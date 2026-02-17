use crate::{BackendType, Capability, ComputeBackend, DeviceId, QueueId, QueueType, Result};
use crate::context::{ComputeContext, BorderMode, ThresholdType, MorphologyType, WarpType};
use cv_core::{Tensor, storage::Storage, TensorShape};
use rayon::prelude::*;
use wide::*;

pub mod simd;

#[derive(Clone, Debug)]
pub struct CpuBackend {
    device_id: DeviceId,
    num_threads: usize,
    simd_available: bool,
}

impl CpuBackend {
    pub fn new() -> Option<Self> {
        Some(Self {
            device_id: DeviceId(0),
            num_threads: rayon::current_num_threads(),
            simd_available: true,
        })
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl ComputeBackend for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn name(&self) -> &str {
        "CPU"
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn supports(&self, capability: Capability) -> bool {
        match capability {
            Capability::Compute => true,
            Capability::Simd => self.simd_available,
            Capability::TensorCore => false,
            Capability::RayTracing => false,
        }
    }

    fn queue(&self, _queue_type: QueueType) -> QueueId {
        QueueId(0)
    }

    fn preferred_queue(&self) -> QueueType {
        QueueType::Compute
    }
}

impl ComputeContext for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn wait_idle(&self) -> Result<()> {
        Ok(())
    }

    fn convolve_2d<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        kernel: &Tensor<f32, S>,
        border_mode: BorderMode,
    ) -> Result<Tensor<f32, S>> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let k_data = kernel.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Kernel not on CPU".into()))?;

        let (h, w) = input.shape.hw();
        let (kh, kw) = kernel.shape.hw();
        let (cx, cy) = (kw / 2, kh / 2);
        
        let mut output_storage = S::new(input.shape.len(), 0.0);
        
        {
            let out_slice = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;
            
            out_slice.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
                for x in 0..w {
                    let mut sum = 0.0;
                    for j in 0..kh {
                        for i in 0..kw {
                            let src_y = y as isize + j as isize - cy as isize;
                            let src_x = x as isize + i as isize - cx as isize;
                            
                            let val = if src_y >= 0 && src_y < h as isize && src_x >= 0 && src_x < w as isize {
                                src[src_y as usize * w + src_x as usize]
                            } else {
                                match border_mode {
                                    BorderMode::Constant(v) => v,
                                    BorderMode::Replicate => {
                                        let c = src_x.clamp(0, w as isize - 1) as usize;
                                        let r = src_y.clamp(0, h as isize - 1) as usize;
                                        src[r * w + c]
                                    }
                                    _ => 0.0,
                                }
                            };
                            
                            sum += val * k_data[j * kw + i];
                        }
                    }
                    row_out[x] = sum;
                }
            });
        }

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn dispatch<S: Storage<u8> + 'static>(
        &self,
        _name: &str,
        _buffers: &[&Tensor<u8, S>],
        _uniforms: &[u8],
        _workgroups: (u32, u32, u32),
    ) -> Result<()> {
        Err(crate::Error::NotSupported("Generic dispatch not supported on CPU".into()))
    }

    fn threshold<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        thresh: u8,
        max_value: u8,
        typ: ThresholdType,
    ) -> Result<Tensor<u8, S>> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let len = src.len();
        
        let mut output_storage = S::new(len, 0);
        let dst = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        let thresh_v = f32x8::splat(thresh as f32);
        let max_v = f32x8::splat(max_value as f32);
        let zero_v = f32x8::ZERO;

        dst.par_chunks_mut(4096).enumerate().for_each(|(chunk_idx, dst_chunk)| {
            let offset = chunk_idx * 4096;
            let src_chunk = &src[offset..offset + dst_chunk.len()];
            
            let n = dst_chunk.len();
            for i in (0..n).step_by(8) {
                if i + 8 <= n {
                    let s_v = f32x8::from([
                        src_chunk[i] as f32,
                        src_chunk[i + 1] as f32,
                        src_chunk[i + 2] as f32,
                        src_chunk[i + 3] as f32,
                        src_chunk[i + 4] as f32,
                        src_chunk[i + 5] as f32,
                        src_chunk[i + 6] as f32,
                        src_chunk[i + 7] as f32,
                    ]);
                    let res = match typ {
                        ThresholdType::Binary => s_v.cmp_gt(thresh_v).blend(max_v, zero_v),
                        ThresholdType::BinaryInv => s_v.cmp_gt(thresh_v).blend(zero_v, max_v),
                        ThresholdType::Trunc => s_v.min(thresh_v),
                        ThresholdType::ToZero => s_v.cmp_gt(thresh_v).blend(s_v, zero_v),
                        ThresholdType::ToZeroInv => s_v.cmp_gt(thresh_v).blend(zero_v, s_v),
                    };
                    let res_arr: [f32; 8] = res.into();
                    for j in 0..8 {
                        dst_chunk[i + j] = res_arr[j] as u8;
                    }
                } else {
                    for j in i..n {
                        let value = src_chunk[j];
                        dst_chunk[j] = match typ {
                            ThresholdType::Binary => if value > thresh { max_value } else { 0 },
                            ThresholdType::BinaryInv => if value > thresh { 0 } else { max_value },
                            ThresholdType::Trunc => value.min(thresh),
                            ThresholdType::ToZero => if value > thresh { value } else { 0 },
                            ThresholdType::ToZeroInv => if value > thresh { 0 } else { value },
                        };
                    }
                }
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn sobel<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        _dx: i32,
        _dy: i32,
        ksize: usize,
    ) -> Result<(Tensor<u8, S>, Tensor<u8, S>)> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let mut gx_storage = S::new(input.shape.len(), 0);
        let mut gy_storage = S::new(input.shape.len(), 0);
        let gx_slice = gx_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Gx output not on CPU".into()))?;
        let gy_slice = gy_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Gy output not on CPU".into()))?;

        let (kx, ky) = if ksize == 3 {
            (
                vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
                vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]
            )
        } else {
            return Err(crate::Error::NotSupported("Only ksize=3 supported for CPU sobel stub".into()));
        };

        gx_slice.par_chunks_mut(w).enumerate().for_each(|(y, row_gx)| {
            for x in 0..w {
                let mut sum_x = 0.0f32;
                for j in 0..3 {
                    for i in 0..3 {
                        let sy = (y as isize + j as isize - 1).clamp(0, h as isize - 1) as usize;
                        let sx = (x as isize + i as isize - 1).clamp(0, w as isize - 1) as usize;
                        sum_x += src[sy * w + sx] as f32 * kx[j * 3 + i];
                    }
                }
                row_gx[x] = sum_x.abs().min(255.0) as u8;
            }
        });

        gy_slice.par_chunks_mut(w).enumerate().for_each(|(y, row_gy)| {
            for x in 0..w {
                let mut sum_y = 0.0f32;
                for j in 0..3 {
                    for i in 0..3 {
                        let sy = (y as isize + j as isize - 1).clamp(0, h as isize - 1) as usize;
                        let sx = (x as isize + i as isize - 1).clamp(0, w as isize - 1) as usize;
                        sum_y += src[sy * w + sx] as f32 * ky[j * 3 + i];
                    }
                }
                row_gy[x] = sum_y.abs().min(255.0) as u8;
            }
        });

        Ok((
            Tensor { storage: gx_storage, shape: input.shape, dtype: input.dtype, _phantom: std::marker::PhantomData },
            Tensor { storage: gy_storage, shape: input.shape, dtype: input.dtype, _phantom: std::marker::PhantomData }
        ))
    }

    fn morphology<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> Result<Tensor<u8, S>> {
        let src_data = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let k_data = kernel.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Kernel not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let (kh, kw) = kernel.shape.hw();
        let (cx, cy) = (kw / 2, kh / 2);

        let mut current_data = src_data.to_vec();
        let mut next_data = vec![0u8; src_data.len()];

        for _ in 0..iterations {
            next_data.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
                for x in 0..w {
                    let mut val = if typ == MorphologyType::Erode { 255u8 } else { 0u8 };
                    for j in 0..kh {
                        for i in 0..kw {
                            if k_data[j * kw + i] == 0 { continue; }
                            let sy = (y as isize + j as isize - cy as isize).clamp(0, h as isize - 1) as usize;
                            let sx = (x as isize + i as isize - cx as isize).clamp(0, w as isize - 1) as usize;
                            let v = current_data[sy * w + sx];
                            if typ == MorphologyType::Erode {
                                val = val.min(v);
                            } else {
                                val = val.max(v);
                            }
                        }
                    }
                    row_out[x] = val;
                }
            });
            std::mem::swap(&mut current_data, &mut next_data);
        }

        Ok(Tensor {
            storage: S::from_vec(current_data),
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn warp<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        matrix: &[[f32; 3]; 3],
        new_shape: (usize, usize),
        _typ: WarpType,
    ) -> Result<Tensor<u8, S>> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let (nw, nh) = (new_shape.0, new_shape.1);
        let mut output_storage = S::new(nw * nh, 0);
        let dst = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        dst.par_chunks_mut(nw).enumerate().for_each(|(y, row_out)| {
            for x in 0..nw {
                let fx = x as f32;
                let fy = y as f32;
                
                let sw = matrix[2][0] * fx + matrix[2][1] * fy + matrix[2][2];
                let sx = (matrix[0][0] * fx + matrix[0][1] * fy + matrix[0][2]) / sw;
                let sy = (matrix[1][0] * fx + matrix[1][1] * fy + matrix[1][2]) / sw;

                if sx >= 0.0 && sx < (w - 1) as f32 && sy >= 0.0 && sy < (h - 1) as f32 {
                    // Bilinear interpolation
                    let x0 = sx as usize;
                    let y0 = sy as usize;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;
                    let dx = sx - x0 as f32;
                    let dy = sy - y0 as f32;

                    let v00 = src[y0 * w + x0] as f32;
                    let v10 = src[y0 * w + x1] as f32;
                    let v01 = src[y1 * w + x0] as f32;
                    let v11 = src[y1 * w + x1] as f32;

                    let val = v00 * (1.0 - dx) * (1.0 - dy) +
                              v10 * dx * (1.0 - dy) +
                              v01 * (1.0 - dx) * dy +
                              v11 * dx * dy;
                    row_out[x] = val as u8;
                }
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: TensorShape::new(1, nh, nw),
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn nms<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        threshold: f32,
        window_size: usize,
    ) -> Result<Tensor<f32, S>> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let mut output_storage = S::new(input.shape.len(), 0.0);
        let dst = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;
        let r = (window_size / 2) as isize;

        dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
            for x in 0..w {
                let val = src[y * w + x];
                if val < threshold { continue; }

                let mut is_max = true;
                for j in -r..=r {
                    for i in -r..=r {
                        if i == 0 && j == 0 { continue; }
                        let sy = y as isize + j;
                        let sx = x as isize + i;
                        if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                            if src[sy as usize * w + sx as usize] > val {
                                is_max = false;
                                break;
                            }
                        }
                    }
                    if !is_max { break; }
                }
                if is_max {
                    row_out[x] = val;
                }
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn nms_boxes<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        let data = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        // Based on common patterns, TensorShape is (channels, height, width).
        // Let's assume input is (N, 5).
        let rows = input.shape.height;
        let cols = input.shape.width;
        if cols != 5 {
            return Err(crate::Error::InvalidInput("NMS Boxes expects (N, 5) tensor".into()));
        }

        let mut boxes: Vec<(usize, f32)> = (0..rows).map(|i| {
            (i, data[i * 5 + 4]) // (index, score)
        }).collect();

        // Sort by score descending
        boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut kept = Vec::new();
        let mut suppressed = vec![false; rows];

        for i in 0..boxes.len() {
            let (idx1, _) = boxes[i];
            if suppressed[idx1] { continue; }

            kept.push(idx1);
            let b1 = cv_core::Rect::new(data[idx1 * 5], data[idx1 * 5 + 1], data[idx1 * 5 + 2] - data[idx1 * 5], data[idx1 * 5 + 3] - data[idx1 * 5 + 1]);

            for j in (i + 1)..boxes.len() {
                let (idx2, _) = boxes[j];
                if suppressed[idx2] { continue; }

                let b2 = cv_core::Rect::new(data[idx2 * 5], data[idx2 * 5 + 1], data[idx2 * 5 + 2] - data[idx2 * 5], data[idx2 * 5 + 3] - data[idx2 * 5 + 1]);
                if b1.iou(&b2) > iou_threshold {
                    suppressed[idx2] = true;
                }
            }
        }

        Ok(kept)
    }

    fn nms_rotated_boxes<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        let data = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let rows = input.shape.height;
        let cols = input.shape.width;
        if cols != 6 {
            return Err(crate::Error::InvalidInput("NMS Rotated Boxes expects (N, 6) tensor".into()));
        }

        let mut boxes: Vec<(usize, f32)> = (0..rows).map(|i| {
            (i, data[i * 6 + 5]) // (index, score)
        }).collect();

        boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut kept = Vec::new();
        let mut suppressed = vec![false; rows];

        for i in 0..boxes.len() {
            let (idx1, _) = boxes[i];
            if suppressed[idx1] { continue; }

            kept.push(idx1);
            let r1 = cv_core::RotatedRect::new(data[idx1 * 6], data[idx1 * 6 + 1], data[idx1 * 6 + 2], data[idx1 * 6 + 3], data[idx1 * 6 + 4]);

            for j in (i + 1)..boxes.len() {
                let (idx2, _) = boxes[j];
                if suppressed[idx2] { continue; }

                let r2 = cv_core::RotatedRect::new(data[idx2 * 6], data[idx2 * 6 + 1], data[idx2 * 6 + 2], data[idx2 * 6 + 3], data[idx2 * 6 + 4]);
                
                // Rotated IOU implementation (stub/simple version)
                if rotated_iou(&r1, &r2) > iou_threshold {
                    suppressed[idx2] = true;
                }
            }
        }

        Ok(kept)
    }

    fn nms_polygons(
        &self,
        polygons: &[cv_core::Polygon],
        scores: &[f32],
        iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        let n = polygons.len();
        if n == 0 { return Ok(Vec::new()); }
        if scores.len() != n {
            return Err(crate::Error::InvalidInput("Scores length must match polygons length".into()));
        }

        let mut items: Vec<(usize, f32)> = (0..n).map(|i| (i, scores[i])).collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut kept = Vec::new();
        let mut suppressed = vec![false; n];

        for i in 0..n {
            let (idx1, _) = items[i];
            if suppressed[idx1] { continue; }

            kept.push(idx1);
            let p1 = &polygons[idx1];

            for j in (i + 1)..n {
                let (idx2, _) = items[j];
                if suppressed[idx2] { continue; }

                let p2 = &polygons[idx2];
                if polygon_iou_internal(p1, p2) > iou_threshold {
                    suppressed[idx2] = true;
                }
            }
        }

        Ok(kept)
    }

    fn pointcloud_transform<S: Storage<f32> + 'static>(
        &self,
        _points: &Tensor<f32, S>,
        _transform: &[[f32; 4]; 4],
    ) -> Result<Tensor<f32, S>> {
        Err(crate::Error::NotSupported("CPU pointcloud_transform pending implementation".into()))
    }

    fn pointcloud_normals<S: Storage<f32> + 'static>(
        &self,
        _points: &Tensor<f32, S>,
        _k_neighbors: u32,
    ) -> Result<Tensor<f32, S>> {
        Err(crate::Error::NotSupported("CPU pointcloud_normals pending implementation".into()))
    }

    fn tsdf_integrate<S: Storage<f32> + 'static>(
        &self,
        _depth_image: &Tensor<f32, S>,
        _camera_pose: &[[f32; 4]; 4],
        _intrinsics: &[f32; 4],
        _tsdf_volume: &mut Tensor<f32, S>,
        _weight_volume: &mut Tensor<f32, S>,
        _voxel_size: f32,
        _truncation: f32,
    ) -> Result<()> {
        Err(crate::Error::NotSupported("CPU tsdf_integrate pending implementation".into()))
    }
}

impl CpuBackend {
    pub fn is_available() -> bool {
        true
    }
}

fn rotated_iou(r1: &cv_core::RotatedRect, r2: &cv_core::RotatedRect) -> f32 {
    let p1 = cv_core::Polygon::new(r1.points().to_vec());
    let p2 = cv_core::Polygon::new(r2.points().to_vec());
    polygon_iou_internal(&p1, &p2)
}

fn polygon_iou_internal(p1: &cv_core::Polygon, p2: &cv_core::Polygon) -> f32 {
    let inter_area = intersection_area_polygons(p1, p2);
    if inter_area <= 0.0 {
        return 0.0;
    }
    let union_area = p1.area() + p2.area() - inter_area;
    inter_area / union_area
}

fn intersection_area_polygons(p1: &cv_core::Polygon, p2: &cv_core::Polygon) -> f32 {
    // Sutherland-Hodgman clipping for generic convex polygons
    let pts1 = &p1.points;
    let pts2 = &p2.points;

    if pts1.len() < 3 || pts2.len() < 3 { return 0.0; }

    let mut poly = pts1.clone();

    // Clip pts1 against each edge of pts2
    for i in 0..pts2.len() {
        let edge_p1 = pts2[i];
        let edge_p2 = pts2[(i + 1) % pts2.len()];

        let mut next_poly = Vec::new();
        if poly.is_empty() { return 0.0; }

        for j in 0..poly.len() {
            let cur = poly[j];
            let prev = poly[(j + poly.len() - 1) % poly.len()];

            let is_cur_inside = is_inside(edge_p1, edge_p2, cur);
            let is_prev_inside = is_inside(edge_p1, edge_p2, prev);

            if is_cur_inside {
                if !is_prev_inside {
                    next_poly.push(intersect(prev, cur, edge_p1, edge_p2));
                }
                next_poly.push(cur);
            } else if is_prev_inside {
                next_poly.push(intersect(prev, cur, edge_p1, edge_p2));
            }
        }
        poly = next_poly;
    }

    if poly.len() < 3 { return 0.0; }
    let mut area = 0.0;
    for i in 0..poly.len() {
        let p1 = poly[i];
        let p2 = poly[(i + 1) % poly.len()];
        area += p1[0] * p2[1] - p2[0] * p1[1];
    }
    area.abs() * 0.5
}

fn is_inside(p1: [f32; 2], p2: [f32; 2], p: [f32; 2]) -> bool {
    (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) >= 0.0
}

fn intersect(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> [f32; 2] {
    let a1 = b[1] - a[1];
    let b1 = a[0] - b[0];
    let c1 = a1 * a[0] + b1 * a[1];

    let a2 = d[1] - c[1];
    let b2 = c[0] - d[0];
    let c2 = a2 * c[0] + b2 * c[1];

    let det = a1 * b2 - a2 * b1;
    if det.abs() < 1e-6 {
        return a; // Parallel
    }
    [(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det]
}

