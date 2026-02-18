use crate::{BackendType, Capability, ComputeBackend, DeviceId, QueueId, QueueType, Result};
use crate::context::{ComputeContext, BorderMode, ThresholdType, MorphologyType, WarpType, ColorConversion};
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
        points: &Tensor<f32, S>,
        transform: &[[f32; 4]; 4],
    ) -> Result<Tensor<f32, S>> {
        let src = points.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Points not on CPU".into()))?;
        let num_points = points.shape.height;
        let mut output_storage = S::new(num_points * 4, 0.0);
        let dst = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        dst.par_chunks_mut(4).enumerate().for_each(|(i, point_out)| {
            let px = src[i * 4];
            let py = src[i * 4 + 1];
            let pz = src[i * 4 + 2];
            let pw = src[i * 4 + 3];

            point_out[0] = transform[0][0] * px + transform[0][1] * py + transform[0][2] * pz + transform[0][3] * pw;
            point_out[1] = transform[1][0] * px + transform[1][1] * py + transform[1][2] * pz + transform[1][3] * pw;
            point_out[2] = transform[2][0] * px + transform[2][1] * py + transform[2][2] * pz + transform[2][3] * pw;
            point_out[3] = transform[3][0] * px + transform[3][1] * py + transform[3][2] * pz + transform[3][3] * pw;
        });

        Ok(Tensor {
            storage: output_storage,
            shape: points.shape,
            dtype: points.dtype,
            _phantom: std::marker::PhantomData,
        })
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

    fn cvt_color<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        code: ColorConversion,
    ) -> Result<Tensor<u8, S>> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let c = input.shape.channels;

        match code {
            ColorConversion::RgbToGray | ColorConversion::BgrToGray => {
                if c != 3 {
                    return Err(crate::Error::InvalidInput("RgbToGray requires 3 channels".into()));
                }
                let mut output_storage = S::new(h * w, 0);
                let dst = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;
                
                dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
                    for x in 0..w {
                        let base = (y * w + x) * 3;
                        let (r, g, b) = if code == ColorConversion::RgbToGray {
                            (src[base], src[base + 1], src[base + 2])
                        } else {
                            (src[base + 2], src[base + 1], src[base])
                        };
                        // Standard luminance formula
                        row_out[x] = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                    }
                });

                Ok(Tensor {
                    storage: output_storage,
                    shape: TensorShape::new(1, h, w),
                    dtype: input.dtype,
                    _phantom: std::marker::PhantomData,
                })
            }
            ColorConversion::GrayToRgb => {
                if c != 1 {
                    return Err(crate::Error::InvalidInput("GrayToRgb requires 1 channel".into()));
                }
                let mut output_storage = S::new(h * w * 3, 0);
                let dst = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

                dst.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_out)| {
                    for x in 0..w {
                        let val = src[y * w + x];
                        row_out[x * 3] = val;
                        row_out[x * 3 + 1] = val;
                        row_out[x * 3 + 2] = val;
                    }
                });

                Ok(Tensor {
                    storage: output_storage,
                    shape: TensorShape::new(3, h, w),
                    dtype: input.dtype,
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    }

    fn resize<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        new_shape: (usize, usize),
    ) -> Result<Tensor<u8, S>> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let c = input.shape.channels;
        let (nw, nh) = new_shape;

        let mut output_storage = S::new(nw * nh * c, 0);
        let dst = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        let scale_x = w as f32 / nw as f32;
        let scale_y = h as f32 / nh as f32;

        dst.par_chunks_mut(nw * c).enumerate().for_each(|(y, row_out)| {
            for x in 0..nw {
                let src_x = (x as f32 * scale_x).min(w as f32 - 1.0);
                let src_y = (y as f32 * scale_y).min(h as f32 - 1.0);
                
                // Nearest neighbor for now (can expand to bilinear)
                let sx = src_x as usize;
                let sy = src_y as usize;
                
                for ch in 0..c {
                    row_out[x * c + ch] = src[(sy * w + sx) * c + ch];
                }
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: TensorShape::new(c, nh, nw),
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn bilateral_filter<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        d: i32,
        sigma_color: f32,
        sigma_space: f32,
    ) -> Result<Tensor<u8, S>> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let c = input.shape.channels;
        if c != 1 {
            return Err(crate::Error::NotSupported("Bilateral filter currently only for grayscale".into()));
        }

        let mut output_storage = S::new(h * w, 0);
        let dst = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        let radius = if d <= 0 { (sigma_space * 1.5).ceil() as i32 } else { d / 2 };
        let color_coeff = -0.5 / (sigma_color * sigma_color);
        let space_coeff = -0.5 / (sigma_space * sigma_space);

        dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
            for x in 0..w {
                let center_val = src[y * w + x] as f32;
                let mut sum = 0.0;
                let mut norm = 0.0;

                for j in -radius..=radius {
                    for i in -radius..=radius {
                        let sy = (y as i32 + j).clamp(0, h as i32 - 1) as usize;
                        let sx = (x as i32 + i).clamp(0, w as i32 - 1) as usize;
                        
                        let val = src[sy * w + sx] as f32;
                        let dist_sq = (i * i + j * j) as f32;
                        let range_sq = (val - center_val).powi(2);
                        
                        let weight = (dist_sq * space_coeff + range_sq * color_coeff).exp();
                        sum += val * weight;
                        norm += weight;
                    }
                }
                row_out[x] = (sum / norm) as u8;
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn fast_detect<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        threshold: u8,
        non_max_suppression: bool,
    ) -> Result<Tensor<u8, S>> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let (h, w) = input.shape.hw();
        let mut output_storage = S::new(h * w, 0);
        let dst = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        dst.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
            if y < 3 || y >= h - 3 { return; }
            for x in 3..w - 3 {
                let p = src[y * w + x];
                let high = p.saturating_add(threshold);
                let low = p.saturating_sub(threshold);

                // Bresenham circle radius 3
                let offsets = [
                    (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
                    (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)
                ];

                let mut brighter = 0u32;
                let mut darker = 0u32;
                let mut vals = [0u8; 16];

                for (i, (dx, dy)) in offsets.iter().enumerate() {
                    let v = src[((y as i32 + dy) as usize) * w + (x as i32 + dx) as usize];
                    vals[i] = v;
                    if v > high { brighter += 1; }
                    else if v < low { darker += 1; }
                }

                if brighter >= 9 || darker >= 9 {
                    // Check contiguous
                    if has_9_contiguous(&vals, high, low) {
                        // Compute score (simple version: sum of abs diff)
                        let mut score = 0u32;
                        for &v in &vals {
                            score += (v as i32 - p as i32).abs() as u32;
                        }
                        row_out[x] = (score / 16).min(255) as u8;
                    }
                }
            }
        });

        if non_max_suppression {
            // TODO: Parallel NMS on score map
        }

        Ok(Tensor {
            storage: output_storage,
            shape: TensorShape::new(1, h, w),
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }
}

fn has_9_contiguous(vals: &[u8; 16], high: u8, low: u8) -> bool {
    let mut b_mask = 0u32;
    let mut d_mask = 0u32;
    for i in 0..16 {
        if vals[i] > high { b_mask |= 1 << i; }
        if vals[i] < low { d_mask |= 1 << i; }
    }
    
    let b_mask_ext = b_mask | (b_mask << 16);
    let d_mask_ext = d_mask | (d_mask << 16);
    
    for i in 0..16 {
        if (b_mask_ext >> i) & 0x1FF == 0x1FF { return true; }
        if (d_mask_ext >> i) & 0x1FF == 0x1FF { return true; }
    }
    false
}

impl CpuBackend {
    pub fn is_available() -> bool {
        true
    }
}

fn rotated_iou(r1: &cv_core::RotatedRect, r2: &cv_core::RotatedRect) -> f32 {
    let mut p1 = cv_core::Polygon::new(r1.points().to_vec());
    let mut p2 = cv_core::Polygon::new(r2.points().to_vec());
    p1.ensure_counter_clockwise();
    p2.ensure_counter_clockwise();
    polygon_iou_internal(&p1, &p2)
}

fn polygon_iou_internal(p1: &cv_core::Polygon, p2: &cv_core::Polygon) -> f32 {
    let inter_area = intersection_area_polygons(p1, p2);
    let a1 = p1.unsigned_area();
    let a2 = p2.unsigned_area();
    if inter_area <= 0.0 {
        return 0.0;
    }
    let union_area = a1 + a2 - inter_area;
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
