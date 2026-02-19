use wgpu::{Device, Queue, Instance, RequestAdapterOptions, PowerPreference, Backends};
use wgpu::util::DeviceExt;
use std::sync::{Arc, OnceLock};
use futures::executor::block_on;
use crate::context::{ComputeContext, BorderMode, ThresholdType, MorphologyType, WarpType, ColorConversion};
use crate::{DeviceId, BackendType};
use cv_core::{Tensor, storage::Storage};

static GLOBAL_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

/// Shared GPU Context containing Device and Queue.
#[derive(Debug)]
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl ComputeContext for GpuContext {
    fn backend_type(&self) -> BackendType {
        BackendType::WebGPU
    }

    fn device_id(&self) -> DeviceId {
        DeviceId(0)
    }

    fn wait_idle(&self) -> crate::Result<()> {
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        Ok(())
    }

    fn convolve_2d<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        kernel: &Tensor<f32, S>,
        border_mode: BorderMode,
    ) -> crate::Result<Tensor<f32, S>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            if let Some(kernel_storage) = kernel.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                let input_gpu = Tensor {
                    storage: input_storage.clone(),
                    shape: input.shape,
                    dtype: input.dtype,
                    _phantom: PhantomData,
                };
                let kernel_gpu = Tensor {
                    storage: kernel_storage.clone(),
                    shape: kernel.shape,
                    dtype: kernel.dtype,
                    _phantom: PhantomData,
                };

                let result_gpu = crate::gpu_kernels::convolve::convolve_2d(self, &input_gpu, &kernel_gpu, border_mode)?;

                // SAFETY: downcast_ref above proved S is GpuStorage<f32>
                let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<f32, S>) };
                std::mem::forget(result_gpu);
                return Ok(result);
            }
        }
        
        Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors. Use .to_gpu() first.".into()))
    }

    fn dispatch<S: Storage<u8> + 'static>(
        &self,
        _name: &str,
        _buffers: &[&Tensor<u8, S>],
        _uniforms: &[u8],
        _workgroups: (u32, u32, u32),
    ) -> crate::Result<()> {
        // TODO: Implement generic dispatch
        Err(crate::Error::NotSupported("Generic GPU dispatch pending implementation".into()))
    }

    fn threshold<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        thresh: u8,
        max_value: u8,
        typ: ThresholdType,
    ) -> crate::Result<Tensor<u8, S>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<u8>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::threshold::threshold(self, &input_gpu, thresh, max_value, typ)?;

            // SAFETY: downcast_ref check above proved S is GpuStorage<u8>
            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<u8, S>) };
            std::mem::forget(result_gpu);
            return Ok(result);
        }
        
        Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors. Use .to_gpu() first.".into()))
    }

    fn sobel<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> crate::Result<(Tensor<u8, S>, Tensor<u8, S>)> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
            let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let input_gpu = unsafe { &*input_ptr };

            let (gx_gpu, gy_gpu) = crate::gpu_kernels::sobel::sobel(self, input_gpu, dx, dy, ksize)?;

            let gx = unsafe { std::ptr::read(&gx_gpu as *const _ as *const Tensor<u8, S>) };
            let gy = unsafe { std::ptr::read(&gy_gpu as *const _ as *const Tensor<u8, S>) };
            std::mem::forget(gx_gpu);
            std::mem::forget(gy_gpu);
            Ok((gx, gy))
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn morphology<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> crate::Result<Tensor<u8, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
            let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let input_gpu = unsafe { &*input_ptr };
            
            let kernel_ptr = kernel as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let kernel_gpu = unsafe { &*kernel_ptr };

            let result_gpu = crate::gpu_kernels::morphology::morphology(self, input_gpu, typ, kernel_gpu, iterations)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<u8, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn warp<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        matrix: &[[f32; 3]; 3],
        new_shape: (usize, usize),
        typ: WarpType,
    ) -> crate::Result<Tensor<u8, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
            let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let input_gpu = unsafe { &*input_ptr };

            let result_gpu = crate::gpu_kernels::warp::warp(self, input_gpu, matrix, new_shape, typ)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<u8, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn nms<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        threshold: f32,
        window_size: usize,
    ) -> crate::Result<Tensor<f32, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let input_ptr = input as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let input_gpu = unsafe { &*input_ptr };

            let result_gpu = crate::gpu_kernels::nms::nms_pixel(self, input_gpu, threshold, window_size)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<f32, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn nms_boxes<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> crate::Result<Vec<usize>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let input_ptr = input as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let input_gpu = unsafe { &*input_ptr };

            crate::gpu_kernels::nms::nms_boxes(self, input_gpu, iou_threshold)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn nms_rotated_boxes<S: Storage<f32> + 'static>(
        &self,
        _input: &Tensor<f32, S>,
        _iou_threshold: f32,
    ) -> crate::Result<Vec<usize>> {
        Err(crate::Error::NotSupported("GPU Rotated NMS pending implementation".into()))
    }

    fn nms_polygons(
        &self,
        _polygons: &[cv_core::Polygon],
        _scores: &[f32],
        _iou_threshold: f32,
    ) -> crate::Result<Vec<usize>> {
        Err(crate::Error::NotSupported("GPU Polygon NMS pending implementation".into()))
    }

    fn pointcloud_transform<S: Storage<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        transform: &[[f32; 4]; 4],
    ) -> crate::Result<Tensor<f32, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let input_ptr = points as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let input_gpu = unsafe { &*input_ptr };

            let result_gpu = crate::gpu_kernels::pointcloud::transform_points(self, input_gpu, transform)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<f32, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn pointcloud_normals<S: Storage<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        k_neighbors: u32,
    ) -> crate::Result<Tensor<f32, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let input_ptr = points as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let input_gpu = unsafe { &*input_ptr };

            let result_gpu = crate::gpu_kernels::pointcloud::compute_normals(self, input_gpu, k_neighbors)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<f32, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn tsdf_integrate<S: Storage<f32> + 'static>(
        &self,
        depth_image: &Tensor<f32, S>,
        camera_pose: &[[f32; 4]; 4],
        intrinsics: &[f32; 4],
        voxel_volume: &mut Tensor<f32, S>,
        voxel_size: f32,
        truncation: f32,
    ) -> crate::Result<()> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let depth_ptr = depth_image as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let voxel_ptr = voxel_volume as *mut Tensor<f32, S> as *mut Tensor<f32, GpuStorage<f32>>;
            
            let depth_gpu = unsafe { &*depth_ptr };
            let voxel_gpu = unsafe { &mut *voxel_ptr };

            crate::gpu_kernels::tsdf::integrate(self, depth_gpu, camera_pose, intrinsics, voxel_gpu, voxel_size, truncation)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn tsdf_raycast<S: Storage<f32> + 'static>(
        &self,
        tsdf_volume: &Tensor<f32, S>,
        camera_pose: &[[f32; 4]; 4],
        intrinsics: &[f32; 4],
        image_size: (u32, u32),
        depth_range: (f32, f32),
        voxel_size: f32,
        truncation: f32,
    ) -> crate::Result<Tensor<f32, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let tsdf_ptr = tsdf_volume as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let tsdf_gpu = unsafe { &*tsdf_ptr };

            let result_gpu = crate::gpu_kernels::tsdf::raycast(self, tsdf_gpu, camera_pose, intrinsics, image_size, depth_range, voxel_size, truncation)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<f32, S>) };
            std::mem::forget(result_gpu);
            
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn tsdf_extract_mesh<S: Storage<f32> + 'static>(
        &self,
        voxel_volume: &Tensor<f32, S>,
        voxel_size: f32,
        iso_level: f32,
        max_triangles: u32,
    ) -> crate::Result<Vec<crate::gpu_kernels::marching_cubes::Vertex>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let voxel_ptr = voxel_volume as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let voxel_gpu = unsafe { &*voxel_ptr };

            crate::gpu_kernels::marching_cubes::extract_mesh(self, voxel_gpu, voxel_size, iso_level, max_triangles)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn optical_flow_lk<S: Storage<f32> + 'static>(
        &self,
        prev_pyramid: &[Tensor<f32, S>],
        next_pyramid: &[Tensor<f32, S>],
        points: &[[f32; 2]],
        window_size: usize,
        max_iters: u32,
    ) -> crate::Result<Vec<[f32; 2]>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            // unsafe cast slice of tensors
            let prev_ptr = prev_pyramid.as_ptr() as *const Tensor<f32, GpuStorage<f32>>;
            let prev_gpu = unsafe { std::slice::from_raw_parts(prev_ptr, prev_pyramid.len()) };

            let next_ptr = next_pyramid.as_ptr() as *const Tensor<f32, GpuStorage<f32>>;
            let next_gpu = unsafe { std::slice::from_raw_parts(next_ptr, next_pyramid.len()) };

            crate::gpu_kernels::optical_flow::lucas_kanade(self, prev_gpu, next_gpu, points, window_size, max_iters)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn cvt_color<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        code: ColorConversion,
    ) -> crate::Result<Tensor<u8, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
            let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let input_gpu = unsafe { &*input_ptr };

            let result_gpu = crate::gpu_kernels::color::cvt_color(self, input_gpu, code)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<u8, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn resize<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        new_shape: (usize, usize),
    ) -> crate::Result<Tensor<u8, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
            let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let input_gpu = unsafe { &*input_ptr };

            let result_gpu = crate::gpu_kernels::resize::resize(self, input_gpu, new_shape)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<u8, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn bilateral_filter<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        d: i32,
        sigma_color: f32,
        sigma_space: f32,
    ) -> crate::Result<Tensor<u8, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
            let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let input_gpu = unsafe { &*input_ptr };

            let result_gpu = crate::gpu_kernels::bilateral::bilateral_filter(self, input_gpu, d, sigma_color, sigma_space)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<u8, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn fast_detect<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        threshold: u8,
        non_max_suppression: bool,
    ) -> crate::Result<Tensor<u8, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
            let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let input_gpu = unsafe { &*input_ptr };

            let result_gpu = crate::gpu_kernels::fast::fast_detect(self, input_gpu, threshold, non_max_suppression)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<u8, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn gaussian_blur<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        sigma: f32,
        k_size: usize,
    ) -> crate::Result<Tensor<u8, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;
        use crate::tensor_ext::TensorCast;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
            let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let input_gpu = unsafe { &*input_ptr };

            // We need f32 for blur accuracy
            let input_f32 = input_gpu.to_f32_ctx(self)?;
            let blurred_f32 = crate::gpu_kernels::convolve::gaussian_blur(self, &input_f32, sigma, k_size)?;
            
            // Convert back to u8
            let blurred_u8 = blurred_f32.to_u8_ctx(self)?;

            // Cleanup intermediate f32 buffers if they were created
            // Note: to_f32_ctx and gaussian_blur might return pooled buffers.
            // Since they return Tensors, we need to decide if Tensor owns the pooled buffer.
            // In our current design, GpuStorage owns an Arc<Buffer>. 
            // If we want to return GpuStorage buffers to pool, we need that Drop impl.
            
            let result = unsafe { std::ptr::read(&blurred_u8 as *const _ as *const Tensor<u8, S>) };
            std::mem::forget(blurred_u8);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn subtract<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug, S: Storage<T> + 'static>(
        &self,
        a: &Tensor<T, S>,
        b: &Tensor<T, S>,
    ) -> crate::Result<Tensor<T, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<T>() == TypeId::of::<f32>() && TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let a_ptr = a as *const Tensor<T, S> as *const Tensor<f32, GpuStorage<f32>>;
            let b_ptr = b as *const Tensor<T, S> as *const Tensor<f32, GpuStorage<f32>>;
            
            let a_gpu = unsafe { &*a_ptr };
            let b_gpu = unsafe { &*b_ptr };

            let result_gpu = crate::gpu_kernels::subtract::subtract(self, a_gpu, b_gpu)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<T, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::NotSupported("GPU subtract only supports f32 for now".into()))
        }
    }

    fn match_descriptors<S: Storage<u8> + 'static>(
        &self,
        query: &Tensor<u8, S>,
        train: &Tensor<u8, S>,
        ratio_threshold: f32,
    ) -> crate::Result<cv_core::Matches> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
            let q_ptr = query as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let t_ptr = train as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let q_gpu = unsafe { &*q_ptr };
            let t_gpu = unsafe { &*t_ptr };

            crate::gpu_kernels::matching::match_descriptors(self, q_gpu, t_gpu, ratio_threshold)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn sift_extrema<S: Storage<f32> + 'static>(
        &self,
        dog_prev: &Tensor<f32, S>,
        dog_curr: &Tensor<f32, S>,
        dog_next: &Tensor<f32, S>,
        threshold: f32,
        edge_threshold: f32,
    ) -> crate::Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;
        use crate::tensor_ext::TensorToCpu;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let p_ptr = dog_prev as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let c_ptr = dog_curr as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let n_ptr = dog_next as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            
            let p_gpu = unsafe { &*p_ptr };
            let c_gpu = unsafe { &*c_ptr };
            let n_gpu = unsafe { &*n_ptr };

            let result_gpu = crate::gpu_kernels::sift::sift_extrema(self, p_gpu, c_gpu, n_gpu, threshold, edge_threshold)?;
            result_gpu.to_cpu_ctx(self)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn compute_sift_descriptors<S: Storage<f32> + 'static>(
        &self,
        image: &Tensor<f32, S>,
        keypoints: &cv_core::KeyPoints,
    ) -> crate::Result<cv_core::Descriptors> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let img_ptr = image as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let img_gpu = unsafe { &*img_ptr };

            // 1. Upload keypoints to GPU
            let kp_data: Vec<[f32; 4]> = keypoints.keypoints.iter().map(|kp| {
                [kp.x as f32, kp.y as f32, kp.size as f32, kp.angle as f32]
            }).collect();
            
            let kp_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SIFT Keypoints Buffer"),
                contents: bytemuck::cast_slice(&kp_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

            // 2. Output buffer for descriptors (num_kps * 128 * 4 bytes)
            let out_byte_size = (kp_data.len() * 128 * 4) as u64;
            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SIFT Descriptors Output"),
                size: out_byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let params = [image.shape.width as u32, image.shape.height as u32, kp_data.len() as u32, 0u32];
            let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SIFT Desc Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let shader_source = include_str!("../shaders/sift_descriptor.wgsl");
            let pipeline = self.create_compute_pipeline(shader_source, "main");

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SIFT Desc Bind Group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: img_gpu.storage.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: kp_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
                ],
            });

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let x = (kp_data.len() as u32 + 63) / 64;
                pass.dispatch_workgroups(x, 1, 1);
            }
            self.submit(encoder);

            // 3. Read back and convert to Descriptor objects
            let raw_descs: Vec<f32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
                self.device.clone(),
                &self.queue,
                &output_buffer,
                0,
                out_byte_size as usize,
            ))?;

            let mut descs = Vec::with_capacity(keypoints.len());
            for i in 0..keypoints.len() {
                let start = i * 128;
                let data: Vec<u8> = raw_descs[start .. start + 128].iter().map(|&v| {
                    (v * 512.0).min(255.0) as u8
                }).collect();
                descs.push(cv_core::Descriptor::new(data, keypoints.keypoints[i].clone()));
            }

            Ok(cv_core::Descriptors { descriptors: descs })
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn icp_correspondences<S: Storage<f32> + 'static>(
        &self,
        src: &Tensor<f32, S>,
        tgt: &Tensor<f32, S>,
        max_dist: f32,
    ) -> crate::Result<Vec<(usize, usize, f32)>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let src_ptr = src as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let tgt_ptr = tgt as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let src_gpu = unsafe { &*src_ptr };
            let tgt_gpu = unsafe { &*tgt_ptr };

            crate::gpu_kernels::icp::icp_correspondences(self, src_gpu, tgt_gpu, max_dist)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn icp_accumulate<S: Storage<f32> + 'static>(
        &self,
        source: &Tensor<f32, S>,
        target: &Tensor<f32, S>,
        target_normals: &Tensor<f32, S>,
        correspondences: &[(u32, u32)],
        transform: &nalgebra::Matrix4<f32>,
    ) -> crate::Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let src_ptr = source as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let tgt_ptr = target as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let n_ptr = target_normals as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            
            let src_gpu = unsafe { &*src_ptr };
            let tgt_gpu = unsafe { &*tgt_ptr };
            let n_gpu = unsafe { &*n_ptr };

            crate::gpu_kernels::icp::icp_accumulate(self, src_gpu, tgt_gpu, n_gpu, correspondences, transform)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn dense_icp_step<S: Storage<f32> + 'static>(
        &self,
        source_depth: &Tensor<f32, S>,
        target_data: &Tensor<f32, S>,
        intrinsics: &[f32; 4],
        initial_guess: &nalgebra::Matrix4<f32>,
        max_dist: f32,
        max_angle: f32,
    ) -> crate::Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let src_ptr = source_depth as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let tgt_ptr = target_data as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            
            let src_gpu = unsafe { &*src_ptr };
            let tgt_gpu = unsafe { &*tgt_ptr };

            crate::gpu_kernels::icp::dense_step(self, src_gpu, tgt_gpu, intrinsics, initial_guess, max_dist, max_angle)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn akaze_diffusion<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        k: f32,
        tau: f32,
    ) -> crate::Result<Tensor<f32, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let input_ptr = input as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let input_gpu = unsafe { &*input_ptr };

            let result_gpu = crate::gpu_kernels::akaze::akaze_diffusion(self, input_gpu, k, tau)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<f32, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn akaze_derivatives<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
    ) -> crate::Result<(Tensor<f32, S>, Tensor<f32, S>, Tensor<f32, S>)> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let input_ptr = input as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let input_gpu = unsafe { &*input_ptr };

            let (lx_gpu, ly_gpu, ldet_gpu) = crate::gpu_kernels::akaze::akaze_derivatives(self, input_gpu)?;

            let lx = unsafe { std::ptr::read(&lx_gpu as *const _ as *const Tensor<f32, S>) };
            let ly = unsafe { std::ptr::read(&ly_gpu as *const _ as *const Tensor<f32, S>) };
            let ldet = unsafe { std::ptr::read(&ldet_gpu as *const _ as *const Tensor<f32, S>) };
            
            std::mem::forget(lx_gpu);
            std::mem::forget(ly_gpu);
            std::mem::forget(ldet_gpu);
            
            Ok((lx, ly, ldet))
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn akaze_contrast_k<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
    ) -> crate::Result<f32> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let input_ptr = input as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let input_gpu = unsafe { &*input_ptr };

            crate::gpu_kernels::akaze::akaze_contrast_k(self, input_gpu)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn spmv<S: Storage<f32> + 'static>(
        &self,
        row_ptr: &[u32],
        col_indices: &[u32],
        values: &[f32],
        x: &Tensor<f32, S>,
    ) -> crate::Result<Tensor<f32, S>> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            let x_ptr = x as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let x_gpu = unsafe { &*x_ptr };

            let result_gpu = crate::gpu_kernels::sparse::spmv(self, row_ptr, col_indices, values, x_gpu)?;

            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<f32, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn mog2_update<S1: Storage<f32> + 'static, S2: Storage<u32> + 'static>(
        &self,
        frame: &Tensor<f32, S1>,
        model: &mut Tensor<f32, S1>,
        mask: &mut Tensor<u32, S2>,
        params: &crate::context::Mog2Params,
    ) -> crate::Result<()> {
        use std::any::TypeId;
        use crate::storage::GpuStorage;
        use wgpu::util::DeviceExt;

        if TypeId::of::<S1>() == TypeId::of::<GpuStorage<f32>>() && TypeId::of::<S2>() == TypeId::of::<GpuStorage<u32>>() {
            let frame_ptr = frame as *const Tensor<f32, S1> as *const Tensor<f32, GpuStorage<f32>>;
            let model_ptr = model as *mut Tensor<f32, S1> as *mut Tensor<f32, GpuStorage<f32>>;
            let mask_ptr = mask as *mut Tensor<u32, S2> as *mut Tensor<u32, GpuStorage<u32>>;
            
            let frame_gpu = unsafe { &*frame_ptr };
            let model_gpu = unsafe { &mut *model_ptr };
            let mask_gpu = unsafe { &mut *mask_ptr };

            let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MOG2 Params"),
                contents: bytemuck::cast_slice(&[*params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let shader_source = include_str!("../shaders/mog2_update.wgsl");
            let pipeline = self.create_compute_pipeline(shader_source, "main");

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MOG2 Bind Group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: frame_gpu.storage.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: model_gpu.storage.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: mask_gpu.storage.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
                ],
            });

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((params.width + 15) / 16, (params.height + 15) / 16, 1);
            }
            self.submit(encoder);
            Ok(())
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }
}

impl GpuContext {
    /// Get the global GPU context, initializing it if necessary.
    pub fn global() -> Option<&'static GpuContext> {
        GLOBAL_CONTEXT.get_or_init(|| {
            Self::new().ok()
        }).as_ref()
    }

    /// Initialize a new GPU context (synchronous wrapper).
    pub fn new() -> crate::Result<Self> {
        block_on(Self::new_async())
    }

    /// Initialize a new GPU context asynchronously.
    pub async fn new_async() -> crate::Result<Self> {
        Self::new_with_policy(PowerPreference::HighPerformance).await
    }

    pub async fn new_with_policy(preference: PowerPreference) -> crate::Result<Self> {
        // Create instance with conservative flags to avoid driver panics on integrated GPUs
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            flags: wgpu::InstanceFlags::default().difference(wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION),
            ..Default::default()
        });

        // Request adapter
        let adapter = instance.request_adapter(&RequestAdapterOptions {
            power_preference: preference,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.map_err(|e| crate::Error::InitError(format!("Failed to find a suitable GPU adapter: {}", e)))?;

        Self::from_adapter(adapter).await
    }

    pub async fn from_adapter(adapter: wgpu::Adapter) -> crate::Result<Self> {
        // Request device
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("CV-HAL Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::default(),
            },
        ).await.map_err(|e| crate::Error::InitError(format!("Failed to create GPU device: {}", e)))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Check if a GPU is available.
    pub fn is_available() -> bool {
        block_on(Self::is_available_async())
    }

    /// Check if a GPU is available asynchronously.
    pub async fn is_available_async() -> bool {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        !instance.enumerate_adapters(Backends::all()).await.is_empty()
    }

    /// Enumerate all available adapters.
    pub async fn enumerate_adapters() -> Vec<wgpu::Adapter> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        instance.enumerate_adapters(Backends::all()).await
    }
    
    /// Get reference to device (convenience method)
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get Arc to device
    pub fn device_arc(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// Get Arc to queue
    pub fn queue_arc(&self) -> Arc<Queue> {
        self.queue.clone()
    }

    /// Submit a command encoder (convenience method)
    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Create a simplified compute pipeline.
    pub fn create_compute_pipeline(&self, shader_source: &str, entry_point: &str) -> wgpu::ComputePipeline {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Get a pooled buffer.
    pub fn get_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        crate::gpu_kernels::buffer_utils::global_pool().get(&self.device, size, usage)
    }

    /// Return a buffer to the pool.
    pub fn return_buffer(&self, buffer: wgpu::Buffer, usage: wgpu::BufferUsages) {
        crate::gpu_kernels::buffer_utils::global_pool().return_buffer(buffer, usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        let ctx = GpuContext::new();
        match ctx {
            Ok(c) => println!("GPU Context created: {:?}", c.device),
            Err(e) => println!("GPU initialization failed (expected on some CI): {}", e),
        }
    }

    #[test]
    fn test_gaussian_blur_gpu_parity() {
        let gpu = if let Some(g) = GpuContext::global() { g } else { return; };
        let cpu = crate::cpu::CpuBackend::new().unwrap();
        
        let width = 64usize;
        let height = 64usize;
        let mut data = vec![0u8; width * height];
        for i in 0..data.len() { data[i] = (i % 255) as u8; }
        
        let shape = cv_core::TensorShape::new(1, height, width);
        let tensor_cpu = cv_core::CpuTensor::from_vec(data, shape);
        
        // GPU execution
        use crate::tensor_ext::{TensorToGpu, TensorToCpu};
        let tensor_gpu = tensor_cpu.to_gpu_ctx(gpu).unwrap();
        let blurred_gpu = gpu.gaussian_blur(&tensor_gpu, 1.5, 7).unwrap();
        let res_gpu = blurred_gpu.to_cpu_ctx(gpu).unwrap();
        
        // CPU execution
        let res_cpu = cpu.gaussian_blur(&tensor_cpu, 1.5, 7).unwrap();
        
        // Check equality
        let slice_gpu = res_gpu.as_slice();
        let slice_cpu = res_cpu.as_slice();
        
        let mut diff_count = 0;
        for i in 0..slice_gpu.len() {
            if (slice_gpu[i] as i32 - slice_cpu[i] as i32).abs() > 1 {
                diff_count += 1;
            }
        }
        
        assert!(diff_count < (width * height) / 100, "Too many differences between GPU and CPU blur: {}", diff_count);
    }
}
