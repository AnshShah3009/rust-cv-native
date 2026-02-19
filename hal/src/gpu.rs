use wgpu::{Device, Queue, Instance, RequestAdapterOptions, PowerPreference, Backends};
use wgpu::util::DeviceExt;
use std::sync::{Arc, OnceLock};
use futures::executor::block_on;
use crate::context::{ComputeContext, BorderMode, ThresholdType, MorphologyType, WarpType, ColorConversion};
use crate::{DeviceId, BackendType};
use cv_core::{Tensor, storage::Storage};

static GLOBAL_CONTEXT: OnceLock<crate::Result<GpuContext>> = OnceLock::new();

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

        if let (Some(input_storage), Some(kernel_storage)) = (
            input.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            kernel.storage.as_any().downcast_ref::<GpuStorage<f32>>()
        ) {
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

            // Safe downcast back to S
            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU result to original storage type".into()))
            }
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors. Use .to_gpu() first.".into()))
        }
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

            // Safe downcast back to S
            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU result to original storage type".into()))
            }
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors. Use .to_gpu() first.".into()))
        }
    }

    fn sobel<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> crate::Result<(Tensor<u8, S>, Tensor<u8, S>)> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<u8>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let (gx_gpu, gy_gpu) = crate::gpu_kernels::sobel::sobel(self, &input_gpu, dx, dy, ksize)?;

            let gx_any = Box::new(gx_gpu.storage).boxed_any();
            let gy_any = Box::new(gy_gpu.storage).boxed_any();

            if let (Ok(gx_s), Ok(gy_s)) = (gx_any.downcast::<S>(), gy_any.downcast::<S>()) {
                Ok((
                    Tensor { storage: *gx_s, shape: gx_gpu.shape, dtype: gx_gpu.dtype, _phantom: PhantomData },
                    Tensor { storage: *gy_s, shape: gy_gpu.shape, dtype: gy_gpu.dtype, _phantom: PhantomData }
                ))
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU Sobel results".into()))
            }
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let (Some(input_storage), Some(kernel_storage)) = (
            input.storage.as_any().downcast_ref::<GpuStorage<u8>>(),
            kernel.storage.as_any().downcast_ref::<GpuStorage<u8>>()
        ) {
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

            let result_gpu = crate::gpu_kernels::morphology::morphology(self, &input_gpu, typ, &kernel_gpu, iterations)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU morphology result".into()))
            }
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<u8>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::warp::warp(self, &input_gpu, matrix, new_shape, typ)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU warp result".into()))
            }
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::nms::nms_pixel(self, &input_gpu, threshold, window_size)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU NMS result".into()))
            }
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn nms_boxes<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> crate::Result<Vec<usize>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            crate::gpu_kernels::nms::nms_boxes(self, &input_gpu, iou_threshold)
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = points.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: points.shape,
                dtype: points.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::pointcloud::transform_points(self, &input_gpu, transform)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU pointcloud result".into()))
            }
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn pointcloud_normals<S: Storage<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        k_neighbors: u32,
    ) -> crate::Result<Tensor<f32, S>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = points.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: points.shape,
                dtype: points.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::pointcloud::compute_normals(self, &input_gpu, k_neighbors)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU normals result".into()))
            }
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let (Some(depth_storage), Some(voxel_storage)) = (
            depth_image.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            voxel_volume.storage.as_any().downcast_ref::<GpuStorage<f32>>()
        ) {
            let depth_gpu = Tensor {
                storage: depth_storage.clone(),
                shape: depth_image.shape,
                dtype: depth_image.dtype,
                _phantom: PhantomData,
            };
            // NOTE: We need to be careful with &mut GpuStorage if we want to modify it in place.
            // For now, TSDF kernels typically take a mutable reference to the underlying buffer.
            let mut voxel_gpu = Tensor {
                storage: voxel_storage.clone(),
                shape: voxel_volume.shape,
                dtype: voxel_volume.dtype,
                _phantom: PhantomData,
            };

            crate::gpu_kernels::tsdf::integrate(self, &depth_gpu, camera_pose, intrinsics, &mut voxel_gpu, voxel_size, truncation)
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = tsdf_volume.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: tsdf_volume.shape,
                dtype: tsdf_volume.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::tsdf::raycast(self, &input_gpu, camera_pose, intrinsics, image_size, depth_range, voxel_size, truncation)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU TSDF result".into()))
            }
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = voxel_volume.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: voxel_volume.shape,
                dtype: voxel_volume.dtype,
                _phantom: PhantomData,
            };

            crate::gpu_kernels::marching_cubes::extract_mesh(self, &input_gpu, voxel_size, iso_level, max_triangles)
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        // Verify all tensors in pyramids are GpuStorage
        let mut prev_gpu = Vec::with_capacity(prev_pyramid.len());
        for t in prev_pyramid {
            if let Some(s) = t.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                prev_gpu.push(Tensor { storage: s.clone(), shape: t.shape, dtype: t.dtype, _phantom: PhantomData });
            } else {
                return Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()));
            }
        }

        let mut next_gpu = Vec::with_capacity(next_pyramid.len());
        for t in next_pyramid {
            if let Some(s) = t.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
                next_gpu.push(Tensor { storage: s.clone(), shape: t.shape, dtype: t.dtype, _phantom: PhantomData });
            } else {
                return Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()));
            }
        }

        crate::gpu_kernels::optical_flow::lucas_kanade(self, &prev_gpu, &next_gpu, points, window_size, max_iters)
    }

    fn cvt_color<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        code: ColorConversion,
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

            let result_gpu = crate::gpu_kernels::color::cvt_color(self, &input_gpu, code)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU result".into()))
            }
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn resize<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        new_shape: (usize, usize),
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

            let result_gpu = crate::gpu_kernels::resize::resize(self, &input_gpu, new_shape)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU result".into()))
            }
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<u8>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::bilateral::bilateral_filter(self, &input_gpu, d, sigma_color, sigma_space)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU result".into()))
            }
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<u8>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::fast::fast_detect(self, &input_gpu, threshold, non_max_suppression)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU result".into()))
            }
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
        use crate::storage::GpuStorage;
        use crate::tensor_ext::TensorCast;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<u8>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            // We need f32 for blur accuracy
            let input_f32 = input_gpu.to_f32_ctx(self)?;
            let blurred_f32 = crate::gpu_kernels::convolve::gaussian_blur(self, &input_f32, sigma, k_size)?;
            
            // Convert back to u8
            let blurred_u8 = blurred_f32.to_u8_ctx(self)?;

            let storage_any = Box::new(blurred_u8.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: blurred_u8.shape,
                    dtype: blurred_u8.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU result".into()))
            }
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors. Use .to_gpu() first.".into()))
        }
    }

    fn subtract<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug, S: Storage<T> + 'static>(
        &self,
        a: &Tensor<T, S>,
        b: &Tensor<T, S>,
    ) -> crate::Result<Tensor<T, S>> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;
        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            if let (Some(a_storage), Some(b_storage)) = (
                a.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
                b.storage.as_any().downcast_ref::<GpuStorage<f32>>()
            ) {
                let a_gpu = Tensor { storage: a_storage.clone(), shape: a.shape, dtype: a.dtype, _phantom: PhantomData };
                let b_gpu = Tensor { storage: b_storage.clone(), shape: b.shape, dtype: b.dtype, _phantom: PhantomData };

                let result_gpu = crate::gpu_kernels::subtract::subtract(self, &a_gpu, &b_gpu)?;

                let storage_any = Box::new(result_gpu.storage).boxed_any();
                if let Ok(storage_s) = storage_any.downcast::<S>() {
                    Ok(Tensor {
                        storage: *storage_s,
                        shape: result_gpu.shape,
                        dtype: result_gpu.dtype,
                        _phantom: PhantomData,
                    })
                } else {
                    Err(crate::Error::InvalidInput("Failed to downcast GPU result".into()))
                }
            } else {
                Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
            }
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let (Some(q_storage), Some(t_storage)) = (
            query.storage.as_any().downcast_ref::<GpuStorage<u8>>(),
            train.storage.as_any().downcast_ref::<GpuStorage<u8>>()
        ) {
            let q_gpu = Tensor { storage: q_storage.clone(), shape: query.shape, dtype: query.dtype, _phantom: PhantomData };
            let t_gpu = Tensor { storage: t_storage.clone(), shape: train.shape, dtype: train.dtype, _phantom: PhantomData };

            crate::gpu_kernels::matching::match_descriptors(self, &q_gpu, &t_gpu, ratio_threshold)
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
        use crate::storage::GpuStorage;
        use crate::tensor_ext::TensorToCpu;
        use std::marker::PhantomData;

        if let (Some(p_storage), Some(c_storage), Some(n_storage)) = (
            dog_prev.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            dog_curr.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            dog_next.storage.as_any().downcast_ref::<GpuStorage<f32>>()
        ) {
            let p_gpu = Tensor { storage: p_storage.clone(), shape: dog_prev.shape, dtype: dog_prev.dtype, _phantom: PhantomData };
            let c_gpu = Tensor { storage: c_storage.clone(), shape: dog_curr.shape, dtype: dog_curr.dtype, _phantom: PhantomData };
            let n_gpu = Tensor { storage: n_storage.clone(), shape: dog_next.shape, dtype: dog_next.dtype, _phantom: PhantomData };

            let result_gpu = crate::gpu_kernels::sift::sift_extrema(self, &p_gpu, &c_gpu, &n_gpu, threshold, edge_threshold)?;
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(img_storage) = image.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let img_gpu = Tensor { storage: img_storage.clone(), shape: image.shape, dtype: image.dtype, _phantom: PhantomData };
            // ... (rest of the implementation remains similar, but using img_gpu)

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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let (Some(src_storage), Some(tgt_storage)) = (
            src.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            tgt.storage.as_any().downcast_ref::<GpuStorage<f32>>()
        ) {
            let src_gpu = Tensor { storage: src_storage.clone(), shape: src.shape, dtype: src.dtype, _phantom: PhantomData };
            let tgt_gpu = Tensor { storage: tgt_storage.clone(), shape: tgt.shape, dtype: tgt.dtype, _phantom: PhantomData };

            crate::gpu_kernels::icp::icp_correspondences(self, &src_gpu, &tgt_gpu, max_dist)
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let (Some(src_storage), Some(tgt_storage), Some(norm_storage)) = (
            source.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            target.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            target_normals.storage.as_any().downcast_ref::<GpuStorage<f32>>()
        ) {
            let src_gpu = Tensor { storage: src_storage.clone(), shape: source.shape, dtype: source.dtype, _phantom: PhantomData };
            let tgt_gpu = Tensor { storage: tgt_storage.clone(), shape: target.shape, dtype: target.dtype, _phantom: PhantomData };
            let n_gpu = Tensor { storage: norm_storage.clone(), shape: target_normals.shape, dtype: target_normals.dtype, _phantom: PhantomData };

            crate::gpu_kernels::icp::icp_accumulate(self, &src_gpu, &tgt_gpu, &n_gpu, correspondences, transform)
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let (Some(src_storage), Some(tgt_storage)) = (
            source_depth.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            target_data.storage.as_any().downcast_ref::<GpuStorage<f32>>()
        ) {
            let src_gpu = Tensor { storage: src_storage.clone(), shape: source_depth.shape, dtype: source_depth.dtype, _phantom: PhantomData };
            let tgt_gpu = Tensor { storage: tgt_storage.clone(), shape: target_data.shape, dtype: target_data.dtype, _phantom: PhantomData };

            crate::gpu_kernels::icp::dense_step(self, &src_gpu, &tgt_gpu, intrinsics, initial_guess, max_dist, max_angle)
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::akaze::akaze_diffusion(self, &input_gpu, k, tau)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU result".into()))
            }
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn akaze_derivatives<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
    ) -> crate::Result<(Tensor<f32, S>, Tensor<f32, S>, Tensor<f32, S>)> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            let (lx_gpu, ly_gpu, ldet_gpu) = crate::gpu_kernels::akaze::akaze_derivatives(self, &input_gpu)?;

            let lx_any = Box::new(lx_gpu.storage).boxed_any();
            let ly_any = Box::new(ly_gpu.storage).boxed_any();
            let ldet_any = Box::new(ldet_gpu.storage).boxed_any();

            if let (Ok(lx_s), Ok(ly_s), Ok(ldet_s)) = (lx_any.downcast::<S>(), ly_any.downcast::<S>(), ldet_any.downcast::<S>()) {
                Ok((
                    Tensor { storage: *lx_s, shape: lx_gpu.shape, dtype: lx_gpu.dtype, _phantom: PhantomData },
                    Tensor { storage: *ly_s, shape: ly_gpu.shape, dtype: ly_gpu.dtype, _phantom: PhantomData },
                    Tensor { storage: *ldet_s, shape: ldet_gpu.shape, dtype: ldet_gpu.dtype, _phantom: PhantomData }
                ))
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU results".into()))
            }
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }

    fn akaze_contrast_k<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
    ) -> crate::Result<f32> {
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(input_storage) = input.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let input_gpu = Tensor {
                storage: input_storage.clone(),
                shape: input.shape,
                dtype: input.dtype,
                _phantom: PhantomData,
            };

            crate::gpu_kernels::akaze::akaze_contrast_k(self, &input_gpu)
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let Some(x_storage) = x.storage.as_any().downcast_ref::<GpuStorage<f32>>() {
            let x_gpu = Tensor {
                storage: x_storage.clone(),
                shape: x.shape,
                dtype: x.dtype,
                _phantom: PhantomData,
            };

            let result_gpu = crate::gpu_kernels::sparse::spmv(self, row_ptr, col_indices, values, &x_gpu)?;

            let storage_any = Box::new(result_gpu.storage).boxed_any();
            if let Ok(storage_s) = storage_any.downcast::<S>() {
                Ok(Tensor {
                    storage: *storage_s,
                    shape: result_gpu.shape,
                    dtype: result_gpu.dtype,
                    _phantom: PhantomData,
                })
            } else {
                Err(crate::Error::InvalidInput("Failed to downcast GPU result".into()))
            }
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
        use crate::storage::GpuStorage;
        use std::marker::PhantomData;

        if let (Some(frame_storage), Some(model_storage), Some(mask_storage)) = (
            frame.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            model.storage.as_any().downcast_ref::<GpuStorage<f32>>(),
            mask.storage.as_any().downcast_ref::<GpuStorage<u32>>()
        ) {
            let frame_gpu = Tensor { storage: frame_storage.clone(), shape: frame.shape, dtype: frame.dtype, _phantom: PhantomData };
            let mut model_gpu = Tensor { storage: model_storage.clone(), shape: model.shape, dtype: model.dtype, _phantom: PhantomData };
            let mut mask_gpu = Tensor { storage: mask_storage.clone(), shape: mask.shape, dtype: mask.dtype, _phantom: PhantomData };

            crate::gpu_kernels::mog2_update(self, &frame_gpu, &mut model_gpu, &mut mask_gpu, params)
        } else {
            Err(crate::Error::InvalidInput("GpuContext requires GpuStorage tensors".into()))
        }
    }
}

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
    pub fn global() -> crate::Result<&'static GpuContext> {
        GLOBAL_CONTEXT.get_or_init(|| {
            Self::new()
        }).as_ref().map_err(|e| crate::Error::InitError(e.to_string()))
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
