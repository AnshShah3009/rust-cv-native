use wgpu::{Device, Queue, Instance, RequestAdapterOptions, PowerPreference, Backends};
use std::sync::{Arc, OnceLock};
use futures::executor::block_on;
use crate::context::{ComputeContext, BorderMode, ThresholdType, MorphologyType, WarpType};
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
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
            // SAFETY: TypeId check ensures S is GpuStorage<f32>, so memory layout is identical
            let input_ptr = input as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            let kernel_ptr = kernel as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
            
            let input_gpu = unsafe { &*input_ptr };
            let kernel_gpu = unsafe { &*kernel_ptr };

            let result_gpu = crate::gpu_kernels::convolve::convolve_2d(self, input_gpu, kernel_gpu, border_mode)?;

            // Move result_gpu into Tensor<f32, S>
            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<f32, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
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
        use std::any::TypeId;
        use crate::storage::GpuStorage;

        if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
            // SAFETY: TypeId check ensures S is GpuStorage<u8>, so memory layout is identical
            let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
            let input_gpu = unsafe { &*input_ptr };

            let result_gpu = crate::gpu_kernels::threshold::threshold(self, input_gpu, thresh, max_value, typ)?;

            // Move result_gpu into Tensor<u8, S>
            let result = unsafe { std::ptr::read(&result_gpu as *const _ as *const Tensor<u8, S>) };
            std::mem::forget(result_gpu);
            Ok(result)
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
        _input: &Tensor<u8, S>,
        _matrix: &[[f32; 3]; 3],
        _new_shape: (usize, usize),
        _typ: WarpType,
    ) -> crate::Result<Tensor<u8, S>> {
        Err(crate::Error::NotSupported("GPU Warp pending implementation".into()))
    }

    fn nms<S: Storage<f32> + 'static>(
        &self,
        _input: &Tensor<f32, S>,
        _threshold: f32,
        _window_size: usize,
    ) -> crate::Result<Tensor<f32, S>> {
        Err(crate::Error::NotSupported("GPU Pixel-wise NMS pending implementation".into()))
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
        _points: &Tensor<f32, S>,
        _transform: &[[f32; 4]; 4],
    ) -> crate::Result<Tensor<f32, S>> {
        Err(crate::Error::NotSupported("GPU pointcloud_transform pending implementation".into()))
    }

    fn pointcloud_normals<S: Storage<f32> + 'static>(
        &self,
        _points: &Tensor<f32, S>,
        _k_neighbors: u32,
    ) -> crate::Result<Tensor<f32, S>> {
        Err(crate::Error::NotSupported("GPU pointcloud_normals pending implementation".into()))
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
    ) -> crate::Result<()> {
        Err(crate::Error::NotSupported("GPU tsdf_integrate pending implementation".into()))
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
        // Create instance
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
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
}
