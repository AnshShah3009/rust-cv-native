use wgpu::{Device, Queue, Instance, RequestAdapterOptions, PowerPreference, Backends};
use std::sync::Arc;
use futures::executor::block_on;

/// Shared GPU Context containing Device and Queue.
#[derive(Debug)]
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl GpuContext {
    /// Initialize a new GPU context.
    /// Selects the best available adapter (HighPerformance).
    pub fn new() -> Option<Self> {
        // Create instance
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        // Request adapter
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).ok()?;

        // Request device (wgpu 28: no trace param)
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("CV-HAL Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        )).ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
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
            Some(c) => println!("GPU Context created: {:?}", c.device),
            None => println!("No suitable GPU adapter found, skipping test."),
        }
    }
}
