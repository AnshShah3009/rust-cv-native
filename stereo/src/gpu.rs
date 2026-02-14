//! GPU-accelerated stereo processing
//!
//! Provides GPU implementations of stereo matching algorithms using wgpu.

use crate::{DisparityMap, StereoError, Result, StereoParams};
use image::GrayImage;

/// GPU-accelerated stereo matcher
pub struct GpuStereoMatcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    stereo_pipeline: wgpu::ComputePipeline,
    params_bind_group_layout: wgpu::BindGroupLayout,
}

/// Stereo matching algorithm type
#[derive(Clone, Copy)]
pub enum GpuStereoAlgorithm {
    BlockMatching { block_size: u32 },
    Sgm,
}

impl GpuStereoMatcher {
    pub async fn new(algorithm: GpuStereoAlgorithm) -> Result<Self> {
        // Initialize wgpu
        let instance = wgpu::Instance::default();
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| StereoError::InvalidParameters(
                "No suitable GPU adapter found".to_string()
            ))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Stereo GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| StereoError::InvalidParameters(
                format!("Failed to create GPU device: {}", e)
            ))?;

        // Create bind group layout for stereo parameters
        let params_bind_group_layout = 
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Stereo Params Bind Group Layout"),
                entries: &[
                    // Left image
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Right image
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Disparity output
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Parameters uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Stereo Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                include_str!("shaders/stereo.wgsl")
            )),
        });

        // Create compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Stereo Pipeline Layout"),
            bind_group_layouts: &[&params_bind_group_layout],
            push_constant_ranges: &[],
        });

        let stereo_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Stereo Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            device,
            queue,
            stereo_pipeline,
            params_bind_group_layout,
        })
    }

    pub fn compute_disparity(
        &self,
        left: &GrayImage,
        right: &GrayImage,
        min_disparity: i32,
        max_disparity: i32,
    ) -> Result<DisparityMap> {
        let width = left.width();
        let height = left.height();

        // Create textures
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Create input textures
        let left_texture = self.create_input_texture(left, texture_size);
        let right_texture = self.create_input_texture(right, texture_size);

        // Create output texture for disparity
        let disparity_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Disparity Output"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Create parameter buffer
        let params = StereoParamsGPU {
            width,
            height,
            min_disparity: min_disparity as u32,
            max_disparity: max_disparity as u32,
            block_size: 11,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stereo Parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Stereo Bind Group"),
            layout: &self.params_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &left_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &right_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &disparity_texture.create_view(&wgpu::TextureViewDescriptor::default())
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and compute pass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Stereo Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Stereo Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.stereo_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch compute shader
            let workgroup_size = 16;
            let dispatch_x = (width + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (height + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Copy result to buffer
        let output_buffer_size = (width * height * 4) as wgpu::BufferAddress;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &disparity_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
            },
            texture_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = output_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let disparity_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        output_buffer.unmap();

        // Create disparity map
        let mut disparity = DisparityMap::new(width, height, min_disparity, max_disparity);
        disparity.data = disparity_data;

        Ok(disparity)
    }

    fn create_input_texture(
        &self,
        image: &GrayImage,
        size: wgpu::Extent3d,
    ) -> wgpu::Texture {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Input Image"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING 
                 | wgpu::TextureUsages::COPY_DST
                 | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        // Upload image data
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image.as_raw(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(image.width()),
                rows_per_image: Some(image.height()),
            },
            size,
        );

        texture
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct StereoParamsGPU {
    width: u32,
    height: u32,
    min_disparity: u32,
    max_disparity: u32,
    block_size: u32,
}

/// Check if GPU acceleration is available
pub async fn is_gpu_available() -> bool {
    let instance = wgpu::Instance::default();
    instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_test_stereo_pair() -> (GrayImage, GrayImage) {
        let width = 64u32;
        let height = 64u32;
        
        let mut left = GrayImage::new(width, height);
        let mut right = GrayImage::new(width, height);
        
        // Create pattern with horizontal shift
        for y in 0..height {
            for x in 0..width {
                let pattern = ((x / 8) % 2) * 200;
                left.put_pixel(x, y, Luma([pattern as u8]));
                
                let shifted_x = if x >= 5 { x - 5 } else { 0 };
                let right_pattern = ((shifted_x / 8) % 2) * 200;
                right.put_pixel(x, y, Luma([right_pattern as u8]));
            }
        }
        
        (left, right)
    }

    #[tokio::test]
    async fn test_gpu_availability() {
        let available = is_gpu_available().await;
        println!("GPU available: {}", available);
    }
}
