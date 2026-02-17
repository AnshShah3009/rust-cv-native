use cv_core::Tensor;
use crate::context::BorderMode;
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use wgpu::util::DeviceExt;
use std::sync::Arc;
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvolveParams {
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
    border_mode: u32,
    border_const: f32,
}

pub fn convolve_2d(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    kernel: &Tensor<f32, GpuStorage<f32>>,
    border_mode: BorderMode,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (h, w) = input.shape.hw();
    let (kh, kw) = kernel.shape.hw();
    
    // Create output buffer
    let output_size = input.shape.len();
    let output_byte_size = (output_size * std::mem::size_of::<f32>()) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Convolve Output"),
        size: output_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Prepare params
    let (mode_int, const_val) = match border_mode {
        BorderMode::Constant(v) => (0, v),
        BorderMode::Replicate => (1, 0.0),
        BorderMode::Reflect => (2, 0.0), // Note: Shader implements Reflect (not Reflect101 perfectly?)
        BorderMode::Wrap => (3, 0.0),
    };

    let params = ConvolveParams {
        width: w as u32,
        height: h as u32,
        kernel_width: kw as u32,
        kernel_height: kh as u32,
        border_mode: mode_int,
        border_const: const_val,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Convolve Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Pipeline setup
    // For now we re-create pipeline. Ideally cached.
    let shader_source = include_str!("../../shaders/convolve_2d.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    // Bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Convolve Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: kernel.storage.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Convolve Dispatch"),
    });
    
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Convolve Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        
        let wg_x = (w as u32 + 15) / 16;
        let wg_y = (h as u32 + 15) / 16;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), output_size),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
