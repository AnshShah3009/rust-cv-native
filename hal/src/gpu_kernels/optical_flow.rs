use cv_core::{Tensor, TensorShape};
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use wgpu::util::DeviceExt;
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LkParams {
    num_points: u32,
    window_radius: i32,
    max_iters: u32,
    min_eigenvalue: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Point {
    x: f32,
    y: f32,
}

pub fn lucas_kanade(
    ctx: &GpuContext,
    prev_pyramid: &Tensor<f32, GpuStorage<f32>>,
    next_pyramid: &Tensor<f32, GpuStorage<f32>>,
    points: &[ [f32; 2] ],
    window_size: usize,
    max_iters: u32,
) -> Result<Vec<[f32; 2]>> {
    let num_points = points.len();
    if num_points == 0 { return Ok(Vec::new()); }

    let points_vec: Vec<Point> = points.iter().map(|p| Point { x: p[0], y: p[1] }).collect();
    let points_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("LK Input Points"),
        contents: bytemuck::cast_slice(&points_vec),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("LK Output Points"),
        size: (num_points * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&output_buffer, 0, bytemuck::cast_slice(&points_vec));

    let params = LkParams {
        num_points: num_points as u32,
        window_radius: (window_size / 2) as i32,
        max_iters,
        min_eigenvalue: 0.001,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("LK Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader_source = include_str!("../../shaders/lucas_kanade.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    // Pyramid levels (assumed stored in single large buffer or handled level by level)
    // For now, let's process just the base level (level 0) or all levels if they are packed.
    // In our current hal, pyramids are often separate Tensors.
    // I will assume for now we are tracking on a single level for simplicity, 
    // or we'll need to update the loop to iterate through levels.

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("LK Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: prev_pyramid.storage.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: next_pyramid.storage.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: points_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { 
                binding: 5, 
                resource: ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[prev_pyramid.shape.width as u32, prev_pyramid.shape.height as u32, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                }).as_entire_binding() 
            },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("LK Dispatch") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((num_points as u32 + 63) / 64, 1, 1);
    }
    ctx.submit(encoder);

    let result_vec: Vec<Point> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &output_buffer,
        0,
        num_points * 8,
    ))?;

    Ok(result_vec.iter().map(|p| [p.x, p.y]).collect())
}
