use cv_core::{Tensor, TensorShape};
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use wgpu::util::DeviceExt;
use std::sync::Arc;

// Placeholder for full tables. In production, these should be complete.
// edge_table[256]
// tri_table[256][16] flattened to [4096]
mod tables {
    pub const EDGE_TABLE: [u32; 256] = [0; 256]; // TODO: Fill me
    pub const TRI_TABLE: [i32; 4096] = [-1; 4096]; // TODO: Fill me
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct McParams {
    vol_x: u32,
    vol_y: u32,
    vol_z: u32,
    voxel_size: f32,
    iso_level: f32,
    max_triangles: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pos: [f32; 4],
    norm: [f32; 4],
}

pub fn extract_mesh(
    ctx: &GpuContext,
    tsdf_volume: &Tensor<f32, GpuStorage<f32>>,
    voxel_size: f32,
    iso_level: f32,
    max_triangles: u32,
) -> Result<Vec<Vertex>> {
    let vol_shape = tsdf_volume.shape;
    let (vx, vy, vz) = (vol_shape.width as u32, vol_shape.height as u32, vol_shape.channels as u32);

    // Params
    let params = McParams {
        vol_x: vx,
        vol_y: vy,
        vol_z: vz,
        voxel_size,
        iso_level,
        max_triangles,
    };
    
    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MC Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Tables
    // Note: In a real impl, we'd use the real tables.
    let edge_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MC Edge Table"),
        contents: bytemuck::cast_slice(&tables::EDGE_TABLE),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    
    let tri_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MC Tri Table"),
        contents: bytemuck::cast_slice(&tables::TRI_TABLE),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Output buffer
    // 3 vertices per triangle * max_triangles * 32 bytes per vertex
    let output_size = (max_triangles as u64) * 3 * 32;
    let vertices_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MC Vertices"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    // Counter
    let counter_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MC Counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Reset counter
    ctx.queue.write_buffer(&counter_buffer, 0, &[0, 0, 0, 0]);

    let shader_source = include_str!("../../shaders/marching_cubes.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tsdf_volume.storage.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vertices_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: counter_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: edge_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: tri_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("MC Dispatch") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((vx + 7) / 8, (vy + 7) / 8, (vz + 7) / 8);
    }
    ctx.submit(encoder);

    // Read counter
    let count_bytes: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &counter_buffer,
        0,
        4,
    ))?;
    let num_verts = count_bytes[0] as u64;

    if num_verts == 0 {
        return Ok(Vec::new());
    }

    // Read vertices
    let vert_data: Vec<Vertex> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &vertices_buffer,
        0,
        (num_verts * 32) as usize,
    ))?;

    Ok(vert_data)
}
