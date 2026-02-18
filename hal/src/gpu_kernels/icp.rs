use cv_core::Tensor;
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use wgpu::util::DeviceExt;
use std::marker::PhantomData;
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ICPParams {
    num_src: u32,
    num_tgt: u32,
    max_dist_sq: f32,
}

pub fn icp_correspondences(
    ctx: &GpuContext,
    src: &Tensor<f32, GpuStorage<f32>>,
    tgt: &Tensor<f32, GpuStorage<f32>>,
    max_dist: f32,
) -> Result<Vec<(usize, usize, f32)>> {
    let num_src = src.shape.height;
    let num_tgt = tgt.shape.height;

    if num_src == 0 || num_tgt == 0 {
        return Ok(Vec::new());
    }

    let byte_size = (num_src * 16) as u64; // [src_idx, tgt_idx, dist_sq, valid]
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ICP Correspondence Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = ICPParams {
        num_src: num_src as u32,
        num_tgt: num_tgt as u32,
        max_dist_sq: max_dist * max_dist,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ICP Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader_source = include_str!("../../shaders/icp_correspondence.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ICP Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.storage.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tgt.storage.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (num_src as u32 + 63) / 64;
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);

    let raw_results: Vec<[f32; 4]> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &output_buffer,
        0,
        byte_size as usize,
    ))?;

    let mut correspondences = Vec::new();
    for res in raw_results {
        if res[3] > 0.5 {
            correspondences.push((res[0] as usize, res[1] as usize, res[2].sqrt()));
        }
    }

    Ok(correspondences)
}
