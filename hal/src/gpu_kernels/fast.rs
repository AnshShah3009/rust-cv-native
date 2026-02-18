use cv_core::{Tensor, TensorShape};
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use wgpu::util::DeviceExt;
use std::marker::PhantomData;
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FastParams {
    width: u32,
    height: u32,
    threshold: u32,
}

pub fn fast_detect(
    ctx: &GpuContext,
    input: &Tensor<u8, GpuStorage<u8>>,
    threshold: u8,
    non_max_suppression: bool,
) -> Result<Tensor<u8, GpuStorage<u8>>> {
    let (h, w) = input.shape.hw();
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported("GPU FAST currently only for grayscale".into()));
    }

    let out_len = w * h;
    let byte_size = ((out_len + 3) / 4 * 4) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("FAST Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = FastParams {
        width: w as u32,
        height: h as u32,
        threshold: threshold as u32,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("FAST Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader_source = include_str!("../../shaders/fast.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FAST Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.storage.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = ((w as u32 + 3) / 4 + 15) / 16;
        let y = (h as u32 + 15) / 16;
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    let score_map = Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    };

    if non_max_suppression {
        // TODO: Implement GPU NMS for FAST
        Ok(score_map)
    } else {
        Ok(score_map)
    }
}
