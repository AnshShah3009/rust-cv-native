use cv_core::Tensor;
use cv_core::storage::Storage;
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::context::MorphologyType;
use crate::Result;
use wgpu::util::DeviceExt;
use std::sync::Arc;
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MorphParams {
    width: u32,
    height: u32,
    kw: u32,
    kh: u32,
    typ: u32,
    iterations: u32,
}

pub fn morphology(
    ctx: &GpuContext,
    input: &Tensor<u8, GpuStorage<u8>>,
    typ: MorphologyType,
    kernel: &Tensor<u8, GpuStorage<u8>>,
    iterations: u32,
) -> Result<Tensor<u8, GpuStorage<u8>>> {
    let len = input.shape.len();
    let (h, w) = input.shape.hw();
    let (kh, kw) = kernel.shape.hw();
    let byte_size = ((len + 3) / 4 * 4) as u64;

    // We implement Open/Close by calling Erode/Dilate twice for now
    if typ == MorphologyType::Open || typ == MorphologyType::Close {
        let (op1, op2) = if typ == MorphologyType::Open {
            (MorphologyType::Erode, MorphologyType::Dilate)
        } else {
            (MorphologyType::Dilate, MorphologyType::Erode)
        };
        let tmp = morphology(ctx, input, op1, kernel, iterations)?;
        return morphology(ctx, &tmp, op2, kernel, iterations);
    }

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Morphology Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Kernel mask needs to be u32 array for shader
    // We convert the kernel tensor (u8) to u32 on GPU? 
    // Ideally we should have uploaded it as u32. 
    // For now, let's assume kernel is small and re-upload as u32 if it's CPU based, 
    // but the trait says it's a Tensor<u8, S>.
    // If S is GpuStorage, we have a problem (it's u8 on GPU).
    // Let's implement a quick fix: copy kernel to CPU, convert to u32, upload.
    // Or add a helper to HAL to convert buffers on GPU.
    // For now, assume kernel is CPU based (common).
    
    let k_data_u8 = kernel.storage.as_slice().ok_or_else(|| crate::Error::NotSupported("GPU Morphology currently requires CPU-based kernel mask".into()))?;
    let k_data_u32: Vec<u32> = k_data_u8.iter().map(|&v| v as u32).collect();
    
    let k_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Morph Kernel Mask"),
        contents: bytemuck::cast_slice(&k_data_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let params = MorphParams {
        width: w as u32,
        height: h as u32,
        kw: kw as u32,
        kh: kh as u32,
        typ: if typ == MorphologyType::Erode { 0 } else { 1 },
        iterations: 1, // We handle iterations by looping in host or in shader (looping in host is easier for now)
    };

    let shader_source = include_str!("../../shaders/morphology.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let current_input = input.storage.buffer.clone();
    let current_output = Arc::new(output_buffer);

    for _ in 0..iterations {
        let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Morph Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Morph Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: current_input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: current_output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = ((w as u32 + 3) / 4 + 15) / 16;
            let wg_y = (h as u32 + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        ctx.submit(encoder);
        
        // Setup for next iteration
        if iterations > 1 {
            // Need to copy current_output back to current_input? 
            // Better: use a temporary ping-pong buffer.
            // For now, let's just use one copy (slow but works).
            let mut copy_encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            copy_encoder.copy_buffer_to_buffer(&current_output, 0, &current_input, 0, byte_size);
            ctx.submit(copy_encoder);
        }
    }

    Ok(Tensor {
        storage: GpuStorage::from_buffer(current_output, len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
