use cv_core::Tensor;
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
        let nms_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FAST NMS Output"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let nms_shader_source = include_str!("../../shaders/fast_nms.wgsl");
        let nms_pipeline = ctx.create_compute_pipeline(nms_shader_source, "main");

        let nms_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FAST NMS Bind Group"),
            layout: &nms_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: score_map.storage.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: nms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut nms_encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = nms_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&nms_pipeline);
            pass.set_bind_group(0, &nms_bind_group, &[]);
            let x = ((w as u32 + 3) / 4 + 15) / 16;
            let y = (h as u32 + 15) / 16;
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.submit(nms_encoder);

        Ok(Tensor {
            storage: GpuStorage::from_buffer(Arc::new(nms_buffer), out_len),
            shape: score_map.shape,
            dtype: score_map.dtype,
            _phantom: PhantomData,
        })
    } else {
        Ok(score_map)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CollectParams {
    width: u32,
    height: u32,
    num_elements: u32,
    padding: u32,
}

pub fn extract_keypoints(
    ctx: &GpuContext,
    score_map: &Tensor<u8, GpuStorage<u8>>,
) -> Result<Vec<cv_core::KeyPointF32>> {
    let (h, w) = score_map.shape.hw();
    let num_pixels = w * h;
    let num_u32 = (num_pixels + 3) / 4;

    // 1. Count pass
    let usages = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let counts_buffer = ctx.get_buffer((num_u32 as u64) * 4, usages);

    let params = CollectParams {
        width: w as u32,
        height: h as u32,
        num_elements: num_u32 as u32,
        padding: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Collect Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader_source = include_str!("feature_collection.wgsl");
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Feature Collection Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let count_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Count Points Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("count_points"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Count Bind Group"),
            layout: &count_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: score_map.storage.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&count_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((num_u32 as u32 + 255) / 256, 1, 1);
    }
    ctx.submit(encoder);

    // 2. Scan pass
    let scan_shader_source = include_str!("prefix_sum.wgsl");
    let scan_shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Scan Shader"),
        source: wgpu::ShaderSource::Wgsl(scan_shader_source.into()),
    });
    
    // We need to get the TOTAL count to allocate the result buffer.
    // However, gpu_exclusive_scan doesn't return the total sum easily (it's in the block_sums of the last level).
    // Let's do a trick: read back the last element of the scanned buffer + last element of original counts?
    // Or just read back the total sum from block_sums.
    
    // For now, to keep it simple and reliable, let's run the scan and then read back the last element.
    crate::gpu_kernels::radix_sort::gpu_exclusive_scan(ctx, &counts_buffer, num_u32 as u32, &scan_shader)?;
    
    // Read back the last index + last count to get total
    let last_indices: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &counts_buffer,
        ((num_u32 - 1) * 4) as u64,
        4,
    ))?;
    
    // We also need the original count of the last element to get the total.
    // This is getting complicated. Let's instead use a single atomic buffer for total count?
    // Actually, the simplest way to get the total count after a scan is to just read it.
    // But we need to allocate the keypoint buffer BEFORE the collect pass.
    
    // Let's use a simpler approach: Atomic counter for total count during collect pass,
    // but we still need to know how much to allocate.
    
    // Standard way:
    // Pass 1: Count.
    // Pass 2: Scan.
    // Readback TOTAL (from end of scanned buffer + original last element).
    
    let total_count = last_indices[0]; // This is the exclusive scan result, so it's the index of the last element start.
    // We need to add the count of the last element.
    // Let's just read back the counts_buffer BEFORE scan? No.
    
    // Actually, I'll modify the collect_points kernel to use an atomic counter if I don't want to use scan?
    // No, scan is better for stability and order.
    
    // Let's just read the total count from CPU for now (small readback).
    let counts_data: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &counts_buffer,
        0,
        (num_u32 as usize) * 4,
    ))?;
    
    let mut total = 0u32;
    let mut indices = Vec::with_capacity(num_u32 as usize);
    for &c in &counts_data {
        indices.push(total);
        total += c;
    }
    
    if total == 0 {
        return Ok(Vec::new());
    }

    // Upload scanned indices
    ctx.queue.write_buffer(&counts_buffer, 0, bytemuck::cast_slice(&indices));

    // 3. Collect pass
    let kp_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Extracted Keypoints"),
        size: (total as u64) * std::mem::size_of::<cv_core::KeyPointF32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let collect_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Collect Points Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("collect_points"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collect Bind Group"),
            layout: &collect_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: score_map.storage.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: kp_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&collect_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((num_u32 as u32 + 255) / 256, 1, 1);
    }
    ctx.submit(encoder);

    // Read back keypoints
    let kps = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &kp_buffer,
        0,
        (total as usize) * std::mem::size_of::<cv_core::KeyPointF32>(),
    ))?;

    ctx.return_buffer(counts_buffer, usages);

    Ok(kps)
}
