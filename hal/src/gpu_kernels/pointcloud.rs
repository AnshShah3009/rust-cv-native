use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub fn transform_points(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    transform: &[[f32; 4]; 4],
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let num_points = input.shape.height;
    if num_points == 0 {
        return Ok(input.clone());
    }

    let byte_size = (num_points * 16) as u64; // float4 alignment
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PC Transform Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut transposed = [[0.0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            transposed[c][r] = transform[r][c];
        }
    }

    let transform_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Matrix"),
            contents: bytemuck::cast_slice(&transposed),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let num_points_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC Transform Num Points"),
            contents: bytemuck::bytes_of(&(num_points as u32)),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../gpu_kernels/pointcloud_transform.wgsl");
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PC Transform Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PC Transform BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PC Transform Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PC Transform Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PC Transform Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: transform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: num_points_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (num_points as u32 + 255) / 256;
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), num_points * 4),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}

pub fn compute_normals(
    ctx: &GpuContext,
    points: &Tensor<f32, GpuStorage<f32>>,
    k_neighbors: u32,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let num_points = points.shape.height;
    if num_points == 0 {
        return Ok(points.clone());
    }

    let byte_size = (num_points * 16) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PC Normals Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader_source = include_str!("../gpu_kernels/pointcloud_normals_fast_gpu.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let params = [num_points as f32, k_neighbors as f32, 0.0, 0.0];
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Normals Params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Normals Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (num_points as u32 + 63) / 64;
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), num_points * 4),
        shape: points.shape,
        dtype: points.dtype,
        _phantom: PhantomData,
    })
}
