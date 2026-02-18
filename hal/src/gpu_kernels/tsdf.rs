use cv_core::{Tensor, TensorShape};
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use wgpu::util::DeviceExt;
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TsdfParams {
    voxel_size: f32,
    truncation_dist: f32,
    image_w: u32,
    image_h: u32,
    vol_x: u32,
    vol_y: u32,
    vol_z: u32,
    padding: u32,
}

pub fn integrate(
    ctx: &GpuContext,
    depth_image: &Tensor<f32, GpuStorage<f32>>,
    camera_pose: &[[f32; 4]; 4],
    intrinsics: &[f32; 4],
    tsdf_volume: &mut Tensor<f32, GpuStorage<f32>>,
    weight_volume: &mut Tensor<f32, GpuStorage<f32>>,
    voxel_size: f32,
    truncation: f32,
) -> Result<()> {
    let (h, w) = depth_image.shape.hw();
    let vol_shape = tsdf_volume.shape;
    let (vx, vy, vz) = (vol_shape.width as u32, vol_shape.height as u32, vol_shape.channels as u32);

    let params = TsdfParams {
        voxel_size,
        truncation_dist: truncation,
        image_w: w as u32,
        image_h: h as u32,
        vol_x: vx,
        vol_y: vy,
        vol_z: vz,
        padding: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TSDF Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let intrinsics_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TSDF Intrinsics"),
        contents: bytemuck::cast_slice(intrinsics),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let mut pose_flat = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            pose_flat[c * 4 + r] = camera_pose[r][c];
        }
    }
    let pose_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TSDF Pose"),
        contents: bytemuck::cast_slice(&pose_flat),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader_source = include_str!("tsdf_integrate.wgsl");
    let pipeline = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("TSDF Integrate Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let compute_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("TSDF Integrate Pipeline"),
        layout: None,
        module: &pipeline,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("TSDF BG 0"),
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: depth_image.storage.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: depth_image.storage.buffer.as_entire_binding() }, // Dummy color
            wgpu::BindGroupEntry { binding: 2, resource: tsdf_volume.storage.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: weight_volume.storage.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: tsdf_volume.storage.buffer.as_entire_binding() }, // Dummy color out
        ],
    });

    let bind_group_1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("TSDF BG 1"),
        layout: &compute_pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pose_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: intrinsics_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("TSDF Integrate") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group_0, &[]);
        pass.set_bind_group(1, &bind_group_1, &[]);
        pass.dispatch_workgroups((vx + 7) / 8, (vy + 7) / 8, (vz + 7) / 8);
    }
    ctx.submit(encoder);

    Ok(())
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RaycastParams {
    width: u32,
    height: u32,
    voxel_size: f32,
    truncation: f32,
    step_factor: f32,
    min_depth: f32,
    max_depth: f32,
    vol_x: u32,
    vol_y: u32,
    vol_z: u32,
    padding: u32,
}

pub fn raycast(
    ctx: &GpuContext,
    tsdf_volume: &Tensor<f32, GpuStorage<f32>>,
    camera_pose: &[[f32; 4]; 4], // Camera-to-World
    intrinsics: &[f32; 4], // fx, fy, cx, cy
    image_size: (u32, u32),
    depth_range: (f32, f32),
    voxel_size: f32,
    truncation: f32,
) -> Result<(Tensor<f32, GpuStorage<f32>>, Tensor<f32, GpuStorage<f32>>)> {
    let (w, h) = image_size;
    let vol_shape = tsdf_volume.shape;
    let (vx, vy, vz) = (vol_shape.width as u32, vol_shape.height as u32, vol_shape.channels as u32);

    let output_depth = ctx.get_buffer((w * h * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let output_normals = ctx.get_buffer((w * h * 16) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);

    let params = RaycastParams {
        width: w,
        height: h,
        voxel_size,
        truncation,
        step_factor: 0.5, // Conservative stepping
        min_depth: depth_range.0,
        max_depth: depth_range.1,
        vol_x: vx,
        vol_y: vy,
        vol_z: vz,
        padding: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Raycast Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let mut pose_flat = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            pose_flat[c * 4 + r] = camera_pose[r][c]; // Column-major
        }
    }
    let pose_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Raycast Pose"),
        contents: bytemuck::cast_slice(&pose_flat),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let inv_intrinsics = [1.0 / intrinsics[0], 1.0 / intrinsics[1], intrinsics[2], intrinsics[3]];
    let intrinsics_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Raycast Intrinsics"),
        contents: bytemuck::cast_slice(&inv_intrinsics),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader_source = include_str!("../../shaders/tsdf_raycast.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Raycast Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tsdf_volume.storage.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_depth.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_normals.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pose_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: intrinsics_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Raycast Dispatch") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((w + 15) / 16, (h + 15) / 16, 1);
    }
    ctx.submit(encoder);

    use cv_core::TensorShape;
    use std::marker::PhantomData;

    let depth_tensor = Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_depth), (w * h * 4) as usize),
        shape: TensorShape::new(1, h as usize, w as usize),
        dtype: cv_core::DType::F32,
        _phantom: PhantomData,
    };
    
    let normal_tensor = Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_normals), (w * h * 16) as usize),
        shape: TensorShape::new(4, h as usize, w as usize), // 4 channels (x, y, z, pad)
        dtype: cv_core::DType::F32,
        _phantom: PhantomData,
    };

    Ok((depth_tensor, normal_tensor))
}
