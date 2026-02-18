use cv_hal::gpu::GpuContext;
use cv_core::{Tensor, TensorShape, DataType};
use cv_hal::storage::GpuStorage;
use cv_hal::context::ComputeContext;
use cv_hal::tensor_ext::{TensorToGpu, TensorToCpu};
use std::sync::Arc;

#[test]
fn test_gpu_nms_pixel() {
    let ctx = match GpuContext::global() {
        Some(ctx) => ctx,
        None => return,
    };

    let shape = TensorShape::new(1, 100, 100);
    let mut data = vec![0.0f32; 10000];
    data[50 * 100 + 50] = 1.0;
    data[50 * 100 + 51] = 0.9; // Should be suppressed
    
    let input = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(data, shape).to_gpu_ctx(ctx).unwrap();
    let output = ctx.nms(&input, 0.5, 3).unwrap();
    
    let out_cpu: Tensor<f32, cv_core::storage::CpuStorage<f32>> = output.to_cpu_ctx(ctx).unwrap();
    let out_slice = out_cpu.as_slice();
    
    assert!(out_slice[50 * 100 + 50] > 0.0);
    assert_eq!(out_slice[50 * 100 + 51], 0.0);
}

#[test]
fn test_gpu_tsdf_pipeline() {
    let ctx = match GpuContext::global() {
        Some(ctx) => ctx,
        None => return,
    };

    // 1. Setup Voxel Volume (channels=2: TSDF, weight)
    let vol_shape = TensorShape::new(2, 64, 64); // width=64, height=64, channels=2? No, TensorShape is (C, H, W)
    // Wait, TensorShape(C, H, W) or (W, H, C)?
    // core/src/tensor.rs: pub struct TensorShape { pub channels: usize, pub height: usize, pub width: usize }
    // Standard is (C, H, W).
    // For a volume, we use width, height as X, Y and channels as Z.
    // But we need 2 values per voxel. 
    // Let's use shape (2, H * W, Z) or similar? 
    // Actually, our TSDF expects (vx, vy, vz) from shape.width, height, channels.
    // If we want 2 values per voxel, we need another dimension or double one.
    // Let's use channels = vz * 2.
    
    let vx = 64; let vy = 64; let vz = 64;
    let vol_shape = TensorShape::new(vz * 2, vy, vx);
    let mut voxels = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(vec![0.0f32; vx*vy*vz*2], vol_shape).to_gpu_ctx(ctx).unwrap();

    // 2. Mock Depth Frame (plane at z=2.0)
    let img_w = 160;
    let img_h = 120;
    let depth_data = vec![2.0f32; img_w * img_h];
    let depth_img = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(depth_data, TensorShape::new(1, img_h, img_w)).to_gpu_ctx(ctx).unwrap();

    let camera_pose = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let intrinsics = [100.0, 100.0, 80.0, 60.0];

    // 3. Integrate
    ctx.tsdf_integrate(&depth_img, &camera_pose, &intrinsics, &mut voxels, 0.05, 0.1).unwrap();

    // 4. Raycast
    let ray_result = ctx.tsdf_raycast(&voxels, &camera_pose, &intrinsics, (img_w as u32, img_h as u32), (0.1, 5.0), 0.05, 0.1).unwrap();

    let rr_cpu: Tensor<f32, cv_core::storage::CpuStorage<f32>> = ray_result.to_cpu_ctx(ctx).unwrap();
    // Channels=4, so pixel (80, 60) is at index (60 * 160 + 80) * 4
    let depth_val = rr_cpu.as_slice()[(60 * 160 + 80) * 4];
    assert!(depth_val > 1.8 && depth_val < 2.2, "Raycast depth {} should be approx 2.0", depth_val);

    // 5. Extract Mesh
    let mesh = ctx.tsdf_extract_mesh(&voxels, 0.05, 0.0, 10000).unwrap();
    println!("Extracted {} vertices", mesh.len());
}

#[test]
fn test_gpu_optical_flow_multi_level() {
    let ctx = match GpuContext::global() {
        Some(ctx) => ctx,
        None => return,
    };

    let shape = TensorShape::new(1, 128, 128);
    let mut data1 = vec![0.0f32; 128 * 128];
    for y in 60..70 { for x in 60..70 { data1[y*128+x] = 1.0; } }
    
    let mut data2 = vec![0.0f32; 128 * 128];
    for y in 62..72 { for x in 62..72 { data2[y*128+x] = 1.0; } }

    let t1 = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(data1, shape).to_gpu_ctx(ctx).unwrap();
    let t2 = Tensor::<f32, cv_core::storage::CpuStorage<f32>>::from_vec(data2, shape).to_gpu_ctx(ctx).unwrap();
    
    let p1 = vec![t1];
    let p2 = vec![t2];
    
    let pts = vec![[65.0, 65.0]];
    let tracked = ctx.optical_flow_lk(&p1, &p2, &pts, 11, 10).unwrap();
    
    let dx = tracked[0][0] - pts[0][0];
    let dy = tracked[0][1] - pts[0][1];
    
    assert!(dx.abs() > 1.0, "Tracked dx {} should be approx 2.0", dx);
    assert!(dy.abs() > 1.0, "Tracked dy {} should be approx 2.0", dy);
}
