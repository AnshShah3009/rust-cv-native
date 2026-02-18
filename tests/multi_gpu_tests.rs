use cv_core::{Tensor, TensorShape, storage::Storage};
use cv_hal::gpu::GpuContext;
use cv_hal::cpu::CpuBackend;
use cv_hal::context::{ComputeContext, ThresholdType, ColorConversion};
use cv_hal::tensor_ext::{TensorToGpu, TensorToCpu};
use futures::executor::block_on;
use std::sync::Arc;

#[test]
fn test_cross_device_parity() {
    // 1. Setup CPU reference
    let cpu = CpuBackend::new().expect("CPU backend unavailable");
    
    // 2. Enumerate all GPUs
    let adapters = block_on(GpuContext::enumerate_adapters());
    println!("Found {} GPU adapters", adapters.len());

    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!("Adapter {}: {} ({:?}) - Backend: {:?}", i, info.name, info.device_type, info.backend);
    }

    for (i, adapter) in adapters.into_iter().enumerate() {
        let info = adapter.get_info();
        println!("--- Testing Adapter {}: {} ---", i, info.name);
        
        // Skip problematic AMD Integrated GPU on Vulkan that panics in driver
        if info.name.contains("AMD") && info.backend == wgpu::Backend::Vulkan {
            println!("  ! Skipping AMD Integrated GPU on Vulkan to avoid driver panic");
            continue;
        }

        // Skip llvmpipe
        if info.name.contains("llvmpipe") {
            println!("  ! Skipping software renderer");
            continue;
        }

        let gpu_res = block_on(GpuContext::from_adapter(adapter));
        let gpu = match gpu_res {
            Ok(g) => g,
            Err(e) => {
                println!("  ! Skipping adapter {}: GPU context creation failed: {}", info.name, e);
                continue;
            }
        };
        
        // --- Run Parity Tests ---
        test_threshold_parity(&cpu, &gpu, &info.name);
        test_pc_transform_parity(&cpu, &gpu, &info.name);
        test_color_cvt_parity(&cpu, &gpu, &info.name);
        test_resize_parity(&cpu, &gpu, &info.name);
        test_bilateral_parity(&cpu, &gpu, &info.name);
        test_fast_parity(&cpu, &gpu, &info.name);
    }
}

fn test_fast_parity(cpu: &CpuBackend, gpu: &GpuContext, gpu_name: &str) {
    let shape = TensorShape::new(1, 128, 128);
    let mut data = vec![0u8; shape.len()];
    // Create some corners
    for y in 30..60 {
        for x in 30..60 {
            data[y * 128 + x] = 255;
        }
    }
    
    let input_cpu = Tensor::from_vec(data, shape);
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Upload failed");
    
    // CPU
    let res_cpu = cpu.fast_detect(&input_cpu, 20, false).unwrap();
    let cpu_slice = res_cpu.storage.as_slice().unwrap();
    
    // GPU (Should return NotSupported for now, we handle it gracefully in the test runner)
    let res_gpu_encoded = gpu.fast_detect(&input_gpu, 20, false);
    
    match res_gpu_encoded {
        Ok(res_gpu_t) => {
            let res_gpu = res_gpu_t.to_cpu_ctx(gpu).unwrap();
            let gpu_slice = res_gpu.storage.as_slice().unwrap();
            for i in 0..cpu_slice.len() {
                if cpu_slice[i] != gpu_slice[i] {
                    panic!("FAST Parity failure on {}: at index {}, CPU={}, GPU={}", gpu_name, i, cpu_slice[i], gpu_slice[i]);
                }
            }
            println!("  ✓ FAST parity passed for {}", gpu_name);
        },
        Err(cv_hal::Error::NotSupported(_)) => {
            println!("  - FAST parity skipped for {} (NotSupported)", gpu_name);
        },
        Err(e) => panic!("FAST failed on {}: {}", gpu_name, e),
    }
}

fn test_bilateral_parity(cpu: &CpuBackend, gpu: &GpuContext, gpu_name: &str) {
    let shape = TensorShape::new(1, 64, 64);
    let mut data = vec![0u8; shape.len()];
    for i in 0..data.len() {
        data[i] = (i % 256) as u8;
    }
    
    let input_cpu = Tensor::from_vec(data, shape);
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Upload failed");
    
    // CPU
    let res_cpu = cpu.bilateral_filter(&input_cpu, 5, 10.0, 10.0).unwrap();
    let cpu_slice = res_cpu.storage.as_slice().unwrap();
    
    // GPU
    let res_gpu_encoded = gpu.bilateral_filter(&input_gpu, 5, 10.0, 10.0).unwrap();
    let res_gpu = res_gpu_encoded.to_cpu_ctx(gpu).unwrap();
    let gpu_slice = res_gpu.storage.as_slice().unwrap();
    
    for i in 0..cpu_slice.len() {
        if (cpu_slice[i] as i32 - gpu_slice[i] as i32).abs() > 1 {
            panic!("Bilateral Parity failure on {}: at index {}, CPU={}, GPU={}", gpu_name, i, cpu_slice[i], gpu_slice[i]);
        }
    }
    println!("  ✓ Bilateral parity passed for {}", gpu_name);
}

fn test_resize_parity(cpu: &CpuBackend, gpu: &GpuContext, gpu_name: &str) {
    let shape = TensorShape::new(1, 128, 128);
    let mut data = vec![0u8; shape.len()];
    for i in 0..data.len() {
        data[i] = (i % 256) as u8;
    }
    
    let input_cpu = Tensor::from_vec(data, shape);
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Upload failed");
    
    let new_shape = (64, 64);
    
    // CPU
    let res_cpu = cpu.resize(&input_cpu, new_shape).unwrap();
    let cpu_slice = res_cpu.storage.as_slice().unwrap();
    
    // GPU
    let res_gpu_encoded = gpu.resize(&input_gpu, new_shape).unwrap();
    let res_gpu = res_gpu_encoded.to_cpu_ctx(gpu).unwrap();
    let gpu_slice = res_gpu.storage.as_slice().unwrap();
    
    for i in 0..cpu_slice.len() {
        if cpu_slice[i] != gpu_slice[i] {
            panic!("Resize Parity failure on {}: at index {}, CPU={}, GPU={}", gpu_name, i, cpu_slice[i], gpu_slice[i]);
        }
    }
    println!("  ✓ Resize parity passed for {}", gpu_name);
}

fn test_color_cvt_parity(cpu: &CpuBackend, gpu: &GpuContext, gpu_name: &str) {
    let shape = TensorShape::new(3, 64, 64);
    let mut data = vec![0u8; shape.len()];
    for i in 0..data.len() {
        data[i] = (i % 256) as u8;
    }
    
    let input_cpu = Tensor::from_vec(data, shape);
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Upload failed");
    
    // CPU
    let res_cpu = cpu.cvt_color(&input_cpu, ColorConversion::RgbToGray).unwrap();
    let cpu_slice = res_cpu.storage.as_slice().unwrap();
    
    // GPU
    let res_gpu_encoded = gpu.cvt_color(&input_gpu, ColorConversion::RgbToGray).unwrap();
    let res_gpu = res_gpu_encoded.to_cpu_ctx(gpu).unwrap();
    let gpu_slice = res_gpu.storage.as_slice().unwrap();
    
    for i in 0..cpu_slice.len() {
        // Use tolerance of 1 due to potential rounding differences
        if (cpu_slice[i] as i32 - gpu_slice[i] as i32).abs() > 1 {
            panic!("Color Cvt Parity failure on {}: at index {}, CPU={}, GPU={}", gpu_name, i, cpu_slice[i], gpu_slice[i]);
        }
    }
    println!("  ✓ Color Cvt parity passed for {}", gpu_name);
}

fn test_pc_transform_parity(cpu: &CpuBackend, gpu: &GpuContext, gpu_name: &str) {
    let num_points = 100;
    let shape = TensorShape::new(1, num_points, 4); // (1, N, 4) for vec4 alignment
    let mut data = vec![0.0f32; shape.len()];
    for i in 0..num_points {
        data[i * 4] = i as f32;
        data[i * 4 + 1] = (i * 2) as f32;
        data[i * 4 + 2] = (i * 3) as f32;
        data[i * 4 + 3] = 1.0;
    }

    let input_cpu = Tensor::from_vec(data, shape);
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Upload failed");

    let transform = [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // Execute on CPU
    let res_cpu = cpu.pointcloud_transform(&input_cpu, &transform).unwrap();
    let cpu_slice = res_cpu.storage.as_slice().unwrap();

    // Execute on GPU
    let res_gpu_encoded = gpu.pointcloud_transform(&input_gpu, &transform).unwrap();
    let res_gpu = res_gpu_encoded.to_cpu_ctx(gpu).unwrap();
    let gpu_slice = res_gpu.storage.as_slice().unwrap();

    for i in 0..cpu_slice.len() {
        if (cpu_slice[i] - gpu_slice[i]).abs() > 1e-5 {
            panic!("PC Transform Parity failure on {}: at index {}, CPU={}, GPU={}", gpu_name, i, cpu_slice[i], gpu_slice[i]);
        }
    }
    
    println!("  ✓ PC Transform parity passed for {}", gpu_name);
}

fn test_threshold_parity(cpu: &CpuBackend, gpu: &GpuContext, gpu_name: &str) {
    let shape = TensorShape::new(1, 128, 128);
    let mut data = vec![0u8; shape.len()];
    for i in 0..data.len() {
        data[i] = (i % 256) as u8;
    }
    
    let input_cpu = Tensor::from_vec(data.clone(), shape);
    let input_gpu = input_cpu.to_gpu_ctx(gpu).expect("Failed to upload to GPU");
    
    let thresh = 128u8;
    let max_val = 255u8;
    
    // Execute on CPU
    let res_cpu = cpu.threshold(&input_cpu, thresh, max_val, ThresholdType::Binary).unwrap();
    
    // Execute on GPU
    let res_gpu_encoded = gpu.threshold(&input_gpu, thresh, max_val, ThresholdType::Binary).unwrap();
    let res_gpu = res_gpu_encoded.to_cpu_ctx(gpu).unwrap();
    
    // Compare
    let cpu_slice = res_cpu.storage.as_slice().unwrap();
    let gpu_slice = res_gpu.storage.as_slice().unwrap();
    
    for i in 0..cpu_slice.len() {
        if cpu_slice[i] != gpu_slice[i] {
            panic!("Parity failure on {}: at index {}, CPU={}, GPU={}", gpu_name, i, cpu_slice[i], gpu_slice[i]);
        }
    }
    println!("  ✓ Threshold parity passed for {}", gpu_name);
}
