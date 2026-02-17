use cv_core::{Tensor, TensorShape, storage::Storage};
use cv_hal::gpu::GpuContext;
use cv_hal::cpu::CpuBackend;
use cv_hal::context::{ComputeContext, ThresholdType};
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
    }
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
