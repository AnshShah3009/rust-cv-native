use cv_core::{Tensor, TensorShape, storage::Storage};
use cv_hal::tensor_ext::{TensorToGpu, TensorToCpu};
use cv_hal::gpu::GpuContext;
use cv_imgproc::convolve::{gaussian_blur, BorderMode};
use image::{GrayImage, Luma};

#[test]
fn test_tensor_transfer_roundtrip() {
    if GpuContext::global().is_none() {
        println!("Skipping test: GPU not available");
        return;
    }

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let shape = TensorShape::new(1, 2, 2);
    let tensor_cpu = Tensor::from_vec(data.clone(), shape);

    let tensor_gpu = tensor_cpu.to_gpu().expect("Failed to upload to GPU");
    let tensor_back = tensor_gpu.to_cpu().expect("Failed to download from GPU");

    assert_eq!(tensor_back.storage.as_slice().unwrap(), &data[..]);
}

#[test]
fn test_gaussian_blur_acceleration_path() {
    // This test ensures the dispatch logic works without crashing.
    // Even if GPU is not available, it should fallback to CPU.
    let mut img = GrayImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            img.put_pixel(x, y, Luma([if (x+y) % 2 == 0 { 255 } else { 0 }]));
        }
    }

    let blurred = gaussian_blur(&img, 1.0);
    assert_eq!(blurred.width(), 64);
    assert_eq!(blurred.height(), 64);
}
