//! Mixture of Gaussians (MOG2) Background Subtraction
//!
//! Robust background/foreground segmentation algorithm that models each pixel
//! as a mixture of multiple Gaussian distributions.

use cv_core::{storage::CpuStorage, storage::Storage, CpuTensor, Tensor, TensorShape};
use cv_hal::compute::ComputeDevice;
use cv_hal::context::{ComputeContext, Mog2Params};
use cv_hal::tensor_ext::{TensorCast, TensorToCpu, TensorToGpu};

pub struct Mog2 {
    history: usize,
    var_threshold: f32,
    _detect_shadows: bool,
    n_mixtures: usize,
    background_ratio: f32,
    var_init: f32,
    var_min: f32,
    var_max: f32,

    // Model state: [H * W * N_MIXTURES * 3]
    model: Option<Box<dyn std::any::Any>>, // Stores Tensor<f32, S>
    width: usize,
    height: usize,
}

impl Mog2 {
    pub fn new(history: usize, var_threshold: f32, detect_shadows: bool) -> Self {
        Self {
            history,
            var_threshold,
            _detect_shadows: detect_shadows,
            n_mixtures: 5,
            background_ratio: 0.9,
            var_init: 15.0,
            var_min: 4.0,
            var_max: 5.0 * 15.0,
            model: None,
            width: 0,
            height: 0,
        }
    }

    pub fn apply_ctx<S: Storage<u8> + 'static>(
        &mut self,
        frame: &Tensor<u8, S>,
        learning_rate: f32,
        ctx: &ComputeDevice,
    ) -> CpuTensor<u8>
    where
        Tensor<u8, S>: TensorToCpu<u8> + TensorToGpu<u8>,
    {
        let (h, w) = frame.shape.hw();
        let width = w;
        let height = h;

        let alpha = if learning_rate < 0.0 {
            1.0 / self.history as f32
        } else {
            learning_rate
        };
        let params = Mog2Params {
            width: width as u32,
            height: height as u32,
            n_mixtures: self.n_mixtures as u32,
            alpha,
            var_threshold: self.var_threshold,
            background_ratio: self.background_ratio,
            var_init: self.var_init / 255.0,
            var_min: self.var_min / 255.0,
            var_max: self.var_max / 255.0,
            _padding: [0; 3],
        };

        match ctx {
            ComputeDevice::Gpu(gpu) => {
                use cv_hal::storage::GpuStorage;
                // For GPU, we need TensorCast to do GPU-side casting
                let frame_gpu = frame.to_gpu_ctx(gpu).expect("Upload to GPU failed");
                let frame_f32 =
                    <Tensor<u8, GpuStorage<u8>> as TensorCast>::to_f32_ctx(&frame_gpu, gpu)
                        .expect("GPU cast failed");

                if self.model.is_none() || self.width != width || self.height != height {
                    self.width = width;
                    self.height = height;
                    let mut data = vec![0.0f32; width * height * self.n_mixtures * 3];
                    let frame_cpu = frame.to_cpu().expect("Download to CPU failed");
                    let frame_raw = frame_cpu.as_slice().expect("Failed to get frame slice");
                    for i in 0..(width * height) {
                        let base = i * self.n_mixtures * 3;
                        data[base + 0] = 1.0;
                        data[base + 1] = frame_raw[i] as f32 / 255.0;
                        data[base + 2] = self.var_init / 255.0;
                    }
                    let model_gpu = CpuTensor::from_vec(
                        data,
                        TensorShape::new(1, 1, width * height * self.n_mixtures * 3),
                    )
                    .expect("Failed to create model tensor")
                    .to_gpu_ctx(gpu)
                    .expect("Upload model to GPU failed");
                    self.model = Some(Box::new(model_gpu));
                }

                let model_gpu = self
                    .model
                    .as_mut()
                    .expect("Model missing")
                    .downcast_mut::<Tensor<f32, GpuStorage<f32>>>()
                    .expect("Downcast failed");
                let mut mask_gpu = Tensor::<u32, GpuStorage<u32>>::from_vec(
                    vec![0u32; width * height],
                    TensorShape::new(1, height, width),
                )
                .expect("Failed to create mask tensor")
                .to_gpu_ctx(gpu)
                .expect("Upload mask to GPU failed");

                gpu.mog2_update(&frame_f32, model_gpu, &mut mask_gpu, &params)
                    .expect("MOG2 GPU update failed");

                let mask_cpu = mask_gpu
                    .to_cpu_ctx(gpu)
                    .expect("Download mask to CPU failed");
                let u8_data: Vec<u8> = mask_cpu
                    .as_slice()
                    .expect("Failed to get mask slice")
                    .iter()
                    .map(|&v| v as u8)
                    .collect();
                CpuTensor::from_vec(u8_data, frame.shape)
                    .expect("Failed to create final result tensor")
            }
            ComputeDevice::Cpu(cpu) => {
                let frame_cpu = frame.to_cpu().expect("Download frame to CPU failed");
                let frame_f32_vec: Vec<f32> = frame_cpu
                    .as_slice()
                    .expect("Failed to get frame slice")
                    .iter()
                    .map(|&v| v as f32 / 255.0)
                    .collect();
                let frame_tensor = CpuTensor::from_vec(frame_f32_vec, frame.shape)
                    .expect("Failed to create frame tensor");

                if self.model.is_none() || self.width != width || self.height != height {
                    self.width = width;
                    self.height = height;
                    let mut data = vec![0.0f32; width * height * self.n_mixtures * 3];
                    let frame_raw = frame_cpu.as_slice().expect("Failed to get frame slice");
                    for i in 0..(width * height) {
                        let base = i * self.n_mixtures * 3;
                        data[base + 0] = 1.0;
                        data[base + 1] = frame_raw[i] as f32 / 255.0;
                        data[base + 2] = self.var_init / 255.0;
                    }
                    self.model = Some(Box::new(
                        CpuTensor::from_vec(
                            data,
                            TensorShape::new(1, 1, width * height * self.n_mixtures * 3),
                        )
                        .expect("Failed to create model tensor"),
                    ));
                }

                let model_cpu = self
                    .model
                    .as_mut()
                    .expect("Model missing")
                    .downcast_mut::<CpuTensor<f32>>()
                    .expect("Downcast failed");
                let mut mask_cpu = Tensor::<u32, CpuStorage<u32>>::from_vec(
                    vec![0u32; width * height],
                    frame.shape,
                )
                .expect("Failed to create mask tensor");

                cpu.mog2_update(&frame_tensor, model_cpu, &mut mask_cpu, &params)
                    .expect("MOG2 CPU update failed");

                let u8_data: Vec<u8> = mask_cpu
                    .as_slice()
                    .expect("Failed to get mask slice")
                    .iter()
                    .map(|&v| v as u8)
                    .collect();
                CpuTensor::from_vec(u8_data, frame.shape)
                    .expect("Failed to create final result tensor")
            }
            ComputeDevice::Mlx(_) => {
                eprintln!("Warning: MOG2 not implemented for MLX backend, returning empty mask");
                CpuTensor::from_vec(vec![0u8; width * height], frame.shape)
                    .expect("Failed to create empty result tensor")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_hal::cpu::CpuBackend;

    #[test]
    fn test_mog2_basic() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let mut mog2 = Mog2::new(100, 16.0, false);

        let width = 64usize;
        let height = 64usize;
        let bg_data = vec![0u8; width * height];
        let bg_frame: CpuTensor<u8> =
            CpuTensor::from_vec(bg_data, TensorShape::new(1, height, width)).unwrap();

        for _ in 0..10 {
            mog2.apply_ctx(&bg_frame, -1.0, &device);
        }

        let mut fg_data = vec![0u8; width * height];
        for y in 20..40 {
            for x in 20..40 {
                fg_data[y * width + x] = 255;
            }
        }
        let fg_frame: CpuTensor<u8> =
            CpuTensor::from_vec(fg_data, TensorShape::new(1, height, width)).unwrap();

        let mask = mog2.apply_ctx(&fg_frame, -1.0, &device);

        let mut fg_count = 0;
        let mask_slice = mask.as_slice().unwrap();
        for y in 20..40 {
            for x in 20..40 {
                if mask_slice[y * width + x] == 255 {
                    fg_count += 1;
                }
            }
        }

        println!("Detected {} foreground pixels in 20x20 square", fg_count);
        assert!(fg_count > 300);
    }

    #[test]
    fn test_mog2_uniform_background() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let mut mog2 = Mog2::new(100, 16.0, false);

        let width = 32usize;
        let height = 32usize;
        let uniform_data = vec![128u8; width * height];
        let frame: CpuTensor<u8> =
            CpuTensor::from_vec(uniform_data, TensorShape::new(1, height, width)).unwrap();

        // Apply to the same frame multiple times
        for _ in 0..20 {
            let _ = mog2.apply_ctx(&frame, -1.0, &device);
        }

        // Uniform background should be learned
        let mask = mog2.apply_ctx(&frame, -1.0, &device);
        let mask_slice = mask.as_slice().unwrap();

        // Count foreground pixels (should be minimal for uniform background)
        let fg_count = mask_slice.iter().filter(|&&v| v > 127).count();
        assert!(fg_count < 100); // Most pixels should be background
    }

    #[test]
    fn test_mog2_initialization() {
        let mog2 = Mog2::new(50, 10.0, false);
        assert_eq!(mog2.history, 50);
        assert!((mog2.var_threshold - 10.0).abs() < 0.01);
        assert!(!mog2._detect_shadows);
        assert_eq!(mog2.n_mixtures, 5);
    }

    #[test]
    fn test_mog2_custom_parameters() {
        let mog2 = Mog2::new(200, 25.0, true);
        assert_eq!(mog2.history, 200);
        assert!((mog2.var_threshold - 25.0).abs() < 0.01);
        assert!(mog2._detect_shadows);
    }

    #[test]
    fn test_mog2_varying_learning_rates() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);

        let width = 32usize;
        let height = 32usize;
        let bg_data = vec![50u8; width * height];
        let bg_frame: CpuTensor<u8> =
            CpuTensor::from_vec(bg_data.clone(), TensorShape::new(1, height, width)).unwrap();

        // Test with automatic learning rate
        let mut mog2_auto = Mog2::new(100, 16.0, false);
        for _ in 0..5 {
            mog2_auto.apply_ctx(&bg_frame, -1.0, &device);
        }

        // Test with explicit learning rate
        let mut mog2_explicit = Mog2::new(100, 16.0, false);
        for _ in 0..5 {
            mog2_explicit.apply_ctx(&bg_frame, 0.01, &device);
        }

        // Both should produce valid masks
        let mask_auto = mog2_auto.apply_ctx(&bg_frame, -1.0, &device);
        let mask_explicit = mog2_explicit.apply_ctx(&bg_frame, 0.01, &device);

        assert_eq!(mask_auto.shape, bg_frame.shape);
        assert_eq!(mask_explicit.shape, bg_frame.shape);
    }

    #[test]
    fn test_mog2_dimension_preservation() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let mut mog2 = Mog2::new(100, 16.0, false);

        let width = 48usize;
        let height = 48usize;
        let frame_data = vec![100u8; width * height];
        let frame: CpuTensor<u8> =
            CpuTensor::from_vec(frame_data, TensorShape::new(1, height, width)).unwrap();

        let mask = mog2.apply_ctx(&frame, -1.0, &device);

        // Output mask should have same dimensions as input
        assert_eq!(mask.shape.height, height);
        assert_eq!(mask.shape.width, width);
    }

    #[test]
    fn test_mog2_sequence_processing() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let mut mog2 = Mog2::new(100, 16.0, false);

        let width = 32usize;
        let height = 32usize;

        // Simulate a sequence with gradually changing frames
        for i in 0..20 {
            let mut frame_data = vec![100u8; width * height];

            // Gradually introduce more white (255) pixels
            for y in 10..22 {
                for x in 10..22 {
                    if i < 10 {
                        frame_data[y * width + x] = 100; // Background
                    } else {
                        frame_data[y * width + x] = 255; // Foreground
                    }
                }
            }

            let frame: CpuTensor<u8> =
                CpuTensor::from_vec(frame_data, TensorShape::new(1, height, width)).unwrap();
            let _ = mog2.apply_ctx(&frame, -1.0, &device);
        }

        // Should complete without crashing and produce valid output
        let final_data = vec![100u8; width * height];
        let final_frame: CpuTensor<u8> =
            CpuTensor::from_vec(final_data, TensorShape::new(1, height, width)).unwrap();
        let mask = mog2.apply_ctx(&final_frame, -1.0, &device);

        // Verify output is valid
        assert_eq!(mask.shape.height, height);
        assert_eq!(mask.shape.width, width);
        assert_eq!(mask.as_slice().unwrap().len(), width * height);
    }

    #[test]
    fn test_mog2_small_frame() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let mut mog2 = Mog2::new(100, 16.0, false);

        let width = 8usize;
        let height = 8usize;
        let frame_data = vec![128u8; width * height];
        let frame: CpuTensor<u8> =
            CpuTensor::from_vec(frame_data, TensorShape::new(1, height, width)).unwrap();

        // Should handle small frames gracefully
        let mask = mog2.apply_ctx(&frame, -1.0, &device);
        assert_eq!(mask.shape.height, height);
        assert_eq!(mask.shape.width, width);
    }

    #[test]
    fn test_mog2_large_frame() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let mut mog2 = Mog2::new(100, 16.0, false);

        let width = 256usize;
        let height = 256usize;
        let frame_data = vec![128u8; width * height];
        let frame: CpuTensor<u8> =
            CpuTensor::from_vec(frame_data, TensorShape::new(1, height, width)).unwrap();

        // Should handle large frames gracefully
        let mask = mog2.apply_ctx(&frame, -1.0, &device);
        assert_eq!(mask.shape.height, height);
        assert_eq!(mask.shape.width, width);
    }

    #[test]
    fn test_mog2_frame_dimension_change() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let mut mog2 = Mog2::new(100, 16.0, false);

        let width1 = 32usize;
        let height1 = 32usize;
        let frame1: CpuTensor<u8> =
            CpuTensor::from_vec(vec![128u8; width1 * height1], TensorShape::new(1, height1, width1)).unwrap();

        let _ = mog2.apply_ctx(&frame1, -1.0, &device);

        // Process frame with different dimensions
        let width2 = 48usize;
        let height2 = 48usize;
        let frame2: CpuTensor<u8> =
            CpuTensor::from_vec(vec![128u8; width2 * height2], TensorShape::new(1, height2, width2)).unwrap();

        let mask = mog2.apply_ctx(&frame2, -1.0, &device);
        assert_eq!(mask.shape.height, height2);
        assert_eq!(mask.shape.width, width2);
    }
}
