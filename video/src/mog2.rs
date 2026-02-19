//! Mixture of Gaussians (MOG2) Background Subtraction
//!
//! Robust background/foreground segmentation algorithm that models each pixel
//! as a mixture of multiple Gaussian distributions.

use cv_core::{Tensor, TensorShape, storage::Storage, CpuTensor, storage::CpuStorage};
use cv_hal::compute::ComputeDevice;
use cv_hal::context::{Mog2Params, ComputeContext};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu, TensorCast};

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

    pub fn apply_ctx<S: Storage<u8> + 'static>(&mut self, frame: &Tensor<u8, S>, learning_rate: f32, ctx: &ComputeDevice) -> CpuTensor<u8> 
    where Tensor<u8, S>: TensorToCpu<u8> + TensorToGpu<u8>
    {
        let (h, w) = frame.shape.hw();
        let width = w;
        let height = h;
        
        let alpha = if learning_rate < 0.0 { 1.0 / self.history as f32 } else { learning_rate };
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
                let frame_gpu = frame.to_gpu_ctx(gpu).unwrap();
                let frame_f32 = <Tensor<u8, GpuStorage<u8>> as TensorCast>::to_f32_ctx(&frame_gpu, gpu).unwrap();
                
                if self.model.is_none() || self.width != width || self.height != height {
                    self.width = width;
                    self.height = height;
                    let mut data = vec![0.0f32; width * height * self.n_mixtures * 3];
                    let frame_cpu = frame.to_cpu().unwrap();
                    let frame_raw = frame_cpu.as_slice();
                    for i in 0..(width * height) {
                        let base = i * self.n_mixtures * 3;
                        data[base + 0] = 1.0;
                        data[base + 1] = frame_raw[i] as f32 / 255.0;
                        data[base + 2] = self.var_init / 255.0;
                    }
                    let model_gpu = CpuTensor::from_vec(data, TensorShape::new(1, 1, width * height * self.n_mixtures * 3)).to_gpu_ctx(gpu).unwrap();
                    self.model = Some(Box::new(model_gpu));
                }

                let model_gpu = self.model.as_mut().unwrap().downcast_mut::<Tensor<f32, GpuStorage<f32>>>().unwrap();
                let mut mask_gpu = Tensor::<u32, GpuStorage<u32>>::from_vec(vec![0u32; width * height], TensorShape::new(1, height, width)).to_gpu_ctx(gpu).unwrap();
                
                gpu.mog2_update(&frame_f32, model_gpu, &mut mask_gpu, &params).unwrap();
                
                let mask_cpu = mask_gpu.to_cpu_ctx(gpu).unwrap();
                let u8_data: Vec<u8> = mask_cpu.as_slice().iter().map(|&v| v as u8).collect();
                CpuTensor::from_vec(u8_data, frame.shape)
            },
            ComputeDevice::Cpu(cpu) => {
                let frame_cpu = frame.to_cpu().unwrap();
                let frame_f32_vec: Vec<f32> = frame_cpu.as_slice().iter().map(|&v| v as f32 / 255.0).collect();
                let frame_tensor = CpuTensor::from_vec(frame_f32_vec, frame.shape);

                if self.model.is_none() || self.width != width || self.height != height {
                    self.width = width;
                    self.height = height;
                    let mut data = vec![0.0f32; width * height * self.n_mixtures * 3];
                    let frame_raw = frame_cpu.as_slice();
                    for i in 0..(width * height) {
                        let base = i * self.n_mixtures * 3;
                        data[base + 0] = 1.0;
                        data[base + 1] = frame_raw[i] as f32 / 255.0;
                        data[base + 2] = self.var_init / 255.0;
                    }
                    self.model = Some(Box::new(CpuTensor::from_vec(data, TensorShape::new(1, 1, width * height * self.n_mixtures * 3))));
                }

                let model_cpu = self.model.as_mut().unwrap().downcast_mut::<CpuTensor<f32>>().unwrap();
                let mut mask_cpu = Tensor::<u32, CpuStorage<u32>>::from_vec(vec![0u32; width * height], frame.shape);
                
                cpu.mog2_update(&frame_tensor, model_cpu, &mut mask_cpu, &params).unwrap();
                
                let u8_data: Vec<u8> = mask_cpu.as_slice().iter().map(|&v| v as u8).collect();
                CpuTensor::from_vec(u8_data, frame.shape)
            },
            ComputeDevice::Mlx(_) => todo!("MOG2 not implemented for MLX"),
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
        let bg_frame: CpuTensor<u8> = CpuTensor::from_vec(bg_data, TensorShape::new(1, height, width));
        
        for _ in 0..10 {
            mog2.apply_ctx(&bg_frame, -1.0, &device);
        }
        
        let mut fg_data = vec![0u8; width * height];
        for y in 20..40 {
            for x in 20..40 {
                fg_data[y * width + x] = 255;
            }
        }
        let fg_frame: CpuTensor<u8> = CpuTensor::from_vec(fg_data, TensorShape::new(1, height, width));
        
        let mask = mog2.apply_ctx(&fg_frame, -1.0, &device);
        
        let mut fg_count = 0;
        let mask_slice = mask.as_slice();
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
}
