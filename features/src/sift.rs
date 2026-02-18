use cv_core::{KeyPoints, Tensor, storage::Storage};
use cv_hal::compute::ComputeDevice;
use cv_hal::tensor_ext::TensorToCpu;

pub struct Sift {
    pub n_octaves: usize,
    pub n_layers: usize,
    pub sigma: f32,
    pub contrast_threshold: f32,
    pub edge_threshold: f32,
}

impl Default for Sift {
    fn default() -> Self {
        Self {
            n_octaves: 4,
            n_layers: 3,
            sigma: 1.6,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
        }
    }
}

impl Sift {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build SIFT scale-space using accelerated primitives
    pub fn build_scale_space<S: Storage<u8> + 'static>(
        &self,
        ctx: &ComputeDevice,
        image: &Tensor<u8, S>,
    ) -> Vec<Vec<Tensor<u8, S>>> {
        let mut pyramid = Vec::with_capacity(self.n_octaves);
        let mut current_base = image.clone();

        for octave in 0..self.n_octaves {
            let mut layers = Vec::with_capacity(self.n_layers + 3);
            
            let mut sig = self.sigma;
            let base_layer = ctx.gaussian_blur(&current_base, sig, 7).unwrap();
            layers.push(base_layer);

            let k = 2.0f32.powf(1.0 / self.n_layers as f32);
            for _ in 1..(self.n_layers + 3) {
                let prev = layers.last().unwrap();
                let sig_prev = sig;
                sig *= k;
                let sig_total = (sig * sig - sig_prev * sig_prev).sqrt();
                
                let blurred = ctx.gaussian_blur(prev, sig_total, 7).unwrap();
                layers.push(blurred);
            }
            pyramid.push(layers);

            if octave < self.n_octaves - 1 {
                let to_sample = &pyramid[octave][self.n_layers];
                let (h, w) = to_sample.shape.hw();
                current_base = ctx.resize(to_sample, (w / 2, h / 2)).unwrap();
            }
        }
        pyramid
    }

    /// Compute Difference of Gaussians (DoG)
    pub fn compute_dog<S: Storage<u8> + 'static>(
        &self,
        ctx: &ComputeDevice,
        gaussian_pyramid: &[Vec<Tensor<u8, S>>],
    ) -> Vec<Vec<Tensor<f32, cv_core::storage::CpuStorage<f32>>>> {
        let mut dog_pyramid = Vec::with_capacity(gaussian_pyramid.len());

        for octave_layers in gaussian_pyramid {
            let mut dog_layers = Vec::with_capacity(octave_layers.len() - 1);
            for i in 0..(octave_layers.len() - 1) {
                let a_f32 = convert_to_f32_cpu(ctx, &octave_layers[i + 1]);
                let b_f32 = convert_to_f32_cpu(ctx, &octave_layers[i]);
                
                // CPU subtraction
                let diff = ctx.subtract(&a_f32, &b_f32).unwrap();
                dog_layers.push(diff);
            }
            dog_pyramid.push(dog_layers);
        }
        dog_pyramid
    }
}

// Helper to convert tensor to f32 on CPU
fn convert_to_f32_cpu<S: Storage<u8> + 'static>(
    ctx: &ComputeDevice,
    input: &Tensor<u8, S>,
) -> Tensor<f32, cv_core::storage::CpuStorage<f32>> {
    use std::any::TypeId;
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;

    let cpu_u8 = if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
        let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
        let input_gpu = unsafe { &*input_ptr };
        let gpu_ctx = match ctx {
            ComputeDevice::Gpu(g) => g,
            _ => panic!("Logic error: GpuStorage with CpuBackend"),
        };
        input_gpu.to_cpu_ctx(gpu_ctx).unwrap()
    } else {
        let input_ptr = input as *const Tensor<u8, S> as *const Tensor<u8, CpuStorage<u8>>;
        let input_cpu = unsafe { &*input_ptr };
        input_cpu.clone()
    };

    let slice_u8 = cpu_u8.storage.as_slice().unwrap();
    let data_f32: Vec<f32> = slice_u8.iter().map(|&v| v as f32).collect();
    
    Tensor::from_vec(data_f32, input.shape)
}

pub fn sift_detect_ctx<S: Storage<u8> + 'static>(
    ctx: &ComputeDevice,
    image: &Tensor<u8, S>,
    params: &Sift,
) -> KeyPoints {
    let _pyramid = params.build_scale_space(ctx, image);
    KeyPoints { keypoints: Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_hal::cpu::CpuBackend;
    use cv_core::{TensorShape, storage::CpuStorage};

    #[test]
    fn test_sift_scale_space() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        
        let shape = TensorShape::new(1, 128, 128);
        let data = vec![128u8; shape.len()];
        let tensor: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(data, shape);
        
        let sift = Sift::new();
        let pyramid = sift.build_scale_space(&device, &tensor);
        
        assert_eq!(pyramid.len(), sift.n_octaves);
        for octave in 0..sift.n_octaves {
            assert_eq!(pyramid[octave].len(), sift.n_layers + 3);
        }
        println!("SIFT Gaussian pyramid built successfully");
    }

    #[test]
    fn test_sift_dog() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        
        let shape = TensorShape::new(1, 128, 128);
        let data = vec![128u8; shape.len()];
        let tensor: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(data, shape);
        
        let sift = Sift::new();
        let gaussian_pyramid = sift.build_scale_space(&device, &tensor);
        let dog_pyramid = sift.compute_dog(&device, &gaussian_pyramid);
        
        assert_eq!(dog_pyramid.len(), sift.n_octaves);
        for octave in 0..sift.n_octaves {
            assert_eq!(dog_pyramid[octave].len(), sift.n_layers + 2);
        }
        println!("SIFT DoG pyramid built successfully");
    }
}
