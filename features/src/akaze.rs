//! AKAZE (Accelerated-KAZE) feature detector and descriptor
//!
//! AKAZE is a fast multi-scale feature detector and descriptor that uses
//! non-linear diffusion scale-space.

use cv_core::{KeyPoint, KeyPoints, Tensor, storage::Storage, CpuTensor};
use cv_hal::compute::ComputeDevice;
use cv_hal::context::ComputeContext;
use cv_hal::tensor_ext::{TensorToGpu, TensorToCpu, TensorCast};
use crate::descriptor::{Descriptor, Descriptors};
use rayon::prelude::*;

/// AKAZE parameters
#[derive(Debug, Clone)]
pub struct AkazeParams {
    pub n_octaves: usize,
    pub n_sublevels: usize,
    pub threshold: f32,
    pub diffusivity: DiffusivityType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusivityType {
    PeronaMalik1,
    PeronaMalik2,
    Weickert,
    Charbonnier,
}

impl Default for AkazeParams {
    fn default() -> Self {
        Self {
            n_octaves: 4,
            n_sublevels: 4,
            threshold: 0.001,
            diffusivity: DiffusivityType::PeronaMalik2,
        }
    }
}

pub struct Akaze {
    params: AkazeParams,
}

impl Akaze {
    pub fn new(params: AkazeParams) -> Self {
        Self { params }
    }

    pub fn detect_ctx<S: Storage<u8> + 'static>(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> KeyPoints 
    where Tensor<u8, S>: TensorToGpu<u8> + TensorToCpu<u8>
    {
        let evolution = self.create_scale_space(ctx, image);
        let keypoints = self.find_extrema(&evolution);
        KeyPoints { keypoints }
    }

    pub fn detect_and_compute_ctx<S: Storage<u8> + 'static>(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> (KeyPoints, Descriptors) 
    where Tensor<u8, S>: TensorToGpu<u8> + TensorToCpu<u8>
    {
        let evolution = self.create_scale_space(ctx, image);
        let keypoints = self.find_extrema(&evolution);
        let descriptors = self.compute_descriptors(&evolution, &keypoints);
        (KeyPoints { keypoints }, descriptors)
    }

    fn create_scale_space<S: Storage<u8> + 'static>(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> Vec<EvolutionLevel> 
    where Tensor<u8, S>: TensorToGpu<u8> + TensorToCpu<u8>
    {
        let mut evolution = Vec::new();
        let mut t = 0.0f32;

        match ctx {
            ComputeDevice::Gpu(gpu) => {
                let current_u8 = ctx.gaussian_blur(image, 1.0, 5).unwrap();
                let gpu_u8 = <Tensor<u8, S> as TensorToGpu<u8>>::to_gpu_ctx(&current_u8, gpu).unwrap();
                let mut current_f32 = <Tensor<u8, _> as TensorCast>::to_f32_ctx(&gpu_u8, gpu).unwrap();
                let k = ctx.akaze_contrast_k(&current_f32).unwrap_or(0.03);

                for o in 0..self.params.n_octaves {
                    for s in 0..self.params.n_sublevels {
                        let sigma = (2.0f32).powf((o as f32) + (s as f32) / (self.params.n_sublevels as f32));
                        let esigma = sigma * sigma / 2.0;
                        let dt = esigma - t;
                        
                        if dt > 0.0 {
                            let n_steps = self.get_fed_steps(dt);
                            let step_tau = dt / (n_steps as f32);
                            for _ in 0..n_steps {
                                current_f32 = gpu.akaze_diffusion(&current_f32, k, step_tau).unwrap();
                            }
                            t = esigma;
                        }
                        
                        let (lx, ly, ldet) = gpu.akaze_derivatives(&current_f32).unwrap();
                        
                        evolution.push(EvolutionLevel {
                            image: current_f32.to_cpu_ctx(gpu).unwrap(),
                            lx: lx.to_cpu_ctx(gpu).unwrap(),
                            ly: ly.to_cpu_ctx(gpu).unwrap(),
                            ldet: ldet.to_cpu_ctx(gpu).unwrap(),
                            sigma,
                            octave: o,
                        });
                    }
                }
            },
            ComputeDevice::Cpu(cpu) => {
                let cpu_u8 = image.to_cpu().unwrap();
                let mut current_u8 = cpu.gaussian_blur(&cpu_u8, 1.0, 5).unwrap();
                let data: Vec<f32> = current_u8.as_slice().iter().map(|&v| v as f32 / 255.0).collect();
                let mut current_f32 = CpuTensor::from_vec(data, current_u8.shape);
                
                let k = cpu.akaze_contrast_k(&current_f32).unwrap_or(0.03);

                for o in 0..self.params.n_octaves {
                    for s in 0..self.params.n_sublevels {
                        let sigma = (2.0f32).powf((o as f32) + (s as f32) / (self.params.n_sublevels as f32));
                        let esigma = sigma * sigma / 2.0;
                        let dt = esigma - t;
                        
                        if dt > 0.0 {
                            let n_steps = self.get_fed_steps(dt);
                            let step_tau = dt / (n_steps as f32);
                            for _ in 0..n_steps {
                                current_f32 = cpu.akaze_diffusion(&current_f32, k, step_tau).unwrap();
                            }
                            t = esigma;
                        }
                        
                        let (lx, ly, ldet) = cpu.akaze_derivatives(&current_f32).unwrap();
                        
                        evolution.push(EvolutionLevel {
                            image: current_f32.clone(),
                            lx,
                            ly,
                            ldet,
                            sigma,
                            octave: o,
                        });
                    }
                }
            }
        }
        
        evolution
    }

    fn get_fed_steps(&self, dt: f32) -> usize {
        let tau_max = 0.25;
        ((dt / tau_max).sqrt().ceil() as usize).max(1)
    }

    fn find_extrema(&self, evolution: &[EvolutionLevel]) -> Vec<KeyPoint> {
        let mut keypoints = Vec::new();
        let threshold = self.params.threshold;

        for curr in evolution {
            let (h, w) = curr.ldet.shape.hw();
            let det_slice = curr.ldet.as_slice();

            let mut level_kps: Vec<KeyPoint> = (1..h-1).into_par_iter().flat_map(|y| {
                let mut row_kps = Vec::new();
                for x in 1..w-1 {
                    let val = det_slice[y * w + x];
                    if val > threshold {
                        let mut is_max = true;
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if dx == 0 && dy == 0 { continue; }
                                if det_slice[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] >= val {
                                    is_max = false;
                                    break;
                                }
                            }
                            if !is_max { break; }
                        }

                        if is_max {
                            row_kps.push(KeyPoint::new(x as f64, y as f64)
                                .with_size(curr.sigma as f64 * 2.0)
                                .with_response(val as f64)
                                .with_octave(curr.octave as i32));
                        }
                    }
                }
                row_kps
            }).collect();

            keypoints.append(&mut level_kps);
        }

        keypoints
    }

    fn compute_descriptors(&self, evolution: &[EvolutionLevel], keypoints: &[KeyPoint]) -> Descriptors {
        let mut descriptors = Vec::with_capacity(keypoints.len());

        for kp in keypoints {
            let level_idx = kp.octave as usize * self.params.n_sublevels + (kp.size.log2() as usize % self.params.n_sublevels);
            let level = &evolution[level_idx.min(evolution.len() - 1)];
            
            if let Some(desc) = self.compute_msurf_descriptor(level, kp) {
                descriptors.push(desc);
            }
        }

        Descriptors { descriptors }
    }

    fn compute_msurf_descriptor(&self, level: &EvolutionLevel, kp: &KeyPoint) -> Option<Descriptor> {
        let (h, w) = level.image.shape.hw();
        let lx_slice = level.lx.as_slice();
        let ly_slice = level.ly.as_slice();

        let mut desc = vec![0u8; 64];
        let mut float_desc = [0.0f32; 64];
        
        let x_center = kp.x as f32;
        let y_center = kp.y as f32;
        let s = kp.size as f32;

        for i in 0..4 {
            for j in 0..4 {
                let mut sum_dx = 0.0;
                let mut sum_dy = 0.0;
                let mut sum_abs_dx = 0.0;
                let mut sum_abs_dy = 0.0;

                for x_sample in 0..5 {
                    for y_sample in 0..5 {
                        let rx = x_center + (i as f32 - 2.0 + x_sample as f32 * 0.2) * s;
                        let ry = y_center + (j as f32 - 2.0 + y_sample as f32 * 0.2) * s;

                        if rx >= 0.0 && rx < (w - 1) as f32 && ry >= 0.0 && ry < (h - 1) as f32 {
                            let idx = ry as usize * w + rx as usize;
                            let dx = lx_slice[idx];
                            let dy = ly_slice[idx];
                            sum_dx += dx;
                            sum_dy += dy;
                            sum_abs_dx += dx.abs();
                            sum_abs_dy += dy.abs();
                        }
                    }
                }

                let base = (i * 4 + j) * 4;
                float_desc[base] = sum_dx;
                float_desc[base + 1] = sum_dy;
                float_desc[base + 2] = sum_abs_dx;
                float_desc[base + 3] = sum_abs_dy;
            }
        }

        let mut norm_sq = 0.0;
        for &v in &float_desc { norm_sq += v * v; }
        let norm = norm_sq.sqrt() + 1e-7;
        for i in 0..64 {
            desc[i] = ((float_desc[i] / norm) * 128.0 + 128.0).clamp(0.0, 255.0) as u8;
        }

        Some(Descriptor::new(desc, kp.clone()))
    }
}

struct EvolutionLevel {
    pub image: CpuTensor<f32>,
    pub lx: CpuTensor<f32>,
    pub ly: CpuTensor<f32>,
    pub ldet: CpuTensor<f32>,
    pub sigma: f32,
    pub octave: usize,
}

fn to_cpu_f32<S: Storage<f32> + 'static>(ctx: &ComputeDevice, tensor: &Tensor<f32, S>) -> CpuTensor<f32> {
    use std::any::TypeId;
    use cv_hal::storage::GpuStorage;
    if TypeId::of::<S>() == TypeId::of::<GpuStorage<f32>>() {
        let ptr = tensor as *const Tensor<f32, S> as *const Tensor<f32, GpuStorage<f32>>;
        let gpu_tensor = unsafe { &*ptr };
        match ctx {
            ComputeDevice::Gpu(gpu) => gpu_tensor.to_cpu_ctx(gpu).unwrap(),
            _ => panic!("Logic error: GpuStorage with non-GPU context"),
        }
    } else {
        let ptr = tensor as *const Tensor<f32, S> as *const CpuTensor<f32>;
        unsafe { &*ptr }.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_hal::cpu::CpuBackend;
    use cv_core::{TensorShape, storage::CpuStorage};

    fn create_test_image() -> Tensor<u8, CpuStorage<u8>> {
        let size = 128usize;
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                if ((x / 16) + (y / 16)) % 2 == 0 { data[y * size + x] = 255; }
            }
        }
        Tensor::from_vec(data, TensorShape::new(1, size, size))
    }

    #[test]
    fn test_akaze_detect() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);
        let tensor = create_test_image();
        let akaze = Akaze::new(AkazeParams::default());
        let kps = akaze.detect_ctx(&device, &tensor);
        println!("Detected {} AKAZE keypoints", kps.len());
        assert!(kps.len() > 0);
    }
}
