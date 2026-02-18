//! ORB (Oriented FAST and Rotated BRIEF) implementation
//!
//! ORB combines the FAST keypoint detector with a modified BRIEF descriptor
//! that includes orientation information for rotation invariance.

use crate::descriptor::{Descriptor, DescriptorExtractor, Descriptors};
use crate::fast::fast_detect;
use cv_core::{KeyPoint, KeyPoints, Tensor, storage::Storage};
use cv_hal::compute::ComputeDevice;
use cv_hal::context::ComputeContext;
use cv_hal::tensor_ext::TensorToCpu;
use image::GrayImage;

/// ORB feature detector and descriptor
pub struct Orb {
    n_features: usize,
    scale_factor: f32,
    n_levels: usize,
    #[allow(dead_code)]
    edge_threshold: i32,
    #[allow(dead_code)]
    first_level: i32,
    #[allow(dead_code)]
    wta_k: i32,
    #[allow(dead_code)]
    score_type: ScoreType,
    patch_size: i32,
    fast_threshold: u8,
}

#[derive(Clone, Copy)]
pub enum ScoreType {
    Harris,
    Fast,
}

impl Default for Orb {
    fn default() -> Self {
        Self {
            n_features: 500,
            scale_factor: 1.2,
            n_levels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            score_type: ScoreType::Harris,
            patch_size: 31,
            fast_threshold: 20,
        }
    }
}

impl Orb {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_n_features(mut self, n: usize) -> Self {
        self.n_features = n;
        self
    }

    pub fn with_n_levels(mut self, n: usize) -> Self {
        self.n_levels = n;
        self
    }

    pub fn with_scale_factor(mut self, factor: f32) -> Self {
        self.scale_factor = factor;
        self
    }

    pub fn with_fast_threshold(mut self, threshold: u8) -> Self {
        self.fast_threshold = threshold;
        self
    }

    /// Detect keypoints using FAST at multiple scales
    pub fn detect(&self, image: &GrayImage) -> KeyPoints {
        let mut all_keypoints = Vec::new();
        let mut scale = 1.0f32;

        for level in 0..self.n_levels {
            let scaled = if level == 0 {
                image.clone()
            } else {
                scale_image(image, scale)
            };

            let kps = fast_detect(&scaled, self.fast_threshold, self.n_features * 2);

            for kp in kps.keypoints {
                let scaled_kp = KeyPoint::new(kp.x / scale as f64, kp.y / scale as f64)
                    .with_size(self.patch_size as f64 * scale as f64)
                    .with_octave(level as i32);

                all_keypoints.push(scaled_kp);
            }

            scale *= self.scale_factor;
        }

        all_keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
        all_keypoints.truncate(self.n_features);

        KeyPoints { keypoints: all_keypoints }
    }

    /// Detect keypoints using FAST at multiple scales with acceleration
    pub fn detect_ctx<S: Storage<u8> + 'static>(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> KeyPoints {
        let mut all_keypoints = Vec::new();
        let mut scale = 1.0f32;

        for level in 0..self.n_levels {
            let scaled = if level == 0 {
                image.clone()
            } else {
                let new_w = (image.shape.width as f32 / scale) as usize;
                let new_h = (image.shape.height as f32 / scale) as usize;
                ctx.resize(image, (new_w, new_h)).unwrap()
            };

            let score_map = ctx.fast_detect(&scaled, self.fast_threshold, true).unwrap();
            let kps = extract_keypoints_from_score_map(ctx, &score_map, self.n_features * 2);

            for kp in kps.keypoints {
                let scaled_kp = KeyPoint::new(kp.x / scale as f64, kp.y / scale as f64)
                    .with_size(self.patch_size as f64 * scale as f64)
                    .with_octave(level as i32)
                    .with_response(kp.response);

                all_keypoints.push(scaled_kp);
            }

            scale *= self.scale_factor;
        }

        all_keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
        all_keypoints.truncate(self.n_features);

        KeyPoints { keypoints: all_keypoints }
    }

    /// Compute orientations for keypoints using intensity centroid
    pub fn compute_orientations(&self, image: &GrayImage, keypoints: &mut KeyPoints) {
        let patch_size = self.patch_size as i32;
        let half_patch = patch_size / 2;

        for kp in &mut keypoints.keypoints {
            let x = kp.x as i32;
            let y = kp.y as i32;

            let mut m01 = 0.0f64;
            let mut m10 = 0.0f64;

            for dy in -half_patch..half_patch {
                for dx in -half_patch..half_patch {
                    let px = x + dx;
                    let py = y + dy;

                    if px >= 0 && px < image.width() as i32 && py >= 0 && py < image.height() as i32 {
                        let intensity = image.get_pixel(px as u32, py as u32)[0] as f64;
                        m01 += intensity * dy as f64;
                        m10 += intensity * dx as f64;
                    }
                }
            }

            let angle = m01.atan2(m10);
            kp.angle = angle.to_degrees();
        }
    }
}

fn extract_keypoints_from_score_map<S: Storage<u8> + 'static>(ctx: &ComputeDevice, score_map: &Tensor<u8, S>, max_kps: usize) -> KeyPoints {
    use std::any::TypeId;
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;

    let cpu_tensor = if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
        let input_ptr = score_map as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
        let input_gpu = unsafe { &*input_ptr };
        let gpu_ctx = match ctx {
            ComputeDevice::Gpu(g) => g,
            _ => panic!("Logic error: GpuStorage with CpuBackend"),
        };
        input_gpu.to_cpu_ctx(gpu_ctx).unwrap()
    } else {
        let input_ptr = score_map as *const Tensor<u8, S> as *const Tensor<u8, CpuStorage<u8>>;
        let input_cpu = unsafe { &*input_ptr };
        input_cpu.clone()
    };

    let slice = cpu_tensor.storage.as_slice().unwrap();
    let (h, w) = cpu_tensor.shape.hw();
    
    let mut kps = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let score = slice[y * w + x];
            if score > 0 {
                kps.push(KeyPoint::new(x as f64, y as f64).with_response(score as f64));
            }
        }
    }
    
    if kps.len() > max_kps {
        kps.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
        kps.truncate(max_kps);
    }
    
    KeyPoints { keypoints: kps }
}

impl DescriptorExtractor for Orb {
    fn extract(&self, image: &GrayImage, keypoints: &KeyPoints) -> Descriptors {
        let mut descriptors = Descriptors::with_capacity(keypoints.len());
        let pattern = generate_steered_brief_pattern(self.patch_size);

        for kp in keypoints.iter() {
            if let Some(desc) = compute_orb_descriptor(image, kp, &pattern, self.patch_size) {
                descriptors.push(desc);
            }
        }

        descriptors
    }
}

/// Generate BRIEF sampling pattern with rotation support
fn generate_steered_brief_pattern(patch_size: i32) -> Vec<(f32, f32, f32, f32)> {
    use rand::thread_rng;
    use rand::Rng;

    let mut rng = thread_rng();
    let num_pairs = 256; // 256 bits = 32 bytes
    let half_size = patch_size as f32 / 2.0;

    let mut pairs = Vec::with_capacity(num_pairs);

    for _ in 0..num_pairs {
        let x1 = rng.gen_range(-half_size..half_size);
        let y1 = rng.gen_range(-half_size..half_size);
        let x2 = rng.gen_range(-half_size..half_size);
        let y2 = rng.gen_range(-half_size..half_size);
        pairs.push((x1, y1, x2, y2));
    }

    pairs
}

/// Compute ORB descriptor with rotation
fn compute_orb_descriptor(
    image: &GrayImage,
    kp: &KeyPoint,
    pattern: &[(f32, f32, f32, f32)],
    patch_size: i32,
) -> Option<Descriptor> {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let cx = kp.x as i32;
    let cy = kp.y as i32;

    let half_patch = patch_size / 2;
    if cx < half_patch || cx >= width - half_patch || cy < half_patch || cy >= height - half_patch {
        return None;
    }

    let angle_rad = kp.angle.to_radians();
    let cos_a = angle_rad.cos() as f32;
    let sin_a = angle_rad.sin() as f32;

    let mut descriptor_data = vec![0u8; 32]; // 256 bits
    let mut bit_idx = 0;
    let mut byte_idx = 0;

    for &(x1, y1, x2, y2) in pattern {
        let rx1 = cos_a * x1 - sin_a * y1;
        let ry1 = sin_a * x1 + cos_a * y1;
        let rx2 = cos_a * x2 - sin_a * y2;
        let ry2 = sin_a * x2 + cos_a * y2;

        let px1 = (cx as f32 + rx1) as i32;
        let py1 = (cy as f32 + ry1) as i32;
        let px2 = (cx as f32 + rx2) as i32;
        let py2 = (cy as f32 + ry2) as i32;

        if px1 < 0 || px1 >= width || py1 < 0 || py1 >= height || px2 < 0 || px2 >= width || py2 < 0 || py2 >= height {
            continue;
        }

        let val1 = image.get_pixel(px1 as u32, py1 as u32)[0];
        let val2 = image.get_pixel(px2 as u32, py2 as u32)[0];

        if val1 < val2 {
            descriptor_data[byte_idx] |= 1 << (7 - bit_idx);
        }

        bit_idx += 1;
        if bit_idx == 8 {
            bit_idx = 0;
            byte_idx += 1;
        }
    }

    Some(Descriptor::new(descriptor_data, kp.clone()))
}

fn scale_image(image: &GrayImage, scale: f32) -> GrayImage {
    let new_width = (image.width() as f32 / scale) as u32;
    let new_height = (image.height() as f32 / scale) as u32;

    image::imageops::resize(
        image,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    )
}

pub fn orb_detect_and_compute(image: &GrayImage, n_features: usize) -> (KeyPoints, Descriptors) {
    let orb = Orb::new().with_n_features(n_features);
    let mut keypoints = orb.detect(image);
    orb.compute_orientations(image, &mut keypoints);
    let descriptors = orb.extract(image, &keypoints);
    (keypoints, descriptors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn create_test_image() -> GrayImage {
        let size = 128u32;
        let mut img = GrayImage::new(size, size);

        let square_size = 16;
        for y in 0..size {
            for x in 0..size {
                let is_white = ((x / square_size) + (y / square_size)) % 2 == 0;
                img.put_pixel(x, y, if is_white { Luma([255]) } else { Luma([0]) });
            }
        }

        img
    }

    #[test]
    fn test_orb_detect_ctx() {
        use cv_hal::cpu::CpuBackend;
        let img_gray = create_test_image();
        let shape = cv_core::TensorShape::new(1, img_gray.height() as usize, img_gray.width() as usize);
        let tensor: cv_core::Tensor<u8, cv_core::storage::CpuStorage<u8>> = cv_core::Tensor::from_vec(img_gray.to_vec(), shape);
        
        let cpu = CpuBackend::new().unwrap();
        let device = cv_hal::compute::ComputeDevice::Cpu(&cpu);
        let orb = Orb::new().with_n_features(50);
        
        let kps = orb.detect_ctx(&device, &tensor);
        println!("Detected {} keypoints with accelerated ORB (CPU)", kps.len());
        assert!(kps.len() > 0);
    }
}
