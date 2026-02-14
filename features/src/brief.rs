use crate::descriptor::{Descriptor, DescriptorExtractor, Descriptors};
use cv_core::KeyPoints;
use image::GrayImage;
use rand::thread_rng;
use rand::Rng;

pub struct BriefDescriptor {
    pairs: Vec<(i32, i32, i32, i32)>,
    patch_size: i32,
    descriptor_size: usize,
}

impl BriefDescriptor {
    pub fn new(descriptor_size: usize, patch_size: i32) -> Self {
        let num_pairs = descriptor_size * 8;
        let mut rng = thread_rng();
        let mut pairs = Vec::with_capacity(num_pairs);

        let half_size = patch_size / 2;

        for _ in 0..num_pairs {
            let x1 = rng.gen_range(-half_size..half_size);
            let y1 = rng.gen_range(-half_size..half_size);
            let x2 = rng.gen_range(-half_size..half_size);
            let y2 = rng.gen_range(-half_size..half_size);
            pairs.push((x1, y1, x2, y2));
        }

        Self {
            pairs,
            patch_size,
            descriptor_size,
        }
    }

    fn compute_brief(&self, image: &GrayImage, x: i32, y: i32) -> Descriptor {
        let width = image.width() as i32;
        let height = image.height() as i32;

        let mut descriptor_data = vec![0u8; self.descriptor_size];
        let mut bit_idx = 0;
        let mut byte_idx = 0;

        for (x1, y1, x2, y2) in &self.pairs {
            let px1 = (x + x1).clamp(0, width - 1) as u32;
            let py1 = (y + y1).clamp(0, height - 1) as u32;
            let px2 = (x + x2).clamp(0, width - 1) as u32;
            let py2 = (y + y2).clamp(0, height - 1) as u32;

            let val1 = image.get_pixel(px1, py1)[0];
            let val2 = image.get_pixel(px2, py2)[0];

            if val1 < val2 {
                descriptor_data[byte_idx] |= 1 << (7 - bit_idx);
            }

            bit_idx += 1;
            if bit_idx == 8 {
                bit_idx = 0;
                byte_idx += 1;
            }
        }

        Descriptor::new(descriptor_data, cv_core::KeyPoint::new(x as f64, y as f64))
    }
}

impl DescriptorExtractor for BriefDescriptor {
    fn extract(&self, image: &GrayImage, keypoints: &KeyPoints) -> Descriptors {
        let mut descriptors = Descriptors::with_capacity(keypoints.len());
        let half_size = self.patch_size / 2;

        for kp in keypoints.iter() {
            let x = kp.x as i32;
            let y = kp.y as i32;

            if x >= half_size
                && x < (image.width() as i32 - half_size)
                && y >= half_size
                && y < (image.height() as i32 - half_size)
            {
                let desc = self.compute_brief(image, x, y);
                descriptors.push(desc);
            }
        }

        descriptors
    }
}

pub fn extract_brief(
    image: &GrayImage,
    keypoints: &KeyPoints,
    descriptor_size: usize,
) -> Descriptors {
    let brief = BriefDescriptor::new(descriptor_size, 31);
    brief.extract(image, keypoints)
}
