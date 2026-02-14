use cv_core::{FeatureMatch, GrayImage, KeyPoint, KeyPoints, Matches};
use cv_imgproc::GrayImage;
use rayon::prelude::*;
use std::collections::VecDeque;

#[derive(Clone)]
pub struct OrbOptions {
    pub n_features: usize,
    pub scale_factor: f32,
    pub n_levels: usize,
    pub edge_threshold: i32,
    pub first_level: i32,
    pub wta_k: i32,
    pub score_type: OrbScoreType,
    pub patch_size: i32,
    pub fast_threshold: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbScoreType {
    Harris,
    Fast,
}

impl Default for OrbOptions {
    fn default() -> Self {
        Self {
            n_features: 500,
            scale_factor: 1.2,
            n_levels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            score_type: OrbScoreType::Harris,
            patch_size: 31,
            fast_threshold: 20,
        }
    }
}

pub struct Orb {
    options: OrbOptions,
    patterns: Vec<[[i32; 2]; 16]>,
    scale_factors: Vec<f32>,
    level_sizes: Vec<(u32, u32)>,
}

impl Orb {
    pub fn new(options: OrbOptions) -> Self {
        let mut orb = Self {
            options,
            patterns: Vec::new(),
            scale_factors: Vec::new(),
            level_sizes: Vec::new(),
        };

        orb.generate_scale_factors();
        orb.generate_pattern();

        orb
    }

    fn generate_scale_factors(&mut self) {
        self.scale_factors.clear();
        self.scale_factors.push(1.0);

        for _ in 1..self.options.n_levels {
            let last = *self.scale_factors.last().unwrap();
            self.scale_factors.push(last * self.options.scale_factor);
        }
    }

    fn generate_pattern(&mut self) {
        self.patterns.clear();

        let n_points = 16;

        for _ in 0..256 {
            let mut pattern = [[0i32; 2]; 16];

            let cx = self.options.patch_size / 2;
            let cy = self.options.patch_size / 2;

            for i in 0..n_points {
                let angle = (i as f32) * (std::f32::consts::PI * 2.0 / n_points);
                let r = self.options.patch_size / 3;

                pattern[i] = [
                    (cx as f32 + r * angle.cos()) as i32,
                    (cy as f32 + r * angle.sin()) as i32,
                ];
            }

            self.patterns.push(pattern);
        }
    }

    pub fn detect(&self, image: &GrayImage) -> Vec<KeyPoint> {
        let mut all_keypoints = Vec::new();
        let mut image_clone = image.clone();

        for level in 0..self.options.n_levels {
            let scale = self.scale_factors[level];

            if level > 0 {
                let new_width = (image.width() as f32 / scale) as u32;
                let new_height = (image.height() as f32 / scale) as u32;
                image_clone = cv_imgproc::resize(
                    image,
                    new_width.max(1),
                    new_height.max(1),
                    cv_imgproc::Interpolation::Linear,
                );
            }

            let kps = self.detect_at_level(&image_clone, scale);
            all_keypoints.extend(kps);
        }

        all_keypoints.truncate(self.options.n_features);

        all_keypoints
    }

    fn detect_at_level(&self, image: &GrayImage, scale: f32) -> Vec<KeyPoint> {
        let fast_kps = fast_detect(image, self.options.fast_threshold as u8);

        if self.options.score_type == OrbScoreType::Harris {
            let responses = compute_harris_response(image, &fast_kps);

            fast_kps
                .iter()
                .zip(responses.iter())
                .map(|(&kp, &response)| {
                    KeyPoint::new((kp.x as f32 * scale) as f64, (kp.y as f32 * scale) as f64)
                        .with_size((self.options.patch_size as f32 * scale) as f64)
                        .with_response(response as f64)
                })
                .collect()
        } else {
            fast_kps
                .iter()
                .map(|&kp| {
                    KeyPoint::new((kp.x as f32 * scale) as f64, (kp.y as f32 * scale) as f64)
                        .with_size((self.options.patch_size as f32 * scale) as f64)
                })
                .collect()
        }
    }

    pub fn compute(&self, image: &GrayImage, keypoints: &mut Vec<KeyPoint>) -> Vec<Vec<u8>> {
        let mut descriptors = Vec::with_capacity(keypoints.len());

        for kp in keypoints.iter_mut() {
            let descriptor = self.compute_single_descriptor(image, kp);
            descriptors.push(descriptor);

            if kp.angle < 0.0 {
                kp.angle = self.compute_orientation(image, kp.x as i32, kp.y as i32);
            }
        }

        descriptors
    }

    fn compute_single_descriptor(&self, image: &GrayImage, kp: &KeyPoint) -> Vec<u8> {
        let mut descriptor = vec![0u8; 32];

        let patch_size = self.options.patch_size as i32;
        let half_patch = patch_size / 2;

        let angle = -kp.angle.to_radians() as f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let kpx = kp.x as i32;
        let kpy = kp.y as i32;

        let patterns = &self.patterns[0];

        for (byte_idx, pattern) in patterns.chunks(4).enumerate() {
            let mut bit = 0u8;

            for (bit_idx, &pt) in pattern.iter().enumerate() {
                let rx = pt[0] - half_patch;
                let ry = pt[1] - half_patch;

                let rot_x = (rx * cos_a - ry * sin_a) as i32 + kpx;
                let rot_y = (rx * sin_a + ry * cos_a) as i32 + kpy;

                let v1 = get_pixel_or_zero(image, rot_x, rot_y);
                let v2 = get_pixel_or_zero(image, rot_x + 1, rot_y + 1);

                if v1 > v2 {
                    bit |= 1 << bit_idx;
                }
            }

            descriptor[byte_idx] = bit;
        }

        descriptor
    }

    fn compute_orientation(&self, image: &GrayImage, x: i32, y: i32) -> f64 {
        let mut m01 = 0i64;
        let mut m10 = 0i64;

        let half_patch = self.options.patch_size / 2;

        for dy in -half_patch..=half_patch {
            for dx in -half_patch..=half_patch {
                let val = get_pixel_or_zero(image, x + dx, y + dy) as i64;
                m01 += val as i64 * dy as i64;
                m10 += val as i64 * dx as i64;
            }
        }

        let angle = m01.atan2(m10 as f64);
        angle.to_degrees() as f64
    }

    pub fn detect_and_compute(&self, image: &GrayImage) -> (Vec<KeyPoint>, Vec<Vec<u8>>) {
        let mut keypoints = self.detect(image);
        let descriptors = self.compute(image, &mut keypoints);
        (keypoints, descriptors)
    }
}

fn get_pixel_or_zero(image: &GrayImage, x: i32, y: i32) -> u8 {
    if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
        image.get_pixel(x as u32, y as u32)[0]
    } else {
        0
    }
}

fn fast_detect(image: &GrayImage, threshold: u8) -> Vec<(i32, i32)> {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let mut keypoints = Vec::new();

    for y in 3..height - 3 {
        for x in 3..width - 3 {
            let p = image.get_pixel(x as u32, y as u32)[0];

            let circle = [
                (x - 3, y),
                (x - 2, y + 1),
                (x - 1, y + 2),
                (x, y + 3),
                (x + 1, y + 2),
                (x + 2, y + 1),
                (x + 3, y),
                (x + 2, y - 1),
                (x + 1, y - 2),
                (x, y - 3),
                (x - 1, y - 2),
                (x - 2, y - 1),
            ];

            let brighter = circle
                .iter()
                .filter(|(cx, cy)| image.get_pixel(*cx as u32, *cy as u32)[0] > p + threshold)
                .count();

            let darker = circle
                .iter()
                .filter(|(cx, cy)| image.get_pixel(*cx as u32, *cy as u32)[0] < p - threshold)
                .count();

            if brighter >= 9 || darker >= 9 {
                keypoints.push((x, y));
            }
        }
    }

    keypoints
}

fn compute_harris_response(image: &GrayImage, keypoints: &[(i32, i32)]) -> Vec<f32> {
    let k = 0.04;
    let half_window = 2;

    keypoints
        .iter()
        .map(|&(x, y)| {
            let mut i_xx = 0.0f32;
            let mut i_yy = 0.0f32;
            let mut i_xy = 0.0f32;

            for dy in -half_window..=half_window {
                for dx in -half_window..=half_window {
                    let gx = get_pixel_or_zero(image, x + dx + 1, y + dy) as f32
                        - get_pixel_or_zero(image, x + dx - 1, y + dy) as f32;
                    let gy = get_pixel_or_zero(image, x + dx, y + dy + 1) as f32
                        - get_pixel_or_zero(image, x + dx, y + dy - 1) as f32;

                    i_xx += gx * gx;
                    i_yy += gy * gy;
                    i_xy += gx * gy;
                }
            }

            let det = i_xx * i_yy - i_xy * i_xy;
            let trace = i_xx + i_yy;
            let response = det - k * trace * trace;

            response.max(0.0)
        })
        .collect()
}
