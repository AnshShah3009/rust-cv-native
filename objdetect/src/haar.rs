//! Haar Cascade Face Detection
//!
//! Implementation of the Viola-Jones object detection framework
//! using Haar-like features and boosted cascades.

use image::GrayImage;
use rayon::prelude::*;
use cv_core::{Rect, Tensor};

pub struct HaarCascade {
    pub stages: Vec<CascadeStage>,
    pub size: (u32, u32),
}

pub struct CascadeStage {
    pub threshold: f32,
    pub features: Vec<HaarFeature>,
}

pub struct HaarFeature {
    pub rects: Vec<(Rect, f32)>, // (rectangle, weight)
    pub threshold: f32,
    pub left_val: f32,
    pub right_val: f32,
}

impl HaarCascade {
    pub fn detect(&self, image: &GrayImage, scale_factor: f32, min_neighbors: u32) -> Vec<Rect> {
        // 1. Compute integral image
        let integral = compute_integral_image(image);
        
        let mut detections = Vec::new();
        let (img_w, img_h) = image.dimensions();
        
        let mut scale = 1.0f32;
        while (scale * self.size.0 as f32) < img_w as f32 && (scale * self.size.1 as f32) < img_h as f32 {
            let win_w = (self.size.0 as f32 * scale) as u32;
            let win_h = (self.size.1 as f32 * scale) as u32;
            let step = (scale * 2.0).max(1.0) as u32;

            for y in (0..img_h - win_h).step_by(step as usize) {
                for x in (0..img_w - win_w).step_by(step as usize) {
                    if self.evaluate_window(&integral, x, y, scale) {
                        detections.push(Rect::new(x as f32, y as f32, win_w as f32, win_h as f32));
                    }
                }
            }
            scale *= scale_factor;
        }

        // 2. Group detections (min_neighbors) - simplified
        detections
    }

    fn evaluate_window(&self, integral: &Tensor<u32>, x: u32, y: u32, scale: f32) -> bool {
        for stage in &self.stages {
            let mut stage_sum = 0.0f32;
            for feature in &stage.features {
                let feature_sum = feature.evaluate(integral, x, y, scale);
                stage_sum += if feature_sum < feature.threshold * scale * scale { feature.left_val } else { feature.right_val };
            }
            if stage_sum < stage.threshold {
                return false;
            }
        }
        true
    }
}

impl HaarFeature {
    fn evaluate(&self, integral: &Tensor<u32>, ox: u32, oy: u32, scale: f32) -> f32 {
        let mut sum = 0.0f32;
        for (r, weight) in &self.rects {
            let rx = ox + (r.x * scale) as u32;
            let ry = oy + (r.y * scale) as u32;
            let rw = (r.w * scale) as u32;
            let rh = (r.h * scale) as u32;
            
            sum += get_rect_sum(integral, rx, ry, rw, rh) as f32 * weight;
        }
        sum
    }
}

fn compute_integral_image(src: &GrayImage) -> Tensor<u32> {
    let (w, h) = src.dimensions();
    let mut integral = vec![0u32; (w as usize + 1) * (h as usize + 1)];
    let src_raw = src.as_raw();
    
    for y in 0..h as usize {
        let mut row_sum = 0u32;
        for x in 0..w as usize {
            row_sum += src_raw[y * w as usize + x] as u32;
            let idx = (y + 1) * (w as usize + 1) + (x + 1);
            integral[idx] = integral[idx - (w as usize + 1)] + row_sum;
        }
    }
    
    Tensor::from_vec(integral, cv_core::TensorShape::new(1, h as usize + 1, w as usize + 1))
}

fn get_rect_sum(integral: &Tensor<u32>, x: u32, y: u32, w: u32, h: u32) -> u32 {
    let iw = integral.shape.width;
    let x0 = x as usize;
    let y0 = y as usize;
    let x1 = (x + w) as usize;
    let y1 = (y + h) as usize;
    
    let data = integral.as_slice();
    data[y1 * iw + x1] + data[y0 * iw + x0] - data[y1 * iw + x0] - data[y0 * iw + x1]
}
