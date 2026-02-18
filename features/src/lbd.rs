//! Line Binary Descriptor (LBD) for line matching
//!
//! LBD is a robust line descriptor that uses local gradient information.

use image::GrayImage;
use crate::line_matcher::LineDescriptor;
use cv_core::{Tensor, TensorShape, storage::CpuStorage};
use rayon::prelude::*;

pub struct LbdParams {
    pub n_layers: usize,
    pub n_bandwidth: usize,
}

impl Default for LbdParams {
    fn default() -> Self {
        Self {
            n_layers: 3,
            n_bandwidth: 7,
        }
    }
}

pub struct Lbd {
    _params: LbdParams,
}

impl Lbd {
    pub fn new(params: LbdParams) -> Self {
        Self { _params: params }
    }

    pub fn compute(&self, image: &GrayImage, segments: &[crate::hough::LineSegment]) -> Vec<LineDescriptor> {
        // Simplified LBD implementation
        // For each segment, compute a descriptor based on orthogonal gradients
        
        let (gx, gy) = cv_imgproc::sobel(image);
        let gx_raw = gx.as_raw();
        let gy_raw = gy.as_raw();
        let (w, h) = image.dimensions();

        segments.par_iter().map(|seg| {
            let mut desc = vec![0u8; 32]; // 256-bit descriptor
            
            // Vector of the line
            let dx = seg.x2 - seg.x1;
            let dy = seg.y2 - seg.y1;
            let len = (dx*dx + dy*dy).sqrt();
            
            if len > 1e-5 {
                // Orthogonal vector (normal)
                let nx = -dy / len;
                let ny = dx / len;
                
                // Sample points along the line and across the normal
                for i in 0..32 {
                    let mut bit_val = 0u8;
                    let t = i as f32 / 31.0;
                    let lx = seg.x1 + t * dx;
                    let ly = seg.y1 + t * dy;
                    
                    // Sample slightly offset along normal
                    let x_p = (lx + nx * 2.0).round() as i32;
                    let y_p = (ly + ny * 2.0).round() as i32;
                    let x_n = (lx - nx * 2.0).round() as i32;
                    let y_n = (ly - ny * 2.0).round() as i32;
                    
                    if x_p >= 0 && x_p < w as i32 && y_p >= 0 && y_p < h as i32 &&
                       x_n >= 0 && x_n < w as i32 && y_n >= 0 && y_n < h as i32 {
                        let idx_p = y_p as usize * w as usize + x_p as usize;
                        let idx_n = y_n as usize * w as usize + x_n as usize;
                        
                        let g_p = (gx_raw[idx_p] as f32).abs() + (gy_raw[idx_p] as f32).abs();
                        let g_n = (gx_raw[idx_n] as f32).abs() + (gy_raw[idx_n] as f32).abs();
                        
                        if g_p > g_n { bit_val = 1; }
                    }
                    
                    desc[i] = bit_val;
                }
            }
            
            // Pack bits into bytes
            let mut packed = vec![0u8; 4];
            for i in 0..32 {
                if desc[i] > 0 {
                    packed[i / 8] |= 1 << (i % 8);
                }
            }
            
            LineDescriptor {
                data: packed,
                segment: *seg,
            }
        }).collect()
    }
}
