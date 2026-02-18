//! Histogram of Oriented Gradients (HOG)
//!
//! Robust descriptor for object detection, particularly pedestrians.

use cv_imgproc::sobel;
use image::GrayImage;
use rayon::prelude::*;

pub struct HogParams {
    pub cell_size: usize,
    pub block_size: usize, // in cells
    pub n_bins: usize,
}

impl Default for HogParams {
    fn default() -> Self {
        Self {
            cell_size: 8,
            block_size: 2,
            n_bins: 9,
        }
    }
}

pub struct Hog {
    params: HogParams,
}

impl Hog {
    pub fn new(params: HogParams) -> Self {
        Self { params }
    }

    pub fn compute(&self, image: &GrayImage) -> Vec<f32> {
        let (gx, gy) = sobel(image);
        let width = image.width() as usize;
        let height = image.height() as usize;
        
        let n_cells_x = width / self.params.cell_size;
        let n_cells_y = height / self.params.cell_size;
        
        // 1. Compute cell histograms
        let mut cell_histograms = vec![0.0f32; n_cells_x * n_cells_y * self.params.n_bins];
        
        // This can be parallelized over cells
        cell_histograms.par_chunks_mut(self.params.n_bins).enumerate().for_each(|(cell_idx, hist)| {
            let cx = cell_idx % n_cells_x;
            let cy = cell_idx / n_cells_x;
            
            let x_start = cx * self.params.cell_size;
            let y_start = cy * self.params.cell_size;
            
            for dy in 0..self.params.cell_size {
                for dx in 0..self.params.cell_size {
                    let x = x_start + dx;
                    let y = y_start + dy;
                    let idx = y * width + x;
                    
                    let dx_val = gx.as_raw()[idx] as f32 - 128.0;
                    let dy_val = gy.as_raw()[idx] as f32 - 128.0;
                    
                    let mag = (dx_val * dx_val + dy_val * dy_val).sqrt();
                    let mut angle = dy_val.atan2(dx_val).to_degrees();
                    if angle < 0.0 { angle += 180.0; } // Unsigned gradients
                    
                    let bin = (angle / (180.0 / self.params.n_bins as f32)) as usize % self.params.n_bins;
                    hist[bin] += mag;
                }
            }
        });

        // 2. Block normalization (L2-Hys)
        let n_blocks_x = n_cells_x - self.params.block_size + 1;
        let n_blocks_y = n_cells_y - self.params.block_size + 1;
        let block_dim = self.params.block_size * self.params.block_size * self.params.n_bins;
        
        let mut descriptor = vec![0.0f32; n_blocks_x * n_blocks_y * block_dim];
        
        descriptor.par_chunks_mut(block_dim).enumerate().for_each(|(block_idx, block_vec)| {
            let bx = block_idx % n_blocks_x;
            let by = block_idx / n_blocks_x;
            
            let mut norm_sq = 0.0f32;
            for j in 0..self.params.block_size {
                for i in 0..self.params.block_size {
                    let cell_x = bx + i;
                    let cell_y = by + j;
                    let cell_base = (cell_y * n_cells_x + cell_x) * self.params.n_bins;
                    
                    for b in 0..self.params.n_bins {
                        let val = cell_histograms[cell_base + b];
                        block_vec[(j * self.params.block_size + i) * self.params.n_bins + b] = val;
                        norm_sq += val * val;
                    }
                }
            }
            
            // Normalize
            let norm = (norm_sq + 1e-6).sqrt();
            for v in block_vec.iter_mut() {
                *v = (*v / norm).min(0.2);
            }
            // Re-normalize after clipping
            let final_norm = (block_vec.iter().map(|&v| v * v).sum::<f32>() + 1e-6).sqrt();
            for v in block_vec.iter_mut() {
                *v /= final_norm;
            }
        });

        descriptor
    }
}
