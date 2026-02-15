use image::GrayImage;
use std::f32::consts::PI;

pub struct HogParams {
    pub cell_size: usize,
    pub block_size: usize,
    pub n_bins: usize,
}

impl Default for HogParams {
    fn default() -> Self {
        Self {
            cell_size: 8,
            block_size: 2, // 2x2 cells
            n_bins: 9,
        }
    }
}

pub fn compute_hog(image: &GrayImage, params: &HogParams) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    
    // 1. Compute gradients
    let mut grad_mag = vec![0.0f32; width * height];
    let mut grad_ori = vec![0.0f32; width * height];
    
    for y in 1..height-1 {
        for x in 1..width-1 {
            let gx = image.get_pixel((x+1) as u32, y as u32)[0] as f32 - image.get_pixel((x-1) as u32, y as u32)[0] as f32;
            let gy = image.get_pixel(x as u32, (y+1) as u32)[0] as f32 - image.get_pixel(x as u32, (y-1) as u32)[0] as f32;
            
            grad_mag[y * width + x] = (gx * gx + gy * gy).sqrt();
            let mut ori = gy.atan2(gx) * 180.0 / PI;
            if ori < 0.0 { ori += 180.0; } // Unsigned gradients
            grad_ori[y * width + x] = ori;
        }
    }
    
    // 2. Cell Histogramming
    let n_cells_x = width / params.cell_size;
    let n_cells_y = height / params.cell_size;
    let mut cell_hists = vec![0.0f32; n_cells_x * n_cells_y * params.n_bins];
    
    for y in 0..height {
        let cy = y / params.cell_size;
        if cy >= n_cells_y { continue; }
        for x in 0..width {
            let cx = x / params.cell_size;
            if cx >= n_cells_x { continue; }
            
            let mag = grad_mag[y * width + x];
            let ori = grad_ori[y * width + x];
            
            let bin = (ori / (180.0 / params.n_bins as f32)).floor() as usize;
            let bin = bin.min(params.n_bins - 1);
            
            cell_hists[((cy * n_cells_x + cx) * params.n_bins) + bin] += mag;
        }
    }
    
    // 3. Block Normalization
    let mut descriptor = Vec::new();
    let b_w = params.block_size;
    
    for by in 0..n_cells_y - b_w + 1 {
        for bx in 0..n_cells_x - b_w + 1 {
            let mut block_vec = Vec::with_capacity(b_w * b_w * params.n_bins);
            for dy in 0..b_w {
                for dx in 0..b_w {
                    let c_idx = (by + dy) * n_cells_x + (bx + dx);
                    for b in 0..params.n_bins {
                        block_vec.push(cell_hists[c_idx * params.n_bins + b]);
                    }
                }
            }
            
            // L2 normalization
            let norm = block_vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for v in &mut block_vec { *v /= norm; }
            }
            descriptor.extend(block_vec);
        }
    }
    
    descriptor
}
