use image::{GrayImage, Luma};
use std::f32::consts::PI;

pub fn bilateral_filter(
    src: &GrayImage,
    d: i32,
    sigma_color: f32,
    sigma_space: f32,
) -> GrayImage {
    let width = src.width() as usize;
    let height = src.height() as usize;
    let mut dst = GrayImage::new(src.width(), src.height());

    let radius = if d <= 0 {
        (sigma_space * 1.5).ceil() as i32
    } else {
        d / 2
    };

    let gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
    let gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

    for y in 0..height {
        for x in 0..width {
            let mut sum_weight = 0.0f32;
            let mut sum_val = 0.0f32;
            let center_val = src.get_pixel(x as u32, y as u32)[0] as f32;

            for dy in -radius..=radius {
                let sy = y as i32 + dy;
                if sy < 0 || sy >= height as i32 { continue; }

                for dx in -radius..=radius {
                    let sx = x as i32 + dx;
                    if sx < 0 || sx >= width as i32 { continue; }

                    let val = src.get_pixel(sx as u32, sy as u32)[0] as f32;
                    
                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let color_diff = val - center_val;
                    let color_diff_sq = color_diff * color_diff;

                    let weight = (dist_sq * gauss_space_coeff + color_diff_sq * gauss_color_coeff).exp();
                    
                    sum_val += val * weight;
                    sum_weight += weight;
                }
            }

            dst.put_pixel(x as u32, y as u32, Luma([(sum_val / sum_weight).min(255.0) as u8]));
        }
    }

    dst
}
