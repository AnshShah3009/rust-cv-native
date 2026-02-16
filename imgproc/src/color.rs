use image::{GrayImage, RgbImage};
use rayon::prelude::*;

pub fn convert_gray_to_rgb(gray: &GrayImage) -> RgbImage {
    let (w, h) = gray.dimensions();
    let count = (w * h) as usize;
    let mut rgb_data = vec![0u8; count * 3];
    let gray_data = gray.as_raw();

    rgb_data
        .par_chunks_mut(3)
        .zip(gray_data.par_iter())
        .for_each(|(rgb_pixel, &g)| {
            rgb_pixel[0] = g;
            rgb_pixel[1] = g;
            rgb_pixel[2] = g;
        });

    RgbImage::from_raw(w, h, rgb_data).unwrap()
}

pub fn convert_rgb_to_gray(rgb: &RgbImage) -> GrayImage {
    let (w, h) = rgb.dimensions();
    let count = (w * h) as usize;
    let mut gray_data = vec![0u8; count];
    let rgb_data = rgb.as_raw();

    gray_data
        .par_iter_mut()
        .zip(rgb_data.par_chunks(3))
        .for_each(|(g, rgb_pixel)| {
            // Using standard luminance weights: 0.299 R + 0.587 G + 0.114 B
            *g = (0.299 * rgb_pixel[0] as f32
                + 0.587 * rgb_pixel[1] as f32
                + 0.114 * rgb_pixel[2] as f32) as u8;
        });

    GrayImage::from_raw(w, h, gray_data).unwrap()
}

pub fn to_gray(img: &RgbImage) -> GrayImage {
    convert_rgb_to_gray(img)
}

pub fn to_rgb(img: &GrayImage) -> RgbImage {
    convert_gray_to_rgb(img)
}

pub fn adjust_brightness(src: &GrayImage, factor: f32) -> GrayImage {
    let mut output = src.clone();
    output.as_mut().par_iter_mut().for_each(|p| {
        *p = (*p as f32 * factor).clamp(0.0, 255.0) as u8;
    });
    output
}

pub fn adjust_contrast(src: &GrayImage, factor: f32) -> GrayImage {
    let mut output = src.clone();
    let mean = compute_mean_intensity(src);
    output.as_mut().par_iter_mut().for_each(|p| {
        *p = ((*p as f32 - mean) * factor + mean).clamp(0.0, 255.0) as u8;
    });
    output
}

pub fn compute_mean_intensity(src: &GrayImage) -> f32 {
    let raw = src.as_raw();
    if raw.is_empty() {
        return 0.0;
    }
    let sum: u64 = raw.par_iter().map(|&p| p as u64).sum();
    (sum as f32) / (raw.len() as f32)
}

pub fn invert(src: &GrayImage) -> GrayImage {
    let mut output = src.clone();
    output.as_mut().par_iter_mut().for_each(|p| {
        *p = 255 - *p;
    });
    output
}

pub fn equalize_histogram(src: &GrayImage) -> GrayImage {
    let raw = src.as_raw();
    if raw.is_empty() {
        return src.clone();
    }

    // Compute histogram in parallel
    let hist = raw
        .par_chunks(4096)
        .fold(
            || [0u32; 256],
            |mut local_hist, chunk| {
                for &val in chunk {
                    local_hist[val as usize] += 1;
                }
                local_hist
            },
        )
        .reduce(
            || [0u32; 256],
            |mut h1, h2| {
                for i in 0..256 {
                    h1[i] += h2[i];
                }
                h1
            },
        );

    // Compute CDF (cumulative distribution function) - Serial is fast enough for 256 elements
    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0);
    let total = raw.len() as u32;

    // Build LUT
    let mut lut = [0u8; 256];
    if total > cdf_min {
        let denom = (total - cdf_min) as f32;
        for i in 0..256 {
            lut[i] = ((cdf[i].saturating_sub(cdf_min)) as f32 / denom * 255.0).round() as u8;
        }
    } else {
        // Edge case: all pixels are same or image is empty
        return src.clone();
    }

    // Apply LUT in parallel
    let mut output = src.clone();
    output.as_mut().par_iter_mut().for_each(|p| {
        *p = lut[*p as usize];
    });
    output
}
