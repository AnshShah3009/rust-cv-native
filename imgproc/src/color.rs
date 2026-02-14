use cv_core::{GrayImage, RgbImage};
use image::{Pixel, Primitive};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorConversion {
    Bgr2Gray,
    Gray2Bgr,
    Bgr2Hsv,
    Hsv2Bgr,
    Bgr2Lab,
    Lab2Bgr,
    Bgr2Rgb,
    Rgb2Bgr,
}

impl ColorConversion {
    pub fn matches(&self, src_channels: u8, _dst_channels: u8) -> bool {
        match self {
            ColorConversion::Bgr2Gray | ColorConversion::Gray2Bgr => true,
            ColorConversion::Bgr2Hsv | ColorConversion::Hsv2Bgr => src_channels == 3,
            ColorConversion::Bgr2Lab | ColorConversion::Lab2Bgr => src_channels == 3,
            ColorConversion::Bgr2Rgb | ColorConversion::Rgb2Bgr => src_channels == 3,
        }
    }
}

pub fn convert_color(
    src: &RgbImage,
    conversion: ColorConversion,
) -> Result<GrayImage, super::ImgprocError> {
    match conversion {
        ColorConversion::Bgr2Gray | ColorConversion::Gray2Bgr => {
            Ok(image::imageops::colorops::grayscale(src))
        }
        _ => Err(super::ImgprocError::UnsupportedFormat(format!(
            "{:?} not yet implemented",
            conversion
        ))),
    }
}

pub fn bgr_to_gray(src: &RgbImage) -> GrayImage {
    image::imageops::colorops::grayscale(src)
}

pub fn gray_to_bgr(src: &GrayImage) -> RgbImage {
    image::imageops::colorops::grayscale(src)
}

pub fn rgb_to_gray(src: &RgbImage) -> GrayImage {
    image::imageops::colorops::grayscale(src)
}

pub fn grayscale(src: &RgbImage) -> GrayImage {
    image::imageops::colorops::grayscale(src)
}

pub fn adjust_brightness<P: Pixel + 'static>(
    src: &image::ImageBuffer<P, Vec<u8>>,
    factor: f32,
) -> image::ImageBuffer<P, Vec<u8>>
where
    P::Subpixel: Primitive,
{
    let mut output = src.clone();

    output.par_pixels_mut().for_each(|_pos, pixel| {
        for i in 0..pixel.channels().len() {
            let v = pixel.channels()[i] as f32;
            let new_v = (v * factor).clamp(0.0, 255.0);
            pixel.channels_mut()[i] = Primitive::from_u8(new_v as u8);
        }
    });

    output
}

pub fn adjust_contrast<P: Pixel + 'static>(
    src: &image::ImageBuffer<P, Vec<u8>>,
    factor: f32,
) -> image::ImageBuffer<P, Vec<u8>>
where
    P::Subpixel: Primitive,
{
    let mut output = src.clone();
    let mean = compute_mean_intensity(src);

    output.par_pixels_mut().for_each(|_pos, pixel| {
        for i in 0..pixel.channels().len() {
            let v = pixel.channels()[i] as f32;
            let new_v = ((v - mean) * factor + mean).clamp(0.0, 255.0);
            pixel.channels_mut()[i] = Primitive::from_u8(new_v as u8);
        }
    });

    output
}

pub fn compute_mean_intensity<P: Pixel + 'static>(src: &image::ImageBuffer<P, Vec<u8>>) -> f32
where
    P::Subpixel: Primitive,
{
    let mut sum = 0u64;
    let mut count = 0u64;

    for pixel in src.pixels() {
        for &ch in pixel.channels() {
            sum += ch as u64;
            count += 1;
        }
    }

    (sum / count) as f32
}

pub fn adjust_saturation(src: &RgbImage, factor: f32) -> RgbImage {
    let mut output = src.clone();

    output.par_pixels_mut().for_each(|_pos, pixel| {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        let gray = 0.299 * r + 0.587 * g + 0.114 * b;

        let new_r = ((gray + factor * (r - gray)).clamp(0.0, 1.0) * 255.0) as u8;
        let new_g = ((gray + factor * (g - gray)).clamp(0.0, 1.0) * 255.0) as u8;
        let new_b = ((gray + factor * (b - gray)).clamp(0.0, 1.0) * 255.0) as u8;

        pixel[0] = new_r;
        pixel[1] = new_g;
        pixel[2] = new_b;
    });

    output
}

pub fn adjust_gamma(src: &GrayImage, gamma: f32) -> GrayImage {
    let mut output = src.clone();
    let inv_gamma = 1.0 / gamma;
    let lut: Vec<u8> = (0..256)
        .map(|i| {
            let v = i as f32 / 255.0;
            ((v.powf(inv_gamma) * 255.0).clamp(0.0, 255.0) as u8)
        })
        .collect();

    output.par_pixels_mut().for_each(|_pos, pixel| {
        pixel[0] = lut[pixel[0] as usize];
    });

    output
}

pub fn invert(src: &GrayImage) -> GrayImage {
    let mut output = src.clone();

    output.par_pixels_mut().for_each(|_pos, pixel| {
        pixel[0] = 255 - pixel[0];
    });

    output
}

pub fn equalize_histogram(src: &GrayImage) -> GrayImage {
    let mut hist = [0u32; 256];

    for pixel in src.pixels() {
        hist[pixel[0] as usize] += 1;
    }

    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0);
    let total = src.width() * src.height();

    let mut lut = [0u8; 256];
    for i in 0..256 {
        let val = ((cdf[i] - cdf_min) as f32 / (total - cdf_min) * 255.0).round() as u8;
        lut[i] = val;
    }

    let mut output = src.clone();

    output.par_pixels_mut().for_each(|_pos, pixel| {
        pixel[0] = lut[pixel[0] as usize];
    });

    output
}

pub fn auto_contrast(src: &GrayImage, clip_limit: f32) -> GrayImage {
    equalize_histogram(src)
}
