use crate::{gaussian_blur_with_border, BorderMode};
use image::GrayImage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdType {
    Binary,
    BinaryInv,
    Trunc,
    ToZero,
    ToZeroInv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMethod {
    MeanC,
    GaussianC,
}

pub fn threshold(src: &GrayImage, thresh: u8, max_value: u8, typ: ThresholdType) -> GrayImage {
    let mut dst = GrayImage::new(src.width(), src.height());

    for (i, out_px) in dst.as_mut().iter_mut().enumerate() {
        let value = src.as_raw()[i];
        *out_px = apply_threshold(value, thresh, max_value, typ);
    }

    dst
}

pub fn threshold_otsu(
    src: &GrayImage,
    max_value: u8,
    typ: ThresholdType,
) -> (u8, GrayImage) {
    let hist = histogram(src);
    let total = (src.width() * src.height()) as f64;

    let mut sum_all = 0.0f64;
    for (i, &count) in hist.iter().enumerate() {
        sum_all += (i as f64) * (count as f64);
    }

    let mut weight_background = 0.0f64;
    let mut sum_background = 0.0f64;
    let mut best_between = -1.0f64;
    let mut best_threshold = 0u8;

    for t in 0u16..=255 {
        let idx = t as usize;
        weight_background += hist[idx] as f64;
        if weight_background <= f64::EPSILON {
            continue;
        }

        let weight_foreground = total - weight_background;
        if weight_foreground <= f64::EPSILON {
            break;
        }

        sum_background += (t as f64) * (hist[idx] as f64);
        let mean_background = sum_background / weight_background;
        let mean_foreground = (sum_all - sum_background) / weight_foreground;
        let diff = mean_background - mean_foreground;
        let between = weight_background * weight_foreground * diff * diff;

        if between > best_between {
            best_between = between;
            best_threshold = t as u8;
        }
    }

    let dst = threshold(src, best_threshold, max_value, typ);
    (best_threshold, dst)
}

pub fn adaptive_threshold(
    src: &GrayImage,
    max_value: u8,
    method: AdaptiveMethod,
    typ: ThresholdType,
    block_size: u32,
    c: f32,
) -> GrayImage {
    assert!(block_size >= 3, "block_size must be >= 3");
    assert!(block_size % 2 == 1, "block_size must be odd");
    assert!(
        matches!(typ, ThresholdType::Binary | ThresholdType::BinaryInv),
        "adaptive threshold supports Binary or BinaryInv types"
    );

    let mut dst = GrayImage::new(src.width(), src.height());
    let local = match method {
        AdaptiveMethod::MeanC => local_mean_image(src, block_size),
        AdaptiveMethod::GaussianC => local_gaussian_image(src, block_size),
    };

    for i in 0..src.as_raw().len() {
        let value = src.as_raw()[i] as f32;
        let threshold = local.as_raw()[i] as f32 - c;
        let out = match typ {
            ThresholdType::Binary => {
                if value > threshold {
                    max_value
                } else {
                    0
                }
            }
            ThresholdType::BinaryInv => {
                if value > threshold {
                    0
                } else {
                    max_value
                }
            }
            _ => 0,
        };
        dst.as_mut()[i] = out;
    }

    dst
}

fn apply_threshold(value: u8, thresh: u8, max_value: u8, typ: ThresholdType) -> u8 {
    match typ {
        ThresholdType::Binary => {
            if value > thresh {
                max_value
            } else {
                0
            }
        }
        ThresholdType::BinaryInv => {
            if value > thresh {
                0
            } else {
                max_value
            }
        }
        ThresholdType::Trunc => value.min(thresh),
        ThresholdType::ToZero => {
            if value > thresh {
                value
            } else {
                0
            }
        }
        ThresholdType::ToZeroInv => {
            if value > thresh {
                0
            } else {
                value
            }
        }
    }
}

fn histogram(src: &GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &px in src.as_raw() {
        hist[px as usize] += 1;
    }
    hist
}

fn local_mean_image(src: &GrayImage, block_size: u32) -> GrayImage {
    let width = src.width() as usize;
    let height = src.height() as usize;
    let radius = (block_size / 2) as i32;
    let stride = width + 1;

    let mut integral = vec![0u32; (width + 1) * (height + 1)];
    for y in 0..height {
        let mut row_sum = 0u32;
        for x in 0..width {
            row_sum += src.as_raw()[y * width + x] as u32;
            let idx = (y + 1) * stride + (x + 1);
            integral[idx] = integral[idx - stride] + row_sum;
        }
    }

    let mut out = GrayImage::new(src.width(), src.height());
    for y in 0..height {
        for x in 0..width {
            let x0 = (x as i32 - radius).max(0) as usize;
            let y0 = (y as i32 - radius).max(0) as usize;
            let x1 = (x as i32 + radius + 1).min(width as i32) as usize;
            let y1 = (y as i32 + radius + 1).min(height as i32) as usize;

            let sum = integral[y1 * stride + x1] + integral[y0 * stride + x0]
                - integral[y0 * stride + x1]
                - integral[y1 * stride + x0];
            let area = ((x1 - x0) * (y1 - y0)) as u32;
            out.as_mut()[y * width + x] = (sum / area).min(255) as u8;
        }
    }
    out
}

fn local_gaussian_image(src: &GrayImage, block_size: u32) -> GrayImage {
    let sigma = 0.3 * (((block_size as f32) - 1.0) * 0.5 - 1.0) + 0.8;
    gaussian_blur_with_border(src, sigma, BorderMode::Reflect101)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn binary_threshold_basic() {
        let mut img = GrayImage::new(4, 1);
        img.put_pixel(0, 0, Luma([10]));
        img.put_pixel(1, 0, Luma([50]));
        img.put_pixel(2, 0, Luma([100]));
        img.put_pixel(3, 0, Luma([200]));

        let out = threshold(&img, 100, 255, ThresholdType::Binary);
        assert_eq!(out.as_raw(), &[0, 0, 0, 255]);
    }

    #[test]
    fn otsu_picks_middle_split_on_bimodal_image() {
        let mut img = GrayImage::new(16, 1);
        for x in 0..8 {
            img.put_pixel(x, 0, Luma([30]));
        }
        for x in 8..16 {
            img.put_pixel(x, 0, Luma([220]));
        }

        let (t, out) = threshold_otsu(&img, 255, ThresholdType::Binary);
        assert!(t >= 30 && t < 220);
        assert_eq!(out.as_raw()[0], 0);
        assert_eq!(out.as_raw()[15], 255);
    }

    #[test]
    fn adaptive_mean_handles_uneven_lighting() {
        let mut img = GrayImage::new(9, 9);
        for y in 0..9 {
            for x in 0..9 {
                let base = 30 + x as u8 * 20;
                img.put_pixel(x, y, Luma([base]));
            }
        }
        img.put_pixel(4, 4, Luma([255]));

        let out = adaptive_threshold(
            &img,
            255,
            AdaptiveMethod::MeanC,
            ThresholdType::Binary,
            5,
            5.0,
        );

        assert_eq!(out.get_pixel(4, 4)[0], 255);
    }
}
