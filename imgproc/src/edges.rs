use image::GrayImage;

fn sobel_kernels_1d(ksize: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    match ksize {
        3 => Some((vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0])),
        5 => Some((vec![-1.0, -2.0, 0.0, 2.0, 1.0], vec![1.0, 4.0, 6.0, 4.0, 1.0])),
        7 => Some((
            vec![-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0],
            vec![1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        )),
        _ => None,
    }
}

fn scharr_kernels_1d() -> (Vec<f32>, Vec<f32>) {
    (vec![-1.0, 0.0, 1.0], vec![3.0, 10.0, 3.0])
}

fn kernel_from_1d(kx: &[f32], ky: &[f32]) -> Kernel {
    let mut data = vec![0.0f32; kx.len() * ky.len()];
    for (y, &vy) in ky.iter().enumerate() {
        for (x, &vx) in kx.iter().enumerate() {
            data[y * kx.len() + x] = vx * vy;
        }
    }
    Kernel::new(data, kx.len(), ky.len())
}

fn apply_linear_transform(mut img: GrayImage, scale: f32, delta: f32) -> GrayImage {
    for px in img.as_mut().iter_mut() {
        let v = (*px as f32) * scale + delta;
        *px = v.clamp(0.0, 255.0) as u8;
    }
    img
}

pub fn sobel_ex(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    ksize: usize,
    scale: f32,
    delta: f32,
    border: BorderMode,
) -> GrayImage {
    let (deriv, smooth) = sobel_kernels_1d(ksize).unwrap_or_else(|| {
        // Fallback to 3x3 Sobel for unsupported sizes to preserve behavior.
        (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0])
    });

    let kx = if dx > 0 { deriv.as_slice() } else { smooth.as_slice() };
    let ky = if dy > 0 { deriv.as_slice() } else { smooth.as_slice() };
    let kernel = kernel_from_1d(kx, ky);
    let out = convolve_with_border(src, &kernel, border);
    apply_linear_transform(out, scale, delta)
}

pub fn scharr_ex(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    scale: f32,
    delta: f32,
    border: BorderMode,
) -> GrayImage {
    let (deriv, smooth) = scharr_kernels_1d();
    let kx = if dx > 0 { deriv.as_slice() } else { smooth.as_slice() };
    let ky = if dy > 0 { deriv.as_slice() } else { smooth.as_slice() };
    let kernel = kernel_from_1d(kx, ky);
    let out = convolve_with_border(src, &kernel, border);
    apply_linear_transform(out, scale, delta)
}

pub fn sobel_with_border(src: &GrayImage, border: BorderMode) -> (GrayImage, GrayImage) {
    let gx = sobel_ex(src, 1, 0, 3, 1.0, 0.0, border);
    let gy = sobel_ex(src, 0, 1, 3, 1.0, 0.0, border);
    (gx, gy)
}

pub fn sobel(src: &GrayImage) -> (GrayImage, GrayImage) {
    sobel_with_border(src, BorderMode::Replicate)
}

pub fn scharr_with_border(src: &GrayImage, border: BorderMode) -> (GrayImage, GrayImage) {
    let gx = scharr_ex(src, 1, 0, 1.0, 0.0, border);
    let gy = scharr_ex(src, 0, 1, 1.0, 0.0, border);
    (gx, gy)
}

pub fn scharr(src: &GrayImage) -> (GrayImage, GrayImage) {
    scharr_with_border(src, BorderMode::Replicate)
}

pub fn sobel_magnitude(gx: &GrayImage, gy: &GrayImage) -> GrayImage {
    let width = gx.width();
    let height = gx.height();
    let mut output = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let gx_val = gx.get_pixel(x, y)[0] as f32;
            let gy_val = gy.get_pixel(x, y)[0] as f32;
            let mag = (gx_val * gx_val + gy_val * gy_val).sqrt();
            output.put_pixel(x, y, image::Luma([mag.min(255.0) as u8]));
        }
    }

    output
}

pub fn laplacian(src: &GrayImage) -> GrayImage {
    let kernel = Kernel::from_slice(&[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], 3, 3);
    convolve_with_border(src, &kernel, BorderMode::Replicate)
}

fn gradients_and_directions(src: &GrayImage) -> (Vec<f32>, Vec<u8>) {
    let width = src.width() as usize;
    let height = src.height() as usize;
    let data = src.as_raw();
    let mut magnitude = vec![0.0f32; width * height];
    let mut direction = vec![0u8; width * height];

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let p = |xx: usize, yy: usize| data[yy * width + xx] as f32;

            let gx = -p(x - 1, y - 1) + p(x + 1, y - 1)
                - 2.0 * p(x - 1, y)
                + 2.0 * p(x + 1, y)
                - p(x - 1, y + 1)
                + p(x + 1, y + 1);
            let gy = -p(x - 1, y - 1)
                - 2.0 * p(x, y - 1)
                - p(x + 1, y - 1)
                + p(x - 1, y + 1)
                + 2.0 * p(x, y + 1)
                + p(x + 1, y + 1);

            let idx = y * width + x;
            magnitude[idx] = (gx * gx + gy * gy).sqrt();

            let angle = gy.atan2(gx).to_degrees().rem_euclid(180.0);
            direction[idx] = if !(22.5..157.5).contains(&angle) {
                0
            } else if angle < 67.5 {
                1
            } else if angle < 112.5 {
                2
            } else {
                3
            };
        }
    }

    (magnitude, direction)
}

fn non_max_suppression(width: usize, height: usize, mag: &[f32], dir: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            let m = mag[idx];
            let (m1, m2) = match dir[idx] {
                0 => (mag[idx - 1], mag[idx + 1]),
                1 => (mag[(y - 1) * width + (x + 1)], mag[(y + 1) * width + (x - 1)]),
                2 => (mag[(y - 1) * width + x], mag[(y + 1) * width + x]),
                _ => (mag[(y - 1) * width + (x - 1)], mag[(y + 1) * width + (x + 1)]),
            };

            if m >= m1 && m >= m2 {
                out[idx] = m;
            }
        }
    }

    out
}

fn hysteresis(width: usize, height: usize, nms: &[f32], low: f32, high: f32) -> GrayImage {
    const STRONG: u8 = 255;
    const WEAK: u8 = 75;

    let mut state = vec![0u8; width * height];
    let mut stack = Vec::new();

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            let v = nms[idx];
            if v >= high {
                state[idx] = STRONG;
                stack.push((x, y));
            } else if v >= low {
                state[idx] = WEAK;
            }
        }
    }

    while let Some((x, y)) = stack.pop() {
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(height - 1);
        let x0 = x.saturating_sub(1);
        let x1 = (x + 1).min(width - 1);
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                let nidx = ny * width + nx;
                if state[nidx] == WEAK {
                    state[nidx] = STRONG;
                    stack.push((nx, ny));
                }
            }
        }
    }

    let mut out = GrayImage::new(width as u32, height as u32);
    for (i, px) in out.as_mut().iter_mut().enumerate() {
        *px = if state[i] == STRONG { 255 } else { 0 };
    }
    out
}

pub fn canny(src: &GrayImage, low_threshold: u8, high_threshold: u8) -> GrayImage {
    let blurred = gaussian_blur_with_border(src, 1.0, BorderMode::Reflect101);
    let width = blurred.width() as usize;
    let height = blurred.height() as usize;
    let (mag, dir) = gradients_and_directions(&blurred);
    let nms = non_max_suppression(width, height, &mag, &dir);
    let low = low_threshold as f32;
    let high = high_threshold.max(low_threshold) as f32;
    hysteresis(width, height, &nms, low, high)
}

use crate::convolve::{convolve_with_border, gaussian_blur_with_border, BorderMode, Kernel};

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn sobel_constant_image_is_zero() {
        let mut img = GrayImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                img.put_pixel(x, y, Luma([100]));
            }
        }
        let (gx, gy) = sobel_with_border(&img, BorderMode::Reflect101);
        assert!(gx.as_raw().iter().all(|&v| v == 0));
        assert!(gy.as_raw().iter().all(|&v| v == 0));
    }

    #[test]
    fn scharr_preserves_size() {
        let img = GrayImage::new(32, 20);
        let (gx, gy) = scharr_with_border(&img, BorderMode::Replicate);
        assert_eq!(gx.width(), 32);
        assert_eq!(gx.height(), 20);
        assert_eq!(gy.width(), 32);
        assert_eq!(gy.height(), 20);
    }

    #[test]
    fn sobel_ex_ksize_5_preserves_size() {
        let img = GrayImage::new(21, 13);
        let out = sobel_ex(&img, 1, 0, 5, 1.0, 0.0, BorderMode::Reflect101);
        assert_eq!(out.width(), 21);
        assert_eq!(out.height(), 13);
    }

    #[test]
    fn canny_detects_step_edge() {
        let mut img = GrayImage::new(64, 32);
        for y in 0..32 {
            for x in 0..64 {
                let v = if x < 32 { 20 } else { 220 };
                img.put_pixel(x, y, Luma([v]));
            }
        }

        let edges = canny(&img, 40, 100);
        let count = edges.as_raw().iter().filter(|&&v| v > 0).count();
        assert!(count > 0, "canny should detect the vertical intensity step");
    }
}
