use image::GrayImage;
use rayon::prelude::*;
use cv_core::{Tensor, storage::Storage};
use cv_hal::compute::{ComputeDevice};
use cv_hal::tensor_ext::{TensorToGpu, TensorToCpu};
use cv_hal::context::ComputeContext;
use cv_runtime::orchestrator::{ResourceGroup, scheduler};

use crate::convolve::{convolve_with_border_into_in_pool, gaussian_blur_with_border_in_pool, BorderMode, Kernel, Pool};

fn sobel_kernels_1d(ksize: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    match ksize {
        3 => Some((vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0])),
        5 => Some((
            vec![-1.0, -2.0, 0.0, 2.0, 1.0],
            vec![1.0, 4.0, 6.0, 4.0, 1.0],
        )),
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
    img.as_mut().par_iter_mut().for_each(|px| {
        let v = (*px as f32) * scale + delta;
        *px = v.clamp(0.0, 255.0) as u8;
    });
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
    let group = scheduler().get_group("default").unwrap().unwrap();
    sobel_ex_ctx(src, dx, dy, ksize, scale, delta, border, &group)
}

pub fn sobel_ex_ctx(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    ksize: usize,
    scale: f32,
    delta: f32,
    border: BorderMode,
    group: &ResourceGroup,
) -> GrayImage {
    let device = group.device();
    
    if let ComputeDevice::Gpu(gpu) = device {
        if dx == 1 && dy == 1 && ksize == 3 {
            if let Ok(result) = sobel_gpu(gpu, src, ksize) {
                let (gx, gy) = result;
                let target = if dx > 0 { gx } else { gy };
                return apply_linear_transform(target, scale, delta);
            }
        }
    }
    
    sobel_ex_in_pool(src, dx, dy, ksize, scale, delta, border, Some(Pool::Group(group)))
}

fn sobel_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    ksize: usize,
) -> cv_hal::Result<(GrayImage, GrayImage)> {
    let input_tensor = Tensor::from_vec(src.as_raw().to_vec(), cv_core::TensorShape::new(1, src.height() as usize, src.width() as usize));
    let input_gpu = input_tensor.to_gpu()?;
    
    let (gx_gpu, gy_gpu) = gpu.sobel(&input_gpu, 1, 1, ksize)?;
    
    let gx_cpu: Tensor<u8, cv_core::CpuStorage<u8>> = gx_gpu.to_cpu()?;
    let gy_cpu: Tensor<u8, cv_core::CpuStorage<u8>> = gy_gpu.to_cpu()?;
    
    let gx_data = gx_cpu.storage.as_slice().unwrap().to_vec();
    let gy_data = gy_cpu.storage.as_slice().unwrap().to_vec();
    
    let gx = GrayImage::from_raw(src.width(), src.height(), gx_data).unwrap();
    let gy = GrayImage::from_raw(src.width(), src.height(), gy_data).unwrap();
    
    Ok((gx, gy))
}

pub fn sobel_ex_in_pool<'a>(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    ksize: usize,
    scale: f32,
    delta: f32,
    border: BorderMode,
    pool: Option<Pool<'a>>,
) -> GrayImage {
    let run = || {
        let (deriv, smooth) = sobel_kernels_1d(ksize).unwrap_or_else(|| {
            (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0])
        });

        let kx = if dx > 0 { deriv.as_slice() } else { smooth.as_slice() };
        let ky = if dy > 0 { deriv.as_slice() } else { smooth.as_slice() };
        let kernel = kernel_from_1d(kx, ky);
        
        let mut out = GrayImage::new(src.width(), src.height());
        convolve_with_border_into_in_pool(src, &mut out, &kernel, border, None as Option<Pool>);
        apply_linear_transform(out, scale, delta)
    };

    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

pub fn scharr_ex(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    scale: f32,
    delta: f32,
    border: BorderMode,
) -> GrayImage {
    let group = scheduler().get_group("default").unwrap().unwrap();
    scharr_ex_ctx(src, dx, dy, scale, delta, border, &group)
}

pub fn scharr_ex_ctx(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    scale: f32,
    delta: f32,
    border: BorderMode,
    group: &ResourceGroup,
) -> GrayImage {
    scharr_ex_in_pool(src, dx, dy, scale, delta, border, Some(Pool::Group(group)))
}

pub fn scharr_ex_in_pool<'a>(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    scale: f32,
    delta: f32,
    border: BorderMode,
    pool: Option<Pool<'a>>,
) -> GrayImage {
    let run = || {
        let (deriv, smooth) = scharr_kernels_1d();
        let kx = if dx > 0 { deriv.as_slice() } else { smooth.as_slice() };
        let ky = if dy > 0 { deriv.as_slice() } else { smooth.as_slice() };
        let kernel = kernel_from_1d(kx, ky);
        
        let mut out = GrayImage::new(src.width(), src.height());
        convolve_with_border_into_in_pool(src, &mut out, &kernel, border, None as Option<Pool>);
        apply_linear_transform(out, scale, delta)
    };

    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

pub fn sobel_with_border(src: &GrayImage, border: BorderMode) -> (GrayImage, GrayImage) {
    let group = scheduler().get_group("default").unwrap().unwrap();
    let gx = sobel_ex_ctx(src, 1, 0, 3, 1.0, 0.0, border, &group);
    let gy = sobel_ex_ctx(src, 0, 1, 3, 1.0, 0.0, border, &group);
    (gx, gy)
}

pub fn sobel_with_border_in_pool<'a>(src: &GrayImage, border: BorderMode, pool: Option<Pool<'a>>) -> (GrayImage, GrayImage) {
    let run = || {
        let gx = sobel_ex_in_pool(src, 1, 0, 3, 1.0, 0.0, border, None as Option<Pool>);
        let gy = sobel_ex_in_pool(src, 0, 1, 3, 1.0, 0.0, border, None as Option<Pool>);
        (gx, gy)
    };
    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

pub fn sobel(src: &GrayImage) -> (GrayImage, GrayImage) {
    sobel_with_border(src, BorderMode::Replicate)
}

pub fn scharr_with_border(src: &GrayImage, border: BorderMode) -> (GrayImage, GrayImage) {
    let group = scheduler().get_group("default").unwrap().unwrap();
    let gx = scharr_ex_ctx(src, 1, 0, 1.0, 0.0, border, &group);
    let gy = scharr_ex_ctx(src, 0, 1, 1.0, 0.0, border, &group);
    (gx, gy)
}

pub fn scharr_with_border_in_pool<'a>(src: &GrayImage, border: BorderMode, pool: Option<Pool<'a>>) -> (GrayImage, GrayImage) {
    let run = || {
        let gx = scharr_ex_in_pool(src, 1, 0, 1.0, 0.0, border, None as Option<Pool>);
        let gy = scharr_ex_in_pool(src, 0, 1, 1.0, 0.0, border, None as Option<Pool>);
        (gx, gy)
    };
    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

pub fn scharr(src: &GrayImage) -> (GrayImage, GrayImage) {
    scharr_with_border(src, BorderMode::Replicate)
}

pub fn sobel_magnitude(gx: &GrayImage, gy: &GrayImage) -> GrayImage {
    sobel_magnitude_in_pool(gx, gy, None as Option<Pool>)
}

pub fn sobel_magnitude_in_pool<'a>(gx: &GrayImage, gy: &GrayImage, pool: Option<Pool<'a>>) -> GrayImage {
    let run = || {
        let width = gx.width();
        let height = gx.height();
        let count = (width * height) as usize;
        let mut output = vec![0u8; count];

        output
            .par_iter_mut()
            .zip(gx.as_raw().par_iter())
            .zip(gy.as_raw().par_iter())
            .for_each(|((out, &gx_val), &gy_val)| {
                let gx_f = gx_val as f32;
                let gy_f = gy_val as f32;
                let mag = (gx_f * gx_f + gy_f * gy_f).sqrt();
                *out = mag.min(255.0) as u8;
            });

        GrayImage::from_raw(width, height, output).unwrap()
    };
    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

pub fn laplacian(src: &GrayImage) -> GrayImage {
    laplacian_in_pool(src, None as Option<Pool>)
}

pub fn laplacian_in_pool<'a>(src: &GrayImage, pool: Option<Pool<'a>>) -> GrayImage {
    let run = || {
        let kernel = Kernel::from_slice(&[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], 3, 3);
        let mut out = GrayImage::new(src.width(), src.height());
        convolve_with_border_into_in_pool(src, &mut out, &kernel, BorderMode::Replicate, None as Option<Pool>);
        out
    };
    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

fn gradients_and_directions(src: &GrayImage) -> (Vec<f32>, Vec<u8>) {
    let width = src.width() as usize;
    let height = src.height() as usize;
    let data = src.as_raw();
    let mut magnitude = vec![0.0f32; width * height];
    let mut direction = vec![0u8; width * height];

    magnitude
        .par_chunks_mut(width)
        .zip(direction.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (mag_row, dir_row))| {
            if y == 0 || y >= height - 1 {
                return;
            }
            let r0_idx = (y - 1) * width;
            let r1_idx = y * width;
            let r2_idx = (y + 1) * width;
            let r0 = &data[r0_idx..r0_idx + width];
            let r1 = &data[r1_idx..r1_idx + width];
            let r2 = &data[r2_idx..r2_idx + width];

            for x in 1..width - 1 {
                let p00 = r0[x - 1] as f32;
                let p02 = r0[x + 1] as f32;
                let p10 = r1[x - 1] as f32;
                let p12 = r1[x + 1] as f32;
                let p20 = r2[x - 1] as f32;
                let p22 = r2[x + 1] as f32;
                let p01 = r0[x] as f32;
                let p21 = r2[x] as f32;

                let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
                let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

                mag_row[x] = (gx * gx + gy * gy).sqrt();

                let angle = gy.atan2(gx).to_degrees().rem_euclid(180.0);
                dir_row[x] = if !(22.5..157.5).contains(&angle) {
                    0
                } else if angle < 67.5 {
                    1
                } else if angle < 112.5 {
                    2
                } else {
                    3
                };
            }
        });

    (magnitude, direction)
}

fn non_max_suppression(width: usize, height: usize, mag: &[f32], dir: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];

    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out_row)| {
            if y == 0 || y >= height - 1 {
                return;
            }
            let r0_idx = (y - 1) * width;
            let r1_idx = y * width;
            let r2_idx = (y + 1) * width;

            for x in 1..width - 1 {
                let m = mag[r1_idx + x];
                let (m1, m2) = match dir[r1_idx + x] {
                    0 => (mag[r1_idx + x - 1], mag[r1_idx + x + 1]),
                    1 => (mag[r0_idx + x + 1], mag[r2_idx + x - 1]),
                    2 => (mag[r0_idx + x], mag[r2_idx + x]),
                    _ => (mag[r0_idx + x - 1], mag[r2_idx + x + 1]),
                };

                if m >= m1 && m >= m2 {
                    out_row[x] = m;
                }
            }
        });

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
    out.as_mut().par_iter_mut().enumerate().for_each(|(i, px)| {
        *px = if state[i] == STRONG { 255 } else { 0 };
    });
    out
}

pub fn canny(src: &GrayImage, low_threshold: u8, high_threshold: u8) -> GrayImage {
    let group = scheduler().get_group("default").unwrap().unwrap();
    canny_ctx(src, low_threshold, high_threshold, &group)
}

pub fn canny_ctx(src: &GrayImage, low_threshold: u8, high_threshold: u8, group: &ResourceGroup) -> GrayImage {
    canny_in_pool(src, low_threshold, high_threshold, Some(Pool::Group(group)))
}

pub fn canny_in_pool<'a>(
    src: &GrayImage, 
    low_threshold: u8, 
    high_threshold: u8, 
    pool: Option<Pool<'a>>
) -> GrayImage {
    let run = || {
        let blurred = gaussian_blur_with_border_in_pool(src, 1.0, BorderMode::Reflect101, None as Option<Pool>);
        let width = blurred.width() as usize;
        let height = blurred.height() as usize;
        let (mag, dir) = gradients_and_directions(&blurred);
        let nms = non_max_suppression(width, height, &mag, &dir);
        let low = low_threshold as f32;
        let high = high_threshold.max(low_threshold) as f32;
        hysteresis(width, height, &nms, low, high)
    };

    if let Some(p) = pool {
        p.install(run)
    } else {
        run()
    }
}

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
}
