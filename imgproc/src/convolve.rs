use image::GrayImage;

#[derive(Debug, Clone)]
pub struct Kernel {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl Kernel {
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            data,
            width,
            height,
        }
    }

    pub fn from_slice(data: &[f32], width: usize, height: usize) -> Self {
        Self::new(data.to_vec(), width, height)
    }

    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn center(&self) -> (isize, isize) {
        ((self.width / 2) as isize, (self.height / 2) as isize)
    }

    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    pub fn normalize(&mut self) {
        let sum: f32 = self.data.iter().sum();
        if sum != 0.0 {
            for v in &mut self.data {
                *v /= sum;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BorderMode {
    Constant(u8),
    Replicate,
    Reflect,
    Reflect101,
    Wrap,
}

pub fn box_kernel(size: usize) -> Kernel {
    let value = 1.0 / (size * size) as f32;
    Kernel::new(vec![value; size * size], size, size)
}

pub fn gaussian_kernel(sigma: f32, size: usize) -> Kernel {
    let mut data = Vec::with_capacity(size * size);
    let center = size / 2;
    let sigma2 = sigma * sigma;
    let mut sum = 0.0f32;

    for y in 0..size {
        for x in 0..size {
            let dx = (x as isize - center as isize).abs() as f32;
            let dy = (y as isize - center as isize).abs() as f32;
            let v = (-(dx * dx + dy * dy) / (2.0 * sigma2)).exp();
            data.push(v);
            sum += v;
        }
    }

    for v in &mut data {
        *v /= sum;
    }

    Kernel::new(data, size, size)
}

pub fn gaussian_kernel_1d(sigma: f32, size: usize) -> Vec<f32> {
    assert!(size % 2 == 1, "gaussian kernel size must be odd");
    let mut kernel = Vec::with_capacity(size);
    let center = (size / 2) as isize;
    let sigma2 = sigma * sigma;
    let mut sum = 0.0f32;

    for i in 0..size {
        let x = (i as isize - center) as f32;
        let v = (-(x * x) / (2.0 * sigma2)).exp();
        kernel.push(v);
        sum += v;
    }

    if sum != 0.0 {
        for v in &mut kernel {
            *v /= sum;
        }
    }

    kernel
}

pub fn sobel_kernel() -> (Kernel, Kernel) {
    let gx = Kernel::from_slice(&[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], 3, 3);
    let gy = Kernel::from_slice(&[-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0], 3, 3);
    (gx, gy)
}

pub fn laplacian_kernel() -> Kernel {
    Kernel::from_slice(&[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], 3, 3)
}

fn map_coord(coord: isize, len: usize, mode: BorderMode) -> Option<usize> {
    let n = len as isize;
    if n <= 0 {
        return None;
    }

    match mode {
        BorderMode::Constant(_) => {
            if coord < 0 || coord >= n {
                None
            } else {
                Some(coord as usize)
            }
        }
        BorderMode::Replicate => Some(coord.clamp(0, n - 1) as usize),
        BorderMode::Wrap => {
            let mut c = coord % n;
            if c < 0 {
                c += n;
            }
            Some(c as usize)
        }
        BorderMode::Reflect => {
            if n == 1 {
                return Some(0);
            }
            let period = 2 * n;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= n {
                c = period - c - 1;
            }
            Some(c as usize)
        }
        BorderMode::Reflect101 => {
            if n == 1 {
                return Some(0);
            }
            let period = 2 * n - 2;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= n {
                c = period - c;
            }
            Some(c as usize)
        }
    }
}

pub fn convolve(image: &GrayImage, kernel: &Kernel) -> GrayImage {
    convolve_with_border(image, kernel, BorderMode::Replicate)
}

pub fn convolve_with_border(image: &GrayImage, kernel: &Kernel, border: BorderMode) -> GrayImage {
    let mut output = GrayImage::new(image.width(), image.height());
    convolve_with_border_into(image, &mut output, kernel, border);
    output
}

pub fn convolve_with_border_into(
    image: &GrayImage,
    output: &mut GrayImage,
    kernel: &Kernel,
    border: BorderMode,
) {
    let (kx_center, ky_center) = kernel.center();
    if output.width() != image.width() || output.height() != image.height() {
        *output = GrayImage::new(image.width(), image.height());
    }
    let width = image.width() as usize;
    let height = image.height() as usize;
    let input_data = image.as_raw();

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;

            for ky in 0..kernel.height {
                for kx in 0..kernel.width {
                    let src_x = x as isize + kx as isize - kx_center;
                    let src_y = y as isize + ky as isize - ky_center;

                    let pixel_val = match (map_coord(src_x, width, border), map_coord(src_y, height, border)) {
                        (Some(ix), Some(iy)) => input_data[iy * width + ix] as f32,
                        _ => match border {
                            BorderMode::Constant(v) => v as f32,
                            _ => 0.0,
                        },
                    };

                    let kernel_val = kernel.get(kx, ky);
                    sum += pixel_val * kernel_val;
                }
            }

            output.as_mut()[y * width + x] = sum.clamp(0.0, 255.0) as u8;
        }
    }
}

use wide::*;

pub fn separable_convolve(image: &GrayImage, kernel_1d: &[f32], border: BorderMode) -> GrayImage {
    let mut out = GrayImage::new(image.width(), image.height());
    separable_convolve_into(image, &mut out, kernel_1d, border);
    out
}

pub fn separable_convolve_into(
    image: &GrayImage,
    out: &mut GrayImage,
    kernel_1d: &[f32],
    border: BorderMode,
) {
    assert!(kernel_1d.len() % 2 == 1, "1D kernel size must be odd");
    if out.width() != image.width() || out.height() != image.height() {
        *out = GrayImage::new(image.width(), image.height());
    }

    let width = image.width() as usize;
    let height = image.height() as usize;
    let radius = (kernel_1d.len() / 2) as isize;
    let src = image.as_raw();

    // Use BufferPool for temporary storage
    let pool = cv_core::BufferPool::global();
    let mut tmp_vec = pool.get(width * height * 4); // 4 bytes per f32
    let tmp: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(tmp_vec.as_mut_ptr() as *mut f32, width * height)
    };

    // Horizontal Pass (SIMD optimized)
    for y in 0..height {
        let row_offset = y * width;
        for x in (0..width).step_by(8) {
            let mut sum_v = f32x8::ZERO;
            let end = (x + 8).min(width);
            
            if x + 8 <= width {
                for (k, &w) in kernel_1d.iter().enumerate() {
                    let w_v = f32x8::splat(w);
                    let mut px_v = [0.0f32; 8];
                    for i in 0..8 {
                        let sx = (x + i) as isize + k as isize - radius;
                        px_v[i] = match map_coord(sx, width, border) {
                            Some(ix) => src[row_offset + ix] as f32,
                            None => match border {
                                BorderMode::Constant(v) => v as f32,
                                _ => 0.0,
                            },
                        };
                    }
                    sum_v += f32x8::from(px_v) * w_v;
                }
                let results: [f32; 8] = sum_v.into();
                tmp[row_offset + x..row_offset + x + 8].copy_from_slice(&results);
            } else {
                // Scalar fallback for row tail
                for cur_x in x..end {
                    let mut sum = 0.0f32;
                    for (k, &w) in kernel_1d.iter().enumerate() {
                        let sx = cur_x as isize + k as isize - radius;
                        let px = match map_coord(sx, width, border) {
                            Some(ix) => src[row_offset + ix] as f32,
                            None => match border {
                                BorderMode::Constant(v) => v as f32,
                                _ => 0.0,
                            },
                        };
                        sum += px * w;
                    }
                    tmp[row_offset + cur_x] = sum;
                }
            }
        }
    }

    // Vertical Pass (SIMD optimized)
    for y in 0..height {
        let row_offset = y * width;
        for x in (0..width).step_by(8) {
            let mut sum_v = f32x8::ZERO;
            let end = (x + 8).min(width);

            if x + 8 <= width {
                for (k, &w) in kernel_1d.iter().enumerate() {
                    let w_v = f32x8::splat(w);
                    let mut py_v = [0.0f32; 8];
                    let sy_base = y as isize + k as isize - radius;
                    let target_y = map_coord(sy_base, height, border);
                    
                    if let Some(iy) = target_y {
                        let src_offset = iy * width + x;
                        py_v.copy_from_slice(&tmp[src_offset..src_offset + 8]);
                    } else if let BorderMode::Constant(v) = border {
                        py_v = [v as f32; 8];
                    }
                    
                    sum_v += f32x8::from(py_v) * w_v;
                }
                let results: [f32; 8] = sum_v.into();
                let out_row = out.as_mut();
                for i in 0..8 {
                    out_row[row_offset + x + i] = results[i].clamp(0.0, 255.0) as u8;
                }
            } else {
                // Scalar fallback for row tail
                for cur_x in x..end {
                    let mut sum = 0.0f32;
                    for (k, &w) in kernel_1d.iter().enumerate() {
                        let sy = y as isize + k as isize - radius;
                        let py = match map_coord(sy, height, border) {
                            Some(iy) => tmp[iy * width + cur_x],
                            None => match border {
                                BorderMode::Constant(v) => v as f32,
                                _ => 0.0,
                            },
                        };
                        sum += py * w;
                    }
                    out.as_mut()[row_offset + cur_x] = sum.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    pool.return_buffer(tmp_vec);
}

pub fn gaussian_blur_with_border(image: &GrayImage, sigma: f32, border: BorderMode) -> GrayImage {
    let mut out = GrayImage::new(image.width(), image.height());
    gaussian_blur_with_border_into(image, &mut out, sigma, border);
    out
}

pub fn gaussian_blur_with_border_into(
    image: &GrayImage,
    out: &mut GrayImage,
    sigma: f32,
    border: BorderMode,
) {
    let size = ((sigma * 6.0).ceil() as usize) | 1;
    let kernel_1d = gaussian_kernel_1d(sigma, size);
    separable_convolve_into(image, out, &kernel_1d, border);
}

pub fn gaussian_blur(image: &GrayImage, sigma: f32) -> GrayImage {
    gaussian_blur_with_border(image, sigma, BorderMode::Replicate)
}

pub fn box_blur(image: &GrayImage, size: usize) -> GrayImage {
    let kernel = box_kernel(size);
    convolve(image, &kernel)
}

pub fn sharpen(image: &GrayImage, amount: f32) -> GrayImage {
    let mut kernel = laplacian_kernel();
    for v in &mut kernel.data {
        *v *= -amount;
    }
    let center = kernel.width / 2;
    kernel.data[center * kernel.width + center] += 1.0 + amount;
    convolve(image, &kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn gaussian_kernel_1d_is_normalized() {
        let k = gaussian_kernel_1d(1.2, 7);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn gaussian_blur_preserves_size() {
        let mut img = GrayImage::new(32, 24);
        img.put_pixel(10, 10, Luma([255]));

        let out = gaussian_blur_with_border(&img, 1.0, BorderMode::Reflect101);
        assert_eq!(out.width(), img.width());
        assert_eq!(out.height(), img.height());
    }
}
