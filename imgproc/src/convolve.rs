use cv_core::{GrayImage, RgbImage};
use rayon::prelude::*;

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

pub fn sobel_kernel() -> (Kernel, Kernel) {
    let gx = Kernel::from_slice(&[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], 3, 3);
    let gy = Kernel::from_slice(&[-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0], 3, 3);
    (gx, gy)
}

pub fn laplacian_kernel() -> Kernel {
    Kernel::from_slice(&[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], 3, 3)
}

pub fn convolve(image: &GrayImage, kernel: &Kernel) -> GrayImage {
    let (kx, ky) = kernel.center();
    let kw = kernel.width as isize;
    let kh = kernel.height as isize;

    let mut output = GrayImage::new(image.width(), image.height());
    let width = image.width() as isize;
    let height = image.height() as isize;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0.0f32;

            for ky in -ky..kh - ky {
                for kx in -kx..kw - kx {
                    let px = x + kx;
                    let py = y + ky;

                    if px >= 0 && px < width && py >= 0 && py < height {
                        let ki = ((ky + ky as isize) * kw + (kx + kx as isize)) as usize;
                        let pi = (py as u32 * image.width() + px as u32) as usize;

                        let pixel_val = image.as_raw()[pi] as f32;
                        let kernel_val = kernel.get(ki / kw as usize, ki % kw as usize);
                        sum += pixel_val * kernel_val;
                    }
                }
            }

            output.put_pixel(
                x as u32,
                y as u32,
                image::Luma([sum.clamp(0.0, 255.0) as u8]),
            );
        }
    }

    output
}

pub fn convolve_parallel(image: &GrayImage, kernel: &Kernel) -> GrayImage {
    let (kx_center, ky_center) = kernel.center();
    let kw = kernel.width;
    let kh = kernel.height;

    let mut output = GrayImage::new(image.width(), image.height());
    let width = image.width() as usize;
    let height = image.height() as usize;
    let input_data = image.as_raw();

    output
        .enumerate_rows()
        .par_collect::<Vec<_>>()
        .into_iter()
        .for_each(|(x, y, row)| {
            for (px, _py) in row.enumerate() {
                if x == 0 || x == width - 1 || y == 0 || y == height - 1 {
                    return;
                }

                let mut sum = 0.0f32;

                for ky in 0..kh {
                    for kx in 0..kw {
                        let idx_x = x + kx - kx_center;
                        let idx_y = y + ky - ky_center;

                        if idx_x > 0 && idx_x < width - 1 && idx_y > 0 && idx_y < height - 1 {
                            let ki = ky * kw + kx;
                            let pi = idx_y * width + idx_x;

                            let pixel_val = input_data[pi] as f32;
                            let kernel_val = kernel.get(kx, ky);
                            sum += pixel_val * kernel_val;
                        }
                    }
                }

                let out_idx = y * width + x;
                output.as_mut_raw()[out_idx] = sum.clamp(0.0, 255.0) as u8;
            }
        });

    output
}

pub fn separable_convolve(image: &GrayImage, kernel_x: &Kernel, kernel_y: &Kernel) -> GrayImage {
    let mut temp = GrayImage::new(image.width(), image.height());

    for y in 0..image.height() {
        for x in 0..image.width() {
            let mut sum = 0.0f32;
            for kx in 0..kernel_x.width {
                let px = (x + kx - kernel_x.width / 2).clamp(0, image.width() as usize - 1) as u32;
                let val = image.get_pixel(px, y)[0] as f32;
                sum += val * kernel_x.get(kx, 0);
            }
            temp.put_pixel(x, y, image::Luma([sum.clamp(0.0, 255.0) as u8]));
        }
    }

    let mut output = GrayImage::new(image.width(), image.height());

    for y in 0..image.height() {
        for x in 0..image.width() {
            let mut sum = 0.0f32;
            for ky in 0..kernel_y.height {
                let py =
                    (y + ky - kernel_y.height / 2).clamp(0, image.height() as usize - 1) as u32;
                let val = temp.get_pixel(x, py)[0] as f32;
                sum += val * kernel_y.get(0, ky);
            }
            output.put_pixel(x, y, image::Luma([sum.clamp(0.0, 255.0) as u8]));
        }
    }

    output
}

pub fn gaussian_blur(image: &GrayImage, sigma: f32) -> GrayImage {
    let size = (sigma * 6.0).ceil() as usize | 1;
    let kernel = gaussian_kernel(sigma, size);
    convolve(image, &kernel)
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
