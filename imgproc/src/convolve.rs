use image::{GrayImage, RgbImage};

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
    let (kx_center, ky_center) = kernel.center();
    let kw = kernel.width as isize;
    let kh = kernel.height as isize;

    let mut output = GrayImage::new(image.width(), image.height());
    let width = image.width() as isize;
    let height = image.height() as isize;
    let input_data = image.as_raw();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0.0f32;

            for ky in 0..kernel.height {
                for kx in 0..kernel.width {
                    let idx_x = x + kx as isize - kx_center;
                    let idx_y = y + ky as isize - ky_center;

                    if idx_x > 0 && idx_x < width - 1 && idx_y > 0 && idx_y < height - 1 {
                        let ki = ky * kernel.width + kx;
                        let pi = (idx_y * width + idx_x) as usize;

                        let pixel_val = input_data[pi] as f32;
                        let kernel_val = kernel.get(kx, ky);
                        sum += pixel_val * kernel_val;
                    }
                }
            }

            let out_idx = (y * width + x) as usize;
            output.as_mut()[out_idx] = sum.clamp(0.0, 255.0) as u8;
        }
    }

    output
}

pub fn gaussian_blur(image: &GrayImage, sigma: f32) -> GrayImage {
    let size = ((sigma * 6.0).ceil() as usize) | 1;
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
