use image::GrayImage;

pub fn sobel(src: &GrayImage) -> (GrayImage, GrayImage) {
    let gx_kernel = Kernel::from_slice(&[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], 3, 3);
    let gy_kernel = Kernel::from_slice(&[-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0], 3, 3);

    let gx = convolve(src, &gx_kernel);
    let gy = convolve(src, &gy_kernel);

    (gx, gy)
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
    convolve(src, &kernel)
}

pub fn canny(src: &GrayImage, _low_threshold: u8, _high_threshold: u8) -> GrayImage {
    let blurred = gaussian_blur(src, 1.0);
    let (gx, gy) = sobel(&blurred);
    sobel_magnitude(&gx, &gy)
}

use crate::convolve::{convolve, gaussian_blur, Kernel};
