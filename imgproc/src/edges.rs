use cv_core::GrayImage;
use rayon::prelude::*;
use std::f32::consts::SQRT_2;

use super::{convolve, gaussian_blur, Kernel};

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

pub fn sobel_direction(gx: &GrayImage, gy: &GrayImage) -> GrayImage {
    let width = gx.width();
    let height = gx.height();
    let mut output = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let gx_val = gx.get_pixel(x, y)[0] as f32;
            let gy_val = gy.get_pixel(x, y)[0] as f32;
            let angle = gy_val.atan2(gx_val);
            let normalized =
                ((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 255.0) as u8;
            output.put_pixel(x, y, image::Luma([normalized]));
        }
    }

    output
}

pub fn scharr(src: &GrayImage) -> (GrayImage, GrayImage) {
    let gx_kernel = Kernel::from_slice(&[-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0], 3, 3);
    let gy_kernel = Kernel::from_slice(&[-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0], 3, 3);

    let gx = convolve(src, &gx_kernel);
    let gy = convolve(src, &gy_kernel);

    (gx, gy)
}

pub fn laplacian(src: &GrayImage) -> GrayImage {
    let kernel = Kernel::from_slice(&[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], 3, 3);
    convolve(src, &kernel)
}

pub fn laplacian_of_gaussian(src: &GrayImage, sigma: f32) -> GrayImage {
    let blurred = gaussian_blur(src, sigma);
    laplacian(&blurred)
}

pub fn canny(src: &GrayImage, low_threshold: u8, high_threshold: u8) -> GrayImage {
    let blurred = gaussian_blur(src, 1.0);

    let (gx, gy) = sobel(&blurred);
    let magnitude = sobel_magnitude(&gx, &gy);
    let direction = sobel_direction(&gx, &gy);

    let mut edges = GrayImage::new(src.width(), src.height());
    let width = src.width() as usize;
    let height = src.height() as usize;

    let low = low_threshold as f32;
    let high = high_threshold as f32;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let mag = magnitude.as_raw()[idx] as f32;
            let dir = direction.as_raw()[idx];

            if mag < low {
                continue;
            }

            let (nx, ny) = get_neighbor_coords(x, y, dir);
            let nidx = ny * width + nx;
            let nmag = magnitude.as_raw()[nidx] as f32;

            if mag > nmag && mag > high {
                edges.as_mut_raw()[idx] = 255;
            }
        }
    }

    let mut output = GrayImage::new(src.width(), src.height());

    for y in 0..height {
        for x in 0..width {
            if edges.get_pixel(x, y)[0] == 255 {
                trace_edges(x as isize, y as isize, &mut output, width, height);
            }
        }
    }

    output
}

fn get_neighbor_coords(x: usize, y: usize, direction: u8) -> (usize, usize) {
    let dir = direction / 45;
    match dir {
        0 => (x + 1, y),
        1 => (x + 1, y + 1),
        2 => (x, y + 1),
        3 => (x - 1, y + 1),
        4 => (x - 1, y),
        5 => (x - 1, y - 1),
        6 => (x, y - 1),
        _ => (x + 1, y - 1),
    }
}

fn trace_edges(x: isize, y: isize, edges: &mut GrayImage, width: usize, height: usize) {
    if x < 0 || x >= width as isize || y < 0 || y >= height as isize {
        return;
    }

    let ux = x as u32;
    let uy = y as u32;

    if edges.get_pixel(ux, uy)[0] == 255 {
        return;
    }

    edges.put_pixel(ux, uy, image::Luma([255]));

    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            trace_edges(x + dx, y + dy, edges, width, height);
        }
    }
}

pub fn prewitt(src: &GrayImage) -> (GrayImage, GrayImage) {
    let gx_kernel = Kernel::from_slice(&[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0], 3, 3);
    let gy_kernel = Kernel::from_slice(&[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 3);

    let gx = convolve(src, &gx_kernel);
    let gy = convolve(src, &gy_kernel);

    (gx, gy)
}

pub fn roberts(src: &GrayImage) -> GrayImage {
    let gx_kernel = Kernel::from_slice(&[1.0, 0.0, 0.0, -1.0], 2, 2);
    let gy_kernel = Kernel::from_slice(&[0.0, 1.0, -1.0, 0.0], 2, 2);

    let gx = convolve(src, &gx_kernel);
    let gy = convolve(src, &gy_kernel);

    sobel_magnitude(&gx, &gy)
}
