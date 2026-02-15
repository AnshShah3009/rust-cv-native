use image::{GrayImage, RgbImage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Nearest,
    Linear,
    Cubic,
    Lanczos,
}

pub fn resize(
    src: &GrayImage,
    width: u32,
    height: u32,
    _interpolation: Interpolation,
) -> GrayImage {
    if width == 0 || height == 0 {
        return GrayImage::new(0, 0);
    }
    resize_linear(src, width, height)
}

#[allow(dead_code)]
fn resize_nearest(src: &GrayImage, width: u32, height: u32) -> GrayImage {
    let mut dst = GrayImage::new(width, height);
    let src_width = src.width() as f32;
    let src_height = src.height() as f32;
    let dst_width = width as f32;
    let dst_height = height as f32;

    for y in 0..height {
        for x in 0..width {
            let sx = ((x as f32 * src_width / dst_width).floor() as u32).min(src.width() - 1);
            let sy = ((y as f32 * src_height / dst_height).floor() as u32).min(src.height() - 1);
            let val = src.get_pixel(sx, sy)[0];
            dst.put_pixel(x, y, image::Luma([val]));
        }
    }
    dst
}

fn resize_linear(src: &GrayImage, width: u32, height: u32) -> GrayImage {
    let mut dst = GrayImage::new(width, height);
    let src_width = src.width() as f32 - 1.0;
    let src_height = src.height() as f32 - 1.0;
    let dst_width = (width - 1) as f32;
    let dst_height = (height - 1) as f32;

    if src_width <= 0.0 || src_height <= 0.0 {
        return dst;
    }

    for y in 0..height {
        for x in 0..width {
            let fx = (x as f32 / dst_width) * src_width;
            let fy = (y as f32 / dst_height) * src_height;

            let x0 = fx as u32;
            let y0 = fy as u32;
            let x1 = (x0 + 1).min(src.width() - 1);
            let y1 = (y0 + 1).min(src.height() - 1);

            let dx = fx - x0 as f32;
            let dy = fy - y0 as f32;

            let v00 = src.get_pixel(x0, y0)[0] as f32;
            let v10 = src.get_pixel(x1, y0)[0] as f32;
            let v01 = src.get_pixel(x0, y1)[0] as f32;
            let v11 = src.get_pixel(x1, y1)[0] as f32;

            let v0 = v00 * (1.0 - dx) + v10 * dx;
            let v1 = v01 * (1.0 - dx) + v11 * dx;
            let v = v0 * (1.0 - dy) + v1 * dy;

            dst.put_pixel(x, y, image::Luma([v.clamp(0.0, 255.0) as u8]));
        }
    }

    dst
}

pub fn resize_rgb(
    src: &RgbImage,
    width: u32,
    height: u32,
    _interpolation: Interpolation,
) -> RgbImage {
    if width == 0 || height == 0 {
        return RgbImage::new(0, 0);
    }

    let mut dst = RgbImage::new(width, height);
    let src_width = src.width() as f32 - 1.0;
    let src_height = src.height() as f32 - 1.0;
    let dst_width = (width - 1) as f32;
    let dst_height = (height - 1) as f32;

    if src_width <= 0.0 || src_height <= 0.0 {
        return dst;
    }

    for y in 0..height {
        for x in 0..width {
            let fx = (x as f32 / dst_width) * src_width;
            let fy = (y as f32 / dst_height) * src_height;

            let x0 = fx as u32;
            let y0 = fy as u32;
            let x1 = (x0 + 1).min(src.width() - 1);
            let y1 = (y0 + 1).min(src.height() - 1);

            let dx = fx - x0 as f32;
            let dy = fy - y0 as f32;

            let mut result = [0u8; 3];

            for c in 0..3 {
                let v00 = src.get_pixel(x0, y0)[c] as f32;
                let v10 = src.get_pixel(x1, y0)[c] as f32;
                let v01 = src.get_pixel(x0, y1)[c] as f32;
                let v11 = src.get_pixel(x1, y1)[c] as f32;

                let v0 = v00 * (1.0 - dx) + v10 * dx;
                let v1 = v01 * (1.0 - dx) + v11 * dx;
                let v = v0 * (1.0 - dy) + v1 * dy;

                result[c] = v.clamp(0.0, 255.0) as u8;
            }

            dst.put_pixel(x, y, image::Rgb(result));
        }
    }

    dst
}

pub fn pyr_down(src: &GrayImage) -> GrayImage {
    let new_width = src.width() / 2;
    let new_height = src.height() / 2;
    if new_width == 0 || new_height == 0 {
        return GrayImage::new(1, 1);
    }
    resize(src, new_width, new_height, Interpolation::Linear)
}

pub fn pyr_up(src: &GrayImage) -> GrayImage {
    let new_width = src.width() * 2;
    let new_height = src.height() * 2;
    resize(src, new_width, new_height, Interpolation::Linear)
}

pub fn build_pyramid(src: &GrayImage, levels: u32) -> Vec<GrayImage> {
    let mut pyramid = vec![src.clone()];

    for _ in 1..levels {
        let prev = pyramid.last().unwrap();
        if prev.width() < 2 || prev.height() < 2 {
            break;
        }
        pyramid.push(pyr_down(prev));
    }

    pyramid
}
