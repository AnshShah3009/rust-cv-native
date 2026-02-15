use crate::convolve::BorderMode;
use image::GrayImage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphShape {
    Rectangle,
    Ellipse,
    Cross,
}

pub fn create_morph_kernel(shape: MorphShape, width: u32, height: u32) -> Vec<(i32, i32)> {
    let mut kernel = Vec::new();
    let cx = width as i32 / 2;
    let cy = height as i32 / 2;

    match shape {
        MorphShape::Rectangle => {
            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    kernel.push((x - cx, y - cy));
                }
            }
        }
        MorphShape::Ellipse => {
            let rx = width as f32 / 2.0;
            let ry = height as f32 / 2.0;

            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    let dx = (x - cx) as f32;
                    let dy = (y - cy) as f32;
                    if (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1.0 {
                        kernel.push((x - cx, y - cy));
                    }
                }
            }
        }
        MorphShape::Cross => {
            for i in -(width as i32 / 2)..=(width as i32 / 2) {
                kernel.push((i, 0));
            }
            for i in -(height as i32 / 2)..=(height as i32 / 2) {
                kernel.push((0, i));
            }
        }
    }

    kernel
}

fn map_coord(coord: i32, len: i32, border: BorderMode) -> Option<i32> {
    if len <= 0 {
        return None;
    }

    match border {
        BorderMode::Constant(_) => {
            if coord < 0 || coord >= len {
                None
            } else {
                Some(coord)
            }
        }
        BorderMode::Replicate => Some(coord.clamp(0, len - 1)),
        BorderMode::Wrap => {
            let mut c = coord % len;
            if c < 0 {
                c += len;
            }
            Some(c)
        }
        BorderMode::Reflect => {
            if len == 1 {
                return Some(0);
            }
            let period = 2 * len;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= len {
                c = period - c - 1;
            }
            Some(c)
        }
        BorderMode::Reflect101 => {
            if len == 1 {
                return Some(0);
            }
            let period = 2 * len - 2;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= len {
                c = period - c;
            }
            Some(c)
        }
    }
}

#[allow(dead_code)]
fn dilate_once(src: &GrayImage, kernel: &[(i32, i32)], border: BorderMode) -> GrayImage {
    let mut output = GrayImage::new(src.width(), src.height());
    dilate_once_into(src, &mut output, kernel, border);
    output
}

fn dilate_once_into(
    src: &GrayImage,
    output: &mut GrayImage,
    kernel: &[(i32, i32)],
    border: BorderMode,
) {
    let width = src.width() as i32;
    let height = src.height() as i32;
    let src_data = src.as_raw();
    if output.width() != src.width() || output.height() != src.height() {
        *output = GrayImage::new(src.width(), src.height());
    }

    for y in 0..height {
        for x in 0..width {
            let mut max_val = 0u8;
            for &(kx, ky) in kernel {
                let px = x + kx;
                let py = y + ky;

                let val = match (map_coord(px, width, border), map_coord(py, height, border)) {
                    (Some(ix), Some(iy)) => src_data[(iy * width + ix) as usize],
                    _ => match border {
                        BorderMode::Constant(v) => v,
                        _ => 0,
                    },
                };
                max_val = max_val.max(val);
            }
            output.as_mut()[(y * width + x) as usize] = max_val;
        }
    }
}

#[allow(dead_code)]
fn erode_once(src: &GrayImage, kernel: &[(i32, i32)], border: BorderMode) -> GrayImage {
    let mut output = GrayImage::new(src.width(), src.height());
    erode_once_into(src, &mut output, kernel, border);
    output
}

fn erode_once_into(
    src: &GrayImage,
    output: &mut GrayImage,
    kernel: &[(i32, i32)],
    border: BorderMode,
) {
    let width = src.width() as i32;
    let height = src.height() as i32;
    let src_data = src.as_raw();
    if output.width() != src.width() || output.height() != src.height() {
        *output = GrayImage::new(src.width(), src.height());
    }

    for y in 0..height {
        for x in 0..width {
            let mut min_val = 255u8;
            for &(kx, ky) in kernel {
                let px = x + kx;
                let py = y + ky;

                let val = match (map_coord(px, width, border), map_coord(py, height, border)) {
                    (Some(ix), Some(iy)) => src_data[(iy * width + ix) as usize],
                    _ => match border {
                        BorderMode::Constant(v) => v,
                        _ => 255,
                    },
                };
                min_val = min_val.min(val);
            }
            output.as_mut()[(y * width + x) as usize] = min_val;
        }
    }
}

pub fn dilate_with_border(
    src: &GrayImage,
    kernel: &[(i32, i32)],
    iterations: u32,
    border: BorderMode,
) -> GrayImage {
    let mut out = GrayImage::new(src.width(), src.height());
    dilate_with_border_into(src, &mut out, kernel, iterations, border);
    out
}

pub fn dilate_with_border_into(
    src: &GrayImage,
    dst: &mut GrayImage,
    kernel: &[(i32, i32)],
    iterations: u32,
    border: BorderMode,
) {
    if iterations == 0 {
        if dst.width() != src.width() || dst.height() != src.height() {
            *dst = src.clone();
        } else {
            dst.as_mut().copy_from_slice(src.as_raw());
        }
        return;
    }
    let mut cur = src.clone();
    let mut tmp = GrayImage::new(src.width(), src.height());
    for _ in 0..iterations {
        dilate_once_into(&cur, &mut tmp, kernel, border);
        std::mem::swap(&mut cur, &mut tmp);
    }
    if dst.width() != cur.width() || dst.height() != cur.height() {
        *dst = cur;
    } else {
        dst.as_mut().copy_from_slice(cur.as_raw());
    }
}

pub fn erode_with_border(
    src: &GrayImage,
    kernel: &[(i32, i32)],
    iterations: u32,
    border: BorderMode,
) -> GrayImage {
    let mut out = GrayImage::new(src.width(), src.height());
    erode_with_border_into(src, &mut out, kernel, iterations, border);
    out
}

pub fn erode_with_border_into(
    src: &GrayImage,
    dst: &mut GrayImage,
    kernel: &[(i32, i32)],
    iterations: u32,
    border: BorderMode,
) {
    if iterations == 0 {
        if dst.width() != src.width() || dst.height() != src.height() {
            *dst = src.clone();
        } else {
            dst.as_mut().copy_from_slice(src.as_raw());
        }
        return;
    }
    let mut cur = src.clone();
    let mut tmp = GrayImage::new(src.width(), src.height());
    for _ in 0..iterations {
        erode_once_into(&cur, &mut tmp, kernel, border);
        std::mem::swap(&mut cur, &mut tmp);
    }
    if dst.width() != cur.width() || dst.height() != cur.height() {
        *dst = cur;
    } else {
        dst.as_mut().copy_from_slice(cur.as_raw());
    }
}

pub fn dilate(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    dilate_with_border(src, kernel, iterations, BorderMode::Replicate)
}

pub fn erode(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    erode_with_border(src, kernel, iterations, BorderMode::Replicate)
}

pub fn opening(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    let eroded = erode(src, kernel, iterations);
    dilate(&eroded, kernel, iterations)
}

pub fn closing(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    let dilated = dilate(src, kernel, iterations);
    erode(&dilated, kernel, iterations)
}

pub fn top_hat(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    let opened = opening(src, kernel, iterations);
    let mut out = GrayImage::new(src.width(), src.height());
    for (dst, (s, o)) in out
        .as_mut()
        .iter_mut()
        .zip(src.as_raw().iter().zip(opened.as_raw().iter()))
    {
        *dst = s.saturating_sub(*o);
    }
    out
}

pub fn black_hat(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    let closed = closing(src, kernel, iterations);
    let mut out = GrayImage::new(src.width(), src.height());
    for (dst, (c, s)) in out
        .as_mut()
        .iter_mut()
        .zip(closed.as_raw().iter().zip(src.as_raw().iter()))
    {
        *dst = c.saturating_sub(*s);
    }
    out
}

pub fn morphological_gradient(src: &GrayImage, kernel_size: u32) -> GrayImage {
    let kernel = create_morph_kernel(MorphShape::Ellipse, kernel_size, kernel_size);
    let dilated = dilate(src, &kernel, 1);
    let eroded = erode(src, &kernel, 1);

    let mut output = GrayImage::new(src.width(), src.height());
    for y in 0..src.height() {
        for x in 0..src.width() {
            let d = dilated.get_pixel(x, y)[0];
            let e = eroded.get_pixel(x, y)[0];
            output.put_pixel(x, y, image::Luma([d.saturating_sub(e)]));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn iterations_zero_is_noop() {
        let mut img = GrayImage::new(8, 8);
        img.put_pixel(4, 4, Luma([255]));
        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        let out = dilate(&img, &k, 0);
        assert_eq!(out.as_raw(), img.as_raw());
    }

    #[test]
    fn opening_closing_preserve_size() {
        let img = GrayImage::new(23, 11);
        let k = create_morph_kernel(MorphShape::Ellipse, 5, 5);
        let o = opening(&img, &k, 1);
        let c = closing(&img, &k, 1);
        assert_eq!(o.width(), img.width());
        assert_eq!(o.height(), img.height());
        assert_eq!(c.width(), img.width());
        assert_eq!(c.height(), img.height());
    }
}
