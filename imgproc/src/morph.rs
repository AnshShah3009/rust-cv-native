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

pub fn dilate(src: &GrayImage, kernel: &[(i32, i32)], _iterations: u32) -> GrayImage {
    let width = src.width() as i32;
    let height = src.height() as i32;
    let mut output = GrayImage::new(src.width(), src.height());

    for y in 0..height {
        for x in 0..width {
            let mut max_val = 0u8;
            for &(kx, ky) in kernel {
                let px = x + kx;
                let py = y + ky;
                if px >= 0 && px < width && py >= 0 && py < height {
                    let val = src.get_pixel(px as u32, py as u32)[0];
                    max_val = max_val.max(val);
                }
            }
            output.put_pixel(x as u32, y as u32, image::Luma([max_val]));
        }
    }

    output
}

pub fn erode(src: &GrayImage, kernel: &[(i32, i32)], _iterations: u32) -> GrayImage {
    let width = src.width() as i32;
    let height = src.height() as i32;
    let mut output = GrayImage::new(src.width(), src.height());

    for y in 0..height {
        for x in 0..width {
            let mut min_val = 255u8;
            for &(kx, ky) in kernel {
                let px = x + kx;
                let py = y + ky;
                if px >= 0 && px < width && py >= 0 && py < height {
                    let val = src.get_pixel(px as u32, py as u32)[0];
                    min_val = min_val.min(val);
                }
            }
            output.put_pixel(x as u32, y as u32, image::Luma([min_val]));
        }
    }

    output
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
