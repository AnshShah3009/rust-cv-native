use cv_core::GrayImage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphType {
    Erode,
    Dilate,
    Open,
    Close,
    Gradient,
    TopHat,
    BlackHat,
}

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

pub fn dilate(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    let mut output = src.clone();

    for _ in 0..iterations {
        output = dilate_once(src, &output, kernel);
    }

    output
}

fn dilate_once(src: &GrayImage, current: &GrayImage, kernel: &[(i32, i32)]) -> GrayImage {
    let mut output = GrayImage::new(src.width(), src.height());
    let width = src.width() as i32;
    let height = src.height() as i32;

    for y in 0..height {
        for x in 0..width {
            let mut max_val = 0u8;

            for &(kx, ky) in kernel {
                let px = x + kx;
                let py = y + ky;

                if px >= 0 && px < width && py >= 0 && py < height {
                    let val = current.get_pixel(px as u32, py as u32)[0];
                    max_val = max_val.max(val);
                }
            }

            output.put_pixel(x as u32, y as u32, image::Luma([max_val]));
        }
    }

    output
}

pub fn erode(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    let mut output = src.clone();

    for _ in 0..iterations {
        output = erode_once(src, &output, kernel);
    }

    output
}

fn erode_once(src: &GrayImage, current: &GrayImage, kernel: &[(i32, i32)]) -> GrayImage {
    let mut output = GrayImage::new(src.width(), src.height());
    let width = src.width() as i32;
    let height = src.height() as i32;

    for y in 0..height {
        for x in 0..width {
            let mut min_val = 255u8;

            for &(kx, ky) in kernel {
                let px = x + kx;
                let py = y + ky;

                if px >= 0 && px < width && py >= 0 && py < height {
                    let val = current.get_pixel(px as u32, py as u32)[0];
                    min_val = min_val.min(val);
                }
            }

            output.put_pixel(x as u32, y as u32, image::Luma([min_val]));
        }
    }

    output
}

pub fn morph(
    src: &GrayImage,
    morph_type: MorphType,
    kernel: &[(i32, i32)],
    iterations: u32,
) -> GrayImage {
    match morph_type {
        MorphType::Erode => erode(src, kernel, iterations),
        MorphType::Dilate => dilate(src, kernel, iterations),
        MorphType::Open => {
            let eroded = erode(src, kernel, iterations);
            dilate(&eroded, kernel, iterations)
        }
        MorphType::Close => {
            let dilated = dilate(src, kernel, iterations);
            erode(&dilated, kernel, iterations)
        }
        MorphType::Gradient => {
            let dilated = dilate(src, kernel, iterations);
            let eroded = erode(src, kernel, iterations);
            subtract(&dilated, &eroded)
        }
        MorphType::TopHat => {
            let opened = {
                let eroded = erode(src, kernel, iterations);
                dilate(&eroded, kernel, iterations)
            };
            subtract(src, &opened)
        }
        MorphType::BlackHat => {
            let closed = {
                let dilated = dilate(src, kernel, iterations);
                erode(&dilated, kernel, iterations)
            };
            subtract(&closed, src)
        }
    }
}

fn subtract(a: &GrayImage, b: &GrayImage) -> GrayImage {
    let mut output = GrayImage::new(a.width(), a.height());

    for (i, pixel) in output.pixels_mut().enumerate() {
        let va = a.as_raw()[i];
        let vb = b.as_raw()[i];
        pixel[0] = va.saturating_sub(vb);
    }

    output
}

pub fn morphological_gradient(src: &GrayImage, kernel_size: u32) -> GrayImage {
    let kernel = create_morph_kernel(MorphShape::Ellipse, kernel_size, kernel_size);
    morph(src, MorphType::Gradient, &kernel, 1)
}

pub fn tophat(src: &GrayImage, kernel_size: u32) -> GrayImage {
    let kernel = create_morph_kernel(MorphShape::Rectangle, kernel_size, kernel_size);
    morph(src, MorphType::TopHat, &kernel, 1)
}

pub fn blackhat(src: &GrayImage, kernel_size: u32) -> GrayImage {
    let kernel = create_morph_kernel(MorphShape::Rectangle, kernel_size, kernel_size);
    morph(src, MorphType::BlackHat, &kernel, 1)
}
