use cv_core::{GrayImage, KeyPoint, KeyPoints};

pub fn harris_detect(
    image: &GrayImage,
    block_size: i32,
    ksize: i32,
    k: f64,
    threshold: f64,
) -> KeyPoints {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let half_block = block_size / 2;
    let mut keypoints = KeyPoints::new();

    let ix = compute_sobel_x(image);
    let iy = compute_sobel_y(image);

    let mut responses = vec![0.0f64; (width * height) as usize];

    for y in half_block..height - half_block {
        for x in half_block..width - half_block {
            let mut i_xx = 0.0f64;
            let mut i_yy = 0.0f64;
            let mut i_xy = 0.0f64;

            for by in -half_block..=half_block {
                for bx in -half_block..=half_block {
                    let idx = ((y + by) * width + (x + bx)) as usize;

                    let gx = ix.get(idx).copied().unwrap_or(0) as f64;
                    let gy = iy.get(idx).copied().unwrap_or(0) as f64;

                    i_xx += gx * gx;
                    i_yy += gy * gy;
                    i_xy += gx * gy;
                }
            }

            let det = i_xx * i_yy - i_xy * i_xy;
            let trace = i_xx + i_yy;
            let response = det - k * trace * trace;

            responses[(y * width + x) as usize] = response;
        }
    }

    for y in (half_block + 1)..(height - half_block - 1) {
        for x in (half_block + 1)..(width - half_block - 1) {
            let idx = (y * width + x) as usize;
            let response = responses[idx];

            if response > threshold {
                let mut is_max = true;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nidx = ((y + dy) * width + (x + dx)) as usize;
                        if responses.get(nidx).copied().unwrap_or(0.0) > response {
                            is_max = false;
                            break;
                        }
                    }
                    if !is_max {
                        break;
                    }
                }

                if is_max {
                    let kp = KeyPoint::new(x as f64, y as f64).with_response(response);
                    keypoints.push(kp);
                }
            }
        }
    }

    keypoints
}

fn compute_sobel_x(image: &GrayImage) -> Vec<i16> {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let mut result = vec![0i16; (width * height) as usize];

    let kernel = [-1, 0, 1, -2, 0, 2, -1, 0, 1];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0i32;

            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x + kx - 1;
                    let py = y + ky - 1;
                    let val = image.get_pixel(px as u32, py as u32)[0] as i32;
                    sum += val * kernel[ky as usize * 3 + kx as usize];
                }
            }

            result[(y * width + x) as usize] = sum.clamp(-32768, 32767) as i16;
        }
    }

    result
}

fn compute_sobel_y(image: &GrayImage) -> Vec<i16> {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let mut result = vec![0i16; (width * height) as usize];

    let kernel = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0i32;

            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x + kx - 1;
                    let py = y + ky - 1;
                    let val = image.get_pixel(px as u32, py as u32)[0] as i32;
                    sum += val * kernel[ky as usize * 3 + kx as usize];
                }
            }

            result[(y * width + x) as usize] = sum.clamp(-32768, 32767) as i16;
        }
    }

    result
}

pub fn shi_tomasi_detect(
    image: &GrayImage,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
) -> KeyPoints {
    harris_detect(image, 3, 3, 0.04, quality_level * 255.0 * 255.0)
}
