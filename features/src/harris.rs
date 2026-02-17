use crate::KeyPoints;
use cv_core::KeyPoint;
use image::GrayImage;
use rayon::prelude::*;

pub fn harris_detect(
    image: &GrayImage,
    _block_size: i32,
    _ksize: i32,
    k: f64,
    threshold: f64,
) -> KeyPoints {
    let width = image.width() as i32;
    let height = image.height() as i32;
    
    let half_window = 1i32;

    let kps: Vec<KeyPoint> = (half_window + 1..height - half_window - 1)
        .into_par_iter()
        .flat_map(|y| {
            let mut row_kps = Vec::new();
            for x in (half_window + 1)..(width - half_window - 1) {
                let mut i_xx = 0.0f64;
                let mut i_yy = 0.0f64;
                let mut i_xy = 0.0f64;

                for by in -half_window..=half_window {
                    for bx in -half_window..=half_window {
                        let gx = image.get_pixel((x + bx + 1) as u32, (y + by) as u32)[0] as f64
                            - image.get_pixel((x + bx - 1) as u32, (y + by) as u32)[0] as f64;
                        let gy = image.get_pixel((x + bx) as u32, (y + by + 1) as u32)[0] as f64
                            - image.get_pixel((x + bx) as u32, (y + by - 1) as u32)[0] as f64;

                        i_xx += gx * gx;
                        i_yy += gy * gy;
                        i_xy += gx * gy;
                    }
                }

                let det = i_xx * i_yy - i_xy * i_xy;
                let trace = i_xx + i_yy;
                let response = det - k * trace * trace;

                if response > threshold {
                    let kp = KeyPoint::new(x as f64, y as f64).with_response(response);
                    row_kps.push(kp);
                }
            }
            row_kps
        })
        .collect();

    KeyPoints { keypoints: kps }
}

pub fn shi_tomasi_detect(
    image: &GrayImage,
    _max_corners: usize,
    quality_level: f64,
    _min_distance: f64,
) -> KeyPoints {
    harris_detect(image, 3, 3, 0.04, quality_level * 255.0 * 255.0)
}
