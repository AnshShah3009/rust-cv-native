use image::GrayImage;
use cv_core::{KeyPoint, KeyPoints};

pub fn gftt_detect(
    image: &GrayImage,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
) -> KeyPoints {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let half_window = 1i32;

    let mut scores = Vec::new();

    let mut max_score = 0.0f64;

    for y in (half_window + 1)..(height - half_window - 1) {
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

            // Min eigenvalue calculation
            let trace = i_xx + i_yy;
            let _det = i_xx * i_yy - i_xy * i_xy;
            let term = ((i_xx - i_yy).powi(2) + 4.0 * i_xy * i_xy).sqrt();
            let lambda_min = (trace - term) * 0.5;

            if lambda_min > 0.0 {
                if lambda_min > max_score {
                    max_score = lambda_min;
                }
                scores.push((x, y, lambda_min));
            }
        }
    }

    // Filter by quality level
    let threshold = max_score * quality_level;
    let mut candidates: Vec<_> = scores.into_iter()
        .filter(|&(_, _, s)| s >= threshold)
        .collect();

    // Sort by score descending
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    let mut corners = KeyPoints::new();
    let min_dist_sq = min_distance * min_distance;

    for (x, y, score) in candidates {
        if corners.len() >= max_corners { break; }

        let mut too_close = false;
        for kp in corners.keypoints.iter() {
            let dx = (x as f64) - kp.x;
            let dy = (y as f64) - kp.y;
            if dx * dx + dy * dy < min_dist_sq {
                too_close = true;
                break;
            }
        }

        if !too_close {
            corners.push(KeyPoint::new(x as f64, y as f64).with_response(score));
        }
    }

    corners
}
