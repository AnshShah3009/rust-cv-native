use cv_core::{GrayImage, KeyPoint, KeyPoints};

pub fn fast_detect(image: &GrayImage, threshold: u8, max_keypoints: usize) -> KeyPoints {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let mut keypoints = Vec::new();

    let circle_offsets = [
        (-3, 0),
        (-2, 1),
        (-1, 2),
        (0, 3),
        (1, 2),
        (2, 1),
        (3, 0),
        (2, -1),
        (1, -2),
        (0, -3),
        (-1, -2),
        (-2, -1),
    ];

    for y in 3..height - 3 {
        for x in 3..width - 3 {
            let p = image.get_pixel(x as u32, y as u32)[0];

            let mut brighter = 0;
            let mut darker = 0;

            for &(dx, dy) in &circle_offsets {
                let px = x + dx;
                let py = y + dy;
                let val = image.get_pixel(px as u32, py as u32)[0];

                if val > p.saturating_add(threshold) {
                    brighter += 1;
                } else if val < p.saturating_sub(threshold) {
                    darker += 1;
                }
            }

            if brighter >= 9 || darker >= 9 {
                let kp = KeyPoint::new(x as f64, y as f64);
                keypoints.push(kp);
            }
        }
    }

    if keypoints.len() > max_keypoints {
        keypoints.truncate(max_keypoints);
    }

    KeyPoints { keypoints }
}

pub fn fast_score(image: &GrayImage, x: i32, y: i32, threshold: u8) -> u32 {
    let p = image.get_pixel(x as u32, y as u32)[0];

    let circle_offsets = [
        (-3, 0),
        (-2, 1),
        (-1, 2),
        (0, 3),
        (1, 2),
        (2, 1),
        (3, 0),
        (2, -1),
        (1, -2),
        (0, -3),
        (-1, -2),
        (-2, -1),
    ];

    let mut brighter = 0u32;
    let mut darker = 0u32;

    for &(dx, dy) in &circle_offsets {
        let px = x + dx;
        let py = y + dy;

        if px >= 0 && px < image.width() as i32 && py >= 0 && py < image.height() as i32 {
            let val = image.get_pixel(px as u32, py as u32)[0];

            if val > p.saturating_add(threshold) {
                brighter += 1;
            } else if val < p.saturating_sub(threshold) {
                darker += 1;
            }
        }
    }

    brighter.max(darker)
}

pub fn non_maximum_suppression(
    keypoints: &mut [KeyPoint],
    distances: &[f32],
    max_keypoints: usize,
) {
    let mut pairs: Vec<_> = keypoints
        .iter()
        .zip(distances.iter())
        .map(|(kp, &d)| (*kp, d))
        .collect();

    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    pairs.truncate(max_keypoints);

    for (i, (kp, _)) in pairs.iter().enumerate() {
        keypoints[i] = *kp;
    }

    keypoints.truncate(pairs.len());
}
