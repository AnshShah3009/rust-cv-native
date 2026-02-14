use cv_core::KeyPoint;
use image::GrayImage;

pub struct KeyPoints {
    pub keypoints: Vec<KeyPoint>,
}

impl KeyPoints {
    pub fn new() -> Self {
        Self {
            keypoints: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            keypoints: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, kp: KeyPoint) {
        self.keypoints.push(kp);
    }

    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
}

pub fn fast_detect(image: &GrayImage, threshold: u8, max_keypoints: usize) -> KeyPoints {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let mut keypoints = Vec::new();

    let circle_offsets: [(i32, i32); 12] = [
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

            let mut brighter = 0u32;
            let mut darker = 0u32;

            for &(dx, dy) in &circle_offsets {
                let px = x + dx;
                let py = y + dy;

                if px >= 0 && px < width && py >= 0 && py < height {
                    let val = image.get_pixel(px as u32, py as u32)[0];

                    if val > p.saturating_add(threshold) {
                        brighter += 1;
                    } else if val < p.saturating_sub(threshold) {
                        darker += 1;
                    }
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
