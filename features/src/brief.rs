use cv_core::{GrayImage, KeyPoint};

pub struct Brief {
    bytes: usize,
    pattern: Vec<[(i32, i32); 2]>,
}

impl Brief {
    pub fn new(bytes: usize) -> Self {
        let mut brief = Self {
            bytes,
            pattern: Vec::with_capacity(bytes * 8),
        };

        let patch_size = 48;

        for _ in 0..(bytes * 8) {
            let p1 = (
                (rand_i32() % patch_size) - patch_size / 2,
                (rand_i32() % patch_size) - patch_size / 2,
            );
            let p2 = (
                (rand_i32() % patch_size) - patch_size / 2,
                (rand_i32() % patch_size) - patch_size / 2,
            );
            brief.pattern.push([p1, p2]);
        }

        brief
    }

    pub fn compute(&self, image: &GrayImage, keypoints: &[KeyPoint]) -> Vec<Vec<u8>> {
        keypoints
            .iter()
            .map(|kp| self.compute_single(image, kp))
            .collect()
    }

    fn compute_single(&self, image: &GrayImage, kp: &KeyPoint) -> Vec<u8> {
        let x = kp.x as i32;
        let y = kp.y as i32;

        let mut descriptor = vec![0u8; self.bytes];

        for (i, pair) in self.pattern.iter().enumerate() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;

            let v1 = get_pixel_safe(image, x + pair[0].0, y + pair[0].1);
            let v2 = get_pixel_safe(image, x + pair[1].0, y + pair[1].1);

            if v1 > v2 {
                descriptor[byte_idx] |= 1 << bit_idx;
            }
        }

        descriptor
    }
}

fn rand_i32() -> i32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let seed = SEED.get_or_init(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    let mut state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(seed.wrapping_add(std::hint::black_box(0u64)));
    (hasher.finish() as i32).abs()
}

fn get_pixel_safe(image: &GrayImage, x: i32, y: i32) -> u8 {
    if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
        image.get_pixel(x as u32, y as u32)[0]
    } else {
        0
    }
}
