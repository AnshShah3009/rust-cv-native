use nalgebra::Point2;

#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    pub x: f64,
    pub y: f64,
    pub size: f64,
    pub angle: f64,
    pub response: f64,
    pub octave: i32,
    pub class_id: i32,
}

impl KeyPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            size: 1.0,
            angle: -1.0,
            response: 0.0,
            octave: 0,
            class_id: -1,
        }
    }

    pub fn with_size(mut self, size: f64) -> Self {
        self.size = size;
        self
    }

    pub fn with_angle(mut self, angle: f64) -> Self {
        self.angle = angle;
        self
    }

    pub fn with_response(mut self, response: f64) -> Self {
        self.response = response;
        self
    }

    pub fn with_octave(mut self, octave: i32) -> Self {
        self.octave = octave;
        self
    }

    pub fn pt(&self) -> Point2<f64> {
        Point2::new(self.x, self.y)
    }

    pub fn scaled_pt(&self, scale: f64) -> Point2<f64> {
        Point2::new(self.x * scale, self.y * scale)
    }
}

impl Default for KeyPoint {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KeyPointF32 {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
    pub class_id: i32,
    pub padding: i32,
}

impl From<KeyPoint> for KeyPointF32 {
    fn from(kp: KeyPoint) -> Self {
        Self {
            x: kp.x as f32,
            y: kp.y as f32,
            size: kp.size as f32,
            angle: kp.angle as f32,
            response: kp.response as f32,
            octave: kp.octave,
            class_id: kp.class_id,
            padding: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FeatureMatch {
    pub query_idx: i32,
    pub train_idx: i32,
    pub distance: f32,
    pub img_idx: i32,
}

impl FeatureMatch {
    pub fn new(query_idx: i32, train_idx: i32, distance: f32) -> Self {
        Self {
            query_idx,
            train_idx,
            distance,
            img_idx: 0,
        }
    }

    pub fn with_img_idx(mut self, img_idx: i32) -> Self {
        self.img_idx = img_idx;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Matches {
    pub matches: Vec<FeatureMatch>,
    pub mask: Option<Vec<u8>>,
}

impl Matches {
    pub fn new() -> Self {
        Self {
            matches: Vec::new(),
            mask: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            matches: Vec::with_capacity(capacity),
            mask: None,
        }
    }

    pub fn push(&mut self, m: FeatureMatch) {
        self.matches.push(m);
    }

    pub fn len(&self) -> usize {
        self.matches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    pub fn filter_by_distance(&mut self, max_distance: f32) {
        self.matches.retain(|m| m.distance <= max_distance);
    }

    pub fn apply_ratio_test(&self, ratio_threshold: f32) -> Vec<FeatureMatch> {
        self.matches
            .iter()
            .filter(|m| m.distance <= ratio_threshold)
            .copied()
            .collect()
    }
}

impl Default for Matches {
    fn default() -> Self {
        Self::new()
    }
}

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

    pub fn iter(&self) -> impl Iterator<Item = &KeyPoint> {
        self.keypoints.iter()
    }
}

impl Default for KeyPoints {
    fn default() -> Self {
        Self::new()
    }
}
