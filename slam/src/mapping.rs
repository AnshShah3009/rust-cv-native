use nalgebra::Point3;
use cv_core::KeyPoint;

pub struct MapPoint {
    pub position: Point3<f64>,
    pub descriptor: Vec<u8>,
    pub observations: Vec<(usize, usize)>, // (frame_idx, keypoint_idx)
}

pub struct Map {
    pub points: Vec<MapPoint>,
}

impl Map {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
        }
    }

    pub fn add_point(&mut self, point: MapPoint) {
        self.points.push(point);
    }
}
