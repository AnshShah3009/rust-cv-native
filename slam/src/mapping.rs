use nalgebra::Point3;

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

    pub fn get_descriptors(&self) -> cv_features::Descriptors {
        let mut descs = cv_features::Descriptors::new();
        for p in &self.points {
            descs.push(cv_features::Descriptor::new(
                p.descriptor.clone(),
                cv_core::KeyPoint::default(), // Placeholder keypoint
            ));
        }
        descs
    }
}
