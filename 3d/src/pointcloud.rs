use nalgebra::{Point3, Vector3};
use cv_runtime::UnifiedBuffer;

pub struct PointCloud {
    pub positions: UnifiedBuffer<f64>, // Store as flat x,y,z
    pub normals: Option<UnifiedBuffer<f64>>,
    pub colors: Option<UnifiedBuffer<f32>>,
    pub num_points: usize,
}

impl PointCloud {
    pub fn new(num_points: usize) -> Self {
        Self {
            positions: UnifiedBuffer::new(num_points * 3),
            normals: None,
            colors: None,
            num_points,
        }
    }

    pub fn with_normals(mut self) -> Self {
        self.normals = Some(UnifiedBuffer::new(self.num_points * 3));
        self
    }

    pub fn with_colors(mut self) -> Self {
        self.colors = Some(UnifiedBuffer::new(self.num_points * 3));
        self
    }

    pub fn get_point(&self, index: usize) -> Option<Point3<f64>> {
        if index >= self.num_points { return None; }
        let host = self.positions.host_view().ok()?;
        Some(Point3::new(host[index * 3], host[index * 3 + 1], host[index * 3 + 2]))
    }

    pub fn set_point(&mut self, index: usize, pt: Point3<f64>) -> bool {
        if index >= self.num_points { return false; }
        if let Ok(mut host) = self.positions.host_view() {
            host[index * 3] = pt.x;
            host[index * 3 + 1] = pt.y;
            host[index * 3 + 2] = pt.z;
            true
        } else {
            false
        }
    }
}
