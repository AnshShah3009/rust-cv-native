use nalgebra::{Point3, Vector3};

/// A 3D Point Cloud with optional colors and normals.
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    pub points: Vec<Point3<f32>>,
    pub colors: Option<Vec<Point3<f32>>>,
    pub normals: Option<Vec<Vector3<f32>>>,
}

impl PointCloud {
    pub fn new(points: Vec<Point3<f32>>) -> Self {
        Self {
            points,
            colors: None,
            normals: None,
        }
    }

    pub fn with_colors(mut self, colors: Vec<Point3<f32>>) -> Self {
        if colors.len() == self.points.len() {
            self.colors = Some(colors);
        } else {
            // In a real scenario, we might return Result. For now, just panic or ignore?
            // Let's inconsistent state is bad.
            panic!("Color count {} does not match point count {}", colors.len(), self.points.len());
        }
        self
    }

    pub fn with_normals(mut self, normals: Vec<Vector3<f32>>) -> Self {
        if normals.len() == self.points.len() {
            self.normals = Some(normals);
        } else {
            panic!("Normal count {} does not match point count {}", normals.len(), self.points.len());
        }
        self
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}
