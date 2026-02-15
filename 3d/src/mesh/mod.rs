//! Triangle Mesh Data Structure and Processing
//!
//! Supports both CPU (rayon) and GPU (wgpu) processing backends.

use cv_core::point_cloud::PointCloud;
use nalgebra::{Point3, Vector3};

/// Triangle mesh with vertices and face indices
#[derive(Debug, Clone)]
pub struct TriangleMesh {
    pub vertices: Vec<Point3<f32>>,
    pub faces: Vec<[usize; 3]>,
    pub normals: Option<Vec<Vector3<f32>>>,
    pub colors: Option<Vec<Point3<f32>>>,
}

impl TriangleMesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
            normals: None,
            colors: None,
        }
    }

    pub fn with_vertices_and_faces(vertices: Vec<Point3<f32>>, faces: Vec<[usize; 3]>) -> Self {
        Self {
            vertices,
            faces,
            normals: None,
            colors: None,
        }
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Compute face normals
    pub fn compute_face_normals(&self) -> Vec<Vector3<f32>> {
        self.faces
            .iter()
            .map(|face| {
                let v0 = self.vertices[face[0]];
                let v1 = self.vertices[face[1]];
                let v2 = self.vertices[face[2]];

                let e1 = v1 - v0;
                let e2 = v2 - v0;

                e1.cross(&e2).normalize()
            })
            .collect()
    }

    /// Compute vertex normals by averaging adjacent face normals
    pub fn compute_vertex_normals(&mut self) {
        let mut vertex_normals: Vec<Vector3<f32>> = vec![Vector3::zeros(); self.vertices.len()];
        let face_normals = self.compute_face_normals();

        for (face_idx, face) in self.faces.iter().enumerate() {
            let normal = face_normals[face_idx];
            for &vertex_idx in face.iter() {
                vertex_normals[vertex_idx] += normal;
            }
        }

        // Normalize
        for normal in vertex_normals.iter_mut() {
            *normal = normal.normalize();
        }

        self.normals = Some(vertex_normals);
    }

    /// Calculate mesh bounds
    pub fn bounds(&self) -> (Point3<f32>, Point3<f32>) {
        if self.vertices.is_empty() {
            return (Point3::origin(), Point3::origin());
        }

        let mut min = self.vertices[0];
        let mut max = self.vertices[0];

        for v in &self.vertices {
            min.x = min.x.min(v.x);
            min.y = min.y.min(v.y);
            min.z = min.z.min(v.z);
            max.x = max.x.max(v.x);
            max.y = max.y.max(v.y);
            max.z = max.z.max(v.z);
        }

        (min, max)
    }

    /// Calculate surface area
    pub fn surface_area(&self) -> f32 {
        let mut area = 0.0;
        for face in &self.faces {
            let v0 = self.vertices[face[0]];
            let v1 = self.vertices[face[1]];
            let v2 = self.vertices[face[2]];

            let e1 = v1 - v0;
            let e2 = v2 - v0;

            area += e1.cross(&e2).norm() * 0.5;
        }
        area
    }

    /// Convert to point cloud (vertex positions)
    pub fn to_point_cloud(&self) -> PointCloud {
        PointCloud::new(self.vertices.clone())
    }
}

impl Default for TriangleMesh {
    fn default() -> Self {
        Self::new()
    }
}

pub mod processing;
pub mod reconstruction;
