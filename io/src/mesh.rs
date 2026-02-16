//! Triangle Mesh Data Structure
//!
//! Placeholder module for mesh I/O functionality

use nalgebra::Point3;

/// Placeholder TriangleMesh type
#[derive(Debug, Clone)]
pub struct TriangleMesh {
    /// Vertices of the mesh
    pub vertices: Vec<Point3<f32>>,
    /// Faces (triangles) as indices into vertices
    pub faces: Vec<[usize; 3]>,
}

impl TriangleMesh {
    /// Create a new mesh from vertices and faces
    pub fn with_vertices_and_faces(vertices: Vec<Point3<f32>>, faces: Vec<[usize; 3]>) -> Self {
        Self { vertices, faces }
    }

    /// Returns true if the mesh has no vertices
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Returns the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the number of faces
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }
}
