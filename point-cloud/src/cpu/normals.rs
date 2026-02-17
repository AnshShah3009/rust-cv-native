//! CPU-based normal estimation operations
//!
//! Normal estimation operations include:
//! - Normal estimation using PCA on k-nearest neighbors
//! - Normal orientation consistency

pub use cv_scientific::point_cloud::{estimate_normals, orient_normals};
