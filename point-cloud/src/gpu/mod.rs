//! GPU-accelerated point cloud operations
//!
//! This module re-exports point cloud functionality from cv-3d GPU module,
//! providing a unified interface for GPU-accelerated operations including:
//! - Normal computation with configurable compute modes
//! - Point cloud filtering (statistical and radius outlier removal)
//! - Point cloud clustering (DBSCAN, Euclidean)
//! - Point cloud downsampling

pub mod normals;
pub mod filtering;

// Re-export main GPU types and functions
pub use cv_3d::gpu::{ComputeMode, NormalComputeConfig};
pub use normals::*;
pub use filtering::*;
