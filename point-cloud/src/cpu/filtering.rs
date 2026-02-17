//! CPU-based point cloud filtering operations
//!
//! Filtering operations include:
//! - Voxel downsampling
//! - Statistical outlier removal
//! - Radius outlier removal

pub use cv_scientific::point_cloud::{
    voxel_down_sample, remove_statistical_outliers, remove_radius_outliers,
};
