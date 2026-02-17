//! GPU-accelerated point cloud filtering operations
//!
//! GPU-based filtering with configurable compute modes:
//! - statistical_outlier_removal: Remove statistical outliers
//! - statistical_outlier_removal_simple: Backward-compatible interface
//! - remove_statistical_outliers: Apply SOR and return filtered points
//! - radius_outlier_removal: Remove radius outliers
//! - radius_outlier_removal_simple: Backward-compatible interface
//! - remove_radius_outliers: Apply ROR and return filtered points
//! - voxel_downsample: Voxel-based downsampling

pub use cv_3d::gpu::filtering::{
    statistical_outlier_removal, statistical_outlier_removal_simple,
    remove_statistical_outliers,
    radius_outlier_removal, radius_outlier_removal_simple,
    remove_radius_outliers,
};

pub use cv_3d::gpu::point_cloud::voxel_downsample;
