//! GPU-accelerated normal estimation operations
//!
//! GPU-based normal estimation with configurable compute modes

pub use cv_3d::gpu::point_cloud::{
    compute_normals, compute_normals_cpu, compute_normals_ctx, compute_normals_simple,
};
