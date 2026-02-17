//! GPU-accelerated normal estimation operations
//!
//! GPU-based normal estimation with configurable compute modes:
//! - compute_normals: Main entry point with ComputeMode selection
//! - compute_normals_simple: Backward-compatible simple interface
//! - compute_normals_with_ctx: GPU computation with explicit context
//! - voxel_based_normals: Voxel-based normal computation
//! - approximate_normals: Approximate normals with regularization
//! - refine_normals: Iterative normal refinement

pub use cv_3d::gpu::point_cloud::{
    compute_normals, compute_normals_simple, compute_normals_with_ctx,
    voxel_based_normals, voxel_based_normals_simple,
    approximate_normals, approximate_normals_simple,
    refine_normals,
};
