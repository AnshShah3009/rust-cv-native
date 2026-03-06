//! CubeCL kernel implementations — one submodule per domain.
//!
//! Kernels are ported from WGSL tier-by-tier.  Each submodule is gated behind
//! the `cubecl` feature (inherited from the parent).

pub mod image;
pub mod icp;
pub mod matching;
pub mod pointcloud;
pub mod sort;
