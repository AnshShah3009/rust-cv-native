//! Point Cloud Operations - Unified workspace crate
//!
//! This crate provides a unified interface for point cloud operations,
//! consolidating functionality from cv-scientific and cv-3d crates.
//!
//! The crate is organized into CPU and GPU modules, each re-exporting
//! implementations from the underlying crates to provide a single entry point.
//!
//! # Module Organization
//!
//! - `cpu`: CPU-based point cloud operations (filtering, normal estimation, segmentation, clustering)
//! - `gpu`: GPU-accelerated point cloud operations
//!
//! # Usage
//!
//! CPU operations:
//! ```ignore
//! use cv_point_cloud::cpu::*;
//! ```
//!
//! GPU operations:
//! ```ignore
//! use cv_point_cloud::gpu::*;
//! ```
//!
//! # Future Consolidation
//!
//! This crate serves as a foundation for consolidating point cloud functionality
//! without breaking existing code. Over time, implementations will be unified
//! into this crate.

pub mod cpu;
pub mod gpu;

// Re-export GPU types and configuration at crate level
pub use gpu::{ComputeMode, NormalComputeConfig};
