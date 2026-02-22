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

#[derive(Debug, Clone, Copy, Default)]
pub enum ComputeMode {
    #[default]
    CPU,
    GPU,
    Hybrid,
    Adaptive,
}

#[derive(Debug, Clone, Default)]
pub struct NormalComputeConfig {
    pub k: usize,
    pub voxel_size: f32,
    pub mode: ComputeMode,
}

impl NormalComputeConfig {
    pub fn default() -> Self {
        Self {
            k: 15,
            voxel_size: 0.01,
            mode: ComputeMode::CPU,
        }
    }

    pub fn cpu() -> Self {
        Self {
            k: 15,
            voxel_size: 0.01,
            mode: ComputeMode::CPU,
        }
    }

    pub fn gpu() -> Self {
        Self {
            k: 15,
            voxel_size: 0.01,
            mode: ComputeMode::GPU,
        }
    }

    pub fn fast() -> Self {
        Self {
            k: 10,
            voxel_size: 0.02,
            mode: ComputeMode::CPU,
        }
    }

    pub fn high_quality() -> Self {
        Self {
            k: 30,
            voxel_size: 0.005,
            mode: ComputeMode::CPU,
        }
    }
}
