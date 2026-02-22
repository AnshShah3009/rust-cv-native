#![warn(missing_docs)]
//! Core computer vision types and traits
//!
//! This crate provides fundamental types and abstractions for computer vision operations,
//! including point clouds, tensors, keypoints, and descriptors.

pub mod descriptor;
pub mod frames;
pub mod geometry;
pub mod image;
pub mod keypoint;
pub mod point_cloud;
pub mod robust;
pub mod runtime;
pub mod storage;
pub mod tensor;
pub mod slam;

pub use descriptor::*;
pub use frames::*;
pub use geometry::*;
pub use image::*;
pub use keypoint::*;
pub use point_cloud::*;
pub use robust::*;
pub use runtime::*;
pub use storage::*;
pub use tensor::*;
pub use slam::*;

/// Error types for core operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Runtime error with message
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Concurrency error with message
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),

    /// Invalid input with message
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Memory allocation or access error
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Tensor or array dimension mismatch
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// I/O error from file operations
    #[error("I/O error: {0}")]
    IoError(String),
}

/// Result type for core operations
pub type Result<T> = std::result::Result<T, Error>;
