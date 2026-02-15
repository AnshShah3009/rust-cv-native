//! Computational photography module
//!
//! This crate provides algorithms for image enhancement, denoising,
//! and reconstruction.

use std::fmt::Debug;

pub type Result<T> = std::result::Result<T, PhotoError>;

#[derive(Debug, thiserror::Error)]
pub enum PhotoError {
    #[error("Algorithm error: {0}")]
    AlgorithmError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub mod bilateral;
pub mod stitcher;

pub use bilateral::*;
pub use stitcher::*;
