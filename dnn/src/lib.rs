//! Deep Neural Network module
//!
//! This crate provides an interface for running pre-trained neural networks
//! using the ONNX Runtime.

use std::fmt::Debug;
use ort::session::{Session, builder::SessionBuilder};

pub type Result<T> = std::result::Result<T, DnnError>;

#[derive(Debug, thiserror::Error)]
pub enum DnnError {
    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] ort::Error),
}

pub struct Net {
    session: Session,
}

impl Net {
    pub fn from_file(path: &str) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)?;
        Ok(Self { session })
    }

    pub fn forward(&mut self, _input_blob: &[f32]) -> Result<Vec<f32>> {
        // Placeholder for inference logic
        Ok(Vec::new())
    }
}

pub mod blob;
