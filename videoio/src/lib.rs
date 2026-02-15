//! Video input/output and camera capture
//!
//! This crate provides a unified interface for capturing video from cameras
//! and reading/writing video files.

use image::GrayImage;
use std::fmt::Debug;

pub type Result<T> = std::result::Result<T, VideoError>;

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Capture failed: {0}")]
    CaptureFailed(String),
}

/// Generic interface for video capture devices
pub trait VideoCapture: Send + Debug {
    /// Check if the device is open and ready
    fn is_opened(&self) -> bool;

    /// Grab the next frame from the stream
    fn grab(&mut self) -> Result<()>;

    /// Retrieve the grabbed frame as a GrayImage
    fn retrieve(&mut self) -> Result<GrayImage>;

    /// Convenience method to grab and retrieve a frame
    fn read(&mut self) -> Result<GrayImage> {
        self.grab()?;
        self.retrieve()
    }
}

/// Video writer interface (planned)
pub trait VideoWriter: Send + Debug {
    fn write(&mut self, frame: &GrayImage) -> Result<()>;
}

pub mod backends;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
