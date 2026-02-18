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
    fn is_opened(&self) -> bool;
    fn grab(&mut self) -> Result<()>;
    fn retrieve(&mut self) -> Result<GrayImage>;
    fn read(&mut self) -> Result<GrayImage> {
        self.grab()?;
        self.retrieve()
    }
}

pub mod backends;

/// Open a video file using native FFmpeg backend
pub fn open_video(path: &str) -> Result<Box<dyn VideoCapture>> {
    let cap = backends::NativeFfmpegCapture::new(path)?;
    Ok(Box::new(cap))
}

/// Open a camera device (V4L2 on Linux)
#[cfg(target_os = "linux")]
pub fn open_camera(path: &str, width: u32, height: u32) -> Result<Box<dyn VideoCapture>> {
    let mut cap = backends::V4L2Capture::new(path)?;
    cap.start_stream(width, height)?;
    Ok(Box::new(cap))
}
