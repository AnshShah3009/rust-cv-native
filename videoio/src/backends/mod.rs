//! Camera capture backends

#[cfg(target_os = "linux")]
pub mod v4l2;

pub mod png_sequence;
pub mod ffmpeg;

pub use ffmpeg::FfmpegCapture;
pub use png_sequence::PngSequenceCapture;
#[cfg(target_os = "linux")]
pub use v4l2::V4L2Capture;
