//! Camera capture backends

#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub mod v4l2;

pub mod png_sequence;
pub mod gif;

#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;

#[cfg(feature = "ffmpeg")]
pub use ffmpeg::NativeFfmpegCapture;

pub use png_sequence::{PngSequenceCapture, PngSequenceWriter};
pub use gif::GifCapture;

#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub use v4l2::V4L2Capture;
