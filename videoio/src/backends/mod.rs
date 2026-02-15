//! Camera capture backends

#[cfg(target_os = "linux")]
pub mod v4l2;

pub mod png_sequence;
