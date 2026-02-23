//! Computational Photography Algorithms
//!
//! Provides algorithms for image enhancement, filtering, denoising,
//! and multi-image processing.
//!
//! # Algorithms
//!
//! - **Bilateral Filter**: Edge-preserving smoothing for denoising
//! - **Image Stitching**: Panoramic image compositing
//!
//! # Use Cases
//!
//! - High-quality image denoising while preserving edges
//! - Panoramic image generation
//! - Image enhancement and restoration
//!
//! # Example: Bilateral Filtering
//!
//! ```no_run
//! # use cv_photo::bilateral::bilateral_filter;
//! # use image::GrayImage;
//! let image = GrayImage::new(640, 480);
//! // let filtered = bilateral_filter(&image, 5, 50.0, 50.0);
//! ```

pub use cv_core::{Error, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Error instead. This type exists only for backward compatibility."
)]
pub type PhotoError = cv_core::Error;

/// Deprecated Result type alias - use cv_core::Result instead
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Result instead. This type alias exists only for backward compatibility."
)]
pub type PhotoResult<T> = cv_core::Result<T>;

/// Edge-preserving bilateral filtering
pub mod bilateral;
/// Panoramic image stitching
pub mod stitcher;

pub use bilateral::*;
pub use stitcher::*;
