pub mod bilateral;
pub mod color;
pub mod contours;
pub mod convolve;
pub mod edges;
pub mod geometry;
pub mod histogram;
pub mod hough;
pub mod local_threshold;
pub mod morph;
pub mod resize;
pub mod simd;
pub mod template_matching;
pub mod threshold;

pub use bilateral::*;
pub use color::*;
pub use contours::*;
pub use convolve::*;
pub use edges::*;
pub use geometry::*;
pub use histogram::*;
pub use local_threshold::*;
pub use morph::*;
pub use resize::*;
pub use template_matching::*;
pub use threshold::*;

pub type Result<T> = std::result::Result<T, ImgprocError>;

#[derive(Debug, thiserror::Error)]
pub enum ImgprocError {
    #[error("Image error: {0}")]
    ImageError(String),

    #[error("Algorithm error: {0}")]
    AlgorithmError(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

pub fn validate_image_size(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(ImgprocError::DimensionMismatch(
            "Image dimensions must be non-zero".into(),
        ));
    }
    Ok(())
}
