pub mod color;
pub mod convolve;
pub mod edges;
pub mod geometry;
pub mod histogram;
pub mod morph;
pub mod resize;

pub use color::*;
pub use convolve::*;
pub use edges::*;
pub use geometry::*;
pub use histogram::*;
pub use morph::*;
pub use resize::*;

use cv_core::{GrayImage, RgbImage, RgbaImage};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, ImgprocError>;

#[derive(Debug, Error)]
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

pub fn to_gray(img: &RgbImage) -> GrayImage {
    cv_core::convert_rgb_to_gray(img)
}

pub fn to_rgb(img: &GrayImage) -> RgbImage {
    cv_core::convert_gray_to_rgb(img)
}

pub fn ensure_gray(img: &GrayImage) -> Result<&GrayImage> {
    Ok(img)
}

pub fn ensure_color(img: &RgbImage) -> Result<&RgbImage> {
    Ok(img)
}

pub fn validate_image_size(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(ImgprocError::DimensionMismatch(
            "Image dimensions must be non-zero".into(),
        ));
    }
    Ok(())
}

pub fn compute_stride(width: u32, bytes_per_pixel: u32) -> usize {
    (width * bytes_per_pixel) as usize
}
