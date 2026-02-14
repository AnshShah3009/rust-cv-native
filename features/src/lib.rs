pub mod fast;
pub mod harris;

pub use fast::*;
pub use harris::*;

use cv_core::{KeyPoint, KeyPoints};
use image::GrayImage;

pub type Result<T> = std::result::Result<T, FeatureError>;

#[derive(Debug, thiserror::Error)]
pub enum FeatureError {
    #[error("Detection error: {0}")]
    DetectionError(String),

    #[error("Descriptor error: {0}")]
    DescriptorError(String),

    #[error("Matching error: {0}")]
    MatchingError(String),
}

pub fn detect_keypoints(image: &GrayImage, max_keypoints: usize) -> KeyPoints {
    let mut kps = fast::fast_detect(image, 20, max_keypoints);
    KeyPoints {
        keypoints: kps.keypoints,
    }
}
