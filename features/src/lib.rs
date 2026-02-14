pub mod brief;
pub mod descriptor;
pub mod fast;
pub mod flann;
pub mod harris;
pub mod matcher;
pub mod orb;
pub mod ransac;

pub use brief::*;
pub use descriptor::*;
pub use fast::*;
pub use flann::*;
pub use harris::*;
pub use matcher::*;
pub use orb::*;
pub use ransac::*;

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
