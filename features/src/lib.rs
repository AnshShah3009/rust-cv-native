pub mod orb;
pub mod fast;
pub mod harris;
pub mod matcher;
pub mod brief;

pub use orb::*;
pub use fast::*;
pub use harris::*;
pub use matcher::*;
pub use brief::*;

use cv_core::{GrayImage, KeyPoint, KeyPoints};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, FeatureError>;

#[derive(Debug, Error)]
pub enum FeatureError {
    #[error("Detection error: {0}")]
    DetectionError(String),
    
    #[error("Descriptor error: {0}")]
    DescriptorError(String),
    
    #[error("Matching error: {0}")]
    MatchingError(String),
}

pub trait FeatureDetector {
    fn detect(&self, image: &GrayImage) -> Result<KeyPoints>;
}

pub trait FeatureDescriptor {
    type Descriptor: AsRef<[u8]>;
    
    fn compute(&self, image: &GrayImage, keypoints: &mut KeyPoints) -> Result<Vec<Self::Descriptor>>;
}

pub trait DescriptorMatcher {
    fn match_descriptors(
        &self,
        query: &[impl AsRef<[u8]>],
        train: &[impl AsRef<[u8]>],
    ) -> Result<Vec<cv_core::FeatureMatch>>;
    
    fn match_with_knn(
        &self,
        query: &[impl AsRef<[u8]>],
        train: &[impl AsRef<[u8]>],
        k: usize,
    ) -> Result<Vec<Vec<cv_core::FeatureMatch>>>;
}

pub fn create_keypoints_from_response(
    image: &GrayImage,
    response: &[f32],
    threshold: f32,
    max_keypoints: usize,
) -> KeyPoints {
    let width = image.width() as usize;
    let height = image.height() as usize;
    
    let mut kps: Vec<(usize, f32)> = response
        .iter()
        .enumerate()
        .filter(|(_, &r)| r > threshold)
        .map(|(i, &r)| (i, r))
        .collect();
    
    kps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    kps.truncate(max_keypoints);
    
    let mut keypoints = KeyPoints::with_capacity(kps.len());
    
    for (idx, &response) in &kps {
        let y = idx / width;
        let x = idx % width;
        
        let kp = KeyPoint::new(x as f64, y as f64)
            .with_response(response as f64);
        
        keypoints.push(kp);
    }
    
    keypoints
}
