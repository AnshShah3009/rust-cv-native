use cv_core::KeyPoints;
pub use cv_core::{Descriptor, Descriptors};
use image::GrayImage;

pub trait DescriptorExtractor {
    fn extract(&self, image: &GrayImage, keypoints: &KeyPoints) -> Descriptors;
}
