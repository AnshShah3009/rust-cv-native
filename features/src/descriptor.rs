use cv_core::{KeyPoint, KeyPoints};
use image::GrayImage;

#[derive(Debug, Clone)]
pub struct Descriptor {
    pub data: Vec<u8>,
    pub keypoint: KeyPoint,
}

impl Descriptor {
    pub fn new(data: Vec<u8>, keypoint: KeyPoint) -> Self {
        Self { data, keypoint }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn hamming_distance(&self, other: &Descriptor) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct Descriptors {
    pub descriptors: Vec<Descriptor>,
}

impl Descriptors {
    pub fn new() -> Self {
        Self {
            descriptors: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            descriptors: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, desc: Descriptor) {
        self.descriptors.push(desc);
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Descriptor> {
        self.descriptors.iter()
    }
}

impl Default for Descriptors {
    fn default() -> Self {
        Self::new()
    }
}

pub trait DescriptorExtractor {
    fn extract(&self, image: &GrayImage, keypoints: &KeyPoints) -> Descriptors;
}
