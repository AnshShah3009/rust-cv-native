//! Visual SLAM module
//!
//! This crate provides algorithms for Simultaneous Localization
//! and Mapping using visual input.

pub mod tracking;
pub mod mapping;

use cv_core::CameraIntrinsics;
use tracking::Tracker;
use mapping::Map;

pub trait SlamSystem {
    fn process_frame(&mut self, image: &image::GrayImage);
}

pub struct Slam {
    pub tracker: Tracker,
    pub map: Map,
}

impl Slam {
    pub fn new(intrinsics: CameraIntrinsics) -> Self {
        Self {
            tracker: Tracker::new(intrinsics),
            map: Map::new(),
        }
    }
}

impl SlamSystem for Slam {
    fn process_frame(&mut self, image: &image::GrayImage) {
        let _pose = self.tracker.process_frame(image);
        // Mapping logic would go here:
        // 1. Create new map points from triangulation
        // 2. Perform local mapping (BA)
        // 3. Check for loop closures
    }
}
