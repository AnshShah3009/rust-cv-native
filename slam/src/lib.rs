pub mod mapping;
pub mod tracking;
pub mod pose_graph;
pub mod kalman;

use cv_core::{CameraExtrinsics, CameraIntrinsics, Tensor, storage::CpuStorage};
use cv_hal::compute::ComputeDevice;
use crate::mapping::Map;
use crate::tracking::Tracker;

pub struct Slam<'a> {
    pub map: Map,
    pub tracker: Tracker<'a>,
    pub intrinsics: CameraIntrinsics,
}

impl<'a> Slam<'a> {
    pub fn new(device: &'a ComputeDevice<'a>, intrinsics: CameraIntrinsics) -> Self {
        Self {
            map: Map::new(),
            tracker: Tracker::new(device, intrinsics.clone()),
            intrinsics,
        }
    }

    pub fn process_image(&mut self, image: &image::GrayImage) -> Result<(CameraExtrinsics, Vec<usize>), String> {
        let shape = cv_core::TensorShape::new(1, image.height() as usize, image.width() as usize);
        let tensor: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(image.to_vec(), shape);
        
        // Front-end tracking
        self.tracker.process_frame(&tensor, &self.map)
    }
}
