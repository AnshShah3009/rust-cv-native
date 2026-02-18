use cv_calib3d::solve_pnp_ransac;
use cv_core::{CameraExtrinsics, CameraIntrinsics, KeyPoints, Tensor, storage::Storage};
use cv_features::{DescriptorExtractor, Descriptors, Orb};
use cv_hal::compute::ComputeDevice;
use cv_hal::tensor_ext::TensorToCpu;
use nalgebra::Point2;

pub struct Frame<S: Storage<u8> + 'static> {
    pub image: Tensor<u8, S>,
    pub keypoints: KeyPoints,
    pub descriptors: Descriptors,
    pub pose: CameraExtrinsics,
}

pub struct Tracker<'a> {
    pub device: &'a ComputeDevice<'a>,
    pub current_frame: Option<Frame<cv_core::storage::CpuStorage<u8>>>,
    pub last_frame: Option<Frame<cv_core::storage::CpuStorage<u8>>>,
    pub intrinsics: CameraIntrinsics,
    pub detector: Orb,
}

impl<'a> Tracker<'a> {
    pub fn new(device: &'a ComputeDevice<'a>, intrinsics: CameraIntrinsics) -> Self {
        Self {
            device,
            current_frame: None,
            last_frame: None,
            intrinsics,
            detector: Orb::default().with_n_features(500),
        }
    }

    pub fn process_frame<S: Storage<u8> + 'static>(
        &mut self,
        image: &Tensor<u8, S>,
        map: &crate::mapping::Map,
    ) -> Result<(CameraExtrinsics, Vec<usize>), String> {
        use cv_core::storage::CpuStorage;
        let mut keypoints = self.detector.detect_ctx(self.device, image);
        
        // Convert to CPU for orientations/descriptors (patch-based)
        let cpu_img_tensor: Tensor<u8, CpuStorage<u8>> = convert_to_cpu(self.device, image);

        let (h, w) = cpu_img_tensor.shape.hw();
        let gray = image::GrayImage::from_raw(w as u32, h as u32, cpu_img_tensor.storage.as_slice().unwrap().to_vec()).unwrap();
        
        self.detector.compute_orientations(&gray, &mut keypoints);
        let descriptors = self.detector.extract(&gray, &keypoints);

        let mut frame = Frame {
            image: cpu_img_tensor,
            keypoints,
            descriptors,
            pose: CameraExtrinsics::default(),
        };

        let mut tracking_success = false;
        let mut tracked_indices = Vec::new();

        if !map.points.is_empty() {
            let map_descs = map.get_descriptors();
            
            let map_desc_tensor: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(
                map_descs.descriptors.iter().flat_map(|d| d.data.clone()).collect(),
                cv_core::TensorShape::new(1, map_descs.len(), 32)
            );
            
            let query_desc_tensor: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(
                frame.descriptors.descriptors.iter().flat_map(|d| d.data.clone()).collect(),
                cv_core::TensorShape::new(1, frame.descriptors.len(), 32)
            );

            let matches = self.device.match_descriptors(&query_desc_tensor, &map_desc_tensor, 0.7)
                .map_err(|e| e.to_string())?;

            if matches.len() >= 10 {
                let mut object_pts = Vec::new();
                let mut image_pts = Vec::new();

                for m in &matches.matches {
                    object_pts.push(map.points[m.train_idx as usize].position);
                    let kp = &frame.keypoints.keypoints[m.query_idx as usize];
                    image_pts.push(Point2::new(kp.x, kp.y));
                }

                if let Ok((pose, inliers)) =
                    solve_pnp_ransac(&object_pts, &image_pts, &self.intrinsics, 2.0, 100)
                {
                    frame.pose = pose;
                    tracked_indices = matches
                        .matches
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| inliers[*i])
                        .map(|(_, m)| m.train_idx as usize)
                        .collect();
                    tracking_success = true;
                }
            }
        }

        if !tracking_success {
            if let Some(ref last) = self.last_frame {
                frame.pose = last.pose;
            } else if self.current_frame.is_none() {
                // First frame ever
                self.current_frame = Some(frame);
                return Err("First frame - tracking not possible".to_string());
            }
        }

        self.last_frame = self.current_frame.take();
        self.current_frame = Some(frame);

        Ok((self.current_frame.as_ref().unwrap().pose, tracked_indices))
    }
}

fn convert_to_cpu<S: Storage<u8> + 'static>(device: &ComputeDevice, tensor: &Tensor<u8, S>) -> Tensor<u8, cv_core::storage::CpuStorage<u8>> {
    use std::any::TypeId;
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;

    if TypeId::of::<S>() == TypeId::of::<GpuStorage<u8>>() {
        let input_ptr = tensor as *const Tensor<u8, S> as *const Tensor<u8, GpuStorage<u8>>;
        let input_gpu = unsafe { &*input_ptr };
        let gpu_ctx = match device {
            ComputeDevice::Gpu(g) => g,
            _ => panic!("Logic error"),
        };
        input_gpu.to_cpu_ctx(gpu_ctx).unwrap()
    } else {
        let input_ptr = tensor as *const Tensor<u8, S> as *const Tensor<u8, CpuStorage<u8>>;
        let input_cpu = unsafe { &*input_ptr };
        input_cpu.clone()
    }
}
