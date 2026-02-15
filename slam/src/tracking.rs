use image::GrayImage;
use nalgebra::Point2;
use cv_core::{KeyPoints, CameraExtrinsics, CameraIntrinsics};
use cv_features::{Orb, Descriptors, Matcher, MatchType, DescriptorExtractor};
use cv_calib3d::solve_pnp_ransac;

pub struct Frame {
    pub image: GrayImage,
    pub keypoints: KeyPoints,
    pub descriptors: Descriptors,
    pub pose: CameraExtrinsics,
}

pub struct Tracker {
    pub current_frame: Option<Frame>,
    pub last_frame: Option<Frame>,
    pub intrinsics: CameraIntrinsics,
    pub detector: Orb,
}

impl Tracker {
    pub fn new(intrinsics: CameraIntrinsics) -> Self {
        Self {
            current_frame: None,
            last_frame: None,
            intrinsics,
            detector: Orb::default().with_n_features(500),
        }
    }

    pub fn process_frame(
        &mut self, 
        image: &image::GrayImage, 
        map: &crate::mapping::Map
    ) -> Result<(CameraExtrinsics, Vec<usize>), String> {
        // 1. Detect and describe features
        let mut keypoints = self.detector.detect(image);
        self.detector.compute_orientations(image, &mut keypoints);
        let descriptors = self.detector.extract(image, &keypoints);
        
        let mut frame = Frame {
            image: image.clone(),
            keypoints,
            descriptors,
            pose: CameraExtrinsics::default(),
        };

        let tracking_success = false;

        // Try map-based tracking if map is not empty
        if !map.points.is_empty() {
            let map_descs = map.get_descriptors();
            let matcher = Matcher::new(MatchType::BruteForceHamming).with_ratio_test(0.7);
            let matches = matcher.match_descriptors(&frame.descriptors, &map_descs);

            if matches.len() >= 10 {
                let mut object_pts = Vec::new();
                let mut image_pts = Vec::new();

                for m in &matches.matches {
                    object_pts.push(map.points[m.train_idx as usize].position);
                    let kp = &frame.keypoints.keypoints[m.query_idx as usize];
                    image_pts.push(Point2::new(kp.x, kp.y));
                }

                if let Ok((pose, inliers)) = solve_pnp_ransac(
                    &object_pts, 
                    &image_pts, 
                    &self.intrinsics, 
                    2.0, 
                    100
                ) {
                    frame.pose = pose;
                    let tracked_indices = matches.matches.iter().enumerate()
                        .filter(|(i, _)| inliers[*i])
                        .map(|(_, m)| m.train_idx as usize)
                        .collect();
                    self.last_frame = self.current_frame.take();
                    self.current_frame = Some(frame);
                    return Ok((self.current_frame.as_ref().unwrap().pose, tracked_indices));
                }
            }
        }

        if !tracking_success {
            if let Some(ref last) = self.last_frame {
                // FALLBACK: Just use last pose
                frame.pose = last.pose; 
            } else {
                self.last_frame = self.current_frame.take();
                self.current_frame = Some(frame);
                return Err("First frame - tracking not possible".to_string());
            }
        }

        self.last_frame = self.current_frame.take();
        self.current_frame = Some(frame);
        
        Ok((self.current_frame.as_ref().unwrap().pose, Vec::new()))
    }
}
