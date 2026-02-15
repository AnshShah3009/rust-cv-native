//! Visual SLAM module
//!
//! This crate provides algorithms for Simultaneous Localization
//! and Mapping using visual input.

pub mod tracking;
pub mod mapping;

use nalgebra::Point2;
use tracking::Tracker;
use mapping::Map;
use std::collections::HashSet;

pub trait SlamSystem: Send {
    fn process_frame(&mut self, image: &image::GrayImage);
}

pub struct KeyFrame {
    pub id: usize,
    pub pose: cv_core::CameraExtrinsics,
    pub keypoints: cv_core::KeyPoints,
    pub descriptors: cv_features::Descriptors,
}

pub struct Slam {
    pub tracker: Tracker,
    pub map: Map,
    pub keyframes: Vec<KeyFrame>,
    pub frame_counter: usize,
}

impl Slam {
    pub fn new(intrinsics: cv_core::CameraIntrinsics) -> Self {
        Self {
            tracker: Tracker::new(intrinsics),
            map: Map::new(),
            keyframes: Vec::new(),
            frame_counter: 0,
        }
    }

    fn check_new_keyframe(&mut self, tracked_indices: &[usize]) {
        if self.map.points.is_empty() { return; }

        let should_spawn = if let Some(_last_kf) = self.keyframes.last() {
            // Spawn if we moved significantly or tracking is getting weak
            // Simplified: every 10 frames or if tracked points are low
            self.frame_counter % 10 == 0 || tracked_indices.len() < 20
        } else {
            // If we initialized the map, the first frames should be keyframes
            true
        };

        if should_spawn {
            if let Some(ref curr) = self.tracker.current_frame {
                let kf = KeyFrame {
                    id: self.keyframes.len(),
                    pose: curr.pose,
                    keypoints: cv_core::KeyPoints { keypoints: curr.keypoints.keypoints.clone() },
                    descriptors: cv_features::Descriptors { descriptors: curr.descriptors.descriptors.clone() },
                };
                self.keyframes.push(kf);
                self.expand_map(tracked_indices);
            }
        }
    }

    fn expand_map(&mut self, tracked_indices: &[usize]) {
        if self.keyframes.len() < 2 { return; }
        
        let kf2 = &self.keyframes[self.keyframes.len() - 1];
        let kf1 = &self.keyframes[self.keyframes.len() - 2];

        // 1. Match current KF against previous KF
        let matcher = cv_features::Matcher::new(cv_features::MatchType::BruteForceHamming).with_ratio_test(0.7);
        let matches = matcher.match_descriptors(&kf1.descriptors, &kf2.descriptors);

        // 2. Identify new potential points (those not already tracked)
        let mut new_pts1 = Vec::new();
        let mut new_pts2 = Vec::new();
        let mut query_indices = Vec::new();
        let mut train_indices = Vec::new();

        let tracked_set: HashSet<usize> = tracked_indices.iter().cloned().collect();

        for m in &matches.matches {
            // For simplicity, we just check if the destination index in KF2 was tracked
            // In a real SLAM we'd check if the feature has a map point
            if !tracked_set.contains(&(m.train_idx as usize)) {
                let kp1 = &kf1.keypoints.keypoints[m.query_idx as usize];
                let kp2 = &kf2.keypoints.keypoints[m.train_idx as usize];
                new_pts1.push(Point2::new(kp1.x, kp1.y));
                new_pts2.push(Point2::new(kp2.x, kp2.y));
                query_indices.push(m.query_idx as usize);
                train_indices.push(m.train_idx as usize);
            }
        }

        if new_pts1.is_empty() { return; }

        // 3. Triangulate
        let k = self.tracker.intrinsics.matrix();
        let proj1 = k * kf1.pose.matrix().fixed_view::<3, 4>(0, 0);
        let proj2 = k * kf2.pose.matrix().fixed_view::<3, 4>(0, 0);

        if let Ok(pts3d) = cv_sfm::triangulate_points(&new_pts1, &new_pts2, &proj1, &proj2) {
            for (i, p) in pts3d.into_iter().enumerate() {
                // Cheirality check in both frames
                let p_cam1 = kf1.pose.rotation * p.coords + kf1.pose.translation;
                let p_cam2 = kf2.pose.rotation * p.coords + kf2.pose.translation;
                
                if p_cam1.z > 0.0 && p_cam2.z > 0.0 {
                    let desc = kf2.descriptors.descriptors[train_indices[i]].data.clone();
                    self.map.add_point(mapping::MapPoint {
                        position: p,
                        descriptor: desc,
                        observations: vec![(kf1.id, query_indices[i]), (kf2.id, train_indices[i])],
                    });
                }
            }
        }
    }
}

impl SlamSystem for Slam {
    fn process_frame(&mut self, image: &image::GrayImage) {
        self.frame_counter += 1;
        let pose_result = self.tracker.process_frame(image, &self.map);
        
        match pose_result {
            Ok((_pose, tracked_indices)) => {
                self.check_new_keyframe(&tracked_indices);
            }
            Err(_) => {
                if self.map.points.is_empty() {
                    self.initialize_map();
                }
            }
        }
    }
}

impl Slam {
    fn initialize_map(&mut self) {
        let (pts1, pts2, matches, descriptors) = {
            let (f1, f2) = match (&self.tracker.last_frame, &self.tracker.current_frame) {
                (Some(f1), Some(f2)) => (f1, f2),
                _ => return,
            };

            // 1. Match features
            let matcher = cv_features::Matcher::new(cv_features::MatchType::BruteForceHamming).with_ratio_test(0.7);
            let matches = matcher.match_descriptors(&f1.descriptors, &f2.descriptors);
            
            if matches.len() < 20 { return; }

            // 2. Get pixel coordinates
            let pts1: Vec<Point2<f64>> = matches.matches.iter().map(|m| {
                let kp = &f1.keypoints.keypoints[m.query_idx as usize];
                Point2::new(kp.x, kp.y)
            }).collect();
            
            let pts2: Vec<Point2<f64>> = matches.matches.iter().map(|m| {
                let kp = &f2.keypoints.keypoints[m.train_idx as usize];
                Point2::new(kp.x, kp.y)
            }).collect();

            // Store descriptors for mapping later to avoid borrowing f2
            let descriptors: Vec<Vec<u8>> = matches.matches.iter().map(|m| {
                f2.descriptors.descriptors[m.train_idx as usize].data.clone()
            }).collect();

            (pts1, pts2, matches, descriptors)
        };

        // 3. Estimate Essential Matrix and Recover Pose
        if let Ok(e) = cv_calib3d::find_essential_mat(&pts1, &pts2, &self.tracker.intrinsics) {
            if let Ok(pose) = cv_calib3d::recover_pose_from_essential(&e, &pts1, &pts2, &self.tracker.intrinsics) {
                // Update current frame pose
                if let Some(ref mut curr) = self.tracker.current_frame {
                    curr.pose = pose;
                }

                // 4. Triangulate points
                let k = self.tracker.intrinsics.matrix();
                let proj1 = k * nalgebra::Matrix3x4::identity();
                let proj2 = k * pose.matrix().fixed_view::<3, 4>(0, 0);

                if let Ok(pts3d) = cv_sfm::triangulate_points(&pts1, &pts2, &proj1, &proj2) {
                    for (i, p) in pts3d.into_iter().enumerate() {
                        // Check if point is in front
                        let p_cam2 = pose.rotation * p.coords + pose.translation;
                        if p.z > 0.0 && p_cam2.z > 0.0 {
                            let m = matches.matches[i];
                            self.map.add_point(mapping::MapPoint {
                                position: p,
                                descriptor: descriptors[i].clone(),
                                observations: vec![(0, m.query_idx as usize), (1, m.train_idx as usize)],
                            });
                        }
                    }
                }
            }
        }
    }
}
