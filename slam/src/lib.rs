//! Visual SLAM (Simultaneous Localization and Mapping)
//!
//! This crate provides algorithms for Simultaneous Localization and Mapping
//! using visual input (cameras). It implements key SLAM components:
//!
//! ## Core Components
//!
//! - [`Slam`]: Main SLAM system integrating tracking, mapping, and loop closure
//! - [`KeyFrame`]: Keyframe representation with pose, features, and descriptors
//! - [`SlamConfig`]: Configuration for SLAM system parameters
//!
//! ## Submodules
//!
//! - [`kalman`]: Kalman filters for state estimation (Linear, Extended)
//! - [`tracking`]: Visual odometry and feature tracking
//! - [`mapping`]: 3D map management and optimization
//!
//! ## Kalman Filters
//!
//! The crate provides robust state estimation via:
//! - [`KalmanFilter`]: Standard linear Kalman filter
//! - [`ExtendedKalmanFilter`] (EKF): For non-linear systems
//!
//! ## Example: Using Kalman Filter for 2D Tracking
//!
//! ```rust
//! use cv_slam::kalman::{KalmanFilter, KalmanFilterState, utils};
//! use nalgebra::SVector;
//!
//! // Create a constant velocity model for 2D tracking
//! let kf = utils::constant_velocity_2d(0.1, 0.01, 0.1);
//! let mut state = KalmanFilterState::zero();
//!
//! // Prediction and update steps
//! let u = SVector::<f64, 4>::zeros();
//! let z = nalgebra::Vector2::new(1.0, 1.0);
//!
//! kf.step(&mut state, &u, &z);
//! println!("State: {:?}", state.x);
//! ```
//!
//! ## Example: Basic SLAM System
//!
//! ```rust
//! // use cv_slam::{Slam, SlamConfig};
//! // use cv_core::CameraIntrinsics;
//! //
//! // let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
//! // let config = SlamConfig::default();
//! // let mut slam = Slam::new(intrinsics).with_config(config);
//! ```

pub mod kalman;
pub mod mapping;
pub mod tracking;

pub use kalman::{ExtendedKalmanFilter, KalmanFilter, KalmanFilterState};

use mapping::Map;
use nalgebra::{Point2, Point3, Vector3};
use std::collections::{HashMap, HashSet};
use tracking::Tracker;

pub trait SlamSystem: Send {
    fn process_frame(&mut self, image: &image::GrayImage);
}

pub struct KeyFrame {
    pub id: usize,
    pub pose: cv_core::CameraExtrinsics,
    pub keypoints: cv_core::KeyPoints,
    pub descriptors: cv_features::Descriptors,
    pub is_loop_closure: bool,
}

pub struct SlamConfig {
    pub max_keyframes: usize,
    pub min_tracked_points: usize,
    pub keyframe_distance_threshold: f64,
    pub loop_closure_distance: f64,
    pub ba_iterations: usize,
}

impl Default for SlamConfig {
    fn default() -> Self {
        Self {
            max_keyframes: 1000,
            min_tracked_points: 20,
            keyframe_distance_threshold: 0.5,
            loop_closure_distance: 3.0,
            ba_iterations: 20,
        }
    }
}

pub struct Slam {
    pub tracker: Tracker,
    pub map: Map,
    pub keyframes: Vec<KeyFrame>,
    pub frame_counter: usize,
    pub config: SlamConfig,
    pub loop_closure_detected: bool,
    pub covisibility_graph: HashMap<usize, HashSet<usize>>,
}

impl Slam {
    pub fn new(intrinsics: cv_core::CameraIntrinsics) -> Self {
        Self {
            tracker: Tracker::new(intrinsics),
            map: Map::new(),
            keyframes: Vec::new(),
            frame_counter: 0,
            config: SlamConfig::default(),
            loop_closure_detected: false,
            covisibility_graph: HashMap::new(),
        }
    }

    pub fn with_config(mut self, config: SlamConfig) -> Self {
        self.config = config;
        self
    }

    fn check_new_keyframe(&mut self, tracked_indices: &[usize]) {
        if self.map.points.is_empty() {
            return;
        }

        let should_spawn = if let Some(last_kf) = self.keyframes.last() {
            if let Some(ref curr) = self.tracker.current_frame {
                let dist = (curr.pose.translation - last_kf.pose.translation).norm();
                dist > self.config.keyframe_distance_threshold
                    || tracked_indices.len() < self.config.min_tracked_points
            } else {
                false
            }
        } else {
            true
        };

        if should_spawn {
            if let Some(ref curr) = self.tracker.current_frame {
                let kf = KeyFrame {
                    id: self.keyframes.len(),
                    pose: curr.pose,
                    keypoints: cv_core::KeyPoints {
                        keypoints: curr.keypoints.keypoints.clone(),
                    },
                    descriptors: cv_features::Descriptors {
                        descriptors: curr.descriptors.descriptors.clone(),
                    },
                    is_loop_closure: false,
                };
                self.keyframes.push(kf);
                self.update_covisibility(tracked_indices);
                self.expand_map(tracked_indices);

                // Check for loop closure
                if self.keyframes.len() > 10 {
                    self.detect_loop_closure();
                }

                // Run bundle adjustment periodically
                if self.keyframes.len() % 5 == 0 {
                    self.run_local_bundle_adjustment();
                }
            }
        }
    }

    fn update_covisibility(&mut self, tracked_indices: &[usize]) {
        if let Some(current_kf_id) = self.keyframes.last().map(|kf| kf.id) {
            let tracked_set: HashSet<usize> = tracked_indices.iter().cloned().collect();

            // Update covisibility with recent keyframes
            for kf in self.keyframes.iter().rev().take(10) {
                if kf.id != current_kf_id {
                    let entry = self
                        .covisibility_graph
                        .entry(current_kf_id)
                        .or_insert_with(HashSet::new);
                    // Count shared observations
                    let shared = tracked_set.len(); // Simplified
                    if shared > 10 {
                        entry.insert(kf.id);
                    }
                }
            }
        }
    }

    fn detect_loop_closure(&mut self) {
        if self.keyframes.len() < 10 {
            return;
        }

        let current_kf = &self.keyframes[self.keyframes.len() - 1];

        // Check against older keyframes
        for kf in self.keyframes.iter().rev().take(20) {
            if kf.id + 10 < current_kf.id {
                // Compute distance between poses
                let dist = (current_kf.pose.translation - kf.pose.translation).norm();

                if dist < self.config.loop_closure_distance {
                    // Potential loop closure detected
                    self.loop_closure_detected = true;
                    self.perform_loop_correction(kf.id, current_kf.id);
                    break;
                }
            }
        }
    }

    fn perform_loop_correction(&mut self, old_kf_id: usize, new_kf_id: usize) {
        // Use pose graph optimization to correct drift
        // This would integrate with cv_3d pose graph
        // For now, simple pose correction
        if let (Some(old_kf), Some(new_kf)) =
            (self.keyframes.get(old_kf_id), self.keyframes.get(new_kf_id))
        {
            let delta = old_kf.pose.translation - new_kf.pose.translation;

            // Apply correction to all keyframes after old_kf_id
            for kf in self.keyframes.iter_mut().skip(old_kf_id + 1) {
                kf.pose.translation += delta;
                kf.is_loop_closure = true;
            }
        }
    }

    fn run_local_bundle_adjustment(&mut self) {
        if self.keyframes.len() < 3 {
            return;
        }

        // Select local keyframes for BA (last N keyframes)
        let local_kf_ids: Vec<usize> = self
            .keyframes
            .iter()
            .rev()
            .take(10)
            .map(|kf| kf.id)
            .collect();

        // This would use the SFM bundle adjustment
        // For now, simple pose refinement
        let mut updated_poses: Vec<(usize, cv_core::CameraExtrinsics)> = Vec::new();

        for &kf_id in &local_kf_ids {
            if let Some(kf) = self.keyframes.get(kf_id) {
                // Refine pose based on reprojection error
                // Simplified: small gradient descent update
                let refined_pose = kf.pose;
                // ... refinement logic would go here
                updated_poses.push((kf_id, refined_pose));
            }
        }

        // Apply updated poses
        for (id, pose) in updated_poses {
            if let Some(kf) = self.keyframes.get_mut(id) {
                kf.pose = pose;
            }
        }
    }

    fn expand_map(&mut self, tracked_indices: &[usize]) {
        if self.keyframes.len() < 2 {
            return;
        }

        let kf2 = &self.keyframes[self.keyframes.len() - 1];
        let kf1 = &self.keyframes[self.keyframes.len() - 2];

        let matcher = cv_features::Matcher::new(cv_features::MatchType::BruteForceHamming)
            .with_ratio_test(0.7);
        let matches = matcher.match_descriptors(&kf1.descriptors, &kf2.descriptors);

        let mut new_pts1 = Vec::new();
        let mut new_pts2 = Vec::new();
        let mut query_indices = Vec::new();
        let mut train_indices = Vec::new();

        let tracked_set: HashSet<usize> = tracked_indices.iter().cloned().collect();

        for m in &matches.matches {
            if !tracked_set.contains(&(m.train_idx as usize)) {
                let kp1 = &kf1.keypoints.keypoints[m.query_idx as usize];
                let kp2 = &kf2.keypoints.keypoints[m.train_idx as usize];
                new_pts1.push(Point2::new(kp1.x, kp1.y));
                new_pts2.push(Point2::new(kp2.x, kp2.y));
                query_indices.push(m.query_idx as usize);
                train_indices.push(m.train_idx as usize);
            }
        }

        if new_pts1.is_empty() {
            return;
        }

        let k = self.tracker.intrinsics.matrix();
        let proj1 = k * kf1.pose.matrix().fixed_view::<3, 4>(0, 0);
        let proj2 = k * kf2.pose.matrix().fixed_view::<3, 4>(0, 0);

        if let Ok(pts3d) = cv_sfm::triangulate_points(&new_pts1, &new_pts2, &proj1, &proj2) {
            for (i, p) in pts3d.into_iter().enumerate() {
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

    pub fn get_current_pose(&self) -> Option<cv_core::CameraExtrinsics> {
        self.tracker.current_frame.as_ref().map(|f| f.pose)
    }

    pub fn get_trajectory(&self) -> Vec<(usize, Point3<f64>)> {
        self.keyframes
            .iter()
            .map(|kf| (kf.id, Point3::from(kf.pose.translation)))
            .collect()
    }

    pub fn get_map_points(&self) -> Vec<Point3<f64>> {
        self.map.points.iter().map(|p| p.position).collect()
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

            let matcher = cv_features::Matcher::new(cv_features::MatchType::BruteForceHamming)
                .with_ratio_test(0.7);
            let matches = matcher.match_descriptors(&f1.descriptors, &f2.descriptors);

            if matches.len() < 20 {
                return;
            }

            let pts1: Vec<Point2<f64>> = matches
                .matches
                .iter()
                .map(|m| {
                    let kp = &f1.keypoints.keypoints[m.query_idx as usize];
                    Point2::new(kp.x, kp.y)
                })
                .collect();

            let pts2: Vec<Point2<f64>> = matches
                .matches
                .iter()
                .map(|m| {
                    let kp = &f2.keypoints.keypoints[m.train_idx as usize];
                    Point2::new(kp.x, kp.y)
                })
                .collect();

            let descriptors: Vec<Vec<u8>> = matches
                .matches
                .iter()
                .map(|m| {
                    f2.descriptors.descriptors[m.train_idx as usize]
                        .data
                        .clone()
                })
                .collect();

            (pts1, pts2, matches, descriptors)
        };

        if let Ok(e) = cv_calib3d::find_essential_mat(&pts1, &pts2, &self.tracker.intrinsics) {
            if let Ok(pose) =
                cv_calib3d::recover_pose_from_essential(&e, &pts1, &pts2, &self.tracker.intrinsics)
            {
                if let Some(ref mut curr) = self.tracker.current_frame {
                    curr.pose = pose;
                }

                let k = self.tracker.intrinsics.matrix();
                let proj1 = k * nalgebra::Matrix3x4::identity();
                let proj2 = k * pose.matrix().fixed_view::<3, 4>(0, 0);

                if let Ok(pts3d) = cv_sfm::triangulate_points(&pts1, &pts2, &proj1, &proj2) {
                    for (i, p) in pts3d.into_iter().enumerate() {
                        let p_cam2 = pose.rotation * p.coords + pose.translation;
                        if p.z > 0.0 && p_cam2.z > 0.0 {
                            let m = matches.matches[i];
                            self.map.add_point(mapping::MapPoint {
                                position: p,
                                descriptor: descriptors[i].clone(),
                                observations: vec![
                                    (0, m.query_idx as usize),
                                    (1, m.train_idx as usize),
                                ],
                            });
                        }
                    }
                }
            }
        }
    }
}
