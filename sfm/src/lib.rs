//! Structure from Motion (SfM)
//!
//! This crate provides algorithms for 3D reconstruction from multiple 2D images.
//! It implements complete photogrammetry pipelines including:
//!
//! ## Core Components
//!
//! - [`bundle_adjustment`]: Non-linear optimization for refining camera poses and 3D points
//! - [`triangulation`]: Triangulating 3D points from multiple views
//!
//! ## Bundle Adjustment
//!
//! The bundle adjustment module provides:
//! - Levenberg-Marquardt optimization
//! - Robust kernel support
//! - Sparse Jacobian computation for efficiency
//! - Configurable convergence criteria
//!
//! ## Example
//!
//! ```rust
//! use cv_sfm::{bundle_adjustment::{BundleAdjustmentConfig, bundle_adjust, SfMState}, CameraIntrinsics, CameraExtrinsics};
//! use nalgebra::{Point2, Point3};
//!
//! // Setup cameras and observations
//! let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
//! let mut state = SfMState::new(intrinsics);
//!
//! // Add cameras and landmarks
//! state.add_camera(CameraExtrinsics::new(
//!     nalgebra::Matrix3::identity(),
//!     nalgebra::Vector3::zeros(),
//! ));
//!
//! state.add_landmark(
//!     Point3::new(0.5, 0.0, 5.0),
//!     vec![(0, Point2::new(320.0, 240.0))],
//! );
//!
//! // Run bundle adjustment
//! let config = BundleAdjustmentConfig::default();
//! bundle_adjust(&mut state, &config);
//! ```

pub mod bundle_adjustment;
pub mod triangulation;

pub use triangulation::*;

pub trait BundleAdjustment {
    fn optimize(&mut self);
}
