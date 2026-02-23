#![warn(missing_docs)]

//! Feature detection, extraction, and matching algorithms.
//!
//! This crate provides implementations of common computer vision feature detection and
//! descriptor extraction algorithms, including:
//!
//! - **AKAZE**: Accelerated-KAZE multi-scale feature detector with M-SURF descriptors
//! - **BRIEF**: Binary Robust Independent Elementary Features descriptor
//! - **FAST**: Features from Accelerated Segment Test corner detection
//! - **GFTT**: Good Features to Track corner detection (Harris corner detector)
//! - **ORB**: Oriented FAST and Rotated BRIEF features
//! - **SIFT**: Scale-Invariant Feature Transform detector and descriptor
//! - **FLANN**: Fast Library for Approximate Nearest Neighbors matching
//! - **Matcher**: Feature descriptor matching with various strategies
//!
//! # Basic Usage
//!
//! ```no_run
//! # use cv_features::*;
//! # use image::GrayImage;
//! let image = GrayImage::new(640, 480);
//!
//! // Detect ORB features
//! let orb = orb::Orb::new().with_n_features(500);
//! let keypoints = orb.detect(&image);
//! let descriptors = orb.extract(&image, &keypoints);
//!
//! // Match features
//! let matcher = matcher::Matcher::new(matcher::MatchType::BruteForceHamming);
//! let matches = matcher.match_descriptors(&descriptors, &descriptors);
//! ```

/// BRIEF (Binary Robust Independent Elementary Features) descriptor
pub mod brief;
/// Feature descriptor types and trait implementations
pub mod descriptor;
/// FAST (Features from Accelerated Segment Test) corner detection
pub mod fast;
/// FLANN (Fast Library for Approximate Nearest Neighbors) matching
pub mod flann;
/// Harris corner detection algorithm
pub mod harris;
/// Feature descriptor matching with various strategies
pub mod matcher;
/// Marker detection and tracking
pub mod markers;
/// ORB (Oriented FAST and Rotated BRIEF) detector and descriptor
pub mod orb;
/// RANSAC-based model estimation and outlier rejection
pub mod ransac;
/// Histogram of Oriented Gradients (HOG) feature extraction
pub mod hog;
/// AKAZE (Accelerated-KAZE) feature detector and descriptor
pub mod akaze;
/// GFTT (Good Features to Track) corner detection
pub mod gftt;
/// SIFT (Scale-Invariant Feature Transform) detector and descriptor
pub mod sift;
/// LBD (Line Band Descriptor) for line feature matching
pub mod lbd;
/// Line feature matching and tracking
pub mod line_matcher;

pub use brief::*;
pub use descriptor::*;
pub use fast::*;
pub use flann::*;
pub use harris::*;
pub use matcher::*;
pub use markers::*;
pub use orb::*;
pub use ransac::*;
pub use hog::*;
pub use akaze::*;
pub use gftt::*;
pub use sift::*;
pub use lbd::*;
pub use line_matcher::*;
pub use cv_imgproc::hough;

pub use cv_core::{KeyPoint, KeyPoints, Error, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Error instead. This type exists only for backward compatibility."
)]
pub type FeatureError = cv_core::Error;

/// Deprecated Result type alias - use cv_core::Result instead
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Result instead. This type alias exists only for backward compatibility."
)]
pub type FeatureResult<T> = cv_core::Result<T>;

