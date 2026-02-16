//! Point cloud registration algorithms
//!
//! This crate provides 3D point cloud registration algorithms:
//! - ICP (Iterative Closest Point)
//! - FGR (Fast Global Registration)
//! - Colored ICP
//! - GNC (Graduated Non-Convexity) for robust registration
//! - Multi-scale ICP

pub mod registration;

pub use registration::{
    evaluate_registration, get_information_matrix_from_point_clouds,
    registration_icp_point_to_plane, registration_multi_scale_icp, ICPResult,
};

pub use registration::colored::{registration_colored_icp, ColoredICPResult};
pub use registration::global::{
    registration_fgr_based_on_feature_matching, registration_ransac_based_on_feature_matching,
    FPFHFeature, FastGlobalRegistrationOption, GlobalRegistrationResult,
};
pub use registration::gnc::{
    registration_gnc, GNCOptimizer, GNCResult, RobustLoss, RobustLossType,
};
