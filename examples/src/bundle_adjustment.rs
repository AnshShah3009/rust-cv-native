//! Example: Bundle Adjustment
//!
//! Run with: cargo run --example bundle_adjustment

use cv_core::CameraIntrinsics;
use cv_sfm::bundle_adjustment::{
    bundle_adjust, BundleAdjustmentConfig, CameraExtrinsics, SfMState,
};
use nalgebra::{Matrix3, Point2, Point3, Vector3};

fn main() {
    println!("=== Bundle Adjustment Example ===\n");

    // Create camera intrinsics
    let intrinsics = CameraIntrinsics::new(
        500.0, 500.0, // fx, fy
        320.0, 240.0, // cx, cy
        640, 480, // width, height
    );

    // Create SfM state
    let mut state = SfMState::new(intrinsics);

    // Add two cameras (poses)
    state.add_camera(CameraExtrinsics::new(Matrix3::identity(), Vector3::zeros()));

    state.add_camera(CameraExtrinsics::new(
        Matrix3::identity(),
        Vector3::new(1.0, 0.0, 0.0), // Translated 1 unit in X
    ));

    // Add landmarks observed by both cameras
    // Landmark 1: at (0.5, 0, 5)
    state.add_landmark(
        Point3::new(0.5, 0.0, 5.0),
        vec![
            (0, Point2::new(320.0, 240.0)), // Camera 0 sees it at center
            (1, Point2::new(420.0, 240.0)), // Camera 1 sees it shifted right
        ],
    );

    // Landmark 2: at (0, 0.5, 5)
    state.add_landmark(
        Point3::new(0.0, 0.5, 5.0),
        vec![
            (0, Point2::new(320.0, 290.0)), // Camera 0 sees it below center
            (1, Point2::new(320.0, 290.0)), // Camera 1 sees it same position
        ],
    );

    println!("Initial state:");
    println!("  Cameras: {}", state.cameras.len());
    println!("  Landmarks: {}", state.landmarks.len());
    println!("  Initial error: {:.6}", state.total_reprojection_error());

    // Configure and run bundle adjustment
    let config = BundleAdjustmentConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        lambda: 0.001,
        use_sparsity: true,
        robust_kernel: true,
    };

    println!("\nRunning bundle adjustment...");
    bundle_adjust(&mut state, &config);

    println!("\nAfter optimization:");
    println!("  Final error: {:.6}", state.total_reprojection_error());

    // Show refined camera poses
    println!("\nRefined camera poses:");
    for (i, cam) in state.cameras.iter().enumerate() {
        println!(
            "  Camera {}: t = ({:.3}, {:.3}, {:.3})",
            i, cam.translation.x, cam.translation.y, cam.translation.z
        );
    }

    // Show refined landmark positions
    println!("\nRefined landmark positions:");
    for (i, lm) in state.landmarks.iter().enumerate() {
        println!(
            "  Landmark {}: ({:.3}, {:.3}, {:.3})",
            i, lm.position.x, lm.position.y, lm.position.z
        );
    }

    println!("\n=== Example Complete ===");
}
