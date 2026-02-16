//! 3D Registration Module
//!
//! Implements various registration algorithms:
//! - ICP (Iterative Closest Point)
//! - Colored ICP
//! - Global Registration (RANSAC, FGR)
//! - GNC (Graduated Non-Convexity) robust registration

pub mod colored;
pub mod global;
pub mod gnc;

pub use colored::{registration_colored_icp, ColoredICPResult};
pub use global::{
    registration_fgr_based_on_feature_matching, registration_ransac_based_on_feature_matching,
    FPFHFeature, FastGlobalRegistrationOption, GlobalRegistrationResult,
};
pub use gnc::{registration_gnc, GNCOptimizer, GNCResult, RobustLoss, RobustLossType};

use cv_core::point_cloud::PointCloud;
use nalgebra::{Matrix4, Point3, Vector3};

/// Simple KD-tree-like structure for nearest neighbor search
struct SimpleNN {
    points: Vec<Point3<f32>>,
}

impl SimpleNN {
    fn new(points: Vec<Point3<f32>>) -> Self {
        Self { points }
    }

    fn nearest(&self, query: &Point3<f32>) -> Option<(Point3<f32>, usize, f32)> {
        let mut min_dist = f32::MAX;
        let mut min_idx = 0;

        for (i, pt) in self.points.iter().enumerate() {
            let dist = (pt - query).norm_squared();
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        if min_dist < f32::MAX {
            Some((self.points[min_idx], min_idx, min_dist))
        } else {
            None
        }
    }
}

/// Standard ICP registration result
#[derive(Debug, Clone)]
pub struct ICPResult {
    pub transformation: Matrix4<f32>,
    pub fitness: f32,
    pub inlier_rmse: f32,
    pub num_iterations: usize,
}

/// Standard point-to-plane ICP
pub fn registration_icp_point_to_plane(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f32,
    init_transformation: &Matrix4<f32>,
    max_iterations: usize,
) -> Option<ICPResult> {
    // Build simple nearest neighbor structure for target
    let target_nn = SimpleNN::new(target.points.clone());

    let mut transformation = *init_transformation;
    let mut best_fitness = 0.0;
    let mut best_rmse = f32::MAX;
    let mut final_iterations = 0;

    for iter in 0..max_iterations {
        let mut correspondences: Vec<(usize, usize, f32)> = Vec::new();

        // Find correspondences
        for (src_idx, src_point) in source.points.iter().enumerate() {
            let transformed = transformation.transform_point(src_point);

            if let Some((target_point, target_idx, dist_sq)) = target_nn.nearest(&transformed) {
                let dist = dist_sq.sqrt();

                if dist <= max_correspondence_distance {
                    correspondences.push((src_idx, target_idx, dist));
                }
            }
        }

        if correspondences.len() < 3 {
            break;
        }

        // Compute point-to-plane error
        let mut ata = nalgebra::Matrix6::<f32>::zeros();
        let mut atb = nalgebra::Vector6::<f32>::zeros();
        let mut total_residual = 0.0;

        for (src_idx, tgt_idx, _) in &correspondences {
            let src_point = source.points[*src_idx];
            let tgt_point = target.points[*tgt_idx];
            let tgt_normal = target.normals.as_ref().map(|n| &n[*tgt_idx]);

            if let Some(normal) = tgt_normal {
                let transformed = transformation.transform_point(&src_point);
                let diff = transformed - tgt_point;
                let residual = diff.dot(normal);

                // Compute Jacobian
                let p = src_point.coords;
                let n = normal;
                let cross = p.cross(n);

                let jacobian = nalgebra::Vector6::new(n.x, n.y, n.z, cross.x, cross.y, cross.z);

                ata += jacobian * jacobian.transpose();
                atb += jacobian * residual;
                total_residual += residual * residual;
            }
        }

        // Solve for update
        if let Some(ata_inv) = ata.try_inverse() {
            let delta = ata_inv * atb;

            // Update transformation using exponential map
            let update = exponential_map_se3(&delta);
            transformation = update * transformation;
        }

        // Evaluate
        let rmse = (total_residual / correspondences.len() as f32).sqrt();
        let fitness = correspondences.len() as f32 / source.points.len() as f32;

        if fitness > best_fitness {
            best_fitness = fitness;
            best_rmse = rmse;
            final_iterations = iter + 1;
        }

        if rmse < 1e-6 {
            break;
        }
    }

    Some(ICPResult {
        transformation,
        fitness: best_fitness,
        inlier_rmse: best_rmse,
        num_iterations: final_iterations,
    })
}

/// Multi-scale ICP
pub fn registration_multi_scale_icp(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distances: &[f32],
    init_transformation: &Matrix4<f32>,
    max_iterations_per_scale: usize,
) -> Option<ICPResult> {
    let mut transformation = *init_transformation;
    let mut best_result = None;

    for &max_dist in max_correspondence_distances {
        if let Some(result) = registration_icp_point_to_plane(
            source,
            target,
            max_dist,
            &transformation,
            max_iterations_per_scale,
        ) {
            transformation = result.transformation;
            best_result = Some(result);
        }
    }

    best_result
}

/// Exponential map from se(3) to SE(3)
fn exponential_map_se3(delta: &nalgebra::Vector6<f32>) -> Matrix4<f32> {
    let omega = nalgebra::Vector3::new(delta[3], delta[4], delta[5]);
    let v = nalgebra::Vector3::new(delta[0], delta[1], delta[2]);

    let theta = omega.norm();

    let rotation = if theta < 1e-6 {
        nalgebra::Matrix3::identity()
    } else {
        let k = omega / theta;
        let k_cross = nalgebra::Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        nalgebra::Matrix3::identity()
            + k_cross * theta.sin()
            + k_cross * k_cross * (1.0 - theta.cos())
    };

    let translation = v;

    let mut transform = Matrix4::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);

    transform
}

/// Compute information matrix from registration
pub fn get_information_matrix_from_point_clouds(
    source: &PointCloud,
    target: &PointCloud,
    transformation: &Matrix4<f32>,
) -> nalgebra::Matrix6<f32> {
    let mut information = nalgebra::Matrix6::<f32>::zeros();

    // Build simple nearest neighbor for target
    let target_nn = SimpleNN::new(target.points.clone());

    // Accumulate information from correspondences
    for src_point in &source.points {
        let transformed = transformation.transform_point(src_point);

        if let Some((target_point, _, dist_sq)) = target_nn.nearest(&transformed) {
            if dist_sq.sqrt() < 0.05 {
                // Small distance threshold
                let diff = transformed - target_point;

                // Compute Jacobian (simplified)
                let p = src_point.coords;
                let jacobian = nalgebra::Vector6::new(
                    diff.x,
                    diff.y,
                    diff.z,
                    p.y * diff.z - p.z * diff.y,
                    p.z * diff.x - p.x * diff.z,
                    p.x * diff.y - p.y * diff.x,
                );

                information += jacobian * jacobian.transpose();
            }
        }
    }

    information
}

/// Evaluate registration
pub fn evaluate_registration(
    source: &PointCloud,
    target: &PointCloud,
    transformation: &Matrix4<f32>,
    max_correspondence_distance: f32,
) -> (f32, f32) {
    let target_nn = SimpleNN::new(target.points.clone());

    let mut inlier_count = 0;
    let mut total_error = 0.0;

    for point in &source.points {
        let transformed = transformation.transform_point(point);
        if let Some((_, _, dist_sq)) = target_nn.nearest(&transformed) {
            let dist = dist_sq.sqrt();
            if dist < max_correspondence_distance {
                inlier_count += 1;
                total_error += dist * dist;
            }
        }
    }

    let fitness = if !source.points.is_empty() {
        inlier_count as f32 / source.points.len() as f32
    } else {
        0.0
    };

    let rmse = if inlier_count > 0 {
        (total_error / inlier_count as f32).sqrt()
    } else {
        0.0
    };

    (fitness, rmse)
}
