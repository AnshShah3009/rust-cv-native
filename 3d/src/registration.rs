use nalgebra::{Point3, Matrix4, Rotation3, Vector3};
use crate::pointcloud::PointCloud;

pub struct RegistrationResult {
    pub transformation: Matrix4<f64>,
    pub fitness: f64,
    pub inlier_rmse: f64,
}

pub fn registration_icp(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f64,
    init_transformation: Matrix4<f64>,
    max_iterations: usize,
) -> RegistrationResult {
    let mut current_transformation = init_transformation;
    let mut fitness = 0.0;
    let mut rmse = 0.0;

    for _ in 0..max_iterations {
        // 1. Find correspondences (Naive O(N*M) for now, should use KD-Tree)
        let mut correspondences = Vec::new();
        let mut total_error = 0.0;

        for i in 0..source.num_points {
            if let Some(s_pt) = source.get_point(i) {
                // Transform source point
                let s_pt_transformed = current_transformation.transform_point(&s_pt);
                
                // Find nearest in target
                let mut min_dist = f64::MAX;
                let mut best_target_pt = None;

                for j in 0..target.num_points {
                    if let Some(t_pt) = target.get_point(j) {
                        let dist = (s_pt_transformed - t_pt).norm();
                        if dist < min_dist {
                            min_dist = dist;
                            best_target_pt = Some(t_pt);
                        }
                    }
                }

                if min_dist < max_correspondence_distance {
                    correspondences.push((s_pt, best_target_pt.unwrap()));
                    total_error += min_dist.powi(2);
                }
            }
        }

        if correspondences.is_empty() { break; }

        fitness = correspondences.len() as f64 / source.num_points as f64;
        rmse = (total_error / correspondences.len() as f64).sqrt();

        // 2. Solve for optimal transformation (Arun's method)
        // ... simplified placeholder for now ...
        // In a real implementation we'd use SVD to find R and t
    }

    RegistrationResult {
        transformation: current_transformation,
        fitness,
        inlier_rmse: rmse,
    }
}
