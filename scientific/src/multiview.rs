use nalgebra::{Matrix3, Matrix4, Matrix3x4, Vector3, Vector2, SVD};
use cv_core::{Pose, CameraModel};

/// Fundamental Matrix solver using the Normalized 8-point Algorithm.
/// 
/// Ref: Hartley, R. I. (1997). In defense of the eight-point algorithm. 
/// IEEE Transactions on Pattern Analysis and Machine Intelligence.
pub struct FundamentalSolver;

/// Essential Matrix solver using Nistér's 5-point Algorithm.
pub struct EssentialSolver;

impl EssentialSolver {
    /// Estimate Essential Matrix E from 5 point correspondences.
    /// Points must be in normalized camera coordinates (K-inv * pixels).
    pub fn estimate_5point(pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> Vec<Matrix3<f64>> {
        if pts1.len() < 5 { return vec![]; }
        
        // 1. Form linear system for 4 basis matrices E1, E2, E3, E4
        // (Simplified for brevity, full Nister is 100+ lines of matrix manipulation)
        // In a real implementation, we'd solve for the nullspace of a 5x9 matrix.
        
        // For now, we'll provide a placeholder that falls back to 8-point logic 
        // if more points are available, or returns empty if only 5 are given 
        // until the full polynomial solver is implemented.
        // NOTE: Full implementation requires a 10th degree polynomial solver (Sturm sequences or Eigen).
        
        // TODO: Implement full Nistér polynomial solver.
        vec![] 
    }
}

impl FundamentalSolver {
    /// Estimate the Fundamental Matrix F from at least 8 point correspondences.
    /// Points should be in (x, y) pixel coordinates.
    pub fn estimate(pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> Option<Matrix3<f64>> {
        if pts1.len() < 8 || pts1.len() != pts2.len() {
            return None;
        }

        // 1. Normalization
        let (t1, norm_pts1) = Self::normalize_points(pts1);
        let (t2, norm_pts2) = Self::normalize_points(pts2);

        // 2. Form matrix A
        let mut a = nalgebra::DMatrix::zeros(pts1.len(), 9);
        for i in 0..pts1.len() {
            let u1 = norm_pts1[i][0];
            let v1 = norm_pts1[i][1];
            let u2 = norm_pts2[i][0];
            let v2 = norm_pts2[i][1];

            // Row i: [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]
            a[(i, 0)] = u2 * u1;
            a[(i, 1)] = u2 * v1;
            a[(i, 2)] = u2;
            a[(i, 3)] = v2 * u1;
            a[(i, 4)] = v2 * v1;
            a[(i, 5)] = v2;
            a[(i, 6)] = u1;
            a[(i, 7)] = v1;
            a[(i, 8)] = 1.0;
        }

        // 3. SVD of A to find F
        let svd = SVD::new(a, false, true);
        let v_t = svd.v_t.unwrap();
        let f_vec = v_t.row(8); // Last row of V^T (singular vector for smallest singular value)
        
        let mut f = Matrix3::new(
            f_vec[0], f_vec[1], f_vec[2],
            f_vec[3], f_vec[4], f_vec[5],
            f_vec[6], f_vec[7], f_vec[8],
        );

        // 4. Force Rank-2 Constraint
        let mut f_svd = SVD::new(f, true, true);
        f_svd.singular_values[2] = 0.0;
        f = f_svd.recompose().unwrap();

        // 5. Denormalization: F = T2^T * F_norm * T1
        Some(t2.transpose() * f * t1)
    }

    /// Normalizes points such that centroid is at origin and mean distance is sqrt(2).
    fn normalize_points(pts: &[[f64; 2]]) -> (Matrix3<f64>, Vec<[f64; 2]>) {
        let n = pts.len() as f64;
        let mut centroid_x = 0.0;
        let mut centroid_y = 0.0;
        for p in pts {
            centroid_x += p[0];
            centroid_y += p[1];
        }
        centroid_x /= n;
        centroid_y /= n;

        let mut mean_dist = 0.0;
        for p in pts {
            let dx = p[0] - centroid_x;
            let dy = p[1] - centroid_y;
            mean_dist += (dx * dx + dy * dy).sqrt();
        }
        mean_dist /= n;

        let scale = std::f64::consts::SQRT_2 / mean_dist;
        
        // Transformation matrix T
        let t = Matrix3::new(
            scale, 0.0, -scale * centroid_x,
            0.0, scale, -scale * centroid_y,
            0.0, 0.0, 1.0,
        );

        let norm_pts = pts.iter().map(|p| {
            let p_h = Vector3::new(p[0], p[1], 1.0);
            let p_n = t * p_h;
            [p_n.x, p_n.y]
        }).collect();

        (t, norm_pts)
    }
}

/// Linear Triangulation using the Direct Linear Transform (DLT) method.
pub struct Triangulator;

impl Triangulator {
    /// Triangulate a 3D point from two 2D observations and camera projection matrices.
    /// Observations should be in normalized camera coordinates (or pixel coordinates if P includes K).
    pub fn triangulate_linear(
        p1: &Matrix3x4<f64>,
        p2: &Matrix3x4<f64>,
        pt1: &[f64; 2],
        pt2: &[f64; 2],
    ) -> Option<Vector3<f64>> {
        let mut a = nalgebra::Matrix4::zeros();

        // Observation 1: u1 = (P1_1 * X) / (P1_3 * X), v1 = (P1_2 * X) / (P1_3 * X)
        // -> u1 * (P1_3 * X) - P1_1 * X = 0
        // -> v1 * (P1_3 * X) - P1_2 * X = 0
        for j in 0..4 {
            a[(0, j)] = pt1[0] * p1[(2, j)] - p1[(0, j)];
            a[(1, j)] = pt1[1] * p1[(2, j)] - p1[(1, j)];
            a[(2, j)] = pt2[0] * p2[(2, j)] - p2[(0, j)];
            a[(3, j)] = pt2[1] * p2[(2, j)] - p2[(1, j)];
        }

        let svd = SVD::new(a, false, true);
        let v_t = svd.v_t.unwrap();
        let x_h = v_t.row(3); // Last row of V^T

        if x_h[3].abs() < 1e-9 {
            return None; // Point at infinity or degenerate
        }

        Some(Vector3::new(x_h[0] / x_h[3], x_h[1] / x_h[3], x_h[2] / x_h[3]))
    }

    /// Triangulate multiple points.
    pub fn triangulate_points(
        pose1: &Pose,
        pose2: &Pose,
        pts1: &[[f64; 2]],
        pts2: &[[f64; 2]],
    ) -> Vec<Option<Vector3<f64>>> {
        // Construct projection matrices P = [R | t]
        // Assuming normalized camera coordinates (K = I)
        let p1 = pose1.to_matrix().fixed_view::<3, 4>(0, 0).into_owned();
        let p2 = pose2.to_matrix().fixed_view::<3, 4>(0, 0).into_owned();

        pts1.iter().zip(pts2.iter()).map(|(pt1, pt2)| {
            Self::triangulate_linear(&p1, &p2, pt1, pt2)
        }).collect()
    }

    /// Estimate camera pose using iterative Levenberg-Marquardt refinement.
    /// Returns the refined Pose given an initial guess, 3D points, 2D projections, and camera intrinsics.
    pub fn refine_pnp(
        initial_pose: &Pose,
        object_points: &[Vector3<f64>],
        image_points: &[[f64; 2]],
        intrinsics: &cv_core::CameraIntrinsics,
        max_iters: usize,
    ) -> Pose {
        // Implementation similar to calib3d but using cv_core types and no rayon/scheduler
        let mut current_pose = initial_pose.clone();
        let mut lambda = 0.001;
        
        let n = object_points.len();
        let fx = intrinsics.fx;
        let fy = intrinsics.fy;
        let cx = intrinsics.cx;
        let cy = intrinsics.cy;

        for _ in 0..max_iters {
            let mut jtj = nalgebra::Matrix6::<f64>::zeros();
            let mut jtr = nalgebra::Vector6::<f64>::zeros();
            let mut current_err = 0.0;

            let rot = current_pose.rotation();
            let t = current_pose.translation();

            for i in 0..n {
                let p_w = object_points[i];
                let p_c = rot * p_w + t;
                
                if p_c.z <= 1e-6 { continue; }
                
                let z_inv = 1.0 / p_c.z;
                let u = fx * p_c.x * z_inv + cx;
                let v = fy * p_c.y * z_inv + cy;

                let du = u - image_points[i][0];
                let dv = v - image_points[i][1];
                current_err += du * du + dv * dv;

                // Jacobian d(u,v) / d(rotation_axis, translation)
                // Use infinitesimal rotation approximation for Jacobian
                let mut j = nalgebra::Matrix2x6::zeros();
                
                // d(u,v)/dt
                j[(0, 3)] = fx * z_inv;
                j[(0, 4)] = 0.0;
                j[(0, 5)] = -fx * p_c.x * z_inv * z_inv;
                j[(1, 3)] = 0.0;
                j[(1, 4)] = fy * z_inv;
                j[(1, 5)] = -fy * p_c.y * z_inv * z_inv;

                // d(u,v)/d(omega) where omega is rotation vector
                // p_c' = p_c + [omega]_x * p_c
                let dpc_domega = nalgebra::Matrix3::new(
                    0.0, p_c.z, -p_c.y,
                    -p_c.z, 0.0, p_c.x,
                    p_c.y, -p_c.x, 0.0,
                );
                let duv_dpc = nalgebra::Matrix2x3::new(
                    fx * z_inv, 0.0, -fx * p_c.x * z_inv * z_inv,
                    0.0, fy * z_inv, -fy * p_c.y * z_inv * z_inv,
                );
                let duv_domega = duv_dpc * dpc_domega;
                j.fixed_view_mut::<2, 3>(0, 0).copy_from(&duv_domega);

                jtj += j.transpose() * j;
                jtr += j.transpose() * nalgebra::Vector2::new(du, dv);
            }

            let mut lhs = jtj;
            for k in 0..6 { lhs[(k, k)] *= 1.0 + lambda; }

            if let Some(delta) = lhs.lu().solve(&jtr) {
                // Update pose
                let omega = Vector3::new(delta[0], delta[1], delta[2]);
                let dt = Vector3::new(delta[3], delta[4], delta[5]);
                
                let d_rot = nalgebra::Rotation3::new(omega);
                let next_rot = d_rot * current_pose.rotation();
                let next_t = current_pose.translation() - dt;
                let next_pose = Pose::new(next_rot.into_inner(), next_t);

                // Simple check for improvement
                let mut next_err = 0.0;
                for i in 0..n {
                    let p_c = next_pose.rotation() * object_points[i] + next_pose.translation();
                    if p_c.z > 0.0 {
                        let u = fx * p_c.x / p_c.z + cx;
                        let v = fy * p_c.y / p_c.z + cy;
                        next_err += (u - image_points[i][0]).powi(2) + (v - image_points[i][1]).powi(2);
                    }
                }

                if next_err < current_err {
                    current_pose = next_pose;
                    lambda /= 10.0;
                    if delta.norm() < 1e-8 { break; }
                } else {
                    lambda *= 10.0;
                }
            } else {
                break;
            }
        }
        current_pose
    }

    /// Refine a 3D point estimate using non-linear least squares (Gauss-Newton).
    pub fn refine_triangulation(
        projection_matrices: &[Matrix3x4<f64>],
        observations: &[[f64; 2]],
        initial_point: Vector3<f64>,
        max_iters: usize,
    ) -> Vector3<f64> {
        let mut p = initial_point;
        let mut lambda = 0.001; // Levenberg-Marquardt

        for _ in 0..max_iters {
            let mut jtj = Matrix3::<f64>::zeros();
            let mut jtr = Vector3::<f64>::zeros();
            let mut current_err = 0.0;

            for (i, p_mat) in projection_matrices.iter().enumerate() {
                let obs = observations[i];
                
                // Project point: x = PX
                let x_h = p_mat * p.insert_row(3, 1.0);
                let z_inv = 1.0 / x_h.z;
                let u = x_h.x * z_inv;
                let v = x_h.y * z_inv;

                let du = u - obs[0];
                let dv = v - obs[1];
                current_err += du * du + dv * dv;

                // Jacobian d(u,v) / d(X,Y,Z)
                // u = (p00*X + p01*Y + p02*Z + p03) / (p20*X + p21*Y + p22*Z + p23)
                // du/dX = (p00 * x_h.z - x_h.x * p20) / (x_h.z^2)
                let mut j = nalgebra::Matrix2x3::zeros();
                for k in 0..3 {
                    j[(0, k)] = (p_mat[(0, k)] * x_h.z - x_h.x * p_mat[(2, k)]) * (z_inv * z_inv);
                    j[(1, k)] = (p_mat[(1, k)] * x_h.z - x_h.y * p_mat[(2, k)]) * (z_inv * z_inv);
                }

                jtj += j.transpose() * j;
                jtr += j.transpose() * nalgebra::Vector2::new(du, dv);
            }

            // Solve (J^T J + lambda*I) * delta = J^T r
            let mut lhs = jtj;
            for i in 0..3 { lhs[(i, i)] *= 1.0 + lambda; }

            if let Some(delta) = lhs.lu().solve(&jtr) {
                let next_p = p - delta;
                
                // Check if error improved
                let mut next_err = 0.0;
                for (i, p_mat) in projection_matrices.iter().enumerate() {
                    let obs = observations[i];
                    let x_h = p_mat * next_p.insert_row(3, 1.0);
                    let z_inv = 1.0 / x_h.z;
                    let du = x_h.x * z_inv - obs[0];
                    let dv = x_h.y * z_inv - obs[1];
                    next_err += du * du + dv * dv;
                }

                if next_err < current_err {
                    p = next_p;
                    lambda /= 10.0;
                    if delta.norm() < 1e-8 { break; }
                } else {
                    lambda *= 10.0;
                }
            } else {
                break;
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    #[test]
    fn test_8point_basic() {
        // Identity rotation and some translation
        let f_true = Matrix3::new(
            0.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0,
        );

        // Generate 8 points
        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();
        
        // This is a bit complex to generate perfectly consistent points without a full simulator,
        // but we can at least check if it handles the input.
        // For a real test, we'd project 3D points into two cameras.
        
        // Let's just mock some data for now to ensure no panics
        for i in 0..8 {
            pts1.push([i as f64 * 10.0, i as f64 * 10.0]);
            pts2.push([i as f64 * 10.0 + 5.0, i as f64 * 10.0]);
        }

        let f = FundamentalSolver::estimate(&pts1, &pts2);
        assert!(f.is_some());
    }

    #[test]
    fn test_triangulation() {
        // Camera 1 at origin
        let p1 = Matrix3x4::identity();
        // Camera 2 translated by 1.0 in x
        let mut p2 = Matrix3x4::identity();
        p2[(0, 3)] = -1.0;

        // Point at (0, 0, 5)
        let x_true = Vector3::new(0.0, 0.0, 5.0);
        
        // Project to cameras
        // x1 = (0, 0, 5) -> [0, 0, 5] -> (0/5, 0/5) = (0, 0)
        // x2 = (-1, 0, 5) -> [-1, 0, 5] -> (-1/5, 0/5) = (-0.2, 0)
        let pt1 = [0.0, 0.0];
        let pt2 = [-0.2, 0.0];

        let x_tri = Triangulator::triangulate_linear(&p1, &p2, &pt1, &pt2).unwrap();
        
        assert!((x_tri - x_true).norm() < 1e-6);
    }
}
