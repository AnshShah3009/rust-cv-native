use nalgebra::{DVector, DMatrix, Vector3, Point3, Point2, Matrix3x4, Rotation3};
use cv_core::{CameraIntrinsics, CameraExtrinsics};

/// Data structure representing a point in 3D and its observations across frames.
pub struct LandMark {
    pub position: Point3<f64>,
    pub observations: Vec<(usize, Point2<f64>)>, // (frame_idx, pixel_coord)
}

/// Represents the state of the SfM system for optimization.
pub struct SfMState {
    pub cameras: Vec<CameraExtrinsics>,
    pub landmarks: Vec<LandMark>,
    pub intrinsics: CameraIntrinsics,
}

impl SfMState {
    pub fn new(intrinsics: CameraIntrinsics) -> Self {
        Self {
            cameras: Vec::new(),
            landmarks: Vec::new(),
            intrinsics,
        }
    }

    /// Compute total reprojection error.
    pub fn total_reprojection_error(&self) -> f64 {
        let mut total_err = 0.0;
        let mut count = 0;

        for landmark in &self.landmarks {
            for (cam_idx, obs) in &landmark.observations {
                let cam = &self.cameras[*cam_idx];
                let pt_cam = cam.rotation * landmark.position + cam.translation;
                
                if pt_cam.z <= 0.0 {
                    total_err += 1e6; // Penalty for points behind camera
                    continue;
                }

                let projected = self.intrinsics.project(&pt_cam.into());
                let err = (projected.x - obs.x).powi(2) + (projected.y - obs.y).powi(2);
                total_err += err;
                count += 1;
            }
        }
        
        if count > 0 {
            total_err / count as f64
        } else {
            0.0
        }
    }

    /// Convert state to a flat parameter vector: [cam_rots, cam_trans, landmark_pos]
    pub fn to_parameters(&self) -> DVector<f64> {
        let n_cam = self.cameras.len();
        let n_lm = self.landmarks.len();
        let mut params = DVector::zeros(6 * n_cam + 3 * n_lm);
        
        for (i, cam) in self.cameras.iter().enumerate() {
            let rotation = Rotation3::from_matrix_unchecked(cam.rotation);
            let axis_angle = rotation.scaled_axis();
            params[6 * i + 0] = axis_angle.x;
            params[6 * i + 1] = axis_angle.y;
            params[6 * i + 2] = axis_angle.z;
            params[6 * i + 3] = cam.translation.x;
            params[6 * i + 4] = cam.translation.y;
            params[6 * i + 5] = cam.translation.z;
        }
        
        let offset = 6 * n_cam;
        for (i, lm) in self.landmarks.iter().enumerate() {
            params[offset + 3 * i + 0] = lm.position.x;
            params[offset + 3 * i + 1] = lm.position.y;
            params[offset + 3 * i + 2] = lm.position.z;
        }
        
        params
    }

    /// Update state from a flat parameter vector.
    pub fn from_parameters(&mut self, params: &DVector<f64>) {
        let n_cam = self.cameras.len();
        for (i, cam) in self.cameras.iter_mut().enumerate() {
            let axis_angle = Vector3::new(params[6 * i], params[6 * i + 1], params[6 * i + 2]);
            cam.rotation = Rotation3::new(axis_angle).into_inner();
            cam.translation = Vector3::new(params[6 * i + 3], params[6 * i + 4], params[6 * i + 5]);
        }
        
        let offset = 6 * n_cam;
        for (i, lm) in self.landmarks.iter_mut().enumerate() {
            lm.position = Point3::new(params[offset + 3 * i], params[offset + 3 * i + 1], params[offset + 3 * i + 2]);
        }
    }

    /// Compute residual vector (projected - observed).
    pub fn residuals(&self) -> DVector<f64> {
        let mut residuals = Vec::new();
        for landmark in &self.landmarks {
            for (cam_idx, obs) in &landmark.observations {
                let cam = &self.cameras[*cam_idx];
                let pt_cam = cam.rotation * landmark.position + cam.translation;
                let projected = self.intrinsics.project(&pt_cam.into());
                residuals.push(projected.x - obs.x);
                residuals.push(projected.y - obs.y);
            }
        }
        DVector::from_vec(residuals)
    }

    /// Compute Jacobian matrix numerically.
    pub fn numerical_jacobian(&mut self) -> DMatrix<f64> {
        let params = self.to_parameters();
        let residuals = self.residuals();
        let n_res = residuals.len();
        let n_params = params.len();
        let mut jacobian = DMatrix::zeros(n_res, n_params);
        let eps = 1e-6;

        for j in 0..n_params {
            let mut params_plus = params.clone();
            params_plus[j] += eps;
            self.from_parameters(&params_plus);
            let residuals_plus = self.residuals();

            let mut params_minus = params.clone();
            params_minus[j] -= eps;
            self.from_parameters(&params_minus);
            let residuals_minus = self.residuals();

            for i in 0..n_res {
                jacobian[(i, j)] = (residuals_plus[i] - residuals_minus[i]) / (2.0 * eps);
            }
        }

        // Restore original state
        self.from_parameters(&params);
        jacobian
    }
}

pub fn bundle_adjust(state: &mut SfMState, max_iters: usize) {
    let mut current_params = state.to_parameters();
    let mut current_residuals = state.residuals();
    let mut current_err = current_residuals.norm_squared();
    let mut lambda = 0.001;

    for _ in 0..max_iters {
        let j = state.numerical_jacobian();
        let r = current_residuals.clone();
        
        let jtj = &j.transpose() * &j;
        let jtr = j.transpose() * r;
        
        // Solve (J^T J + lambda * I) * delta = -J^T * r
        let mut lhs = jtj;
        for i in 0..lhs.nrows() {
            lhs[(i, i)] += lambda;
        }
        
        // Use Cholesky or SVD to solve
        let delta = match lhs.clone().cholesky() {
            Some(ch) => ch.solve(&-jtr),
            None => {
                // Fallback to SVD if not positive definite
                lhs.svd(true, true).solve(&-jtr, 1e-9).unwrap()
            }
        };

        let next_params = &current_params + &delta;
        state.from_parameters(&next_params);
        let next_residuals = state.residuals();
        let next_err = next_residuals.norm_squared();

        if next_err < current_err {
            current_params = next_params;
            current_residuals = next_residuals;
            current_err = next_err;
            lambda /= 10.0;
            if delta.norm() < 1e-6 {
                break;
            }
        } else {
            lambda *= 10.0;
            state.from_parameters(&current_params); // Rollback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3, Rotation3};

    #[test]
    fn test_ba_convergence() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let mut state = SfMState::new(intrinsics);

        // Define 2 cameras
        state.cameras.push(CameraExtrinsics::default());
        state.cameras.push(CameraExtrinsics::new(
            Rotation3::from_axis_angle(&Vector3::y_axis(), 0.1).into_inner(),
            Vector3::new(-0.2, 0.0, 0.0),
        ));

        // Define 5 landmarks
        let pts = vec![
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(0.5, 0.5, 5.5),
            Point3::new(-0.5, 0.5, 4.5),
            Point3::new(0.3, -0.4, 6.0),
            Point3::new(-0.2, 0.1, 5.2),
        ];

        for p in pts.iter() {
            let mut lm = LandMark {
                position: *p,
                observations: Vec::new(),
            };
            for cam_idx in 0..2 {
                let cam = &state.cameras[cam_idx];
                let pt_cam = cam.rotation * lm.position.coords + cam.translation;
                let projected = state.intrinsics.project(&Point3::from(pt_cam));
                lm.observations.push((cam_idx, projected));
            }
            // Add noise to initial position
            lm.position += Vector3::new(0.1, -0.1, 0.2);
            state.landmarks.push(lm);
        }

        let initial_err = state.total_reprojection_error();
        println!("Initial error: {}", initial_err);
        
        bundle_adjust(&mut state, 20);
        
        let final_err = state.total_reprojection_error();
        println!("Final error: {}", final_err);
        
        assert!(final_err < initial_err);
        assert!(final_err < 1e-4); // Numerical Jacobian might not be perfect
    }
}
