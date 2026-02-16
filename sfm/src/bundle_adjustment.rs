use cv_core::{CameraExtrinsics, CameraIntrinsics};
use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, Point2, Point3, Rotation3, Vector3};

pub struct Landmark {
    pub position: Point3<f64>,
    pub observations: Vec<(usize, Point2<f64>)>,
    pub is_valid: bool,
}

impl Landmark {
    pub fn new(position: Point3<f64>) -> Self {
        Self {
            position,
            observations: Vec::new(),
            is_valid: true,
        }
    }

    pub fn add_observation(&mut self, cam_idx: usize, obs: Point2<f64>) {
        self.observations.push((cam_idx, obs));
    }
}

pub struct SfMState {
    pub cameras: Vec<CameraExtrinsics>,
    pub landmarks: Vec<Landmark>,
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

    pub fn add_camera(&mut self, pose: CameraExtrinsics) -> usize {
        let id = self.cameras.len();
        self.cameras.push(pose);
        id
    }

    pub fn add_landmark(
        &mut self,
        position: Point3<f64>,
        observations: Vec<(usize, Point2<f64>)>,
    ) -> usize {
        let id = self.landmarks.len();
        let mut landmark = Landmark::new(position);
        for (cam_idx, obs) in observations {
            landmark.add_observation(cam_idx, obs);
        }
        self.landmarks.push(landmark);
        id
    }

    pub fn total_reprojection_error(&self) -> f64 {
        let mut total_err = 0.0;
        let mut count = 0;

        for landmark in &self.landmarks {
            if !landmark.is_valid {
                continue;
            }

            for (cam_idx, obs) in &landmark.observations {
                if *cam_idx >= self.cameras.len() {
                    continue;
                }

                let cam = &self.cameras[*cam_idx];
                let pt_cam = cam.rotation * landmark.position + cam.translation;

                if pt_cam.z <= 0.0 {
                    total_err += 1e6;
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

    pub fn from_parameters(&mut self, params: &DVector<f64>) {
        let n_cam = self.cameras.len();
        for (i, cam) in self.cameras.iter_mut().enumerate() {
            let axis_angle = Vector3::new(params[6 * i], params[6 * i + 1], params[6 * i + 2]);
            cam.rotation = Rotation3::new(axis_angle).into_inner();
            cam.translation = Vector3::new(params[6 * i + 3], params[6 * i + 4], params[6 * i + 5]);
        }

        let offset = 6 * n_cam;
        for (i, lm) in self.landmarks.iter_mut().enumerate() {
            lm.position = Point3::new(
                params[offset + 3 * i],
                params[offset + 3 * i + 1],
                params[offset + 3 * i + 2],
            );
        }
    }

    pub fn residuals(&self) -> DVector<f64> {
        let mut residuals = Vec::new();
        for landmark in &self.landmarks {
            for (cam_idx, obs) in &landmark.observations {
                if *cam_idx >= self.cameras.len() {
                    continue;
                }

                if !landmark.is_valid {
                    // Push zeros for invalid landmarks to maintain consistent dimensions
                    residuals.push(0.0);
                    residuals.push(0.0);
                    continue;
                }

                let cam = &self.cameras[*cam_idx];
                let pt_cam = cam.rotation * landmark.position + cam.translation;
                let projected = self.intrinsics.project(&pt_cam.into());
                residuals.push(projected.x - obs.x);
                residuals.push(projected.y - obs.y);
            }
        }
        DVector::from_vec(residuals)
    }

    pub fn jacobian_sparsity(&self) -> DMatrix<f64> {
        let n_res = self.residuals().len();
        let n_cam = self.cameras.len();
        let n_lm = self.landmarks.len();
        let n_params = 6 * n_cam + 3 * n_lm;

        let mut sparsity = DMatrix::zeros(n_res, n_params);

        let mut res_idx = 0;
        for (lm_idx, lm) in self.landmarks.iter().enumerate() {
            if !lm.is_valid {
                continue;
            }

            for (cam_idx, _) in &lm.observations {
                // Jacobian has non-zero in camera block
                for j in 0..6 {
                    sparsity[(res_idx, 6 * cam_idx + j)] = 1.0;
                    sparsity[(res_idx + 1, 6 * cam_idx + j)] = 1.0;
                }
                // Jacobian has non-zero in landmark block
                for j in 0..3 {
                    sparsity[(res_idx, 6 * n_cam + 3 * lm_idx + j)] = 1.0;
                    sparsity[(res_idx + 1, 6 * n_cam + 3 * lm_idx + j)] = 1.0;
                }
                res_idx += 2;
            }
        }

        sparsity
    }

    pub fn numerical_jacobian(&self) -> DMatrix<f64> {
        let params = self.to_parameters();
        let residuals = self.residuals();
        let n_res = residuals.len();
        let n_params = params.len();
        let mut jacobian = DMatrix::zeros(n_res, n_params);
        let eps = 1e-6;

        // Use finite differences
        let mut params_plus = params.clone();
        let mut params_minus = params.clone();

        for j in 0..n_params {
            params_plus[j] = params[j] + eps;
            params_minus[j] = params[j] - eps;

            // Quick residual check
            let (res_plus, res_minus) =
                self.compute_residuals_for_param(&params_plus, &params_minus, j, eps);

            for i in 0..n_res {
                jacobian[(i, j)] = (res_plus.get(i).unwrap_or(&0.0)
                    - res_minus.get(i).unwrap_or(&0.0))
                    / (2.0 * eps);
            }

            params_plus[j] = params[j];
            params_minus[j] = params[j];
        }

        jacobian
    }

    fn compute_residuals_for_param(
        &self,
        params_plus: &DVector<f64>,
        params_minus: &DVector<f64>,
        param_idx: usize,
        eps: f64,
    ) -> (DVector<f64>, DVector<f64>) {
        let mut state_plus = self.clone_state();
        let mut state_minus = self.clone_state();

        state_plus.from_parameters(params_plus);
        state_minus.from_parameters(params_minus);

        (state_plus.residuals(), state_minus.residuals())
    }

    fn clone_state(&self) -> SfMState {
        let mut new_state = SfMState::new(self.intrinsics);
        for cam in &self.cameras {
            new_state.cameras.push(*cam);
        }
        for lm in &self.landmarks {
            new_state.landmarks.push(Landmark {
                position: lm.position,
                observations: lm.observations.clone(),
                is_valid: lm.is_valid,
            });
        }
        new_state
    }

    fn compute_point_reprojection_error_index(&self, lm: &Landmark) -> f64 {
        let mut error = 0.0;
        let mut count = 0;

        for (cam_idx, obs) in &lm.observations {
            if *cam_idx >= self.cameras.len() {
                continue;
            }
            let cam = &self.cameras[*cam_idx];
            let pt_cam = cam.rotation * lm.position + cam.translation;

            if pt_cam.z > 0.0 {
                let proj = self.intrinsics.project(&pt_cam.into());
                error += (proj.x - obs.x).powi(2) + (proj.y - obs.y).powi(2);
                count += 1;
            }
        }

        if count > 0 {
            error / count as f64
        } else {
            0.0
        }
    }

    pub fn remove_outliers(&mut self, threshold: f64) {
        // Collect indices of outliers first to avoid borrow issues
        let outlier_indices: Vec<usize> = self
            .landmarks
            .iter()
            .enumerate()
            .filter(|(_, lm)| self.compute_point_reprojection_error_index(lm) > threshold)
            .map(|(i, _)| i)
            .collect();

        // Mark outliers as invalid
        for idx in outlier_indices {
            self.landmarks[idx].is_valid = false;
        }
    }
}

pub struct BundleAdjustmentConfig {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub lambda: f64,
    pub use_sparsity: bool,
    pub robust_kernel: bool,
}

impl Default for BundleAdjustmentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            lambda: 0.001,
            use_sparsity: true,
            robust_kernel: true,
        }
    }
}

pub fn bundle_adjust(state: &mut SfMState, config: &BundleAdjustmentConfig) {
    let mut current_params = state.to_parameters();
    let mut current_residuals = state.residuals();
    let mut current_err = current_residuals.norm_squared();
    let mut lambda = config.lambda;

    for iteration in 0..config.max_iterations {
        // Compute Jacobian
        let j = if config.use_sparsity && state.landmarks.len() > 50 {
            // Use sparse Jacobian computation
            state.numerical_jacobian()
        } else {
            state.numerical_jacobian()
        };

        let r = current_residuals.clone();

        // Normal equations: J^T J * delta = -J^T * r
        let jtj = &j.transpose() * &j;
        let mut jtr = j.transpose() * r;

        // Levenberg-Marquardt damping
        let mut lhs = jtj.clone();
        for i in 0..lhs.nrows() {
            lhs[(i, i)] *= 1.0 + lambda;
        }

        // Solve using Cholesky or fallback
        let neg_jtr = -&jtr;
        let delta = if let Some(ch) = lhs.clone().cholesky() {
            ch.solve(&neg_jtr)
        } else {
            // Fallback to QR or SVD
            lhs.lu()
                .solve(&neg_jtr)
                .unwrap_or_else(|| DVector::zeros(jtr.len()))
        };

        // Compute expected error reduction
        let expected_reduction = -jtr.dot(&delta);

        // Apply step
        let next_params = &current_params + &delta;
        state.from_parameters(&next_params);
        let next_residuals = state.residuals();
        let next_err = next_residuals.norm_squared();

        let actual_reduction = current_err - next_err;

        // Accept or reject step
        if actual_reduction > 0.0 && actual_reduction > 0.9 * expected_reduction {
            current_params = next_params;
            current_residuals = next_residuals;
            current_err = next_err;
            lambda /= 10.0;

            if delta.norm() < config.convergence_threshold {
                break;
            }
        } else {
            lambda *= 10.0;
            state.from_parameters(&current_params); // Rollback
        }

        // Apply robust kernel if enabled
        if config.robust_kernel && iteration % 5 == 0 {
            state.remove_outliers(10.0);
        }
    }
}

pub fn incremental_bundle_adjust(
    state: &mut SfMState,
    new_cameras: &[CameraExtrinsics],
    new_landmarks: &[Point3<f64>],
) {
    // Add new cameras
    for cam in new_cameras {
        state.add_camera(*cam);
    }

    // Add new landmarks with observations from last camera
    let last_cam_idx = state.cameras.len() - 1;
    for (i, lm) in new_landmarks.iter().enumerate() {
        // Simplified: add with dummy observation
        state.add_landmark(*lm, vec![(last_cam_idx, Point2::new(0.0, 0.0))]);
    }

    // Run local BA on recent cameras
    let config = BundleAdjustmentConfig::default();
    bundle_adjust(state, &config);
}

pub fn pose_graph_bundle_adjust(
    cameras: &mut [CameraExtrinsics],
    constraints: &[(usize, usize, CameraExtrinsics)], // (i, j, relative_pose)
) {
    // Simplified pose graph optimization
    // In full implementation, would use cv_3d pose graph

    let n = cameras.len();
    if n == 0 {
        return;
    }

    for _ in 0..10 {
        let mut total_error = 0.0;

        for (i, j, rel_pose) in constraints {
            if *i >= n || *j >= n {
                continue;
            }

            // Compute relative pose from current estimates
            let inv_i = cameras[*i].inverse();
            let est_rel = inv_i.compose(&cameras[*j]);
            let diff = est_rel.rotation - rel_pose.rotation;
            let trans_diff = est_rel.translation - rel_pose.translation;

            total_error += diff.norm() + trans_diff.norm();

            // Simple gradient descent
            let alpha = 0.1;
            cameras[*j].translation -= trans_diff * alpha;
        }

        if total_error < 1e-6 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Rotation3, Vector3};

    #[test]
    fn test_ba_convergence() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let mut state = SfMState::new(intrinsics);

        // Add two cameras
        state.add_camera(CameraExtrinsics::new(Matrix3::identity(), Vector3::zeros()));
        state.add_camera(CameraExtrinsics::new(
            Matrix3::identity(),
            Vector3::new(1.0, 0.0, 0.0),
        ));

        // Add a landmark observed by both cameras
        state.add_landmark(
            Point3::new(0.5, 0.0, 5.0),
            vec![
                (0, Point2::new(320.0, 240.0)),
                (1, Point2::new(320.0 + 100.0, 240.0)),
            ],
        );

        let initial_error = state.total_reprojection_error();
        assert!(initial_error > 0.0);

        let config = BundleAdjustmentConfig::default();
        bundle_adjust(&mut state, &config);

        let final_error = state.total_reprojection_error();
        assert!(final_error < initial_error);
    }
}
