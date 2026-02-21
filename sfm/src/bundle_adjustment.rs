use cv_core::{Pose, CameraIntrinsics};
use nalgebra::{DMatrix, DVector, Point2, Point3, Rotation3, Vector3};
use rayon::prelude::*;
use cv_runtime::orchestrator::{ResourceGroup, scheduler};

use cv_optimize::sparse::{SparseMatrix, Triplet};

#[derive(Clone)]
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

#[derive(Clone)]
pub struct SfMState {
    pub cameras: Vec<Pose>,
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

    pub fn add_camera(&mut self, pose: Pose) -> usize {
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

    pub fn numerical_jacobian(&self) -> DMatrix<f64> {
        if let Ok(s) = scheduler() {
            if let Ok(group) = s.get_default_group() {
                return self.numerical_jacobian_ctx(&group);
            }
        }
        
        // Fallback to sequential execution if scheduler fails
        let params = self.to_parameters();
        let n_res = self.residuals().len();
        let n_params = params.len();
        let eps = 1e-6;

        let mut jacobian_data = vec![0.0; n_res * n_params];
        for j in 0..n_params {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[j] += eps;
            params_minus[j] -= eps;

            let (res_plus, res_minus) = self.compute_residuals_for_param(&params_plus, &params_minus);
            for i in 0..n_res {
                jacobian_data[j * n_res + i] = (res_plus.get(i).unwrap_or(&0.0) - res_minus.get(i).unwrap_or(&0.0)) / (2.0 * eps);
            }
        }
        DMatrix::from_vec(n_res, n_params, jacobian_data)
    }

    pub fn numerical_jacobian_ctx(&self, group: &ResourceGroup) -> DMatrix<f64> {
        let params = self.to_parameters();
        let n_res = self.residuals().len();
        let n_params = params.len();
        let eps = 1e-6;

        let jacobian_data: Vec<f64> = group.run(|| {
            (0..n_params).into_par_iter().flat_map(|j| {
                let mut params_plus = params.clone();
                let mut params_minus = params.clone();
                params_plus[j] += eps;
                params_minus[j] -= eps;

                let (res_plus, res_minus) = self.compute_residuals_for_param(&params_plus, &params_minus);
                
                let mut col = vec![0.0; n_res];
                for i in 0..n_res {
                    col[i] = (res_plus.get(i).unwrap_or(&0.0) - res_minus.get(i).unwrap_or(&0.0)) / (2.0 * eps);
                }
                col
            }).collect()
        });

        // DMatrix is column-major by default in nalgebra for from_vec
        DMatrix::from_vec(n_res, n_params, jacobian_data)
    }

    pub fn numerical_jacobian_sparse(&self) -> SparseMatrix {
        let params = self.to_parameters();
        let n_res = self.residuals().len();
        let n_params = params.len();
        let n_cam = self.cameras.len();
        let eps = 1e-6;

        let mut triplets = Vec::new();

        let mut res_idx = 0;
        for (lm_idx, lm) in self.landmarks.iter().enumerate() {
            if !lm.is_valid { continue; }
            for (cam_idx, _obs) in &lm.observations {
                // Camera block (6 params)
                for k in 0..6 {
                    let mut p_perturbed = params.clone();
                    p_perturbed[6 * cam_idx + k] += eps;
                    let (res_plus, _) = self.compute_residuals_for_param_local(&p_perturbed, *cam_idx, lm_idx);
                    
                    let base_pt = self.cameras[*cam_idx].rotation * lm.position + self.cameras[*cam_idx].translation;
                    let base_proj = self.intrinsics.project(&base_pt.into());
                    
                    triplets.push(Triplet::new(res_idx, 6 * cam_idx + k, (res_plus.x - base_proj.x) / eps));
                    triplets.push(Triplet::new(res_idx + 1, 6 * cam_idx + k, (res_plus.y - base_proj.y) / eps));
                }

                // Landmark block (3 params)
                let offset = 6 * n_cam;
                for k in 0..3 {
                    let mut p_perturbed = params.clone();
                    p_perturbed[offset + 3 * lm_idx + k] += eps;
                    let (res_plus, _) = self.compute_residuals_for_param_local(&p_perturbed, *cam_idx, lm_idx);
                    
                    let base_pt = self.cameras[*cam_idx].rotation * lm.position + self.cameras[*cam_idx].translation;
                    let base_proj = self.intrinsics.project(&base_pt.into());

                    triplets.push(Triplet::new(res_idx, offset + 3 * lm_idx + k, (res_plus.x - base_proj.x) / eps));
                    triplets.push(Triplet::new(res_idx + 1, offset + 3 * lm_idx + k, (res_plus.y - base_proj.y) / eps));
                }
                res_idx += 2;
            }
        }

        SparseMatrix::from_triplets(n_res, n_params, &triplets)
    }

    fn compute_residuals_for_param_local(&self, params: &DVector<f64>, cam_idx: usize, lm_idx: usize) -> (Point2<f64>, Point2<f64>) {
        let axis_angle = Vector3::new(params[6 * cam_idx], params[6 * cam_idx + 1], params[6 * cam_idx + 2]);
        let rot = Rotation3::new(axis_angle).into_inner();
        let trans = Vector3::new(params[6 * cam_idx + 3], params[6 * cam_idx + 4], params[6 * cam_idx + 5]);
        
        let n_cam = self.cameras.len();
        let offset = 6 * n_cam;
        let lm_pos = Point3::new(params[offset + 3 * lm_idx], params[offset + 3 * lm_idx + 1], params[offset + 3 * lm_idx + 2]);
        
        let pt_cam = rot * lm_pos + trans;
        let proj = self.intrinsics.project(&pt_cam.into());
        (proj, proj) // Dummy second for signature parity
    }

    fn compute_residuals_for_param(
        &self,
        params_plus: &DVector<f64>,
        params_minus: &DVector<f64>,
    ) -> (DVector<f64>, DVector<f64>) {
        let mut state_plus = self.clone();
        let mut state_minus = self.clone();

        state_plus.from_parameters(params_plus);
        state_minus.from_parameters(params_minus);

        (state_plus.residuals(), state_minus.residuals())
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
        let outlier_indices: Vec<usize> = self
            .landmarks
            .iter()
            .enumerate()
            .filter(|(_, lm)| self.compute_point_reprojection_error_index(lm) > threshold)
            .map(|(i, _)| i)
            .collect();

        for idx in outlier_indices {
            self.landmarks[idx].is_valid = false;
        }
    }
}

use cv_optimize::{CostFunction, SparseLMSolver};

impl CostFunction for SfMState {
    fn dimensions(&self) -> (usize, usize) {
        let n_res = self.landmarks.iter().filter(|l| l.is_valid).map(|l| l.observations.len() * 2).sum();
        let n_params = 6 * self.cameras.len() + 3 * self.landmarks.len();
        (n_res, n_params)
    }

    fn residuals(&self, params: &DVector<f64>) -> DVector<f64> {
        let mut temp_state = self.clone();
        temp_state.from_parameters(params);
        temp_state.residuals()
    }

    fn jacobian(&self, params: &DVector<f64>) -> SparseMatrix {
        let mut temp_state = self.clone();
        temp_state.from_parameters(params);
        temp_state.numerical_jacobian_sparse()
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
    if let Ok(s) = scheduler() {
        if let Ok(group) = s.get_default_group() {
            bundle_adjust_ctx(state, config, &group);
            return;
        }
    }
    
    // Fallback: Use a temporary dummy group or implement a sequential version of bundle_adjust_ctx.
    // For now, since bundle_adjust_ctx doesn't use the group for much other than calling numerical_jacobian_ctx,
    // we can implement a basic loop.
    let mut current_params = state.to_parameters();
    let mut current_residuals = state.residuals();
    let mut current_err = current_residuals.norm_squared();
    let mut lambda = config.lambda;

    for iteration in 0..config.max_iterations {
        // Use the sequential version of numerical_jacobian() which handles its own fallback
        let j = state.numerical_jacobian();
        let r = current_residuals.clone();

        let jtj = &j.transpose() * &j;
        let jtr = j.transpose() * r;

        let mut lhs = jtj.clone();
        for i in 0..lhs.nrows() {
            lhs[(i, i)] *= 1.0 + lambda;
        }

        let neg_jtr = -&jtr;
        let delta = if let Some(ch) = lhs.clone().cholesky() {
            ch.solve(&neg_jtr)
        } else {
            lhs.lu().solve(&neg_jtr).unwrap_or_else(|| DVector::zeros(jtr.len()))
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
            if delta.norm() < config.convergence_threshold { break; }
        } else {
            lambda *= 10.0;
            state.from_parameters(&current_params);
        }

        if config.robust_kernel && iteration % 5 == 0 {
            state.remove_outliers(10.0);
        }
    }
}

pub fn bundle_adjust_ctx(state: &mut SfMState, config: &BundleAdjustmentConfig, group: &ResourceGroup) {
    let device = group.device();
    let solver = SparseLMSolver {
        ctx: &device,
        config: cv_optimize::LMConfig {
            max_iters: config.max_iterations,
            lambda: config.lambda,
            tolerance: config.convergence_threshold,
        },
    };

    let initial_params = state.to_parameters();
    if let Ok(final_params) = solver.minimize(state, initial_params) {
        state.from_parameters(&final_params);
    }
}
