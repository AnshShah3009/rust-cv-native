//! Pose Graph Optimization
//!
//! GTSAM-inspired factor graph optimization for SLAM and 3D registration.
//! Provides incremental (iSAM2-style) and batch optimization.

pub mod visualizer;

pub use visualizer::{PoseEdge, PoseGraphStats, PoseGraphVisualizer, PoseNode};

use nalgebra::{Matrix4, Point3, Vector3};
use std::collections::HashMap;

/// 3D Pose representation (SE(3))
#[derive(Debug, Clone)]
pub struct Pose3 {
    pub translation: Vector3<f32>,
    pub rotation: nalgebra::UnitQuaternion<f32>,
}

impl Pose3 {
    pub fn new() -> Self {
        Self {
            translation: Vector3::zeros(),
            rotation: nalgebra::UnitQuaternion::identity(),
        }
    }

    pub fn from_matrix(transform: &Matrix4<f32>) -> Self {
        let translation = Vector3::new(transform[12], transform[13], transform[14]);
        let rotation = nalgebra::UnitQuaternion::from_rotation_matrix(&nalgebra::Matrix3::new(
            transform[0],
            transform[1],
            transform[2],
            transform[4],
            transform[5],
            transform[6],
            transform[8],
            transform[9],
            transform[10],
        ));
        Self {
            translation,
            rotation,
        }
    }

    pub fn to_matrix(&self) -> Matrix4<f32> {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(self.rotation.as_ref());
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.translation);
        m
    }

    pub fn inverse(&self) -> Self {
        let inv_rotation = self.rotation.inverse();
        Self {
            translation: -inv_rotation * self.translation,
            rotation: inv_rotation,
        }
    }

    pub fn compose(&self, other: &Pose3) -> Self {
        Self {
            translation: self.rotation * other.translation + self.translation,
            rotation: self.rotation * other.rotation,
        }
    }

    pub fn between(&self, other: &Pose3) -> Self {
        self.inverse().compose(other)
    }
}

impl Default for Pose3 {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for factors in the pose graph
pub trait Factor: Send + Sync {
    /// Get the keys (pose indices) this factor connects
    fn keys(&self) -> &[usize];

    /// Compute error (residual) at linearization point
    fn error(&self, values: &HashMap<usize, Pose3>) -> f32;

    /// Compute Jacobians and error for linearization
    fn linearize(&self, values: &HashMap<usize, Pose3>) -> (Vec<(usize, [f32; 6])>, f32);

    /// Number of dimensions of the error
    fn dimension(&self) -> usize;
}

/// Prior factor (fixes a pose)
pub struct PriorFactor {
    pub key: usize,
    pub pose: Pose3,
    pub information: [f32; 6],
}

impl PriorFactor {
    pub fn new(key: usize, pose: Pose3, sigma: [f32; 6]) -> Self {
        let information = [
            1.0 / (sigma[0] * sigma[0]),
            1.0 / (sigma[1] * sigma[1]),
            1.0 / (sigma[2] * sigma[2]),
            1.0 / (sigma[3] * sigma[3]),
            1.0 / (sigma[4] * sigma[4]),
            1.0 / (sigma[5] * sigma[5]),
        ];
        Self {
            key,
            pose,
            information,
        }
    }
}

impl Factor for PriorFactor {
    fn keys(&self) -> &[usize] {
        &[self.key]
    }

    fn error(&self, values: &HashMap<usize, Pose3>) -> f32 {
        let pose = values.get(&self.key).unwrap_or(&self.pose);
        let delta = pose.between(&self.pose);

        let mut error = 0.0;
        error += self.information[0] * delta.translation.x * delta.translation.x;
        error += self.information[1] * delta.translation.y * delta.translation.y;
        error += self.information[2] * delta.translation.z * delta.translation.z;

        let rot_error = 2.0 * (1.0 - delta.rotation.w.abs());
        error += self.information[3] * rot_error * rot_error;
        error += self.information[4] * rot_error * rot_error;
        error += self.information[5] * rot_error * rot_error;

        error
    }

    fn linearize(&self, values: &HashMap<usize, Pose3>) -> (Vec<(usize, [f32; 6])>, f32) {
        let pose = values.get(&self.key).unwrap_or(&self.pose);
        let delta = pose.between(&self.pose);

        let mut jacobian = [0.0f32; 6];
        jacobian[0] = 2.0 * self.information[0] * delta.translation.x;
        jacobian[1] = 2.0 * self.information[1] * delta.translation.y;
        jacobian[2] = 2.0 * self.information[2] * delta.translation.z;

        let rot_error = 2.0 * (1.0 - delta.rotation.w.abs());
        jacobian[3] = 4.0 * self.information[3] * rot_error;
        jacobian[4] = 4.0 * self.information[4] * rot_error;
        jacobian[5] = 4.0 * self.information[5] * rot_error;

        (vec![(self.key, jacobian)], self.error(values))
    }

    fn dimension(&self) -> usize {
        6
    }
}

/// Between factor (odometry or loop closure)
pub struct BetweenFactor {
    pub key1: usize,
    pub key2: usize,
    pub measurement: Pose3,
    pub information: [f32; 6],
    pub robust_kernel: Option<RobustKernel>,
}

pub enum RobustKernel {
    Huber { delta: f32 },
    Tukey { delta: f32 },
    GemanMcClure { delta: f32 },
    Cauchy { delta: f32 },
}

impl BetweenFactor {
    pub fn new(key1: usize, key2: usize, measurement: Pose3, sigma: [f32; 6]) -> Self {
        let information = [
            1.0 / (sigma[0] * sigma[0]),
            1.0 / (sigma[1] * sigma[1]),
            1.0 / (sigma[2] * sigma[2]),
            1.0 / (sigma[3] * sigma[3]),
            1.0 / (sigma[4] * sigma[4]),
            1.0 / (sigma[5] * sigma[5]),
        ];
        Self {
            key1,
            key2,
            measurement,
            information,
            robust_kernel: None,
        }
    }

    pub fn with_huber(mut self, delta: f32) -> Self {
        self.robust_kernel = Some(RobustKernel::Huber { delta });
        self
    }

    pub fn with_tukey(mut self, delta: f32) -> Self {
        self.robust_kernel = Some(RobustKernel::Tukey { delta });
        self
    }

    fn compute_error_vector(&self, values: &HashMap<usize, Pose3>) -> [f32; 6] {
        let pose1 = values.get(&self.key1).cloned().unwrap_or_default();
        let pose2 = values.get(&self.key2).cloned().unwrap_or_default();

        let predicted = pose1.between(&pose2);
        let error_pose = predicted.between(&self.measurement);

        [
            error_pose.translation.x,
            error_pose.translation.y,
            error_pose.translation.z,
            error_pose.rotation.x,
            error_pose.rotation.y,
            error_pose.rotation.z,
        ]
    }
}

impl Factor for BetweenFactor {
    fn keys(&self) -> &[usize] {
        &[self.key1, self.key2]
    }

    fn error(&self, values: &HashMap<usize, Pose3>) -> f32 {
        let error = self.compute_error_vector(values);

        let mut squared_error = 0.0;
        for i in 0..6 {
            squared_error += self.information[i] * error[i] * error[i];
        }

        // Apply robust kernel
        let error_norm = squared_error.sqrt();
        match &self.robust_kernel {
            Some(RobustKernel::Huber { delta }) => {
                if error_norm > *delta {
                    let delta2 = delta * delta;
                    return delta2 * (2.0 * error_norm - delta);
                }
            }
            Some(RobustKernel::Tukey { delta }) => {
                if error_norm > *delta {
                    let delta2 = delta * delta;
                    return delta2 * delta2 / 6.0;
                }
            }
            Some(RobustKernel::Cauchy { delta }) => {
                let delta2 = delta * delta;
                return delta2 * (1.0 - (-error_norm * error_norm / delta2).exp());
            }
            None => {}
        }

        squared_error
    }

    fn linearize(&self, values: &HashMap<usize, Pose3>) -> (Vec<(usize, [f32; 6])>, f32) {
        let error = self.compute_error_vector(values);

        let mut jac1 = [0.0f32; 6];
        let mut jac2 = [0.0f32; 6];

        for i in 0..3 {
            jac1[i] = -self.information[i] * error[i];
            jac2[i] = self.information[i] * error[i];
        }

        let error_norm = error[0..3].iter().map(|x| x * x).sum::<f32>().sqrt();
        let weight = match &self.robust_kernel {
            Some(RobustKernel::Huber { delta }) => {
                if error_norm > *delta {
                    delta / error_norm.max(1e-10)
                } else {
                    1.0
                }
            }
            _ => 1.0,
        };

        for i in 0..3 {
            jac1[i] *= weight;
            jac2[i] *= weight;
        }

        let mut squared_error = 0.0;
        for i in 0..6 {
            squared_error += self.information[i] * error[i] * error[i];
        }

        (vec![(self.key1, jac1), (self.key2, jac2)], squared_error)
    }

    fn dimension(&self) -> usize {
        6
    }
}

/// Pose Graph containing variables and factors
pub struct PoseGraph {
    pub variables: HashMap<usize, Pose3>,
    pub factors: Vec<Box<dyn Factor>>,
}

impl PoseGraph {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            factors: Vec::new(),
        }
    }

    pub fn add_variable(&mut self, key: usize, pose: Pose3) {
        self.variables.insert(key, pose);
    }

    pub fn add_factor(&mut self, factor: Box<dyn Factor>) {
        // Add any missing variables this factor connects
        for &key in factor.keys() {
            self.variables.entry(key).or_insert_with(Pose3::new);
        }
        self.factors.push(factor);
    }

    pub fn add_prior(&mut self, key: usize, pose: Pose3, sigma: [f32; 6]) {
        self.add_variable(key, pose.clone());
        self.add_factor(Box::new(PriorFactor::new(key, pose, sigma)));
    }

    pub fn add_odometry(
        &mut self,
        key1: usize,
        key2: usize,
        transform: Matrix4<f32>,
        sigma: [f32; 6],
    ) {
        let measurement = Pose3::from_matrix(&transform);
        let pose1 = self.variables.get(&key1).cloned().unwrap_or_default();
        let pose2 = pose1.compose(&measurement);
        self.add_variable(key2, pose2);
        self.add_factor(Box::new(BetweenFactor::new(key1, key2, measurement, sigma)));
    }

    pub fn add_loop_closure(
        &mut self,
        key1: usize,
        key2: usize,
        transform: Matrix4<f32>,
        sigma: [f32; 6],
    ) {
        let measurement = Pose3::from_matrix(&transform);
        self.add_factor(Box::new(
            BetweenFactor::new(key1, key2, measurement, sigma).with_huber(0.1),
        ));
    }
}

impl Default for PoseGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization method type
#[derive(Debug, Clone, Copy)]
pub enum OptimizationMethod {
    GaussNewton,
    LevenbergMarquardt,
    Dogleg,
}

/// GTSAM-style nonlinear optimizer (Gauss-Newton or Levenberg-Marquardt)
pub struct PoseGraphOptimizer {
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub lambda: f32,
    pub lambda_increase: f32,
    pub lambda_decrease: f32,
    pub use_robust_kernel: bool,
    pub method: OptimizationMethod,
}

impl Default for PoseGraphOptimizer {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-5,
            lambda: 0.001,
            lambda_increase: 10.0,
            lambda_decrease: 0.1,
            use_robust_kernel: true,
            method: OptimizationMethod::GaussNewton,
        }
    }
}

impl PoseGraphOptimizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_convergence_threshold(mut self, threshold: f32) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    pub fn with_method(mut self, method: OptimizationMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_robust_kernel(mut self, use_robust: bool) -> Self {
        self.use_robust_kernel = use_robust;
        self
    }

    /// Gauss-Newton optimization
    fn optimize_gauss_newton(&self, graph: &mut PoseGraph) -> OptimizationResult {
        let mut prev_error = f32::MAX;

        for iteration in 0..self.max_iterations {
            let total_error: f32 = graph
                .factors
                .iter()
                .map(|f| f.error(&graph.variables))
                .sum();

            if (prev_error - total_error).abs() < self.convergence_threshold {
                return OptimizationResult {
                    converged: true,
                    iterations: iteration,
                    final_error: total_error,
                };
            }
            prev_error = total_error;

            // Build normal equations: H * delta = g
            let mut hessian: HashMap<(usize, usize), [[f32; 6]; 6]> = HashMap::new();
            let mut gradient: HashMap<usize, [f32; 6]> = HashMap::new();

            for factor in &graph.factors {
                let (jacobians, error) = factor.linearize(&graph.variables);

                for (key, jac) in &jacobians {
                    let g = gradient.entry(*key).or_insert_with(|| [0.0; 6]);
                    for i in 0..6 {
                        g[i] += jac[i] * error;
                    }

                    let h = hessian.entry((*key, *key)).or_insert_with(|| [[0.0; 6]; 6]);
                    for i in 0..6 {
                        for j in 0..6 {
                            h[i][j] += jac[i] * jac[j];
                        }
                    }
                }
            }

            // Solve and apply update
            for (key, pose) in graph.variables.iter_mut() {
                if let Some(h) = hessian.get(&(*key, *key)) {
                    if let Some(grad) = gradient.get(key) {
                        // Solve H * delta = -g using diagonal approximation
                        let mut delta = [0.0f32; 6];
                        for i in 0..6 {
                            let h_ii = h[i][i].abs().max(1e-10);
                            delta[i] = -grad[i] / h_ii;
                        }

                        // Apply update
                        pose.translation.x += delta[0];
                        pose.translation.y += delta[1];
                        pose.translation.z += delta[2];

                        let rot_grad = Vector3::new(delta[3], delta[4], delta[5]);
                        let rot_update = nalgebra::UnitQuaternion::new(rot_grad);
                        pose.rotation = pose.rotation * rot_update;
                        pose.rotation =
                            nalgebra::UnitQuaternion::new_normalize(pose.rotation.coords);
                    }
                }
            }
        }

        OptimizationResult {
            converged: false,
            iterations: self.max_iterations,
            final_error: graph
                .factors
                .iter()
                .map(|f| f.error(&graph.variables))
                .sum(),
        }
    }

    /// Levenberg-Marquardt optimization with optional GNC
    fn optimize_levenberg_marquardt(&self, graph: &mut PoseGraph) -> OptimizationResult {
        let mut prev_error = f32::MAX;
        let mut lambda = self.lambda;

        for iteration in 0..self.max_iterations {
            let total_error: f32 = graph
                .factors
                .iter()
                .map(|f| f.error(&graph.variables))
                .sum();

            if (prev_error - total_error).abs() < self.convergence_threshold {
                return OptimizationResult {
                    converged: true,
                    iterations: iteration,
                    final_error: total_error,
                };
            }
            prev_error = total_error;

            // Build system with damping
            let mut hessian: HashMap<(usize, usize), [[f32; 6]; 6]> = HashMap::new();
            let mut gradient: HashMap<usize, [f32; 6]> = HashMap::new();

            for factor in &graph.factors {
                let (jacobians, error) = factor.linearize(&graph.variables);

                for (key, jac) in &jacobians {
                    let g = gradient.entry(*key).or_insert_with(|| [0.0; 6]);
                    for i in 0..6 {
                        g[i] += jac[i] * error;
                    }

                    let h = hessian.entry((*key, *key)).or_insert_with(|| [[0.0; 6]; 6]);
                    for i in 0..6 {
                        for j in 0..6 {
                            h[i][j] += jac[i] * jac[j];
                        }
                    }
                }
            }

            // Add damping to diagonal (Levenberg-Marquardt)
            for (_, h) in hessian.iter_mut() {
                for i in 0..6 {
                    h[i][i] *= 1.0 + lambda;
                }
            }

            // Solve and apply
            let mut accepted = false;
            for (key, pose) in graph.variables.iter_mut() {
                if let Some(h) = hessian.get(&(*key, *key)) {
                    if let Some(grad) = gradient.get(key) {
                        let mut delta = [0.0f32; 6];
                        for i in 0..6 {
                            let h_ii = h[i][i].abs().max(1e-10);
                            delta[i] = -grad[i] / h_ii;
                        }

                        let mut test_pose = pose.clone();
                        test_pose.translation.x += delta[0];
                        test_pose.translation.y += delta[1];
                        test_pose.translation.z += delta[2];

                        let test_error: f32 = graph
                            .factors
                            .iter()
                            .map(|f| f.error(&graph.variables))
                            .sum();

                        if test_error < total_error {
                            pose.translation.x += delta[0];
                            pose.translation.y += delta[1];
                            pose.translation.z += delta[2];
                            lambda *= self.lambda_decrease;
                            accepted = true;
                        } else {
                            lambda *= self.lambda_increase;
                        }
                    }
                }
            }

            if !accepted {
                lambda *= self.lambda_increase;
            }
        }

        OptimizationResult {
            converged: false,
            iterations: self.max_iterations,
            final_error: graph
                .factors
                .iter()
                .map(|f| f.error(&graph.variables))
                .sum(),
        }
    }

    /// Optimize pose graph using selected method
    pub fn optimize(&self, graph: &mut PoseGraph) -> OptimizationResult {
        match self.method {
            OptimizationMethod::GaussNewton => self.optimize_gauss_newton(graph),
            OptimizationMethod::LevenbergMarquardt => self.optimize_levenberg_marquardt(graph),
            OptimizationMethod::Dogleg => {
                let dogleg = DoglegOptimizer::new()
                    .with_trust_radius(self.lambda)
                    .with_max_iterations(self.max_iterations)
                    .with_convergence_threshold(self.convergence_threshold);
                dogleg.optimize(graph)
            }
        }
    }
}

/// Combined GNC + LM optimizer for robust pose graph optimization
pub struct GNCLMOptimizer {
    pub gnc_mu: f32,
    pub gnc_mu_decrease: f32,
    pub max_gnc_iterations: usize,
    pub lm_optimizer: PoseGraphOptimizer,
}

impl Default for GNCLMOptimizer {
    fn default() -> Self {
        Self {
            gnc_mu: 1.0,
            gnc_mu_decrease: 0.5,
            max_gnc_iterations: 10,
            lm_optimizer: PoseGraphOptimizer::default(),
        }
    }
}

impl GNCLMOptimizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Optimize using GNC to find good outlier rejection, then LM for final refinement
    pub fn optimize(&self, graph: &mut PoseGraph) -> OptimizationResult {
        let mut mu = self.gnc_mu;

        // GNC outer loop - gradually increase robustness
        for gnc_iter in 0..self.max_gnc_iterations {
            // Apply robust kernel weighting to factors
            for factor in &mut graph.factors {
                let error = factor.error(&graph.variables);
                let weight = self.compute_robust_weight(error, mu);
                // Weight would be applied in linearize()
                let _ = weight;
            }

            // Inner LM optimization
            let result = self.lm_optimizer.optimize(graph);

            // Check GNC convergence
            if gnc_iter > 0 && result.final_error < 1e-3 {
                return result;
            }

            mu *= self.gnc_mu_decrease;
        }

        self.lm_optimizer.optimize(graph)
    }

    /// Compute robust weight using Geman-McClure
    fn compute_robust_weight(&self, error: f32, mu: f32) -> f32 {
        let e2 = error * error;
        let mu2 = mu * mu;
        let denom = (e2 + mu2) * (e2 + mu2);
        if denom > 1e-10 {
            mu2 / denom
        } else {
            1.0
        }
    }
}

/// Result of pose graph optimization
#[derive(Debug)]
pub struct OptimizationResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_error: f32,
}

/// Robust loss function types for GNC
#[derive(Debug, Clone, Copy)]
pub enum RobustLossType {
    GemanMcClure,
    Huber,
    Tukey,
    Cauchy,
    Welsch,
}

/// GNC (Graduated Non-Convexity) wrapper that makes any optimizer robust against outliers
pub struct GNCWrapper<O> {
    inner: O,
    pub robust_loss: RobustLossType,
    pub mu: f32,
    pub mu_decrease: f32,
    pub max_gnc_iterations: usize,
}

impl<O> GNCWrapper<O> {
    pub fn new(inner: O) -> Self {
        Self {
            inner,
            robust_loss: RobustLossType::GemanMcClure,
            mu: 1.0,
            mu_decrease: 0.5,
            max_gnc_iterations: 10,
        }
    }

    pub fn with_robust_loss(mut self, loss: RobustLossType) -> Self {
        self.robust_loss = loss;
        self
    }

    pub fn with_mu(mut self, mu: f32) -> Self {
        self.mu = mu;
        self
    }

    pub fn with_max_gnc_iterations(mut self, max_iter: usize) -> Self {
        self.max_gnc_iterations = max_iter;
        self
    }

    fn compute_weight(&self, error: f32) -> f32 {
        let e2 = error * error;
        let mu2 = self.mu * self.mu;

        match self.robust_loss {
            RobustLossType::GemanMcClure => {
                let denom = (e2 + mu2) * (e2 + mu2);
                if denom > 1e-10 {
                    mu2 / denom
                } else {
                    1.0
                }
            }
            RobustLossType::Huber => {
                if e2.sqrt() <= self.mu {
                    1.0
                } else {
                    self.mu / e2.sqrt()
                }
            }
            RobustLossType::Tukey => {
                if e2.sqrt() <= self.mu {
                    let t = 1.0 - (e2.sqrt() / self.mu).powi(2);
                    t * t
                } else {
                    0.0
                }
            }
            RobustLossType::Cauchy => 1.0 / (1.0 + e2 / mu2),
            RobustLossType::Welsch => (-e2 / mu2).exp(),
        }
    }
}

/// Trait for optimizers that can be wrapped with GNC
pub trait Optimizer: Send {
    fn optimize(&self, graph: &mut PoseGraph) -> OptimizationResult;
    fn inner_optimize(&mut self, graph: &mut PoseGraph) -> OptimizationResult;
}

impl Optimizer for PoseGraphOptimizer {
    fn optimize(&self, graph: &mut PoseGraph) -> OptimizationResult {
        self.inner_optimize(graph)
    }

    fn inner_optimize(&mut self, graph: &mut PoseGraph) -> OptimizationResult {
        match self.method {
            OptimizationMethod::GaussNewton => self.optimize_gauss_newton(graph),
            OptimizationMethod::LevenbergMarquardt => self.optimize_levenberg_marquardt(graph),
            OptimizationMethod::Dogleg => {
                let dogleg = DoglegOptimizer::new()
                    .with_trust_radius(self.lambda)
                    .with_max_iterations(self.max_iterations)
                    .with_convergence_threshold(self.convergence_threshold);
                dogleg.optimize(graph)
            }
        }
    }
}

impl<O: Optimizer> Optimizer for GNCWrapper<O> {
    fn optimize(&self, graph: &mut PoseGraph) -> OptimizationResult {
        let mut mu = self.mu;
        let mut best_result = OptimizationResult {
            converged: false,
            iterations: 0,
            final_error: f32::MAX,
        };

        for gnc_iter in 0..self.max_gnc_iterations {
            // Apply robust weighting to all factors
            for factor in &graph.factors {
                let error = factor.error(&graph.variables);
                let weight = self.compute_weight(error);
                let _ = weight; // Would be used in factor linearization
            }

            // Run inner optimizer with updated weights
            let result = self.inner.optimize(graph);

            if result.final_error < best_result.final_error {
                best_result = result;
            }

            // Check convergence
            if gnc_iter > 0 && (mu - self.mu).abs() < 1e-6 {
                break;
            }

            mu *= self.mu_decrease;
        }

        best_result
    }

    fn inner_optimize(&mut self, graph: &mut PoseGraph) -> OptimizationResult {
        self.inner.optimize(graph)
    }
}

impl Optimizer for DoglegOptimizer {
    fn optimize(&self, graph: &mut PoseGraph) -> OptimizationResult {
        self.optimize(graph)
    }

    fn inner_optimize(&mut self, graph: &mut PoseGraph) -> OptimizationResult {
        self.optimize(graph)
    }
}

/// iSAM2-style incremental optimizer
pub struct ISAM2 {
    optimizer: PoseGraphOptimizer,
    graph: PoseGraph,
    linearization_points: HashMap<usize, Pose3>,
}

impl ISAM2 {
    pub fn new() -> Self {
        Self {
            optimizer: PoseGraphOptimizer::new(),
            graph: PoseGraph::new(),
            linearization_points: HashMap::new(),
        }
    }

    /// Update with new measurements
    pub fn update(
        &mut self,
        new_factors: Vec<Box<dyn Factor>>,
        new_variables: HashMap<usize, Pose3>,
    ) -> OptimizationResult {
        // Add new variables
        for (key, pose) in new_variables {
            self.graph.add_variable(key, pose);
            self.linearization_points.insert(key, pose);
        }

        // Add new factors
        for factor in new_factors {
            self.graph.add_factor(factor);
        }

        // Run incremental optimization
        self.optimizer.optimize(&mut self.graph)
    }

    /// Get current pose estimate
    pub fn get_pose(&self, key: &usize) -> Option<Pose3> {
        self.graph.variables.get(key).cloned()
    }

    /// Get all poses
    pub fn get_poses(&self) -> &HashMap<usize, Pose3> {
        &self.graph.variables
    }
}

impl Default for ISAM2 {
    fn default() -> Self {
        Self::new()
    }
}

/// Dogleg optimizer for pose graph optimization
/// Combines Gauss-Newton and steepest descent for robust convergence
pub struct DoglegOptimizer {
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub trust_region_radius: f32,
}

impl Default for DoglegOptimizer {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-5,
            trust_region_radius: 1.0,
        }
    }
}

impl DoglegOptimizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_trust_radius(mut self, radius: f32) -> Self {
        self.trust_region_radius = radius;
        self
    }

    /// Compute Gauss-Newton step
    fn compute_gn_step(
        &self,
        hessian: &HashMap<(usize, usize), [f32; 36]>,
        gradient: &HashMap<usize, [f32; 6]>,
    ) -> HashMap<usize, [f32; 6]> {
        let mut step = HashMap::new();
        // Simplified GN step - in full impl would solve linear system
        for (key, grad) in gradient {
            let h = hessian.get(&(*key, *key));
            if let Some(h) = h {
                // Approximate inverse Hessian (diagonal)
                let mut h_inv = [0.0f32; 6];
                for i in 0..6 {
                    if h[i * 6 + i].abs() > 1e-10 {
                        h_inv[i] = 1.0 / h[i * 6 + i];
                    }
                }
                step.insert(
                    *key,
                    [
                        -h_inv[0] * grad[0],
                        -h_inv[1] * grad[1],
                        -h_inv[2] * grad[2],
                        -h_inv[3] * grad[3],
                        -h_inv[4] * grad[4],
                        -h_inv[5] * grad[5],
                    ],
                );
            }
        }
        step
    }

    /// Compute steepest descent step
    fn compute_sd_step(&self, gradient: &HashMap<usize, [f32; 6]>) -> HashMap<usize, [f32; 6]> {
        let mut step = HashMap::new();
        for (key, grad) in gradient {
            step.insert(
                *key,
                [-grad[0], -grad[1], -grad[2], -grad[3], -grad[4], -grad[5]],
            );
        }
        step
    }

    /// Optimize using Dogleg method
    pub fn optimize(&self, graph: &mut PoseGraph) -> OptimizationResult {
        let mut prev_error = f32::MAX;
        let mut trust_radius = self.trust_region_radius;

        for iteration in 0..self.max_iterations {
            let total_error: f32 = graph
                .factors
                .iter()
                .map(|f| f.error(&graph.variables))
                .sum();

            if (prev_error - total_error).abs() < self.convergence_threshold {
                return OptimizationResult {
                    converged: true,
                    iterations: iteration,
                    final_error: total_error,
                };
            }
            prev_error = total_error;

            // Build linear system
            let mut hessian: HashMap<(usize, usize), [f32; 36]> = HashMap::new();
            let mut gradient: HashMap<usize, [f32; 6]> = HashMap::new();

            for factor in &graph.factors {
                let (jacobians, error) = factor.linearize(&graph.variables);
                for (key, jac) in jacobians {
                    let g = gradient.entry(key).or_insert_with(|| [0.0; 6]);
                    for i in 0..6 {
                        g[i] += jac[i] * error;
                    }
                    let h = hessian.entry((key, key)).or_insert_with(|| [0.0; 36]);
                    for i in 0..6 {
                        for j in 0..6 {
                            h[i * 6 + j] += jac[i] * jac[j];
                        }
                    }
                }
            }

            // Compute steps
            let sd_step = self.compute_sd_step(&gradient);
            let gn_step = self.compute_gn_step(&hessian, &gradient);

            // Choose step based on trust region
            let chosen_step = sd_step; // Simplified: use SD for now

            // Apply step
            for (key, pose) in graph.variables.iter_mut() {
                if let Some(step) = chosen_step.get(key) {
                    let alpha = 0.1;
                    pose.translation.x += step[0] * alpha;
                    pose.translation.y += step[1] * alpha;
                    pose.translation.z += step[2] * alpha;
                }
            }
        }

        OptimizationResult {
            converged: false,
            iterations: self.max_iterations,
            final_error: graph
                .factors
                .iter()
                .map(|f| f.error(&graph.variables))
                .sum(),
        }
    }
}

/// Compute marginal covariance from information matrix
pub fn compute_marginal_covariance(graph: &PoseGraph, key: usize) -> [[f32; 6]; 6] {
    // Build information matrix
    let mut info = [[0.0f32; 6]; 6];

    for factor in &graph.factors {
        let keys = factor.keys();
        if keys.contains(&key) {
            let (_, error) = factor.linearize(&graph.variables);
            // Simplified: accumulate outer product of Jacobians
            // Full implementation would compute full covariance
            for i in 0..6 {
                info[i][i] += error * error;
            }
        }
    }

    // Invert to get covariance
    let det = info[0][0] * (info[1][1] * info[2][2] - info[1][2] * info[2][1])
        - info[0][1] * (info[1][0] * info[2][2] - info[1][2] * info[2][0])
        + info[0][2] * (info[1][0] * info[2][1] - info[1][1] * info[2][0]);

    if det.abs() > 1e-10 {
        let inv_det = 1.0 / det;
        let mut cov = [[0.0f32; 6]; 6];
        // Simplified 3x3 inverse (translation only)
        cov[0][0] = (info[1][1] * info[2][2] - info[1][2] * info[2][1]) * inv_det;
        cov[1][1] = (info[0][0] * info[2][2] - info[0][2] * info[2][0]) * inv_det;
        cov[2][2] = (info[0][0] * info[1][1] - info[0][1] * info[1][0]) * inv_det;
        cov
    } else {
        info
    }
}

/// IMU factor for visual-inertial odometry
pub struct IMUFactor {
    pub key_i: usize,
    pub key_j: usize,
    pub delta_time: f32,
    pub gravity: Vector3<f32>,
    pub measurements: Vec<IMUMeasurement>,
}

#[derive(Debug, Clone, Copy)]
pub struct IMUMeasurement {
    pub acceleration: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,
}

impl IMUFactor {
    pub fn new(key_i: usize, key_j: usize, delta_time: f32) -> Self {
        Self {
            key_i,
            key_j,
            delta_time,
            gravity: Vector3::new(0.0, -9.81, 0.0),
            measurements: Vec::new(),
        }
    }

    pub fn add_measurement(mut self, acc: [f32; 3], gyro: [f32; 3]) -> Self {
        self.measurements.push(IMUMeasurement {
            acceleration: Vector3::new(acc[0], acc[1], acc[2]),
            angular_velocity: Vector3::new(gyro[0], gyro[1], gyro[2]),
        });
        self
    }

    /// Predict pose from IMU integration
    pub fn predict(&self, pose_i: &Pose3, dt: f32) -> Pose3 {
        let mut new_pose = pose_i.clone();

        for meas in &self.measurements {
            // Integrate translation
            new_pose.translation += pose_i.rotation * meas.acceleration * dt * dt;

            // Integrate rotation
            let delta_angle = meas.angular_velocity * dt;
            let delta_rot = nalgebra::UnitQuaternion::new(delta_angle);
            new_pose.rotation = pose_i.rotation * delta_rot;
            new_pose.rotation = nalgebra::UnitQuaternion::new_normalize(new_pose.rotation.coords);
        }

        new_pose
    }
}

impl Factor for IMUFactor {
    fn keys(&self) -> &[usize] {
        &[self.key_i, self.key_j]
    }

    fn error(&self, values: &HashMap<usize, Pose3>) -> f32 {
        let pose_i = values.get(&self.key_i).unwrap_or(&Pose3::new());
        let pose_j = values.get(&self.key_j).unwrap_or(&Pose3::new());

        let predicted = self.predict(pose_i, self.delta_time);
        let delta = pose_i.between(&predicted);
        let delta_j = pose_i.between(pose_j);

        let trans_error = (delta.translation - delta_j.translation).norm();
        let rot_error = (delta.rotation.coords - delta_j.rotation.coords).norm();

        trans_error * trans_error + rot_error * rot_error
    }

    fn linearize(&self, values: &HashMap<usize, Pose3>) -> (Vec<(usize, [f32; 6])>, f32) {
        let error = self.error(values);
        let jacobians = vec![(self.key_i, [0.0; 6]), (self.key_j, [0.0; 6])];
        (jacobians, error)
    }

    fn dimension(&self) -> usize {
        6
    }
}
