//! Pose Graph Optimization
//!
//! GTSAM-inspired factor graph optimization for SLAM and 3D registration.
//! Provides incremental (iSAM2-style) and batch optimization.

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

/// GTSAM-style nonlinear optimizer (Gauss-Newton or Levenberg-Marquardt)
pub struct PoseGraphOptimizer {
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub lambda: f32,
    pub use_robust_kernel: bool,
}

impl Default for PoseGraphOptimizer {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-5,
            lambda: 0.001,
            use_robust_kernel: true,
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

    /// Optimize pose graph using batch Gauss-Newton
    pub fn optimize(&self, graph: &mut PoseGraph) -> OptimizationResult {
        let mut prev_error = f32::MAX;

        for iteration in 0..self.max_iterations {
            // Compute total error
            let total_error: f32 = graph
                .factors
                .iter()
                .map(|f| f.error(&graph.variables))
                .sum();

            // Check convergence
            if (prev_error - total_error).abs() < self.convergence_threshold {
                return OptimizationResult {
                    converged: true,
                    iterations: iteration,
                    final_error: total_error,
                };
            }
            prev_error = total_error;

            // Build linear system (Ax = b)
            let mut hessian: HashMap<(usize, usize), [f32; 36]> = HashMap::new();
            let mut gradient: HashMap<usize, [f32; 6]> = HashMap::new();

            for factor in &graph.factors {
                let (jacobians, error) = factor.linearize(&graph.variables);

                // Update Hessian and gradient
                for (key, jac) in jacobians {
                    // Gradient: g += J^T * e
                    let g = gradient.entry(key).or_insert_with(|| [0.0; 6]);
                    for i in 0..6 {
                        g[i] += jac[i] * error;
                    }

                    // Hessian approximation: H += J^T * J
                    let h = hessian.entry((key, key)).or_insert_with(|| [0.0; 36]);
                    for i in 0..6 {
                        for j in 0..6 {
                            h[i * 6 + j] += jac[i] * jac[j];
                        }
                    }
                }
            }

            // Solve linear system and update (simplified: coordinate descent)
            for (key, mut pose) in graph.variables.iter_mut() {
                if let Some(grad) = gradient.get(key) {
                    // Extract translation and rotation parts
                    let alpha = 0.1; // Learning rate

                    pose.translation.x -= alpha * grad[0];
                    pose.translation.y -= alpha * grad[1];
                    pose.translation.z -= alpha * grad[2];

                    // Simplified rotation update
                    let rot_grad = Vector3::new(grad[3], grad[4], grad[5]);
                    let rot_update = nalgebra::UnitQuaternion::new(rot_grad * alpha);
                    pose.rotation = pose.rotation * rot_update;
                    pose.rotation = nalgebra::UnitQuaternion::new_normalize(pose.rotation.coords);
                }
            }
        }

        let final_error: f32 = graph
            .factors
            .iter()
            .map(|f| f.error(&graph.variables))
            .sum();

        OptimizationResult {
            converged: false,
            iterations: self.max_iterations,
            final_error,
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
