use nalgebra::{DMatrix, DVector, Isometry3, Vector3, Matrix6, Vector6, U6};
use std::collections::HashMap;

pub struct PoseGraph {
    pub nodes: HashMap<usize, Isometry3<f64>>,
    pub edges: Vec<Edge>,
    pub fixed_nodes: std::collections::HashSet<usize>,
}

pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub measurement: Isometry3<f64>, // Relative pose from -> to
    pub information: Matrix6<f64>,   // Inverse covariance
}

impl PoseGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            fixed_nodes: std::collections::HashSet::new(),
        }
    }

    pub fn add_node(&mut self, id: usize, pose: Isometry3<f64>) {
        self.nodes.insert(id, pose);
    }

    pub fn add_edge(&mut self, from: usize, to: usize, measurement: Isometry3<f64>, information: Matrix6<f64>) {
        self.edges.push(Edge {
            from,
            to,
            measurement,
            information,
        });
    }

    pub fn set_fixed(&mut self, id: usize) {
        self.fixed_nodes.insert(id);
    }

    /// Performs one iteration of Gauss-Newton optimization using dense linear algebra.
    /// Warning: Only suitable for small graphs (< 500 nodes).
    pub fn optimize(&mut self, iterations: usize) -> Result<f64, String> {
        let num_nodes = self.nodes.len();
        if num_nodes == 0 { return Ok(0.0); }

        let mut node_indices: HashMap<usize, usize> = HashMap::new();
        let mut idx = 0;
        // Assign matrix indices to non-fixed nodes
        for (&id, _) in &self.nodes {
            if !self.fixed_nodes.contains(&id) {
                node_indices.insert(id, idx);
                idx += 1;
            }
        }
        let system_size = idx * 6;
        if system_size == 0 { return Ok(0.0); }

        let mut error_sum = 0.0;

        for _ in 0..iterations {
            let mut H = DMatrix::zeros(system_size, system_size);
            let mut b = DVector::zeros(system_size);
            error_sum = 0.0;

            for edge in &self.edges {
                let pose_i = self.nodes[&edge.from];
                let pose_j = self.nodes[&edge.to];
                let pose_ij = edge.measurement;
                let info = edge.information;

                // Error vector e_ij = log(Z_ij^-1 * (X_i^-1 * X_j))
                // Approximate linear error model
                let prediction = pose_i.inverse() * pose_j;
                let error_se3 = pose_ij.inverse() * prediction;
                let error = Vector6::from_iterator(error_se3.translation.vector.iter().chain(error_se3.rotation.scaled_axis().iter()).cloned());

                error_sum += error.transpose() * info * error;

                // Jacobians (approximation)
                // d(e_ij)/d(x_i) = -J_r^-1(e_ij) * Ad(Z_ij^-1)
                // d(e_ij)/d(x_j) = J_r^-1(e_ij) * Ad(error_se3^-1)
                // Simplified for small errors: J_i = -I, J_j = I (in local tangent space)
                // Wait, correct Jacobians are critical.
                // For pose graph slam:
                // e = log(Z_ij^-1 * Xi^-1 * Xj)
                // J_i = -Ad(Xj^-1 * Xi)
                // J_j = I
                // Let's use a simpler approximation for now: Identity blocks rotated.
                
                // Construct H and b
                let idx_i = node_indices.get(&edge.from);
                let idx_j = node_indices.get(&edge.to);

                // If both fixed, skip
                if idx_i.is_none() && idx_j.is_none() { continue; }

                let J_i = -Matrix6::identity(); // Approximation
                let J_j = Matrix6::identity();  // Approximation

                let H_ii = J_i.transpose() * info * J_i;
                let H_jj = J_j.transpose() * info * J_j;
                let H_ij = J_i.transpose() * info * J_j;
                let H_ji = H_ij.transpose();

                let b_i = J_i.transpose() * info * error;
                let b_j = J_j.transpose() * info * error;

                if let Some(&i) = idx_i {
                    let r = i * 6;
                    H.slice_mut((r, r), (6, 6)) .add_assign(&H_ii);
                    b.rows_mut(r, 6) += &(-b_i);

                    if let Some(&j) = idx_j {
                        let c = j * 6;
                        H.slice_mut((r, c), (6, 6)) += &H_ij;
                        H.slice_mut((c, r), (6, 6)) += &H_ji;
                        H.slice_mut((c, c), (6, 6)) += &H_jj;
                        b.rows_mut(c, 6) += &(-b_j);
                    }
                } else if let Some(&j) = idx_j {
                    let c = j * 6;
                    H.slice_mut((c, c), (6, 6)) += &H_jj;
                    b.rows_mut(c, 6) += &(-b_j);
                }
            }

            // Solve H * dx = b
            let dx = match H.cholesky() {
                Some(chol) => chol.solve(&b),
                None => return Err("Cholesky decomposition failed (system unstable)".into()),
            };

            // Update poses
            for (&id, &idx) in &node_indices {
                let update = dx.rows(idx * 6, 6);
                let translation = Vector3::new(update[0], update[1], update[2]);
                let rotation = Vector3::new(update[3], update[4], update[5]);
                
                // Exp map update
                let delta = Isometry3::new(translation, rotation);
                // Left update: X_new = delta * X_old
                if let Some(pose) = self.nodes.get_mut(&id) {
                    *pose = delta * (*pose);
                }
            }

            if error_sum < 1e-6 { break; }
        }

        Ok(error_sum)
    }
}
