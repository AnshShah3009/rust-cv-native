pub mod gpu_solver;
pub mod sparse;
pub mod pose_graph;

use nalgebra::DVector;
use cv_hal::compute::ComputeDevice;
use sparse::{SparseMatrix, LinearSolver, CgSolver};

pub struct SparseLMSolver<'a> {
    pub ctx: &'a ComputeDevice<'a>,
    pub config: LMConfig,
}

pub struct LMConfig {
    pub max_iters: usize,
    pub lambda: f64,
    pub tolerance: f64,
}

impl Default for LMConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            lambda: 0.001,
            tolerance: 1e-6,
        }
    }
}

impl<'a> SparseLMSolver<'a> {
    pub fn new(ctx: &'a ComputeDevice<'a>) -> Self {
        Self {
            ctx,
            config: LMConfig::default(),
        }
    }

    /// Solves the normal equations SparseMatrix * x = b
    pub fn solve(&self, a: &SparseMatrix, b: &DVector<f64>) -> Result<DVector<f64>, String> {
        let solver = CgSolver {
            max_iters: 1000,
            tolerance: self.config.tolerance,
        };
        solver.solve(self.ctx, a, b)
    }
}
