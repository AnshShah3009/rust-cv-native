use nalgebra::{DVector, DMatrix};
// use std::sync::Arc;

pub mod sparse;

pub trait Factor: Send + Sync {
    fn residual(&self, variables: &[DVector<f64>]) -> DVector<f64>;
    fn jacobian(&self, variables: &[DVector<f64>]) -> Vec<DMatrix<f64>>;
    fn variable_indices(&self) -> Vec<usize>;
}

pub struct FactorGraph {
    pub factors: Vec<Box<dyn Factor>>,
    pub variables: Vec<DVector<f64>>,
}

impl FactorGraph {
    pub fn new() -> Self {
        Self {
            factors: Vec::new(),
            variables: Vec::new(),
        }
    }

    pub fn add_factor(&mut self, factor: Box<dyn Factor>) {
        self.factors.push(factor);
    }

    pub fn add_variable(&mut self, values: DVector<f64>) -> usize {
        let idx = self.variables.len();
        self.variables.push(values);
        idx
    }

    pub fn optimize_lm(&mut self, max_iterations: usize) -> Result<(), String> {
        // Levenberg-Marquardt implementation foundation
        for _ in 0..max_iterations {
            // 1. Build Linear System (Hessian approximation and Gradient)
            // 2. Solve for update
            // 3. Apply update if error decreases
        }
        Ok(())
    }
}
