use nalgebra::DVector;
pub use faer::sparse::Triplet;
use faer::sparse::SparseColMat;
use faer::Mat;
use faer::prelude::Solve;

/// Sparse Matrix representation using Faer as backend.
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub triplets: Vec<Triplet<usize, usize, f64>>,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            triplets: Vec::new(),
        }
    }

    pub fn add(&mut self, row: usize, col: usize, value: f64) {
        self.triplets.push(Triplet::new(row, col, value));
    }

    /// Convert to Faer SparseColMat
    pub fn to_faer(&self) -> SparseColMat<usize, f64> {
        SparseColMat::try_new_from_triplets(
            self.rows,
            self.cols,
            &self.triplets,
        ).expect("Failed to create sparse matrix")
    }
}


pub trait LinearSolver {
    fn solve(&self, A: &SparseMatrix, b: &DVector<f64>) -> Result<DVector<f64>, String>;
}

pub struct CholeskySolver;

impl LinearSolver for CholeskySolver {
    fn solve(&self, A: &SparseMatrix, b: &DVector<f64>) -> Result<DVector<f64>, String> {
        // TEMPORARY: Convert sparse to dense and use dense Cholesky
        // TODO: Implement proper sparse Cholesky or GPU compute shader solver
        // (avoiding CUDA bindings as per user preference)
        
        use faer::Mat;
        use faer::linalg::cholesky::llt;
        
        // Convert sparse matrix to dense
        let mut dense = Mat::zeros(A.rows, A.cols);
        for triplet in &A.triplets {
            *dense.get_mut(triplet.row, triplet.col) = triplet.val;
        }
        
        // Convert b to faer Mat
        let mut b_mat = Mat::zeros(b.len(), 1);
        for i in 0..b.len() {
            *b_mat.get_mut(i, 0) = b[i];
        }
        
        // Use LU decomposition for general linear solve
        // TODO: Implement proper sparse solver or GPU compute shader
        let lu = dense.full_piv_lu();
        let x_faer = lu.solve(b_mat.as_ref());
        
        // Convert back to DVector
        let mut x_vec = Vec::with_capacity(x_faer.nrows());
        for i in 0..x_faer.nrows() {
            x_vec.push(*x_faer.get(i, 0));
        }
        
        Ok(DVector::from_vec(x_vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_solver() {
        // Solve A x = b
        // A = [[2, 0], [0, 2]]
        // b = [2, 4]
        // x = [1, 2]
        
        let mut mat = SparseMatrix::new(2, 2);
        mat.add(0, 0, 2.0);
        mat.add(1, 1, 2.0);
        
        let b = DVector::from_vec(vec![2.0, 4.0]);
        
        let solver = CholeskySolver;
        let x = solver.solve(&mat, &b).expect("Solve failed");
        
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 2.0).abs() < 1e-6);
    }
}
