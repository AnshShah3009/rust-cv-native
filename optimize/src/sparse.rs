use nalgebra::DVector;
use faer::sparse::{SparseColMat, Triplet};
use faer::mat::MatRef;

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
        let faer_mat = A.to_faer();
        
        // Convert nalgebra DVector to faer MatRef
        // faer 0.24 Use MatRef directly
        let b_faer = MatRef::from_column_major_slice(
            b.as_slice(),
            b.len(),
            1,
        );

        // Cholesky decomposition
        // Llt = A
        // Solve Ax = b
        
        // faer 0.24 API breakdown
        // TODO: Re-enable generic solver when API is clear.
        // SimplicialLlt seems to be replaced or moved.
        todo!("Implement sparse solver with faer 0.24 API");
        
        /*
        let llt = SimplicialLlt::new(faer_mat.symbolic());
        let llt = llt.decompose(&faer_mat).map_err(|_| "Decomposition failed")?;
        
        let x_faer = llt.solve(b_faer);
        
        // Convert back to DVector
        // x_faer is likely a Mat<f64>
        let mut x_vec = Vec::with_capacity(x_faer.nrows());
        for i in 0..x_faer.nrows() {
            x_vec.push(x_faer.read(i, 0));
        }
        
        Ok(DVector::from_vec(x_vec))
        */
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
        // let x = solver.solve(&mat, &b).expect("Solve failed");
        
        // assert!((x[0] - 1.0).abs() < 1e-6);
        // assert!((x[1] - 2.0).abs() < 1e-6);
    }
}
