use faer::sparse::SparseColMat;
pub use faer::sparse::Triplet;
use nalgebra::DVector;
use cv_hal::compute::ComputeDevice;
use cv_hal::context::ComputeContext;
use cv_core::Tensor;

/// Sparse Matrix representation in CSR format for GPU optimization
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl SparseMatrix {
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[Triplet<usize, usize, f64>]) -> Self {
        // Convert COO (triplets) to CSR
        let mut row_counts = vec![0; rows];
        for t in triplets {
            row_counts[t.row] += 1;
        }

        let mut row_ptr = vec![0; rows + 1];
        for i in 0..rows {
            row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        }

        let mut current_row_pos = vec![0; rows];
        let mut col_indices = vec![0; triplets.len()];
        let mut values = vec![0.0; triplets.len()];

        for t in triplets {
            let pos = row_ptr[t.row] as usize + current_row_pos[t.row];
            col_indices[pos] = t.col as u32;
            values[pos] = t.val;
            current_row_pos[t.row] += 1;
        }

        Self {
            rows,
            cols,
            row_ptr,
            col_indices,
            values,
        }
    }

    pub fn spmv_ctx(&self, ctx: &ComputeDevice, x: &DVector<f64>) -> DVector<f64> {
        // Convert f64 to f32 for GPU SpMV
        let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let values_f32: Vec<f32> = self.values.iter().map(|&v| v as f32).collect();
        
        let x_tensor = Tensor::from_vec(x_f32, cv_core::TensorShape::new(1, x.len(), 1));
        
        // If GPU, upload x
        let result_tensor = match ctx {
            ComputeDevice::Gpu(gpu) => {
                let x_gpu = cv_hal::tensor_ext::TensorToGpu::to_gpu_ctx(&x_tensor, gpu).unwrap();
                ctx.spmv(&self.row_ptr, &self.col_indices, &values_f32, &x_gpu).unwrap()
            },
            _ => ctx.spmv(&self.row_ptr, &self.col_indices, &values_f32, &x_tensor).unwrap(),
        };

        // Download result
        let res_cpu = match ctx {
            ComputeDevice::Gpu(gpu) => cv_hal::tensor_ext::TensorToCpu::to_cpu_ctx(&result_tensor, gpu).unwrap(),
            _ => result_tensor,
        };

        DVector::from_vec(res_cpu.storage.as_slice().unwrap().iter().map(|&v| v as f64).collect())
    }
}

pub trait LinearSolver {
    fn solve(&self, ctx: &ComputeDevice, a: &SparseMatrix, b: &DVector<f64>) -> Result<DVector<f64>, String>;
}

/// Conjugate Gradient solver on GPU/CPU
pub struct CgSolver {
    pub max_iters: usize,
    pub tolerance: f64,
}

impl LinearSolver for CgSolver {
    fn solve(&self, ctx: &ComputeDevice, a: &SparseMatrix, b: &DVector<f64>) -> Result<DVector<f64>, String> {
        let mut x = DVector::zeros(a.cols);
        let mut r = b - a.spmv_ctx(ctx, &x);
        let mut p = r.clone();
        let mut rsold = r.dot(&r);

        for _ in 0..self.max_iters {
            let ap = a.spmv_ctx(ctx, &p);
            let alpha = rsold / p.dot(&ap);
            x += alpha * &p;
            r -= alpha * &ap;
            
            let rsnew = r.dot(&r);
            if rsnew.sqrt() < self.tolerance {
                break;
            }
            p = &r + (rsnew / rsold) * &p;
            rsold = rsnew;
        }

        Ok(x)
    }
}
