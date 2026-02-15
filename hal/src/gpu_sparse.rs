use wgpu::{Device, Queue, Buffer, BufferUsages};
use wgpu::util::DeviceExt;
use std::sync::Arc;

/// GPU-resident sparse matrix in CSR (Compressed Sparse Row) format.
/// 
/// CSR format stores:
/// - row_ptr: offsets into col_indices/values for each row
/// - col_indices: column index for each non-zero
/// - values: value for each non-zero
pub struct GpuSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize, // number of non-zeros
    
    // GPU buffers
    pub row_ptr_buffer: Buffer,      // size: (rows + 1) * 4 bytes
    pub col_indices_buffer: Buffer,  // size: nnz * 4 bytes
    pub values_buffer: Buffer,       // size: nnz * 4 bytes (f32)
    
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl GpuSparseMatrix {
    /// Upload a sparse matrix from triplet format to GPU in CSR format.
    pub fn from_triplets(
        device: Arc<Device>,
        queue: Arc<Queue>,
        rows: usize,
        cols: usize,
        triplets: &[(usize, usize, f64)], // (row, col, value)
    ) -> Self {
        // Convert triplets to CSR format
        let (row_ptr, col_indices, values) = Self::triplets_to_csr(rows, triplets);
        let nnz = col_indices.len();
        
        // Create GPU buffers
        let row_ptr_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sparse Matrix Row Ptr"),
            contents: bytemuck::cast_slice(&row_ptr),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        let col_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sparse Matrix Col Indices"),
            contents: bytemuck::cast_slice(&col_indices),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        // Convert f64 to f32 for GPU
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sparse Matrix Values"),
            contents: bytemuck::cast_slice(&values_f32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        Self {
            rows,
            cols,
            nnz,
            row_ptr_buffer,
            col_indices_buffer,
            values_buffer,
            device,
            queue,
        }
    }
    
    /// Convert triplets to CSR format (CPU-side).
    fn triplets_to_csr(
        rows: usize,
        triplets: &[(usize, usize, f64)],
    ) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        // Sort triplets by row, then column
        let mut sorted_triplets = triplets.to_vec();
        sorted_triplets.sort_by_key(|(r, c, _)| (*r, *c));
        
        // Build CSR arrays
        let mut row_ptr = vec![0u32; rows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        
        let mut current_row = 0;
        for (row, col, val) in sorted_triplets {
            // Fill row_ptr for empty rows
            while current_row < row {
                current_row += 1;
                row_ptr[current_row] = col_indices.len() as u32;
            }
            
            col_indices.push(col as u32);
            values.push(val);
        }
        
        // Fill remaining row_ptr entries
        while current_row < rows {
            current_row += 1;
            row_ptr[current_row] = col_indices.len() as u32;
        }
        
        (row_ptr, col_indices, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_triplets_to_csr() {
        // Matrix:
        // [1, 0, 2]
        // [0, 3, 0]
        // [4, 0, 5]
        let triplets = vec![
            (0, 0, 1.0),
            (0, 2, 2.0),
            (1, 1, 3.0),
            (2, 0, 4.0),
            (2, 2, 5.0),
        ];
        
        let (row_ptr, col_indices, values) = GpuSparseMatrix::triplets_to_csr(3, &triplets);
        
        assert_eq!(row_ptr, vec![0, 2, 3, 5]);
        assert_eq!(col_indices, vec![0, 2, 1, 0, 2]);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
