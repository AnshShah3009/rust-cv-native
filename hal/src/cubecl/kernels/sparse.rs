//! CubeCL sparse linear algebra kernels.
//!
//! Tier 1 per-element kernels for sparse and dense vector operations:
//!
//!   - `spmv`      — sparse matrix-vector multiply (CSR format)
//!   - `dot`       — dense vector dot product (GPU element-wise, CPU reduce)
//!   - `axpy`      — y = alpha * x + y
//!   - `vec_scale` — v = alpha * v
//!
//! # CSR format
//!
//! The SpMV kernel expects standard CSR (Compressed Sparse Row) layout:
//!   - `row_ptr`:     u32 array of length `n_rows + 1`
//!   - `col_indices`:  u32 array of length `nnz`
//!   - `values`:       f32 array of length `nnz`
//!
//! # Reduction strategy
//!
//! CubeCL 0.9 does not expose `Atomic<f32>`, so the `dot` kernel performs
//! per-element multiplication on GPU and final summation on CPU.  This
//! matches the Tier 2 accumulation strategy used in `icp.rs`.

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;

use crate::cubecl::WgpuClient;

// ---------------------------------------------------------------------------
// SpMV — Sparse Matrix-Vector multiply (CSR)
// ---------------------------------------------------------------------------
//
// One thread per row.  Each thread reads its row_ptr range, iterates over
// non-zero entries, and accumulates the dot product with the dense vector x.

#[cube(launch)]
fn spmv_kernel(
    row_ptr: &Array<u32>,
    col_indices: &Array<u32>,
    values: &Array<f32>,
    x: &Array<f32>,
    y: &mut Array<f32>,
    #[comptime] n_rows: usize,
) {
    let row = ABSOLUTE_POS;
    if row < n_rows {
        let start = usize::cast_from(row_ptr[row]);
        let end = usize::cast_from(row_ptr[row + 1]);

        let sum = RuntimeCell::<f32>::new(0.0f32);

        for j in start..end {
            let col = usize::cast_from(col_indices[j]);
            sum.store(sum.read() + values[j] * x[col]);
        }

        y[row] = sum.read();
    }
}

/// Sparse matrix-vector multiply: `y = A * x` where `A` is in CSR format.
///
/// `row_ptr`:     u32 slice of length `n_rows + 1`.
/// `col_indices`: u32 slice of length `nnz`.
/// `values`:      f32 slice of length `nnz`.
/// `x`:           dense f32 vector of length `n_cols`.
///
/// Returns the dense result vector `y` of length `n_rows`.
pub fn spmv(
    client: &WgpuClient,
    row_ptr: &[u32],
    col_indices: &[u32],
    values: &[f32],
    x: &[f32],
) -> Vec<f32> {
    let n_rows = row_ptr.len() - 1;
    let nnz = values.len();

    let rp_h = client.create_from_slice(u32::as_bytes(row_ptr));
    let ci_h = client.create_from_slice(u32::as_bytes(col_indices));
    let val_h = client.create_from_slice(f32::as_bytes(values));
    let x_h = client.create_from_slice(f32::as_bytes(x));
    let y_h = client.empty(n_rows * 4);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_rows, cube_dim);

    unsafe {
        spmv_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<u32>(&rp_h, row_ptr.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&ci_h, nnz, 1),
            ArrayArg::from_raw_parts::<f32>(&val_h, nnz, 1),
            ArrayArg::from_raw_parts::<f32>(&x_h, x.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&y_h, n_rows, 1),
            n_rows,
        )
    }
    .unwrap();

    let bytes = client.read_one(y_h);
    f32::from_bytes(&bytes).to_vec()
}

// ---------------------------------------------------------------------------
// Dot product — per-element multiply on GPU, CPU reduce
// ---------------------------------------------------------------------------

#[cube(launch)]
fn dot_kernel(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>, #[comptime] size: usize) {
    let idx = ABSOLUTE_POS;
    if idx < size {
        out[idx] = a[idx] * b[idx];
    }
}

/// Dense vector dot product: `result = sum(a[i] * b[i])`.
///
/// Element-wise multiplication runs on GPU; final summation on CPU.
/// Both vectors must have the same length.
pub fn dot(client: &WgpuClient, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: vectors must have equal length");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let a_h = client.create_from_slice(f32::as_bytes(a));
    let b_h = client.create_from_slice(f32::as_bytes(b));
    let out_h = client.empty(n * 4);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n, cube_dim);

    unsafe {
        dot_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&a_h, n, 1),
            ArrayArg::from_raw_parts::<f32>(&b_h, n, 1),
            ArrayArg::from_raw_parts::<f32>(&out_h, n, 1),
            n,
        )
    }
    .unwrap();

    let bytes = client.read_one(out_h);
    let products = f32::from_bytes(&bytes);

    // CPU reduction
    products.iter().sum()
}

// ---------------------------------------------------------------------------
// AXPY — y = alpha * x + y
// ---------------------------------------------------------------------------

#[cube(launch)]
fn axpy_kernel(
    x: &Array<f32>,
    y: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] size: usize,
    #[comptime] alpha_u: u32, // alpha * 1_000_000
) {
    let idx = ABSOLUTE_POS;
    if idx < size {
        let alpha = alpha_u as f32 / 1_000_000.0f32;
        out[idx] = alpha * x[idx] + y[idx];
    }
}

/// AXPY: `result[i] = alpha * x[i] + y[i]`.
///
/// Both vectors must have the same length.
pub fn axpy(client: &WgpuClient, alpha: f32, x: &[f32], y: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), y.len(), "axpy: vectors must have equal length");
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }

    let x_h = client.create_from_slice(f32::as_bytes(x));
    let y_h = client.create_from_slice(f32::as_bytes(y));
    let out_h = client.empty(n * 4);

    let alpha_u = (alpha * 1_000_000.0) as u32;

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n, cube_dim);

    unsafe {
        axpy_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&x_h, n, 1),
            ArrayArg::from_raw_parts::<f32>(&y_h, n, 1),
            ArrayArg::from_raw_parts::<f32>(&out_h, n, 1),
            n,
            alpha_u,
        )
    }
    .unwrap();

    let bytes = client.read_one(out_h);
    f32::from_bytes(&bytes).to_vec()
}

// ---------------------------------------------------------------------------
// Vec Scale — v = alpha * v
// ---------------------------------------------------------------------------

#[cube(launch)]
fn vec_scale_kernel(
    v: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] size: usize,
    #[comptime] alpha_u: u32, // alpha * 1_000_000
) {
    let idx = ABSOLUTE_POS;
    if idx < size {
        let alpha = alpha_u as f32 / 1_000_000.0f32;
        out[idx] = alpha * v[idx];
    }
}

/// Scale a vector: `result[i] = alpha * v[i]`.
pub fn vec_scale(client: &WgpuClient, alpha: f32, v: &[f32]) -> Vec<f32> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }

    let v_h = client.create_from_slice(f32::as_bytes(v));
    let out_h = client.empty(n * 4);

    let alpha_u = (alpha * 1_000_000.0) as u32;

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n, cube_dim);

    unsafe {
        vec_scale_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&v_h, n, 1),
            ArrayArg::from_raw_parts::<f32>(&out_h, n, 1),
            n,
            alpha_u,
        )
    }
    .unwrap();

    let bytes = client.read_one(out_h);
    f32::from_bytes(&bytes).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl::get_client;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_spmv_identity() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_spmv_identity");
            return;
        };
        // 3x3 identity matrix in CSR:
        //   row_ptr     = [0, 1, 2, 3]
        //   col_indices = [0, 1, 2]
        //   values      = [1, 1, 1]
        let row_ptr: Vec<u32> = vec![0, 1, 2, 3];
        let col_indices: Vec<u32> = vec![0, 1, 2];
        let values: Vec<f32> = vec![1.0, 1.0, 1.0];
        let x: Vec<f32> = vec![10.0, 20.0, 30.0];

        let y = spmv(&client, &row_ptr, &col_indices, &values, &x);

        assert_eq!(y.len(), 3);
        for i in 0..3 {
            assert!(
                (y[i] - x[i]).abs() < 1e-4,
                "row {i}: expected {}, got {}",
                x[i],
                y[i]
            );
        }
    }

    #[test]
    #[serial]
    fn test_spmv_known_matrix() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_spmv_known_matrix");
            return;
        };
        // Matrix:
        //   [2  0  1]     [1]     [2*1 + 0*2 + 1*3]   [5]
        //   [0  3  0]  *  [2]  =  [0*1 + 3*2 + 0*3] = [6]
        //   [4  0  5]     [3]     [4*1 + 0*2 + 5*3]   [19]
        let row_ptr: Vec<u32> = vec![0, 2, 3, 5];
        let col_indices: Vec<u32> = vec![0, 2, 1, 0, 2];
        let values: Vec<f32> = vec![2.0, 1.0, 3.0, 4.0, 5.0];
        let x: Vec<f32> = vec![1.0, 2.0, 3.0];
        let expected: Vec<f32> = vec![5.0, 6.0, 19.0];

        let y = spmv(&client, &row_ptr, &col_indices, &values, &x);

        assert_eq!(y.len(), 3);
        for i in 0..3 {
            assert!(
                (y[i] - expected[i]).abs() < 1e-4,
                "row {i}: expected {}, got {}",
                expected[i],
                y[i]
            );
        }
    }

    #[test]
    #[serial]
    fn test_dot_product() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_dot_product");
            return;
        };
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        // dot = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let result = dot(&client, &a, &b);
        assert!((result - 70.0).abs() < 1e-3, "expected 70.0, got {result}");
    }

    #[test]
    #[serial]
    fn test_dot_empty() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_dot_empty");
            return;
        };
        let result = dot(&client, &[], &[]);
        assert!((result - 0.0).abs() < 1e-6, "empty dot should be 0.0");
    }

    #[test]
    #[serial]
    fn test_axpy() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_axpy");
            return;
        };
        let x: Vec<f32> = vec![1.0, 2.0, 3.0];
        let y: Vec<f32> = vec![10.0, 20.0, 30.0];
        let alpha = 2.0f32;
        // result = 2*[1,2,3] + [10,20,30] = [12, 24, 36]
        let result = axpy(&client, alpha, &x, &y);

        assert_eq!(result.len(), 3);
        let expected = [12.0f32, 24.0, 36.0];
        for i in 0..3 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-2,
                "index {i}: expected {}, got {}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    #[serial]
    fn test_vec_scale() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_vec_scale");
            return;
        };
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let alpha = 0.5f32;
        // result = 0.5 * [1,2,3,4] = [0.5, 1.0, 1.5, 2.0]
        let result = vec_scale(&client, alpha, &v);

        assert_eq!(result.len(), 4);
        let expected = [0.5f32, 1.0, 1.5, 2.0];
        for i in 0..4 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-2,
                "index {i}: expected {}, got {}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    #[serial]
    fn test_vec_scale_empty() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_vec_scale_empty");
            return;
        };
        let result = vec_scale(&client, 3.0, &[]);
        assert!(result.is_empty(), "scaling empty vec should return empty");
    }
}
