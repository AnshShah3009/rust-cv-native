//! CubeCL point cloud kernels.
//!
//! Tier 1 (per-element, no shared memory):
//!   `transform_points` — vec4 affine transform
//!   `compute_normals_from_covariances` — batch analytic 3×3 eigensolver

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;

use crate::cubecl::WgpuClient;

// ---------------------------------------------------------------------------
// Transform points — vec4 affine (4×4 column-major matrix)
// ---------------------------------------------------------------------------

#[cube(launch)]
fn transform_points_kernel(
    points_in: &Array<f32>,
    matrix: &Array<f32>,
    points_out: &mut Array<f32>,
    #[comptime] num_points: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx < num_points {
        let base = idx * 4;
        let x = points_in[base];
        let y = points_in[base + 1];
        let z = points_in[base + 2];
        let w = points_in[base + 3];

        // column-major 4×4 multiply: out[row] = sum_col(M[col][row] * v[col])
        points_out[base] = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12] * w;
        points_out[base + 1] = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13] * w;
        points_out[base + 2] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14] * w;
        points_out[base + 3] = 1.0f32;
    }
}

/// Transform `points` (f32 slice, stride 4: XYZW) by a 4×4 column-major matrix.
pub fn transform_points(client: &WgpuClient, points: &[f32], matrix: &[f32; 16]) -> Vec<f32> {
    let num_points = points.len() / 4;
    let in_handle = client.create_from_slice(f32::as_bytes(points));
    let mat_handle = client.create_from_slice(f32::as_bytes(matrix.as_slice()));
    let out_handle = client.empty(points.len() * 4);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, num_points, cube_dim);

    unsafe {
        transform_points_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&in_handle, points.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&mat_handle, 16, 1),
            ArrayArg::from_raw_parts::<f32>(&out_handle, points.len(), 1),
            num_points,
        )
    }
    .unwrap();

    let result = client.read_one(out_handle);
    f32::from_bytes(&result).to_vec()
}

// ---------------------------------------------------------------------------
// Analytic 3×3 minimum eigenvector (Open3D / Geometric Tools algorithm)
// ---------------------------------------------------------------------------
//
// CubeCL DSL does not support tuple returns from `#[cube]` fns.
// We write all 3 components of the eigenvector directly into the output buffer
// from within the launch kernel, inlining the eigensolver.

/// Batch analytic minimum eigenvector from `n_points` covariance matrices.
///
/// Input layout per point (8 × f32):
///   `[cxx, cxy, cxz, cyy, cyz, czz, 0, 0]`
///
/// Output layout per point (4 × f32):
///   `[nx, ny, nz, 0]`
#[cube(launch)]
fn batch_pca_kernel(covs: &Array<f32>, normals: &mut Array<f32>, #[comptime] num_points: usize) {
    let idx = ABSOLUTE_POS;
    if idx < num_points {
        let base = idx * 8;
        let cxx = covs[base];
        let cxy = covs[base + 1];
        let cxz = covs[base + 2];
        let cyy = covs[base + 3];
        let cyz = covs[base + 4];
        let czz = covs[base + 5];

        // === Analytic 3×3 minimum eigenvector (Open3D / Geometric Tools) ===
        let max_c = f32::max(
            f32::max(f32::abs(cxx), f32::max(f32::abs(cxy), f32::abs(cxz))),
            f32::max(f32::abs(cyy), f32::max(f32::abs(cyz), f32::abs(czz))),
        );

        let out = idx * 4;
        if max_c < 1e-30f32 {
            normals[out] = 0.0f32;
            normals[out + 1] = 0.0f32;
            normals[out + 2] = 1.0f32;
            normals[out + 3] = 0.0f32;
        } else {
            let s = 1.0f32 / max_c;
            let a00 = cxx * s;
            let a01 = cxy * s;
            let a02 = cxz * s;
            let a11 = cyy * s;
            let a12 = cyz * s;
            let a22 = czz * s;

            let norm = a01 * a01 + a02 * a02 + a12 * a12;
            let q = (a00 + a11 + a22) / 3.0f32;
            let b00 = a00 - q;
            let b11 = a11 - q;
            let b22 = a22 - q;
            let p = f32::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + 2.0f32 * norm) / 6.0f32);

            if p < 1e-10f32 {
                normals[out] = 0.0f32;
                normals[out + 1] = 0.0f32;
                normals[out + 2] = 1.0f32;
                normals[out + 3] = 0.0f32;
            } else {
                let c00 = b11 * b22 - a12 * a12;
                let c01 = a01 * b22 - a12 * a02;
                let c02 = a01 * a12 - b11 * a02;
                let det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);
                let half_det = f32::clamp(det * 0.5f32, -1.0f32, 1.0f32);
                let angle = f32::acos(half_det) / 3.0f32;

                let two_thirds_pi = 2.094_395_1_f32;
                let eval_min = q + p * f32::cos(angle + two_thirds_pi) * 2.0f32;

                // Rows of (A - λ_min * I)
                let r0x = a00 - eval_min;
                let r0y = a01;
                let r0z = a02;
                let r1x = a01;
                let r1y = a11 - eval_min;
                let r1z = a12;
                let r2x = a02;
                let r2y = a12;
                let r2z = a22 - eval_min;

                // Cross products
                let c01x = r0y * r1z - r0z * r1y;
                let c01y = r0z * r1x - r0x * r1z;
                let c01z = r0x * r1y - r0y * r1x;

                let c02x = r0y * r2z - r0z * r2y;
                let c02y = r0z * r2x - r0x * r2z;
                let c02z = r0x * r2y - r0y * r2x;

                let c12x = r1y * r2z - r1z * r2y;
                let c12y = r1z * r2x - r1x * r2z;
                let c12z = r1x * r2y - r1y * r2x;

                let d0 = c01x * c01x + c01y * c01y + c01z * c01z;
                let d1 = c02x * c02x + c02y * c02y + c02z * c02z;
                let d2 = c12x * c12x + c12y * c12y + c12z * c12z;

                let bx = if d0 >= d1 && d0 >= d2 {
                    c01x
                } else if d1 >= d2 {
                    c02x
                } else {
                    c12x
                };
                let by = if d0 >= d1 && d0 >= d2 {
                    c01y
                } else if d1 >= d2 {
                    c02y
                } else {
                    c12y
                };
                let bz = if d0 >= d1 && d0 >= d2 {
                    c01z
                } else if d1 >= d2 {
                    c02z
                } else {
                    c12z
                };

                let blen = f32::sqrt(bx * bx + by * by + bz * bz);

                if blen < 1e-10f32 {
                    normals[out] = 0.0f32;
                    normals[out + 1] = 0.0f32;
                    normals[out + 2] = 1.0f32;
                } else {
                    normals[out] = bx / blen;
                    normals[out + 1] = by / blen;
                    normals[out + 2] = bz / blen;
                }
                normals[out + 3] = 0.0f32;
            }
        }
    }
}

/// Compute normals from pre-computed covariance matrices using analytic eigensolver.
///
/// `covs`: f32 slice, 8 floats per point: `[cxx, cxy, cxz, cyy, cyz, czz, 0, 0]`.
///
/// Returns `Vec<f32>` of length `4 * n_points`: `[nx, ny, nz, 0]` per point.
pub fn compute_normals_from_covariances(client: &WgpuClient, covs: &[f32]) -> Vec<f32> {
    let num_points = covs.len() / 8;
    let out_floats = num_points * 4;

    let cov_handle = client.create_from_slice(f32::as_bytes(covs));
    let norm_handle = client.empty(out_floats * 4);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, num_points, cube_dim);

    unsafe {
        batch_pca_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&cov_handle, covs.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&norm_handle, out_floats, 1),
            num_points,
        )
    }
    .unwrap();

    let result = client.read_one(norm_handle);
    f32::from_bytes(&result).to_vec()
}

// ---------------------------------------------------------------------------
// Morton-sorted kNN covariances → batch PCA normals (Tier 2)
// ---------------------------------------------------------------------------
//
// Pipeline:
//   CPU: Morton-code computation + radix sort → sorted_pts, sorted_indices
//   GPU: range-window kNN covariance accumulation  (this kernel)
//   GPU: batch analytic eigensolver  (reuses batch_pca_kernel above)

/// For each sorted point, accumulate a 3×3 covariance from its k-nearest
/// neighbours found in a Morton-index window.
///
/// `sorted_pts`: f32 slice, stride 4 (x,y,z,0) in Morton order.
/// `window`:     number of points to scan each side in sorted order.
/// Output:       covs f32[8*n] layout: `[cxx,cxy,cxz, cyy,cyz,czz, 0,0]`.
#[cube(launch)]
fn morton_knn_cov_kernel(
    sorted_pts: &Array<f32>,
    covs: &mut Array<f32>,
    #[comptime] n_pts: usize,
    #[comptime] window: usize, // search ±window neighbours in sorted order
) {
    let idx = ABSOLUTE_POS;
    if idx < n_pts {
        let base = idx * 4;
        let px = sorted_pts[base];
        let py = sorted_pts[base + 1];
        let pz = sorted_pts[base + 2];

        let lo = select(idx >= window, idx - window, 0usize);
        let hi = (idx + window + 1).min(n_pts);

        // Accumulate mean and covariance using RuntimeCell
        let sx = RuntimeCell::<f32>::new(0.0f32);
        let sy = RuntimeCell::<f32>::new(0.0f32);
        let sz = RuntimeCell::<f32>::new(0.0f32);
        let cnt = RuntimeCell::<u32>::new(0u32);

        for ni in lo..hi {
            let nb = ni * 4;
            sx.store(sx.read() + sorted_pts[nb]);
            sy.store(sy.read() + sorted_pts[nb + 1]);
            sz.store(sz.read() + sorted_pts[nb + 2]);
            cnt.store(cnt.read() + 1u32);
        }
        let n = f32::cast_from(cnt.read());
        let mx = sx.read() / n;
        let my = sy.read() / n;
        let mz = sz.read() / n;

        // px, py, pz used implicitly via the point we are computing for
        let _ = px + py + pz; // touch to avoid unused-variable errors

        let cxx = RuntimeCell::<f32>::new(0.0f32);
        let cxy = RuntimeCell::<f32>::new(0.0f32);
        let cxz = RuntimeCell::<f32>::new(0.0f32);
        let cyy = RuntimeCell::<f32>::new(0.0f32);
        let cyz = RuntimeCell::<f32>::new(0.0f32);
        let czz = RuntimeCell::<f32>::new(0.0f32);

        for ni in lo..hi {
            let nb = ni * 4;
            let dx = sorted_pts[nb] - mx;
            let dy = sorted_pts[nb + 1] - my;
            let dz = sorted_pts[nb + 2] - mz;
            cxx.store(cxx.read() + dx * dx);
            cxy.store(cxy.read() + dx * dy);
            cxz.store(cxz.read() + dx * dz);
            cyy.store(cyy.read() + dy * dy);
            cyz.store(cyz.read() + dy * dz);
            czz.store(czz.read() + dz * dz);
        }

        let out = idx * 8;
        covs[out] = cxx.read();
        covs[out + 1] = cxy.read();
        covs[out + 2] = cxz.read();
        covs[out + 3] = cyy.read();
        covs[out + 4] = cyz.read();
        covs[out + 5] = czz.read();
        covs[out + 6] = 0.0f32;
        covs[out + 7] = 0.0f32;
    }
}

/// Compute normals via Morton-sorted kNN + analytic eigensolver.
///
/// `points`: f32 slice, stride 4 (x,y,z,0).
/// `k`:      approximate neighbourhood size (window = k/2 in sorted order).
///
/// Returns normals as f32 slice, stride 4 `[nx,ny,nz,0]`, in the ORIGINAL
/// (unsorted) point order.
pub fn compute_normals_morton(
    client: &WgpuClient,
    points: &[f32], // stride 4
    k: usize,
) -> Vec<f32> {
    let n_pts = points.len() / 4;
    if n_pts == 0 {
        return Vec::new();
    }

    // ---- CPU: compute Morton codes ----
    // Normalise points into [0, 1023] grid
    let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
    let (mut max_x, mut max_y, mut max_z) = (f32::MIN, f32::MIN, f32::MIN);
    for i in 0..n_pts {
        let b = i * 4;
        min_x = min_x.min(points[b]);
        min_y = min_y.min(points[b + 1]);
        min_z = min_z.min(points[b + 2]);
        max_x = max_x.max(points[b]);
        max_y = max_y.max(points[b + 1]);
        max_z = max_z.max(points[b + 2]);
    }
    let rx = (max_x - min_x).max(1e-6);
    let ry = (max_y - min_y).max(1e-6);
    let rz = (max_z - min_z).max(1e-6);

    let mut codes: Vec<u32> = (0..n_pts)
        .map(|i| {
            let b = i * 4;
            let gx = (((points[b] - min_x) / rx) * 1023.0) as u32;
            let gy = (((points[b + 1] - min_y) / ry) * 1023.0) as u32;
            let gz = (((points[b + 2] - min_z) / rz) * 1023.0) as u32;
            crate::gpu_kernels::morton_encode(gx, gy, gz)
        })
        .collect();

    // ---- CPU: sort by Morton code (radix sort) ----
    let indices: Vec<u32> = (0..n_pts as u32).collect();
    let (sorted_codes, sorted_indices) =
        crate::cubecl::kernels::sort::radix_sort_by_key(client, &codes, &indices);
    let _ = sorted_codes;

    // Build sorted point array
    let sorted_pts: Vec<f32> = sorted_indices
        .iter()
        .flat_map(|&si| {
            let b = si as usize * 4;
            [points[b], points[b + 1], points[b + 2], 0.0f32]
        })
        .collect();

    // ---- GPU: kNN covariances from Morton window ----
    let spts_h = client.create_from_slice(f32::as_bytes(&sorted_pts));
    let covs_h = client.empty(n_pts * 8 * 4);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_pts, cube_dim);
    let window = k / 2;

    macro_rules! launch_win {
        ($w:expr) => {
            unsafe {
                morton_knn_cov_kernel::launch::<WgpuRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<f32>(&spts_h, sorted_pts.len(), 1),
                    ArrayArg::from_raw_parts::<f32>(&covs_h, n_pts * 8, 1),
                    n_pts,
                    $w,
                )
            }
            .unwrap()
        };
    }
    match window {
        0..=4 => launch_win!(4),
        5..=8 => launch_win!(8),
        9..=16 => launch_win!(16),
        _ => launch_win!(32),
    }

    let cov_bytes = client.read_one(covs_h);
    let covs = f32::from_bytes(&cov_bytes).to_vec();

    // ---- GPU: batch analytic eigensolver ----
    let sorted_normals = compute_normals_from_covariances(client, &covs);

    // ---- CPU: unsort normals back to original order ----
    let mut normals = vec![0.0f32; n_pts * 4];
    for (sorted_idx, &orig_idx) in sorted_indices.iter().enumerate() {
        let src = sorted_idx * 4;
        let dst = orig_idx as usize * 4;
        normals[dst] = sorted_normals[src];
        normals[dst + 1] = sorted_normals[src + 1];
        normals[dst + 2] = sorted_normals[src + 2];
        normals[dst + 3] = 0.0;
    }
    // Drain Morton codes vec (suppress unused warning)
    codes.clear();

    normals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl::get_client;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_transform_points_identity() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_transform_points_identity");
            return;
        };
        let points: Vec<f32> = vec![1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 1.0];
        // Column-major 4×4 identity
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, // col 0
            0.0, 1.0, 0.0, 0.0, // col 1
            0.0, 0.0, 1.0, 0.0, // col 2
            0.0, 0.0, 0.0, 1.0, // col 3
        ];
        let result = transform_points(&client, &points, &identity);
        for i in 0..points.len() {
            assert!(
                (result[i] - points[i]).abs() < 1e-4,
                "mismatch at {i}: {} vs {}",
                result[i],
                points[i]
            );
        }
    }

    #[test]
    #[serial]
    fn test_batch_pca_flat_plane() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_batch_pca_flat_plane");
            return;
        };
        // Flat Z=0 plane: dominant variance X and Y, minimal Z.
        // Cov = diag(1, 1, 0.001) → normal ≈ (0, 0, 1)
        let covs: Vec<f32> = vec![
            1.0f32, 0.0, 0.0, // cxx, cxy, cxz
            1.0, 0.0, 0.001, // cyy, cyz, czz
            0.0, 0.0, // padding
        ];
        let normals = compute_normals_from_covariances(&client, &covs);
        let nz = normals[2].abs();
        assert!(nz > 0.9, "normal z component should be large, got {nz}");
    }

    #[test]
    #[serial]
    fn test_normals_morton_flat_plane() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_normals_morton_flat_plane");
            return;
        };
        // 20 points on Z=0 plane in a 4×5 grid → normals should be ≈ (0,0,±1)
        let mut pts: Vec<f32> = Vec::new();
        for row in 0..4 {
            for col in 0..5 {
                pts.extend_from_slice(&[row as f32, col as f32, 0.0, 0.0]);
            }
        }
        let normals = compute_normals_morton(&client, &pts, 8);
        assert_eq!(normals.len(), 20 * 4);
        for i in 0..20 {
            let nz = normals[i * 4 + 2].abs();
            assert!(nz > 0.8, "point {i}: normal z should be large, got {nz}");
        }
    }
}
