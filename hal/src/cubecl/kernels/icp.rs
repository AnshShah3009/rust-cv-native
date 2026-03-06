//! CubeCL ICP (Iterative Closest Point) kernels.
//!
//! Tier 2 — moderate complexity:
//!   `correspondences` — brute-force nearest-neighbour with max-distance threshold
//!   `dense_step`      — per-pixel depth ICP (point-to-plane), returns JtJ + Jtb
//!
//! # CPU-side accumulation strategy
//!
//! The JtJ/Jtb accumulation requires atomic f32 adds. CubeCL 0.9 exposes
//! `Atomic<u32>` (integer), not `Atomic<f32>`.  We therefore perform
//! accumulation on CPU after reading correspondences back from GPU.
//! A full GPU reduction will be added in the Tier 3 pass using shared memory.

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;

use crate::cubecl::WgpuClient;

// ---------------------------------------------------------------------------
// Correspondences — brute-force 1NN with max-distance threshold
// ---------------------------------------------------------------------------
//
// Source points: f32 slice, stride 3 (x,y,z per point).
// Target points: f32 slice, stride 3.
// Transform:     column-major f32[16] 4×4 matrix applied to source.
//
// Output corr:  u32 array [n_src] — target index, or 0xFFFF_FFFF if no match.
// Output dist:  f32 array [n_src] — sqrt distance, or -1.0 if no match.

#[cube(launch)]
fn correspondences_kernel(
    src: &Array<f32>,
    tgt: &Array<f32>,
    transform: &Array<f32>,
    corr: &mut Array<u32>,
    dist: &mut Array<f32>,
    #[comptime] n_src: usize,
    #[comptime] n_tgt: usize,
    #[comptime] max_dist_sq_u: u32, // max_dist_sq * 1_000_000
) {
    let si = ABSOLUTE_POS;
    if si < n_src {
        let sb = si * 3;
        let sx = src[sb];
        let sy = src[sb + 1];
        let sz = src[sb + 2];

        // Apply 4×4 transform (column-major)
        let tx = transform[0] * sx + transform[4] * sy + transform[8] * sz + transform[12];
        let ty = transform[1] * sx + transform[5] * sy + transform[9] * sz + transform[13];
        let tz = transform[2] * sx + transform[6] * sy + transform[10] * sz + transform[14];

        let best_idx = RuntimeCell::<u32>::new(0xFFFF_FFFFu32);
        let max_dsq = max_dist_sq_u as f32 / 1_000_000.0f32;
        let best_dsq = RuntimeCell::<f32>::new(max_dsq + 1.0f32);

        for ti in 0usize..n_tgt {
            let tb = ti * 3;
            let dx = tx - tgt[tb];
            let dy = ty - tgt[tb + 1];
            let dz = tz - tgt[tb + 2];
            let dsq = dx * dx + dy * dy + dz * dz;
            if dsq < best_dsq.read() {
                best_dsq.store(dsq);
                best_idx.store(u32::cast_from(ti));
            }
        }

        if best_dsq.read() <= max_dsq {
            corr[si] = best_idx.read();
            dist[si] = f32::sqrt(best_dsq.read());
        } else {
            corr[si] = 0xFFFF_FFFFu32;
            dist[si] = -1.0f32;
        }
    }
}

/// Brute-force ICP correspondences with transform.
///
/// `src` / `tgt`: f32 slices, stride 3 (x,y,z).
/// `transform`: column-major f32[16].
/// `max_dist`: maximum correspondence distance.
///
/// Returns `(corr_indices, distances)` where `corr_indices[i] = 0xFFFFFFFF`
/// means no valid correspondence.
pub fn icp_correspondences(
    client: &WgpuClient,
    src: &[f32],
    tgt: &[f32],
    transform: &[f32; 16],
    max_dist: f32,
) -> (Vec<u32>, Vec<f32>) {
    let n_src = src.len() / 3;
    let n_tgt = tgt.len() / 3;

    let src_h = client.create_from_slice(f32::as_bytes(src));
    let tgt_h = client.create_from_slice(f32::as_bytes(tgt));
    let xfm_h = client.create_from_slice(f32::as_bytes(transform.as_slice()));
    let cor_h = client.empty(n_src * 4);
    let dst_h = client.empty(n_src * 4);

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_src, cube_dim);

    let max_dist_sq = max_dist * max_dist;
    let max_dist_sq_u = (max_dist_sq * 1_000_000.0) as u32;
    unsafe {
        correspondences_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&src_h, src.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&tgt_h, tgt.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&xfm_h, 16, 1),
            ArrayArg::from_raw_parts::<u32>(&cor_h, n_src, 1),
            ArrayArg::from_raw_parts::<f32>(&dst_h, n_src, 1),
            n_src,
            n_tgt,
            max_dist_sq_u,
        )
    }
    .unwrap();

    let cb = client.read_one(cor_h);
    let db = client.read_one(dst_h);
    (u32::from_bytes(&cb).to_vec(), f32::from_bytes(&db).to_vec())
}

// ---------------------------------------------------------------------------
// Dense depth ICP step — per-pixel point-to-plane
// ---------------------------------------------------------------------------
//
// Each thread computes the contribution of one depth pixel to JtJ and Jtb.
// Because we cannot do f32 atomic accumulation on GPU in CubeCL 0.9, we
// compute per-pixel JtJ rows and accumulate on CPU. This is a 2-pass approach:
//   Pass 1 (GPU): compute per-pixel 6-element jacobian row `j[6]` and residual `r`
//   Pass 2 (CPU): compute JtJ (21 unique elements) and Jtb (6 elements)
//
// The dense step returns 7 floats per pixel: [j0..j5, residual].
// The caller accumulates the 6×6 JtJ + 6×1 Jtb and solves the linear system.

#[cube(launch)]
fn dense_icp_jacobians_kernel(
    depth_prev: &Array<f32>,
    depth_curr: &Array<f32>,
    normals: &Array<f32>,  // stride-4 (nx, ny, nz, 0)
    intrinsics: &Array<f32>, // [fx, fy, cx, cy]
    transform: &Array<f32>, // col-major f32[16]
    out: &mut Array<f32>,  // [n_pixels * 7] → [j0..j5, residual] per pixel
    #[comptime] w: usize,
    #[comptime] h: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < w * h {
        let out_base = pos * 7;
        let px = f32::cast_from(pos % w);
        let py = f32::cast_from(pos / w);

        let d_prev = depth_prev[pos];
        let d_curr = depth_curr[pos];

        // Mark invalid pixels
        if d_prev <= 0.0f32 || d_curr <= 0.0f32 {
            for k in 0usize..7usize {
                out[out_base + k] = 0.0f32;
            }
        } else {
            let fx = intrinsics[0];
            let fy = intrinsics[1];
            let cx = intrinsics[2];
            let cy = intrinsics[3];

            // Back-project prev pixel to 3D
            let x = (px - cx) / fx * d_prev;
            let y = (py - cy) / fy * d_prev;
            let z = d_prev;

            // Apply transform
            let tx = transform[0] * x + transform[4] * y + transform[8] * z + transform[12];
            let ty = transform[1] * x + transform[5] * y + transform[9] * z + transform[13];
            let tz = transform[2] * x + transform[6] * y + transform[10] * z + transform[14];

            // Project to current frame
            let u_proj = fx * tx / tz + cx;
            let v_proj = fy * ty / tz + cy;

            let ui = usize::cast_from(f32::round(u_proj) as u32);
            let vi = usize::cast_from(f32::round(v_proj) as u32);

            if ui < w && vi < h {
                let curr_d = depth_curr[vi * w + ui];
                let ni_base = (vi * w + ui) * 4;
                let nx = normals[ni_base];
                let ny = normals[ni_base + 1];
                let nz = normals[ni_base + 2];

                // Point-to-plane residual: n · (p_curr - p_transformed)
                let cx_proj = (u_proj - cx) / fx * curr_d;
                let cy_proj = (v_proj - cy) / fy * curr_d;
                let cz_proj = curr_d;

                let residual = nx * (cx_proj - tx) + ny * (cy_proj - ty) + nz * (cz_proj - tz);

                // Jacobian: J = [n^T, (p_t × n)^T]
                let cpx = ty * nz - tz * ny;
                let cpy = tz * nx - tx * nz;
                let cpz = tx * ny - ty * nx;

                out[out_base] = nx;
                out[out_base + 1] = ny;
                out[out_base + 2] = nz;
                out[out_base + 3] = cpx;
                out[out_base + 4] = cpy;
                out[out_base + 5] = cpz;
                out[out_base + 6] = residual;
            } else {
                for k in 0usize..7usize {
                    out[out_base + k] = 0.0f32;
                }
            }
        }
    }
}

/// Dense ICP step: compute per-pixel Jacobians for point-to-plane cost.
///
/// Returns `(jtj_upper, jtb)` where:
/// - `jtj_upper`: 21 f32 (upper triangle of 6×6 symmetric matrix)
/// - `jtb`: 6 f32 (right-hand side vector)
///
/// Jacobians computed on GPU; accumulation done on CPU (Tier 2 approach).
/// For a fully GPU-accelerated path, use a parallel reduction in Tier 3.
pub fn dense_icp_step(
    client: &WgpuClient,
    depth_prev: &[f32],
    depth_curr: &[f32],
    normals: &[f32], // stride-4 per pixel
    intrinsics: &[f32; 4],
    transform: &[f32; 16],
    width: usize,
    height: usize,
) -> ([f32; 21], [f32; 6]) {
    let n_pixels = width * height;
    let out_floats = n_pixels * 7;

    let dp_h = client.create_from_slice(f32::as_bytes(depth_prev));
    let dc_h = client.create_from_slice(f32::as_bytes(depth_curr));
    let nm_h = client.create_from_slice(f32::as_bytes(normals));
    let in_h = client.create_from_slice(f32::as_bytes(intrinsics.as_slice()));
    let xf_h = client.create_from_slice(f32::as_bytes(transform.as_slice()));
    let out_h = client.empty(out_floats * 4);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_pixels, cube_dim);

    unsafe {
        dense_icp_jacobians_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&dp_h, n_pixels, 1),
            ArrayArg::from_raw_parts::<f32>(&dc_h, n_pixels, 1),
            ArrayArg::from_raw_parts::<f32>(&nm_h, normals.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&in_h, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&xf_h, 16, 1),
            ArrayArg::from_raw_parts::<f32>(&out_h, out_floats, 1),
            width,
            height,
        )
    }
    .unwrap();

    let bytes = client.read_one(out_h);
    let jacs = f32::from_bytes(&bytes);

    // CPU accumulation of JtJ and Jtb
    let mut jtj = [0.0f32; 21];
    let mut jtb = [0.0f32; 6];

    for i in 0..n_pixels {
        let base = i * 7;
        let j = &jacs[base..base + 6];
        let r = jacs[base + 6];
        if r == 0.0 {
            continue;
        }

        // Upper triangle of 6×6 JtJ
        let mut k = 0usize;
        for row in 0usize..6usize {
            for col in row..6usize {
                jtj[k] += j[row] * j[col];
                k += 1;
            }
        }
        for row in 0usize..6usize {
            jtb[row] -= j[row] * r; // negative because we minimise
        }
    }

    (jtj, jtb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl::get_client;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_icp_correspondences_self_match() {
        let client = get_client();
        // 3 source points = 3 target points; identity transform; small threshold
        let pts: Vec<f32> = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let identity: [f32; 16] = [
            1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        ];
        let (corr, dists) = icp_correspondences(&client, &pts, &pts, &identity, 0.1);
        // Each source point should match itself
        assert_eq!(corr[0], 0, "pt0 → target 0");
        assert_eq!(corr[1], 1, "pt1 → target 1");
        assert_eq!(corr[2], 2, "pt2 → target 2");
        for d in &dists {
            assert!(*d >= 0.0 && *d < 0.01, "distance should be ~0, got {d}");
        }
    }
}
