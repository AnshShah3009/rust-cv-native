//! CubeCL TSDF (Truncated Signed Distance Function) kernels — Tier 3.
//!
//! Operations:
//!   `tsdf_integrate` — update voxel grid from a new depth frame
//!   `tsdf_raycast`   — render depth + normals from the TSDF volume
//!
//! # Voxel layout
//!
//! `voxels[z * VY * VX + y * VX + x] = (tsdf: f32, weight: f32)`.
//! Total f32 count = 2 * VX * VY * VZ.
//!
//! # CubeCL constraints applied
//! - No closures — matrix access inlined
//! - No `if-else` expressions — use `select()`
//! - RuntimeCell for mutable loop state
//! - `i32::cast_from(f32_val)` for float→int; `f32::cast_from(u_val)` for int→float

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;

use crate::cubecl::WgpuClient;

// ---------------------------------------------------------------------------
// TSDF helper: safe voxel lookup with bounds clamp
// ---------------------------------------------------------------------------

#[cube]
fn tsdf_at(voxels: &Array<f32>, ix: i32, iy: i32, iz: i32, vx: usize, vy: usize, vz: usize) -> f32 {
    let out = ix < 0i32
        || iy < 0i32
        || iz < 0i32
        || ix >= vx as i32
        || iy >= vy as i32
        || iz >= vz as i32;
    let safe_ix = i32::max(i32::min(ix, vx as i32 - 1i32), 0i32);
    let safe_iy = i32::max(i32::min(iy, vy as i32 - 1i32), 0i32);
    let safe_iz = i32::max(i32::min(iz, vz as i32 - 1i32), 0i32);
    let vi = usize::cast_from(safe_iz) * vy * vx
        + usize::cast_from(safe_iy) * vx
        + usize::cast_from(safe_ix);
    select(out, 1.0f32, voxels[vi * 2])
}

// ---------------------------------------------------------------------------
// TSDF Integration kernel
// ---------------------------------------------------------------------------

#[cube(launch)]
fn tsdf_integrate_kernel(
    voxels: &mut Array<f32>,
    depth: &Array<f32>,
    pose: &Array<f32>, // col-major 4×4 world→camera
    intr: &Array<f32>, // [fx, fy, cx, cy]
    #[comptime] vx: usize,
    #[comptime] vy: usize,
    #[comptime] vz: usize,
    #[comptime] img_w: usize,
    #[comptime] img_h: usize,
    #[comptime] voxel_size_u: u32,
    #[comptime] truncation_u: u32,
) {
    let vi = ABSOLUTE_POS;
    if vi < vx * vy * vz {
        let voxel_size = voxel_size_u as f32 / 1_000_000.0f32;
        let truncation = truncation_u as f32 / 1_000_000.0f32;

        let iz_u = vi / (vx * vy);
        let iy_u = (vi % (vx * vy)) / vx;
        let ix_u = vi % vx;

        let wx = (f32::cast_from(ix_u) + 0.5f32) * voxel_size;
        let wy = (f32::cast_from(iy_u) + 0.5f32) * voxel_size;
        let wz = (f32::cast_from(iz_u) + 0.5f32) * voxel_size;

        // world→camera
        let cx = pose[0] * wx + pose[4] * wy + pose[8] * wz + pose[12];
        let cy = pose[1] * wx + pose[5] * wy + pose[9] * wz + pose[13];
        let cz = pose[2] * wx + pose[6] * wy + pose[10] * wz + pose[14];

        if cz > 0.0f32 {
            let fx = intr[0];
            let fy = intr[1];
            let ccx = intr[2];
            let ccy = intr[3];

            let u = f32::round(fx * cx / cz + ccx);
            let v = f32::round(fy * cy / cz + ccy);
            let ui = usize::cast_from(u32::cast_from(u32::max(
                u32::cast_from(i32::cast_from(u)),
                0u32,
            )));
            let vi_img = usize::cast_from(u32::cast_from(u32::max(
                u32::cast_from(i32::cast_from(v)),
                0u32,
            )));

            if ui < img_w && vi_img < img_h {
                let d = depth[vi_img * img_w + ui];
                if d > 0.0f32 {
                    let sdf = d - cz;
                    if sdf >= -truncation {
                        let tsdf = f32::min(sdf / truncation, 1.0f32);
                        let vb = vi * 2;
                        let tsdf_old = voxels[vb];
                        let w_old = voxels[vb + 1];
                        let w_new = w_old + 1.0f32;
                        voxels[vb] = (tsdf_old * w_old + tsdf) / w_new;
                        voxels[vb + 1] = w_new;
                    }
                }
            }
        }
    }
}

/// Integrate a depth frame into the TSDF voxel grid.
pub fn tsdf_integrate(
    client: &WgpuClient,
    voxels: &mut Vec<f32>,
    depth: &[f32],
    pose: &[f32; 16],
    intr: &[f32; 4],
    vx: usize,
    vy: usize,
    vz: usize,
    img_w: usize,
    img_h: usize,
    voxel_size: f32,
    truncation: f32,
) {
    let n_voxels = vx * vy * vz;
    let vox_h = client.create_from_slice(f32::as_bytes(voxels));
    let dep_h = client.create_from_slice(f32::as_bytes(depth));
    let pos_h = client.create_from_slice(f32::as_bytes(pose.as_slice()));
    let int_h = client.create_from_slice(f32::as_bytes(intr.as_slice()));

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_voxels, cube_dim);

    unsafe {
        tsdf_integrate_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&vox_h, voxels.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&dep_h, depth.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&pos_h, 16, 1),
            ArrayArg::from_raw_parts::<f32>(&int_h, 4, 1),
            vx,
            vy,
            vz,
            img_w,
            img_h,
            (voxel_size * 1_000_000.0) as u32,
            (truncation * 1_000_000.0) as u32,
        )
    }
    .unwrap();

    let result = client.read_one(vox_h);
    let updated = f32::from_bytes(&result);
    voxels.clear();
    voxels.extend_from_slice(updated);
}

// ---------------------------------------------------------------------------
// TSDF Raycast kernel
// ---------------------------------------------------------------------------

#[cube(launch)]
fn tsdf_raycast_kernel(
    voxels: &Array<f32>,
    output: &mut Array<f32>,   // [depth, nx, ny, nz] per pixel
    cam_to_world: &Array<f32>, // 4×4 col-major camera→world
    intr_inv: &Array<f32>,     // [1/fx, 1/fy, cx, cy]
    #[comptime] img_w: usize,
    #[comptime] img_h: usize,
    #[comptime] vx: usize,
    #[comptime] vy: usize,
    #[comptime] vz: usize,
    #[comptime] voxel_size_u: u32,
    #[comptime] step_factor_u: u32,
    #[comptime] max_depth_u: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos < img_w * img_h {
        let voxel_size = voxel_size_u as f32 / 1_000_000.0f32;
        let step = step_factor_u as f32 / 1000.0f32 * voxel_size;
        let max_d = max_depth_u as f32 / 1000.0f32;

        let px = f32::cast_from(pos % img_w);
        let py = f32::cast_from(pos / img_w);

        let ifx = intr_inv[0];
        let ify = intr_inv[1];
        let ccx = intr_inv[2];
        let ccy = intr_inv[3];

        // Ray direction in camera space (normalised)
        let rdx = (px - ccx) * ifx;
        let rdy = (py - ccy) * ify;
        let rlen = f32::sqrt(rdx * rdx + rdy * rdy + 1.0f32);
        let rx = rdx / rlen;
        let ry = rdy / rlen;
        let rz = 1.0f32 / rlen;

        // Transform to world space
        let wrx = cam_to_world[0] * rx + cam_to_world[4] * ry + cam_to_world[8] * rz;
        let wry = cam_to_world[1] * rx + cam_to_world[5] * ry + cam_to_world[9] * rz;
        let wrz = cam_to_world[2] * rx + cam_to_world[6] * ry + cam_to_world[10] * rz;
        let ox = cam_to_world[12];
        let oy = cam_to_world[13];
        let oz = cam_to_world[14];

        let depth_out = RuntimeCell::<f32>::new(-1.0f32);
        let nx_out = RuntimeCell::<f32>::new(0.0f32);
        let ny_out = RuntimeCell::<f32>::new(0.0f32);
        let nz_out = RuntimeCell::<f32>::new(0.0f32);
        let t_cell = RuntimeCell::<f32>::new(voxel_size);
        let prev_tsdf = RuntimeCell::<f32>::new(1.0f32);
        let found = RuntimeCell::<u32>::new(0u32);

        while t_cell.read() < max_d && found.read() == 0u32 {
            let t = t_cell.read();
            let qx = ox + wrx * t;
            let qy = oy + wry * t;
            let qz = oz + wrz * t;
            let tsdf = tsdf_at(
                voxels,
                i32::cast_from(qx / voxel_size),
                i32::cast_from(qy / voxel_size),
                i32::cast_from(qz / voxel_size),
                vx,
                vy,
                vz,
            );

            if prev_tsdf.read() > 0.0f32 && tsdf <= 0.0f32 {
                let alpha = prev_tsdf.read() / (prev_tsdf.read() - tsdf);
                depth_out.store(t - step + step * alpha);

                let sx = ox + wrx * depth_out.read();
                let sy = oy + wry * depth_out.read();
                let sz = oz + wrz * depth_out.read();
                let ix_c = i32::cast_from(sx / voxel_size);
                let iy_c = i32::cast_from(sy / voxel_size);
                let iz_c = i32::cast_from(sz / voxel_size);

                let dfdx = tsdf_at(voxels, ix_c + 1i32, iy_c, iz_c, vx, vy, vz)
                    - tsdf_at(voxels, ix_c - 1i32, iy_c, iz_c, vx, vy, vz);
                let dfdy = tsdf_at(voxels, ix_c, iy_c + 1i32, iz_c, vx, vy, vz)
                    - tsdf_at(voxels, ix_c, iy_c - 1i32, iz_c, vx, vy, vz);
                let dfdz = tsdf_at(voxels, ix_c, iy_c, iz_c + 1i32, vx, vy, vz)
                    - tsdf_at(voxels, ix_c, iy_c, iz_c - 1i32, vx, vy, vz);
                let nlen = f32::sqrt(dfdx * dfdx + dfdy * dfdy + dfdz * dfdz);

                if nlen > 1e-6f32 {
                    nx_out.store(dfdx / nlen);
                    ny_out.store(dfdy / nlen);
                    nz_out.store(dfdz / nlen);
                }
                found.store(1u32);
            }
            prev_tsdf.store(tsdf);
            t_cell.store(t + step);
        }

        let ob = pos * 4;
        output[ob] = depth_out.read();
        output[ob + 1] = nx_out.read();
        output[ob + 2] = ny_out.read();
        output[ob + 3] = nz_out.read();
    }
}

/// Raycast TSDF volume.
/// Returns `Vec<f32>` of length `4 * img_w * img_h`: `[depth, nx, ny, nz]` per pixel.
pub fn tsdf_raycast(
    client: &WgpuClient,
    voxels: &[f32],
    cam_to_world: &[f32; 16],
    intr_inv: &[f32; 4],
    img_w: usize,
    img_h: usize,
    vx: usize,
    vy: usize,
    vz: usize,
    voxel_size: f32,
    _truncation: f32,
    step_factor: f32,
    max_depth: f32,
) -> Vec<f32> {
    let n_pixels = img_w * img_h;
    let vox_h = client.create_from_slice(f32::as_bytes(voxels));
    let out_h = client.empty(n_pixels * 4 * 4);
    let c2w_h = client.create_from_slice(f32::as_bytes(cam_to_world.as_slice()));
    let inv_h = client.create_from_slice(f32::as_bytes(intr_inv.as_slice()));

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_pixels, cube_dim);

    unsafe {
        tsdf_raycast_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&vox_h, voxels.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&out_h, n_pixels * 4, 1),
            ArrayArg::from_raw_parts::<f32>(&c2w_h, 16, 1),
            ArrayArg::from_raw_parts::<f32>(&inv_h, 4, 1),
            img_w,
            img_h,
            vx,
            vy,
            vz,
            (voxel_size * 1_000_000.0) as u32,
            (step_factor * 1000.0) as u32,
            (max_depth * 1000.0) as u32,
        )
    }
    .unwrap();

    let result = client.read_one(out_h);
    f32::from_bytes(&result).to_vec()
}
