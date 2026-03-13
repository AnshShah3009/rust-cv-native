//! CubeCL ray-mesh intersection kernel — Möller–Trumbore algorithm.
//!
//! Each GPU thread processes one ray against ALL triangles in the mesh,
//! finding the closest intersection (smallest positive t).
//!
//! # Input layout
//!
//! - `origins`:  f32 array, stride 3 — ray origin (x, y, z) per ray
//! - `dirs`:     f32 array, stride 3 — ray direction (x, y, z) per ray
//! - `vertices`: f32 array, stride 3 — mesh vertex positions (x, y, z)
//! - `faces`:    u32 array, stride 3 — triangle face indices into `vertices`
//!
//! # Output layout
//!
//! - `hit_dist`:  f32 array [num_rays] — hit distance, or -1.0 for miss
//! - `hit_point`: f32 array [num_rays * 3] — hit point (x, y, z)
//! - `hit_normal`: f32 array [num_rays * 3] — face normal (x, y, z)
//!
//! # CubeCL constraints applied
//! - No closures — all math inlined
//! - RuntimeCell for mutable loop state (best_t, best triangle index)
//! - `f32::cast_from()` for int-to-float conversions
//! - `f32::abs()`, `f32::sqrt()` for math intrinsics

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;

use crate::cubecl::WgpuClient;

// ---------------------------------------------------------------------------
// Möller–Trumbore ray-triangle intersection kernel
// ---------------------------------------------------------------------------
//
// For each ray (one thread per ray), iterate over all faces and track the
// closest hit. Writes hit distance, hit point, and face normal to output
// arrays.

#[cube(launch)]
fn moller_trumbore_kernel(
    origins: &Array<f32>,
    dirs: &Array<f32>,
    vertices: &Array<f32>,
    faces: &Array<u32>,
    hit_dist: &mut Array<f32>,
    hit_point: &mut Array<f32>,
    hit_normal: &mut Array<f32>,
    #[comptime] num_rays: usize,
    #[comptime] num_faces: usize,
) {
    let ray_idx = ABSOLUTE_POS;
    if ray_idx < num_rays {
        let rb = ray_idx * 3;

        // Ray origin
        let ox = origins[rb];
        let oy = origins[rb + 1];
        let oz = origins[rb + 2];

        // Ray direction
        let dx = dirs[rb];
        let dy = dirs[rb + 1];
        let dz = dirs[rb + 2];

        let eps = 1e-7f32;

        // Track closest hit using RuntimeCell
        let best_t = RuntimeCell::<f32>::new(3.402823e+38f32); // f32::MAX approx
        let best_face = RuntimeCell::<u32>::new(0xFFFF_FFFFu32);

        for fi in 0usize..num_faces {
            let fb = fi * 3;
            let i0 = usize::cast_from(faces[fb]);
            let i1 = usize::cast_from(faces[fb + 1]);
            let i2 = usize::cast_from(faces[fb + 2]);

            let v0b = i0 * 3;
            let v1b = i1 * 3;
            let v2b = i2 * 3;

            // Vertices of the triangle
            let v0x = vertices[v0b];
            let v0y = vertices[v0b + 1];
            let v0z = vertices[v0b + 2];
            let v1x = vertices[v1b];
            let v1y = vertices[v1b + 1];
            let v1z = vertices[v1b + 2];
            let v2x = vertices[v2b];
            let v2y = vertices[v2b + 1];
            let v2z = vertices[v2b + 2];

            // edge1 = v1 - v0
            let e1x = v1x - v0x;
            let e1y = v1y - v0y;
            let e1z = v1z - v0z;

            // edge2 = v2 - v0
            let e2x = v2x - v0x;
            let e2y = v2y - v0y;
            let e2z = v2z - v0z;

            // h = dir x edge2
            let hx = dy * e2z - dz * e2y;
            let hy = dz * e2x - dx * e2z;
            let hz = dx * e2y - dy * e2x;

            // a = edge1 . h
            let a = e1x * hx + e1y * hy + e1z * hz;

            // Skip near-parallel rays (|a| < eps)
            if f32::abs(a) > eps {
                let inv_a = 1.0f32 / a;

                // s = origin - v0
                let sx = ox - v0x;
                let sy = oy - v0y;
                let sz = oz - v0z;

                // u = (s . h) / a
                let u = (sx * hx + sy * hy + sz * hz) * inv_a;

                if u >= 0.0f32 && u <= 1.0f32 {
                    // q = s x edge1
                    let qx = sy * e1z - sz * e1y;
                    let qy = sz * e1x - sx * e1z;
                    let qz = sx * e1y - sy * e1x;

                    // v = (dir . q) / a
                    let v = (dx * qx + dy * qy + dz * qz) * inv_a;

                    if v >= 0.0f32 && u + v <= 1.0f32 {
                        // t = (edge2 . q) / a
                        let t = (e2x * qx + e2y * qy + e2z * qz) * inv_a;

                        if t > eps && t < best_t.read() {
                            best_t.store(t);
                            best_face.store(u32::cast_from(fi));
                        }
                    }
                }
            }
        }

        // Write results
        if best_face.read() != 0xFFFF_FFFFu32 {
            let t = best_t.read();
            hit_dist[ray_idx] = t;

            // Hit point = origin + t * dir
            let pb = ray_idx * 3;
            hit_point[pb] = ox + t * dx;
            hit_point[pb + 1] = oy + t * dy;
            hit_point[pb + 2] = oz + t * dz;

            // Recompute face normal for the best hit triangle
            let bfi = usize::cast_from(best_face.read()) * 3;
            let bi0 = usize::cast_from(faces[bfi]) * 3;
            let bi1 = usize::cast_from(faces[bfi + 1]) * 3;
            let bi2 = usize::cast_from(faces[bfi + 2]) * 3;

            let be1x = vertices[bi1] - vertices[bi0];
            let be1y = vertices[bi1 + 1] - vertices[bi0 + 1];
            let be1z = vertices[bi1 + 2] - vertices[bi0 + 2];
            let be2x = vertices[bi2] - vertices[bi0];
            let be2y = vertices[bi2 + 1] - vertices[bi0 + 1];
            let be2z = vertices[bi2 + 2] - vertices[bi0 + 2];

            // normal = edge1 x edge2
            let nx = be1y * be2z - be1z * be2y;
            let ny = be1z * be2x - be1x * be2z;
            let nz = be1x * be2y - be1y * be2x;

            // Normalise
            let nlen = f32::sqrt(nx * nx + ny * ny + nz * nz);
            if nlen > 1e-12f32 {
                hit_normal[pb] = nx / nlen;
                hit_normal[pb + 1] = ny / nlen;
                hit_normal[pb + 2] = nz / nlen;
            } else {
                hit_normal[pb] = 0.0f32;
                hit_normal[pb + 1] = 0.0f32;
                hit_normal[pb + 2] = 0.0f32;
            }
        } else {
            hit_dist[ray_idx] = -1.0f32;

            let pb = ray_idx * 3;
            hit_point[pb] = 0.0f32;
            hit_point[pb + 1] = 0.0f32;
            hit_point[pb + 2] = 0.0f32;
            hit_normal[pb] = 0.0f32;
            hit_normal[pb + 1] = 0.0f32;
            hit_normal[pb + 2] = 0.0f32;
        }
    }
}

/// Ray-mesh intersection using the Moller-Trumbore algorithm on GPU.
///
/// Tests each ray against every triangle in the mesh and returns the closest
/// hit per ray.
///
/// # Arguments
///
/// * `origins` — f32 slice, stride 3: ray origin (x, y, z) per ray
/// * `dirs` — f32 slice, stride 3: ray direction (x, y, z) per ray
/// * `vertices` — f32 slice, stride 3: mesh vertex positions (x, y, z)
/// * `faces` — u32 slice, stride 3: triangle face indices into `vertices`
///
/// # Returns
///
/// `(hit_dist, hit_point, hit_normal)` where:
/// - `hit_dist`: `[num_rays]` — distance to closest hit, or -1.0 for miss
/// - `hit_point`: `[num_rays * 3]` — hit point (x, y, z)
/// - `hit_normal`: `[num_rays * 3]` — unit face normal (x, y, z)
pub fn raycast_mesh(
    client: &WgpuClient,
    origins: &[f32],
    dirs: &[f32],
    vertices: &[f32],
    faces: &[u32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_rays = origins.len() / 3;
    let num_faces = faces.len() / 3;

    let ori_h = client.create_from_slice(f32::as_bytes(origins));
    let dir_h = client.create_from_slice(f32::as_bytes(dirs));
    let vtx_h = client.create_from_slice(f32::as_bytes(vertices));
    let fac_h = client.create_from_slice(u32::as_bytes(faces));

    let dist_h = client.empty(num_rays * 4);
    let point_h = client.empty(num_rays * 3 * 4);
    let normal_h = client.empty(num_rays * 3 * 4);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, num_rays, cube_dim);

    unsafe {
        moller_trumbore_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&ori_h, origins.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&dir_h, dirs.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&vtx_h, vertices.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&fac_h, faces.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&dist_h, num_rays, 1),
            ArrayArg::from_raw_parts::<f32>(&point_h, num_rays * 3, 1),
            ArrayArg::from_raw_parts::<f32>(&normal_h, num_rays * 3, 1),
            num_rays,
            num_faces,
        )
    }
    .unwrap();

    let dist_bytes = client.read_one(dist_h);
    let point_bytes = client.read_one(point_h);
    let normal_bytes = client.read_one(normal_h);

    (
        f32::from_bytes(&dist_bytes).to_vec(),
        f32::from_bytes(&point_bytes).to_vec(),
        f32::from_bytes(&normal_bytes).to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl::get_client;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_ray_hits_triangle() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping");
            return;
        };

        // Triangle on z=1 plane: (0,0,1), (1,0,1), (0,1,1)
        let vertices: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let faces: Vec<u32> = vec![0, 1, 2];

        // Ray from origin aimed at centre of triangle — should hit
        let origins: Vec<f32> = vec![0.2, 0.2, 0.0];
        let dirs: Vec<f32> = vec![0.0, 0.0, 1.0];

        let (dist, point, normal) = raycast_mesh(&client, &origins, &dirs, &vertices, &faces);

        assert_eq!(dist.len(), 1);
        assert!(dist[0] > 0.0, "expected hit, got dist={}", dist[0]);
        assert!(
            (dist[0] - 1.0).abs() < 1e-4,
            "expected t~1.0, got {}",
            dist[0]
        );

        // Hit point should be (0.2, 0.2, 1.0)
        assert!((point[0] - 0.2).abs() < 1e-4, "hit_x={}", point[0]);
        assert!((point[1] - 0.2).abs() < 1e-4, "hit_y={}", point[1]);
        assert!((point[2] - 1.0).abs() < 1e-4, "hit_z={}", point[2]);

        // Normal should point in -z direction (towards ray origin)
        // edge1 x edge2 = (1,0,0) x (0,1,0) = (0,0,1) — normalised
        assert!((normal[2].abs() - 1.0).abs() < 1e-4, "nz={}", normal[2]);
    }

    #[test]
    #[serial]
    fn test_ray_misses_triangle() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping");
            return;
        };

        // Same triangle on z=1 plane
        let vertices: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let faces: Vec<u32> = vec![0, 1, 2];

        // Ray aimed away from the triangle
        let origins: Vec<f32> = vec![5.0, 5.0, 0.0];
        let dirs: Vec<f32> = vec![0.0, 0.0, 1.0];

        let (dist, _point, _normal) = raycast_mesh(&client, &origins, &dirs, &vertices, &faces);

        assert_eq!(dist.len(), 1);
        assert!(
            dist[0] < 0.0,
            "expected miss (dist<0), got dist={}",
            dist[0]
        );
    }

    #[test]
    #[serial]
    fn test_closest_of_two_triangles() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping");
            return;
        };

        // Two parallel triangles at z=1 and z=3
        #[rustfmt::skip]
        let vertices: Vec<f32> = vec![
            // Triangle 0 at z=1
            0.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            // Triangle 1 at z=3
            0.0, 0.0, 3.0,
            1.0, 0.0, 3.0,
            0.0, 1.0, 3.0,
        ];
        let faces: Vec<u32> = vec![0, 1, 2, 3, 4, 5];

        // Ray through both triangles
        let origins: Vec<f32> = vec![0.2, 0.2, 0.0];
        let dirs: Vec<f32> = vec![0.0, 0.0, 1.0];

        let (dist, _point, _normal) = raycast_mesh(&client, &origins, &dirs, &vertices, &faces);

        assert_eq!(dist.len(), 1);
        // Should hit the closer triangle at z=1 (t=1.0)
        assert!(
            (dist[0] - 1.0).abs() < 1e-4,
            "expected closest hit at t~1.0, got {}",
            dist[0]
        );
    }
}
