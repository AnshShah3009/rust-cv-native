/// Perspective-n-Point (PnP) pose estimation module
///
/// This module provides functions to estimate camera pose (rotation and translation)
/// given a set of 3D object points and their 2D image projections.
use crate::Result;
use cv_core::{CameraIntrinsics, Pose};
use cv_hal;
use cv_runtime::RuntimeRunner;
use nalgebra::{DMatrix, Matrix3, Matrix3x4, Point2, Point3, Rotation3, Vector3};
use rayon::prelude::*;

/// Solves the Perspective-n-Point problem using Direct Linear Transform (DLT)
pub fn solve_pnp_dlt(
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> Result<Pose> {
    if object_points.len() != image_points.len() {
        return Err(cv_core::Error::CalibrationError(
            "object_points and image_points must have equal length".to_string(),
        ));
    }
    if object_points.len() < 6 {
        return Err(cv_core::Error::CalibrationError(
            "solve_pnp_dlt needs at least 6 correspondences".to_string(),
        ));
    }

    let k_inv = intrinsics.inverse_matrix();
    let n = object_points.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 12);

    for (i, (obj, pix)) in object_points.iter().zip(image_points.iter()).enumerate() {
        let x = k_inv * Vector3::new(pix.x, pix.y, 1.0);
        let xn = x[0] / x[2];
        let yn = x[1] / x[2];
        let xw = obj.x;
        let yw = obj.y;
        let zw = obj.z;

        let r0 = 2 * i;
        let r1 = r0 + 1;

        a[(r0, 0)] = xw;
        a[(r0, 1)] = yw;
        a[(r0, 2)] = zw;
        a[(r0, 3)] = 1.0;
        a[(r0, 8)] = -xn * xw;
        a[(r0, 9)] = -xn * yw;
        a[(r0, 10)] = -xn * zw;
        a[(r0, 11)] = -xn;

        a[(r1, 4)] = xw;
        a[(r1, 5)] = yw;
        a[(r1, 6)] = zw;
        a[(r1, 7)] = 1.0;
        a[(r1, 8)] = -yn * xw;
        a[(r1, 9)] = -yn * yw;
        a[(r1, 10)] = -yn * zw;
        a[(r1, 11)] = -yn;
    }

    let svd = a.svd(true, true);
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD failed in solve_pnp_dlt".to_string())
    })?;
    let p = vt.row(vt.nrows() - 1);

    let mut pmat = Matrix3x4::<f64>::zeros();
    for r in 0..3 {
        for c in 0..4 {
            pmat[(r, c)] = p[(0, r * 4 + c)];
        }
    }

    let m = Matrix3::new(
        pmat[(0, 0)],
        pmat[(0, 1)],
        pmat[(0, 2)],
        pmat[(1, 0)],
        pmat[(1, 1)],
        pmat[(1, 2)],
        pmat[(2, 0)],
        pmat[(2, 1)],
        pmat[(2, 2)],
    );
    let mut t = Vector3::new(pmat[(0, 3)], pmat[(1, 3)], pmat[(2, 3)]);

    let svd_m = m.svd(true, true);
    let u = svd_m.u.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD U missing in solve_pnp_dlt".to_string())
    })?;
    let vt_m = svd_m.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD V^T missing in solve_pnp_dlt".to_string())
    })?;

    let mut r = u * vt_m;
    let scale =
        (svd_m.singular_values[0] + svd_m.singular_values[1] + svd_m.singular_values[2]) / 3.0;
    if scale.abs() < 1e-12 {
        return Err(cv_core::Error::CalibrationError(
            "Degenerate solve_pnp_dlt scale".to_string(),
        ));
    }
    t /= scale;

    if r.determinant() < 0.0 {
        r = -r;
        t = -t;
    }

    Ok(Pose::new(r, t))
}

/// Solves the PnP problem using RANSAC
pub fn solve_pnp_ransac(
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    reprojection_threshold_px: f64,
    max_iters: usize,
) -> Result<(Pose, Vec<bool>)> {
    if object_points.len() != image_points.len() || object_points.len() < 6 {
        return Err(cv_core::Error::CalibrationError(
            "solve_pnp_ransac needs >=6 paired points".to_string(),
        ));
    }

    let n = object_points.len();
    let sample_k = 6usize;
    let iters = max_iters.max(64);
    let mut best_pose = None;
    let mut best_inliers = vec![false; n];
    let mut best_count = 0usize;
    let mut best_error = f64::INFINITY;

    for i in 0..iters {
        let idx = sample_unique_indices(n, sample_k, i as u64 + 11);
        let sample_obj: Vec<Point3<f64>> = idx.iter().map(|&j| object_points[j]).collect();
        let sample_img: Vec<Point2<f64>> = idx.iter().map(|&j| image_points[j]).collect();

        let pose = match solve_pnp_dlt(&sample_obj, &sample_img, intrinsics) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let mut inliers = vec![false; n];
        let mut count = 0usize;
        let mut sum_err = 0.0f64;
        for j in 0..n {
            let err = reprojection_error_px_dist(
                &pose,
                intrinsics,
                distortion,
                &object_points[j],
                &image_points[j],
            );
            if err.is_finite() && err <= reprojection_threshold_px {
                inliers[j] = true;
                count += 1;
                sum_err += err;
            }
        }
        if count == 0 {
            continue;
        }
        let mean_err = sum_err / count as f64;
        if count > best_count || (count == best_count && mean_err < best_error) {
            best_pose = Some(pose);
            best_inliers = inliers;
            best_count = count;
            best_error = mean_err;
        }
    }

    let best_pose = best_pose.ok_or_else(|| {
        cv_core::Error::CalibrationError("RANSAC failed to estimate PnP pose".to_string())
    })?;

    let inlier_obj: Vec<Point3<f64>> = object_points
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(p, &m)| if m { Some(*p) } else { None })
        .collect();
    let inlier_img: Vec<Point2<f64>> = image_points
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(p, &m)| if m { Some(*p) } else { None })
        .collect();

    let refined_pose = if inlier_obj.len() >= 6 {
        solve_pnp_refine(
            &best_pose,
            &inlier_obj,
            &inlier_img,
            intrinsics,
            distortion,
            20,
        )
        .unwrap_or(best_pose)
    } else {
        best_pose
    };

    Ok((refined_pose, best_inliers))
}

fn reprojection_error_px_dist(
    extrinsics: &Pose,
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    object_point: &Point3<f64>,
    image_point: &Point2<f64>,
) -> f64 {
    let pred = project_point_dist(intrinsics, distortion, extrinsics, object_point);
    ((pred.x - image_point.x).powi(2) + (pred.y - image_point.y).powi(2)).sqrt()
}

pub fn solve_pnp_refine(
    initial: &Pose,
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    max_iters: usize,
) -> Result<Pose> {
    let runner = cv_runtime::default_runner().unwrap_or_else(|_| {
        // Fallback to CPU registry on error
        cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
    });
    solve_pnp_refine_ctx(
        initial,
        object_points,
        image_points,
        intrinsics,
        distortion,
        max_iters,
        &runner,
    )
}

/// Compute the Rodrigues rotation Jacobian: returns (R, dR/dr).
///
/// Given a 3-vector `r` (axis-angle / Rodrigues), returns the 3x3 rotation
/// matrix `R` and the 9x3 Jacobian `dR/dr` where `dR` is stored column-major
/// as a 9-vector (i.e., `dR/dr_k` = k-th column reshaped from R's column-major
/// layout).
///
/// This is the standard formulation used in OpenCV's `Rodrigues()`.
fn rodrigues_jacobian(rvec: &Vector3<f64>) -> (Matrix3<f64>, [[f64; 9]; 3]) {
    let theta = rvec.norm();
    if theta < 1e-15 {
        // Small angle: R ≈ I + [r]×
        let r = Matrix3::new(
            1.0, -rvec[2], rvec[1], rvec[2], 1.0, -rvec[0], -rvec[1], rvec[0], 1.0,
        );
        // dR/dr_k ≈ d[r]×/dr_k (the k-th generator of so(3))
        // [e1]× = [[0,0,0],[0,0,-1],[0,1,0]]
        // [e2]× = [[0,0,1],[0,0,0],[-1,0,0]]
        // [e3]× = [[0,-1,0],[1,0,0],[0,0,0]]
        // Stored column-major: col0, col1, col2 of the 3×3 matrix.
        let dr0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0]; // d/dr_x
        let dr1 = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // d/dr_y
        let dr2 = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // d/dr_z
        return (r, [dr0, dr1, dr2]);
    }

    let inv_theta = 1.0 / theta;
    let inv_theta2 = inv_theta * inv_theta;
    let ct = theta.cos();
    let st = theta.sin();
    let alpha = st * inv_theta; // sin(θ)/θ

    // R = I + alpha * [r]× + ((1-cos(θ))/θ²) * [r]×²
    // Using the identity [r]×² = r*rᵀ − ||r||² I
    let kx = Matrix3::new(
        0.0, -rvec[2], rvec[1], rvec[2], 0.0, -rvec[0], -rvec[1], rvec[0], 0.0,
    );
    let kx2 = rvec * rvec.transpose() - Matrix3::identity() * (theta * theta);

    let rot = Matrix3::identity() + kx * alpha + kx2 * inv_theta2 * (1.0 - ct);

    // For derivatives, we differentiate:
    //   R = I + (sin θ / θ) [r]× + ((1-cos θ) / θ²) [r]×²
    // w.r.t. each component r_k.
    //
    // d_alpha/dr_k = (θ cos θ − sin θ) / θ² · (r_k/θ) = r_k (cos θ − alpha) / θ²
    // d_beta/dr_k  = (θ sin θ − 2(1−cos θ)) / θ³ · (r_k/θ) = ... (computed below)
    //
    // d[r]×/dr_k = [e_k]×  (generator matrix)
    // d([r]×²)/dr_k = [e_k]× [r]× + [r]× [e_k]×  = e_k rᵀ + r e_kᵀ − 2 r_k I

    let d_alpha_dtheta = (theta * ct - st) * inv_theta2; // d(sinθ/θ)/dθ
    let d_beta_dtheta = (theta * st - 2.0 * (1.0 - ct)) / (theta * theta * theta); // d((1-cosθ)/θ²)/dθ

    // Generator matrices [e_k]× stored column-major
    let gen: [Matrix3<f64>; 3] = [
        Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0),
        Matrix3::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
        Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ];

    let mut dr_all = [[0.0f64; 9]; 3];
    let coeff = (1.0 - ct) * inv_theta2; // (1-cos(θ))/θ²

    for ki in 0..3 {
        let rk = rvec[ki];
        let dtheta_drk = rk * inv_theta;

        let dalpha = d_alpha_dtheta * dtheta_drk;
        let dcoeff = d_beta_dtheta * dtheta_drk;

        // d([r]×²)/dr_k = e_k rᵀ + r e_kᵀ − 2 r_k I
        let mut ek = Vector3::zeros();
        ek[ki] = 1.0;
        let dkx2 = ek * rvec.transpose() + rvec * ek.transpose() - Matrix3::identity() * (2.0 * rk);

        // dR/dr_k = dalpha * [r]× + alpha * [e_k]×
        //         + dcoeff * ([r]×²) + coeff * d([r]×²)/dr_k
        let dr_mat = kx * dalpha + gen[ki] * alpha + kx2 * dcoeff + dkx2 * coeff;

        // Store column-major: dr_all[ki][col*3 + row]
        for col in 0..3 {
            for row in 0..3 {
                dr_all[ki][col * 3 + row] = dr_mat[(row, col)];
            }
        }
    }

    (rot, dr_all)
}

/// Compute the analytical 2x6 Jacobian of the projection of a 3D point
/// w.r.t. pose parameters [r0, r1, r2, tx, ty, tz].
///
/// The pose parameterisation is: params[0..3] = Rodrigues rotation vector,
/// params[3..6] = translation.  Projection: pc = R*X + t, then
/// u = fx * pc.x/pc.z + cx,  v = fy * pc.y/pc.z + cy.
///
/// When `distortion` is `Some(...)`, falls back to numerical (central)
/// differences because the distortion chain rule is model-dependent.
fn analytical_jacobian(
    params: &[f64; 6],
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    point: &Point3<f64>,
) -> nalgebra::Matrix2x6<f64> {
    // Distortion present: fall back to numerical Jacobian (central differences).
    if distortion.is_some() {
        return numerical_jacobian(params, intrinsics, distortion, point);
    }

    let rvec = Vector3::new(params[0], params[1], params[2]);
    let t = Vector3::new(params[3], params[4], params[5]);

    let (rot, dr_dr) = rodrigues_jacobian(&rvec);

    let x = point.coords; // 3-vector
    let pc = rot * x + t; // camera-space point

    let z_inv = if pc[2].abs() <= 1e-12 {
        return nalgebra::Matrix2x6::zeros();
    } else {
        1.0 / pc[2]
    };
    let z_inv2 = z_inv * z_inv;

    let fx = intrinsics.fx;
    let fy = intrinsics.fy;

    // d(u,v)/d(pc) -- 2x3 Jacobian of the pinhole projection
    //   u = fx * pc[0]/pc[2] + cx
    //   v = fy * pc[1]/pc[2] + cy
    let duv_dpc = nalgebra::Matrix2x3::new(
        fx * z_inv,
        0.0,
        -fx * pc[0] * z_inv2,
        0.0,
        fy * z_inv,
        -fy * pc[1] * z_inv2,
    );

    // --- Translation part: d(pc)/dt = I₃ ---
    // J_t = duv_dpc * I = duv_dpc  (2×3)

    // --- Rotation part: d(pc)/dr_k = dR/dr_k * X ---
    // For each k in 0..3, dR/dr_k is a 3×3 matrix (stored column-major in dr_dr[k]).
    // d(pc)/dr_k = dR/dr_k * X  (3-vector)
    let mut dpc_dr = nalgebra::Matrix3::zeros(); // columns = d(pc)/dr_k
    for ki in 0..3 {
        // Reconstruct dR/dr_k from column-major storage
        let dr = &dr_dr[ki];
        // dR * X: row i = sum_j dR(i,j) * X[j]
        for row in 0..3 {
            let mut val = 0.0;
            for col in 0..3 {
                val += dr[col * 3 + row] * x[col];
            }
            dpc_dr[(row, ki)] = val;
        }
    }

    // J_r = duv_dpc * dpc_dr  (2×3)
    let j_r = duv_dpc * dpc_dr;
    let j_t = duv_dpc; // 2×3

    // Full Jacobian: [dr | dt] layout: columns [r0,r1,r2,tx,ty,tz]
    nalgebra::Matrix2x6::from_row_slice(&[
        j_r[(0, 0)],
        j_r[(0, 1)],
        j_r[(0, 2)],
        j_t[(0, 0)],
        j_t[(0, 1)],
        j_t[(0, 2)],
        j_r[(1, 0)],
        j_r[(1, 1)],
        j_r[(1, 2)],
        j_t[(1, 0)],
        j_t[(1, 1)],
        j_t[(1, 2)],
    ])
}

/// Numerical Jacobian via forward differences (fallback for distorted projection).
fn numerical_jacobian(
    params: &[f64; 6],
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    point: &Point3<f64>,
) -> nalgebra::Matrix2x6<f64> {
    let eps = 1e-7;
    let base = params_to_extrinsics(params);
    let pred0 = project_point_dist(intrinsics, distortion, &base, point);

    let mut j = [[0.0f64; 6]; 2];
    for k in 0..6 {
        let mut p = *params;
        p[k] += eps;
        let ext = params_to_extrinsics(&p);
        let pred1 = project_point_dist(intrinsics, distortion, &ext, point);
        j[0][k] = (pred1.x - pred0.x) / eps;
        j[1][k] = (pred1.y - pred0.y) / eps;
    }

    nalgebra::Matrix2x6::from_row_slice(&[
        j[0][0], j[0][1], j[0][2], j[0][3], j[0][4], j[0][5], j[1][0], j[1][1], j[1][2], j[1][3],
        j[1][4], j[1][5],
    ])
}

/// Context-aware PnP refinement using Levenberg-Marquardt
pub fn solve_pnp_refine_ctx(
    initial: &Pose,
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    max_iters: usize,
    group: &RuntimeRunner,
) -> Result<Pose> {
    if object_points.len() != image_points.len() || object_points.len() < 6 {
        return Err(cv_core::Error::CalibrationError(
            "solve_pnp_refine needs >=6 paired points".to_string(),
        ));
    }

    let mut params = extrinsics_to_params(initial);
    let mut lambda = 0.001;
    let n_pts = object_points.len();

    let mut current_err = group.run(|| {
        let base = params_to_extrinsics(&params);
        object_points
            .par_iter()
            .zip(image_points.par_iter())
            .map(|(p3, p2)| {
                let pred = project_point_dist(intrinsics, distortion, &base, p3);
                (pred.x - p2.x).powi(2) + (pred.y - p2.y).powi(2)
            })
            .sum::<f64>()
    });

    for _ in 0..max_iters {
        let base = params_to_extrinsics(&params);

        // Parallel analytical Jacobian and residual accumulation
        let (jtj, jtr) = group.run(|| {
            let results: Vec<(nalgebra::Matrix6<f64>, nalgebra::Vector6<f64>)> = (0..n_pts)
                .into_par_iter()
                .map(|i| {
                    let p3 = &object_points[i];
                    let p2 = &image_points[i];
                    let pred0 = project_point_dist(intrinsics, distortion, &base, p3);

                    let j = analytical_jacobian(&params, intrinsics, distortion, p3);
                    let r = nalgebra::Vector2::new(pred0.x - p2.x, pred0.y - p2.y);

                    (j.transpose() * j, j.transpose() * r)
                })
                .collect();

            let mut local_ata = nalgebra::Matrix6::<f64>::zeros();
            let mut local_atb = nalgebra::Vector6::<f64>::zeros();
            for (a, b) in results {
                local_ata += a;
                local_atb += b;
            }
            (local_ata, local_atb)
        });

        // Levenberg-Marquardt
        let mut lhs = jtj;
        for i in 0..6 {
            lhs[(i, i)] *= 1.0 + lambda;
        }

        if let Some(delta) = lhs.lu().solve(&jtr) {
            let mut next_params = params;
            for k in 0..6 {
                next_params[k] -= delta[k];
            }

            let next_err = group.run(|| {
                let next_ext = params_to_extrinsics(&next_params);
                object_points
                    .par_iter()
                    .zip(image_points.par_iter())
                    .map(|(p3, p2)| {
                        let pred = project_point_dist(intrinsics, distortion, &next_ext, p3);
                        (pred.x - p2.x).powi(2) + (pred.y - p2.y).powi(2)
                    })
                    .sum::<f64>()
            });

            if next_err < current_err {
                params = next_params;
                current_err = next_err;
                lambda /= 10.0;
                if delta.norm() < 1e-8 {
                    break;
                }
            } else {
                lambda *= 10.0;
            }
        } else {
            break;
        }
    }

    Ok(params_to_extrinsics(&params))
}

fn extrinsics_to_params(ext: &Pose) -> [f64; 6] {
    let r = Rotation3::from_matrix_unchecked(ext.rotation_matrix());
    let omega = r.scaled_axis();
    [
        omega[0],
        omega[1],
        omega[2],
        ext.translation[0],
        ext.translation[1],
        ext.translation[2],
    ]
}

fn params_to_extrinsics(params: &[f64; 6]) -> Pose {
    let rot = Rotation3::new(Vector3::new(params[0], params[1], params[2])).into_inner();
    let t = Vector3::new(params[3], params[4], params[5]);
    Pose::new(rot, t)
}

fn project_point_dist(
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    ext: &Pose,
    p: &Point3<f64>,
) -> Point2<f64> {
    let pc = ext.rotation * p.coords + ext.translation;
    if pc[2].abs() <= 1e-12 {
        return Point2::new(0.0, 0.0);
    }
    let x = pc[0] / pc[2];
    let y = pc[1] / pc[2];
    let (xd, yd) = if let Some(dist) = distortion {
        dist.apply(x, y)
    } else {
        (x, y)
    };
    Point2::new(
        intrinsics.fx * xd + intrinsics.cx,
        intrinsics.fy * yd + intrinsics.cy,
    )
}

fn sample_unique_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut out = Vec::with_capacity(k);
    let mut used = vec![false; n];
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    while out.len() < k {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (state as usize) % n;
        if !used[idx] {
            used[idx] = true;
            out.push(idx);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Vector3};

    /// Compute a central-difference numerical Jacobian for comparison.
    fn central_difference_jacobian(
        params: &[f64; 6],
        intrinsics: &CameraIntrinsics,
        point: &Point3<f64>,
    ) -> nalgebra::Matrix2x6<f64> {
        let eps = 1e-7;
        let mut j = [[0.0f64; 6]; 2];
        for k in 0..6 {
            let mut p_plus = *params;
            let mut p_minus = *params;
            p_plus[k] += eps;
            p_minus[k] -= eps;
            let ext_plus = params_to_extrinsics(&p_plus);
            let ext_minus = params_to_extrinsics(&p_minus);
            let pred_plus = project_point_dist(intrinsics, None, &ext_plus, point);
            let pred_minus = project_point_dist(intrinsics, None, &ext_minus, point);
            j[0][k] = (pred_plus.x - pred_minus.x) / (2.0 * eps);
            j[1][k] = (pred_plus.y - pred_minus.y) / (2.0 * eps);
        }
        nalgebra::Matrix2x6::from_row_slice(&[
            j[0][0], j[0][1], j[0][2], j[0][3], j[0][4], j[0][5], j[1][0], j[1][1], j[1][2],
            j[1][3], j[1][4], j[1][5],
        ])
    }

    #[test]
    fn test_analytical_jacobian_matches_numerical() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);

        // Test several combinations of rotation, translation, and 3D point
        let test_cases: Vec<([f64; 6], Point3<f64>)> = vec![
            // Near-identity rotation, point in front of camera
            (
                [0.01, -0.02, 0.03, 0.0, 0.0, 5.0],
                Point3::new(1.0, 2.0, 0.5),
            ),
            // Moderate rotation around Y axis
            ([0.0, 0.5, 0.0, 1.0, -0.5, 8.0], Point3::new(-1.0, 0.5, 3.0)),
            // Large rotation (close to pi)
            ([1.0, 0.8, -0.6, 0.2, 0.3, 10.0], Point3::new(0.0, 0.0, 2.0)),
            // Near-zero rotation (small angle regime)
            (
                [1e-10, 1e-10, 1e-10, 0.0, 0.0, 3.0],
                Point3::new(0.5, -0.5, 1.0),
            ),
            // Rotation around X with off-center point
            ([0.3, 0.0, 0.0, -2.0, 1.0, 6.0], Point3::new(2.0, -1.0, 0.0)),
            // Mixed rotation and translation
            (
                [-0.2, 0.4, -0.1, 0.5, -0.3, 4.0],
                Point3::new(0.1, 0.2, 0.3),
            ),
        ];

        for (case_idx, (params, point)) in test_cases.iter().enumerate() {
            let j_analytical = analytical_jacobian(params, &intrinsics, None, point);
            let j_numerical = central_difference_jacobian(params, &intrinsics, point);

            let diff = j_analytical - j_numerical;
            let max_err = diff.abs().max();

            assert!(
                max_err < 1e-4,
                "Case {}: analytical Jacobian differs from numerical by {:.2e} (threshold 1e-4)\n\
                 params: {:?}\n\
                 point: {:?}\n\
                 analytical:\n{:.6}\n\
                 numerical:\n{:.6}\n\
                 diff:\n{:.2e}",
                case_idx,
                max_err,
                params,
                point,
                j_analytical,
                j_numerical,
                diff,
            );
        }
    }

    #[test]
    fn test_rodrigues_jacobian_small_angle() {
        // Near-zero rotation should produce R ≈ I and sensible derivatives
        let rvec = Vector3::new(1e-12, 1e-12, 1e-12);
        let (rot, _dr) = rodrigues_jacobian(&rvec);

        // R should be very close to identity
        let diff = rot - Matrix3::identity();
        assert!(
            diff.norm() < 1e-8,
            "Small angle Rodrigues should give near-identity rotation"
        );
    }

    #[test]
    fn test_rodrigues_jacobian_consistency() {
        // Verify that the Jacobian is consistent with finite differences of R(r)*X
        let rvec = Vector3::new(0.3, -0.5, 0.2);
        let x = Vector3::new(1.0, 2.0, 3.0);
        let (rot, dr) = rodrigues_jacobian(&rvec);

        let eps = 1e-7;
        for ki in 0..3 {
            let mut rv_plus = rvec;
            let mut rv_minus = rvec;
            rv_plus[ki] += eps;
            rv_minus[ki] -= eps;

            let r_plus = Rotation3::new(rv_plus).into_inner();
            let r_minus = Rotation3::new(rv_minus).into_inner();
            let fd = (r_plus * x - r_minus * x) / (2.0 * eps);

            // Reconstruct dR/dr_k * X from our Jacobian
            let mut analytical = Vector3::zeros();
            for row in 0..3 {
                for col in 0..3 {
                    analytical[row] += dr[ki][col * 3 + row] * x[col];
                }
            }

            let err = (analytical - fd).norm();
            assert!(
                err < 1e-4,
                "Rodrigues Jacobian column {} differs from FD by {:.2e}",
                ki,
                err
            );
        }

        // Also verify R matches nalgebra's Rotation3
        let expected = Rotation3::new(rvec).into_inner();
        let diff = (rot - expected).norm();
        assert!(
            diff < 1e-12,
            "Rodrigues rotation differs from nalgebra by {:.2e}",
            diff
        );
    }
}
