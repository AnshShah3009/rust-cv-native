/// Perspective-n-Point (PnP) pose estimation module
///
/// This module provides functions to estimate camera pose (rotation and translation)
/// given a set of 3D object points and their 2D image projections.

use crate::CalibError;
use cv_core::{CameraExtrinsics, CameraIntrinsics};
use nalgebra::{DMatrix, Matrix3, Matrix3x4, Point2, Point3, Rotation3, Vector3};

pub type Result<T> = std::result::Result<T, CalibError>;

/// Solves the Perspective-n-Point problem using Direct Linear Transform (DLT)
///
/// Requires at least 6 point correspondences. This is a direct method that
/// solves a linear system without iterative refinement.
///
/// # Arguments
/// * `object_points` - 3D points in world coordinates
/// * `image_points` - 2D points in image coordinates
/// * `intrinsics` - Camera intrinsic parameters
///
/// # Returns
/// `CameraExtrinsics` containing the estimated rotation matrix and translation vector
pub fn solve_pnp_dlt(
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> Result<CameraExtrinsics> {
    if object_points.len() != image_points.len() {
        return Err(CalibError::InvalidParameters(
            "object_points and image_points must have equal length".to_string(),
        ));
    }
    if object_points.len() < 6 {
        return Err(CalibError::InvalidParameters(
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
        CalibError::InvalidParameters("SVD failed in solve_pnp_dlt".to_string())
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
        CalibError::InvalidParameters("SVD U missing in solve_pnp_dlt".to_string())
    })?;
    let vt_m = svd_m.v_t.ok_or_else(|| {
        CalibError::InvalidParameters("SVD V^T missing in solve_pnp_dlt".to_string())
    })?;

    let mut r = u * vt_m;
    let scale = (svd_m.singular_values[0] + svd_m.singular_values[1] + svd_m.singular_values[2])
        / 3.0;
    if scale.abs() < 1e-12 {
        return Err(CalibError::InvalidParameters(
            "Degenerate solve_pnp_dlt scale".to_string(),
        ));
    }
    t /= scale;

    if r.determinant() < 0.0 {
        r = -r;
        t = -t;
    }

    Ok(CameraExtrinsics::new(r, t))
}

/// Solves the PnP problem using RANSAC to handle outliers
///
/// This method iteratively samples minimal sets (6 points) and fits poses
/// using DLT, keeping the solution with the most inliers. The final solution
/// is refined using all inliers.
///
/// # Arguments
/// * `object_points` - 3D points in world coordinates
/// * `image_points` - 2D points in image coordinates
/// * `intrinsics` - Camera intrinsic parameters
/// * `reprojection_threshold_px` - Maximum reprojection error (pixels) for inlier classification
/// * `max_iters` - Maximum number of RANSAC iterations
///
/// # Returns
/// Tuple of (pose, inlier_mask) where inlier_mask is a boolean vector
pub fn solve_pnp_ransac(
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    reprojection_threshold_px: f64,
    max_iters: usize,
) -> Result<(CameraExtrinsics, Vec<bool>)> {
    if object_points.len() != image_points.len() || object_points.len() < 6 {
        return Err(CalibError::InvalidParameters(
            "solve_pnp_ransac needs >=6 paired points".to_string(),
        ));
    }
    if reprojection_threshold_px <= 0.0 {
        return Err(CalibError::InvalidParameters(
            "reprojection_threshold_px must be > 0".to_string(),
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
            let err = reprojection_error_px(&pose, intrinsics, &object_points[j], &image_points[j]);
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
        CalibError::InvalidParameters("RANSAC failed to estimate PnP pose".to_string())
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
        let dlt_pose = solve_pnp_dlt(&inlier_obj, &inlier_img, intrinsics).unwrap_or(best_pose);
        // P1: Add iterative refinement post-DLT for maximum precision
        solve_pnp_refine(&dlt_pose, &inlier_obj, &inlier_img, intrinsics, 20).unwrap_or(dlt_pose)
    } else {
        best_pose
    };

    Ok((refined_pose, best_inliers))
}

/// Refines a PnP pose estimate using iterative Levenberg-Marquardt-style minimization
///
/// Minimizes the reprojection error through gradient descent on pose parameters.
/// Uses numerical differentiation to compute the Jacobian.
///
/// # Arguments
/// * `initial` - Initial pose estimate
/// * `object_points` - 3D points in world coordinates
/// * `image_points` - 2D points in image coordinates
/// * `intrinsics` - Camera intrinsic parameters
/// * `max_iters` - Maximum number of refinement iterations
///
/// # Returns
/// Refined `CameraExtrinsics`
pub fn solve_pnp_refine(
    initial: &CameraExtrinsics,
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    max_iters: usize,
) -> Result<CameraExtrinsics> {
    if object_points.len() != image_points.len() || object_points.len() < 6 {
        return Err(CalibError::InvalidParameters(
            "solve_pnp_refine needs >=6 paired points".to_string(),
        ));
    }

    let mut params = extrinsics_to_params(initial);
    let iters = max_iters.max(1);

    for _ in 0..iters {
        let base = params_to_extrinsics(&params);
        let mut j = DMatrix::<f64>::zeros(2 * object_points.len(), 6);
        let mut r = DMatrix::<f64>::zeros(2 * object_points.len(), 1);

        for (i, (p3, p2)) in object_points.iter().zip(image_points.iter()).enumerate() {
            let pred = project_point(intrinsics, &base, p3);
            r[(2 * i, 0)] = pred.x - p2.x;
            r[(2 * i + 1, 0)] = pred.y - p2.y;
        }

        let eps = 1e-6;
        for k in 0..6 {
            let mut p_perturbed = params;
            p_perturbed[k] += eps;
            let ext_p = params_to_extrinsics(&p_perturbed);

            for (i, p3) in object_points.iter().enumerate() {
                let p0 = project_point(intrinsics, &base, p3);
                let p1 = project_point(intrinsics, &ext_p, p3);
                j[(2 * i, k)] = (p1.x - p0.x) / eps;
                j[(2 * i + 1, k)] = (p1.y - p0.y) / eps;
            }
        }

        let jt = j.transpose();
        let h = &jt * &j;
        let g = &jt * &r;
        let delta = h
            .lu()
            .solve(&g)
            .ok_or_else(|| CalibError::InvalidParameters("solve_pnp_refine normal equation failed".to_string()))?;

        let mut max_step = 0.0f64;
        for k in 0..6 {
            let step = delta[(k, 0)];
            params[k] -= step;
            max_step = max_step.max(step.abs());
        }
        if max_step < 1e-9 {
            break;
        }
    }

    Ok(params_to_extrinsics(&params))
}

/// Computes the reprojection error in pixels for a single point
///
/// # Arguments
/// * `extrinsics` - Camera extrinsics (pose)
/// * `intrinsics` - Camera intrinsics (K matrix)
/// * `object_point` - 3D point in world coordinates
/// * `image_point` - 2D point in image coordinates
///
/// # Returns
/// Euclidean distance in pixels between projected and actual image point
fn reprojection_error_px(
    extrinsics: &CameraExtrinsics,
    intrinsics: &CameraIntrinsics,
    object_point: &Point3<f64>,
    image_point: &Point2<f64>,
) -> f64 {
    let pc = extrinsics.rotation * object_point.coords + extrinsics.translation;
    if !pc.iter().all(|v| v.is_finite()) || pc[2].abs() <= 1e-12 {
        return f64::INFINITY;
    }
    let u = intrinsics.fx * (pc[0] / pc[2]) + intrinsics.cx;
    let v = intrinsics.fy * (pc[1] / pc[2]) + intrinsics.cy;
    ((u - image_point.x).powi(2) + (v - image_point.y).powi(2)).sqrt()
}

/// Converts CameraExtrinsics to a 6D parameter vector [wx, wy, wz, tx, ty, tz]
///
/// The rotation is represented as an axis-angle (scaled axis)
fn extrinsics_to_params(ext: &CameraExtrinsics) -> [f64; 6] {
    let r = Rotation3::from_matrix_unchecked(ext.rotation);
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

/// Converts a 6D parameter vector back to CameraExtrinsics
///
/// Inverse of `extrinsics_to_params`. The first 3 components are axis-angle.
fn params_to_extrinsics(params: &[f64; 6]) -> CameraExtrinsics {
    let rot = Rotation3::new(Vector3::new(params[0], params[1], params[2])).into_inner();
    let t = Vector3::new(params[3], params[4], params[5]);
    CameraExtrinsics::new(rot, t)
}

/// Projects a 3D point onto the image plane
///
/// Applies camera extrinsics (world to camera) and intrinsics (camera to image)
fn project_point(intrinsics: &CameraIntrinsics, ext: &CameraExtrinsics, p: &Point3<f64>) -> Point2<f64> {
    let pc = ext.rotation * p.coords + ext.translation;
    Point2::new(
        intrinsics.fx * (pc[0] / pc[2]) + intrinsics.cx,
        intrinsics.fy * (pc[1] / pc[2]) + intrinsics.cy,
    )
}

/// Randomly samples k unique indices from [0, n) using a simple LCG
///
/// Uses a linear congruential generator seeded with the provided seed for determinism.
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
