use crate::{Result, StereoError};
use cv_core::{skew_symmetric, CameraExtrinsics, CameraIntrinsics};
use nalgebra::{
    DMatrix, Matrix3, Matrix3x4, Matrix4, Point2, Point3, Vector3,
};

#[derive(Debug, Clone)]
pub struct StereoRectifyMatrices {
    pub r1: Matrix3<f64>,
    pub r2: Matrix3<f64>,
    pub p1: Matrix3x4<f64>,
    pub p2: Matrix3x4<f64>,
    pub q: Matrix4<f64>,
}

pub fn solve_pnp_dlt(
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> Result<CameraExtrinsics> {
    if object_points.len() != image_points.len() {
        return Err(StereoError::InvalidParameters(
            "object_points and image_points must have equal length".to_string(),
        ));
    }
    if object_points.len() < 6 {
        return Err(StereoError::InvalidParameters(
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
        StereoError::InvalidParameters("SVD failed in solve_pnp_dlt".to_string())
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
        StereoError::InvalidParameters("SVD U missing in solve_pnp_dlt".to_string())
    })?;
    let vt_m = svd_m.v_t.ok_or_else(|| {
        StereoError::InvalidParameters("SVD V^T missing in solve_pnp_dlt".to_string())
    })?;

    let mut r = u * vt_m;
    let scale = (svd_m.singular_values[0] + svd_m.singular_values[1] + svd_m.singular_values[2])
        / 3.0;
    if scale.abs() < 1e-12 {
        return Err(StereoError::InvalidParameters(
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

pub fn essential_from_extrinsics(extrinsics: &CameraExtrinsics) -> Matrix3<f64> {
    skew_symmetric(&extrinsics.translation) * extrinsics.rotation
}

pub fn fundamental_from_essential(
    essential: &Matrix3<f64>,
    intrinsics1: &CameraIntrinsics,
    intrinsics2: &CameraIntrinsics,
) -> Matrix3<f64> {
    let k1_inv = intrinsics1.inverse_matrix();
    let k2_inv_t = intrinsics2.inverse_matrix().transpose();
    k2_inv_t * essential * k1_inv
}

pub fn find_essential_mat(
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> Result<Matrix3<f64>> {
    if pts1.len() != pts2.len() || pts1.len() < 8 {
        return Err(StereoError::InvalidParameters(
            "find_essential_mat needs >=8 paired points".to_string(),
        ));
    }
    let (n1, n2) = normalize_with_intrinsics(pts1, pts2, intrinsics);
    estimate_essential_8_point(&n1, &n2)
}

pub fn find_essential_mat_ransac(
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    threshold_px: f64,
    max_iters: usize,
) -> Result<(Matrix3<f64>, Vec<bool>)> {
    if pts1.len() != pts2.len() || pts1.len() < 8 {
        return Err(StereoError::InvalidParameters(
            "find_essential_mat_ransac needs >=8 paired points".to_string(),
        ));
    }
    if threshold_px <= 0.0 {
        return Err(StereoError::InvalidParameters(
            "threshold_px must be > 0".to_string(),
        ));
    }

    let (n1, n2) = normalize_with_intrinsics(pts1, pts2, intrinsics);
    let n = n1.len();
    let f = 0.5 * (intrinsics.fx + intrinsics.fy);
    let thresh_norm = threshold_px / f.max(1e-12);
    let thresh2 = thresh_norm * thresh_norm;

    let mut best_inliers = vec![false; n];
    let mut best_count = 0usize;
    let mut best_e = None;

    let iters = max_iters.max(32);
    for i in 0..iters {
        let idx = sample_unique_indices(n, 8, i as u64 + 1);
        let s1: Vec<Point2<f64>> = idx.iter().map(|&j| n1[j]).collect();
        let s2: Vec<Point2<f64>> = idx.iter().map(|&j| n2[j]).collect();

        let e = match estimate_essential_8_point(&s1, &s2) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let mut mask = vec![false; n];
        let mut count = 0usize;
        for j in 0..n {
            let err = sampson_error(&e, &n1[j], &n2[j]);
            if err <= thresh2 {
                mask[j] = true;
                count += 1;
            }
        }

        if count > best_count {
            best_count = count;
            best_inliers = mask;
            best_e = Some(e);
        }
    }

    let best_e = best_e.ok_or_else(|| {
        StereoError::InvalidParameters("RANSAC failed to estimate essential matrix".to_string())
    })?;

    let in1: Vec<Point2<f64>> = n1
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(p, &m)| if m { Some(*p) } else { None })
        .collect();
    let in2: Vec<Point2<f64>> = n2
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(p, &m)| if m { Some(*p) } else { None })
        .collect();

    let refined = if in1.len() >= 8 {
        estimate_essential_8_point(&in1, &in2).unwrap_or(best_e)
    } else {
        best_e
    };

    Ok((refined, best_inliers))
}

fn normalize_with_intrinsics(
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> (Vec<Point2<f64>>, Vec<Point2<f64>>) {
    let k_inv = intrinsics.inverse_matrix();
    let n1: Vec<Point2<f64>> = pts1
        .iter()
        .map(|p| {
            let v = k_inv * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0] / v[2], v[1] / v[2])
        })
        .collect();
    let n2: Vec<Point2<f64>> = pts2
        .iter()
        .map(|p| {
            let v = k_inv * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0] / v[2], v[1] / v[2])
        })
        .collect();
    (n1, n2)
}

pub fn find_fundamental_mat(pts1: &[Point2<f64>], pts2: &[Point2<f64>]) -> Result<Matrix3<f64>> {
    if pts1.len() != pts2.len() || pts1.len() < 8 {
        return Err(StereoError::InvalidParameters(
            "find_fundamental_mat needs >=8 paired points".to_string(),
        ));
    }
    estimate_fundamental_8_point(pts1, pts2)
}

pub fn find_fundamental_mat_ransac(
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    threshold_px: f64,
    max_iters: usize,
) -> Result<(Matrix3<f64>, Vec<bool>)> {
    if pts1.len() != pts2.len() || pts1.len() < 8 {
        return Err(StereoError::InvalidParameters(
            "find_fundamental_mat_ransac needs >=8 paired points".to_string(),
        ));
    }
    if threshold_px <= 0.0 {
        return Err(StereoError::InvalidParameters(
            "threshold_px must be > 0".to_string(),
        ));
    }

    let n = pts1.len();
    let thresh2 = threshold_px * threshold_px;
    let mut best_inliers = vec![false; n];
    let mut best_count = 0usize;
    let mut best_f = None;
    let iters = max_iters.max(32);

    for i in 0..iters {
        let idx = sample_unique_indices(n, 8, i as u64 + 7);
        let s1: Vec<Point2<f64>> = idx.iter().map(|&j| pts1[j]).collect();
        let s2: Vec<Point2<f64>> = idx.iter().map(|&j| pts2[j]).collect();
        let f = match estimate_fundamental_8_point(&s1, &s2) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let mut mask = vec![false; n];
        let mut count = 0usize;
        for j in 0..n {
            let err = sampson_error(&f, &pts1[j], &pts2[j]);
            if err <= thresh2 {
                mask[j] = true;
                count += 1;
            }
        }

        if count > best_count {
            best_count = count;
            best_inliers = mask;
            best_f = Some(f);
        }
    }

    let best_f = best_f.ok_or_else(|| {
        StereoError::InvalidParameters("RANSAC failed to estimate fundamental matrix".to_string())
    })?;

    let in1: Vec<Point2<f64>> = pts1
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(p, &m)| if m { Some(*p) } else { None })
        .collect();
    let in2: Vec<Point2<f64>> = pts2
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(p, &m)| if m { Some(*p) } else { None })
        .collect();

    let refined = if in1.len() >= 8 {
        estimate_fundamental_8_point(&in1, &in2).unwrap_or(best_f)
    } else {
        best_f
    };
    Ok((refined, best_inliers))
}

pub fn stereo_rectify_matrices(
    left_intrinsics: &CameraIntrinsics,
    right_intrinsics: &CameraIntrinsics,
    left_extrinsics: &CameraExtrinsics,
    right_extrinsics: &CameraExtrinsics,
) -> Result<StereoRectifyMatrices> {
    let rel_r = left_extrinsics.rotation.transpose() * right_extrinsics.rotation;
    let rel_t = left_extrinsics.rotation.transpose()
        * (right_extrinsics.translation - left_extrinsics.translation);
    let baseline = rel_t.norm();
    if baseline <= 1e-12 {
        return Err(StereoError::InvalidParameters(
            "stereo_rectify_matrices requires non-zero baseline".to_string(),
        ));
    }

    let ex = rel_t / baseline;
    let helper = if ex[2].abs() < 0.9 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let ey = helper.cross(&ex).normalize();
    let ez = ex.cross(&ey).normalize();
    let basis = Matrix3::from_columns(&[ex, ey, ez]);
    let r_rect = basis.transpose();

    let r1 = r_rect;
    let r2 = r_rect * rel_r;

    let fx = 0.5 * (left_intrinsics.fx + right_intrinsics.fx);
    let fy = 0.5 * (left_intrinsics.fy + right_intrinsics.fy);
    let cx1 = 0.5 * (left_intrinsics.cx + right_intrinsics.cx);
    let cx2 = cx1;
    let cy = 0.5 * (left_intrinsics.cy + right_intrinsics.cy);
    let tx = -fx * baseline;

    let p1 = Matrix3x4::new(
        fx, 0.0, cx1, 0.0, //
        0.0, fy, cy, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );
    let p2 = Matrix3x4::new(
        fx, 0.0, cx2, tx, //
        0.0, fy, cy, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );

    let mut q = Matrix4::<f64>::zeros();
    q[(0, 0)] = 1.0;
    q[(0, 3)] = -cx1;
    q[(1, 1)] = 1.0;
    q[(1, 3)] = -cy;
    q[(2, 3)] = fx;
    q[(3, 2)] = -1.0 / tx;
    q[(3, 3)] = (cx1 - cx2) / tx;

    Ok(StereoRectifyMatrices { r1, r2, p1, p2, q })
}

fn estimate_essential_8_point(pts1: &[Point2<f64>], pts2: &[Point2<f64>]) -> Result<Matrix3<f64>> {
    if pts1.len() != pts2.len() || pts1.len() < 8 {
        return Err(StereoError::InvalidParameters(
            "estimate_essential_8_point needs >=8 paired points".to_string(),
        ));
    }

    let n = pts1.len();
    let mut a = DMatrix::<f64>::zeros(n, 9);
    for i in 0..n {
        let x1 = pts1[i].x;
        let y1 = pts1[i].y;
        let x2 = pts2[i].x;
        let y2 = pts2[i].y;
        a[(i, 0)] = x2 * x1;
        a[(i, 1)] = x2 * y1;
        a[(i, 2)] = x2;
        a[(i, 3)] = y2 * x1;
        a[(i, 4)] = y2 * y1;
        a[(i, 5)] = y2;
        a[(i, 6)] = x1;
        a[(i, 7)] = y1;
        a[(i, 8)] = 1.0;
    }

    let svd = a.svd(true, true);
    let vt = svd.v_t.ok_or_else(|| {
        StereoError::InvalidParameters("SVD failed in estimate_essential_8_point".to_string())
    })?;
    let evec = vt.row(vt.nrows() - 1);
    let e = Matrix3::new(
        evec[(0, 0)],
        evec[(0, 1)],
        evec[(0, 2)],
        evec[(0, 3)],
        evec[(0, 4)],
        evec[(0, 5)],
        evec[(0, 6)],
        evec[(0, 7)],
        evec[(0, 8)],
    );
    enforce_essential_constraints(&e)
}

fn enforce_essential_constraints(e: &Matrix3<f64>) -> Result<Matrix3<f64>> {
    let svd = e.svd(true, true);
    let u = svd.u.ok_or_else(|| {
        StereoError::InvalidParameters("SVD U missing in essential constraints".to_string())
    })?;
    let vt = svd.v_t.ok_or_else(|| {
        StereoError::InvalidParameters("SVD V^T missing in essential constraints".to_string())
    })?;
    let s = 0.5 * (svd.singular_values[0] + svd.singular_values[1]);
    let sigma = Matrix3::new(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0);
    Ok(u * sigma * vt)
}

fn estimate_fundamental_8_point(pts1: &[Point2<f64>], pts2: &[Point2<f64>]) -> Result<Matrix3<f64>> {
    let (n1, t1) = normalize_points_hartley(pts1)?;
    let (n2, t2) = normalize_points_hartley(pts2)?;
    let n = n1.len();
    let mut a = DMatrix::<f64>::zeros(n, 9);
    for i in 0..n {
        let x1 = n1[i].x;
        let y1 = n1[i].y;
        let x2 = n2[i].x;
        let y2 = n2[i].y;
        a[(i, 0)] = x2 * x1;
        a[(i, 1)] = x2 * y1;
        a[(i, 2)] = x2;
        a[(i, 3)] = y2 * x1;
        a[(i, 4)] = y2 * y1;
        a[(i, 5)] = y2;
        a[(i, 6)] = x1;
        a[(i, 7)] = y1;
        a[(i, 8)] = 1.0;
    }

    let svd = a.svd(true, true);
    let vt = svd.v_t.ok_or_else(|| {
        StereoError::InvalidParameters("SVD failed in estimate_fundamental_8_point".to_string())
    })?;
    let fvec = vt.row(vt.nrows() - 1);
    let f0 = Matrix3::new(
        fvec[(0, 0)],
        fvec[(0, 1)],
        fvec[(0, 2)],
        fvec[(0, 3)],
        fvec[(0, 4)],
        fvec[(0, 5)],
        fvec[(0, 6)],
        fvec[(0, 7)],
        fvec[(0, 8)],
    );
    let f_rank2 = enforce_rank2(&f0)?;
    let f = t2.transpose() * f_rank2 * t1;
    Ok(f)
}

fn normalize_points_hartley(pts: &[Point2<f64>]) -> Result<(Vec<Point2<f64>>, Matrix3<f64>)> {
    if pts.len() < 2 {
        return Err(StereoError::InvalidParameters(
            "normalize_points_hartley requires at least 2 points".to_string(),
        ));
    }

    let mx = pts.iter().map(|p| p.x).sum::<f64>() / pts.len() as f64;
    let my = pts.iter().map(|p| p.y).sum::<f64>() / pts.len() as f64;
    let mean_dist = pts
        .iter()
        .map(|p| ((p.x - mx) * (p.x - mx) + (p.y - my) * (p.y - my)).sqrt())
        .sum::<f64>()
        / pts.len() as f64;
    if mean_dist <= 1e-12 {
        return Err(StereoError::InvalidParameters(
            "degenerate points in normalize_points_hartley".to_string(),
        ));
    }

    let s = (2.0f64).sqrt() / mean_dist;
    let t = Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0);
    let out = pts
        .iter()
        .map(|p| {
            let v = t * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0], v[1])
        })
        .collect();
    Ok((out, t))
}

fn enforce_rank2(m: &Matrix3<f64>) -> Result<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let u = svd.u.ok_or_else(|| {
        StereoError::InvalidParameters("SVD U missing in enforce_rank2".to_string())
    })?;
    let vt = svd.v_t.ok_or_else(|| {
        StereoError::InvalidParameters("SVD V^T missing in enforce_rank2".to_string())
    })?;
    let sigma = Matrix3::new(
        svd.singular_values[0],
        0.0,
        0.0,
        0.0,
        svd.singular_values[1],
        0.0,
        0.0,
        0.0,
        0.0,
    );
    Ok(u * sigma * vt)
}

fn sampson_error(e: &Matrix3<f64>, p1: &Point2<f64>, p2: &Point2<f64>) -> f64 {
    let x1 = Vector3::new(p1.x, p1.y, 1.0);
    let x2 = Vector3::new(p2.x, p2.y, 1.0);
    let ex1 = e * x1;
    let etx2 = e.transpose() * x2;
    let x2tex1 = x2.dot(&ex1);
    let denom = ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1];
    if denom <= 1e-18 {
        f64::INFINITY
    } else {
        (x2tex1 * x2tex1) / denom
    }
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

pub fn triangulate_points(
    p1: &Matrix3x4<f64>,
    p2: &Matrix3x4<f64>,
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
) -> Result<Vec<Point3<f64>>> {
    if pts1.len() != pts2.len() {
        return Err(StereoError::InvalidParameters(
            "triangulate_points requires equal point counts".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(pts1.len());
    for (a, b) in pts1.iter().zip(pts2.iter()) {
        let mut m = Matrix4::<f64>::zeros();
        for c in 0..4 {
            m[(0, c)] = a.x * p1[(2, c)] - p1[(0, c)];
            m[(1, c)] = a.y * p1[(2, c)] - p1[(1, c)];
            m[(2, c)] = b.x * p2[(2, c)] - p2[(0, c)];
            m[(3, c)] = b.y * p2[(2, c)] - p2[(1, c)];
        }
        let svd = m.svd(true, true);
        let vt = svd.v_t.ok_or_else(|| {
            StereoError::InvalidParameters("SVD failed in triangulate_points".to_string())
        })?;
        let xh = vt.row(3);
        let w = xh[(0, 3)];
        if w.abs() < 1e-12 {
            out.push(Point3::new(0.0, 0.0, 0.0));
            continue;
        }
        out.push(Point3::new(xh[(0, 0)] / w, xh[(0, 1)] / w, xh[(0, 2)] / w));
    }

    Ok(out)
}

pub fn recover_pose_from_essential(
    essential: &Matrix3<f64>,
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> Result<CameraExtrinsics> {
    if pts1.len() != pts2.len() || pts1.len() < 5 {
        return Err(StereoError::InvalidParameters(
            "recover_pose_from_essential needs >=5 paired points".to_string(),
        ));
    }

    let svd = essential.svd(true, true);
    let mut u = svd.u.ok_or_else(|| {
        StereoError::InvalidParameters("SVD U missing in recover_pose_from_essential".to_string())
    })?;
    let mut vt = svd.v_t.ok_or_else(|| {
        StereoError::InvalidParameters("SVD V^T missing in recover_pose_from_essential".to_string())
    })?;

    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r1 = u * w * vt;
    let r2 = u * w.transpose() * vt;
    let t = u.column(2).into_owned();

    let candidates = [
        CameraExtrinsics::new(r1, t),
        CameraExtrinsics::new(r1, -t),
        CameraExtrinsics::new(r2, t),
        CameraExtrinsics::new(r2, -t),
    ];

    let k_inv = intrinsics.inverse_matrix();
    let norm1: Vec<Point2<f64>> = pts1
        .iter()
        .map(|p| {
            let v = k_inv * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0] / v[2], v[1] / v[2])
        })
        .collect();
    let norm2: Vec<Point2<f64>> = pts2
        .iter()
        .map(|p| {
            let v = k_inv * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0] / v[2], v[1] / v[2])
        })
        .collect();

    let p1 = Matrix3x4::new(
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );

    let mut best = None;
    let mut best_score = i32::MIN;
    for cand in candidates {
        let p2 = Matrix3x4::new(
            cand.rotation[(0, 0)],
            cand.rotation[(0, 1)],
            cand.rotation[(0, 2)],
            cand.translation[0],
            cand.rotation[(1, 0)],
            cand.rotation[(1, 1)],
            cand.rotation[(1, 2)],
            cand.translation[1],
            cand.rotation[(2, 0)],
            cand.rotation[(2, 1)],
            cand.rotation[(2, 2)],
            cand.translation[2],
        );

        let tri = triangulate_points(&p1, &p2, &norm1, &norm2)?;
        let mut score = 0i32;
        for x in &tri {
            let z1 = x.z;
            let x2 = cand.rotation * x.coords + cand.translation;
            let z2 = x2[2];
            if z1 > 0.0 && z2 > 0.0 {
                score += 1;
            }
        }
        if score > best_score {
            best_score = score;
            best = Some(cand);
        }
    }

    best.ok_or_else(|| StereoError::InvalidParameters("No valid pose candidate found".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    fn project_point(k: &CameraIntrinsics, ext: &CameraExtrinsics, p: &Point3<f64>) -> Point2<f64> {
        let pc = ext.rotation * p.coords + ext.translation;
        let u = k.fx * (pc[0] / pc[2]) + k.cx;
        let v = k.fy * (pc[1] / pc[2]) + k.cy;
        Point2::new(u, v)
    }

    #[test]
    fn triangulate_points_recovers_geometry() {
        let p1 = Matrix3x4::new(
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        );
        let p2 = Matrix3x4::new(
            1.0, 0.0, 0.0, 0.2, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        );
        let world = vec![
            Point3::new(0.0, 0.0, 3.0),
            Point3::new(0.2, -0.1, 4.0),
            Point3::new(-0.3, 0.2, 5.0),
        ];
        let pts1: Vec<Point2<f64>> = world
            .iter()
            .map(|p| Point2::new(p.x / p.z, p.y / p.z))
            .collect();
        let pts2: Vec<Point2<f64>> = world
            .iter()
            .map(|p| Point2::new((p.x + 0.2) / p.z, p.y / p.z))
            .collect();

        let out = triangulate_points(&p1, &p2, &pts1, &pts2).unwrap();
        for (a, b) in out.iter().zip(world.iter()) {
            assert!((a.coords - b.coords).norm() < 1e-6);
        }
    }

    #[test]
    fn solve_pnp_dlt_reprojects_well() {
        let k = CameraIntrinsics::new(800.0, 780.0, 320.0, 240.0, 640, 480);
        let rot = Rotation3::from_euler_angles(0.08, -0.04, 0.06)
            .matrix()
            .clone_owned();
        let t = Vector3::new(0.15, -0.1, 0.4);
        let gt = CameraExtrinsics::new(rot, t);

        let world = vec![
            Point3::new(-0.4, -0.2, 3.8),
            Point3::new(0.3, -0.1, 4.1),
            Point3::new(0.1, 0.2, 4.5),
            Point3::new(-0.2, 0.3, 3.9),
            Point3::new(0.4, 0.4, 4.7),
            Point3::new(-0.5, 0.1, 5.0),
            Point3::new(0.2, -0.4, 4.3),
            Point3::new(-0.1, -0.3, 5.2),
        ];
        let pixels: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &gt, p)).collect();

        let est = solve_pnp_dlt(&world, &pixels, &k).unwrap();
        let reproj_err = world
            .iter()
            .zip(pixels.iter())
            .map(|(w, p)| (project_point(&k, &est, w) - p).norm())
            .sum::<f64>()
            / world.len() as f64;
        assert!(reproj_err < 1e-6);
    }

    #[test]
    fn recover_pose_from_essential_selects_valid_candidate() {
        let k = CameraIntrinsics::new(700.0, 700.0, 320.0, 240.0, 640, 480);
        let rot = Rotation3::from_euler_angles(0.04, -0.03, 0.02)
            .matrix()
            .clone_owned();
        let t = Vector3::new(0.2, 0.0, 0.02).normalize();
        let gt = CameraExtrinsics::new(rot, t);
        let e = essential_from_extrinsics(&gt);

        let world = vec![
            Point3::new(-0.2, -0.1, 3.0),
            Point3::new(0.2, -0.2, 3.5),
            Point3::new(0.1, 0.15, 4.1),
            Point3::new(-0.3, 0.1, 4.4),
            Point3::new(0.25, 0.2, 3.7),
            Point3::new(-0.1, -0.25, 5.0),
        ];

        let i_ext = CameraExtrinsics::default();
        let pts1: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &i_ext, p)).collect();
        let pts2: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &gt, p)).collect();

        let recovered = recover_pose_from_essential(&e, &pts1, &pts2, &k).unwrap();
        assert!(recovered.rotation.determinant() > 0.0);
        let dir_dot = recovered.translation.normalize().dot(&gt.translation.normalize());
        assert!(dir_dot > 0.9);
    }

    #[test]
    fn find_essential_mat_ransac_handles_outliers() {
        let k = CameraIntrinsics::new(750.0, 760.0, 320.0, 240.0, 640, 480);
        let rot = Rotation3::from_euler_angles(0.03, -0.02, 0.01)
            .matrix()
            .clone_owned();
        let t = Vector3::new(0.18, -0.01, 0.02).normalize();
        let gt = CameraExtrinsics::new(rot, t);
        let i_ext = CameraExtrinsics::default();

        let mut world = vec![];
        for i in 0..20 {
            let x = -0.5 + 0.05 * i as f64;
            let y = -0.2 + 0.03 * (i % 7) as f64;
            let z = 3.0 + 0.2 * (i % 5) as f64;
            world.push(Point3::new(x, y, z));
        }
        let pts1: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &i_ext, p)).collect();
        let mut pts2: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &gt, p)).collect();

        // Inject outliers.
        for i in 0..5 {
            pts2[i] = Point2::new(50.0 + i as f64 * 20.0, 400.0 - i as f64 * 15.0);
        }

        let (e, inliers) = find_essential_mat_ransac(&pts1, &pts2, &k, 3.0, 600).unwrap();
        let inlier_count = inliers.iter().filter(|&&m| m).count();
        assert!(inlier_count >= 10);

        let in1: Vec<Point2<f64>> = pts1
            .iter()
            .zip(inliers.iter())
            .filter_map(|(p, &m)| if m { Some(*p) } else { None })
            .collect();
        let in2: Vec<Point2<f64>> = pts2
            .iter()
            .zip(inliers.iter())
            .filter_map(|(p, &m)| if m { Some(*p) } else { None })
            .collect();

        let recovered = recover_pose_from_essential(&e, &in1, &in2, &k).unwrap();
        assert!(recovered.rotation.determinant() > 0.0);
        assert!(recovered.translation.norm() > 1e-6);
    }

    #[test]
    fn find_fundamental_mat_ransac_handles_outliers() {
        let k = CameraIntrinsics::new(720.0, 710.0, 320.0, 240.0, 640, 480);
        let rot = Rotation3::from_euler_angles(0.02, -0.01, 0.015)
            .matrix()
            .clone_owned();
        let t = Vector3::new(0.15, 0.01, 0.0);
        let gt = CameraExtrinsics::new(rot, t);
        let i_ext = CameraExtrinsics::default();

        let mut world = vec![];
        for i in 0..24 {
            let x = -0.4 + 0.04 * i as f64;
            let y = -0.2 + 0.03 * (i % 6) as f64;
            let z = 2.8 + 0.2 * (i % 5) as f64;
            world.push(Point3::new(x, y, z));
        }
        let pts1: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &i_ext, p)).collect();
        let mut pts2: Vec<Point2<f64>> = world.iter().map(|p| project_point(&k, &gt, p)).collect();
        for i in 0..6 {
            pts2[i] = Point2::new(600.0 - i as f64 * 17.0, 30.0 + i as f64 * 11.0);
        }

        let (f, inliers) = find_fundamental_mat_ransac(&pts1, &pts2, 2.5, 600).unwrap();
        let inlier_count = inliers.iter().filter(|&&m| m).count();
        assert!(inlier_count >= 12);

        let mean_epi = pts1
            .iter()
            .zip(pts2.iter())
            .zip(inliers.iter())
            .filter_map(|((p1, p2), &m)| {
                if m {
                    let x1 = Vector3::new(p1.x, p1.y, 1.0);
                    let x2 = Vector3::new(p2.x, p2.y, 1.0);
                    Some((x2.dot(&(f * x1))).abs())
                } else {
                    None
                }
            })
            .sum::<f64>()
            / inlier_count as f64;
        assert!(mean_epi < 0.5);
    }

    #[test]
    fn stereo_rectify_matrices_has_expected_projection_shape() {
        let k1 = CameraIntrinsics::new(700.0, 700.0, 320.0, 240.0, 640, 480);
        let k2 = CameraIntrinsics::new(710.0, 705.0, 322.0, 241.0, 640, 480);
        let left = CameraExtrinsics::default();
        let right = CameraExtrinsics::new(Matrix3::identity(), Vector3::new(0.2, 0.0, 0.0));

        let rect = stereo_rectify_matrices(&k1, &k2, &left, &right).unwrap();
        assert!(rect.r1.determinant() > 0.0);
        assert!(rect.r2.determinant() > 0.0);
        assert!(rect.p2[(0, 3)] < 0.0);
        assert!(rect.q[(3, 2)].is_finite());
        assert!(rect.q[(3, 2)].abs() > 0.0);
    }
}
