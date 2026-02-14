use crate::{Result, StereoError};
use cv_core::{skew_symmetric, CameraExtrinsics, CameraIntrinsics};
use nalgebra::{
    DMatrix, Matrix3, Matrix3x4, Matrix4, Point2, Point3, Vector3,
};

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
}
