use crate::{Result, StereoError};
use cv_core::{skew_symmetric, CameraExtrinsics, CameraIntrinsics};
use image::GrayImage;
use nalgebra::{
    DMatrix, Matrix2, Matrix3, Matrix3x4, Matrix4, Point2, Point3, SymmetricEigen, Vector2,
    Vector3,
};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct StereoRectifyMatrices {
    pub r1: Matrix3<f64>,
    pub r2: Matrix3<f64>,
    pub p1: Matrix3x4<f64>,
    pub p2: Matrix3x4<f64>,
    pub q: Matrix4<f64>,
}

#[derive(Debug, Clone)]
pub struct CameraCalibrationResult {
    pub intrinsics: CameraIntrinsics,
    pub extrinsics: Vec<CameraExtrinsics>,
    pub rms_reprojection_error: f64,
}

#[derive(Debug, Clone)]
pub struct StereoCalibrationResult {
    pub left: CameraCalibrationResult,
    pub right: CameraCalibrationResult,
    pub relative_extrinsics: CameraExtrinsics,
    pub essential_matrix: Matrix3<f64>,
    pub fundamental_matrix: Matrix3<f64>,
}

#[derive(Debug, Clone)]
pub struct CalibrationFileReport {
    pub total_images: usize,
    pub used_images: usize,
    pub rejected_images: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct StereoCalibrationFileReport {
    pub total_pairs: usize,
    pub used_pairs: usize,
    pub rejected_pairs: Vec<usize>,
}

pub fn generate_chessboard_object_points(
    pattern_size: (usize, usize),
    square_size: f64,
) -> Vec<Point3<f64>> {
    let (cols, rows) = pattern_size;
    let mut points = Vec::with_capacity(cols * rows);
    for y in 0..rows {
        for x in 0..cols {
            points.push(Point3::new(
                x as f64 * square_size,
                y as f64 * square_size,
                0.0,
            ));
        }
    }
    points
}

pub fn calibrate_camera_planar(
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
    image_size: (u32, u32),
) -> Result<CameraCalibrationResult> {
    if object_points.len() != image_points.len() || object_points.len() < 3 {
        return Err(StereoError::InvalidParameters(
            "calibrate_camera_planar needs >=3 views with matching point sets".to_string(),
        ));
    }

    let mut homographies = Vec::with_capacity(object_points.len());
    for (obj, img) in object_points.iter().zip(image_points.iter()) {
        if obj.len() != img.len() || obj.len() < 4 {
            return Err(StereoError::InvalidParameters(
                "each calibration view needs >=4 correspondences".to_string(),
            ));
        }
        if obj.iter().any(|p| p.z.abs() > 1e-9) {
            return Err(StereoError::InvalidParameters(
                "calibrate_camera_planar expects planar object points (z=0)".to_string(),
            ));
        }
        let obj2d: Vec<Point2<f64>> = obj.iter().map(|p| Point2::new(p.x, p.y)).collect();
        homographies.push(estimate_homography_dlt(&obj2d, img)?);
    }

    let k = intrinsics_from_planar_homographies(&homographies)?;
    let intrinsics = CameraIntrinsics::new(
        k[(0, 0)],
        k[(1, 1)],
        k[(0, 2)],
        k[(1, 2)],
        image_size.0,
        image_size.1,
    );
    let k_inv = intrinsics.inverse_matrix();
    let mut extrinsics = Vec::with_capacity(homographies.len());
    for h in &homographies {
        extrinsics.push(extrinsics_from_homography(&k_inv, h)?);
    }

    let rms = compute_rms_reprojection(&intrinsics, &extrinsics, object_points, image_points)?;
    let result = CameraCalibrationResult {
        intrinsics,
        extrinsics,
        rms_reprojection_error: rms,
    };
    if !is_valid_camera_calibration(&result) {
        return Err(StereoError::InvalidParameters(
            "calibrate_camera_planar produced non-finite or degenerate calibration".to_string(),
        ));
    }
    Ok(result)
}

pub fn calibrate_camera_from_chessboard_images(
    images: &[GrayImage],
    pattern_size: (usize, usize),
    square_size: f64,
) -> Result<CameraCalibrationResult> {
    if images.is_empty() {
        return Err(StereoError::InvalidParameters(
            "calibrate_camera_from_chessboard_images: images cannot be empty".to_string(),
        ));
    }
    let (w, h) = images[0].dimensions();
    if images.iter().any(|img| img.dimensions() != (w, h)) {
        return Err(StereoError::InvalidParameters(
            "all calibration images must have the same dimensions".to_string(),
        ));
    }

    let board = generate_chessboard_object_points(pattern_size, square_size);
    let mut object_points = Vec::new();
    let mut image_points = Vec::new();
    for img in images {
        if let Ok(corners) = find_chessboard_corners(img, pattern_size) {
            object_points.push(board.clone());
            image_points.push(corners);
        }
    }

    if object_points.len() < 3 {
        return Err(StereoError::InvalidParameters(format!(
            "need at least 3 valid chessboard frames, found {}",
            object_points.len()
        )));
    }

    calibrate_camera_planar(&object_points, &image_points, (w, h))
}

pub fn calibrate_camera_from_chessboard_files<P: AsRef<Path>>(
    image_paths: &[P],
    pattern_size: (usize, usize),
    square_size: f64,
) -> Result<(CameraCalibrationResult, CalibrationFileReport)> {
    if image_paths.is_empty() {
        return Err(StereoError::InvalidParameters(
            "calibration file list cannot be empty".to_string(),
        ));
    }

    let board = generate_chessboard_object_points(pattern_size, square_size);
    let mut object_points = Vec::new();
    let mut image_points = Vec::new();
    let mut rejected = Vec::new();
    let mut expected_dims = None;

    for (idx, path) in image_paths.iter().enumerate() {
        let img = match image::open(path) {
            Ok(i) => i.to_luma8(),
            Err(_) => {
                rejected.push(idx);
                continue;
            }
        };

        if let Some((w, h)) = expected_dims {
            if img.dimensions() != (w, h) {
                rejected.push(idx);
                continue;
            }
        } else {
            expected_dims = Some(img.dimensions());
        }

        match find_chessboard_corners(&img, pattern_size) {
            Ok(corners) => {
                object_points.push(board.clone());
                image_points.push(corners);
            }
            Err(_) => rejected.push(idx),
        }
    }

    if object_points.len() < 3 {
        return Err(StereoError::InvalidParameters(format!(
            "need at least 3 valid chessboard images, found {}",
            object_points.len()
        )));
    }
    let dims = expected_dims.ok_or_else(|| {
        StereoError::InvalidParameters("no readable images in provided file list".to_string())
    })?;

    let calib = calibrate_camera_planar(&object_points, &image_points, dims).map_err(|e| {
        StereoError::InvalidParameters(format!(
            "camera calibration failed for file subset (used {} / {} images): {}",
            object_points.len(),
            image_paths.len(),
            e
        ))
    })?;
    let report = CalibrationFileReport {
        total_images: image_paths.len(),
        used_images: object_points.len(),
        rejected_images: rejected,
    };
    Ok((calib, report))
}

pub fn stereo_calibrate_planar(
    object_points: &[Vec<Point3<f64>>],
    left_image_points: &[Vec<Point2<f64>>],
    right_image_points: &[Vec<Point2<f64>>],
    image_size: (u32, u32),
) -> Result<StereoCalibrationResult> {
    if object_points.len() != left_image_points.len()
        || object_points.len() != right_image_points.len()
    {
        return Err(StereoError::InvalidParameters(
            "stereo_calibrate_planar expects matching batch sizes".to_string(),
        ));
    }
    if object_points.len() < 3 {
        return Err(StereoError::InvalidParameters(
            "stereo_calibrate_planar needs at least 3 views".to_string(),
        ));
    }

    let left = calibrate_camera_planar(object_points, left_image_points, image_size)?;
    let right = calibrate_camera_planar(object_points, right_image_points, image_size)?;

    let n = left.extrinsics.len().min(right.extrinsics.len());
    if n == 0 {
        return Err(StereoError::InvalidParameters(
            "stereo_calibrate_planar: no usable extrinsics".to_string(),
        ));
    }

    let mut t_sum = Vector3::zeros();
    let mut r_sum = Matrix3::<f64>::zeros();
    for i in 0..n {
        let r_l = left.extrinsics[i].rotation;
        let t_l = left.extrinsics[i].translation;
        let r_r = right.extrinsics[i].rotation;
        let t_r = right.extrinsics[i].translation;

        let r_rel = r_r * r_l.transpose();
        let t_rel = t_r - r_rel * t_l;
        r_sum += r_rel;
        t_sum += t_rel;
    }
    t_sum /= n as f64;

    let svd = r_sum.svd(true, true);
    let u = svd.u.ok_or_else(|| {
        StereoError::InvalidParameters("SVD U missing in stereo_calibrate_planar".to_string())
    })?;
    let vt = svd.v_t.ok_or_else(|| {
        StereoError::InvalidParameters("SVD V^T missing in stereo_calibrate_planar".to_string())
    })?;
    let mut r = u * vt;
    if r.determinant() < 0.0 {
        r = -r;
    }

    let relative_extrinsics = CameraExtrinsics::new(r, t_sum);
    let essential_matrix = essential_from_extrinsics(&relative_extrinsics);
    let fundamental_matrix =
        fundamental_from_essential(&essential_matrix, &left.intrinsics, &right.intrinsics);

    Ok(StereoCalibrationResult {
        left,
        right,
        relative_extrinsics,
        essential_matrix,
        fundamental_matrix,
    })
}

pub fn stereo_calibrate_from_chessboard_images(
    left_images: &[GrayImage],
    right_images: &[GrayImage],
    pattern_size: (usize, usize),
    square_size: f64,
) -> Result<StereoCalibrationResult> {
    if left_images.len() != right_images.len() || left_images.is_empty() {
        return Err(StereoError::InvalidParameters(
            "left/right image lists must be non-empty and equal-sized".to_string(),
        ));
    }

    let (w, h) = left_images[0].dimensions();
    if left_images.iter().any(|i| i.dimensions() != (w, h))
        || right_images.iter().any(|i| i.dimensions() != (w, h))
    {
        return Err(StereoError::InvalidParameters(
            "all stereo calibration images must share the same dimensions".to_string(),
        ));
    }

    let board = generate_chessboard_object_points(pattern_size, square_size);
    let mut object_points = Vec::new();
    let mut left_points = Vec::new();
    let mut right_points = Vec::new();

    for (l, r) in left_images.iter().zip(right_images.iter()) {
        let cl = find_chessboard_corners(l, pattern_size);
        let cr = find_chessboard_corners(r, pattern_size);
        if let (Ok(pl), Ok(pr)) = (cl, cr) {
            object_points.push(board.clone());
            left_points.push(pl);
            right_points.push(pr);
        }
    }

    if object_points.len() < 3 {
        return Err(StereoError::InvalidParameters(format!(
            "need at least 3 valid stereo chessboard pairs, found {}",
            object_points.len()
        )));
    }

    stereo_calibrate_planar(&object_points, &left_points, &right_points, (w, h))
}

pub fn stereo_calibrate_from_chessboard_files<P: AsRef<Path>>(
    left_paths: &[P],
    right_paths: &[P],
    pattern_size: (usize, usize),
    square_size: f64,
) -> Result<(StereoCalibrationResult, StereoCalibrationFileReport)> {
    if left_paths.len() != right_paths.len() || left_paths.is_empty() {
        return Err(StereoError::InvalidParameters(
            "left/right file lists must be non-empty and equal-sized".to_string(),
        ));
    }

    let board = generate_chessboard_object_points(pattern_size, square_size);
    let mut object_points = Vec::new();
    let mut left_points = Vec::new();
    let mut right_points = Vec::new();
    let mut rejected = Vec::new();
    let mut expected_dims = None;

    for i in 0..left_paths.len() {
        let left = image::open(&left_paths[i]).map(|v| v.to_luma8());
        let right = image::open(&right_paths[i]).map(|v| v.to_luma8());
        let (left, right) = match (left, right) {
            (Ok(l), Ok(r)) => (l, r),
            _ => {
                rejected.push(i);
                continue;
            }
        };

        if let Some((w, h)) = expected_dims {
            if left.dimensions() != (w, h) || right.dimensions() != (w, h) {
                rejected.push(i);
                continue;
            }
        } else {
            expected_dims = Some(left.dimensions());
        }

        let cl = find_chessboard_corners(&left, pattern_size);
        let cr = find_chessboard_corners(&right, pattern_size);
        if let (Ok(pl), Ok(pr)) = (cl, cr) {
            object_points.push(board.clone());
            left_points.push(pl);
            right_points.push(pr);
        } else {
            rejected.push(i);
        }
    }

    if object_points.len() < 3 {
        return Err(StereoError::InvalidParameters(format!(
            "need at least 3 valid stereo pairs, found {}",
            object_points.len()
        )));
    }
    let dims = expected_dims.ok_or_else(|| {
        StereoError::InvalidParameters("no readable stereo pairs in provided file lists".to_string())
    })?;

    let calib =
        stereo_calibrate_planar(&object_points, &left_points, &right_points, dims).map_err(
            |e| {
                StereoError::InvalidParameters(format!(
                    "stereo calibration failed for file subset (used {} / {} pairs): {}",
                    object_points.len(),
                    left_paths.len(),
                    e
                ))
            },
        )?;
    let report = StereoCalibrationFileReport {
        total_pairs: left_paths.len(),
        used_pairs: object_points.len(),
        rejected_pairs: rejected,
    };
    Ok((calib, report))
}

pub fn find_chessboard_corners(
    image: &GrayImage,
    pattern_size: (usize, usize),
) -> Result<Vec<Point2<f64>>> {
    let (cols, rows) = pattern_size;
    let need = cols * rows;
    if cols < 2 || rows < 2 {
        return Err(StereoError::InvalidParameters(
            "pattern_size must be at least (2,2)".to_string(),
        ));
    }
    if image.width() < 8 || image.height() < 8 {
        return Err(StereoError::InvalidParameters(
            "image too small for chessboard detection".to_string(),
        ));
    }

    let (response, width, height) = harris_response(image, 0.04, 1);
    let max_r = response
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    if max_r <= 0.0 {
        return Err(StereoError::InvalidParameters(
            "no chessboard-like corners found".to_string(),
        ));
    }
    let threshold = max_r * 0.01;
    let mut cands = non_max_suppression_response(&response, width, height, threshold);
    if cands.len() < need {
        return Err(StereoError::InvalidParameters(format!(
            "insufficient corner candidates: found {}, need {need}",
            cands.len()
        )));
    }

    cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate((need * 10).max(need));
    let mut ordered = assign_grid_points(&cands, pattern_size)?;
    corner_subpix(image, &mut ordered, 3, 25, 1e-3)?;
    Ok(ordered)
}

pub fn corner_subpix(
    image: &GrayImage,
    corners: &mut [Point2<f64>],
    win_radius: usize,
    max_iters: usize,
    eps: f64,
) -> Result<()> {
    if win_radius == 0 {
        return Err(StereoError::InvalidParameters(
            "win_radius must be >= 1".to_string(),
        ));
    }
    let w = image.width() as i32;
    let h = image.height() as i32;
    for p in corners.iter_mut() {
        let mut x = p.x;
        let mut y = p.y;
        for _ in 0..max_iters {
            let mut sw = 0.0f64;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let cx = x.round() as i32;
            let cy = y.round() as i32;
            for dy in -(win_radius as i32)..=(win_radius as i32) {
                for dx in -(win_radius as i32)..=(win_radius as i32) {
                    let xx = cx + dx;
                    let yy = cy + dy;
                    if xx <= 0 || yy <= 0 || xx >= w - 1 || yy >= h - 1 {
                        continue;
                    }
                    let gx = (image.get_pixel((xx + 1) as u32, yy as u32)[0] as f64
                        - image.get_pixel((xx - 1) as u32, yy as u32)[0] as f64)
                        * 0.5;
                    let gy = (image.get_pixel(xx as u32, (yy + 1) as u32)[0] as f64
                        - image.get_pixel(xx as u32, (yy - 1) as u32)[0] as f64)
                        * 0.5;
                    let wgt = (gx * gx + gy * gy).sqrt();
                    if wgt <= 1e-9 {
                        continue;
                    }
                    sw += wgt;
                    sx += wgt * xx as f64;
                    sy += wgt * yy as f64;
                }
            }
            if sw <= 1e-9 {
                break;
            }
            let nx = sx / sw;
            let ny = sy / sw;
            let shift = ((nx - x) * (nx - x) + (ny - y) * (ny - y)).sqrt();
            x = nx;
            y = ny;
            if shift < eps {
                break;
            }
        }
        p.x = x.clamp(0.0, (image.width() - 1) as f64);
        p.y = y.clamp(0.0, (image.height() - 1) as f64);
    }
    Ok(())
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

fn estimate_homography_dlt(src: &[Point2<f64>], dst: &[Point2<f64>]) -> Result<Matrix3<f64>> {
    if src.len() != dst.len() || src.len() < 4 {
        return Err(StereoError::InvalidParameters(
            "estimate_homography_dlt needs >=4 paired points".to_string(),
        ));
    }

    let (src_n, ts) = normalize_points_hartley(src)?;
    let (dst_n, td) = normalize_points_hartley(dst)?;
    let n = src.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for i in 0..n {
        let x = src_n[i].x;
        let y = src_n[i].y;
        let u = dst_n[i].x;
        let v = dst_n[i].y;
        let r0 = 2 * i;
        let r1 = r0 + 1;
        a[(r0, 0)] = -x;
        a[(r0, 1)] = -y;
        a[(r0, 2)] = -1.0;
        a[(r0, 6)] = u * x;
        a[(r0, 7)] = u * y;
        a[(r0, 8)] = u;

        a[(r1, 3)] = -x;
        a[(r1, 4)] = -y;
        a[(r1, 5)] = -1.0;
        a[(r1, 6)] = v * x;
        a[(r1, 7)] = v * y;
        a[(r1, 8)] = v;
    }

    let svd = a.svd(true, true);
    let vt = svd.v_t.ok_or_else(|| {
        StereoError::InvalidParameters("SVD failed in estimate_homography_dlt".to_string())
    })?;
    let h = vt.row(vt.nrows() - 1);
    let hn = Matrix3::new(
        h[(0, 0)],
        h[(0, 1)],
        h[(0, 2)],
        h[(0, 3)],
        h[(0, 4)],
        h[(0, 5)],
        h[(0, 6)],
        h[(0, 7)],
        h[(0, 8)],
    );
    let mut hdenorm = td.try_inverse().unwrap_or(Matrix3::identity()) * hn * ts;
    if hdenorm[(2, 2)].abs() > 1e-12 {
        hdenorm /= hdenorm[(2, 2)];
    }
    Ok(hdenorm)
}

fn intrinsics_from_planar_homographies(homographies: &[Matrix3<f64>]) -> Result<Matrix3<f64>> {
    if homographies.len() < 3 {
        return Err(StereoError::InvalidParameters(
            "need at least 3 homographies for planar calibration".to_string(),
        ));
    }

    let mut v = DMatrix::<f64>::zeros(2 * homographies.len(), 6);
    for (i, h) in homographies.iter().enumerate() {
        let v12 = v_ij(h, 0, 1);
        let v11 = v_ij(h, 0, 0);
        let v22 = v_ij(h, 1, 1);
        for j in 0..6 {
            v[(2 * i, j)] = v12[j];
            v[(2 * i + 1, j)] = v11[j] - v22[j];
        }
    }

    let svd = v.svd(true, true);
    let vt = svd.v_t.ok_or_else(|| {
        StereoError::InvalidParameters(
            "SVD failed in intrinsics_from_planar_homographies".to_string(),
        )
    })?;
    let b = vt.row(vt.nrows() - 1);
    let mut b11 = b[(0, 0)];
    let mut b12 = b[(0, 1)];
    let mut b22 = b[(0, 2)];
    let mut b13 = b[(0, 3)];
    let mut b23 = b[(0, 4)];
    let mut b33 = b[(0, 5)];

    let mut denom = b11 * b22 - b12 * b12;
    if denom.abs() < 1e-18 || b11.abs() < 1e-18 {
        return Err(StereoError::InvalidParameters(
            "degenerate calibration system".to_string(),
        ));
    }

    let mut v0 = (b12 * b13 - b11 * b23) / denom;
    let mut lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;

    // Nullspace sign is arbitrary; flip once if needed.
    if lambda <= 0.0 {
        b11 = -b11;
        b12 = -b12;
        b22 = -b22;
        b13 = -b13;
        b23 = -b23;
        b33 = -b33;
        denom = b11 * b22 - b12 * b12;
        if denom.abs() < 1e-18 || b11.abs() < 1e-18 {
            return Err(StereoError::InvalidParameters(
                "degenerate calibration system after sign flip".to_string(),
            ));
        }
        v0 = (b12 * b13 - b11 * b23) / denom;
        lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;
    }
    if lambda <= 0.0 {
        return Err(StereoError::InvalidParameters(
            "invalid lambda in planar calibration".to_string(),
        ));
    }
    let alpha = (lambda / b11).sqrt();
    let beta = (lambda * b11 / denom).sqrt();
    let gamma = -b12 * alpha * alpha * beta / lambda;
    let u0 = gamma * v0 / beta - b13 * alpha * alpha / lambda;

    Ok(Matrix3::new(alpha, gamma, u0, 0.0, beta, v0, 0.0, 0.0, 1.0))
}

fn extrinsics_from_homography(k_inv: &Matrix3<f64>, h: &Matrix3<f64>) -> Result<CameraExtrinsics> {
    let h1 = h.column(0).into_owned();
    let h2 = h.column(1).into_owned();
    let h3 = h.column(2).into_owned();

    let r1_raw = k_inv * h1;
    let r2_raw = k_inv * h2;
    let t_raw = k_inv * h3;
    let scale = 1.0 / r1_raw.norm().max(1e-18);

    let r1 = r1_raw * scale;
    let r2 = r2_raw * scale;
    let r3 = r1.cross(&r2);
    let mut r = Matrix3::from_columns(&[r1, r2, r3]);

    let svd = r.svd(true, true);
    let u = svd.u.ok_or_else(|| {
        StereoError::InvalidParameters("SVD U missing in extrinsics_from_homography".to_string())
    })?;
    let vt = svd.v_t.ok_or_else(|| {
        StereoError::InvalidParameters("SVD V^T missing in extrinsics_from_homography".to_string())
    })?;
    r = u * vt;
    if r.determinant() < 0.0 {
        r = -r;
    }

    let t = t_raw * scale;
    Ok(CameraExtrinsics::new(r, t))
}

fn compute_rms_reprojection(
    intrinsics: &CameraIntrinsics,
    extrinsics: &[CameraExtrinsics],
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
) -> Result<f64> {
    if extrinsics.len() != object_points.len() || object_points.len() != image_points.len() {
        return Err(StereoError::InvalidParameters(
            "compute_rms_reprojection: mismatched batch sizes".to_string(),
        ));
    }

    let mut sq_sum = 0.0f64;
    let mut count = 0usize;
    for ((ext, obj), img) in extrinsics
        .iter()
        .zip(object_points.iter())
        .zip(image_points.iter())
    {
        for (p3, p2) in obj.iter().zip(img.iter()) {
            let pc = ext.rotation * p3.coords + ext.translation;
            if pc[2].abs() <= 1e-18 {
                continue;
            }
            let u = intrinsics.fx * (pc[0] / pc[2]) + intrinsics.cx;
            let v = intrinsics.fy * (pc[1] / pc[2]) + intrinsics.cy;
            let du = u - p2.x;
            let dv = v - p2.y;
            sq_sum += du * du + dv * dv;
            count += 1;
        }
    }
    if count == 0 {
        return Err(StereoError::InvalidParameters(
            "compute_rms_reprojection: no valid points".to_string(),
        ));
    }
    Ok((sq_sum / count as f64).sqrt())
}

fn is_valid_camera_calibration(result: &CameraCalibrationResult) -> bool {
    let k = &result.intrinsics;
    let intrinsics_valid = k.fx.is_finite()
        && k.fy.is_finite()
        && k.cx.is_finite()
        && k.cy.is_finite()
        && k.fx.abs() > 1e-12
        && k.fy.abs() > 1e-12;
    if !intrinsics_valid || !result.rms_reprojection_error.is_finite() {
        return false;
    }

    result.extrinsics.iter().all(|ext| {
        ext.rotation.iter().all(|v| v.is_finite()) && ext.translation.iter().all(|v| v.is_finite())
    })
}

pub fn refine_camera_calibration_iterative(
    initial: &CameraCalibrationResult,
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
    max_iters: usize,
) -> Result<CameraCalibrationResult> {
    if object_points.len() != image_points.len() || object_points.len() != initial.extrinsics.len() {
        return Err(StereoError::InvalidParameters(
            "refine_camera_calibration_iterative: inconsistent input sizes".to_string(),
        ));
    }

    let mut intr = initial.intrinsics;
    let mut extr = initial.extrinsics.clone();
    let mut prev = compute_rms_reprojection(&intr, &extr, object_points, image_points)?;

    for _ in 0..max_iters {
        intr = estimate_intrinsics_from_extrinsics(&extr, object_points, image_points, intr)?;

        for i in 0..extr.len() {
            extr[i] = solve_pnp_dlt(&object_points[i], &image_points[i], &intr)?;
        }

        let cur = compute_rms_reprojection(&intr, &extr, object_points, image_points)?;
        if (prev - cur).abs() < 1e-10 {
            prev = cur;
            break;
        }
        prev = cur;
    }

    Ok(CameraCalibrationResult {
        intrinsics: intr,
        extrinsics: extr,
        rms_reprojection_error: prev,
    })
}

fn estimate_intrinsics_from_extrinsics(
    extrinsics: &[CameraExtrinsics],
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
    fallback: CameraIntrinsics,
) -> Result<CameraIntrinsics> {
    let mut sx2 = 0.0f64;
    let mut sxu = 0.0f64;
    let mut sx = 0.0f64;
    let mut su = 0.0f64;
    let mut n_x = 0usize;

    let mut sy2 = 0.0f64;
    let mut syv = 0.0f64;
    let mut sy = 0.0f64;
    let mut sv = 0.0f64;
    let mut n_y = 0usize;

    for ((ext, obj), img) in extrinsics
        .iter()
        .zip(object_points.iter())
        .zip(image_points.iter())
    {
        for (p3, p2) in obj.iter().zip(img.iter()) {
            let pc = ext.rotation * p3.coords + ext.translation;
            if pc[2].abs() <= 1e-12 {
                continue;
            }
            let xn = pc[0] / pc[2];
            let yn = pc[1] / pc[2];

            sx2 += xn * xn;
            sxu += xn * p2.x;
            sx += xn;
            su += p2.x;
            n_x += 1;

            sy2 += yn * yn;
            syv += yn * p2.y;
            sy += yn;
            sv += p2.y;
            n_y += 1;
        }
    }

    if n_x < 2 || n_y < 2 {
        return Err(StereoError::InvalidParameters(
            "estimate_intrinsics_from_extrinsics: insufficient valid points".to_string(),
        ));
    }

    let det_x = sx2 * n_x as f64 - sx * sx;
    let det_y = sy2 * n_y as f64 - sy * sy;
    if det_x.abs() < 1e-18 || det_y.abs() < 1e-18 {
        return Ok(fallback);
    }

    let fx = (sxu * n_x as f64 - sx * su) / det_x;
    let cx = (sx2 * su - sx * sxu) / det_x;
    let fy = (syv * n_y as f64 - sy * sv) / det_y;
    let cy = (sy2 * sv - sy * syv) / det_y;

    if !fx.is_finite() || !fy.is_finite() || fx.abs() < 1e-12 || fy.abs() < 1e-12 {
        return Ok(fallback);
    }

    Ok(CameraIntrinsics::new(
        fx,
        fy,
        cx,
        cy,
        fallback.width,
        fallback.height,
    ))
}

fn v_ij(h: &Matrix3<f64>, i: usize, j: usize) -> [f64; 6] {
    [
        h[(0, i)] * h[(0, j)],
        h[(0, i)] * h[(1, j)] + h[(1, i)] * h[(0, j)],
        h[(1, i)] * h[(1, j)],
        h[(2, i)] * h[(0, j)] + h[(0, i)] * h[(2, j)],
        h[(2, i)] * h[(1, j)] + h[(1, i)] * h[(2, j)],
        h[(2, i)] * h[(2, j)],
    ]
}

fn harris_response(image: &GrayImage, k: f64, win_radius: usize) -> (Vec<f64>, usize, usize) {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut ix = vec![0.0f64; width * height];
    let mut iy = vec![0.0f64; width * height];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = image.get_pixel((x + 1) as u32, y as u32)[0] as f64
                - image.get_pixel((x - 1) as u32, y as u32)[0] as f64;
            let gy = image.get_pixel(x as u32, (y + 1) as u32)[0] as f64
                - image.get_pixel(x as u32, (y - 1) as u32)[0] as f64;
            ix[y * width + x] = gx * 0.5;
            iy[y * width + x] = gy * 0.5;
        }
    }

    let mut resp = vec![0.0f64; width * height];
    let r = win_radius as i32;
    for y in win_radius..(height - win_radius) {
        for x in win_radius..(width - win_radius) {
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            let mut syy = 0.0;
            for dy in -r..=r {
                for dx in -r..=r {
                    let xx = (x as i32 + dx) as usize;
                    let yy = (y as i32 + dy) as usize;
                    let gx = ix[yy * width + xx];
                    let gy = iy[yy * width + xx];
                    sxx += gx * gx;
                    sxy += gx * gy;
                    syy += gy * gy;
                }
            }
            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            resp[y * width + x] = det - k * trace * trace;
        }
    }
    (resp, width, height)
}

fn non_max_suppression_response(
    response: &[f64],
    width: usize,
    height: usize,
    threshold: f64,
) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::new();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let r = response[y * width + x];
            if r <= threshold {
                continue;
            }
            let mut is_max = true;
            for yy in (y - 1)..=(y + 1) {
                for xx in (x - 1)..=(x + 1) {
                    if (xx != x || yy != y) && response[yy * width + xx] > r {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }
            if is_max {
                out.push((x as f64, y as f64, r));
            }
        }
    }
    out
}

fn assign_grid_points(
    candidates: &[(f64, f64, f64)],
    pattern_size: (usize, usize),
) -> Result<Vec<Point2<f64>>> {
    let (cols, rows) = pattern_size;
    let points: Vec<Vector2<f64>> = candidates
        .iter()
        .map(|(x, y, _)| Vector2::new(*x, *y))
        .collect();
    if points.len() < cols * rows {
        return Err(StereoError::InvalidParameters(
            "not enough candidates to assign grid".to_string(),
        ));
    }

    let mean = points
        .iter()
        .fold(Vector2::zeros(), |acc, p| acc + p)
        / points.len() as f64;
    let mut cov = Matrix2::<f64>::zeros();
    for p in &points {
        let d = p - mean;
        cov += d * d.transpose();
    }
    cov /= points.len() as f64;
    let eig = SymmetricEigen::new(cov);
    let (i0, i1) = if eig.eigenvalues[0] >= eig.eigenvalues[1] {
        (0usize, 1usize)
    } else {
        (1usize, 0usize)
    };
    let e0 = eig.eigenvectors.column(i0).into_owned();
    let e1 = eig.eigenvectors.column(i1).into_owned();

    let mut uv = Vec::with_capacity(points.len());
    for p in &points {
        let d = p - mean;
        uv.push((d.dot(&e0), d.dot(&e1)));
    }
    let u_vals: Vec<f64> = uv.iter().map(|(u, _)| *u).collect();
    let v_vals: Vec<f64> = uv.iter().map(|(_, v)| *v).collect();
    let mut u_centers = kmeans_1d(&u_vals, cols, 30);
    let mut v_centers = kmeans_1d(&v_vals, rows, 30);
    u_centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v_centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut used = vec![false; points.len()];
    let mut out = Vec::with_capacity(cols * rows);
    for vc in &v_centers {
        for uc in &u_centers {
            let mut best = None;
            let mut best_cost = f64::INFINITY;
            for (i, (u, v)) in uv.iter().enumerate() {
                if used[i] {
                    continue;
                }
                let du = u - uc;
                let dv = v - vc;
                let cost = du * du + dv * dv;
                if cost < best_cost {
                    best_cost = cost;
                    best = Some(i);
                }
            }
            let idx = best.ok_or_else(|| {
                StereoError::InvalidParameters("failed to assign all chessboard corners".to_string())
            })?;
            used[idx] = true;
            out.push(Point2::new(points[idx][0], points[idx][1]));
        }
    }
    Ok(out)
}

fn kmeans_1d(values: &[f64], k: usize, iters: usize) -> Vec<f64> {
    let min_v = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if k == 1 || (max_v - min_v).abs() < 1e-12 {
        return vec![0.5 * (min_v + max_v); k];
    }

    let mut centers = (0..k)
        .map(|i| min_v + (i as f64) * (max_v - min_v) / (k as f64 - 1.0))
        .collect::<Vec<_>>();

    for _ in 0..iters {
        let mut sums = vec![0.0f64; k];
        let mut cnts = vec![0usize; k];
        for &v in values {
            let mut bi = 0usize;
            let mut bd = (v - centers[0]).abs();
            for (i, &c) in centers.iter().enumerate().skip(1) {
                let d = (v - c).abs();
                if d < bd {
                    bd = d;
                    bi = i;
                }
            }
            sums[bi] += v;
            cnts[bi] += 1;
        }
        for i in 0..k {
            if cnts[i] > 0 {
                centers[i] = sums[i] / cnts[i] as f64;
            }
        }
    }
    centers
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
    use cv_imgproc::{warp_perspective_ex, BorderMode, Interpolation};
    use image::Luma;
    use nalgebra::Rotation3;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn project_point(k: &CameraIntrinsics, ext: &CameraExtrinsics, p: &Point3<f64>) -> Point2<f64> {
        let pc = ext.rotation * p.coords + ext.translation;
        let u = k.fx * (pc[0] / pc[2]) + k.cx;
        let v = k.fy * (pc[1] / pc[2]) + k.cy;
        Point2::new(u, v)
    }

    fn synthetic_checkerboard(
        pattern: (usize, usize),
        square: u32,
        margin_x: u32,
        margin_y: u32,
    ) -> (GrayImage, Vec<Point2<f64>>) {
        let (cols, rows) = pattern;
        let squares_x = cols as u32 + 1;
        let squares_y = rows as u32 + 1;
        let width = margin_x * 2 + squares_x * square;
        let height = margin_y * 2 + squares_y * square;
        let mut img = GrayImage::from_pixel(width, height, Luma([180]));

        for sy in 0..squares_y {
            for sx in 0..squares_x {
                let is_black = (sx + sy) % 2 == 0;
                let val = if is_black { 30u8 } else { 220u8 };
                let x0 = margin_x + sx * square;
                let y0 = margin_y + sy * square;
                for y in y0..(y0 + square) {
                    for x in x0..(x0 + square) {
                        img.put_pixel(x, y, Luma([val]));
                    }
                }
            }
        }

        let mut gt = Vec::with_capacity(cols * rows);
        for y in 0..rows {
            for x in 0..cols {
                gt.push(Point2::new(
                    (margin_x + (x as u32 + 1) * square) as f64,
                    (margin_y + (y as u32 + 1) * square) as f64,
                ));
            }
        }
        (img, gt)
    }

    fn synthetic_checkerboard_fixed(
        pattern: (usize, usize),
        square: u32,
        width: u32,
        height: u32,
        offset_x: u32,
        offset_y: u32,
    ) -> GrayImage {
        let (cols, rows) = pattern;
        let squares_x = cols as u32 + 1;
        let squares_y = rows as u32 + 1;
        let mut img = GrayImage::from_pixel(width, height, Luma([180]));

        for sy in 0..squares_y {
            for sx in 0..squares_x {
                let is_black = (sx + sy) % 2 == 0;
                let val = if is_black { 30u8 } else { 220u8 };
                let x0 = offset_x + sx * square;
                let y0 = offset_y + sy * square;
                if x0 + square > width || y0 + square > height {
                    continue;
                }
                for y in y0..(y0 + square) {
                    for x in x0..(x0 + square) {
                        img.put_pixel(x, y, Luma([val]));
                    }
                }
            }
        }
        img
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

    #[test]
    fn generate_chessboard_object_points_layout() {
        let pts = generate_chessboard_object_points((4, 3), 0.05);
        assert_eq!(pts.len(), 12);
        assert!((pts[0].coords - Point3::new(0.0, 0.0, 0.0).coords).norm() < 1e-12);
        assert!((pts[3].coords - Point3::new(0.15, 0.0, 0.0).coords).norm() < 1e-12);
        assert!((pts[11].coords - Point3::new(0.15, 0.10, 0.0).coords).norm() < 1e-12);
    }

    #[test]
    fn calibrate_camera_planar_recovers_intrinsics() {
        let board = generate_chessboard_object_points((7, 6), 0.04);
        let gt_k = CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
        let views = [
            CameraExtrinsics::new(
                Rotation3::from_euler_angles(0.08, -0.03, 0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.05, -0.03, 2.6),
            ),
            CameraExtrinsics::new(
                Rotation3::from_euler_angles(-0.06, 0.04, -0.05)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.08, 0.02, 2.9),
            ),
            CameraExtrinsics::new(
                Rotation3::from_euler_angles(0.03, 0.07, -0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.02, 0.06, 2.4),
            ),
            CameraExtrinsics::new(
                Rotation3::from_euler_angles(-0.04, -0.05, 0.04)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.03, -0.05, 3.1),
            ),
        ];

        let mut obj_sets = Vec::new();
        let mut img_sets = Vec::new();
        for ext in &views {
            obj_sets.push(board.clone());
            img_sets.push(board.iter().map(|p| project_point(&gt_k, ext, p)).collect());
        }

        let calib = calibrate_camera_planar(&obj_sets, &img_sets, (640, 480)).unwrap();
        assert!((calib.intrinsics.fx - gt_k.fx).abs() < 1e-2);
        assert!((calib.intrinsics.fy - gt_k.fy).abs() < 1e-2);
        assert!((calib.intrinsics.cx - gt_k.cx).abs() < 1e-2);
        assert!((calib.intrinsics.cy - gt_k.cy).abs() < 1e-2);
        assert!(calib.rms_reprojection_error < 1e-5);
        assert_eq!(calib.extrinsics.len(), views.len());
    }

    #[test]
    fn stereo_calibrate_planar_recovers_relative_transform() {
        let board = generate_chessboard_object_points((7, 6), 0.04);
        let k_l = CameraIntrinsics::new(810.0, 800.0, 320.0, 240.0, 640, 480);
        let k_r = CameraIntrinsics::new(815.0, 805.0, 318.0, 242.0, 640, 480);
        let r_lr = Rotation3::from_euler_angles(0.01, -0.015, 0.005)
            .matrix()
            .clone_owned();
        let t_lr = Vector3::new(0.20, 0.002, -0.001);

        let board_poses = [
            CameraExtrinsics::new(
                Rotation3::from_euler_angles(0.08, -0.03, 0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.05, -0.03, 2.6),
            ),
            CameraExtrinsics::new(
                Rotation3::from_euler_angles(-0.06, 0.04, -0.05)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.08, 0.02, 2.9),
            ),
            CameraExtrinsics::new(
                Rotation3::from_euler_angles(0.03, 0.07, -0.02)
                    .matrix()
                    .clone_owned(),
                Vector3::new(0.02, 0.06, 2.4),
            ),
            CameraExtrinsics::new(
                Rotation3::from_euler_angles(-0.04, -0.05, 0.04)
                    .matrix()
                    .clone_owned(),
                Vector3::new(-0.03, -0.05, 3.1),
            ),
        ];

        let mut obj_sets = Vec::new();
        let mut left_sets = Vec::new();
        let mut right_sets = Vec::new();
        for ext_l in &board_poses {
            let ext_r = CameraExtrinsics::new(r_lr * ext_l.rotation, r_lr * ext_l.translation + t_lr);
            obj_sets.push(board.clone());
            left_sets.push(board.iter().map(|p| project_point(&k_l, ext_l, p)).collect());
            right_sets.push(board.iter().map(|p| project_point(&k_r, &ext_r, p)).collect());
        }

        let out = stereo_calibrate_planar(&obj_sets, &left_sets, &right_sets, (640, 480)).unwrap();
        let t_err = (out.relative_extrinsics.translation - t_lr).norm();
        let r_err = (out.relative_extrinsics.rotation - r_lr).norm();
        assert!(t_err < 1e-2);
        assert!(r_err < 1e-2);
    }

    #[test]
    fn find_chessboard_corners_detects_expected_count() {
        let pattern = (7, 6);
        let (img, gt) = synthetic_checkerboard(pattern, 20, 40, 30);
        let corners = find_chessboard_corners(&img, pattern).unwrap();
        assert_eq!(corners.len(), pattern.0 * pattern.1);

        // Validate board-like coverage even if ordering and exact localization vary.
        let min_x = corners.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let max_x = corners.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let min_y = corners.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let max_y = corners.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

        let gt_min_x = gt.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let gt_max_x = gt.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let gt_min_y = gt.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let gt_max_y = gt.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

        assert!((min_x - gt_min_x).abs() < 25.0);
        assert!((max_x - gt_max_x).abs() < 25.0);
        assert!((min_y - gt_min_y).abs() < 25.0);
        assert!((max_y - gt_max_y).abs() < 25.0);
    }

    #[test]
    fn corner_subpix_refines_toward_local_corner() {
        let pattern = (7, 6);
        let (img, gt) = synthetic_checkerboard(pattern, 24, 30, 30);
        let mut p = vec![Point2::new(gt[10].x + 2.3, gt[10].y - 1.9)];
        let before = (p[0] - gt[10]).norm();
        corner_subpix(&img, &mut p, 4, 40, 1e-4).unwrap();
        let after = (p[0] - gt[10]).norm();
        assert!(after < before);
    }

    #[test]
    fn calibrate_camera_from_chessboard_files_reports_usage() {
        let pattern = (7, 6);
        let mut paths: Vec<PathBuf> = Vec::new();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let base = synthetic_checkerboard_fixed(pattern, 16, 320, 240, 70, 50);
        let transforms = [
            Matrix3::new(1.00, 0.00, 0.0, 0.00, 1.00, 0.0, 0.0000, 0.0000, 1.0),
            Matrix3::new(0.92, -0.03, 14.0, 0.04, 1.00, -9.0, 0.0024, -0.0012, 1.0),
            Matrix3::new(1.06, 0.06, -20.0, -0.03, 0.94, 13.0, -0.0020, 0.0015, 1.0),
            Matrix3::new(0.89, 0.02, 18.0, -0.05, 1.08, 8.0, 0.0016, 0.0021, 1.0),
            Matrix3::new(1.04, -0.07, -15.0, 0.03, 0.90, -7.0, -0.0018, -0.0013, 1.0),
            Matrix3::new(0.95, 0.05, 10.0, 0.02, 1.04, 6.0, 0.0012, -0.0020, 1.0),
        ];

        for (i, m) in transforms.iter().enumerate() {
            let img = warp_perspective_ex(
                &base,
                m,
                320,
                240,
                Interpolation::Linear,
                BorderMode::Constant(180),
            );
            let p = std::env::temp_dir().join(format!(
                "rustcv_calib_{}_{}_{}.png",
                std::process::id(),
                stamp,
                i
            ));
            img.save(&p).unwrap();
            paths.push(p);
        }

        paths.push(std::env::temp_dir().join(format!(
            "rustcv_calib_missing_{}_{}.png",
            std::process::id(),
            stamp
        )));

        let (calib, report) =
            calibrate_camera_from_chessboard_files(&paths, pattern, 0.04).unwrap();
        assert_eq!(report.total_images, 7);
        assert!(report.used_images >= 3);
        assert!(!report.rejected_images.is_empty());
        assert!(calib.intrinsics.fx.is_finite() && calib.intrinsics.fy.is_finite());
        assert!(calib.intrinsics.fx.abs() > 1e-6 && calib.intrinsics.fy.abs() > 1e-6);

        for p in paths {
            let _ = std::fs::remove_file(p);
        }
    }
}
