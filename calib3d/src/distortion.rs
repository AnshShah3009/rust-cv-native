//! Distortion correction functions for camera calibration
//!
//! This module provides functions to correct lens distortion in images and points.

use crate::{CalibError, Result};
use cv_core::{CameraIntrinsics, Distortion};
use cv_imgproc::{remap, BorderMode, Interpolation};
use image::GrayImage;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

/// Undistort a set of 2D points using camera intrinsics and distortion model.
///
/// Takes distorted image points and returns their undistorted coordinates in the same space.
///
/// # Arguments
///
/// * `distorted_points` - The distorted 2D image points
/// * `intrinsics` - Camera intrinsic matrix and parameters
/// * `distortion` - The distortion model (k1, k2, p1, p2, k3)
///
/// # Returns
///
/// A vector of undistorted 2D points, or an error if intrinsics are invalid
pub fn undistort_points(
    distorted_points: &[nalgebra::Point2<f64>],
    intrinsics: &CameraIntrinsics,
    distortion: &Distortion,
) -> Result<Vec<nalgebra::Point2<f64>>> {
    if intrinsics.fx.abs() <= 1e-12 || intrinsics.fy.abs() <= 1e-12 {
        return Err(CalibError::InvalidParameters(
            "undistort_points requires non-zero focal lengths".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(distorted_points.len());
    for p in distorted_points {
        let xd = (p.x - intrinsics.cx) / intrinsics.fx;
        let yd = (p.y - intrinsics.cy) / intrinsics.fy;
        let (xu, yu) = distortion.remove(xd, yd);
        out.push(nalgebra::Point2::new(
            intrinsics.fx * xu + intrinsics.cx,
            intrinsics.fy * yu + intrinsics.cy,
        ));
    }
    Ok(out)
}

/// Create remap matrices for undistortion with optional rectification.
///
/// Precomputes lookup maps used for efficient image undistortion via remapping.
/// Optionally applies rectification to correct for relative camera positions in stereo systems.
///
/// # Arguments
///
/// * `image_size` - (width, height) of the output image
/// * `intrinsics` - Original camera intrinsic matrix
/// * `distortion` - The distortion model to correct
/// * `rectification` - Rectification rotation matrix (use `Matrix3::identity()` for no rectification)
/// * `new_intrinsics` - Intrinsics for the output image (can differ from original)
///
/// # Returns
///
/// A tuple of (map_x, map_y) float vectors for use with remap functions,
/// or an error if parameters are invalid
pub fn init_undistort_rectify_map(
    image_size: (u32, u32),
    intrinsics: &CameraIntrinsics,
    distortion: &Distortion,
    rectification: &Matrix3<f64>,
    new_intrinsics: &CameraIntrinsics,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if image_size.0 == 0 || image_size.1 == 0 {
        return Err(CalibError::InvalidParameters(
            "init_undistort_rectify_map requires non-zero image size".to_string(),
        ));
    }
    if intrinsics.fx.abs() <= 1e-12
        || intrinsics.fy.abs() <= 1e-12
        || new_intrinsics.fx.abs() <= 1e-12
        || new_intrinsics.fy.abs() <= 1e-12
    {
        return Err(CalibError::InvalidParameters(
            "init_undistort_rectify_map requires non-zero focal lengths".to_string(),
        ));
    }
    let (width, height) = image_size;
    let mut map_x = vec![0.0f32; (width * height) as usize];
    let mut map_y = vec![0.0f32; (width * height) as usize];

    let k_new_inv = new_intrinsics
        .matrix()
        .try_inverse()
        .unwrap_or(Matrix3::identity());
    let r_inv = rectification.try_inverse().unwrap_or(Matrix3::identity());

    map_x
        .par_chunks_mut(width as usize)
        .zip(map_y.par_chunks_mut(width as usize))
        .enumerate()
        .for_each(|(y, (row_x, row_y))| {
            for x in 0..width {
                let dst = Vector3::new(x as f64, y as f64, 1.0);
                let rectified_norm = &k_new_inv * dst;
                let original_norm = &r_inv * rectified_norm;

                if original_norm[2].abs() <= 1e-12 {
                    continue;
                }
                let xn = original_norm[0] / original_norm[2];
                let yn = original_norm[1] / original_norm[2];
                let (xd, yd) = distortion.apply(xn, yn);
                let src_x = intrinsics.fx * xd + intrinsics.cx;
                let src_y = intrinsics.fy * yd + intrinsics.cy;

                row_x[x as usize] = src_x as f32;
                row_y[x as usize] = src_y as f32;
            }
        });

    Ok((map_x, map_y))
}

/// Undistort a grayscale image using camera intrinsics and distortion model.
///
/// Applies lens distortion correction to the entire image using remapping.
/// Useful for preprocessing images before analysis or feature detection.
///
/// # Arguments
///
/// * `src` - Source grayscale image
/// * `intrinsics` - Original camera intrinsic matrix
/// * `distortion` - The distortion model to correct
/// * `new_intrinsics` - Optional new intrinsics for the output image.
///   If `None`, uses the original intrinsics.
///
/// # Returns
///
/// Undistorted grayscale image, or an error if parameters are invalid
pub fn undistort_image(
    src: &GrayImage,
    intrinsics: &CameraIntrinsics,
    distortion: &Distortion,
    new_intrinsics: Option<&CameraIntrinsics>,
) -> Result<GrayImage> {
    let k_new = new_intrinsics.unwrap_or(intrinsics);
    let (map_x, map_y) = init_undistort_rectify_map(
        (src.width(), src.height()),
        intrinsics,
        distortion,
        &Matrix3::identity(),
        k_new,
    )?;
    Ok(remap(
        src,
        &map_x,
        &map_y,
        src.width(),
        src.height(),
        Interpolation::Linear,
        BorderMode::Constant(0),
    ))
}
