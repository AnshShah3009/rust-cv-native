//! RGBD Odometry
//!
//! Estimates camera motion from consecutive RGBD frames.
//! Implements multi-scale odometry with different loss functions.

use nalgebra::{Matrix3, Matrix4, Point3, Vector3};

/// RGBD Odometry result
#[derive(Debug, Clone)]
pub struct OdometryResult {
    pub transformation: Matrix4<f32>,
    pub fitness: f32,
    pub inlier_rmse: f32,
}

/// Odometry method
#[derive(Debug, Clone, Copy)]
pub enum OdometryMethod {
    PointToPlane,
    Intensity,
    Hybrid,
}

/// Compute RGBD odometry between two frames
pub fn compute_rgbd_odometry(
    source_depth: &[f32],
    target_depth: &[f32],
    source_color: Option<&[Vector3<u8>]>,
    target_color: Option<&[Vector3<u8>]>,
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    method: OdometryMethod,
) -> Option<OdometryResult> {
    // Multi-scale pyramid
    let scales = vec![1.0, 0.5, 0.25, 0.125];
    let mut transformation = Matrix4::identity();

    for scale in scales {
        let scaled_width = (width as f32 * scale) as usize;
        let scaled_height = (height as f32 * scale) as usize;

        // Create scaled intrinsics
        let scaled_intrinsics = crate::tsdf::CameraIntrinsics {
            fx: intrinsics.fx * scale,
            fy: intrinsics.fy * scale,
            cx: intrinsics.cx * scale,
            cy: intrinsics.cy * scale,
        };

        // Downsample depth images
        let source_scaled = downsample_depth(source_depth, width, height, scale);
        let target_scaled = downsample_depth(target_depth, width, height, scale);

        // Compute odometry at this scale
        let scale_result = match method {
            OdometryMethod::PointToPlane => compute_point_to_plane(
                &source_scaled,
                &target_scaled,
                &scaled_intrinsics,
                scaled_width,
                scaled_height,
                &transformation,
            ),
            OdometryMethod::Intensity => compute_intensity(
                &source_scaled,
                &target_scaled,
                source_color,
                target_color,
                &scaled_intrinsics,
                scaled_width,
                scaled_height,
                &transformation,
            ),
            OdometryMethod::Hybrid => compute_hybrid(
                &source_scaled,
                &target_scaled,
                source_color,
                target_color,
                &scaled_intrinsics,
                scaled_width,
                scaled_height,
                &transformation,
            ),
        };

        if let Some(result) = scale_result {
            transformation = result.transformation * transformation;
        }
    }

    // Compute final fitness
    let (fitness, rmse) = evaluate_odometry(
        source_depth,
        target_depth,
        intrinsics,
        width,
        height,
        &transformation,
    );

    Some(OdometryResult {
        transformation,
        fitness,
        inlier_rmse: rmse,
    })
}

/// Point-to-plane odometry
fn compute_point_to_plane(
    source_depth: &[f32],
    target_depth: &[f32],
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    init_transform: &Matrix4<f32>,
) -> Option<OdometryResult> {
    let mut transformation = *init_transform;
    let max_iterations = 10;

    // Precompute target vertex and normal maps
    let (target_vertices, target_normals) =
        compute_vertex_normal_map(target_depth, intrinsics, width, height);

    for _ in 0..max_iterations {
        // Build linear system
        let mut ata = Matrix3::<f32>::zeros();
        let mut atb = Vector3::<f32>::zeros();
        let mut total_residual = 0.0;
        let mut valid_points = 0;

        for v in 0..height {
            for u in 0..width {
                let idx = v * width + u;
                let depth = source_depth[idx];

                if depth <= 0.0 {
                    continue;
                }

                // Backproject source point
                let x = (u as f32 - intrinsics.cx) * depth / intrinsics.fx;
                let y = (v as f32 - intrinsics.cy) * depth / intrinsics.fy;
                let z = depth;
                let source_point = Point3::new(x, y, z);

                // Transform to target frame
                let target_point = transformation.transform_point(&source_point);

                // Project to target image
                let tu = (target_point.x * intrinsics.fx / target_point.z + intrinsics.cx) as i32;
                let tv = (target_point.y * intrinsics.fy / target_point.z + intrinsics.cy) as i32;

                if tu < 0 || tu >= width as i32 || tv < 0 || tv >= height as i32 {
                    continue;
                }

                let tidx = (tv as usize) * width + (tu as usize);
                let target_vertex = target_vertices[tidx];
                let target_normal = target_normals[tidx];

                if target_vertex.z <= 0.0 {
                    continue;
                }

                // Point-to-plane error
                let residual = (target_point - target_vertex).dot(&target_normal);

                // Jacobian (simplified for translation only)
                // Full implementation would include rotation
                let jacobian = target_normal;

                ata += jacobian * jacobian.transpose();
                atb += jacobian * residual;
                total_residual += residual * residual;
                valid_points += 1;
            }
        }

        if valid_points < 100 {
            return None;
        }

        // Solve for update
        if let Some(ata_inv) = ata.try_inverse() {
            let delta = ata_inv * atb;

            // Update transformation (translation only for simplicity)
            let update = Matrix4::new_translation(&delta);
            transformation = update * transformation;
        }
    }

    Some(OdometryResult {
        transformation,
        fitness: 0.0,
        inlier_rmse: 0.0,
    })
}

/// Intensity-based odometry (uses color)
fn compute_intensity(
    _source_depth: &[f32],
    _target_depth: &[f32],
    _source_color: Option<&[Vector3<u8>]>,
    _target_color: Option<&[Vector3<u8>]>,
    _intrinsics: &crate::tsdf::CameraIntrinsics,
    _width: usize,
    _height: usize,
    _init_transform: &Matrix4<f32>,
) -> Option<OdometryResult> {
    // Placeholder - full implementation would compute photometric error
    None
}

/// Hybrid odometry (combines depth and intensity)
fn compute_hybrid(
    _source_depth: &[f32],
    _target_depth: &[f32],
    _source_color: Option<&[Vector3<u8>]>,
    _target_color: Option<&[Vector3<u8>]>,
    _intrinsics: &crate::tsdf::CameraIntrinsics,
    _width: usize,
    _height: usize,
    init_transform: &Matrix4<f32>,
) -> Option<OdometryResult> {
    // Placeholder - would combine point-to-plane and intensity
    Some(OdometryResult {
        transformation: *init_transform,
        fitness: 0.0,
        inlier_rmse: 0.0,
    })
}

/// Downsample depth image
fn downsample_depth(input: &[f32], width: usize, height: usize, scale: f32) -> Vec<f32> {
    let new_width = (width as f32 * scale) as usize;
    let new_height = (height as f32 * scale) as usize;
    let mut output = vec![0.0; new_width * new_height];

    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = (x as f32 / scale) as usize;
            let src_y = (y as f32 / scale) as usize;
            let src_idx = (src_y.min(height - 1)) * width + (src_x.min(width - 1));
            output[y * new_width + x] = input[src_idx];
        }
    }

    output
}

/// Compute vertex and normal maps from depth
fn compute_vertex_normal_map(
    depth: &[f32],
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
) -> (Vec<Point3<f32>>, Vec<Vector3<f32>>) {
    let mut vertices = vec![Point3::origin(); width * height];
    let mut normals = vec![Vector3::zeros(); width * height];

    // Compute vertices
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let z = depth[idx];

            if z > 0.0 {
                let vx = (x as f32 - intrinsics.cx) * z / intrinsics.fx;
                let vy = (y as f32 - intrinsics.cy) * z / intrinsics.fy;
                vertices[idx] = Point3::new(vx, vy, z);
            }
        }
    }

    // Compute normals using central differences
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;

            let left = vertices[idx - 1];
            let right = vertices[idx + 1];
            let up = vertices[(y - 1) * width + x];
            let down = vertices[(y + 1) * width + x];

            if left.z > 0.0 && right.z > 0.0 && up.z > 0.0 && down.z > 0.0 {
                let dx = right.coords - left.coords;
                let dy = down.coords - up.coords;
                normals[idx] = dx.cross(&dy).normalize();
            }
        }
    }

    (vertices, normals)
}

/// Evaluate odometry quality
fn evaluate_odometry(
    source_depth: &[f32],
    target_depth: &[f32],
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    transformation: &Matrix4<f32>,
) -> (f32, f32) {
    let mut total_error = 0.0;
    let mut valid_points = 0;

    for v in 0..height {
        for u in 0..width {
            let idx = v * width + u;
            let depth = source_depth[idx];

            if depth <= 0.0 {
                continue;
            }

            let x = (u as f32 - intrinsics.cx) * depth / intrinsics.fx;
            let y = (v as f32 - intrinsics.cy) * depth / intrinsics.fy;
            let z = depth;
            let point = Point3::new(x, y, z);

            let transformed = transformation.transform_point(&point);
            let tu = (transformed.x * intrinsics.fx / transformed.z + intrinsics.cx) as i32;
            let tv = (transformed.y * intrinsics.fy / transformed.z + intrinsics.cy) as i32;

            if tu >= 0 && tu < width as i32 && tv >= 0 && tv < height as i32 {
                let tidx = (tv as usize) * width + (tu as usize);
                let target_z = target_depth[tidx];

                if target_z > 0.0 {
                    let error = (transformed.z - target_z).abs();
                    total_error += error * error;
                    valid_points += 1;
                }
            }
        }
    }

    if valid_points == 0 {
        return (0.0, 0.0);
    }

    let rmse = (total_error / valid_points as f32).sqrt();
    let fitness = valid_points as f32 / (width * height) as f32;

    (fitness, rmse)
}
