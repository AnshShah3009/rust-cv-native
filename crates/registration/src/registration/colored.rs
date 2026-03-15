//! Colored ICP Registration
//!
//! ICP variant that uses both geometry and color for alignment.
//! Useful for registering RGBD scans with rich texture.

use cv_core::point_cloud::PointCloud;
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};

/// KDTree-backed nearest neighbor for O(log N) queries.
struct SimpleNN {
    tree: cv_3d::spatial::KDTree<usize>,
}

impl SimpleNN {
    fn new(points: Vec<Point3<f32>>) -> Self {
        let mut items: Vec<_> = points.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        let tree = cv_3d::spatial::KDTree::build(&mut items);
        Self { tree }
    }

    fn nearest(&self, query: &Point3<f32>) -> Option<(Point3<f32>, usize, f32)> {
        self.tree.nearest_neighbor(query)
    }
}

/// Colored ICP result
#[derive(Debug, Clone)]
pub struct ColoredICPResult {
    pub transformation: Matrix4<f32>,
    pub fitness: f32,
    pub inlier_rmse: f32,
}

/// Colored ICP registration
#[allow(clippy::needless_range_loop)]
pub fn registration_colored_icp(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f32,
    init_transformation: &Matrix4<f32>,
    max_iterations: usize,
    lambda_geometric: f32, // Weight for geometric vs photometric (0-1)
) -> Option<ColoredICPResult> {
    // Check if colors are available
    let source_colors = source.colors.as_ref()?;
    let target_colors = target.colors.as_ref()?;

    // Build simple NN for target
    let target_nn = SimpleNN::new(target.points.clone());

    let mut transformation = *init_transformation;
    let mut best_fitness = 0.0;
    let mut best_rmse = f32::MAX;

    for iter in 0..max_iterations {
        // Build linear system
        let mut ata = nalgebra::Matrix6::<f32>::zeros();
        let mut atb = nalgebra::Vector6::<f32>::zeros();
        let mut total_residual = 0.0;
        let mut valid_points = 0;

        for i in 0..source.points.len() {
            let source_point = source.points[i];
            let source_color = source_colors[i];

            // Transform to target frame
            let transformed = transformation.transform_point(&source_point);

            // Find nearest neighbor in target
            if let Some((target_point, target_idx, dist_sq)) = target_nn.nearest(&transformed) {
                let dist = dist_sq.sqrt();

                if dist > max_correspondence_distance {
                    continue;
                }

                let target_color = target_colors[target_idx];

                // Compute residuals
                let geometric_residual = dist;

                // Photometric residual (color difference in grayscale)
                let source_gray =
                    0.299 * source_color.x + 0.587 * source_color.y + 0.114 * source_color.z;
                let target_gray =
                    0.299 * target_color.x + 0.587 * target_color.y + 0.114 * target_color.z;
                let photometric_residual = (source_gray - target_gray).abs();

                // Combined residual
                let residual = lambda_geometric * geometric_residual
                    + (1.0 - lambda_geometric) * photometric_residual * 0.1;

                // Compute jacobian (simplified)
                // Full implementation would compute SE(3) jacobian
                let normal = target
                    .normals
                    .as_ref()
                    .map(|n| n[target_idx])
                    .unwrap_or_else(|| {
                        let diff = transformed - target_point;
                        diff.normalize()
                    });

                // Geometric jacobian
                let jacobian_geo = compute_point_to_plane_jacobian(&source_point, &normal);

                // Photometric jacobian (gradient of intensity w.r.t. pose)
                let jacobian_photo = compute_photometric_jacobian(
                    &transformed,
                    &source_color,
                    &target_color,
                    &target_point,
                    &normal,
                    target,
                    target_colors,
                    target_idx,
                );

                // Combined jacobian
                let jacobian =
                    jacobian_geo * lambda_geometric + jacobian_photo * (1.0 - lambda_geometric);

                // Accumulate (jacobian * jacobian.transpose() gives 6x6)
                ata += jacobian * jacobian.transpose();
                atb += jacobian * residual;
                total_residual += residual * residual;
                valid_points += 1;
            }
        }

        if valid_points < 10 {
            break;
        }

        // Solve for update
        if let Some(ata_inv) = ata.try_inverse() {
            let delta = -(ata_inv * atb);

            // Convert delta to transformation update
            let update = exponential_map(&delta);
            transformation = update * transformation;
        }

        // Track best
        let rmse = (total_residual / valid_points as f32).sqrt();
        let fitness = valid_points as f32 / source.points.len() as f32;

        if fitness > best_fitness {
            best_fitness = fitness;
            best_rmse = rmse;
        }

        // Convergence check
        if iter > 0 && rmse < 0.001 {
            break;
        }
    }

    Some(ColoredICPResult {
        transformation,
        fitness: best_fitness,
        inlier_rmse: best_rmse,
    })
}

/// Compute point-to-plane jacobian
fn compute_point_to_plane_jacobian(
    point: &Point3<f32>,
    normal: &Vector3<f32>,
) -> nalgebra::Vector6<f32> {
    // J = [n^T, (p x n)^T]
    let p = point.coords;
    let n = normal;
    let cross = p.cross(n);

    nalgebra::Vector6::new(n.x, n.y, n.z, cross.x, cross.y, cross.z)
}

/// Compute photometric jacobian per Park et al. (ICCV 2017).
///
/// The photometric error is e_C = I(target) - I(source), where I is grayscale
/// intensity.  The full Jacobian is:
///     J_C = (dI/dp)^T · (dp/dξ)
/// where dI/dp is the intensity gradient projected onto the tangent plane at
/// the target point and dp/dξ is the SE(3) Jacobian for the point.
///
/// The intensity gradient at the target point is estimated via finite
/// differences over the k-nearest neighbours (in this simplified version the
/// single nearest neighbour is already provided by the caller).
#[allow(clippy::too_many_arguments)]
fn compute_photometric_jacobian(
    transformed_point: &Point3<f32>,
    source_color: &Point3<f32>,
    target_color: &Point3<f32>,
    target_point: &Point3<f32>,
    normal: &Vector3<f32>,
    target_cloud: &PointCloud,
    target_colors: &[Point3<f32>],
    target_idx: usize,
) -> nalgebra::Vector6<f32> {
    // 1. Estimate the colour gradient at the target point.
    //    We approximate dI/dp by finite differences over the neighbours of the
    //    target point.  For every other target point within a small radius we
    //    accumulate a least-squares gradient in 3-D.  When no nearby points
    //    are available, fall back to the colour difference direction.
    let target_gray = 0.299 * target_color.x + 0.587 * target_color.y + 0.114 * target_color.z;

    // Collect neighbour intensity differences and position offsets.
    let search_radius_sq: f32 = {
        // Use a radius proportional to the distance to the nearest neighbour.
        let d = (transformed_point - target_point).norm();
        let r = (d * 3.0).max(0.01);
        r * r
    };

    // Accumulate A^T A and A^T b for the 3-D gradient  (dI/dx, dI/dy, dI/dz).
    let mut ata = Matrix3::<f32>::zeros();
    let mut atb = Vector3::<f32>::zeros();
    let mut neighbour_count = 0u32;

    // Sample neighbours (cap iteration to avoid O(n^2) for very large clouds).
    let max_samples = target_cloud.points.len().min(200);
    let step = if target_cloud.points.len() > max_samples {
        target_cloud.points.len() / max_samples
    } else {
        1
    };

    let mut j = 0;
    while j < target_cloud.points.len() {
        if j != target_idx {
            let dp = target_cloud.points[j] - target_point;
            let dist_sq = dp.norm_squared();
            if dist_sq < search_radius_sq && dist_sq > 1e-12 {
                let neighbor_color = &target_colors[j];
                let neighbor_gray =
                    0.299 * neighbor_color.x + 0.587 * neighbor_color.y + 0.114 * neighbor_color.z;
                let di = neighbor_gray - target_gray;

                // Rank-1 update of A^T A and A^T b
                ata += dp * dp.transpose();
                atb += dp * di;
                neighbour_count += 1;
            }
        }
        j += step;
    }

    let intensity_gradient_3d = if neighbour_count >= 3 {
        // Solve the 3x3 system for the gradient
        ata.try_inverse().map(|inv| inv * atb).unwrap_or_else(|| {
            // Fallback: finite-difference along the colour difference direction
            color_difference_gradient(source_color, target_color, transformed_point, target_point)
        })
    } else {
        // Not enough neighbours — approximate from the colour difference
        color_difference_gradient(source_color, target_color, transformed_point, target_point)
    };

    // 2. Project the intensity gradient onto the tangent plane at the target.
    let n = normal.normalize();
    let grad_tangent = intensity_gradient_3d - n * n.dot(&intensity_gradient_3d);

    // 3. Build the SE(3) photometric Jacobian.
    //    For a point p, the Jacobian dp/dξ (the derivative of the transformed
    //    point w.r.t. the twist ξ = [v; ω]) is:
    //        dp/dξ = [ I | -[p]_x ]     (3×6)
    //    The photometric Jacobian is:
    //        J_C = grad_tangent^T · dp/dξ   (1×6 → stored as a 6-vector)
    let p = transformed_point.coords;
    let px = Vector3::new(0.0, -p.z, p.y);
    let py = Vector3::new(p.z, 0.0, -p.x);
    let pz = Vector3::new(-p.y, p.x, 0.0);

    // J_C = [ grad^T , grad^T · (-[p]_x) ]
    let g = &grad_tangent;
    nalgebra::Vector6::new(
        g.x,
        g.y,
        g.z,
        -(g.x * px.x + g.y * px.y + g.z * px.z),
        -(g.x * py.x + g.y * py.y + g.z * py.z),
        -(g.x * pz.x + g.y * pz.y + g.z * pz.z),
    )
}

/// Fallback intensity gradient approximation from the colour difference
/// projected along the direction between the two matched points.
fn color_difference_gradient(
    source_color: &Point3<f32>,
    target_color: &Point3<f32>,
    transformed_point: &Point3<f32>,
    target_point: &Point3<f32>,
) -> Vector3<f32> {
    let source_gray = 0.299 * source_color.x + 0.587 * source_color.y + 0.114 * source_color.z;
    let target_gray = 0.299 * target_color.x + 0.587 * target_color.y + 0.114 * target_color.z;
    let di = target_gray - source_gray;

    let direction = transformed_point - target_point;
    let dist = direction.norm();
    if dist > 1e-8 {
        direction * (di / (dist * dist))
    } else {
        Vector3::new(di, di, di) * 0.01
    }
}

/// Exponential map from se(3) to SE(3)
fn exponential_map(delta: &nalgebra::Vector6<f32>) -> Matrix4<f32> {
    // Extract rotation and translation components
    let omega = Vector3::new(delta[3], delta[4], delta[5]);
    let v = Vector3::new(delta[0], delta[1], delta[2]);

    // Rodrigues' formula for rotation
    let theta = omega.norm();
    let rotation = if theta < 1e-6 {
        Matrix3::identity()
    } else {
        let k = omega / theta;
        let k_cross = Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        Matrix3::identity() + k_cross * theta.sin() + k_cross * k_cross * (1.0 - theta.cos())
    };

    // Proper SE(3) exponential map using left Jacobian
    let translation = if theta < 1e-6 {
        v
    } else {
        let k = omega / theta;
        let k_cross_v = Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        let k_cross_sq_v = k_cross_v * k_cross_v;
        let left_jacobian = Matrix3::identity()
            + k_cross_v * ((1.0 - theta.cos()) / theta)
            + k_cross_sq_v * ((theta - theta.sin()) / theta);
        left_jacobian * v
    };

    // Build transformation
    let mut transform = Matrix4::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);

    transform
}
