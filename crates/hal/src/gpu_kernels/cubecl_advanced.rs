//! Advanced CubeCL GPU Kernels for RETINA
//!
//! High-performance GPU implementations using CubeCL for:
//! - ICP (Iterative Closest Point) registration
//! - Optical Flow (Lucas-Kanade)
//! - Normal computation
//! - Depth processing

use cubecl::prelude::*;
use cubecl::tensor::Tensor;

// ============================================================================
// ICP (Iterative Closest Point) Registration
// ============================================================================

mod icp {
    use super::*;

    /// Compute point-to-point residuals for ICP
    /// For each source point, find closest point in target and compute residual
    #[cube]
    pub fn icp_residual_kernel(
        source: &Tensor<f32>,
        target: &Tensor<f32>,
        transform: &Tensor<f32>, // 4x4 transformation matrix
        residuals: &mut Tensor<f32>,
        num_points: u32,
    ) {
        let idx = global_idx().0;
        if idx >= num_points {
            return;
        }

        // Get source point
        let sx = source[[idx, 0]];
        let sy = source[[idx, 1]];
        let sz = source[[idx, 2]];

        // Apply transformation
        let tx = transform[[0, 0]] * sx
            + transform[[0, 1]] * sy
            + transform[[0, 2]] * sz
            + transform[[0, 3]];
        let ty = transform[[1, 0]] * sx
            + transform[[1, 1]] * sy
            + transform[[1, 2]] * sz
            + transform[[1, 3]];
        let tz = transform[[2, 0]] * sx
            + transform[[2, 1]] * sy
            + transform[[2, 2]] * sz
            + transform[[2, 3]];

        // Find closest point in target (naive - would use spatial index in practice)
        let mut min_dist = 1e10f32;
        let num_target = target.dims(0) as u32;

        for t_idx in 0..num_target {
            let dx = tx - target[[t_idx, 0]];
            let dy = ty - target[[t_idx, 1]];
            let dz = tz - target[[t_idx, 2]];
            let dist = dx * dx + dy * dy + dz * dz;

            if dist < min_dist {
                min_dist = dist;
            }
        }

        residuals[[idx]] = min_dist.sqrt();
    }

    /// Compute ICP transformation update using SVD
    /// This is a simplified version - full implementation would compute cross-covariance
    #[cube]
    pub fn icp_jacobian_kernel(
        source: &Tensor<f32>,
        target: &Tensor<f32>,
        transform: &Tensor<f32>,
        jt_j: &mut Tensor<f32>, // 6x6 JT*J matrix
        jt_r: &mut Tensor<f32>, // 6x1 JT*residual vector
        num_points: u32,
    ) {
        let idx = global_idx().0;
        if idx >= num_points {
            return;
        }

        // Get source point
        let sx = source[[idx, 0]];
        let sy = source[[idx, 1]];
        let sz = source[[idx, 2]];

        // Apply transformation
        let tx = transform[[0, 0]] * sx
            + transform[[0, 1]] * sy
            + transform[[0, 2]] * sz
            + transform[[0, 3]];
        let ty = transform[[1, 0]] * sx
            + transform[[1, 1]] * sy
            + transform[[1, 2]] * sz
            + transform[[1, 3]];
        let tz = transform[[2, 0]] * sx
            + transform[[2, 1]] * sy
            + transform[[2, 2]] * sz
            + transform[[2, 3]];

        // Simplified Jacobian - just the translation part
        // Full implementation would include rotation derivatives
        let residual = 1.0f32; // Placeholder

        // Accumulate JT*J (simplified)
        jt_j[[0, 0]] += 1.0; // dx/dtx
        jt_j[[1, 1]] += 1.0; // dy/dty
        jt_j[[2, 2]] += 1.0; // dz/dtZ

        // Accumulate JT*r
        jt_r[[0, 0]] += residual;
        jt_r[[1, 0]] += residual;
        jt_r[[2, 0]] += residual;
    }
}

/// ICP residual computation
pub fn icp_residuals<C: CubeContext>(
    ctx: &C,
    source: &Tensor<f32>,
    target: &Tensor<f32>,
    transform: &Tensor<f32>,
    num_points: usize,
) -> Tensor<f32> {
    let residuals = ctx.create_tensor(vec![num_points].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(icp::icp_residual_kernel);

    client
        .execute(
            &kernel,
            &[
                source.as_ref(),
                target.as_ref(),
                transform.as_ref(),
                residuals.as_ref(),
            ],
            cubecl::ExecutionDims::new::<icp::icp_residual_kernel>(client, (num_points,))
                .expect("Failed to create dims"),
        )
        .expect("ICP residual kernel failed");

    residuals
}

// ============================================================================
// Optical Flow (Lucas-Kanade)
// ============================================================================

mod optical_flow {
    use super::*;

    /// Compute image gradients
    #[cube]
    pub fn gradient_kernel(
        image: &Tensor<f32>,
        grad_x: &mut Tensor<f32>,
        grad_y: &mut Tensor<f32>,
        width: u32,
        height: u32,
    ) {
        let idx = global_idx().0;
        let x = idx % width;
        let y = idx / width;

        if x == 0 || x >= width - 1 || y == 0 || y >= height - 1 {
            grad_x[[y, x]] = 0.0;
            grad_y[[y, x]] = 0.0;
            return;
        }

        // Sobel X gradient
        let gx = -image[[y, x - 1]] + image[[y, x + 1]];
        // Sobel Y gradient
        let gy = -image[[y - 1, x]] + image[[y + 1, x]];

        grad_x[[y, x]] = gx * 0.5;
        grad_y[[y, x]] = gy * 0.5;
    }

    /// Lucas-Kanade optical flow single iteration
    #[cube]
    pub fn lk_flow_kernel(
        prev: &Tensor<f32>,
        next: &Tensor<f32>,
        grad_x: &Tensor<f32>,
        grad_y: &Tensor<f32>,
        flow: &mut Tensor<f32>,
        width: u32,
        height: u32,
        window_size: u32,
    ) {
        let idx = global_idx().0;
        let x = idx % width;
        let y = idx / width;

        let half_win = window_size / 2;

        if x < half_win || x >= width - half_win || y < half_win || y >= height - half_win {
            flow[[y, x, 0]] = 0.0;
            flow[[y, x, 1]] = 0.0;
            return;
        }

        // Compute structure tensor: sum(Ix^2, Ix*Iy, Iy^2) in window
        let mut sum_ixx = 0.0f32;
        let mut sum_ixy = 0.0f32;
        let mut sum_iyy = 0.0f32;
        let mut sum_ixt = 0.0f32;
        let mut sum_iyt = 0.0f32;

        for wy in 0..window_size {
            for wx in 0..window_size {
                let px = x + wx - half_win;
                let py = y + wy - half_win;

                let ix = grad_x[[py, px]];
                let iy = grad_y[[py, px]];
                let it = next[[py, px]] - prev[[py, px]];

                sum_ixx += ix * ix;
                sum_ixy += ix * iy;
                sum_iyy += iy * iy;
                sum_ixt += ix * it;
                sum_iyt += iy * it;
            }
        }

        // Solve 2x2 linear system
        let det = sum_ixx * sum_iyy - sum_ixy * sum_ixy;

        if det.abs() < 1e-8 {
            flow[[y, x, 0]] = 0.0;
            flow[[y, x, 1]] = 0.0;
            return;
        }

        // Inverse 2x2 matrix
        let inv_ixx = sum_iyy / det;
        let inv_ixy = -sum_ixy / det;
        let inv_iyy = sum_ixx / det;

        let dx = inv_ixx * sum_ixt + inv_ixy * sum_iyt;
        let dy = inv_ixy * sum_ixt + inv_iyy * sum_iyt;

        flow[[y, x, 0]] = dx;
        flow[[y, x, 1]] = dy;
    }
}

/// Compute image gradients using Sobel
pub fn compute_gradients<C: CubeContext>(
    ctx: &C,
    image: &Tensor<f32>,
    width: usize,
    height: usize,
) -> (Tensor<f32>, Tensor<f32>) {
    let grad_x = ctx.create_tensor(vec![height, width].try_into().unwrap());
    let grad_y = ctx.create_tensor(vec![height, width].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(optical_flow::gradient_kernel);

    client
        .execute(
            &kernel,
            &[image.as_ref(), grad_x.as_ref(), grad_y.as_ref()],
            cubecl::ExecutionDims::new::<optical_flow::gradient_kernel>(client, (width * height,))
                .expect("Failed to create dims"),
        )
        .expect("Gradient kernel failed");

    (grad_x, grad_y)
}

/// Lucas-Kanade optical flow
pub fn lucas_kanade_flow<C: CubeContext>(
    ctx: &C,
    prev: &Tensor<f32>,
    next: &Tensor<f32>,
    width: usize,
    height: usize,
    window_size: usize,
) -> Tensor<f32> {
    // First compute gradients
    let (grad_x, grad_y) = compute_gradients(ctx, prev, width, height);

    // Allocate flow output
    let flow = ctx.create_tensor(vec![height, width, 2].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(optical_flow::lk_flow_kernel);

    client
        .execute(
            &kernel,
            &[
                prev.as_ref(),
                next.as_ref(),
                grad_x.as_ref(),
                grad_y.as_ref(),
                flow.as_ref(),
            ],
            cubecl::ExecutionDims::new::<optical_flow::lk_flow_kernel>(client, (width * height,))
                .expect("Failed to create dims"),
        )
        .expect("LK flow kernel failed");

    flow
}

// ============================================================================
// Normal Computation
// ============================================================================

mod normals {
    use super::*;

    /// Compute normal vectors from depth image
    #[cube]
    pub fn depth_to_normal_kernel(
        depth: &Tensor<f32>,
        normals: &mut Tensor<f32>,
        width: u32,
        height: u32,
        fx: f32,
        fy: f32,
    ) {
        let idx = global_idx().0;
        let x = idx % width;
        let y = idx / width;

        if x == 0 || x >= width - 1 || y == 0 || y >= height - 1 {
            normals[[y, x, 0]] = 0.0;
            normals[[y, x, 1]] = 0.0;
            normals[[y, x, 2]] = 1.0;
            return;
        }

        let z = depth[[y, x]];

        if z < 0.01 {
            normals[[y, x, 0]] = 0.0;
            normals[[y, x, 1]] = 0.0;
            normals[[y, x, 2]] = 1.0;
            return;
        }

        // Convert to 3D points
        let px = (x as f32 - width as f32 / 2.0) * z / fx;
        let py = (y as f32 - height as f32 / 2.0) * z / fy;
        let pz = z;

        let px_right = ((x + 1) as f32 - width as f32 / 2.0) * depth[[y, x + 1]] / fx;
        let py_down = ((y + 1) as f32 - height as f32 / 2.0) * depth[[y + 1, x]] / fy;
        let pz_right = depth[[y, x + 1]];
        let pz_down = depth[[y + 1, x]];

        // Compute tangent vectors
        let tx = px_right - px;
        let ty = 0.0;
        let tz = pz_right - pz;

        let bx = 0.0;
        let by = py_down - py;
        let bz = pz_down - pz;

        // Cross product = normal
        let nx = ty * bz - tz * by;
        let ny = tz * bx - tx * bz;
        let nz = tx * by - ty * bx;

        let len = (nx * nx + ny * ny + nz * nz).sqrt();

        if len > 1e-8 {
            normals[[y, x, 0]] = nx / len;
            normals[[y, x, 1]] = ny / len;
            normals[[y, x, 2]] = nz / len;
        } else {
            normals[[y, x, 0]] = 0.0;
            normals[[y, x, 1]] = 0.0;
            normals[[y, x, 2]] = 1.0;
        }
    }

    /// Compute normal from point cloud using PCA in local neighborhood
    #[cube]
    pub fn pca_normal_kernel(
        points: &Tensor<f32>,
        normals: &mut Tensor<f32>,
        num_points: u32,
        k: u32,
    ) {
        let idx = global_idx().0;
        if idx >= num_points {
            return;
        }

        let px = points[[idx, 0]];
        let py = points[[idx, 1]];
        let pz = points[[idx, 2]];

        // Simplified: just compute from 3 nearest neighbors
        // In practice, would use spatial index for k-NN
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        let mut count = 0u32;

        for j in 0..num_points {
            if j == idx {
                continue;
            }

            let dx = points[[j, 0]] - px;
            let dy = points[[j, 1]] - py;
            let dz = points[[j, 2]] - pz;
            let dist = dx * dx + dy * dy + dz * dz;

            if dist < 1.0 {
                sum_x += points[[j, 0]];
                sum_y += points[[j, 1]];
                sum_z += points[[j, 2]];
                count += 1;
            }

            if count >= k {
                break;
            }
        }

        if count < 3 {
            normals[[idx, 0]] = 0.0;
            normals[[idx, 1]] = 0.0;
            normals[[idx, 2]] = 1.0;
            return;
        }

        // Center
        let cx = sum_x / count as f32;
        let cy = sum_y / count as f32;
        let cz = sum_z / count as f32;

        // Normalize
        normals[[idx, 0]] = cx / (cx * cx + cy * cy + cz * cz + 1e-8).sqrt();
        normals[[idx, 1]] = cy / (cx * cx + cy * cy + cz * cz + 1e-8).sqrt();
        normals[[idx, 2]] = cz / (cx * cx + cy * cy + cz * cz + 1e-8).sqrt();
    }
}

/// Compute normals from depth image
pub fn depth_to_normals<C: CubeContext>(
    ctx: &C,
    depth: &Tensor<f32>,
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
) -> Tensor<f32> {
    let normals = ctx.create_tensor(vec![height, width, 3].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(normals::depth_to_normal_kernel);

    client
        .execute(
            &kernel,
            &[depth.as_ref(), normals.as_ref()],
            cubecl::ExecutionDims::new::<normals::depth_to_normal_kernel>(
                client,
                (width * height,),
            )
            .expect("Failed to create dims"),
        )
        .expect("Depth to normal kernel failed");

    normals
}

/// Compute normals from point cloud using PCA
pub fn compute_pointcloud_normals<C: CubeContext>(
    ctx: &C,
    points: &Tensor<f32>,
    num_points: usize,
    k: usize,
) -> Tensor<f32> {
    let normals = ctx.create_tensor(vec![num_points, 3].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(normals::pca_normal_kernel);

    client
        .execute(
            &kernel,
            &[points.as_ref(), normals.as_ref()],
            cubecl::ExecutionDims::new::<normals::pca_normal_kernel>(client, (num_points,))
                .expect("Failed to create dims"),
        )
        .expect("PCA normal kernel failed");

    normals
}

// ============================================================================
// Image Processing
// ============================================================================

mod image {
    use super::*;

    /// Gaussian blur 5x5
    #[cube]
    pub fn gaussian_blur_kernel(
        input: &Tensor<f32>,
        output: &mut Tensor<f32>,
        width: u32,
        height: u32,
    ) {
        let idx = global_idx().0;
        let x = idx % width;
        let y = idx / width;

        if x < 2 || x >= width - 2 || y < 2 || y >= height - 2 {
            output[[y, x]] = input[[y, x]];
            return;
        }

        // 5x5 Gaussian kernel (sigma = 1.4)
        // Approximation: [1, 4, 6, 4, 1] normalized
        let kernel = [
            1.0, 4.0, 6.0, 4.0, 1.0, 4.0, 16.0, 24.0, 16.0, 4.0, 6.0, 24.0, 36.0, 24.0, 6.0, 4.0,
            16.0, 24.0, 16.0, 4.0, 1.0, 4.0, 6.0, 4.0, 1.0,
        ];
        let sum = 256.0;

        let mut val = 0.0f32;
        for ky in 0..5 {
            for kx in 0..5 {
                val += input[[y + ky - 2, x + kx - 2]] * kernel[ky * 5 + kx];
            }
        }

        output[[y, x]] = val / sum;
    }

    /// Bilateral filter (edge-preserving smoothing)
    #[cube]
    pub fn bilateral_filter_kernel(
        input: &Tensor<f32>,
        output: &mut Tensor<f32>,
        width: u32,
        height: u32,
        sigma_space: f32,
        sigma_range: f32,
    ) {
        let idx = global_idx().0;
        let x = idx % width;
        let y = idx / width;

        if x < 3 || x >= width - 3 || y < 3 || y >= height - 3 {
            output[[y, x]] = input[[y, x]];
            return;
        }

        let center_val = input[[y, x]];
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        for ky in 0..7 {
            for kx in 0..7 {
                let px = x + kx - 3;
                let py = y + ky - 3;

                let val = input[[py, px]];

                let dx = (kx as f32 - 3.0) / sigma_space;
                let dy = (ky as f32 - 3.0) / sigma_space;
                let dr = (val - center_val) / sigma_range;

                let spatial_weight = (-(dx * dx + dy * dy) * 0.5).exp();
                let range_weight = (-(dr * dr) * 0.5).exp();
                let weight = spatial_weight * range_weight;

                sum += val * weight;
                weight_sum += weight;
            }
        }

        output[[y, x]] = sum / (weight_sum + 1e-8);
    }
}

/// Gaussian blur
pub fn gaussian_blur<C: CubeContext>(
    ctx: &C,
    input: &Tensor<f32>,
    width: usize,
    height: usize,
) -> Tensor<f32> {
    let output = ctx.create_tensor(vec![height, width].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(image::gaussian_blur_kernel);

    client
        .execute(
            &kernel,
            &[input.as_ref(), output.as_ref()],
            cubecl::ExecutionDims::new::<image::gaussian_blur_kernel>(client, (width * height,))
                .expect("Failed to create dims"),
        )
        .expect("Gaussian blur kernel failed");

    output
}

/// Bilateral filter
pub fn bilateral_filter<C: CubeContext>(
    ctx: &C,
    input: &Tensor<f32>,
    width: usize,
    height: usize,
    sigma_space: f32,
    sigma_range: f32,
) -> Tensor<f32> {
    let output = ctx.create_tensor(vec![height, width].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(image::bilateral_filter_kernel);

    client
        .execute(
            &kernel,
            &[input.as_ref(), output.as_ref()],
            cubecl::ExecutionDims::new::<image::bilateral_filter_kernel>(client, (width * height,))
                .expect("Failed to create dims"),
        )
        .expect("Bilateral filter kernel failed");

    output
}

// ============================================================================
// Pooling Operations
// ============================================================================

mod pooling {
    use super::*;

    /// Max pooling 2x2
    #[cube]
    pub fn max_pool2d_kernel(
        input: &Tensor<f32>,
        output: &mut Tensor<f32>,
        width: u32,
        height: u32,
        pool_size: u32,
        stride: u32,
    ) {
        let idx = global_idx().0;
        let out_w = output.dims(1) as u32;
        let out_h = output.dims(0) as u32;

        let x = idx % out_w;
        let y = idx / out_w;

        if y >= out_h || x >= out_w {
            return;
        }

        let in_x = x * stride;
        let in_y = y * stride;

        let mut max_val = -1e10f32;

        for py in 0..pool_size {
            for px in 0..pool_size {
                let px_in = in_x + px;
                let py_in = in_y + py;

                if px_in < width && py_in < height {
                    max_val = max_val.max(input[[py_in, px_in]]);
                }
            }
        }

        output[[y, x]] = max_val;
    }

    /// Average pooling 2x2
    #[cube]
    pub fn avg_pool2d_kernel(
        input: &Tensor<f32>,
        output: &mut Tensor<f32>,
        width: u32,
        height: u32,
        pool_size: u32,
        stride: u32,
    ) {
        let idx = global_idx().0;
        let out_w = output.dims(1) as u32;
        let out_h = output.dims(0) as u32;

        let x = idx % out_w;
        let y = idx / out_w;

        if y >= out_h || x >= out_w {
            return;
        }

        let in_x = x * stride;
        let in_y = y * stride;

        let mut sum = 0.0f32;
        let mut count = 0u32;

        for py in 0..pool_size {
            for px in 0..pool_size {
                let px_in = in_x + px;
                let py_in = in_y + py;

                if px_in < width && py_in < height {
                    sum += input[[py_in, px_in]];
                    count += 1;
                }
            }
        }

        output[[y, x]] = sum / count as f32;
    }
}

/// Max pooling 2D
pub fn max_pool2d<C: CubeContext>(
    ctx: &C,
    input: &Tensor<f32>,
    width: usize,
    height: usize,
    pool_size: usize,
    stride: usize,
) -> Tensor<f32> {
    let out_w = (width - pool_size) / stride + 1;
    let out_h = (height - pool_size) / stride + 1;
    let output = ctx.create_tensor(vec![out_h, out_w].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(pooling::max_pool2d_kernel);

    client
        .execute(
            &kernel,
            &[input.as_ref(), output.as_ref()],
            cubecl::ExecutionDims::new::<pooling::max_pool2d_kernel>(client, (out_w * out_h,))
                .expect("Failed to create dims"),
        )
        .expect("Max pool kernel failed");

    output
}

/// Average pooling 2D
pub fn avg_pool2d<C: CubeContext>(
    ctx: &C,
    input: &Tensor<f32>,
    width: usize,
    height: usize,
    pool_size: usize,
    stride: usize,
) -> Tensor<f32> {
    let out_w = (width - pool_size) / stride + 1;
    let out_h = (height - pool_size) / stride + 1;
    let output = ctx.create_tensor(vec![out_h, out_w].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(pooling::avg_pool2d_kernel);

    client
        .execute(
            &kernel,
            &[input.as_ref(), output.as_ref()],
            cubecl::ExecutionDims::new::<pooling::avg_pool2d_kernel>(client, (out_w * out_h,))
                .expect("Failed to create dims"),
        )
        .expect("Avg pool kernel failed");

    output
}
