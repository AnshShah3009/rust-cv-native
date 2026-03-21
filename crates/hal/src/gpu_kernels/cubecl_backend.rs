//! CubeCL GPU Backend for RETINA
//!
//! This module provides GPU acceleration using CubeCL, which can target
//! CUDA, Vulkan, Metal, and other backends through a unified API.
//!
//! CubeCL provides a more Rust-native approach to GPU computing compared
//! to WGPU, with direct tensor operations and automatic differentiation support.

use cubecl::prelude::*;
use cubecl::tensor::Tensor;
use cv_core::TensorShape;

#[derive(Clone, Debug)]
pub struct CubeCLContext {
    device: WgpuDevice,
}

impl CubeCLContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let device = WgpuDevice::default();
        Ok(Self { device })
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }
}

impl Default for CubeCLContext {
    fn default() -> Self {
        Self::new().expect("Failed to create CubeCL context")
    }
}

// ============================================================================
// Element-wise Operations
// ============================================================================

mod element_wise {
    use super::*;

    #[cube]
    pub fn add_kernel(lhs: &f32, rhs: &f32) -> f32 {
        lhs + rhs
    }

    #[cube]
    pub fn mul_kernel(lhs: &f32, rhs: &f32) -> f32 {
        lhs * rhs
    }

    #[cube]
    pub fn sub_kernel(lhs: &f32, rhs: &f32) -> f32 {
        lhs - rhs
    }

    #[cube]
    pub fn div_kernel(lhs: &f32, rhs: &f32) -> f32 {
        lhs / (rhs + 1e-8)
    }

    #[cube]
    pub fn relu_kernel(x: &f32) -> f32 {
        x.max(0.0)
    }

    #[cube]
    pub fn sigmoid_kernel(x: &f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[cube]
    pub fn tanh_kernel(x: &f32) -> f32 {
        let e2x = (2.0 * x).exp();
        (e2x - 1.0) / (e2x + 1.0)
    }
}

/// Element-wise addition: output = lhs + rhs
pub fn add<C: CubeContext>(ctx: &C, lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    launch_element_wise::<f32, _>(ctx, lhs, rhs, element_wise::add_kernel)
}

/// Element-wise multiplication: output = lhs * rhs
pub fn mul<C: CubeContext>(ctx: &C, lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    launch_element_wise::<f32, _>(ctx, lhs, rhs, element_wise::mul_kernel)
}

/// Element-wise subtraction: output = lhs - rhs
pub fn sub<C: CubeContext>(ctx: &C, lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    launch_element_wise::<f32, _>(ctx, lhs, rhs, element_wise::sub_kernel)
}

/// Element-wise division: output = lhs / rhs
pub fn div<C: CubeContext>(ctx: &C, lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> Tensor<f32> {
    launch_element_wise::<f32, _>(ctx, lhs, rhs, element_wise::div_kernel)
}

/// ReLU activation: output = max(0, x)
pub fn relu<C: CubeContext>(ctx: &C, input: &Tensor<f32>) -> Tensor<f32> {
    launch_unary::<f32, _>(ctx, input, element_wise::relu_kernel)
}

/// Sigmoid activation: output = 1 / (1 + exp(-x))
pub fn sigmoid<C: CubeContext>(ctx: &C, input: &Tensor<f32>) -> Tensor<f32> {
    launch_unary::<f32, _>(ctx, input, element_wise::sigmoid_kernel)
}

/// Tanh activation
pub fn tanh<C: CubeContext>(ctx: &C, input: &Tensor<f32>) -> Tensor<f32> {
    launch_unary::<f32, _>(ctx, input, element_wise::tanh_kernel)
}

// ============================================================================
// Reduction Operations
// ============================================================================

mod reduction {
    use super::*;

    #[cube]
    pub fn sum_reduce(acc: &f32, val: &f32) -> f32 {
        acc + val
    }

    #[cube]
    pub fn max_reduce(acc: &f32, val: &f32) -> f32 {
        acc.max(*val)
    }

    #[cube]
    pub fn min_reduce(acc: &f32, val: &f32) -> f32 {
        acc.min(*val)
    }
}

/// Sum reduction along specified axis
pub fn sum<C: CubeContext>(ctx: &C, input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    input.sum(axis)
}

/// Max reduction along specified axis
pub fn max<C: CubeContext>(ctx: &C, input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    input.max(axis)
}

/// Min reduction along specified axis
pub fn min<C: CubeContext>(ctx: &C, input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    input.min(axis)
}

/// Mean reduction along specified axis
pub fn mean<C: CubeContext>(ctx: &C, input: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    input.mean(axis)
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Matrix multiplication: C = A @ B
/// Uses CubeCL's built-in matmul for optimal performance
pub fn matmul<C: CubeContext>(ctx: &C, a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    a.matmul(b)
}

// ============================================================================
// Convolution (2D)
// ============================================================================

mod conv2d {
    use super::*;

    #[cube]
    pub fn conv2d_kernel(
        input: &Tensor<f32>,
        kernel: &Tensor<f32>,
        output: &mut Tensor<f32>,
        kernel_size: u32,
        stride: u32,
        padding: u32,
    ) {
        let batch = input.dims(0);
        let channels_in = input.dims(1);
        let height_in = input.dims(2);
        let width_in = input.dims(3);

        let height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
        let width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

        let b = global_idx().0;
        if b >= batch {
            return;
        }

        let index = global_idx();
        let c_out = index / (height_out * width_out);
        let h_out = (index % (height_out * width_out)) / width_out;
        let w_out = index % width_out;

        let mut sum = 0.0f32;
        for c_in in 0..channels_in {
            for kh in 0..kernel_size {
                for kw in 0..kernel_size {
                    let h_in = h_out * stride + kh - padding;
                    let w_in = w_out * stride + kw - padding;

                    if h_in < height_in && w_in < width_in {
                        let input_val = input[[b, c_in, h_in, w_in]];
                        let kernel_val = kernel[[c_out, c_in, kh, kw]];
                        sum += input_val * kernel_val;
                    }
                }
            }
        }

        output[[b, c_out, h_out, w_out]] = sum;
    }
}

/// 2D Convolution
///
/// # Arguments
/// * `ctx` - CubeCL context
/// * `input` - Input tensor [batch, channels_in, height, width]
/// * `kernel` - Convolution kernel [channels_out, channels_in, kernel_h, kernel_w]
/// * `stride` - Convolution stride
/// * `padding` - Zero padding
pub fn conv2d<C: CubeContext>(
    ctx: &C,
    input: &Tensor<f32>,
    kernel: &Tensor<f32>,
    stride: usize,
    padding: usize,
) -> Tensor<f32> {
    let kernel_size = kernel.dims(2) as usize;
    let batch = input.dims(0);
    let channels_out = kernel.dims(0);
    let height_in = input.dims(2);
    let width_in = input.dims(3);

    let height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    let width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    let output = ctx.create_tensor(
        vec![batch, channels_out, height_out, width_out]
            .try_into()
            .unwrap(),
    );

    let client = ctx.client();
    let kernel = ctx.compile(conv2d::conv2d_kernel);

    client
        .execute(
            &kernel,
            // Args
            &[input.as_ref(), kernel.as_ref(), output.as_ref()],
            // Dimensions
            launch_params::<conv2d::conv2d_kernel>(
                ctx,
                (batch, height_out * width_out * channels_out),
            ),
        )
        .expect("Conv2D kernel failed");

    output
}

// ============================================================================
// Point Cloud Operations
// ============================================================================

mod pointcloud {
    use super::*;

    /// Compute squared Euclidean distance between all pairs of points
    #[cube]
    pub fn pairwise_distance_kernel(
        points: &Tensor<f32>,
        output: &mut Tensor<f32>,
        num_points: u32,
    ) {
        let i = global_idx().0;
        if i >= num_points * num_points {
            return;
        }

        let p1_idx = i / num_points;
        let p2_idx = i % num_points;

        let px1 = points[[p1_idx, 0]];
        let py1 = points[[p1_idx, 1]];
        let pz1 = points[[p1_idx, 2]];

        let px2 = points[[p2_idx, 0]];
        let py2 = points[[p2_idx, 1]];
        let pz2 = points[[p2_idx, 2]];

        let dx = px1 - px2;
        let dy = py1 - py2;
        let dz = pz1 - pz2;

        output[[p1_idx, p2_idx]] = dx * dx + dy * dy + dz * dz;
    }

    /// K-Nearest Neighbors search
    #[cube]
    pub fn knn_kernel(
        points: &Tensor<f32>,
        queries: &Tensor<f32>,
        distances: &mut Tensor<f32>,
        indices: &mut Tensor<u32>,
        k: u32,
        num_points: u32,
        num_queries: u32,
    ) {
        let q_idx = global_idx().0;
        if q_idx >= num_queries {
            return;
        }

        let qx = queries[[q_idx, 0]];
        let qy = queries[[q_idx, 1]];
        let qz = queries[[q_idx, 2]];

        // Compute distances to all points
        let mut dists: [f32; 32] = [0.0; 32]; // Max k=32
        let mut indices_arr: [u32; 32] = [0; 32];

        for p_idx in 0..num_points {
            let px = points[[p_idx, 0]];
            let py = points[[p_idx, 1]];
            let pz = points[[p_idx, 2]];

            let dx = qx - px;
            let dy = qy - py;
            let dz = qz - pz;
            let dist = dx * dx + dy * dy + dz * dz;

            // Simple insertion sort for top-k
            for k_idx in 0..k {
                if dist < dists[k_idx as usize] {
                    // Shift
                    for shift in (k_idx + 1..k).rev() {
                        dists[shift as usize] = dists[(shift - 1) as usize];
                        indices_arr[shift as usize] = indices_arr[(shift - 1) as usize];
                    }
                    dists[k_idx as usize] = dist;
                    indices_arr[k_idx as usize] = p_idx;
                    break;
                }
            }
        }

        for k_idx in 0..k {
            distances[[q_idx, k_idx]] = dists[k_idx as usize];
            indices[[q_idx, k_idx]] = indices_arr[k_idx as usize];
        }
    }

    /// Voxel grid downsampling - assign each point to a voxel and reduce
    #[cube]
    pub fn voxel_hash_kernel(
        points: &Tensor<f32>,
        voxel_keys: &mut Tensor<u32>,
        voxel_counts: &mut Tensor<u32>,
        num_points: u32,
        voxel_size: f32,
    ) {
        let p_idx = global_idx().0;
        if p_idx >= num_points {
            return;
        }

        let px = points[[p_idx, 0]];
        let py = points[[p_idx, 1]];
        let pz = points[[p_idx, 2]];

        let vx = (px / voxel_size) as i32;
        let vy = (py / voxel_size) as i32;
        let vz = (pz / voxel_size) as i32;

        // Simple hash function
        let hash = ((vx * 73856093) ^ (vy * 19349663) ^ (vz * 83492791)).abs() as u32;

        voxel_keys[[p_idx, 0]] = hash;
        voxel_keys[[p_idx, 1]] = p_idx; // Store original index
    }
}

/// Compute pairwise squared distances between all points
pub fn pairwise_squared_distance<C: CubeContext>(
    ctx: &C,
    points: &Tensor<f32>,
    num_points: usize,
) -> Tensor<f32> {
    let output = ctx.create_tensor(vec![num_points, num_points].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(pointcloud::pairwise_distance_kernel);

    client
        .execute(
            &kernel,
            &[points.as_ref(), output.as_ref()],
            launch_params::<pointcloud::pairwise_distance_kernel>(ctx, (num_points * num_points,)),
        )
        .expect("Pairwise distance kernel failed");

    output
}

/// K-Nearest Neighbors search
pub fn knn<C: CubeContext>(
    ctx: &C,
    points: &Tensor<f32>,
    queries: &Tensor<f32>,
    k: usize,
) -> (Tensor<f32>, Tensor<u32>) {
    let num_points = points.dims(0) as usize;
    let num_queries = queries.dims(0) as usize;

    let distances = ctx.create_tensor(vec![num_queries, k].try_into().unwrap());
    let indices = ctx.create_tensor(vec![num_queries, k].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(pointcloud::knn_kernel);

    client
        .execute(
            &kernel,
            &[
                points.as_ref(),
                queries.as_ref(),
                distances.as_ref(),
                indices.as_ref(),
            ],
            launch_params::<pointcloud::knn_kernel>(ctx, (num_queries,)),
        )
        .expect("KNN kernel failed");

    (distances, indices)
}

/// Voxel grid hashing for downsampling
pub fn voxel_hash<C: CubeContext>(ctx: &C, points: &Tensor<f32>, voxel_size: f32) -> Tensor<u32> {
    let num_points = points.dims(0) as usize;
    let output = ctx.create_tensor(vec![num_points, 2].try_into().unwrap());

    let client = ctx.client();
    let kernel = ctx.compile(pointcloud::voxel_hash_kernel);

    client
        .execute(
            &kernel,
            &[points.as_ref(), output.as_ref()],
            launch_params::<pointcloud::voxel_hash_kernel>(ctx, (num_points,)),
        )
        .expect("Voxel hash kernel failed");

    output
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Helper to launch binary element-wise kernels
fn launch_element_wise<F: Float, K: cubecl::Kernel<F>>(
    ctx: &impl CubeContext,
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    _kernel: K,
) -> Tensor<F> {
    let output = ctx.create_tensor(lhs.dims());

    let client = ctx.client();
    let kernel = ctx.compile(_kernel);

    let total_elements = lhs.dims().iter().product::<usize>();

    client
        .execute(
            &kernel,
            &[lhs.as_ref(), rhs.as_ref(), output.as_ref()],
            launch_params::<K>(ctx, (total_elements,)),
        )
        .expect("Element-wise kernel failed");

    output
}

/// Helper to launch unary element-wise kernels
fn launch_unary<F: Float, K: cubecl::Kernel<F>>(
    ctx: &impl CubeContext,
    input: &Tensor<F>,
    _kernel: K,
) -> Tensor<F> {
    let output = ctx.create_tensor(input.dims());

    let client = ctx.client();
    let kernel = ctx.compile(_kernel);

    let total_elements = input.dims().iter().product::<usize>();

    client
        .execute(
            &kernel,
            &[input.as_ref(), output.as_ref()],
            launch_params::<K>(ctx, (total_elements,)),
        )
        .expect("Unary kernel failed");

    output
}

/// Helper to get launch parameters for a kernel
fn launch_params<K: cubecl::Kernel<f32>>(
    ctx: &impl CubeContext,
    dims: impl Into<cubecl::TensorDim>,
) -> cubecl::ExecutionDims {
    let client = ctx.client();
    let numel = dims.clone().into_num_elements();
    cubecl::ExecutionDims::new::<K>(client, dims).expect("Failed to create execution dims")
}
