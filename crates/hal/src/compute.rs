use crate::context::{
    BorderMode, ColorConversion, ComputeContext, MorphologyType, ThresholdType, WarpType,
};
use crate::cpu::CpuBackend;
use crate::gpu::GpuContext;
use crate::mlx::MlxContext;
use crate::Result;
use cv_core::{storage::Storage, Float, Tensor};
use std::sync::OnceLock;

static CPU_CONTEXT: OnceLock<CpuBackend> = OnceLock::new();
static MLX_CONTEXT: OnceLock<MlxContext> = OnceLock::new();

/// Dispatch a method call to whichever backend variant `ComputeDevice` holds.
macro_rules! dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ComputeDevice::Cpu(c) => c.$method($($arg),*),
            ComputeDevice::Gpu(g) => g.$method($($arg),*),
            ComputeDevice::Mlx(m) => m.$method($($arg),*),
        }
    };
}

/// A reference to the active compute backend.
///
/// `ComputeDevice` is the main entry-point for dispatching operations.
/// Obtain one via [`get_device`] (auto-select best backend) or
/// [`get_device_by_id`] (specific device).
#[derive(Clone, Copy, Debug)]
pub enum ComputeDevice<'a> {
    /// CPU backend (Rayon + SIMD).
    Cpu(&'a CpuBackend),
    /// GPU backend (wgpu / WebGPU).
    Gpu(&'a GpuContext),
    /// Apple MLX backend (experimental).
    Mlx(&'a MlxContext),
}

impl<'a> ComputeDevice<'a> {
    pub fn backend_type(&self) -> crate::BackendType {
        dispatch!(self, backend_type)
    }

    pub fn device_id(&self) -> crate::DeviceId {
        dispatch!(self, device_id)
    }

    pub fn convolve_2d<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        kernel: &Tensor<T, S>,
        border_mode: BorderMode<T>,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, convolve_2d, input, kernel, border_mode)
    }

    pub fn threshold<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        thresh: T,
        max_value: T,
        typ: ThresholdType,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, threshold, input, thresh, max_value, typ)
    }

    pub fn sobel<
        T: Float + bytemuck::Pod + std::fmt::Debug + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> Result<(Tensor<T, S>, Tensor<T, S>)> {
        dispatch!(self, sobel, input, dx, dy, ksize)
    }

    pub fn morphology<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> Result<Tensor<u8, S>> {
        dispatch!(self, morphology, input, typ, kernel, iterations)
    }

    pub fn warp<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        matrix: &[[T; 3]; 3],
        new_shape: (usize, usize),
        typ: WarpType,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, warp, input, matrix, new_shape, typ)
    }

    pub fn nms<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
        window_size: usize,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, nms, input, threshold, window_size)
    }

    pub fn nms_boxes<T: Float + 'static, S: Storage<T> + cv_core::StorageFactory<T> + 'static>(
        &self,
        input: &Tensor<T, S>,
        iou_threshold: T,
    ) -> Result<Vec<usize>> {
        dispatch!(self, nms_boxes, input, iou_threshold)
    }

    pub fn nms_rotated_boxes<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        iou_threshold: T,
    ) -> Result<Vec<usize>> {
        dispatch!(self, nms_rotated_boxes, input, iou_threshold)
    }

    pub fn nms_polygons<T: Float + 'static>(
        &self,
        polygons: &[cv_core::Polygon],
        scores: &[T],
        iou_threshold: T,
    ) -> Result<Vec<usize>> {
        dispatch!(self, nms_polygons, polygons, scores, iou_threshold)
    }

    pub fn pointcloud_transform<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        transform: &[[T; 4]; 4],
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, pointcloud_transform, points, transform)
    }

    pub fn pointcloud_normals<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        points: &Tensor<T, S>,
        k_neighbors: u32,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, pointcloud_normals, points, k_neighbors)
    }

    pub fn tsdf_integrate<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        depth_image: &Tensor<T, S>,
        camera_pose: &[[T; 4]; 4],
        intrinsics: &[T; 4],
        voxel_volume: &mut Tensor<T, S>,
        voxel_size: T,
        truncation: T,
    ) -> Result<()> {
        dispatch!(
            self,
            tsdf_integrate,
            depth_image,
            camera_pose,
            intrinsics,
            voxel_volume,
            voxel_size,
            truncation
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tsdf_raycast<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        tsdf_volume: &Tensor<T, S>,
        camera_pose: &[[T; 4]; 4],
        intrinsics: &[T; 4],
        image_size: (u32, u32),
        depth_range: (T, T),
        voxel_size: T,
        truncation: T,
    ) -> Result<Tensor<T, S>> {
        dispatch!(
            self,
            tsdf_raycast,
            tsdf_volume,
            camera_pose,
            intrinsics,
            image_size,
            depth_range,
            voxel_size,
            truncation
        )
    }

    pub fn tsdf_extract_mesh<
        T: Float + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        tsdf_volume: &Tensor<T, S>,
        voxel_size: T,
        iso_level: T,
        max_triangles: u32,
    ) -> Result<Vec<crate::gpu_kernels::marching_cubes::Vertex>> {
        dispatch!(
            self,
            tsdf_extract_mesh,
            tsdf_volume,
            voxel_size,
            iso_level,
            max_triangles
        )
    }

    pub fn optical_flow_lk<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        prev_pyramid: &[Tensor<T, S>],
        next_pyramid: &[Tensor<T, S>],
        points: &[[T; 2]],
        window_size: usize,
        max_iters: u32,
    ) -> Result<Vec<[T; 2]>> {
        dispatch!(
            self,
            optical_flow_lk,
            prev_pyramid,
            next_pyramid,
            points,
            window_size,
            max_iters
        )
    }

    pub fn cvt_color<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        code: ColorConversion,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, cvt_color, input, code)
    }

    pub fn resize<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        new_shape: (usize, usize),
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, resize, input, new_shape)
    }

    pub fn pyramid_down<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, pyramid_down, input)
    }

    pub fn bilateral_filter<
        T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        d: i32,
        sigma_color: T,
        sigma_space: T,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, bilateral_filter, input, d, sigma_color, sigma_space)
    }

    pub fn fast_detect<
        T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        threshold: T,
        non_max_suppression: bool,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, fast_detect, input, threshold, non_max_suppression)
    }

    pub fn gaussian_blur<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        sigma: T,
        k_size: usize,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, gaussian_blur, input, sigma, k_size)
    }

    pub fn subtract<
        T: Float + 'static + bytemuck::Pod,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        a: &Tensor<T, S>,
        b: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, subtract, a, b)
    }

    pub fn match_descriptors<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &self,
        query: &Tensor<u8, S>,
        train: &Tensor<u8, S>,
        ratio_threshold: f32,
    ) -> Result<cv_core::Matches> {
        dispatch!(self, match_descriptors, query, train, ratio_threshold)
    }

    pub fn sift_extrema<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        dog_prev: &Tensor<T, S>,
        dog_curr: &Tensor<T, S>,
        dog_next: &Tensor<T, S>,
        threshold: T,
        edge_threshold: T,
    ) -> Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>> {
        dispatch!(
            self,
            sift_extrema,
            dog_prev,
            dog_curr,
            dog_next,
            threshold,
            edge_threshold
        )
    }

    pub fn compute_sift_descriptors<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        image: &Tensor<T, S>,
        keypoints: &cv_core::KeyPoints,
    ) -> Result<cv_core::Descriptors> {
        dispatch!(self, compute_sift_descriptors, image, keypoints)
    }

    pub fn icp_correspondences<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        src: &Tensor<T, S>,
        tgt: &Tensor<T, S>,
        max_dist: T,
    ) -> Result<Vec<(usize, usize, T)>> {
        dispatch!(self, icp_correspondences, src, tgt, max_dist)
    }

    pub fn icp_accumulate<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        source: &Tensor<T, S>,
        target: &Tensor<T, S>,
        target_normals: &Tensor<T, S>,
        correspondences: &[(u32, u32)],
        transform: &nalgebra::Matrix4<T>,
    ) -> Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)> {
        dispatch!(
            self,
            icp_accumulate,
            source,
            target,
            target_normals,
            correspondences,
            transform
        )
    }

    pub fn dense_icp_step<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        source_depth: &Tensor<T, S>,
        target_data: &Tensor<T, S>,
        intrinsics: &[T; 4],
        initial_guess: &nalgebra::Matrix4<T>,
        max_dist: T,
        max_angle: T,
    ) -> Result<(nalgebra::Matrix6<T>, nalgebra::Vector6<T>)> {
        dispatch!(
            self,
            dense_icp_step,
            source_depth,
            target_data,
            intrinsics,
            initial_guess,
            max_dist,
            max_angle
        )
    }

    pub fn akaze_diffusion<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
        k: T,
        tau: T,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, akaze_diffusion, input, k, tau)
    }

    #[allow(clippy::type_complexity)]
    pub fn akaze_derivatives<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> Result<(Tensor<T, S>, Tensor<T, S>, Tensor<T, S>)> {
        dispatch!(self, akaze_derivatives, input)
    }

    pub fn akaze_contrast_k<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        input: &Tensor<T, S>,
    ) -> Result<T> {
        dispatch!(self, akaze_contrast_k, input)
    }

    pub fn spmv<
        T: Float + bytemuck::Pod + 'static,
        S: Storage<T> + cv_core::StorageFactory<T> + 'static,
    >(
        &self,
        row_ptr: &[u32],
        col_indices: &[u32],
        values: &[T],
        x: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        dispatch!(self, spmv, row_ptr, col_indices, values, x)
    }

    pub fn mog2_update<
        T: Float + bytemuck::Pod + 'static,
        S1: Storage<T> + 'static,
        S2: Storage<u32> + 'static,
    >(
        &self,
        frame: &Tensor<T, S1>,
        model: &mut Tensor<T, S1>,
        mask: &mut Tensor<u32, S2>,
        params: &crate::context::Mog2Params<T>,
    ) -> Result<()> {
        dispatch!(self, mog2_update, frame, model, mask, params)
    }

    /// Get a pooled GPU buffer if this is a GPU device.
    pub fn get_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> Result<wgpu::Buffer> {
        match self {
            ComputeDevice::Gpu(gpu) => Ok(gpu.get_buffer(size, usage)),
            ComputeDevice::Cpu(_) | ComputeDevice::Mlx(_) => Err(crate::Error::NotSupported(
                "GPU buffer pooling not available".into(),
            )),
        }
    }

    /// Return a buffer to the pool if this is a GPU device.
    pub fn return_buffer(&self, buffer: wgpu::Buffer, usage: wgpu::BufferUsages) -> Result<()> {
        match self {
            ComputeDevice::Gpu(gpu) => {
                gpu.return_buffer(buffer, usage);
                Ok(())
            }
            ComputeDevice::Cpu(_) | ComputeDevice::Mlx(_) => Err(crate::Error::NotSupported(
                "GPU buffer pooling not available".into(),
            )),
        }
    }
}

/// Get a compute device by its ID.
pub fn get_device_by_id(id: crate::DeviceId) -> Result<ComputeDevice<'static>> {
    // Check CPU
    if let Some(cpu) = CPU_CONTEXT.get() {
        if cpu.device_id() == id {
            return Ok(ComputeDevice::Cpu(cpu));
        }
    } else {
        // Try to init if it's the only one
        if let Some(cpu_backend) = CpuBackend::new() {
            let cpu = CPU_CONTEXT.get_or_init(move || cpu_backend);
            if cpu.device_id() == id {
                return Ok(ComputeDevice::Cpu(cpu));
            }
        }
    }

    // Check GPU
    if let Ok(gpu) = GpuContext::global() {
        if gpu.device_id() == id {
            return Ok(ComputeDevice::Gpu(gpu));
        }
    }

    // Check MLX
    if let Some(mlx) = MLX_CONTEXT.get() {
        if mlx.device_id() == id {
            return Ok(ComputeDevice::Mlx(mlx));
        }
    }

    Err(crate::Error::DeviceError(format!(
        "Device {:?} not found in global contexts",
        id
    )))
}

/// Get the best available compute device.
pub fn get_device() -> Result<ComputeDevice<'static>> {
    if let Some(mlx) = MlxContext::new() {
        return Ok(ComputeDevice::Mlx(MLX_CONTEXT.get_or_init(|| mlx)));
    }

    match GpuContext::global() {
        Ok(gpu) => Ok(ComputeDevice::Gpu(gpu)),
        Err(_) => {
            let cpu_backend = CpuBackend::new()
                .ok_or_else(|| crate::Error::InitError("CPU backend unavailable".into()))?;
            let cpu = CPU_CONTEXT.get_or_init(|| cpu_backend);
            Ok(ComputeDevice::Cpu(cpu))
        }
    }
}
