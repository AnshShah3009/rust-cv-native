use crate::gpu::GpuContext;
use crate::cpu::CpuBackend;
use crate::context::{ComputeContext, BorderMode, ThresholdType, MorphologyType, WarpType, ColorConversion};
use crate::Result;
use cv_core::{Tensor, storage::Storage};
use std::sync::OnceLock;

static CPU_CONTEXT: OnceLock<CpuBackend> = OnceLock::new();

#[derive(Clone, Copy, Debug)]
pub enum ComputeDevice<'a> {
    Cpu(&'a CpuBackend),
    Gpu(&'a GpuContext),
}

impl<'a> ComputeDevice<'a> {
    pub fn backend_type(&self) -> crate::BackendType {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.backend_type(),
            ComputeDevice::Gpu(gpu) => gpu.backend_type(),
        }
    }

    pub fn convolve_2d<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        kernel: &Tensor<f32, S>,
        border_mode: BorderMode,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.convolve_2d(input, kernel, border_mode),
            ComputeDevice::Gpu(gpu) => gpu.convolve_2d(input, kernel, border_mode),
        }
    }

    pub fn threshold<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        thresh: u8,
        max_value: u8,
        typ: ThresholdType,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.threshold(input, thresh, max_value, typ),
            ComputeDevice::Gpu(gpu) => gpu.threshold(input, thresh, max_value, typ),
        }
    }

    pub fn sobel<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> Result<(Tensor<u8, S>, Tensor<u8, S>)> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.sobel(input, dx, dy, ksize),
            ComputeDevice::Gpu(gpu) => gpu.sobel(input, dx, dy, ksize),
        }
    }

    pub fn morphology<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.morphology(input, typ, kernel, iterations),
            ComputeDevice::Gpu(gpu) => gpu.morphology(input, typ, kernel, iterations),
        }
    }

    pub fn warp<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        matrix: &[[f32; 3]; 3],
        new_shape: (usize, usize),
        typ: WarpType,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.warp(input, matrix, new_shape, typ),
            ComputeDevice::Gpu(gpu) => gpu.warp(input, matrix, new_shape, typ),
        }
    }

    pub fn nms<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        threshold: f32,
        window_size: usize,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.nms(input, threshold, window_size),
            ComputeDevice::Gpu(gpu) => gpu.nms(input, threshold, window_size),
        }
    }

    pub fn nms_boxes<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.nms_boxes(input, iou_threshold),
            ComputeDevice::Gpu(gpu) => gpu.nms_boxes(input, iou_threshold),
        }
    }

    pub fn nms_rotated_boxes<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.nms_rotated_boxes(input, iou_threshold),
            ComputeDevice::Gpu(gpu) => gpu.nms_rotated_boxes(input, iou_threshold),
        }
    }

    pub fn nms_polygons(
        &self,
        polygons: &[cv_core::Polygon],
        scores: &[f32],
        iou_threshold: f32,
    ) -> Result<Vec<usize>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.nms_polygons(polygons, scores, iou_threshold),
            ComputeDevice::Gpu(gpu) => gpu.nms_polygons(polygons, scores, iou_threshold),
        }
    }

    pub fn pointcloud_transform<S: Storage<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        transform: &[[f32; 4]; 4],
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.pointcloud_transform(points, transform),
            ComputeDevice::Gpu(gpu) => gpu.pointcloud_transform(points, transform),
        }
    }

    pub fn pointcloud_normals<S: Storage<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        k_neighbors: u32,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.pointcloud_normals(points, k_neighbors),
            ComputeDevice::Gpu(gpu) => gpu.pointcloud_normals(points, k_neighbors),
        }
    }

    pub fn tsdf_integrate<S: Storage<f32> + 'static>(
        &self,
        depth_image: &Tensor<f32, S>,
        camera_pose: &[[f32; 4]; 4],
        intrinsics: &[f32; 4],
        tsdf_volume: &mut Tensor<f32, S>,
        weight_volume: &mut Tensor<f32, S>,
        voxel_size: f32,
        truncation: f32,
    ) -> Result<()> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.tsdf_integrate(depth_image, camera_pose, intrinsics, tsdf_volume, weight_volume, voxel_size, truncation),
            ComputeDevice::Gpu(gpu) => gpu.tsdf_integrate(depth_image, camera_pose, intrinsics, tsdf_volume, weight_volume, voxel_size, truncation),
        }
    }

    pub fn cvt_color<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        code: ColorConversion,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.cvt_color(input, code),
            ComputeDevice::Gpu(gpu) => gpu.cvt_color(input, code),
        }
    }

    pub fn resize<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        new_shape: (usize, usize),
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.resize(input, new_shape),
            ComputeDevice::Gpu(gpu) => gpu.resize(input, new_shape),
        }
    }

    pub fn bilateral_filter<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        d: i32,
        sigma_color: f32,
        sigma_space: f32,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.bilateral_filter(input, d, sigma_color, sigma_space),
            ComputeDevice::Gpu(gpu) => gpu.bilateral_filter(input, d, sigma_color, sigma_space),
        }
    }

    pub fn fast_detect<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        threshold: u8,
        non_max_suppression: bool,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.fast_detect(input, threshold, non_max_suppression),
            ComputeDevice::Gpu(gpu) => gpu.fast_detect(input, threshold, non_max_suppression),
        }
    }

    pub fn gaussian_blur<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        sigma: f32,
        k_size: usize,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.gaussian_blur(input, sigma, k_size),
            ComputeDevice::Gpu(gpu) => gpu.gaussian_blur(input, sigma, k_size),
        }
    }

    pub fn subtract<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug, S: Storage<T> + 'static>(
        &self,
        a: &Tensor<T, S>,
        b: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.subtract(a, b),
            ComputeDevice::Gpu(gpu) => gpu.subtract(a, b),
        }
    }

    pub fn match_descriptors<S: Storage<u8> + 'static>(
        &self,
        query: &Tensor<u8, S>,
        train: &Tensor<u8, S>,
        ratio_threshold: f32,
    ) -> Result<cv_core::Matches> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.match_descriptors(query, train, ratio_threshold),
            ComputeDevice::Gpu(gpu) => gpu.match_descriptors(query, train, ratio_threshold),
        }
    }

    pub fn sift_extrema<S: Storage<f32> + 'static>(
        &self,
        dog_prev: &Tensor<f32, S>,
        dog_curr: &Tensor<f32, S>,
        dog_next: &Tensor<f32, S>,
        threshold: f32,
        edge_threshold: f32,
    ) -> Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.sift_extrema(dog_prev, dog_curr, dog_next, threshold, edge_threshold),
            ComputeDevice::Gpu(gpu) => gpu.sift_extrema(dog_prev, dog_curr, dog_next, threshold, edge_threshold),
        }
    }

    pub fn compute_sift_descriptors<S: Storage<f32> + 'static>(
        &self,
        image: &Tensor<f32, S>,
        keypoints: &cv_core::KeyPoints,
    ) -> Result<cv_core::Descriptors> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.compute_sift_descriptors(image, keypoints),
            ComputeDevice::Gpu(gpu) => gpu.compute_sift_descriptors(image, keypoints),
        }
    }

    pub fn icp_correspondences<S: Storage<f32> + 'static>(
        &self,
        src: &Tensor<f32, S>,
        tgt: &Tensor<f32, S>,
        max_dist: f32,
    ) -> Result<Vec<(usize, usize, f32)>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.icp_correspondences(src, tgt, max_dist),
            ComputeDevice::Gpu(gpu) => gpu.icp_correspondences(src, tgt, max_dist),
        }
    }

    pub fn icp_accumulate<S: Storage<f32> + 'static>(
        &self,
        source: &Tensor<f32, S>,
        target: &Tensor<f32, S>,
        target_normals: &Tensor<f32, S>,
        correspondences: &[(u32, u32)],
        transform: &nalgebra::Matrix4<f32>,
    ) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.icp_accumulate(source, target, target_normals, correspondences, transform),
            ComputeDevice::Gpu(gpu) => gpu.icp_accumulate(source, target, target_normals, correspondences, transform),
        }
    }

    pub fn akaze_diffusion<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        k: f32,
        tau: f32,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.akaze_diffusion(input, k, tau),
            ComputeDevice::Gpu(gpu) => gpu.akaze_diffusion(input, k, tau),
        }
    }

    pub fn akaze_derivatives<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
    ) -> Result<(Tensor<f32, S>, Tensor<f32, S>, Tensor<f32, S>)> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.akaze_derivatives(input),
            ComputeDevice::Gpu(gpu) => gpu.akaze_derivatives(input),
        }
    }

    pub fn akaze_contrast_k<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
    ) -> Result<f32> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.akaze_contrast_k(input),
            ComputeDevice::Gpu(gpu) => gpu.akaze_contrast_k(input),
        }
    }

    pub fn spmv<S: Storage<f32> + 'static>(
        &self,
        row_ptr: &[u32],
        col_indices: &[u32],
        values: &[f32],
        x: &Tensor<f32, S>,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.spmv(row_ptr, col_indices, values, x),
            ComputeDevice::Gpu(gpu) => gpu.spmv(row_ptr, col_indices, values, x),
        }
    }

    pub fn mog2_update<S1: Storage<f32> + 'static, S2: Storage<u32> + 'static>(
        &self,
        frame: &Tensor<f32, S1>,
        model: &mut Tensor<f32, S1>,
        mask: &mut Tensor<u32, S2>,
        params: &crate::context::Mog2Params,
    ) -> Result<()> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.mog2_update(frame, model, mask, params),
            ComputeDevice::Gpu(gpu) => gpu.mog2_update(frame, model, mask, params),
        }
    }

    /// Get a pooled GPU buffer if this is a GPU device.
    pub fn get_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> Result<wgpu::Buffer> {
        match self {
            ComputeDevice::Gpu(gpu) => Ok(gpu.get_buffer(size, usage)),
            ComputeDevice::Cpu(_) => Err(crate::Error::NotSupported("GPU buffer pooling not available on CPU".into())),
        }
    }

    /// Return a buffer to the pool if this is a GPU device.
    pub fn return_buffer(&self, buffer: wgpu::Buffer, usage: wgpu::BufferUsages) -> Result<()> {
        match self {
            ComputeDevice::Gpu(gpu) => {
                gpu.return_buffer(buffer, usage);
                Ok(())
            }
            ComputeDevice::Cpu(_) => Err(crate::Error::NotSupported("GPU buffer pooling not available on CPU".into())),
        }
    }
}

/// Get the best available compute device.
pub fn get_device() -> ComputeDevice<'static> {
    if let Some(gpu) = GpuContext::global() {
        ComputeDevice::Gpu(gpu)
    } else {
        ComputeDevice::Cpu(CPU_CONTEXT.get_or_init(|| CpuBackend::new().unwrap()))
    }
}
