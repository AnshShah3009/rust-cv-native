use crate::gpu::GpuContext;
use crate::cpu::CpuBackend;
use crate::context::{ComputeContext, BorderMode, ThresholdType, MorphologyType, WarpType};
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
}

/// Get the best available compute device.
pub fn get_device() -> ComputeDevice<'static> {
    if let Some(gpu) = GpuContext::global() {
        ComputeDevice::Gpu(gpu)
    } else {
        ComputeDevice::Cpu(CPU_CONTEXT.get_or_init(|| CpuBackend::new().unwrap()))
    }
}
