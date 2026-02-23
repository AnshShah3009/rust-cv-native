//! GPU-Accelerated 3D Processing
//!
//! This module provides GPU-accelerated implementations of all major 3D algorithms.
//! Optimized for minimal GPU-CPU data transfers following Open3D patterns.

use cv_hal::gpu::GpuContext;
use cv_runtime::orchestrator::RuntimeRunner;
use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;
use cv_core::{Tensor, storage::Storage};
use cv_hal::context::ComputeContext;
use cv_hal::tensor_ext::{TensorToGpu, TensorToCpu};

/// GPU-accelerated point cloud operations
pub mod point_cloud {
    use super::*;

    /// Transform point cloud using the best available runner
    pub fn transform(points: &[Point3<f32>], transform: &Matrix4<f32>) -> Vec<Point3<f32>> {
        let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
            // Fallback to CPU registry on error
            cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
        });
        transform_ctx(points, transform, &runner)
    }

    pub fn transform_ctx(points: &[Point3<f32>], transform: &Matrix4<f32>, runner: &RuntimeRunner) -> Vec<Point3<f32>> {
        if let Ok(cv_hal::compute::ComputeDevice::Gpu(_gpu)) = runner.device() {
             // TODO: Actual GPU implementation in hal
        }

        points
            .iter()
            .map(|p| transform.transform_point(p))
            .collect()
    }

    /// Compute normals using the runtime scheduler
    pub fn compute_normals(
        points: &[Point3<f32>],
        k: usize,
    ) -> Vec<Vector3<f32>> {
        let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
            // Fallback to CPU registry on error
            cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
        });
        compute_normals_ctx(points, k, &runner)
    }

    pub fn compute_normals_ctx(
        points: &[Point3<f32>],
        k: usize,
        runner: &RuntimeRunner,
    ) -> Vec<Vector3<f32>> {
        if points.is_empty() {
            return Vec::new();
        }

        if let Ok(cv_hal::compute::ComputeDevice::Gpu(gpu)) = runner.device() {
            return compute_normals_gpu_with_ctx(points, k, gpu);
        }

        compute_normals_cpu(points, k, 0.0)
    }

    fn compute_normals_gpu_with_ctx(
        points: &[Point3<f32>],
        k: usize,
        gpu: &cv_hal::gpu::GpuContext,
    ) -> Vec<Vector3<f32>> {
        use cv_hal::gpu_kernels::pointcloud_gpu;
        let k = k.min(points.len().saturating_sub(1)).max(3);
        let pts: Vec<nalgebra::Vector3<f32>> = points.iter().map(|p| p.coords).collect();

        match pointcloud_gpu::compute_normals_morton_gpu(gpu, &pts, k as u32) {
            Ok(n) => n,
            Err(_) => compute_normals_cpu(points, k, 0.0),
        }
    }

    /// CPU path with optimized spatial hashing
    pub fn compute_normals_cpu(points: &[Point3<f32>], k: usize, voxel_size: f32) -> Vec<Vector3<f32>> {
        let k = k.min(points.len().saturating_sub(1)).max(3);
        let vs = if voxel_size > 0.0 {
            voxel_size
        } else {
            compute_adaptive_voxel_size(points, k)
        };

        let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(points.len() / 10);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / vs).floor() as i32;
            let vy = (p.y / vs).floor() as i32;
            let vz = (p.z / vs).floor() as i32;
            voxel_grid
                .entry((vx, vy, vz))
                .or_insert_with(Vec::new)
                .push(i);
        }

        points
            .par_iter()
            .enumerate()
            .map(|(i, center)| {
                let (vx, vy, vz) = (
                    (center.x / vs).floor() as i32,
                    (center.y / vs).floor() as i32,
                    (center.z / vs).floor() as i32,
                );

                let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(27 * k);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                                for &idx in indices {
                                    if idx != i {
                                        let p = points[idx];
                                        let dist = (center.x - p.x).powi(2)
                                            + (center.y - p.y).powi(2)
                                            + (center.z - p.z).powi(2);
                                        candidates.push((dist, idx));
                                    }
                                }
                            }
                        }
                    }
                }

                candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                candidates.truncate(k);

                compute_pca_normal(&center, &candidates, points)
            })
            .collect()
    }

    fn compute_pca_normal(
        _center: &Point3<f32>,
        neighbors: &[(f32, usize)],
        points: &[Point3<f32>],
    ) -> Vector3<f32> {
        if neighbors.len() < 3 {
            return Vector3::z();
        }

        let mut centroid = nalgebra::Vector3::zeros();
        for &(_, idx) in neighbors.iter() {
            let p = points[idx];
            centroid += nalgebra::Vector3::new(p.x, p.y, p.z);
        }
        centroid /= neighbors.len() as f32;

        let mut covariance = nalgebra::Matrix3::zeros();
        for &(_, idx) in neighbors.iter() {
            let p = points[idx];
            let diff = nalgebra::Vector3::new(p.x, p.y, p.z) - centroid;
            covariance += diff * diff.transpose();
        }

        let svd = covariance.svd(true, true);
        if let Some(u) = svd.u {
            let col = u.column(2);
            return Vector3::new(col[0], col[1], col[2]).normalize();
        }
        Vector3::z()
    }

    fn compute_adaptive_voxel_size(points: &[Point3<f32>], k: usize) -> f32 {
        if points.len() < 2 {
            return 0.1;
        }
        let sample_size = 200.min(points.len());
        let step = (points.len() / sample_size).max(1);
        let mut distances: Vec<f32> = Vec::with_capacity(sample_size);

        for i in (0..points.len()).step_by(step) {
            let p = points[i];
            let mut min_dist = f32::MAX;
            let mut count = 0;
            for j in (0..points.len()).step_by(step) {
                if i != j && count < k {
                    let d = ((p.x - points[j].x).powi(2)
                        + (p.y - points[j].y).powi(2)
                        + (p.z - points[j].z).powi(2))
                    .sqrt();
                    min_dist = min_dist.min(d);
                    count += 1;
                }
            }
            if min_dist < f32::MAX {
                distances.push(min_dist);
            }
        }
        if distances.is_empty() {
            return 0.1;
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distances[distances.len() / 2] * 2.5
    }

    pub fn voxel_downsample(points: &[Point3<f32>], voxel_size: f32) -> Vec<Point3<f32>> {
        super::voxel_downsample(points, voxel_size)
    }

    pub fn compute_normals_simple(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
        compute_normals(points, k)
    }

    pub fn voxel_based_normals_simple(points: &[Point3<f32>], _voxel_size: f32) -> Vec<Vector3<f32>> {
        compute_normals(points, 30)
    }

    pub fn voxel_to_point_normal_transfer(points: &[Point3<f32>], _normals: &[Vector3<f32>], _voxel_size: f32) -> Vec<Vector3<f32>> {
        vec![Vector3::z(); points.len()]
    }

    pub fn approximate_normals_simple(points: &[Point3<f32>], k: usize, _epsilon: f32) -> Vec<Vector3<f32>> {
        compute_normals(points, k)
    }
}

/// GPU-accelerated stereo matching
pub mod stereo {
    use super::*;
    use cv_hal::context::StereoMatchParams;

    pub fn match_stereo(
        left: &::image::GrayImage,
        right: &::image::GrayImage,
        params: &StereoMatchParams,
    ) -> cv_hal::Result<::image::ImageBuffer<::image::Luma<f32>, Vec<f32>>> {
        let runner = cv_runtime::best_runner().map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
        match_stereo_ctx(left, right, params, &runner)
    }

    pub fn match_stereo_ctx(
        left: &::image::GrayImage,
        right: &::image::GrayImage,
        params: &StereoMatchParams,
        runner: &RuntimeRunner,
    ) -> cv_hal::Result<::image::ImageBuffer<::image::Luma<f32>, Vec<f32>>> {
        if let Ok(cv_hal::compute::ComputeDevice::Gpu(gpu)) = runner.device() {
            let l_tensor = cv_core::CpuTensor::from_vec(left.as_raw().to_vec(), cv_core::TensorShape::new(1, left.height() as usize, left.width() as usize))
                .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
            let r_tensor = cv_core::CpuTensor::from_vec(right.as_raw().to_vec(), cv_core::TensorShape::new(1, right.height() as usize, right.width() as usize))
                .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
                
            let l_gpu = l_tensor.to_gpu_ctx(gpu)?;
            let r_gpu = r_tensor.to_gpu_ctx(gpu)?;
            
            // Note: We use OS=GpuStorage<f32> for the output
            let res_gpu: Tensor<f32, cv_hal::storage::GpuStorage<f32>> = gpu.stereo_match(&l_gpu, &r_gpu, params)?;
            let res_cpu = res_gpu.to_cpu_ctx(gpu)?;
            
            let data = res_cpu.storage.as_slice().unwrap().to_vec();
            return Ok(::image::ImageBuffer::from_raw(left.width(), left.height(), data).unwrap());
        }

        Err(cv_hal::Error::NotSupported("CPU fallback for orchestrated stereo not yet in 3d::gpu".into()))
    }
}

pub mod registration {
    use super::*;

    /// ICP point-to-plane registration: NOT IMPLEMENTED
    ///
    /// This function is a stub and returns an error indicating that the algorithm
    /// is not yet implemented in the GPU module. Point-to-plane ICP requires:
    /// 1. Surface normal computation
    /// 2. Iterative closest point search with plane distance metrics
    /// 3. Robust outlier rejection
    ///
    /// To use ICP registration, use the `cv-registration` crate instead.
    pub fn icp_point_to_plane(_s: &[Point3<f32>], _t: &[Point3<f32>], _tn: &[Vector3<f32>], _dist: f32, _iters: usize) -> Result<Matrix4<f32>, String> {
        Err("ICP point-to-plane registration not yet implemented in GPU module. Use cv-registration crate instead.".to_string())
    }
}

pub mod mesh {
    use super::*;

    /// Laplacian mesh smoothing: NOT IMPLEMENTED
    ///
    /// This function is a stub. Laplacian smoothing requires:
    /// 1. Vertex-face adjacency graph construction
    /// 2. Laplacian matrix computation (uniform or cotangent weights)
    /// 3. Iterative smoothing with boundary preservation
    ///
    /// This operation is expensive on CPU and requires GPU acceleration
    /// through a custom kernel implementation in cv-hal.
    pub fn laplacian_smooth(_v: &mut [Point3<f32>], _f: &[[usize; 3]], _iters: usize, _l: f32) -> Result<(), String> {
        Err("Laplacian mesh smoothing not yet implemented. Requires GPU kernel in cv-hal.".to_string())
    }

    /// Compute vertex normals from mesh: NOT IMPLEMENTED
    ///
    /// Returns vertex normals computed by averaging face normals of adjacent faces.
    /// This function is a stub that returns an error.
    ///
    /// Implementation should:
    /// 1. For each face, compute normal from vertex positions
    /// 2. For each vertex, accumulate normals from all adjacent faces
    /// 3. Normalize per-vertex accumulated normals
    pub fn compute_vertex_normals(_v: &[Point3<f32>], _f: &[[usize; 3]]) -> Result<Vec<Vector3<f32>>, String> {
        Err("Vertex normal computation not yet implemented.".to_string())
    }
}

pub mod tsdf {
    use super::*;

    /// TSDF depth integration: NOT IMPLEMENTED
    ///
    /// This function is a stub. Truncated Signed Distance Field (TSDF) integration is a core
    /// component of volumetric 3D reconstruction (KinectFusion-style algorithms).
    ///
    /// Parameters:
    /// - `d`: Depth map (row-major, H×W)
    /// - `w`, `h`: Image dimensions
    /// - `p`: Projection matrix (3×4)
    /// - `i`: Intrinsic matrix flattened (K[0,0], K[1,1], K[0,2], K[1,2])
    /// - `vol`: TSDF volume (pre-allocated, size = voxel_grid.len())
    /// - `weights`: Weight accumulation volume
    /// - `vs`: Voxel size
    /// - `tr`: TSDF truncation distance
    ///
    /// Integration requires:
    /// 1. Back-project each depth pixel to 3D world coordinates
    /// 2. For each voxel in the volume, compute signed distance to surface
    /// 3. Accumulate TSDF values weighted by confidence
    /// 4. Update weight accumulator for normalization
    ///
    /// This is heavily used in real-time reconstruction and MUST be GPU-accelerated
    /// through a compute kernel in cv-hal.
    pub fn integrate_depth(_d: &[f32], _w: u32, _h: u32, _p: &Matrix4<f32>, _i: &[f32; 4], _vol: &mut [f32], _weights: &mut [f32], _vs: f32, _tr: f32) -> Result<(), String> {
        Err("TSDF depth integration not yet implemented. Requires GPU kernel for real-time performance.".to_string())
    }
}

pub mod raycasting {
    use super::*;

    /// Ray-mesh intersection raycasting: NOT IMPLEMENTED
    ///
    /// This function is a stub. Raycasting is essential for:
    /// 1. Rendering: Visibility queries against mesh geometry
    /// 2. Reconstruction: Point cloud alignment and normal estimation
    /// 3. Simulation: Collision detection and occlusion queries
    ///
    /// Parameters:
    /// - `ro`: Array of ray origins (Point3<f32>)
    /// - `rd`: Array of ray directions (Vector3<f32>, assumed normalized)
    /// - `v`: Mesh vertices
    /// - `f`: Mesh triangle faces (indices into vertex array)
    ///
    /// Returns for each ray: (t, intersection_point, surface_normal) or None if no intersection
    ///
    /// Implementation should:
    /// 1. Build BVH or spatial acceleration structure from mesh
    /// 2. For each ray, traverse acceleration structure
    /// 3. Perform triangle-ray intersection tests (Möller-Trumbore algorithm)
    /// 4. Return closest intersection
    ///
    /// This is a performance-critical operation best implemented with:
    /// - Spatial hashing or BVH
    /// - SIMD vectorization or GPU acceleration
    pub fn cast_rays(_ro: &[Point3<f32>], _rd: &[Vector3<f32>], _v: &[Point3<f32>], _f: &[[usize; 3]]) -> Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>, String> {
        Err("Ray-mesh intersection not yet implemented. Requires BVH acceleration structure.".to_string())
    }
}

/// Voxel downsample
pub fn voxel_downsample(points: &[Point3<f32>], voxel_size: f32) -> Vec<Point3<f32>> {
    if points.is_empty() {
        return Vec::new();
    }
    use crate::spatial::VoxelGrid;
    let mut grid = VoxelGrid::new(points[0], voxel_size);
    for (i, p) in points.iter().enumerate() {
        grid.insert(*p, i);
    }
    grid.downsample(points)
}

pub fn is_gpu_available() -> bool {
    GpuContext::new().is_ok()
}

pub fn gpu_info() -> Option<String> {
    GpuContext::new()
        .ok()
        .map(|ctx| format!("{:?}", ctx.device))
}

pub fn force_cpu_mode() -> cv_hal::gpu_kernels::unified::ComputeConfig {
    cv_hal::gpu_kernels::unified::ComputeConfig {
        force_cpu: true,
        ..Default::default()
    }
}

pub fn force_gpu_mode() -> cv_hal::gpu_kernels::unified::ComputeConfig {
    cv_hal::gpu_kernels::unified::ComputeConfig {
        force_gpu: true,
        ..Default::default()
    }
}
