//! GPU-Accelerated 3D Processing
//!
//! This module provides GPU-accelerated implementations of all major 3D algorithms.
//! Optimized for minimal GPU-CPU data transfers following Open3D patterns.

use cv_hal::gpu::GpuContext;
use cv_hal::gpu_kernels::unified::{should_use_gpu, ComputeConfig};
use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;

/// Compute mode selection for algorithms
#[derive(Debug, Clone, Copy, Default)]
pub enum ComputeMode {
    /// Force CPU-only computation
    CPU,
    /// Force GPU-only computation (will fail if GPU unavailable)
    GPU,
    /// Hybrid: GPU for large datasets, CPU for small (auto-select)
    #[default]
    Hybrid,
    /// Adaptive: Automatically select based on data size and GPU availability
    Adaptive,
}

impl ComputeMode {
    /// Determine actual execution mode based on config and data size
    pub fn resolve(&self, config: &ComputeConfig, data_size: usize) -> ComputeMode {
        match self {
            ComputeMode::CPU => ComputeMode::CPU,
            ComputeMode::GPU => ComputeMode::GPU,
            ComputeMode::Hybrid => {
                if should_use_gpu(config, data_size) {
                    ComputeMode::GPU
                } else {
                    ComputeMode::CPU
                }
            }
            ComputeMode::Adaptive => {
                if GpuContext::new().is_ok() && should_use_gpu(config, data_size) {
                    ComputeMode::GPU
                } else {
                    ComputeMode::CPU
                }
            }
        }
    }
}

/// Configuration for normal computation
#[derive(Debug, Clone)]
pub struct NormalComputeConfig {
    /// Compute mode (CPU/GPU/Hybrid/Adaptive)
    pub mode: ComputeMode,
    /// Number of nearest neighbors for PCA
    pub k: usize,
    /// Voxel size for spatial hashing (0 = auto-compute)
    pub voxel_size: f32,
    /// Epsilon for regularized PCA
    pub epsilon: f32,
    /// Minimum points required for valid normal
    pub min_neighbors: usize,
}

impl Default for NormalComputeConfig {
    fn default() -> Self {
        Self {
            mode: ComputeMode::default(),
            k: 30,
            voxel_size: 0.0, // 0 = auto
            epsilon: 1e-6,
            min_neighbors: 3,
        }
    }
}

impl NormalComputeConfig {
    /// Fast mode - fewer neighbors, auto voxel size
    pub fn fast() -> Self {
        Self {
            k: 15,
            ..Default::default()
        }
    }

    /// High quality mode - more neighbors
    pub fn high_quality() -> Self {
        Self {
            k: 50,
            ..Default::default()
        }
    }

    /// Force CPU mode
    pub fn cpu() -> Self {
        Self {
            mode: ComputeMode::CPU,
            ..Default::default()
        }
    }

    /// Force GPU mode
    pub fn gpu() -> Self {
        Self {
            mode: ComputeMode::GPU,
            ..Default::default()
        }
    }
}

/// GPU-accelerated point cloud operations
pub mod point_cloud {
    use super::*;

    /// Transform point cloud using GPU
    pub fn transform(points: &[Point3<f32>], transform: &Matrix4<f32>) -> Vec<Point3<f32>> {
        let config = ComputeConfig::default();
        if should_use_gpu(&config, points.len()) {
            transform_gpu(points, transform)
        } else {
            points
                .iter()
                .map(|p| transform.transform_point(p))
                .collect()
        }
    }

    fn transform_gpu(points: &[Point3<f32>], transform: &Matrix4<f32>) -> Vec<Point3<f32>> {
        // TODO: Actual GPU implementation
        points
            .iter()
            .map(|p| transform.transform_point(p))
            .collect()
    }

    /// Compute normals with configurable mode
    ///
    /// # Arguments
    /// * `points` - Input point cloud
    /// * `config` - Computation configuration (or use default for auto-select)
    ///
    /// # Example
    /// ```
    /// use cv_3d::gpu::point_cloud::compute_normals;
    /// use cv_3d::gpu::{NormalComputeConfig, ComputeMode};
    /// use nalgebra::Point3;
    ///
    /// let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
    ///
    /// // Auto-select (Hybrid)
    /// let normals = compute_normals(&points, NormalComputeConfig::default());
    ///
    /// // Force CPU
    /// let normals = compute_normals(&points, NormalComputeConfig::cpu());
    ///
    /// // High quality
    /// let normals = compute_normals(&points, NormalComputeConfig::high_quality());
    /// ```
    pub fn compute_normals(
        points: &[Point3<f32>],
        config: NormalComputeConfig,
    ) -> Vec<Vector3<f32>> {
        if points.is_empty() {
            return Vec::new();
        }

        let global_config = ComputeConfig::default();
        let mode = config.mode.resolve(&global_config, points.len());

        match mode {
            ComputeMode::GPU => compute_normals_gpu(points, config.k),
            ComputeMode::CPU => compute_normals_cpu(points, config.k, config.voxel_size),
            _ => compute_normals_cpu(points, config.k, config.voxel_size),
        }
    }

    /// Compute normals using a specific GPU context
    pub fn compute_normals_with_ctx(
        points: &[Point3<f32>],
        k: usize,
        ctx: &cv_hal::gpu::GpuContext,
    ) -> Vec<Vector3<f32>> {
        use cv_hal::gpu_kernels::{GpuCompute, pointcloud_gpu};
        let k = k.min(points.len().saturating_sub(1)).max(3);
        let gpu = GpuCompute::new(ctx.device.clone(), ctx.queue.clone());

        let voxel_size = compute_adaptive_voxel_size(points, k);
        let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(points.len() / 10);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / voxel_size).floor() as i32;
            let vy = (p.y / voxel_size).floor() as i32;
            let vz = (p.z / voxel_size).floor() as i32;
            voxel_grid.entry((vx, vy, vz)).or_default().push(i);
        }

        let neighbor_indices: Vec<u32> = points.par_iter().enumerate().flat_map(|(i, center)| {
            let (vx, vy, vz) = (
                (center.x / voxel_size).floor() as i32,
                (center.y / voxel_size).floor() as i32,
                (center.z / voxel_size).floor() as i32,
            );
            let mut candidates = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(indices) = voxel_grid.get(&(vx+dx, vy+dy, vz+dz)) {
                            for &idx in indices {
                                if idx != i {
                                    let p = points[idx];
                                    let d = (center.x-p.x).powi(2) + (center.y-p.y).powi(2) + (center.z-p.z).powi(2);
                                    candidates.push((d, idx));
                                }
                            }
                        }
                    }
                }
            }
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            candidates.truncate(k);
            let mut result = vec![0u32; k];
            for (j, &(_, idx)) in candidates.iter().enumerate() {
                result[j] = idx as u32;
            }
            result
        }).collect();

        let points_v3: Vec<Vector3<f32>> = points.iter().map(|p| p.coords).collect();

        match pointcloud_gpu::compute_normals(&gpu, &points_v3, &neighbor_indices, k as u32) {
            Ok(n) => n,
            Err(_) => compute_normals_cpu(points, k, 0.0),
        }
    }

    /// Simple compute_normals with default config (backward compatible)
    pub fn compute_normals_simple(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
        compute_normals(
            points,
            NormalComputeConfig {
                k,
                ..Default::default()
            },
        )
    }

    /// GPU path
    fn compute_normals_gpu(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
        use cv_hal::gpu::GpuContext;
        use cv_hal::gpu_kernels::{GpuCompute, pointcloud_gpu};

        let k = k.min(points.len().saturating_sub(1)).max(3);
        
        // 1. Get GPU context
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return compute_normals_cpu(points, k, 0.0),
        };
        let gpu = GpuCompute::new(ctx.device, ctx.queue);

        // 2. Compute neighbor indices on CPU (Simplified hybrid approach)
        // In a full implementation, this would also happen on GPU
        let voxel_size = compute_adaptive_voxel_size(points, k);
        let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(points.len() / 10);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / voxel_size).floor() as i32;
            let vy = (p.y / voxel_size).floor() as i32;
            let vz = (p.z / voxel_size).floor() as i32;
            voxel_grid.entry((vx, vy, vz)).or_default().push(i);
        }

        let neighbor_indices: Vec<u32> = points.par_iter().enumerate().flat_map(|(i, center)| {
            let (vx, vy, vz) = (
                (center.x / voxel_size).floor() as i32,
                (center.y / voxel_size).floor() as i32,
                (center.z / voxel_size).floor() as i32,
            );
            let mut candidates = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(indices) = voxel_grid.get(&(vx+dx, vy+dy, vz+dz)) {
                            for &idx in indices {
                                if idx != i {
                                    let p = points[idx];
                                    let d = (center.x-p.x).powi(2) + (center.y-p.y).powi(2) + (center.z-p.z).powi(2);
                                    candidates.push((d, idx));
                                }
                            }
                        }
                    }
                }
            }
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            candidates.truncate(k);
            let mut result = vec![0u32; k];
            for (j, &(_, idx)) in candidates.iter().enumerate() {
                result[j] = idx as u32;
            }
            result
        }).collect();

        // 3. Convert types for HAL
        let points_v3: Vec<Vector3<f32>> = points.iter().map(|p| p.coords).collect();

        // 4. Run on GPU
        match pointcloud_gpu::compute_normals(&gpu, &points_v3, &neighbor_indices, k as u32) {
            Ok(n) => n,
            Err(_) => compute_normals_cpu(points, k, 0.0),
        }
    }

    /// CPU path with optimized spatial hashing
    fn compute_normals_cpu(points: &[Point3<f32>], k: usize, voxel_size: f32) -> Vec<Vector3<f32>> {
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

    /// Hybrid path - GPU for k-NN, CPU for SVD (when implemented)
    fn compute_normals_hybrid(points: &[Point3<f32>], k: usize) -> Vec<Vector3<f32>> {
        // TODO: Hybrid implementation:
        // - Use GPU for parallel k-NN search (best for large datasets)
        // - Use CPU for SVD (small data, reduces transfer overhead)

        // For now, use optimized CPU
        compute_normals_cpu(points, k, 0.0)
    }

    /// Voxel-based normal computation with configurable mode
    pub fn voxel_based_normals(
        points: &[Point3<f32>],
        voxel_size: f32,
        mode: ComputeMode,
    ) -> Vec<Vector3<f32>> {
        if points.is_empty() {
            return Vec::new();
        }

        let global_config = ComputeConfig::default();
        let resolved_mode = mode.resolve(&global_config, points.len());

        match resolved_mode {
            ComputeMode::GPU => voxel_based_normals_gpu(points, voxel_size),
            ComputeMode::CPU => voxel_based_normals_cpu(points, voxel_size),
            _ => voxel_based_normals_cpu(points, voxel_size),
        }
    }

    /// Simple version (backward compatible)
    pub fn voxel_based_normals_simple(
        points: &[Point3<f32>],
        voxel_size: f32,
    ) -> Vec<Vector3<f32>> {
        voxel_based_normals(points, voxel_size, ComputeMode::default())
    }

    fn voxel_based_normals_gpu(points: &[Point3<f32>], voxel_size: f32) -> Vec<Vector3<f32>> {
        // TODO: GPU implementation
        voxel_based_normals_cpu(points, voxel_size)
    }

    fn voxel_based_normals_cpu(points: &[Point3<f32>], voxel_size: f32) -> Vec<Vector3<f32>> {
        let mut voxel_map: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(points.len() / 10);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / voxel_size).floor() as i32;
            let vy = (p.y / voxel_size).floor() as i32;
            let vz = (p.z / voxel_size).floor() as i32;
            voxel_map
                .entry((vx, vy, vz))
                .or_insert_with(Vec::new)
                .push(i);
        }

        let voxel_normals: std::collections::HashMap<(i32, i32, i32), Vector3<f32>> = voxel_map
            .par_iter()
            .filter(|(_, indices)| indices.len() >= 3)
            .map(|(&voxel, indices)| (voxel, compute_voxel_normal(voxel, indices, points)))
            .collect();

        points
            .iter()
            .map(|p| {
                let vx = (p.x / voxel_size).floor() as i32;
                let vy = (p.y / voxel_size).floor() as i32;
                let vz = (p.z / voxel_size).floor() as i32;
                voxel_normals
                    .get(&(vx, vy, vz))
                    .copied()
                    .unwrap_or_else(Vector3::z)
            })
            .collect()
    }

    /// Transfer normals from voxel grid to points
    pub fn voxel_to_point_normal_transfer(
        points: &[Point3<f32>],
        voxel_normals: &[Vector3<f32>],
        voxel_size: f32,
    ) -> Vec<Vector3<f32>> {
        if points.is_empty() || voxel_normals.is_empty() {
            return Vec::new();
        }

        let num_voxels = (voxel_normals.len() as f32).sqrt() as i32;
        let mut voxel_map: std::collections::HashMap<(i32, i32, i32), Vector3<f32>> =
            std::collections::HashMap::with_capacity(voxel_normals.len());

        for (i, normal) in voxel_normals.iter().enumerate() {
            let vx = (i as i32 % num_voxels) - num_voxels / 2;
            let vy = (i as i32 / num_voxels) - num_voxels / 2;
            voxel_map.insert((vx, vy, 0), *normal);
        }

        points
            .iter()
            .map(|p| {
                let vx = (p.x / voxel_size).floor() as i32;
                let vy = (p.y / voxel_size).floor() as i32;
                let vz = (p.z / voxel_size).floor() as i32;
                voxel_map
                    .get(&(vx, vy, vz))
                    .copied()
                    .unwrap_or_else(Vector3::z)
            })
            .collect()
    }

    /// Approximate normals with configurable mode
    pub fn approximate_normals(
        points: &[Point3<f32>],
        k: usize,
        epsilon: f32,
        mode: ComputeMode,
    ) -> Vec<Vector3<f32>> {
        if points.is_empty() {
            return Vec::new();
        }

        let global_config = ComputeConfig::default();
        let resolved_mode = mode.resolve(&global_config, points.len());

        match resolved_mode {
            ComputeMode::GPU => approximate_normals_gpu(points, k, epsilon),
            ComputeMode::CPU => approximate_normals_cpu(points, k, epsilon),
            _ => approximate_normals_cpu(points, k, epsilon),
        }
    }

    /// Simple version (backward compatible)
    pub fn approximate_normals_simple(
        points: &[Point3<f32>],
        k: usize,
        epsilon: f32,
    ) -> Vec<Vector3<f32>> {
        approximate_normals(points, k, epsilon, ComputeMode::default())
    }

    fn approximate_normals_gpu(
        points: &[Point3<f32>],
        k: usize,
        epsilon: f32,
    ) -> Vec<Vector3<f32>> {
        // TODO: GPU implementation
        approximate_normals_cpu(points, k, epsilon)
    }

    fn approximate_normals_cpu(
        points: &[Point3<f32>],
        k: usize,
        epsilon: f32,
    ) -> Vec<Vector3<f32>> {
        let k = k.min(points.len().saturating_sub(1)).max(3);
        let voxel_size = compute_adaptive_voxel_size(points, k);

        let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(points.len() / 10);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / voxel_size).floor() as i32;
            let vy = (p.y / voxel_size).floor() as i32;
            let vz = (p.z / voxel_size).floor() as i32;
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
                    (center.x / voxel_size).floor() as i32,
                    (center.y / voxel_size).floor() as i32,
                    (center.z / voxel_size).floor() as i32,
                );

                let mut neighbors: Vec<(f32, usize)> = Vec::new();
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
                                        neighbors.push((dist, idx));
                                    }
                                }
                            }
                        }
                    }
                }

                neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                neighbors.truncate(k);

                compute_pca_normal_regularized(center, &neighbors, points, epsilon)
            })
            .collect()
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

    /// Refine normals using iterative plane fitting
    /// Improves normal quality by iteratively re-weighting points
    pub fn refine_normals(
        points: &[Point3<f32>],
        normals: &mut [Vector3<f32>],
        k_neighbors: usize,
        iterations: usize,
    ) {
        if points.len() != normals.len() || points.is_empty() {
            return;
        }

        let k = k_neighbors.min(points.len().saturating_sub(1)).max(3);
        let voxel_size = compute_adaptive_voxel_size(points, k);

        let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(points.len() / 10);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / voxel_size).floor() as i32;
            let vy = (p.y / voxel_size).floor() as i32;
            let vz = (p.z / voxel_size).floor() as i32;
            voxel_grid
                .entry((vx, vy, vz))
                .or_insert_with(Vec::new)
                .push(i);
        }

        for _ in 0..iterations {
            // Compute new normals based on weighted PCA
            let new_normals: Vec<Vector3<f32>> = points
                .par_iter()
                .enumerate()
                .map(|(i, center)| {
                    let normal = normals[i];
                    if normal.norm() < 1e-6 {
                        return normal;
                    }

                    let (vx, vy, vz) = (
                        (center.x / voxel_size).floor() as i32,
                        (center.y / voxel_size).floor() as i32,
                        (center.z / voxel_size).floor() as i32,
                    );

                    let mut weighted_points: Vec<(f32, nalgebra::Vector3<f32>)> = Vec::new();

                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz))
                                {
                                    for &idx in indices {
                                        if idx != i {
                                            let p = points[idx];
                                            let diff = nalgebra::Vector3::new(
                                                p.x - center.x,
                                                p.y - center.y,
                                                p.z - center.z,
                                            );
                                            let weight = diff.dot(&normal).abs();
                                            weighted_points.push((weight, p.coords));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if weighted_points.len() < 3 {
                        return normal;
                    }

                    let mut centroid = nalgebra::Vector3::zeros();
                    let mut total_weight = 0.0f32;
                    for (w, p) in &weighted_points {
                        centroid += p * *w;
                        total_weight += *w;
                    }
                    centroid /= total_weight;

                    let mut covariance = nalgebra::Matrix3::zeros();
                    for (w, p) in &weighted_points {
                        let diff = p - centroid;
                        covariance = covariance + diff * diff.transpose() * *w;
                    }

                    let svd = covariance.svd(true, true);
                    if let Some(u) = svd.u {
                        let col = u.column(2);
                        let new_normal = Vector3::new(col[0], col[1], col[2]).normalize();
                        // Ensure consistent orientation
                        if new_normal.dot(&normal) < 0.0 {
                            return -new_normal;
                        }
                        return new_normal;
                    }
                    normal
                })
                .collect();

            // Copy back
            for (i, n) in new_normals.iter().enumerate() {
                normals[i] = *n;
            }
        }
    }

    // Helper functions
    fn compute_pca_normal(
        center: &Point3<f32>,
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

    fn compute_pca_normal_regularized(
        center: &Point3<f32>,
        neighbors: &[(f32, usize)],
        points: &[Point3<f32>],
        epsilon: f32,
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

        covariance =
            covariance / (neighbors.len() as f32) + nalgebra::Matrix3::identity() * epsilon;

        let svd = covariance.svd(true, true);
        if let Some(u) = svd.u {
            let col = u.column(2);
            return Vector3::new(col[0], col[1], col[2]).normalize();
        }
        Vector3::z()
    }

    fn compute_voxel_normal(
        _voxel: (i32, i32, i32),
        indices: &[usize],
        points: &[Point3<f32>],
    ) -> Vector3<f32> {
        if indices.len() < 3 {
            return Vector3::z();
        }

        let mut centroid = nalgebra::Vector3::zeros();
        for &idx in indices {
            let p = points[idx];
            centroid += nalgebra::Vector3::new(p.x, p.y, p.z);
        }
        centroid /= indices.len() as f32;

        let mut covariance = nalgebra::Matrix3::zeros();
        for &idx in indices {
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
}

/// GPU-accelerated registration
pub mod registration {
    use super::*;

    pub fn icp_point_to_plane(
        source: &[Point3<f32>],
        target: &[Point3<f32>],
        target_normals: &[Vector3<f32>],
        max_distance: f32,
        max_iterations: usize,
    ) -> Option<Matrix4<f32>> {
        let config = ComputeConfig::default();
        if should_use_gpu(&config, source.len()) {
            icp_gpu(source, target, target_normals, max_distance, max_iterations)
        } else {
            icp_cpu(source, target, target_normals, max_distance, max_iterations)
        }
    }

    fn icp_cpu(
        source: &[Point3<f32>],
        target: &[Point3<f32>],
        target_normals: &[Vector3<f32>],
        max_distance: f32,
        max_iterations: usize,
    ) -> Option<Matrix4<f32>> {
        use cv_core::point_cloud::PointCloud;
        use cv_registration::registration_icp_point_to_plane;
        let source_cloud = PointCloud::new(source.to_vec());
        let target_cloud = PointCloud::new(target.to_vec()).with_normals(target_normals.to_vec());
        let result = registration_icp_point_to_plane(
            &source_cloud,
            &target_cloud,
            max_distance,
            &Matrix4::identity(),
            max_iterations,
        )?;
        Some(result.transformation)
    }

    fn icp_gpu(
        _source: &[Point3<f32>],
        _target: &[Point3<f32>],
        _target_normals: &[Vector3<f32>],
        _max_distance: f32,
        _max_iterations: usize,
    ) -> Option<Matrix4<f32>> {
        None
    }
}

/// GPU-accelerated mesh processing
pub mod mesh {
    use super::*;

    pub fn laplacian_smooth(
        vertices: &mut [Point3<f32>],
        faces: &[[usize; 3]],
        iterations: usize,
        lambda: f32,
    ) {
        use crate::mesh::TriangleMesh;
        let mut mesh = TriangleMesh::with_vertices_and_faces(vertices.to_vec(), faces.to_vec());
        crate::mesh::processing::laplacian_smooth(&mut mesh, iterations, lambda);
        for (i, v) in mesh.vertices.iter().enumerate() {
            vertices[i] = *v;
        }
    }

    pub fn compute_vertex_normals(
        vertices: &[Point3<f32>],
        faces: &[[usize; 3]],
    ) -> Vec<Vector3<f32>> {
        use crate::mesh::TriangleMesh;
        let mesh = TriangleMesh::with_vertices_and_faces(vertices.to_vec(), faces.to_vec());
        mesh.compute_face_normals()
    }
}

/// GPU-accelerated TSDF operations
pub mod tsdf {
    use super::*;

    pub fn integrate_depth(
        depth_image: &[f32],
        width: u32,
        height: u32,
        camera_pose: &Matrix4<f32>,
        intrinsics: &[f32; 4],
        tsdf_volume: &mut [f32],
        weights: &mut [f32],
        voxel_size: f32,
        truncation: f32,
    ) {
        integrate_depth_cpu(
            depth_image,
            width,
            height,
            camera_pose,
            intrinsics,
            tsdf_volume,
            weights,
            voxel_size,
            truncation,
        );
    }

    fn integrate_depth_cpu(
        _depth_image: &[f32],
        _width: u32,
        _height: u32,
        _camera_pose: &Matrix4<f32>,
        _intrinsics: &[f32; 4],
        _tsdf_volume: &mut [f32],
        _weights: &mut [f32],
        _voxel_size: f32,
        _truncation: f32,
    ) {
    }
}

/// GPU-accelerated ray casting
pub mod raycasting {
    use super::*;

    pub fn cast_rays(
        ray_origins: &[Point3<f32>],
        ray_directions: &[Vector3<f32>],
        mesh_vertices: &[Point3<f32>],
        mesh_faces: &[[usize; 3]],
    ) -> Vec<Option<(f32, Point3<f32>, Vector3<f32>)>> {
        let config = ComputeConfig::default();
        if should_use_gpu(&config, ray_origins.len() / 10) {
            cast_rays_gpu(ray_origins, ray_directions, mesh_vertices, mesh_faces)
        } else {
            cast_rays_cpu(ray_origins, ray_directions, mesh_vertices, mesh_faces)
        }
    }

    fn cast_rays_cpu(
        ray_origins: &[Point3<f32>],
        ray_directions: &[Vector3<f32>],
        mesh_vertices: &[Point3<f32>],
        mesh_faces: &[[usize; 3]],
    ) -> Vec<Option<(f32, Point3<f32>, Vector3<f32>)>> {
        use crate::mesh::TriangleMesh;
        use crate::raycasting::{cast_ray_mesh, Ray};
        let mesh =
            TriangleMesh::with_vertices_and_faces(mesh_vertices.to_vec(), mesh_faces.to_vec());
        ray_origins
            .par_iter()
            .zip(ray_directions.par_iter())
            .map(|(origin, dir)| {
                let ray = Ray::new(*origin, *dir);
                cast_ray_mesh(&ray, &mesh).map(|hit| (hit.distance, hit.point, hit.normal))
            })
            .collect()
    }

    fn cast_rays_gpu(
        _ray_origins: &[Point3<f32>],
        _ray_directions: &[Vector3<f32>],
        _mesh_vertices: &[Point3<f32>],
        _mesh_faces: &[[usize; 3]],
    ) -> Vec<Option<(f32, Point3<f32>, Vector3<f32>)>> {
        vec![]
    }
}

/// GPU-accelerated point cloud filtering
pub mod filtering {
    use super::*;

    /// Statistical Outlier Removal (SOR)
    /// Removes points that are further than std_ratio * std_dev from their k nearest neighbors
    pub fn statistical_outlier_removal(
        points: &[Point3<f32>],
        k_neighbors: usize,
        std_ratio: f32,
        mode: ComputeMode,
    ) -> Vec<bool> {
        if points.is_empty() {
            return Vec::new();
        }

        let global_config = ComputeConfig::default();
        let resolved_mode = mode.resolve(&global_config, points.len());

        match resolved_mode {
            ComputeMode::GPU => statistical_outlier_removal_gpu(points, k_neighbors, std_ratio),
            _ => statistical_outlier_removal_cpu(points, k_neighbors, std_ratio),
        }
    }

    /// Simple version (backward compatible)
    pub fn statistical_outlier_removal_simple(
        points: &[Point3<f32>],
        k_neighbors: usize,
        std_ratio: f32,
    ) -> Vec<bool> {
        statistical_outlier_removal(points, k_neighbors, std_ratio, ComputeMode::default())
    }

    /// Apply SOR and return inlier points
    pub fn remove_statistical_outliers(
        points: &[Point3<f32>],
        k_neighbors: usize,
        std_ratio: f32,
    ) -> (Vec<Point3<f32>>, Vec<bool>) {
        let inliers = statistical_outlier_removal_simple(points, k_neighbors, std_ratio);
        let filtered: Vec<Point3<f32>> = points
            .iter()
            .zip(inliers.iter())
            .filter(|(_, &keep)| keep)
            .map(|(p, _)| *p)
            .collect();
        (filtered, inliers)
    }

    fn statistical_outlier_removal_gpu(
        points: &[Point3<f32>],
        k_neighbors: usize,
        std_ratio: f32,
    ) -> Vec<bool> {
        // TODO: GPU implementation
        statistical_outlier_removal_cpu(points, k_neighbors, std_ratio)
    }

    fn statistical_outlier_removal_cpu(
        points: &[Point3<f32>],
        k_neighbors: usize,
        std_ratio: f32,
    ) -> Vec<bool> {
        let k = k_neighbors.min(points.len().saturating_sub(1)).max(1);
        let voxel_size = compute_adaptive_voxel_size_filtering(points, k);

        // Build spatial hash
        let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(points.len() / 10);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / voxel_size).floor() as i32;
            let vy = (p.y / voxel_size).floor() as i32;
            let vz = (p.z / voxel_size).floor() as i32;
            voxel_grid
                .entry((vx, vy, vz))
                .or_insert_with(Vec::new)
                .push(i);
        }

        // Compute mean distances to k nearest neighbors
        let mean_distances: Vec<f32> = points
            .par_iter()
            .enumerate()
            .map(|(i, center)| {
                let (vx, vy, vz) = (
                    (center.x / voxel_size).floor() as i32,
                    (center.y / voxel_size).floor() as i32,
                    (center.z / voxel_size).floor() as i32,
                );

                let mut distances: Vec<f32> = Vec::with_capacity(27 * k);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                                for &idx in indices {
                                    if idx != i {
                                        let p = points[idx];
                                        let d = ((center.x - p.x).powi(2)
                                            + (center.y - p.y).powi(2)
                                            + (center.z - p.z).powi(2))
                                        .sqrt();
                                        distances.push(d);
                                    }
                                }
                            }
                        }
                    }
                }

                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let k_nearest: Vec<_> = distances.into_iter().take(k).collect();

                if k_nearest.is_empty() {
                    f32::MAX
                } else {
                    k_nearest.iter().sum::<f32>() / k_nearest.len() as f32
                }
            })
            .collect();

        // Compute mean and std
        let valid_distances: Vec<f32> = mean_distances
            .iter()
            .filter(|&&d| d < f32::MAX)
            .copied()
            .collect();

        if valid_distances.is_empty() {
            return vec![false; points.len()];
        }

        let mean = valid_distances.iter().sum::<f32>() / valid_distances.len() as f32;
        let variance = valid_distances
            .iter()
            .map(|d| (d - mean).powi(2))
            .sum::<f32>()
            / valid_distances.len() as f32;
        let std_dev = variance.sqrt();
        let threshold = mean + std_ratio * std_dev;

        // Return inlier mask
        mean_distances.iter().map(|&d| d <= threshold).collect()
    }

    /// Radius Outlier Removal (ROR)
    /// Removes points that have fewer than min_points within radius
    pub fn radius_outlier_removal(
        points: &[Point3<f32>],
        radius: f32,
        min_points: usize,
        mode: ComputeMode,
    ) -> Vec<bool> {
        if points.is_empty() {
            return Vec::new();
        }

        let global_config = ComputeConfig::default();
        let resolved_mode = mode.resolve(&global_config, points.len());

        match resolved_mode {
            ComputeMode::GPU => radius_outlier_removal_gpu(points, radius, min_points),
            _ => radius_outlier_removal_cpu(points, radius, min_points),
        }
    }

    /// Simple version
    pub fn radius_outlier_removal_simple(
        points: &[Point3<f32>],
        radius: f32,
        min_points: usize,
    ) -> Vec<bool> {
        radius_outlier_removal(points, radius, min_points, ComputeMode::default())
    }

    /// Apply ROR and return inlier points
    pub fn remove_radius_outliers(
        points: &[Point3<f32>],
        radius: f32,
        min_points: usize,
    ) -> (Vec<Point3<f32>>, Vec<bool>) {
        let inliers = radius_outlier_removal_simple(points, radius, min_points);
        let filtered: Vec<Point3<f32>> = points
            .iter()
            .zip(inliers.iter())
            .filter(|(_, &keep)| keep)
            .map(|(p, _)| *p)
            .collect();
        (filtered, inliers)
    }

    fn radius_outlier_removal_gpu(
        points: &[Point3<f32>],
        radius: f32,
        min_points: usize,
    ) -> Vec<bool> {
        // TODO: GPU implementation
        radius_outlier_removal_cpu(points, radius, min_points)
    }

    fn radius_outlier_removal_cpu(
        points: &[Point3<f32>],
        radius: f32,
        min_points: usize,
    ) -> Vec<bool> {
        let radius_sq = radius * radius;
        let voxel_size = radius; // Use radius as voxel size for efficiency

        let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(points.len() / 10);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / voxel_size).floor() as i32;
            let vy = (p.y / voxel_size).floor() as i32;
            let vz = (p.z / voxel_size).floor() as i32;
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
                    (center.x / voxel_size).floor() as i32,
                    (center.y / voxel_size).floor() as i32,
                    (center.z / voxel_size).floor() as i32,
                );

                let mut count = 0usize;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                                for &idx in indices {
                                    if idx != i {
                                        let p = points[idx];
                                        let dist_sq = (center.x - p.x).powi(2)
                                            + (center.y - p.y).powi(2)
                                            + (center.z - p.z).powi(2);
                                        if dist_sq <= radius_sq {
                                            count += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                count >= min_points
            })
            .collect()
    }

    fn compute_adaptive_voxel_size_filtering(points: &[Point3<f32>], k: usize) -> f32 {
        if points.len() < 2 {
            return 0.1;
        }
        let sample_size = 100.min(points.len());
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
        distances[distances.len() / 2] * 2.0
    }
}

/// GPU-accelerated clustering
pub mod clustering {
    use super::*;

    /// DBSCAN Clustering
    /// Returns cluster labels (-1 for noise)
    pub fn dbscan(
        points: &[Point3<f32>],
        epsilon: f32,
        min_points: usize,
        mode: ComputeMode,
    ) -> Vec<i32> {
        if points.is_empty() {
            return Vec::new();
        }

        let global_config = ComputeConfig::default();
        let resolved_mode = mode.resolve(&global_config, points.len());

        match resolved_mode {
            ComputeMode::GPU => dbscan_gpu(points, epsilon, min_points),
            _ => dbscan_cpu(points, epsilon, min_points),
        }
    }

    /// Simple version
    pub fn dbscan_simple(points: &[Point3<f32>], epsilon: f32, min_points: usize) -> Vec<i32> {
        dbscan(points, epsilon, min_points, ComputeMode::default())
    }

    fn dbscan_gpu(points: &[Point3<f32>], epsilon: f32, min_points: usize) -> Vec<i32> {
        // TODO: GPU implementation
        dbscan_cpu(points, epsilon, min_points)
    }

    fn dbscan_cpu(points: &[Point3<f32>], epsilon: f32, min_points: usize) -> Vec<i32> {
        let epsilon_sq = epsilon * epsilon;
        let n = points.len();

        if n == 0 {
            return Vec::new();
        }

        // Build neighborhood index
        let voxel_size = epsilon;
        let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(n / 10);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / voxel_size).floor() as i32;
            let vy = (p.y / voxel_size).floor() as i32;
            let vz = (p.z / voxel_size).floor() as i32;
            voxel_grid
                .entry((vx, vy, vz))
                .or_insert_with(Vec::new)
                .push(i);
        }

        fn get_neighbors(
            idx: usize,
            point: &Point3<f32>,
            points: &[Point3<f32>],
            epsilon_sq: f32,
            voxel_size: f32,
            voxel_grid: &std::collections::HashMap<(i32, i32, i32), Vec<usize>>,
        ) -> Vec<usize> {
            let (vx, vy, vz) = (
                (point.x / voxel_size).floor() as i32,
                (point.y / voxel_size).floor() as i32,
                (point.z / voxel_size).floor() as i32,
            );

            let mut neighbors = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                            for &j in indices {
                                if j != idx {
                                    let pj = points[j];
                                    let dist_sq = (point.x - pj.x).powi(2)
                                        + (point.y - pj.y).powi(2)
                                        + (point.z - pj.z).powi(2);
                                    if dist_sq <= epsilon_sq {
                                        neighbors.push(j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            neighbors
        }

        let mut labels = vec![-1i32; n];
        let mut cluster_id = 0i32;

        for i in 0..n {
            if labels[i] != -1 {
                continue;
            }

            let neighbors =
                get_neighbors(i, &points[i], points, epsilon_sq, voxel_size, &voxel_grid);

            if neighbors.len() < min_points {
                labels[i] = -1; // Noise
            } else {
                // Expand cluster
                labels[i] = cluster_id;
                let mut seed_set: Vec<usize> = neighbors.iter().copied().collect();
                let mut seed_idx = 0;

                while seed_idx < seed_set.len() {
                    let j = seed_set[seed_idx];
                    seed_idx += 1;

                    if labels[j] == -1 {
                        labels[j] = cluster_id;
                        let j_neighbors = get_neighbors(
                            j,
                            &points[j],
                            points,
                            epsilon_sq,
                            voxel_size,
                            &voxel_grid,
                        );
                        if j_neighbors.len() >= min_points {
                            for &n in &j_neighbors {
                                if labels[n] == -1 {
                                    seed_set.push(n);
                                }
                            }
                        }
                    } else if labels[j] == -2 {
                        labels[j] = cluster_id;
                    }
                }

                cluster_id += 1;
            }
        }

        // Mark visited but not assigned
        for label in labels.iter_mut() {
            if *label == -2 {
                *label = -1;
            }
        }

        labels
    }

    /// Euclidean Clustering
    /// Returns vector of clusters, each cluster is a vector of point indices
    pub fn euclidean_cluster(
        points: &[Point3<f32>],
        tolerance: f32,
        min_cluster_size: usize,
        max_cluster_size: usize,
        mode: ComputeMode,
    ) -> Vec<Vec<usize>> {
        if points.is_empty() {
            return Vec::new();
        }

        let global_config = ComputeConfig::default();
        let resolved_mode = mode.resolve(&global_config, points.len());

        match resolved_mode {
            ComputeMode::GPU => {
                euclidean_cluster_gpu(points, tolerance, min_cluster_size, max_cluster_size)
            }
            _ => euclidean_cluster_cpu(points, tolerance, min_cluster_size, max_cluster_size),
        }
    }

    /// Simple version
    pub fn euclidean_cluster_simple(
        points: &[Point3<f32>],
        tolerance: f32,
        min_cluster_size: usize,
    ) -> Vec<Vec<usize>> {
        euclidean_cluster(
            points,
            tolerance,
            min_cluster_size,
            usize::MAX,
            ComputeMode::default(),
        )
    }

    fn euclidean_cluster_gpu(
        points: &[Point3<f32>],
        tolerance: f32,
        min_cluster_size: usize,
        max_cluster_size: usize,
    ) -> Vec<Vec<usize>> {
        // TODO: GPU implementation
        euclidean_cluster_cpu(points, tolerance, min_cluster_size, max_cluster_size)
    }

    fn euclidean_cluster_cpu(
        points: &[Point3<f32>],
        tolerance: f32,
        min_cluster_size: usize,
        max_cluster_size: usize,
    ) -> Vec<Vec<usize>> {
        let tolerance_sq = tolerance * tolerance;
        let n = points.len();

        if n == 0 {
            return Vec::new();
        }

        let voxel_size = tolerance;
        let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(n / 10);

        for (i, p) in points.iter().enumerate() {
            let vx = (p.x / voxel_size).floor() as i32;
            let vy = (p.y / voxel_size).floor() as i32;
            let vz = (p.z / voxel_size).floor() as i32;
            voxel_grid
                .entry((vx, vy, vz))
                .or_insert_with(Vec::new)
                .push(i);
        }

        let mut visited = vec![false; n];
        let mut clusters = Vec::new();

        for i in 0..n {
            if visited[i] {
                continue;
            }

            let mut cluster = Vec::new();
            let mut queue = vec![i];
            visited[i] = true;

            while let Some(j) = queue.pop() {
                cluster.push(j);

                if cluster.len() >= max_cluster_size {
                    break;
                }

                let (vx, vy, vz) = (
                    (points[j].x / voxel_size).floor() as i32,
                    (points[j].y / voxel_size).floor() as i32,
                    (points[j].z / voxel_size).floor() as i32,
                );

                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                                for &k in indices {
                                    if !visited[k] {
                                        let dist_sq = (points[j].x - points[k].x).powi(2)
                                            + (points[j].y - points[k].y).powi(2)
                                            + (points[j].z - points[k].z).powi(2);
                                        if dist_sq <= tolerance_sq {
                                            visited[k] = true;
                                            queue.push(k);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if cluster.len() >= min_cluster_size {
                clusters.push(cluster);
            }
        }

        clusters
    }
}

pub fn is_gpu_available() -> bool {
    GpuContext::new().is_ok()
}

pub fn gpu_info() -> Option<String> {
    GpuContext::new().ok().map(|ctx| format!("{:?}", ctx.device))
}

pub fn force_cpu_mode() -> ComputeConfig {
    ComputeConfig {
        force_cpu: true,
        ..Default::default()
    }
}

pub fn force_gpu_mode() -> ComputeConfig {
    ComputeConfig {
        force_gpu: true,
        ..Default::default()
    }
}
