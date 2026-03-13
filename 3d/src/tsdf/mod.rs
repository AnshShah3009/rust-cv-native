//! TSDF (Truncated Signed Distance Function) Integration
//!
//! Core algorithm for fusing multiple RGBD images into a 3D volume.
//! Based on "KinectFusion: Real-Time Dense Surface Mapping and Tracking" by Newcombe et al.

use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;
use std::collections::HashMap;

use cv_hal::compute::ComputeDevice;
use cv_runtime::orchestrator::RuntimeRunner;

/// Camera intrinsics
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// Voxel block for TSDF volume
#[derive(Debug, Clone)]
pub struct VoxelBlock {
    pub coords: (i32, i32, i32),  // Block coordinates in grid
    pub tsdf: Vec<f32>,           // TSDF values (8x8x8 = 512 voxels per block)
    pub weights: Vec<f32>,        // Integration weights
    pub colors: Vec<Vector3<u8>>, // RGB colors (optional)
}

impl VoxelBlock {
    pub const BLOCK_SIZE: usize = 8;
    pub const VOXELS_PER_BLOCK: usize = 512; // 8^3

    pub fn new(coords: (i32, i32, i32)) -> Self {
        Self {
            coords,
            tsdf: vec![1.0; Self::VOXELS_PER_BLOCK], // Initialize to empty (positive)
            weights: vec![0.0; Self::VOXELS_PER_BLOCK],
            colors: vec![Vector3::new(0, 0, 0); Self::VOXELS_PER_BLOCK],
        }
    }

    /// Get linear index from local voxel coordinates
    pub fn voxel_index(local: (usize, usize, usize)) -> usize {
        local.0 + local.1 * Self::BLOCK_SIZE + local.2 * Self::BLOCK_SIZE * Self::BLOCK_SIZE
    }

    /// Get voxel value at local coordinates
    pub fn get_tsdf(&self, local: (usize, usize, usize)) -> f32 {
        self.tsdf[Self::voxel_index(local)]
    }

    /// Set voxel value
    pub fn set_voxel(
        &mut self,
        local: (usize, usize, usize),
        tsdf: f32,
        weight: f32,
        color: Vector3<u8>,
    ) {
        let idx = Self::voxel_index(local);
        self.tsdf[idx] = tsdf;
        self.weights[idx] = weight;
        self.colors[idx] = color;
    }
}

/// TSDF Volume for RGBD integration
pub struct TSDFVolume {
    voxel_size: f32,
    truncation_distance: f32,
    blocks: HashMap<(i32, i32, i32), VoxelBlock>,
    depth_scale: f32,
    max_weight: f32,
}

impl TSDFVolume {
    pub fn new(voxel_size: f32, truncation_distance: f32) -> Self {
        Self {
            voxel_size,
            truncation_distance,
            blocks: HashMap::new(),
            depth_scale: 1000.0, // Default: depth in mm
            max_weight: 100.0,
        }
    }

    pub fn with_depth_scale(mut self, scale: f32) -> Self {
        self.depth_scale = scale;
        self
    }

    /// Integrate a single RGBD frame using best available runner
    pub fn integrate(
        &mut self,
        depth_image: &[f32],
        color_image: Option<&[Vector3<u8>]>,
        intrinsics: &CameraIntrinsics,
        extrinsics: &Matrix4<f32>,
        width: usize,
        height: usize,
    ) {
        if let Ok(runner) = cv_runtime::best_runner() {
            self.integrate_ctx(
                depth_image,
                color_image,
                intrinsics,
                extrinsics,
                width,
                height,
                &runner,
            );
        }
    }

    /// Integrate a single RGBD frame with explicit context
    #[allow(clippy::too_many_arguments)]
    pub fn integrate_ctx(
        &mut self,
        depth_image: &[f32],
        color_image: Option<&[Vector3<u8>]>,
        intrinsics: &CameraIntrinsics,
        extrinsics: &Matrix4<f32>,
        width: usize,
        height: usize,
        group: &RuntimeRunner,
    ) {
        // GPU Path
        if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
            // For GPU integration, we currently require a fixed-size dense volume.
            // In a real sparse TSDF system, we'd use a Voxel Hashing approach on GPU.
            // For now, we'll implement a dense fallback for GPU if the area is limited.
            let vol_size = 256; // 256^3 dense volume
            let shape = cv_core::TensorShape::new(2, vol_size, vol_size); // (TSDF, Weight) packed in channels

            use cv_core::DataType;
            use cv_hal::storage::GpuStorage;
            use std::marker::PhantomData;

            let mut gpu_vol: cv_core::Tensor<f32, GpuStorage<f32>> = cv_core::Tensor {
                storage: GpuStorage::new_with_ctx(gpu, vol_size * vol_size * vol_size * 2, 0.0)
                    .map_err(|_| ())
                    .unwrap(),
                shape,
                dtype: DataType::F32,
                _phantom: PhantomData,
            };

            let intrinsics_arr = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy];
            let pose_arr = extrinsics.as_slice();
            let mut pose_mat = [[0.0f32; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    pose_mat[i][j] = pose_arr[j * 4 + i];
                }
            }

            // Create depth tensor on GPU
            let depth_tensor: cv_core::Tensor<f32, GpuStorage<f32>> = cv_core::Tensor {
                storage: GpuStorage::from_slice_ctx(gpu, depth_image)
                    .map_err(|_| ())
                    .unwrap(),
                shape: cv_core::TensorShape::new(1, height, width),
                dtype: DataType::F32,
                _phantom: PhantomData,
            };

            if cv_hal::gpu_kernels::tsdf::integrate(
                gpu,
                &depth_tensor,
                &pose_mat,
                &intrinsics_arr,
                &mut gpu_vol,
                self.voxel_size,
                self.truncation_distance,
            )
            .is_ok()
            {
                // In a full implementation, we'd read back and merge into sparse blocks.
                // For now, we just demonstrate the dispatch.
                return;
            }
        }

        // CPU Fallback (Rayon)
        // Transform camera origin to world space
        let camera_origin = extrinsics.transform_point(&Point3::origin());

        let updates: Vec<_> = group.run(|| {
            (0..height)
                .into_par_iter()
                .flat_map(|v| {
                    let mut local_updates = Vec::new();

                    for u in 0..width {
                        let idx = v * width + u;
                        let depth = depth_image[idx];

                        if depth <= 0.0 || depth > 10.0 {
                            continue;
                        }

                        // Backproject to 3D in camera space
                        let x = (u as f32 - intrinsics.cx) * depth / intrinsics.fx;
                        let y = (v as f32 - intrinsics.cy) * depth / intrinsics.fy;
                        let z = depth;
                        let point_camera = Point3::new(x, y, z);

                        // Transform to world space
                        let point_world = extrinsics.transform_point(&point_camera);

                        // Get color
                        let color = color_image
                            .map(|c| c[idx])
                            .unwrap_or(Vector3::new(128, 128, 128));

                        // Find affected voxels along ray
                        let ray_dir = (point_world - camera_origin).normalize();
                        let start =
                            camera_origin + ray_dir * (depth - self.truncation_distance).max(0.01);
                        let end = camera_origin + ray_dir * (depth + self.truncation_distance);

                        // March along ray and update voxels
                        let steps = ((end - start).norm() / self.voxel_size).ceil() as usize;
                        for i in 0..=steps {
                            let t = i as f32 / steps.max(1) as f32;
                            let voxel_pos = start + (end - start) * t;

                            // Calculate TSDF value
                            let dist = (voxel_pos - point_world).norm();
                            let sdf = if voxel_pos.coords.dot(&ray_dir)
                                > point_world.coords.dot(&ray_dir)
                            {
                                dist // Behind surface
                            } else {
                                -dist // In front of surface
                            };

                            let tsdf = sdf
                                .clamp(-self.truncation_distance, self.truncation_distance)
                                / self.truncation_distance;

                            local_updates.push((voxel_pos, tsdf, color));
                        }
                    }

                    local_updates
                })
                .collect()
        });

        // Apply updates
        for (pos, tsdf, color) in updates {
            self.update_voxel(pos, tsdf, color);
        }
    }

    /// Update a single voxel with new TSDF value
    fn update_voxel(&mut self, world_pos: Point3<f32>, tsdf: f32, color: Vector3<u8>) {
        // Convert to voxel coordinates
        let voxel_x = (world_pos.x / self.voxel_size).floor() as i32;
        let voxel_y = (world_pos.y / self.voxel_size).floor() as i32;
        let voxel_z = (world_pos.z / self.voxel_size).floor() as i32;

        // Calculate block coordinates
        let block_x = voxel_x.div_euclid(VoxelBlock::BLOCK_SIZE as i32);
        let block_y = voxel_y.div_euclid(VoxelBlock::BLOCK_SIZE as i32);
        let block_z = voxel_z.div_euclid(VoxelBlock::BLOCK_SIZE as i32);

        let block_coords = (block_x, block_y, block_z);

        // Calculate local voxel coordinates within block
        let local_x = voxel_x.rem_euclid(VoxelBlock::BLOCK_SIZE as i32) as usize;
        let local_y = voxel_y.rem_euclid(VoxelBlock::BLOCK_SIZE as i32) as usize;
        let local_z = voxel_z.rem_euclid(VoxelBlock::BLOCK_SIZE as i32) as usize;
        let local = (local_x, local_y, local_z);

        // Get or create block
        let block = self
            .blocks
            .entry(block_coords)
            .or_insert_with(|| VoxelBlock::new(block_coords));

        // Integrate with running average
        let idx = VoxelBlock::voxel_index(local);
        let old_tsdf = block.tsdf[idx];
        let old_weight = block.weights[idx];

        let new_weight = (old_weight + 1.0).min(self.max_weight);
        let new_tsdf = (old_tsdf * old_weight + tsdf) / new_weight;

        block.tsdf[idx] = new_tsdf;
        block.weights[idx] = new_weight;
        block.colors[idx] = color;
    }

    /// Extract surface mesh using Marching Cubes
    pub fn extract_mesh(&self) -> Vec<Triangle> {
        let mut triangles = Vec::new();

        for (block_coords, block) in &self.blocks {
            for z in 0..VoxelBlock::BLOCK_SIZE - 1 {
                for y in 0..VoxelBlock::BLOCK_SIZE - 1 {
                    for x in 0..VoxelBlock::BLOCK_SIZE - 1 {
                        // Get corner values
                        let corners = [
                            block.get_tsdf((x, y, z)),
                            block.get_tsdf((x + 1, y, z)),
                            block.get_tsdf((x + 1, y + 1, z)),
                            block.get_tsdf((x, y + 1, z)),
                            block.get_tsdf((x, y, z + 1)),
                            block.get_tsdf((x + 1, y, z + 1)),
                            block.get_tsdf((x + 1, y + 1, z + 1)),
                            block.get_tsdf((x, y + 1, z + 1)),
                        ];

                        // Marching cubes on this cell
                        let cell_tris =
                            marching_cubes_cell(block_coords, (x, y, z), &corners, self.voxel_size);

                        triangles.extend(cell_tris);
                    }
                }
            }
        }

        triangles
    }

    /// Extract point cloud from surface (zero-crossings)
    pub fn extract_point_cloud(&self) -> Vec<(Point3<f32>, Vector3<f32>, Vector3<u8>)> {
        let mut points = Vec::new();

        for (block_coords, block) in &self.blocks {
            for z in 0..VoxelBlock::BLOCK_SIZE {
                for y in 0..VoxelBlock::BLOCK_SIZE {
                    for x in 0..VoxelBlock::BLOCK_SIZE {
                        let local = (x, y, z);
                        let tsdf = block.get_tsdf(local);
                        let weight = block.weights[VoxelBlock::voxel_index(local)];

                        // Check if this is a surface voxel (near zero crossing)
                        if weight > 0.0 && tsdf.abs() < 0.1 {
                            let world_pos = self.voxel_to_world(*block_coords, local);
                            let normal = self.estimate_normal(*block_coords, local);
                            let color = block.colors[VoxelBlock::voxel_index(local)];

                            points.push((world_pos, normal, color));
                        }
                    }
                }
            }
        }

        points
    }

    fn voxel_to_world(
        &self,
        block_coords: (i32, i32, i32),
        local: (usize, usize, usize),
    ) -> Point3<f32> {
        let vx = block_coords.0 * VoxelBlock::BLOCK_SIZE as i32 + local.0 as i32;
        let vy = block_coords.1 * VoxelBlock::BLOCK_SIZE as i32 + local.1 as i32;
        let vz = block_coords.2 * VoxelBlock::BLOCK_SIZE as i32 + local.2 as i32;

        Point3::new(
            vx as f32 * self.voxel_size,
            vy as f32 * self.voxel_size,
            vz as f32 * self.voxel_size,
        )
    }

    fn estimate_normal(
        &self,
        block_coords: (i32, i32, i32),
        local: (usize, usize, usize),
    ) -> Vector3<f32> {
        // Simple central difference gradient
        let x = local.0;
        let y = local.1;
        let z = local.2;

        let dx = self.get_tsdf_safe(block_coords, (x + 1, y, z))
            - self.get_tsdf_safe(block_coords, (x.saturating_sub(1), y, z));
        let dy = self.get_tsdf_safe(block_coords, (x, y + 1, z))
            - self.get_tsdf_safe(block_coords, (x, y.saturating_sub(1), z));
        let dz = self.get_tsdf_safe(block_coords, (x, y, z + 1))
            - self.get_tsdf_safe(block_coords, (x, y, z.saturating_sub(1)));

        Vector3::new(dx, dy, dz).normalize()
    }

    fn get_tsdf_safe(&self, block_coords: (i32, i32, i32), local: (usize, usize, usize)) -> f32 {
        if let Some(block) = self.blocks.get(&block_coords) {
            let x = local.0.min(VoxelBlock::BLOCK_SIZE - 1);
            let y = local.1.min(VoxelBlock::BLOCK_SIZE - 1);
            let z = local.2.min(VoxelBlock::BLOCK_SIZE - 1);
            block.get_tsdf((x, y, z))
        } else {
            1.0 // Empty space
        }
    }
}

/// Triangle for mesh extraction
#[derive(Debug, Clone)]
pub struct Triangle {
    pub vertices: [Point3<f32>; 3],
    pub normals: [Vector3<f32>; 3],
    pub colors: [Vector3<u8>; 3],
}

/// Marching cubes on a single cell
#[allow(clippy::needless_range_loop)]
fn marching_cubes_cell(
    _block_coords: &(i32, i32, i32),
    _local: (usize, usize, usize),
    corners: &[f32; 8],
    _voxel_size: f32,
) -> Vec<Triangle> {
    let triangles = Vec::new();

    // Determine which edges are intersected
    let mut cube_index = 0;
    for i in 0..8 {
        if corners[i] < 0.0 {
            cube_index |= 1 << i;
        }
    }

    // No surface in this cell
    if cube_index == 0 || cube_index == 255 {
        return triangles;
    }

    // Calculate edge vertices (simplified - full implementation would interpolate)
    // For now, return empty - full marching cubes requires ~4k lines

    triangles
}
