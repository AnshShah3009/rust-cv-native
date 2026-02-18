//! GPU Compute Kernels
//!
//! This module provides GPU-accelerated implementations using WebGPU compute shaders.
//! All shaders are written in WGSL (WebGPU Shading Language).

use wgpu::{
    CommandEncoder, Device, Queue
};
use nalgebra::Vector3;
use std::sync::Arc;

/// GPU Compute Context (Deprecated: Use GpuContext instead)
#[deprecated(since = "0.1.1", note = "Use cv_hal::gpu::GpuContext directly instead")]
pub struct GpuCompute {
    device: Arc<Device>,
    queue: Arc<Queue>,
}

#[allow(deprecated)]
impl GpuCompute {
    /// Create a new GpuCompute from Device and Queue (Deprecated)
    #[deprecated(since = "0.1.1", note = "Use cv_hal::gpu::GpuContext::new() instead")]
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self { device, queue }
    }

    /// Get reference to device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get reference to queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Submit command encoder
    pub fn submit(&self, encoder: CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}

pub mod convolve;
pub mod threshold;
pub mod sobel;
pub mod morphology;
pub mod nms;
pub mod pointcloud;
pub mod color;
pub mod resize;
pub mod bilateral;
pub mod fast;
pub mod matching;
pub mod sift;
pub mod icp;
pub mod subtract;
pub mod akaze;
pub mod sparse;

/// Shader compilation utilities
pub mod shaders {
    use std::collections::HashMap;

    /// Collection of WGSL shaders for 3D processing
    pub struct ShaderLibrary {
        shaders: HashMap<&'static str, &'static str>,
    }

    impl ShaderLibrary {
        pub fn new() -> Self {
            let mut shaders = HashMap::new();

            // Point cloud transformations
            shaders.insert(
                "pointcloud_transform",
                include_str!("pointcloud_transform.wgsl"),
            );

            // Point cloud normals
            shaders.insert(
                "pointcloud_normals",
                include_str!("pointcloud_normals.wgsl"),
            );

            // Point cloud normals - fast GPU
            shaders.insert(
                "pointcloud_normals_fast_gpu",
                include_str!("pointcloud_normals_fast_gpu.wgsl"),
            );

            // Point cloud normals - Morton code optimized
            shaders.insert(
                "pointcloud_normals_morton",
                include_str!("pointcloud_normals_morton.wgsl"),
            );

            // TSDF integration
            shaders.insert(
                "tsdf_integrate",
                include_str!("tsdf_integrate.wgsl"),
            );

            // TSDF raycasting (surface extraction)
            shaders.insert(
                "tsdf_raycast",
                include_str!("tsdf_raycast.wgsl"),
            );

            // ICP correspondence finding
            shaders.insert(
                "icp_correspondences",
                include_str!("icp_correspondences.wgsl"),
            );

            // ICP accumulation (J^T J, J^T r)
            shaders.insert(
                "icp_accumulate",
                include_str!("../../shaders/icp_accumulate.wgsl"),
            );

            // ICP reduction (sum of errors)
            shaders.insert(
                "icp_reduction",
                include_str!("icp_reduction.wgsl"),
            );

            // KDTree build (parallel construction)
            shaders.insert(
                "kdtree_build",
                include_str!("kdtree_build.wgsl"),
            );

            // KDTree nearest neighbor search
            shaders.insert(
                "kdtree_search",
                include_str!("kdtree_search.wgsl"),
            );

            // Voxel grid operations
            shaders.insert(
                "voxel_grid_downsample",
                include_str!("voxel_grid_downsample.wgsl"),
            );

            // Mesh vertex operations
            shaders.insert(
                "mesh_laplacian",
                include_str!("mesh_laplacian.wgsl"),
            );

            // Ray casting
            shaders.insert(
                "ray_mesh_intersection",
                include_str!("ray_mesh_intersection.wgsl"),
            );

            // Distance field computation
            shaders.insert(
                "distance_field",
                include_str!("distance_field.wgsl"),
            );

            // RGBD odometry
            shaders.insert(
                "rgbd_odometry",
                include_str!("rgbd_odometry.wgsl"),
            );

            // Parallel reduction (sum, min, max)
            shaders.insert(
                "parallel_reduce",
                include_str!("parallel_reduce.wgsl"),
            );

            // Gaussian blur (separable)
            shaders.insert(
                "gaussian_blur_separable",
                include_str!("../../shaders/gaussian_blur_separable.wgsl"),
            );

            // Image subtraction
            shaders.insert(
                "subtract",
                include_str!("../../shaders/subtract.wgsl"),
            );

            // Matrix multiplication
            shaders.insert(
                "matrix_multiply",
                include_str!("matrix_multiply.wgsl"),
            );

            // Prefix sum (scan)
            shaders.insert(
                "prefix_sum",
                include_str!("prefix_sum.wgsl"),
            );

            // Radix sort (for KDTree, Octree)
            shaders.insert(
                "radix_sort",
                include_str!("radix_sort.wgsl"),
            );

            Self { shaders }
        }

        pub fn get(&self, name: &str) -> Option<&&str> {
            self.shaders.get(name)
        }
    }

    impl Default for ShaderLibrary {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// GPU Buffer utilities
pub mod buffer_utils {
    use wgpu::{Buffer, Device, BufferDescriptor, BufferUsages, MapMode};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::collections::HashMap;

    /// A bucketed pool for reusing GPU buffers
    pub struct GpuBufferPool {
        // Buckets: usage -> (size_bucket -> Vec<Buffer>)
        buckets: Mutex<HashMap<BufferUsages, HashMap<u64, Vec<Buffer>>>>,
    }

    impl GpuBufferPool {
        pub fn new() -> Self {
            Self {
                buckets: Mutex::new(HashMap::new()),
            }
        }

        pub(crate) fn get_size_bucket(size: u64) -> u64 {
            // Round up to nearest power of 2 or multiple of 1MB
            if size <= 1024 * 1024 {
                size.next_power_of_two().max(256)
            } else {
                ((size + 1024 * 1024 - 1) / (1024 * 1024)) * 1024 * 1024
            }
        }

        pub fn get(&self, device: &Device, size: u64, usage: BufferUsages) -> Buffer {
            let bucket_size = Self::get_size_bucket(size);
            let mut buckets = self.buckets.lock().unwrap();
            
            let usage_map = buckets.entry(usage).or_insert_with(HashMap::new);
            if let Some(pool) = usage_map.get_mut(&bucket_size) {
                if let Some(buffer) = pool.pop() {
                    return buffer;
                }
            }

            device.create_buffer(&BufferDescriptor {
                label: Some("Pooled Compute Buffer"),
                size: bucket_size,
                usage,
                mapped_at_creation: false,
            })
        }

        pub fn return_buffer(&self, buffer: Buffer, usage: BufferUsages) {
            let size = buffer.size();
            let mut buckets = self.buckets.lock().unwrap();
            let usage_map = buckets.entry(usage).or_insert_with(HashMap::new);
            let pool = usage_map.entry(size).or_insert_with(Vec::new);
            if pool.len() < 8 {
                pool.push(buffer);
            }
        }
    }

    static GLOBAL_GPU_POOL: OnceLock<GpuBufferPool> = OnceLock::new();

    pub fn global_pool() -> &'static GpuBufferPool {
        GLOBAL_GPU_POOL.get_or_init(GpuBufferPool::new)
    }

    /// Create a GPU buffer from data
    pub fn create_buffer<T: bytemuck::Pod>(
        device: &Device,
        data: &[T],
        usage: BufferUsages,
    ) -> Buffer {
        use wgpu::util::DeviceExt;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compute Buffer"),
            contents: bytemuck::cast_slice(data),
            usage,
        })
    }

    /// Create an uninitialized GPU buffer
    pub fn create_buffer_uninit(
        device: &Device,
        size: usize,
        usage: BufferUsages,
    ) -> Buffer {
        device.create_buffer(&BufferDescriptor {
            label: Some("Compute Buffer (uninit)"),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Download data from GPU buffer
    pub async fn read_buffer<T: bytemuck::Pod + std::fmt::Debug>(
        device: Arc<Device>,
        queue: &wgpu::Queue,
        buffer: &Buffer,
        offset: u64,
        size: usize,
    ) -> crate::Result<Vec<T>> {
        let aligned_size = (size + 3) & !3;
        
        // Create a staging buffer
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: aligned_size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from source to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(
            buffer,
            offset,
            &staging_buffer,
            0,
            aligned_size as u64,
        );

        let index = queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = tokio::sync::oneshot::channel();
        
        let slice = staging_buffer.slice(..);
        slice.map_async(MapMode::Read, move |res| {
            tx.send(res).ok();
        });

        let _ = device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None });

        rx.await
            .map_err(|_| crate::Error::DeviceError("Readback channel closed".to_string()))?
            .map_err(|e| crate::Error::DeviceError(format!("Buffer mapping failed: {}", e)))?;

        let data = slice.get_mapped_range();
        // Cast to aligned slice then truncate to original size if needed
        let result_full: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let num_elements = size / std::mem::size_of::<T>();
        Ok(result_full[..num_elements].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::buffer_utils::*;
    use wgpu::BufferUsages;

    #[test]
    fn test_buffer_pool_buckets() {
        let pool = GpuBufferPool::new();
        
        // Test size bucket logic
        assert_eq!(GpuBufferPool::get_size_bucket(100), 256);
        assert_eq!(GpuBufferPool::get_size_bucket(1024), 1024);
        assert_eq!(GpuBufferPool::get_size_bucket(1024 * 1024 + 1), 2 * 1024 * 1024);
    }

    #[test]
    fn test_gpu_buffer_pool_reuse() {
        // We can't easily test the 'get' without a real Device, 
        // but we can test the internal structure if we exposed it, 
        // or just ensure 'return_buffer' doesn't crash.
        let pool = GpuBufferPool::new();
        // Since we can't create a real wgpu::Buffer without a device easily in a unit test,
        // we'll focus on the logic we've implemented.
    }
}

/// Workgroup size configuration
pub const WORKGROUP_SIZE_1D: u32 = 256;
pub const WORKGROUP_SIZE_2D: u32 = 16;
pub const WORKGROUP_SIZE_3D: u32 = 8;

/// Compute dispatch helpers
pub fn dispatch_size_1d(count: u32) -> u32 {
    (count + WORKGROUP_SIZE_1D - 1) / WORKGROUP_SIZE_1D
}

pub fn dispatch_size_2d(width: u32, height: u32) -> (u32, u32) {
    (
        (width + WORKGROUP_SIZE_2D - 1) / WORKGROUP_SIZE_2D,
        (height + WORKGROUP_SIZE_2D - 1) / WORKGROUP_SIZE_2D,
    )
}

pub fn dispatch_size_3d(width: u32, height: u32, depth: u32) -> (u32, u32, u32) {
    (
        (width + WORKGROUP_SIZE_3D - 1) / WORKGROUP_SIZE_3D,
        (height + WORKGROUP_SIZE_3D - 1) / WORKGROUP_SIZE_3D,
        (depth + WORKGROUP_SIZE_3D - 1) / WORKGROUP_SIZE_3D,
    )
}

/// GPU-accelerated point cloud operations
pub mod pointcloud_gpu {
    use super::*;
    use nalgebra::{Matrix4, Vector3};

    /// Transform point cloud on GPU
    pub fn transform_points(
        _gpu: &GpuCompute,
        _points: &[Vector3<f32>],
        _transform: &Matrix4<f32>,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        // Implementation: dispatch compute shader
        Err(crate::Error::not_supported("GPU point cloud transform"))
    }

    /// Compute normals using fast GPU implementation
    pub fn compute_normals_fast_gpu(
        gpu: &GpuCompute,
        points: &[Vector3<f32>],
        k_neighbors: u32,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        // Use Morton code optimized version
        compute_normals_morton_gpu(gpu, points, k_neighbors)
    }

    /// Compute normals from points on GPU
    pub fn compute_normals(
        gpu: &GpuCompute,
        points: &[Vector3<f32>],
        neighbor_indices: &[u32],
        k_neighbors: u32,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        use wgpu::BufferUsages;
        use crate::gpu_kernels::buffer_utils::{create_buffer, create_buffer_uninit, read_buffer};

        let device = gpu.device();
        let queue = gpu.queue();
        let num_points = points.len() as u32;

        // 1. Create buffers
        // Convert Vector3 to [f32; 4] for 16-byte alignment (vec4 in shader)
        let points_data: Vec<[f32; 4]> = points.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        
        let points_buf = create_buffer(device, &points_data, BufferUsages::STORAGE);
        let indices_buf = create_buffer(device, neighbor_indices, BufferUsages::STORAGE);
        let normals_buf = create_buffer_uninit(device, points.len() * 16, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let num_points_buf = create_buffer(device, &[num_points], BufferUsages::UNIFORM);
        let k_buf = create_buffer(device, &[k_neighbors], BufferUsages::UNIFORM);

        // 2. Create pipeline
        let shader_source = include_str!("pointcloud_normals.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point Cloud Normals Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Normals Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // 3. Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Normals Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: points_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: normals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: indices_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: num_points_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: k_buf.as_entire_binding() },
            ],
        });

        // 4. Dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let x = dispatch_size_1d(num_points);
            pass.dispatch_workgroups(x, 1, 1);
        }

        gpu.submit(encoder);

        // 5. Read back
        let result_data: Vec<[f32; 4]> = pollster::block_on(read_buffer(gpu.device.clone(), queue, &normals_buf, 0, points.len() * 16))?;
        let result: Vec<Vector3<f32>> = result_data.into_iter().map(|v| Vector3::new(v[0], v[1], v[2])).collect();
        Ok(result)
    }

    /// Compute normals using GPU with Morton code spatial indexing
    /// This is the optimized GPU version - O(n log n) via sorting
    pub fn compute_normals_morton_gpu(
        gpu: &GpuCompute,
        points: &[Vector3<f32>],
        k_neighbors: u32,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        use wgpu::BufferUsages;
        use crate::gpu_kernels::buffer_utils::{create_buffer, create_buffer_uninit, read_buffer};

        let device = gpu.device();
        let queue = gpu.queue();
        let num_points = points.len() as u32;
        let k = k_neighbors.min(32).max(3);

        // Step 1: Compute Morton codes on CPU
        let (min_bound, max_bound) = points.iter().fold(
            (Vector3::new(f32::MAX, f32::MAX, f32::MAX), Vector3::new(f32::MIN, f32::MIN, f32::MIN)),
            |(min, max), p| (min.inf(p), max.sup(p))
        );
        let span = (max_bound - min_bound).max();
        let grid_size = span / 1024.0;

        let mut morton_data: Vec<(u32, usize)> = points.iter().enumerate().map(|(i, p)| {
            let x = ((p.x - min_bound.x) / grid_size).max(0.0).min(1023.0) as u32;
            let y = ((p.y - min_bound.y) / grid_size).max(0.0).min(1023.0) as u32;
            let z = ((p.z - min_bound.z) / grid_size).max(0.0).min(1023.0) as u32;
            (morton_encode(x, y, z), i)
        }).collect();

        // Step 2: Sort by Morton code
        morton_data.sort_by_key(|&(code, _)| code);

        let morton_codes: Vec<u32> = morton_data.iter().map(|&(code, _)| code).collect();
        let sorted_indices: Vec<u32> = morton_data.iter().map(|&(_, i)| i as u32).collect();

        // Step 3: Create GPU buffers
        let points_data: Vec<[f32; 4]> = points.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        
        let points_buf = create_buffer(device, &points_data, BufferUsages::STORAGE);
        let normals_buf = create_buffer_uninit(device, points.len() * 16, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let sorted_buf = create_buffer(device, &sorted_indices, BufferUsages::STORAGE);
        let morton_buf = create_buffer(device, &morton_codes, BufferUsages::STORAGE);
        
        let params_data: [f32; 4] = [num_points as f32, k as f32, grid_size, 0.0];
        let params_buf = create_buffer(device, &params_data, BufferUsages::UNIFORM);

        // Step 4: Create pipeline
        let shader_source = include_str!("pointcloud_normals_morton.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point Cloud Normals Morton GPU"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout explicitly
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Normals Morton Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Normals Morton Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Normals Morton GPU Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Normals Morton Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: points_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: normals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: morton_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        // Step 5: Dispatch
        let workgroups = (num_points + 255) / 256;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Step 6: Read back
        let result_data: Vec<[f32; 4]> = pollster::block_on(read_buffer(gpu.device.clone(), queue, &normals_buf, 0, points.len() * 16))?;
        let result: Vec<Vector3<f32>> = result_data.into_iter().map(|v| Vector3::new(v[0], v[1], v[2])).collect();
        Ok(result)
    }
}

/// Encode 3D coordinates to Morton code
fn morton_encode(x: u32, y: u32, z: u32) -> u32 {
    let mut mx = x & 0x000003FFu32;
    let mut my = y & 0x000003FFu32;
    let mut mz = z & 0x000003FFu32;
    
    mx = (mx | (mx << 16)) & 0x030000FF;
    mx = (mx | (mx << 8)) & 0x0300F00F;
    mx = (mx | (mx << 4)) & 0x030C30C3;
    mx = (mx | (mx << 2)) & 0x09249249;
    
    my = (my | (my << 16)) & 0x030000FF;
    my = (my | (my << 8)) & 0x0300F00F;
    my = (my | (my << 4)) & 0x030C30C3;
    my = (my | (my << 2)) & 0x09249249;
    
    mz = (mz | (mz << 16)) & 0x030000FF;
    mz = (mz | (mz << 8)) & 0x0300F00F;
    mz = (mz | (mz << 4)) & 0x030C30C3;
    mz = (mz | (mz << 2)) & 0x09249249;
    
    mx | (my << 1) | (mz << 2)
}

/// GPU-accelerated TSDF operations
pub mod tsdf_gpu {
    use super::*;

    /// Integrate depth frame into TSDF volume on GPU
    pub fn integrate_depth(
        _gpu: &GpuCompute,
        _depth_image: &[f32],
        _color_image: Option<&[u32]>, // Packed RGB
        _camera_pose: &[f32; 16],
        _intrinsics: &[f32; 4], // fx, fy, cx, cy
        _width: u32,
        _height: u32,
        _tsdf_volume: &mut [f32],
        _weights: &mut [f32],
        _colors: &mut [u32],
        _voxel_size: f32,
        _truncation: f32,
    ) -> crate::Result<()> {
        Err(crate::Error::not_supported("GPU TSDF integration"))
    }

    /// Raycast TSDF volume to generate surface on GPU
    pub fn raycast_volume(
        _gpu: &GpuCompute,
        _tsdf_volume: &[f32],
        _camera_pose: &[f32; 16],
        _intrinsics: &[f32; 4],
        _width: u32,
        _height: u32,
        _voxel_size: f32,
    ) -> crate::Result<(Vec<Vector3<f32>>, Vec<Vector3<f32>>)> {
        // Returns (vertices, normals)
        Err(crate::Error::not_supported("GPU TSDF raycasting"))
    }

    /// Extract surface points from TSDF on GPU
    pub fn extract_surface(
        _gpu: &GpuCompute,
        _tsdf_volume: &[f32],
        _threshold: f32,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        Err(crate::Error::not_supported("GPU surface extraction"))
    }
}

/// GPU-accelerated ICP operations
pub mod icp_gpu {
    use super::*;
    use nalgebra::{Matrix4, Vector3};

    /// Find correspondences on GPU
    pub fn find_correspondences(
        _gpu: &GpuCompute,
        _source: &[Vector3<f32>],
        _target: &[Vector3<f32>],
        _transform: &Matrix4<f32>,
        _max_distance: f32,
    ) -> crate::Result<Vec<(u32, u32, f32)>> { // (src_idx, tgt_idx, distance)
        Err(crate::Error::not_supported("GPU correspondence finding"))
    }

    /// Compute ICP residuals on GPU
    pub fn compute_residuals(
        _gpu: &GpuCompute,
        _source: &[Vector3<f32>],
        _target: &[Vector3<f32>],
        _target_normals: &[Vector3<f32>],
        _correspondences: &[(u32, u32)],
        _transform: &Matrix4<f32>,
    ) -> crate::Result<(f32, f32)> { // (sum_squared_error, inlier_count)
        Err(crate::Error::not_supported("GPU residual computation"))
    }

    /// One ICP iteration on GPU
    pub fn icp_iteration(
        _gpu: &GpuCompute,
        _source: &[Vector3<f32>],
        _target: &[Vector3<f32>],
        _target_normals: &[Vector3<f32>],
        _transform: &Matrix4<f32>,
        _max_distance: f32,
    ) -> crate::Result<Matrix4<f32>> {
        Err(crate::Error::not_supported("GPU ICP iteration"))
    }
}

/// GPU-accelerated spatial queries
pub mod spatial_gpu {
    use super::*;
    use nalgebra::Vector3;

    /// Build KDTree on GPU (parallel construction)
    pub fn build_kdtree(
        _gpu: &GpuCompute,
        _points: &[Vector3<f32>],
    ) -> crate::Result<GpuKDTree> {
        Err(crate::Error::not_supported("GPU KDTree construction"))
    }

    /// Batch nearest neighbor search on GPU
    pub fn batch_nearest_neighbors(
        _gpu: &GpuCompute,
        _kdtree: &GpuKDTree,
        _queries: &[Vector3<f32>],
        _k: u32,
    ) -> crate::Result<Vec<Vec<(u32, f32)>>> { // (point_idx, distance)
        Err(crate::Error::not_supported("GPU batch NN search"))
    }

    /// Batch radius search on GPU
    pub fn batch_radius_search(
        _gpu: &GpuCompute,
        _kdtree: &GpuKDTree,
        _queries: &[Vector3<f32>],
        _radius: f32,
    ) -> crate::Result<Vec<Vec<(u32, f32)>>> {
        Err(crate::Error::not_supported("GPU batch radius search"))
    }

    /// Build VoxelGrid on GPU
    pub fn build_voxel_grid(
        _gpu: &GpuCompute,
        _points: &[Vector3<f32>],
        _voxel_size: f32,
    ) -> crate::Result<GpuVoxelGrid> {
        Err(crate::Error::not_supported("GPU VoxelGrid construction"))
    }

    /// GPU KDTree handle
    pub struct GpuKDTree {
        nodes_buffer: wgpu::Buffer,
        points_buffer: wgpu::Buffer,
        num_points: u32,
    }

    /// GPU VoxelGrid handle
    pub struct GpuVoxelGrid {
        voxel_buffer: wgpu::Buffer,
        occupied_buffer: wgpu::Buffer,
        voxel_size: f32,
    }
}

/// GPU-accelerated mesh operations
pub mod mesh_gpu {
    use super::*;
    use nalgebra::{Vector3, Point3};

    /// Compute vertex normals on GPU
    pub fn compute_vertex_normals(
        _gpu: &GpuCompute,
        _vertices: &[Point3<f32>],
        _faces: &[[u32; 3]],
    ) -> crate::Result<Vec<Vector3<f32>>> {
        Err(crate::Error::not_supported("GPU vertex normals"))
    }

    /// Laplacian smoothing on GPU
    pub fn laplacian_smooth(
        _gpu: &GpuCompute,
        _vertices: &mut [Point3<f32>],
        _faces: &[[u32; 3]],
        _iterations: u32,
        _lambda: f32,
    ) -> crate::Result<()> {
        Err(crate::Error::not_supported("GPU Laplacian smoothing"))
    }

    /// Simplify mesh on GPU
    pub fn simplify_mesh(
        _gpu: &GpuCompute,
        _vertices: &[Point3<f32>],
        _faces: &[[u32; 3]],
        _target_ratio: f32,
    ) -> crate::Result<(Vec<Point3<f32>>, Vec<[u32; 3]>)> {
        Err(crate::Error::not_supported("GPU mesh simplification"))
    }

    /// Compute mesh bounds on GPU
    pub fn compute_bounds(
        _gpu: &GpuCompute,
        _vertices: &[Point3<f32>],
    ) -> crate::Result<(Point3<f32>, Point3<f32>)> {
        Err(crate::Error::not_supported("GPU bounds computation"))
    }
}

/// GPU-accelerated ray casting
pub mod raycasting_gpu {
    use super::*;
    use nalgebra::{Point3, Vector3};

    /// Batch ray-mesh intersection on GPU
    pub fn cast_rays(
        _gpu: &GpuCompute,
        _rays: &[(Point3<f32>, Vector3<f32>)], // (origin, direction)
        _vertices: &[Point3<f32>],
        _faces: &[[u32; 3]],
    ) -> crate::Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>> { // (distance, hit_point, normal)
        Err(crate::Error::not_supported("GPU ray casting"))
    }

    /// Compute distance field on GPU
    pub fn compute_distance_field(
        _gpu: &GpuCompute,
        _mesh_vertices: &[Point3<f32>],
        _mesh_faces: &[[u32; 3]],
        _grid_origin: Point3<f32>,
        _grid_size: (u32, u32, u32),
        _voxel_size: f32,
    ) -> crate::Result<Vec<f32>> {
        Err(crate::Error::not_supported("GPU distance field"))
    }
}

/// GPU-accelerated RGBD odometry
pub mod odometry_gpu {
    use super::*;

    /// Compute RGBD odometry on GPU
    pub fn compute_odometry(
        _gpu: &GpuCompute,
        _source_depth: &[f32],
        _target_depth: &[f32],
        _source_color: Option<&[u32]>,
        _target_color: Option<&[u32]>,
        _intrinsics: &[f32; 4],
        _width: u32,
        _height: u32,
        _init_transform: &[f32; 16],
    ) -> crate::Result<([f32; 16], f32, f32)> { // (transform, fitness, rmse)
        Err(crate::Error::not_supported("GPU RGBD odometry"))
    }

    /// Compute vertex map from depth on GPU
    pub fn compute_vertex_map(
        _gpu: &GpuCompute,
        _depth: &[f32],
        _intrinsics: &[f32; 4],
        _width: u32,
        _height: u32,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        Err(crate::Error::not_supported("GPU vertex map computation"))
    }

    /// Compute normal map from vertex map on GPU
    pub fn compute_normal_map(
        _gpu: &GpuCompute,
        _vertex_map: &[Vector3<f32>],
        _width: u32,
        _height: u32,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        Err(crate::Error::not_supported("GPU normal map computation"))
    }
}

/// Unified dispatch API that automatically chooses CPU or GPU
pub mod unified {
    
    use nalgebra::{Matrix4, Vector3};

    /// Unified point cloud transform (CPU or GPU)
    pub fn transform_points(
        points: &[Vector3<f32>],
        transform: &Matrix4<f32>,
        use_gpu: bool,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        if use_gpu {
            // Dispatch to GPU
            Err(crate::Error::not_supported("GPU dispatch in unified API"))
        } else {
            // CPU fallback
            Ok(points.iter()
                .map(|p| {
                    let p_h = p.insert_row(3, 1.0);
                    let transformed = transform * p_h;
                    Vector3::new(transformed.x, transformed.y, transformed.z)
                })
                .collect())
        }
    }

    /// Configuration for automatic CPU/GPU selection
    pub struct ComputeConfig {
        pub use_gpu_threshold: usize, // Minimum elements to use GPU
        pub force_cpu: bool,
        pub force_gpu: bool,
    }

    impl Default for ComputeConfig {
        fn default() -> Self {
            Self {
                use_gpu_threshold: 10000, // Use GPU for >10k points
                force_cpu: false,
                force_gpu: false,
            }
        }
    }

    /// Auto-select CPU or GPU based on problem size
    pub fn should_use_gpu(config: &ComputeConfig, element_count: usize) -> bool {
        if config.force_cpu {
            false
        } else if config.force_gpu {
            true
        } else {
            element_count >= config.use_gpu_threshold
        }
    }
}
