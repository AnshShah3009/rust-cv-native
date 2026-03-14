//! Benchmark: 3D pipeline operations (marching cubes, TSDF, ICP, KDTree, Poisson)
//!
//! Run:
//!   cargo bench --bench pipeline_3d
//!   cargo bench --bench pipeline_3d -- marching_cubes   # single group
//!   cargo bench --bench pipeline_3d -- kdtree           # single group

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cv_3d::mesh::reconstruction::poisson_reconstruction;
use cv_3d::spatial::KDTree;
use cv_3d::tsdf::CameraIntrinsics;
use cv_3d::TSDFVolume;
use cv_core::PointCloud;
use cv_hal::context::ComputeContext;
use cv_hal::gpu::GpuContext;
use cv_hal::storage::WgpuGpuStorage;
use cv_registration::registration_icp_point_to_plane;
use nalgebra::{Matrix4, Point3, Vector3};
use std::marker::PhantomData;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helpers: synthetic data generators
// ---------------------------------------------------------------------------

/// Generate a sphere point cloud with outward normals.
fn sphere_cloud(num_points: usize, radius: f32) -> PointCloud {
    let mut points = Vec::with_capacity(num_points);
    let mut normals = Vec::with_capacity(num_points);

    // Fibonacci sphere for even distribution
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    for i in 0..num_points {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / golden_ratio;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) / num_points as f32).acos();
        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();
        let pt = Point3::new(x, y, z);
        let n = Vector3::new(x, y, z).normalize();
        points.push(pt);
        normals.push(n);
    }

    PointCloud::new(points).with_normals(normals).unwrap()
}

/// Generate a sphere signed-distance field on a regular grid.
/// Returns flat f32 array in (x, y, z) order with TSDF-like values.
fn sphere_sdf(grid_size: usize, radius: f32) -> Vec<f32> {
    let voxel_size = 2.2 * radius / grid_size as f32;
    let origin = -1.1 * radius;
    let mut data = vec![0.0f32; grid_size * grid_size * grid_size * 2]; // tsdf + weight pairs
    for z in 0..grid_size {
        for y in 0..grid_size {
            for x in 0..grid_size {
                let px = origin + (x as f32 + 0.5) * voxel_size;
                let py = origin + (y as f32 + 0.5) * voxel_size;
                let pz = origin + (z as f32 + 0.5) * voxel_size;
                let dist = (px * px + py * py + pz * pz).sqrt() - radius;
                let truncated = dist.clamp(-0.1, 0.1);
                let idx = (z * grid_size * grid_size + y * grid_size + x) * 2;
                data[idx] = truncated; // tsdf value
                data[idx + 1] = 1.0; // weight
            }
        }
    }
    data
}

fn try_gpu() -> Option<GpuContext> {
    GpuContext::new().ok()
}

// ---------------------------------------------------------------------------
// 1. Marching Cubes
// ---------------------------------------------------------------------------

fn bench_marching_cubes(c: &mut Criterion) {
    let gpu = try_gpu();

    let mut group = c.benchmark_group("marching_cubes");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(15);

    for &grid_size in &[64usize, 128] {
        let label = format!("{}^3", grid_size);
        let sdf_data = sphere_sdf(grid_size, 0.4);
        let voxel_size = 2.2 * 0.4 / grid_size as f32;

        // -- CPU (via Poisson's internal extract_isosurface — benchmarked via poisson group)
        // CPU marching cubes is internal; we benchmark the GPU path here.

        // -- WGPU GPU --
        if let Some(ref ctx) = gpu {
            // Upload SDF volume to GPU
            let storage =
                WgpuGpuStorage::from_slice_ctx(ctx, &sdf_data).expect("GPU upload failed");
            let shape = cv_core::TensorShape::new(
                grid_size * 2, // channels: tsdf+weight pairs
                grid_size,     // height
                grid_size,     // width
            );
            let gpu_vol = cv_core::Tensor {
                storage,
                shape,
                dtype: cv_core::DataType::F32,
                _phantom: PhantomData,
            };

            // Warmup
            let _ = cv_hal::gpu_kernels::marching_cubes::extract_mesh(
                ctx, &gpu_vol, voxel_size, 0.0, 500_000,
            );
            ctx.wait_idle().ok();

            group.bench_with_input(BenchmarkId::new("wgpu", &label), &(), |b, _| {
                b.iter(|| {
                    let out = cv_hal::gpu_kernels::marching_cubes::extract_mesh(
                        black_box(ctx),
                        black_box(&gpu_vol),
                        voxel_size,
                        0.0,
                        500_000,
                    )
                    .unwrap();
                    ctx.wait_idle().ok();
                    black_box(out)
                });
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. KDTree queries
// ---------------------------------------------------------------------------

fn bench_kdtree(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree_queries");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &n in &[10_000usize, 100_000] {
        let label = format!("n={}", n);
        let cloud = sphere_cloud(n, 1.0);

        // Build KDTree
        let mut tree = KDTree::new();
        for (i, pt) in cloud.points.iter().enumerate() {
            tree.insert(*pt, i);
        }

        let query = Point3::new(0.5, 0.5, 0.5);

        // -- Nearest neighbor --
        group.bench_with_input(BenchmarkId::new("nearest/1", &label), &(), |b, _| {
            b.iter(|| black_box(tree.nearest_neighbor(black_box(&query))));
        });

        // -- K nearest neighbors (k=20) --
        group.bench_with_input(BenchmarkId::new("knn/20", &label), &(), |b, _| {
            b.iter(|| black_box(tree.k_nearest_neighbors(black_box(&query), 20)));
        });

        // -- Radius search (r=0.1) --
        group.bench_with_input(BenchmarkId::new("radius/0.1", &label), &(), |b, _| {
            b.iter(|| black_box(tree.search_radius(black_box(&query), 0.1)));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. ICP Registration
// ---------------------------------------------------------------------------

fn bench_icp(c: &mut Criterion) {
    let mut group = c.benchmark_group("icp_registration");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    for &n in &[1_000usize, 10_000] {
        let label = format!("n={}", n);
        let target = sphere_cloud(n, 1.0);

        // Source = target translated by small offset (simulates registration scenario)
        let offset = Vector3::new(0.05, 0.03, 0.02);
        let source_pts: Vec<_> = target.points.iter().map(|p| p + offset).collect();
        let source = PointCloud::new(source_pts)
            .with_normals(target.normals.clone().unwrap())
            .unwrap();

        let init = Matrix4::identity();

        group.bench_with_input(BenchmarkId::new("point_to_plane", &label), &(), |b, _| {
            b.iter(|| {
                black_box(registration_icp_point_to_plane(
                    black_box(&source),
                    black_box(&target),
                    0.5,
                    black_box(&init),
                    30,
                ))
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Poisson Reconstruction
// ---------------------------------------------------------------------------

fn bench_poisson(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_reconstruction");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    for &n in &[10_000usize, 50_000] {
        let label = format!("n={}_d6", n);
        let cloud = sphere_cloud(n, 1.0);

        group.bench_with_input(BenchmarkId::new("depth_6", &label), &(), |b, _| {
            b.iter(|| black_box(poisson_reconstruction(black_box(&cloud), 6, 1.0)));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. TSDF Volume Integration
// ---------------------------------------------------------------------------

fn bench_tsdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("tsdf_integration");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    for &(w, h) in &[(320usize, 240usize), (640, 480)] {
        let label = format!("{}x{}", w, h);

        // Synthetic depth image: hemisphere at z=1.0
        let depth: Vec<f32> = (0..w * h)
            .map(|i| {
                let x = (i % w) as f32 / w as f32 - 0.5;
                let y = (i / w) as f32 / h as f32 - 0.5;
                let r2 = x * x + y * y;
                if r2 < 0.2 {
                    1.0 - (0.2 - r2).sqrt() * 0.3
                } else {
                    0.0 // no depth
                }
            })
            .collect();

        let intrinsics = CameraIntrinsics::new(
            w as f32,       // fx
            h as f32,       // fy
            w as f32 / 2.0, // cx
            h as f32 / 2.0, // cy
            w as u32,       // width
            h as u32,       // height
        );
        let extrinsics = Matrix4::identity();

        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| {
                let mut volume = TSDFVolume::new(0.01, 0.03);
                volume.integrate(
                    black_box(&depth),
                    None,
                    black_box(&intrinsics),
                    black_box(&extrinsics),
                    w,
                    h,
                );
                black_box(&volume);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion entry point
// ---------------------------------------------------------------------------

criterion_group!(
    pipeline_3d,
    bench_marching_cubes,
    bench_kdtree,
    bench_icp,
    bench_poisson,
    bench_tsdf,
);
criterion_main!(pipeline_3d);
