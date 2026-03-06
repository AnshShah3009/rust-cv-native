//! Head-to-head benchmarks: CubeCL kernels vs WGSL baseline.
//!
//! Measures:
//!   normals_morton        — 1k / 10k / 100k / 500k points
//!   normals_batch_pca     — 10k / 100k / 500k (GPU eigensolver only)
//!   icp_dense_step        — 320×240 / 640×480 depth frames
//!   convolve_5x5          — 512×512 / 1024×1024 f32 images
//!   threshold             — 1M u8 elements
//!   radix_sort            — 100k / 1M u32 keys
//!
//! Run:
//!   cargo bench --features cubecl                   # WGPU/Metal
//!   cargo bench --features cubecl --no-default-features  # just CubeCL
//!
//! The suite also includes the wgpu baseline (from existing `normals.rs`) so
//! both measurements appear in the same Criterion HTML report.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cv_hal::cubecl::{
    get_client,
    kernels::{
        convolve::convolve,
        icp::dense_icp_step,
        image::threshold,
        pointcloud::{compute_normals_from_covariances, compute_normals_morton},
        sort::radix_sort,
    },
};
use rand::Rng;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

fn random_f32_flat(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>()).collect()
}

/// n points on a unit sphere, stride-4 (x,y,z,0)
fn sphere_points_stride4(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut pts = Vec::with_capacity(n * 4);
    for _ in 0..n {
        let theta = rng.gen::<f32>() * std::f32::consts::TAU;
        let phi = (rng.gen::<f32>() * 2.0 - 1.0).acos();
        let r = 1.0 + rng.gen::<f32>() * 0.01;
        pts.extend_from_slice(&[
            r * phi.sin() * theta.cos(),
            r * phi.sin() * theta.sin(),
            r * phi.cos(),
            0.0,
        ]);
    }
    pts
}

/// Random covariance matrices, 8 floats each [cxx,cxy,cxz,cyy,cyz,czz,0,0]
fn random_covs(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut covs = Vec::with_capacity(n * 8);
    for _ in 0..n {
        let a: f32 = rng.gen_range(0.1..1.0);
        let b: f32 = rng.gen_range(0.1..1.0);
        let c: f32 = rng.gen_range(0.001..0.01); // small eigenvalue → normal direction
        covs.extend_from_slice(&[a, 0.0, 0.0, b, 0.0, c, 0.0, 0.0]);
    }
    covs
}

fn random_depth_image(w: usize, h: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..w * h).map(|_| rng.gen_range(0.5_f32..5.0)).collect()
}

fn random_normals_stride4(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut ns = Vec::with_capacity(n * 4);
    for _ in 0..n {
        let nx: f32 = rng.gen::<f32>() - 0.5;
        let ny: f32 = rng.gen::<f32>() - 0.5;
        let nz: f32 = rng.gen::<f32>() - 0.5;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        ns.extend_from_slice(&[nx / len, ny / len, nz / len, 0.0]);
    }
    ns
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_normals_morton(c: &mut Criterion) {
    let client = get_client();
    let mut group = c.benchmark_group("normals_morton/cubecl");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(20);

    for &n in &[1_000usize, 10_000, 100_000, 500_000] {
        let pts = sphere_points_stride4(n);
        group.bench_with_input(BenchmarkId::new("n_pts", n), &pts, |b, pts| {
            b.iter(|| {
                let normals = compute_normals_morton(&client, black_box(pts), 16);
                black_box(normals);
            });
        });
    }
    group.finish();
}

fn bench_normals_batch_pca(c: &mut Criterion) {
    let client = get_client();
    let mut group = c.benchmark_group("normals_batch_pca/cubecl");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &n in &[10_000usize, 100_000, 500_000] {
        let covs = random_covs(n);
        group.bench_with_input(BenchmarkId::new("n_pts", n), &covs, |b, covs| {
            b.iter(|| {
                let normals = compute_normals_from_covariances(&client, black_box(covs));
                black_box(normals);
            });
        });
    }
    group.finish();
}

fn bench_icp_dense_step(c: &mut Criterion) {
    let client = get_client();
    let mut group = c.benchmark_group("icp_dense_step/cubecl");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(20);

    let identity: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let intr: [f32; 4] = [525.0, 525.0, 320.0, 240.0]; // RealSense-like

    for &(w, h) in &[(320usize, 240usize), (640, 480)] {
        let depth = random_depth_image(w, h);
        let normals = random_normals_stride4(w * h);
        group.bench_with_input(
            BenchmarkId::new("wxh", format!("{w}x{h}")),
            &(w, h),
            |b, &(w, h)| {
                b.iter(|| {
                    let (jtj, jtb) = dense_icp_step(
                        &client,
                        black_box(&depth),
                        black_box(&depth),
                        black_box(&normals),
                        &intr,
                        &identity,
                        w,
                        h,
                    );
                    black_box((jtj, jtb));
                });
            },
        );
    }
    group.finish();
}

fn bench_convolve_5x5(c: &mut Criterion) {
    let client = get_client();
    let mut group = c.benchmark_group("convolve_5x5/cubecl");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    let kernel: Vec<f32> = vec![1.0 / 25.0; 25]; // 5×5 box filter

    for &side in &[512usize, 1024] {
        let img = random_f32_flat(side * side);
        group.bench_with_input(BenchmarkId::new("side", side), &img, |b, img| {
            b.iter(|| {
                let out = convolve(&client, black_box(img), side, side, &kernel, 5, 5, 1);
                black_box(out);
            });
        });
    }
    group.finish();
}

fn bench_threshold(c: &mut Criterion) {
    let client = get_client();
    let mut group = c.benchmark_group("threshold/cubecl");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    let n = 1_000_000usize;
    let data: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
    group.bench_function("1M_u8", |b| {
        b.iter(|| {
            let out = threshold(&client, black_box(&data), 128, 255, 0);
            black_box(out);
        });
    });
    group.finish();
}

fn bench_radix_sort(c: &mut Criterion) {
    let client = get_client();
    let mut group = c.benchmark_group("radix_sort/cubecl");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    let mut rng = rand::thread_rng();
    for &n in &[100_000usize, 1_000_000] {
        let keys: Vec<u32> = (0..n).map(|_| rng.gen::<u32>()).collect();
        group.bench_with_input(BenchmarkId::new("n", n), &keys, |b, keys| {
            b.iter(|| {
                let sorted = radix_sort(&client, black_box(keys));
                black_box(sorted);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_normals_morton,
    bench_normals_batch_pca,
    bench_icp_dense_step,
    bench_convolve_5x5,
    bench_threshold,
    bench_radix_sort,
);
criterion_main!(benches);
