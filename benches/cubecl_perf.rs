//! Performance Benchmark: CubeCL Optimized vs Standard vs WGPU vs CPU
//!
//! Run with:
//!   cargo bench --bench cubecl_perf
//!
//! Or quick test:
//!   cargo bench --bench cubecl_perf -- --quick

use cubecl::prelude::*;
use cv_core::{CpuTensor, Tensor, TensorShape};
use std::marker::PhantomData;
use std::time::Instant;

const BENCH_SIZES: &[usize] = &[1024, 4096, 16384, 65536, 262144];
const IMAGE_SIZES: &[(usize, usize)] = &[
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
];
const POINT_SIZES: &[usize] = &[1024, 4096, 16384, 65536];

// ============================================================================
// CPU Benchmarks (Baseline)
// ============================================================================

mod cpu {
    use super::*;

    pub fn bench_element_wise_add(c: &mut Criterion, size: usize) {
        let mut group = c.benchmark_group("cpu_element_add");
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
                    let b: Vec<f32> = (0..size).map(|i| i as f32 * 0.2).collect();
                    let mut result = vec![0.0f32; size];
                    for i in 0..size {
                        result[i] = a[i] + b[i];
                    }
                    criterion::black_box(result);
                });
            },
        );
        group.finish();
    }

    pub fn bench_conv2d_3x3(c: &mut Criterion, (width, height): (usize, usize)) {
        let mut group = c.benchmark_group("cpu_conv2d_3x3");
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter((width, height)),
            &(width, height),
            |b, &(w, h)| {
                b.iter(|| {
                    let input: Vec<f32> = (0..w * h).map(|i| (i % 255) as f32).collect();
                    let kernel = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
                    let mut output = vec![0.0f32; w * h];
                    let khalf = 1;

                    for y in khalf..h - khalf {
                        for x in khalf..w - khalf {
                            let mut sum = 0.0f32;
                            for ky in 0..3 {
                                for kx in 0..3 {
                                    let px = x + kx - khalf;
                                    let py = y + ky - khalf;
                                    sum += input[py * w + px] * kernel[ky * 3 + kx];
                                }
                            }
                            output[y * w + x] = sum / 16.0;
                        }
                    }
                    criterion::black_box(output);
                });
            },
        );
        group.finish();
    }

    pub fn bench_knn(c: &mut Criterion, num_points: usize) {
        let mut group = c.benchmark_group("cpu_knn");
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(num_points),
            &num_points,
            |b, &n| {
                b.iter(|| {
                    let points: Vec<f32> = (0..n * 3).map(|i| (i as f32) * 0.01).collect();
                    let queries: Vec<f32> = vec![0.0, 0.0, 0.0];

                    // KNN with k=8
                    let k = 8;
                    let mut results = vec![0.0f32; k];
                    let mut indices = vec![0u32; k];

                    for q_idx in 0..1 {
                        let qx = queries[q_idx * 3];
                        let qy = queries[q_idx * 3 + 1];
                        let qz = queries[q_idx * 3 + 2];

                        let mut dists: Vec<(f32, usize)> = (0..n)
                            .map(|i| {
                                let dx = qx - points[i * 3];
                                let dy = qy - points[i * 3 + 1];
                                let dz = qz - points[i * 3 + 2];
                                (dx * dx + dy * dy + dz * dz, i)
                            })
                            .collect();

                        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        for i in 0..k.min(n) {
                            results[i] = dists[i].0;
                            indices[i] = dists[i].1 as u32;
                        }
                    }
                    criterion::black_box((results, indices));
                });
            },
        );
        group.finish();
    }

    pub fn bench_matmul(c: &mut Criterion, size: usize) {
        let mut group = c.benchmark_group("cpu_matmul");
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            &size,
            |b, &n| {
                b.iter(|| {
                    let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
                    let b: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.02).collect();
                    let mut c = vec![0.0f32; n * n];

                    for i in 0..n {
                        for j in 0..n {
                            for k in 0..n {
                                c[i * n + j] += a[i * n + k] * b[k * n + j];
                            }
                        }
                    }
                    criterion::black_box(c);
                });
            },
        );
        group.finish();
    }
}

// ============================================================================
// WGPU Benchmarks
// ============================================================================

mod wgpu {
    use super::*;

    pub fn bench_element_wise_add(c: &mut Criterion, size: usize) {
        // Note: Actual WGPU benchmarks would require async runtime
        // This is a placeholder showing the structure
        let mut group = c.benchmark_group("wgpu_element_add");
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.to_async(&std::thread::spawn(|| WgpuDevice::default()))
                    .iter(|| async {
                        // Would create tensors and run kernel
                        criterion::black_box(());
                    });
            },
        );
        group.finish();
    }
}

// ============================================================================
// Criterion Main
// ============================================================================

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_element_add(c: &mut Criterion) {
    for &size in BENCH_SIZES {
        cpu::bench_element_wise_add(c, size);
    }
}

fn bench_conv2d(c: &mut Criterion) {
    for &size in IMAGE_SIZES {
        cpu::bench_conv2d_3x3(c, size);
    }
}

fn bench_knn(c: &mut Criterion) {
    for &size in POINT_SIZES {
        cpu::bench_knn(c, size);
    }
}

fn bench_matmul(c: &mut Criterion) {
    for &size in &[64, 128, 256, 512] {
        cpu::bench_matmul(c, size);
    }
}

criterion_group!(
    benches,
    bench_element_add,
    bench_conv2d,
    bench_knn,
    bench_matmul
);
criterion_main!(benches);
