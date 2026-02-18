use criterion::{criterion_group, criterion_main, Criterion};
use cv_runtime::orchestrator::{scheduler, GroupPolicy};
use cv_imgproc::{threshold, threshold_ctx};
use image::GrayImage;
use std::sync::Arc;

fn bench_threading_isolation(c: &mut Criterion) {
    let size = 2048;
    let img = GrayImage::new(size, size);
    
    let s = scheduler();
    
    // 1. Global pool (shared)
    let global_group = s.get_group("default").unwrap().unwrap();
    
    // 2. Isolated pool (pinned)
    let isolated_group = s.create_group(
        "isolated_bench", 
        4, 
        Some(vec![0, 1, 2, 3]), 
        GroupPolicy { allow_work_stealing: false, allow_dynamic_scaling: false }
    ).unwrap();

    let mut group = c.benchmark_group("Threading Isolation");
    
    group.bench_function("Global Pool", |b| {
        b.iter(|| {
            threshold_ctx(&img, 128, 255, cv_imgproc::ThresholdType::Binary, &global_group);
        })
    });

    group.bench_function("Isolated Pool (4 cores)", |b| {
        b.iter(|| {
            threshold_ctx(&img, 128, 255, cv_imgproc::ThresholdType::Binary, &isolated_group);
        })
    });

    group.finish();
}

fn bench_gpu_acceleration(c: &mut Criterion) {
    let size = 2048;
    let img = GrayImage::new(size, size);
    
    let s = scheduler();
    let cpu_group = s.get_group("default").unwrap().unwrap();
    let gpu_group = s.best_gpu_or_cpu();

    let mut group = c.benchmark_group("Hardware Acceleration");

    group.bench_function("CPU (Parallel SIMD)", |b| {
        b.iter(|| {
            threshold_ctx(&img, 128, 255, cv_imgproc::ThresholdType::Binary, &cpu_group);
        })
    });

    if let cv_hal::compute::ComputeDevice::Gpu(_) = gpu_group.device() {
        group.bench_function("GPU (WebGPU)", |b| {
            b.iter(|| {
                threshold_ctx(&img, 128, 255, cv_imgproc::ThresholdType::Binary, &gpu_group);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_threading_isolation, bench_gpu_acceleration);
criterion_main!(benches);
