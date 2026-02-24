use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cv_stereo::BlockMatcher;
use image::GrayImage;
use tokio::runtime::Runtime;

fn benchmark_async_bridge(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (left, right) = (GrayImage::new(256, 256), GrayImage::new(256, 256));

    let mut group = c.benchmark_group("async_bridge");
    group.sample_size(10); // Reduce sample size for speed

    // Baseline: Direct synchronous call (Global Pool)
    group.bench_function("sync_direct", |b| {
        b.iter(|| {
            let matcher = BlockMatcher::new();
            matcher
                .compute(black_box(&left), black_box(&right))
                .unwrap();
        })
    });

    // Wrapped: Via spawn_blocking (Simulates async_ops)
    group.bench_function("async_wrapper", |b| {
        b.to_async(&rt).iter(|| async {
            tokio::task::spawn_blocking({
                let l = left.clone();
                let r = right.clone();
                move || {
                    let matcher = BlockMatcher::new();
                    matcher.compute(&l, &r).unwrap()
                }
            })
            .await
            .unwrap()
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_async_bridge);
criterion_main!(benches);
