use cv_3d::gpu::point_cloud::*;
use cv_hal::gpu::GpuContext;
use nalgebra::Point3;

#[tokio::test]
async fn test_multi_gpu_normals() {
    let adapters = GpuContext::enumerate_adapters().await;
    println!("Found {} GPU adapters", adapters.len());

    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.1, 0.1, 0.0),
    ];

    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!("Testing Adapter {}: {} ({:?})", i, info.name, info.device_type);

        let ctx = match GpuContext::from_adapter(adapter.clone()).await {
            Ok(c) => c,
            Err(e) => {
                println!("  Failed to create context: {}", e);
                continue;
            }
        };

        let normals = compute_normals_with_ctx(&points, 3, &ctx);
        println!("  Computed {} normals", normals.len());
        
        if !normals.is_empty() {
            println!("  First normal: {:?}", normals[0]);
            assert!(normals[0].z.abs() > 0.9);
        }
    }
}
