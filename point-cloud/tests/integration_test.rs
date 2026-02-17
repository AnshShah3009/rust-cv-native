/// Integration tests for point-cloud crate
/// Verifies that all CPU and GPU functions can be imported and used correctly

#[test]
fn test_cpu_filtering_imports() {
    use cv_point_cloud::cpu::filtering::{
        voxel_down_sample, remove_statistical_outliers, remove_radius_outliers,
    };

    // Just verifying that the imports work
    // The actual function calls would require test data
    let _ = voxel_down_sample;
    let _ = remove_statistical_outliers;
    let _ = remove_radius_outliers;
}

#[test]
fn test_cpu_normals_imports() {
    use cv_point_cloud::cpu::normals::{estimate_normals, orient_normals};

    // Just verifying that the imports work
    let _ = estimate_normals;
    let _ = orient_normals;
}

#[test]
fn test_cpu_segmentation_imports() {
    use cv_point_cloud::cpu::segmentation::{
        segment_plane, cluster_dbscan, compute_fpfh_feature,
    };

    // Just verifying that the imports work
    let _ = segment_plane;
    let _ = cluster_dbscan;
    let _ = compute_fpfh_feature;
}

#[test]
fn test_gpu_config_imports() {
    use cv_point_cloud::gpu::{ComputeMode, NormalComputeConfig};

    // Verify ComputeMode variants
    let _ = ComputeMode::CPU;
    let _ = ComputeMode::GPU;
    let _ = ComputeMode::Hybrid;
    let _ = ComputeMode::Adaptive;

    // Verify NormalComputeConfig construction
    let _config = NormalComputeConfig::default();
    let _config = NormalComputeConfig::cpu();
    let _config = NormalComputeConfig::gpu();
    let _config = NormalComputeConfig::fast();
    let _config = NormalComputeConfig::high_quality();
}

#[test]
fn test_gpu_normals_imports() {
    use cv_point_cloud::gpu::normals::{
        compute_normals, compute_normals_simple, voxel_based_normals,
        approximate_normals, refine_normals,
    };

    // Just verifying that the imports work
    let _ = compute_normals;
    let _ = compute_normals_simple;
    let _ = voxel_based_normals;
    let _ = approximate_normals;
    let _ = refine_normals;
}

#[test]
fn test_gpu_filtering_imports() {
    use cv_point_cloud::gpu::filtering::{
        statistical_outlier_removal, statistical_outlier_removal_simple,
        remove_statistical_outliers, radius_outlier_removal,
        radius_outlier_removal_simple, remove_radius_outliers,
        voxel_downsample,
    };

    // Just verifying that the imports work
    let _ = statistical_outlier_removal;
    let _ = statistical_outlier_removal_simple;
    let _ = remove_statistical_outliers;
    let _ = radius_outlier_removal;
    let _ = radius_outlier_removal_simple;
    let _ = remove_radius_outliers;
    let _ = voxel_downsample;
}

#[test]
fn test_crate_level_imports() {
    // Test that key types are available at crate level
    use cv_point_cloud::{ComputeMode, NormalComputeConfig};

    let _ = ComputeMode::CPU;
    let _ = NormalComputeConfig::default();
}

#[test]
fn test_unified_cpu_module_import() {
    // Verify we can import cpu module as a whole
    use cv_point_cloud::cpu;

    // Access functions through module
    let _ = cpu::voxel_down_sample;
    let _ = cpu::estimate_normals;
    let _ = cpu::segment_plane;
}

#[test]
fn test_unified_gpu_module_import() {
    // Verify we can import gpu module as a whole
    use cv_point_cloud::gpu;

    // Access types and functions through module
    let _ = gpu::ComputeMode::CPU;
    let _ = gpu::NormalComputeConfig::default();
    let _ = gpu::compute_normals;
    let _ = gpu::statistical_outlier_removal;
}
