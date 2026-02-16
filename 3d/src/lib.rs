pub mod async_ops;
pub mod batch;
pub mod gaussian_splatting;
pub mod gpu;
pub mod mesh;
pub mod odometry;
// pub mod pose_graph;
pub mod raycasting;
pub mod registration;
pub mod spatial;
pub mod tsdf;

pub use async_ops::{
    AsyncConfig,
    point_cloud as async_point_cloud,
    registration as async_registration,
    mesh as async_mesh,
    raycasting as async_raycasting,
    tsdf as async_tsdf,
    pipeline as async_pipeline,
};

pub use batch::{
    BatchConfig,
    point_cloud as batch_point_cloud,
    registration as batch_registration,
    mesh as batch_mesh,
    raycasting as batch_raycasting,
    bench as batch_bench,
};

pub use gpu::{
    gpu_info, is_gpu_available, force_cpu_mode, force_gpu_mode,
    point_cloud as gpu_point_cloud,
    registration as gpu_registration,
    mesh as gpu_mesh,
    tsdf as gpu_tsdf,
    raycasting as gpu_raycasting,
};
pub use mesh::TriangleMesh;
pub use odometry::{compute_rgbd_odometry, OdometryMethod, OdometryResult};
pub use cv_core::PointCloud;
pub use raycasting::{
    cast_ray_mesh, cast_rays_mesh, closest_point_on_mesh, closest_points_on_mesh,
    mesh_to_mesh_distance, point_inside_mesh, Ray, RayHit,
};
pub use registration::{
    evaluate_registration, get_information_matrix_from_point_clouds, registration_colored_icp,
    registration_fgr_based_on_feature_matching, registration_gnc, registration_icp_point_to_plane,
    registration_multi_scale_icp, registration_ransac_based_on_feature_matching, ColoredICPResult,
    FPFHFeature, FastGlobalRegistrationOption, GNCOptimizer, GNCResult, GlobalRegistrationResult,
    ICPResult, RobustLoss, RobustLossType,
};
pub use spatial::{KDTree, Octree, VoxelGrid};
pub use tsdf::{CameraIntrinsics, TSDFVolume, Triangle, VoxelBlock};
pub use gaussian_splatting::{
    Gaussian, GaussianCloud, SphericalHarmonics, GaussianRasterizer, DifferentiableRasterizer,
    GaussianOptimizer, DensificationConfig, TrainingConfig, Camera, RasterizationResult,
    read_ply_gaussian_cloud, write_ply_gaussian_cloud,
};
