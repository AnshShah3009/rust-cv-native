use std::sync::Arc;
use crate::{BackendType, DeviceId, Result};
use cv_core::{Tensor, storage::Storage};

/// A unified context for executing compute operations.
/// 
/// This trait abstracts over different compute backends (CPU, CUDA, Vulkan/WebGPU),
/// allowing high-level algorithms to be written in a backend-agnostic way.
pub trait ComputeContext: Send + Sync {
    /// Get the backend type of this context
    fn backend_type(&self) -> BackendType;

    /// Get the unique device ID
    fn device_id(&self) -> DeviceId;

    /// Wait for all pending operations to complete
    fn wait_idle(&self) -> Result<()>;

    // --- Core Operations ---

    /// Execute a 2D convolution
    fn convolve_2d<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        kernel: &Tensor<f32, S>,
        border_mode: BorderMode,
    ) -> Result<Tensor<f32, S>>;

    /// Execute a generic compute shader/kernel
    /// 
    /// * `name`: Name of the kernel (e.g., "gaussian_blur")
    /// * `buffers`: List of buffers (input/output)
    /// * `uniforms`: Uniform data (constants)
    /// * `workgroups`: (x, y, z) dispatch size
    fn dispatch<S: Storage<u8> + 'static>(
        &self,
        name: &str,
        buffers: &[&Tensor<u8, S>],
        uniforms: &[u8],
        workgroups: (u32, u32, u32),
    ) -> Result<()>;

    /// Execute a threshold operation
    fn threshold<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        thresh: u8,
        max_value: u8,
        typ: ThresholdType,
    ) -> Result<Tensor<u8, S>>;

    /// Execute a Sobel operator
    fn sobel<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        dx: i32,
        dy: i32,
        ksize: usize,
    ) -> Result<(Tensor<u8, S>, Tensor<u8, S>)>;

    /// Execute a morphological operation
    fn morphology<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        typ: MorphologyType,
        kernel: &Tensor<u8, S>,
        iterations: u32,
    ) -> Result<Tensor<u8, S>>;

    /// Execute a warp operation (affine or perspective)
    fn warp<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        matrix: &[[f32; 3]; 3],
        new_shape: (usize, usize),
        typ: WarpType,
    ) -> Result<Tensor<u8, S>>;

    /// Execute Non-Maximum Suppression
    fn nms<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        threshold: f32,
        window_size: usize,
    ) -> Result<Tensor<f32, S>>;

    /// Execute Bounding Box NMS
    /// input: (N, 5) tensor [x1, y1, x2, y2, score]
    fn nms_boxes<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> Result<Vec<usize>>;

    /// Execute Rotated Bounding Box NMS
    /// input: (N, 6) tensor [cx, cy, w, h, angle, score]
    fn nms_rotated_boxes<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        iou_threshold: f32,
    ) -> Result<Vec<usize>>;

    /// Execute Polygon NMS
    /// input: list of polygons, list of scores
    fn nms_polygons(
        &self,
        polygons: &[cv_core::Polygon],
        scores: &[f32],
        iou_threshold: f32,
    ) -> Result<Vec<usize>>;

    /// Transform a point cloud
    fn pointcloud_transform<S: Storage<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        transform: &[[f32; 4]; 4],
    ) -> Result<Tensor<f32, S>>;

    /// Compute normals for a point cloud
    fn pointcloud_normals<S: Storage<f32> + 'static>(
        &self,
        points: &Tensor<f32, S>,
        k_neighbors: u32,
    ) -> Result<Tensor<f32, S>>;

    /// TSDF Volume Integration
    fn tsdf_integrate<S: Storage<f32> + 'static>(
        &self,
        depth_image: &Tensor<f32, S>,
        camera_pose: &[[f32; 4]; 4],
        intrinsics: &[f32; 4],
        tsdf_volume: &mut Tensor<f32, S>,
        weight_volume: &mut Tensor<f32, S>,
        voxel_size: f32,
        truncation: f32,
    ) -> Result<()>;

    /// Color space conversion
    fn cvt_color<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        code: ColorConversion,
    ) -> Result<Tensor<u8, S>>;

    /// Resize an image
    fn resize<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        new_shape: (usize, usize),
    ) -> Result<Tensor<u8, S>>;

    /// Bilateral Filter
    fn bilateral_filter<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        d: i32,
        sigma_color: f32,
        sigma_space: f32,
    ) -> Result<Tensor<u8, S>>;

    /// FAST Keypoint Detection
    /// Returns a score map (1 channel, same size as input)
    fn fast_detect<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        threshold: u8,
        non_max_suppression: bool,
    ) -> Result<Tensor<u8, S>>;

    /// Gaussian Blur
    fn gaussian_blur<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        sigma: f32,
        k_size: usize,
    ) -> Result<Tensor<u8, S>>;

    /// Elementwise Subtraction (A - B)
    /// Input: Signed output often needed for DoG, but we might use f32 or i16.
    /// For SIFT DoG, we usually use f32.
    fn subtract<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug, S: Storage<T> + 'static>(
        &self,
        a: &Tensor<T, S>,
        b: &Tensor<T, S>,
    ) -> Result<Tensor<T, S>>;

    /// Feature Matching
    fn match_descriptors<S: Storage<u8> + 'static>(
        &self,
        query: &Tensor<u8, S>,
        train: &Tensor<u8, S>,
        ratio_threshold: f32,
    ) -> Result<cv_core::Matches>;

    /// SIFT Local Extrema Detection
    /// Finds local maxima/minima in 3x3x3 scale-space neighborhood.
    /// Returns a U8 score map on CPU for refinement.
    fn sift_extrema<S: Storage<f32> + 'static>(
        &self,
        dog_prev: &Tensor<f32, S>,
        dog_curr: &Tensor<f32, S>,
        dog_next: &Tensor<f32, S>,
        threshold: f32,
        edge_threshold: f32,
    ) -> Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>>;

    /// SIFT Descriptor Computation
    /// Returns a descriptor tensor (num_kps x 128)
    fn compute_sift_descriptors<S: Storage<f32> + 'static>(
        &self,
        image: &Tensor<f32, S>,
        keypoints: &cv_core::KeyPoints,
    ) -> Result<cv_core::Descriptors>;

    /// ICP Point Correspondence
    /// Finds nearest neighbors from src points to tgt points
    fn icp_correspondences<S: Storage<f32> + 'static>(
        &self,
        src: &Tensor<f32, S>,
        tgt: &Tensor<f32, S>,
        max_dist: f32,
    ) -> Result<Vec<(usize, usize, f32)>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorConversion {
    RgbToGray,
    BgrToGray,
    GrayToRgb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpType {
    Affine,
    Perspective,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphologyType {
    Erode,
    Dilate,
    Open,
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdType {
    Binary,
    BinaryInv,
    Trunc,
    ToZero,
    ToZeroInv,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BorderMode {
    Constant(f32),
    Replicate,
    Reflect,
    Wrap,
}

/// A handle to a compute device and its queues
pub struct Context {
    backend: BackendType,
    // future: wgpu::Device, wgpu::Queue, etc.
}
