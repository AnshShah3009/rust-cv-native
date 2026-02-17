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
