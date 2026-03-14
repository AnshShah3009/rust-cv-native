//! Hardware Abstraction Layer (HAL) for computer vision compute operations.
//!
//! This crate provides a unified interface over CPU, GPU (via wgpu/WebGPU), and
//! experimental MLX backends. Algorithm crates depend only on the [`ComputeContext`]
//! trait, making them backend-agnostic.

pub mod backend;
pub mod compute;
pub mod context;
pub mod cpu;
pub mod gpu;
pub mod gpu_kernels;
pub mod gpu_sparse;
pub mod gpu_storage;
pub mod gpu_timer;
pub mod gpu_utils;
pub mod mlx;
pub mod storage;
pub mod tensor_ext;

#[cfg(feature = "cubecl")]
pub mod cubecl;

pub use gpu::*;
pub use gpu_kernels::*;
pub use gpu_storage::GpuStorage;
/// Convenience type alias for a GPU-backed tensor with typed storage.
pub type GpuTensor<T> = cv_core::Tensor<T, crate::storage::WgpuGpuStorage<T>>;

pub use backend::{
    BackendType, Capability, ComputeBackend, DeviceId, QueueId, QueueType, SubmissionIndex,
};
pub use cpu::*;
pub use gpu_utils::{
    estimate_image_buffer_size, fits_in_budget, parse_bytes_with_suffix,
    read_gpu_max_bytes_from_env,
};

/// Crate-wide result type using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during HAL operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The requested backend (e.g. CUDA, Vulkan) is not available on this system.
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    /// A device-level failure such as lost connection or driver crash.
    #[error("Device error: {0}")]
    DeviceError(String),

    /// GPU/CPU memory allocation or budget exceeded.
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// A command queue operation failed.
    #[error("Queue error: {0}")]
    QueueError(String),

    /// A compute kernel (shader) failed to compile or execute.
    #[error("Kernel error: {0}")]
    KernelError(String),

    /// The requested operation is not supported by the current backend.
    #[error("Not supported: {0}")]
    NotSupported(String),

    /// Backend or device initialization failed.
    #[error("Initialization error: {0}")]
    InitError(String),

    /// Caller supplied invalid arguments (wrong shape, type mismatch, etc.).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// An unexpected runtime failure.
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Propagated error from [`cv_core`].
    #[error("Core error: {0}")]
    CoreError(#[from] cv_core::Error),
}

impl Error {
    /// Create a [`BackendNotAvailable`](Error::BackendNotAvailable) error.
    pub fn backend_not_available(backend: impl Into<String>) -> Self {
        Self::BackendNotAvailable(backend.into())
    }

    /// Create a [`NotSupported`](Error::NotSupported) error.
    pub fn not_supported(feature: impl Into<String>) -> Self {
        Self::NotSupported(feature.into())
    }
}
