pub mod backend;
pub mod cpu;
pub mod device;
pub mod gpu;
pub mod gpu_utils;
pub mod gpu_sparse;
pub mod gpu_kernels;

pub use gpu::*;
pub use gpu_kernels::*;

pub use backend::{BackendType, Capability, ComputeBackend, DeviceId, QueueId, QueueType};
pub use cpu::*;
pub use device::*;
pub use gpu_utils::{
    read_gpu_max_bytes_from_env, parse_bytes_with_suffix, estimate_image_buffer_size, fits_in_budget,
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Queue error: {0}")]
    QueueError(String),

    #[error("Kernel error: {0}")]
    KernelError(String),

    #[error("Not supported: {0}")]
    NotSupported(String),

    #[error("Initialization error: {0}")]
    InitError(String),
}

impl Error {
    pub fn backend_not_available(backend: impl Into<String>) -> Self {
        Self::BackendNotAvailable(backend.into())
    }

    pub fn not_supported(feature: impl Into<String>) -> Self {
        Self::NotSupported(feature.into())
    }
}
