use std::fmt;

/// Identifies which compute backend is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// CPU backend using Rayon + SIMD.
    Cpu,
    /// NVIDIA CUDA backend.
    Cuda,
    /// Vulkan compute backend.
    Vulkan,
    /// Apple Metal compute backend.
    Metal,
    /// Microsoft DirectML backend.
    DirectML,
    /// NVIDIA TensorRT inference backend.
    TensorRT,
    /// Cross-platform WebGPU/wgpu backend.
    WebGPU,
    /// Apple MLX framework backend (experimental).
    Mlx,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::Cpu => write!(f, "CPU"),
            BackendType::Cuda => write!(f, "CUDA"),
            BackendType::Vulkan => write!(f, "Vulkan"),
            BackendType::Metal => write!(f, "Metal"),
            BackendType::DirectML => write!(f, "DirectML"),
            BackendType::TensorRT => write!(f, "TensorRT"),
            BackendType::WebGPU => write!(f, "WebGPU"),
            BackendType::Mlx => write!(f, "MLX"),
        }
    }
}

/// Hardware capability that a backend may or may not support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// General-purpose compute shader support.
    Compute,
    /// SIMD / vectorised instruction support.
    Simd,
    /// Tensor-core / matrix-multiply accelerator support.
    TensorCore,
    /// Hardware ray-tracing support.
    RayTracing,
}

/// Kind of command queue exposed by a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    /// Compute-only queue.
    Compute,
    /// Async copy / DMA transfer queue.
    Transfer,
    /// Graphics + compute queue.
    Graphics,
}

/// Unique identifier for a compute device within this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u32);

/// Unique identifier for a command queue on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueueId(pub u32);

/// Monotonically increasing submission index for tracking GPU operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct SubmissionIndex(pub u64);

impl SubmissionIndex {
    /// Increment and return the new submission index.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Self {
        self.0 += 1;
        *self
    }
}

/// Low-level backend descriptor providing device metadata and queue access.
///
/// Higher-level compute work is dispatched through [`crate::context::ComputeContext`]
/// instead; this trait is for device enumeration and capability queries.
pub trait ComputeBackend: Send + Sync {
    /// Return the backend type (CPU, Vulkan, CUDA, etc.).
    fn backend_type(&self) -> BackendType;
    /// Human-readable backend name.
    fn name(&self) -> &str;
    /// Unique device identifier.
    fn device_id(&self) -> DeviceId;
    /// Query whether this device supports a given capability.
    fn supports(&self, capability: Capability) -> bool;
    /// Get the queue ID for a given queue type.
    fn queue(&self, queue_type: QueueType) -> QueueId;
    /// Return the preferred queue type for this backend.
    fn preferred_queue(&self) -> QueueType;
}
