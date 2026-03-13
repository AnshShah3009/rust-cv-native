//! Runtime orchestration layer for `rust-cv-native`.
//!
//! Provides task scheduling, device management, pipeline execution, memory
//! management, and observability for CPU and GPU compute backends.

pub mod device_registry;
pub mod distributed;
pub mod error;
pub mod executor;
pub mod memory;
pub mod memory_manager;
pub mod observe;
pub mod orchestrator;
pub mod pipeline;

pub use device_registry::{registry, DeviceRegistry, DeviceRuntime, SubmissionIndex};
pub use error::ErrorContext;
pub use memory::UnifiedBuffer;
pub use observe::{observability, Metrics, ObservabilityLayer, RuntimeEvent};
pub use orchestrator::{
    best_runner, default_runner, scheduler, try_best_runner, try_default_runner, GroupPolicy,
    ResourceGroup, RuntimeRunner, TaskPriority, TaskScheduler, WorkloadHint,
};
pub use pipeline::{AsyncPipelineHandle, ExecutionEvent, PipelineResult};
pub use pipeline::{BufferAlloc, TransientBufferPool};
pub use pipeline::{BufferId, Pipeline, PipelineNode};
pub use pipeline::{ExecutionGraph, NodeDependency, NodeId};
pub use pipeline::{FusedKernel, FusionPattern, KernelFuser};
pub use pipeline::{KernelDispatcher, NoOpDispatcher};

/// Top-level runtime error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// General runtime failure (scheduler, executor, device lookup).
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Buffer allocation, sync, or out-of-bounds error.
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Lock poisoning or other concurrency issue.
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),

    /// Operation not available on the current backend.
    #[error("Not supported: {0}")]
    NotSupported(String),

    /// Error propagated from the HAL layer.
    #[error("HAL error: {0}")]
    HalError(#[from] cv_hal::Error),

    /// Error propagated from `cv-core`.
    #[error("Core error: {0}")]
    CoreError(#[from] cv_core::Error),

    /// Failure during subsystem initialization.
    #[error("Initialization error: {0}")]
    InitError(String),
}

/// Convenience alias for results returned by the runtime crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Submit a closure to a named resource group via the global [`TaskScheduler`].
///
/// Returns `Err` if the scheduler is not initialized or the group does not exist.
#[macro_export]
macro_rules! submit_to {
    ($group_name:expr, $f:block) => {
        match $crate::scheduler() {
            Ok(s) => s.submit($group_name, move || $f),
            Err(e) => Err(e),
        }
    };
}
