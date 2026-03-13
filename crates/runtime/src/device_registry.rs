use crate::executor::ExecutorPool;
use crate::memory_manager::MemoryManager;
use crate::Result;
pub use cv_hal::{context::ComputeContext, BackendType, ComputeBackend, DeviceId, SubmissionIndex};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Supported compute backend contexts.
pub enum BackendContext {
    /// CPU backend (always available).
    Cpu(Arc<cv_hal::cpu::CpuBackend>),
    /// GPU backend (WebGPU/Vulkan).
    Gpu(Arc<cv_hal::gpu::GpuContext>),
    /// Apple MLX backend.
    Mlx(Arc<cv_hal::mlx::MlxContext>),
}

impl BackendContext {
    /// Return the HAL device identifier for this backend.
    pub fn device_id(&self) -> DeviceId {
        match self {
            BackendContext::Cpu(c) => ComputeContext::device_id(c.as_ref()),
            BackendContext::Gpu(c) => ComputeContext::device_id(c.as_ref()),
            BackendContext::Mlx(c) => ComputeContext::device_id(c.as_ref()),
        }
    }

    /// Return the backend type (CPU, WebGPU, Vulkan, etc.).
    pub fn backend_type(&self) -> BackendType {
        match self {
            BackendContext::Cpu(c) => ComputeContext::backend_type(c.as_ref()),
            BackendContext::Gpu(c) => ComputeContext::backend_type(c.as_ref()),
            BackendContext::Mlx(c) => ComputeContext::backend_type(c.as_ref()),
        }
    }
}

/// Runtime state for a specific compute device.
///
/// Owns the compute context and manages execution and memory for that device.
pub struct DeviceRuntime {
    id: DeviceId,
    backend: BackendType,
    context: BackendContext,

    // Track GPU submission ordering
    last_submitted: Mutex<SubmissionIndex>,
    last_completed: Mutex<SubmissionIndex>,

    executors: Mutex<ExecutorPool>,
    memory: Arc<MemoryManager>,
}

impl DeviceRuntime {
    /// Create a new device runtime wrapping the given backend context.
    pub fn new(context: BackendContext) -> Self {
        let id = context.device_id();

        // MemoryManager needs Arc<Device> for GPU pooling.
        // If it's CPU, we still need a MemoryManager but it doesn't do GPU pooling.
        // We'll provide a dummy device for CPU or handle it in MemoryManager.
        // For now, let's extract the device from the context.

        let memory = match &context {
            BackendContext::Gpu(c) => Arc::new(MemoryManager::new(id, Some(c.device.clone()))),
            BackendContext::Mlx(_) => {
                // MLX contexts don't use standard wgpu buffer pooling yet
                Arc::new(MemoryManager::new(id, None))
            }
            BackendContext::Cpu(_) => Arc::new(MemoryManager::new(id, None)),
        };

        Self {
            id,
            backend: context.backend_type(),
            context,
            last_submitted: Mutex::new(SubmissionIndex(0)),
            last_completed: Mutex::new(SubmissionIndex(0)),
            executors: Mutex::new(ExecutorPool::new(id)),
            memory,
        }
    }

    /// Return the device identifier.
    pub fn id(&self) -> DeviceId {
        self.id
    }

    /// Return the backend type for this device.
    pub fn backend(&self) -> BackendType {
        self.backend
    }

    /// Return a reference to the underlying backend context.
    pub fn context(&self) -> &BackendContext {
        &self.context
    }

    /// Return the memory manager for this device.
    pub fn memory(&self) -> &Arc<MemoryManager> {
        &self.memory
    }

    /// Return the executor pool mutex for this device.
    pub fn executors(&self) -> &Mutex<ExecutorPool> {
        &self.executors
    }

    /// Advance and return the next GPU submission index.
    pub fn next_submission(&self) -> SubmissionIndex {
        if let Ok(mut last) = self.last_submitted.lock() {
            last.next()
        } else {
            SubmissionIndex(0)
        }
    }

    /// Mark a submission as completed and trigger garbage collection.
    pub fn mark_completed(&self, index: SubmissionIndex) {
        if let Ok(mut last) = self.last_completed.lock() {
            if index > *last {
                *last = index;
            }
        }

        // Collect garbage when GPU operations complete
        self.memory.collect_garbage(index);
    }

    /// Return the highest completed submission index for this device.
    pub fn last_completed(&self) -> SubmissionIndex {
        match self.last_completed.lock() {
            Ok(l) => *l,
            Err(_) => {
                // Lock is poisoned. This indicates a panic occurred while holding the lock.
                // We conservatively return 0 to avoid cascading panics in Drop or cleanup paths.
                // NOTE: A poisoned lock means system integrity may be compromised.
                eprintln!(
                    "WARNING: last_completed lock poisoned for device {:?}",
                    self.id
                );
                SubmissionIndex(0)
            }
        }
    }
}

/// Global registry of all available compute devices.
///
/// This is the primary owner of all device-specific resources.
pub struct DeviceRegistry {
    devices: Mutex<HashMap<DeviceId, Arc<DeviceRuntime>>>,
    default_cpu: DeviceId,
    default_gpu: Mutex<Option<DeviceId>>,
}

impl DeviceRegistry {
    /// Create a new registry with the CPU backend pre-registered.
    pub fn new() -> Result<Self> {
        let mut devices = HashMap::new();

        // Always initialize CPU backend
        let cpu_backend = cv_hal::cpu::CpuBackend::new().ok_or_else(|| {
            crate::Error::RuntimeError("Failed to initialize CPU backend".to_string())
        })?;
        let cpu_id = ComputeContext::device_id(&cpu_backend);
        let cpu_runtime = Arc::new(DeviceRuntime::new(BackendContext::Cpu(Arc::new(
            cpu_backend,
        ))));
        devices.insert(cpu_id, cpu_runtime);

        Ok(Self {
            devices: Mutex::new(devices),
            default_cpu: cpu_id,
            default_gpu: Mutex::new(None),
        })
    }

    /// Register a new device context
    pub fn register_device(&self, context: BackendContext, is_default_gpu: bool) {
        if let Ok(mut devices) = self.devices.lock() {
            let id = context.device_id();
            let runtime = Arc::new(DeviceRuntime::new(context));
            devices.insert(id, runtime);

            if is_default_gpu {
                if let Ok(mut default_gpu) = self.default_gpu.lock() {
                    *default_gpu = Some(id);
                }
            }
        }
    }

    /// Look up a device runtime by its ID.
    pub fn get_device(&self, id: DeviceId) -> Option<Arc<DeviceRuntime>> {
        self.devices.lock().ok()?.get(&id).cloned()
    }

    /// Return the default CPU device runtime. Panics if the CPU device is missing.
    pub fn default_cpu(&self) -> Arc<DeviceRuntime> {
        self.get_device(self.default_cpu)
            .expect("CPU device must exist")
    }

    /// Return the default GPU device runtime, if one was registered.
    pub fn default_gpu(&self) -> Option<Arc<DeviceRuntime>> {
        let id = *self.default_gpu.lock().ok()?;
        id.and_then(|id| self.get_device(id))
    }

    /// Return all registered device runtimes.
    pub fn all_devices(&self) -> Vec<Arc<DeviceRuntime>> {
        self.devices
            .lock()
            .ok()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }
}

use std::sync::OnceLock;

static GLOBAL_REGISTRY: OnceLock<Result<Arc<DeviceRegistry>>> = OnceLock::new();

/// Return the global [`DeviceRegistry`] singleton, auto-detecting GPU and MLX backends.
pub fn registry() -> Result<Arc<DeviceRegistry>> {
    GLOBAL_REGISTRY
        .get_or_init(|| {
            let registry = Arc::new(DeviceRegistry::new()?);

            // Attempt to auto-detect GPU
            if let Ok(gpu_context) = cv_hal::gpu::GpuContext::global() {
                registry.register_device(BackendContext::Gpu(Arc::new(gpu_context.clone())), true);
            }

            // Attempt to auto-detect MLX (on Apple Silicon)
            if let Some(mlx_context) = cv_hal::mlx::MlxContext::new() {
                registry.register_device(BackendContext::Mlx(Arc::new(mlx_context)), false);
            }

            Ok(registry)
        })
        .as_ref()
        .map(|r| r.clone())
        .map_err(|e| crate::Error::RuntimeError(e.to_string()))
}
