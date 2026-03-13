//! Cross-process load coordination for multi-process GPU sharing.

mod file;
mod shared_memory;

pub use file::FileCoordinator;
pub use shared_memory::ShmCoordinator;

use cv_hal::DeviceId;
use std::collections::HashMap;

/// Trait for publishing and reading per-device load across cooperating processes.
pub trait LoadCoordinator: Send + Sync {
    /// Publish this process's per-device load.
    fn update_load(&self, device_load: &HashMap<DeviceId, usize>) -> std::io::Result<()>;

    /// Read the aggregate load from all cooperating processes.
    fn get_global_load(&self) -> std::io::Result<HashMap<DeviceId, usize>>;

    /// Release resources owned by this coordinator (e.g. lock files, shared memory slots).
    fn cleanup(&self);

    /// Register this process with the coordinator (no-op by default).
    fn register(&self) -> std::io::Result<()> {
        Ok(())
    }

    /// Send a heartbeat to indicate this process is still alive (no-op by default).
    fn heartbeat(&self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Selects the inter-process coordination mechanism.
pub enum CoordinatorType {
    /// File-system based coordination (one `.load` file per process).
    File { path: std::path::PathBuf },
    /// POSIX shared-memory based coordination.
    SharedMemory { name: String, size: usize },
}

/// Create a coordinator of the requested type.
pub fn create_coordinator(
    coord_type: CoordinatorType,
) -> std::io::Result<Box<dyn LoadCoordinator>> {
    match coord_type {
        CoordinatorType::File { path } => Ok(Box::new(FileCoordinator::new(path))),
        CoordinatorType::SharedMemory { name, size } => {
            Ok(Box::new(ShmCoordinator::new(&name, size)?))
        }
    }
}

/// Auto-detects an appropriate coordinator based on environment variables.
///
/// Checks `CV_RUNTIME_COORDINATOR` (for file-based coordination) and `CV_RUNTIME_SHM`
/// (for shared-memory coordination), in that order.
///
/// # Important
/// This function creates a new coordinator instance each time it's called.
/// To avoid creating multiple coordinators for the same resource, callers
/// **MUST** cache the result using `OnceLock` or similar synchronization primitive.
/// Creating duplicate coordinators for the same resource can lead to
/// resource leaks and stale state.
pub fn auto_detect_coordinator() -> Option<Box<dyn LoadCoordinator>> {
    if let Ok(path) = std::env::var("CV_RUNTIME_COORDINATOR") {
        let path_buf = std::path::PathBuf::from(path);
        Some(Box::new(FileCoordinator::new(path_buf)))
    } else if let Ok(name) = std::env::var("CV_RUNTIME_SHM") {
        ShmCoordinator::new(&name, 4096)
            .ok()
            .map(|c| Box::new(c) as Box<dyn LoadCoordinator>)
    } else {
        None
    }
}
