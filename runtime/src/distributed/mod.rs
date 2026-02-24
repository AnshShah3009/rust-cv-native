mod file;
mod shared_memory;

pub use file::FileCoordinator;
pub use shared_memory::ShmCoordinator;

use cv_hal::DeviceId;
use std::collections::HashMap;

pub trait LoadCoordinator: Send + Sync {
    fn update_load(&self, device_load: &HashMap<DeviceId, usize>) -> std::io::Result<()>;

    fn get_global_load(&self) -> std::io::Result<HashMap<DeviceId, usize>>;

    fn cleanup(&self);

    fn register(&self) -> std::io::Result<()> {
        Ok(())
    }

    fn heartbeat(&self) -> std::io::Result<()> {
        Ok(())
    }
}

pub enum CoordinatorType {
    File { path: std::path::PathBuf },
    SharedMemory { name: String, size: usize },
}

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
