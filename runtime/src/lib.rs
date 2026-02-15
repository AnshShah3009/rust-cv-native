pub mod orchestrator;
pub mod memory;

pub use orchestrator::{TaskScheduler, TaskPriority, ResourceGroup, scheduler};
pub use memory::{UnifiedBuffer, BufferLocation};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Runtime error: {0}")]
    RuntimeError(String),
    
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    #[error("HAL error: {0}")]
    HalError(#[from] cv_hal::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! submit_to {
    ($group_name:expr, $f:block) => {
        if let Some(group) = $crate::scheduler().get_group($group_name) {
            group.spawn(move || { $f });
        }
    };
}
