pub mod orchestrator;
pub mod memory;

pub use orchestrator::{TaskScheduler, ResourceGroup, GroupPolicy, scheduler};
pub use memory::{UnifiedBuffer, BufferLocation};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Runtime error: {0}")]
    RuntimeError(String),
    
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),
    
    #[error("HAL error: {0}")]
    HalError(#[from] cv_hal::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! submit_to {
    ($group_name:expr, $f:block) => {
        $crate::scheduler().submit($group_name, move || { $f })
    };
}
