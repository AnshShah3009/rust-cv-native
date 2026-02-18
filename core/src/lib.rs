pub mod descriptor;
pub mod frames;
pub mod geometry;
pub mod image;
pub mod keypoint;
pub mod point_cloud;
pub mod robust;
pub mod runtime;
pub mod storage;
pub mod tensor;

pub use descriptor::*;
pub use frames::*;
pub use geometry::*;
pub use image::*;
pub use keypoint::*;
pub use point_cloud::*;
pub use robust::*;
pub use runtime::*;
pub use storage::*;
pub use tensor::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Runtime error: {0}")]
    RuntimeError(String),
    
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, Error>;
