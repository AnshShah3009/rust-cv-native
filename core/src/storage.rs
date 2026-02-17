use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
    Dml,
    TensorRT,
}

/// A trait for tensor storage backends.
///
/// This allows a `Tensor` to hold data on different devices (CPU, GPU, etc.)
/// while providing a common interface for metadata and (where applicable) access.
pub trait Storage<T>: Debug + Clone {
    /// The device where this storage resides.
    fn device(&self) -> DeviceType;

    /// Get the data as a slice, if possible.
    /// Returns `None` if the data is not on CPU or not contiguous.
    fn as_slice(&self) -> Option<&[T]>;

    /// Get the data as a mutable slice, if possible.
    /// Returns `None` if the data is not on CPU or not contiguous.
    fn as_mut_slice(&mut self) -> Option<&mut [T]>;

    /// Creates a new storage with the given size and default value.
    fn new(size: usize, default_value: T) -> Self where T: Clone;

    /// Creates a new storage from a vector.
    fn from_vec(data: Vec<T>) -> Self;
}

/// Standard CPU-based storage using `Vec<T>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuStorage<T> {
    pub data: Vec<T>,
}

impl<T: Clone + Debug> Storage<T> for CpuStorage<T> {
    fn device(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn as_slice(&self) -> Option<&[T]> {
        Some(&self.data)
    }

    fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        Some(&mut self.data)
    }

    fn new(size: usize, default_value: T) -> Self {
        Self {
            data: vec![default_value; size],
        }
    }

    fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }
}
