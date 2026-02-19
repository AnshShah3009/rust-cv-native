use std::fmt::Debug;
use std::any::Any;

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
pub trait Storage<T: 'static>: Debug + Clone + Any {
    /// The device where this storage resides.
    fn device(&self) -> DeviceType;

    /// The number of elements in this storage.
    fn len(&self) -> usize;

    /// Get the data as a slice, if possible.
    /// Returns `None` if the data is not on CPU or not contiguous.
    fn as_slice(&self) -> Option<&[T]>;

    /// Get the data as a mutable slice, if possible.
    /// Returns `None` if the data is not on CPU or not contiguous.
    fn as_mut_slice(&mut self) -> Option<&mut [T]>;

    /// Creates a new storage with the given size and default value.
    fn new(size: usize, default_value: T) -> std::result::Result<Self, String> where T: Clone;

    /// Creates a new storage from a vector.
    fn from_vec(data: Vec<T>) -> std::result::Result<Self, String>;

    /// NEW: For safe downcasting
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn boxed_any(self: Box<Self>) -> Box<dyn Any>;
}

/// Standard CPU-based storage using `Vec<T>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuStorage<T> {
    pub data: Vec<T>,
}

impl<T: Clone + Debug + Any + 'static> Storage<T> for CpuStorage<T> {
    fn device(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn as_slice(&self) -> Option<&[T]> {
        Some(&self.data)
    }

    fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        Some(&mut self.data)
    }

    fn new(size: usize, default_value: T) -> std::result::Result<Self, String> {
        Ok(Self {
            data: vec![default_value; size],
        })
    }

    fn from_vec(data: Vec<T>) -> std::result::Result<Self, String> {
        Ok(Self { data })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn boxed_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}
