use std::any::Any;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enumeration of supported compute devices.
pub enum DeviceType {
    /// CPU (system RAM)
    Cpu,
    /// NVIDIA CUDA GPU
    Cuda,
    /// Vulkan GPU
    Vulkan,
    /// Apple Metal GPU
    Metal,
    /// Windows DirectML GPU
    Dml,
    /// NVIDIA TensorRT
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
    fn new(size: usize, default_value: T) -> std::result::Result<Self, String>
    where
        T: Clone;

    /// Creates a new storage from a vector.
    fn from_vec(data: Vec<T>) -> std::result::Result<Self, String>;

    /// Returns a reference to the storage as a dynamically typed object for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Returns a mutable reference to the storage as a dynamically typed object for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Converts this boxed storage into a boxed dynamically typed object for downcasting.
    fn boxed_any(self: Box<Self>) -> Box<dyn Any>;
}

/// Standard CPU-based storage using `Vec<T>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuStorage<T> {
    /// The vector containing the actual data on the CPU.
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

#[cfg(test)]
#[allow(missing_docs)]
mod tests {
    use super::*;

    #[test]
    fn cpu_storage_new_correct_size_and_default() {
        let s = CpuStorage::<f32>::new(10, 0.0).unwrap();
        assert_eq!(s.len(), 10);
        assert_eq!(s.device(), DeviceType::Cpu);
        assert!(s.as_slice().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn cpu_storage_from_vec_preserves_data() {
        let data = vec![1.0f32, 2.0, 3.0];
        let s = CpuStorage::from_vec(data.clone()).unwrap();
        assert_eq!(s.as_slice().unwrap(), data.as_slice());
    }

    #[test]
    fn cpu_storage_mut_slice_modifies_data() {
        let mut s = CpuStorage::<i32>::new(3, 0).unwrap();
        s.as_mut_slice().unwrap()[1] = 42;
        assert_eq!(s.as_slice().unwrap()[1], 42);
    }

    #[test]
    fn cpu_storage_empty_is_valid() {
        let s = CpuStorage::<u8>::from_vec(vec![]).unwrap();
        assert_eq!(s.len(), 0);
        assert_eq!(s.as_slice().unwrap().len(), 0);
    }
}
