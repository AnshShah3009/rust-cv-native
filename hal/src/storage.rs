use cv_core::storage::{Storage, DeviceType};
use crate::gpu::GpuContext;
use wgpu::util::DeviceExt;
use std::sync::Arc;
use std::marker::PhantomData;
use std::fmt;
use std::any::Any;

/// GPU storage using wgpu buffers.
///
/// This storage holds data on the GPU. Accessing it as a slice on the CPU is not supported directly.
/// Use `to_cpu()` to download data (not part of Storage trait).
#[derive(Clone)]
pub struct GpuStorage<T> {
    pub buffer: Option<Arc<wgpu::Buffer>>,
    pub len: usize,
    pub usage: wgpu::BufferUsages,
    _phantom: PhantomData<T>,
}

impl<T> GpuStorage<T> {
    pub fn from_buffer(buffer: Arc<wgpu::Buffer>, len: usize) -> Self {
        Self::from_buffer_with_usage(buffer, len, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST)
    }

    pub fn from_buffer_with_usage(buffer: Arc<wgpu::Buffer>, len: usize, usage: wgpu::BufferUsages) -> Self {
        Self {
            buffer: Some(buffer),
            len,
            usage,
            _phantom: PhantomData,
        }
    }

    /// Access the underlying buffer. Panics if the buffer has been dropped (which should not happen during normal use).
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("GpuStorage buffer accessed after drop")
    }
}

impl<T> Drop for GpuStorage<T> {
    fn drop(&mut self) {
        if let Some(arc_buf) = self.buffer.take() {
            if let Ok(buffer) = Arc::try_unwrap(arc_buf) {
                if let Ok(ctx) = GpuContext::global() {
                    ctx.return_buffer(buffer, self.usage);
                }
            }
        }
    }
}

impl<T: bytemuck::Pod + fmt::Debug + Any> Storage<T> for GpuStorage<T> {
    fn device(&self) -> DeviceType {
        DeviceType::Vulkan 
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> Option<&[T]> {
        None
    }

    fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }

    fn new(size: usize, _default_value: T) -> Self where T: Clone {
        let ctx = GpuContext::global().expect("GPU not available for GpuStorage::new");
        let byte_size = (size * std::mem::size_of::<T>()) as u64;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        
        let buffer = ctx.get_buffer(byte_size, usage);

        Self {
            buffer: Some(Arc::new(buffer)),
            len: size,
            usage,
            _phantom: PhantomData,
        }
    }

    fn from_vec(data: Vec<T>) -> Self {
        let ctx = GpuContext::global().expect("GPU not available for GpuStorage::from_vec");
        let byte_size = (data.len() * std::mem::size_of::<T>()) as u64;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        
        let buffer = ctx.get_buffer(byte_size, usage);
        ctx.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data));

        Self {
            buffer: Some(Arc::new(buffer)),
            len: data.len(),
            usage,
            _phantom: PhantomData,
        }
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

impl<T> fmt::Debug for GpuStorage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuStorage(len={})", self.len)
    }
}

impl<T> PartialEq for GpuStorage<T> {
    fn eq(&self, other: &Self) -> bool {
        // Pointer equality for buffer
        match (&self.buffer, &other.buffer) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b) && self.len == other.len,
            (None, None) => self.len == other.len,
            _ => false,
        }
    }
}

impl<T> Eq for GpuStorage<T> {}
