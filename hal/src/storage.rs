use cv_core::storage::{Storage, DeviceType};
use crate::gpu::GpuContext;
use wgpu::util::DeviceExt;
use std::sync::Arc;
use std::marker::PhantomData;
use std::fmt;

/// GPU storage using wgpu buffers.
///
/// This storage holds data on the GPU. Accessing it as a slice on the CPU is not supported directly.
/// Use `to_cpu()` to download data (not part of Storage trait).
#[derive(Clone)]
pub struct GpuStorage<T> {
    pub buffer: Arc<wgpu::Buffer>,
    pub len: usize,
    _phantom: PhantomData<T>,
}

impl<T> GpuStorage<T> {
    pub fn from_buffer(buffer: Arc<wgpu::Buffer>, len: usize) -> Self {
        Self {
            buffer,
            len,
            _phantom: PhantomData,
        }
    }
}

impl<T: bytemuck::Pod + fmt::Debug> Storage<T> for GpuStorage<T> {
    fn device(&self) -> DeviceType {
        // We assume Vulkan or Metal based on platform, but for now just say "Vulkan" or generic "GPU"
        // core::DeviceType has Cuda, Vulkan, Metal.
        // We can't easily know specific backend from Buffer alone without context.
        DeviceType::Vulkan 
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
        
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuStorage Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Note: We don't initialize with default value efficiently here because
        // we'd need to upload it. For now, uninitialized or zeroed?
        // wgpu buffers are zero-initialized by default on creation if not mapped.
        // So default_value is ignored! (Limitation)
        
        Self {
            buffer: Arc::new(buffer),
            len: size,
            _phantom: PhantomData,
        }
    }

    fn from_vec(data: Vec<T>) -> Self {
        let ctx = GpuContext::global().expect("GPU not available for GpuStorage::from_vec");
        
        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuStorage From Vec"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        Self {
            buffer: Arc::new(buffer),
            len: data.len(),
            _phantom: PhantomData,
        }
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
        Arc::ptr_eq(&self.buffer, &other.buffer) && self.len == other.len
    }
}

impl<T> Eq for GpuStorage<T> {}
