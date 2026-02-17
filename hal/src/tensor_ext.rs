use cv_core::{Tensor, CpuStorage, TensorShape, DataType};
use crate::storage::GpuStorage;
use crate::gpu::GpuContext;
use crate::gpu_kernels::buffer_utils;
use wgpu::util::DeviceExt;
use std::sync::Arc;
use std::marker::PhantomData;

/// Extension trait for transferring data to the GPU.
pub trait TensorToGpu<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug> {
    /// Uploads the tensor to the GPU.
    fn to_gpu(&self) -> crate::Result<Tensor<T, GpuStorage<T>>>;
}

impl<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug> TensorToGpu<T> for Tensor<T, CpuStorage<T>> {
    fn to_gpu(&self) -> crate::Result<Tensor<T, GpuStorage<T>>> {
        let ctx = GpuContext::global().ok_or_else(|| {
            crate::Error::InitError("GPU context not initialized".into())
        })?;

        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tensor Upload"),
            contents: bytemuck::cast_slice(self.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        Ok(Tensor {
            storage: GpuStorage::from_buffer(Arc::new(buffer), self.shape.len()),
            shape: self.shape,
            dtype: self.dtype,
            _phantom: PhantomData,
        })
    }
}

/// Extension trait for transferring data back to the CPU.
pub trait TensorToCpu<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug> {
    /// Downloads the tensor from the GPU.
    fn to_cpu(&self) -> crate::Result<Tensor<T, CpuStorage<T>>>;
}

impl<T: Clone + Copy + bytemuck::Pod + std::fmt::Debug> TensorToCpu<T> for Tensor<T, GpuStorage<T>> {
    fn to_cpu(&self) -> crate::Result<Tensor<T, CpuStorage<T>>> {
        let ctx = GpuContext::global().ok_or_else(|| {
            crate::Error::InitError("GPU context not initialized".into())
        })?;

        // We use the synchronous readback for now to match the blocking API.
        // buffer_utils::read_buffer is async, so we block on it.
        
        let byte_size = self.storage.len * std::mem::size_of::<T>();
        
        let data: Vec<T> = pollster::block_on(buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &self.storage.buffer,
            0,
            byte_size,
        ))?;

        Ok(Tensor::from_vec(data, self.shape))
    }
}
