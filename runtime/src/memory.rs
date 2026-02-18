use std::sync::{Arc, Mutex};
use crate::Result;
use cv_hal::gpu::GpuContext;
use cv_hal::compute::ComputeDevice;
use wgpu::BufferUsages;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLocation {
    Host,
    Device,
    Both,
}

pub struct UnifiedBuffer<T> {
    host_data: Arc<Mutex<Vec<T>>>,
    device_data: Option<wgpu::Buffer>,
    location: BufferLocation,
    len: usize,
}

impl<T: bytemuck::Pod + Clone + Default + Send + 'static + std::fmt::Debug> UnifiedBuffer<T> {
    pub fn new(len: usize) -> Self {
        Self {
            host_data: Arc::new(Mutex::new(vec![T::default(); len])),
            device_data: None,
            location: BufferLocation::Host,
            len,
        }
    }

    pub fn host_view(&self) -> Result<std::sync::MutexGuard<'_, Vec<T>>> {
        if self.location == BufferLocation::Device {
            return Err(crate::Error::MemoryError("Buffer data only on device. Call sync_to_host() first.".to_string()));
        }
        self.host_data.lock()
            .map_err(|_| crate::Error::ConcurrencyError("UnifiedBuffer host_data poisoned".to_string()))
    }

    pub fn sync_to_device_ctx(&mut self, ctx: &GpuContext) -> Result<()> {
        self.sync_to_device(&ComputeDevice::Gpu(ctx))
    }

    pub fn sync_to_device(&mut self, device: &ComputeDevice) -> Result<()> {
        if self.location == BufferLocation::Device || self.location == BufferLocation::Both {
            return Ok(());
        }

        let host_guard = self.host_data.lock()
            .map_err(|_| crate::Error::ConcurrencyError("UnifiedBuffer host_data poisoned".to_string()))?;

        let size = (self.len * std::mem::size_of::<T>()) as u64;
        let usages = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        
        let buffer = if let Some(ref b) = self.device_data {
            if b.size() >= size {
                b
            } else {
                // Return old buffer to pool if it was pooled?
                // For now just create new one from pool
                let new_b = device.get_buffer(size, usages)?;
                self.device_data = Some(new_b);
                self.device_data.as_ref().unwrap()
            }
        } else {
            let new_b = device.get_buffer(size, usages)?;
            self.device_data = Some(new_b);
            self.device_data.as_ref().unwrap()
        };

        if let ComputeDevice::Gpu(gpu) = device {
            gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&host_guard));
        }
        
        self.location = BufferLocation::Both;
        Ok(())
    }

    pub async fn sync_to_host_ctx(&mut self, ctx: &GpuContext) -> Result<()> {
        self.sync_to_host(&ComputeDevice::Gpu(ctx)).await
    }

    pub async fn sync_to_host(&mut self, device: &ComputeDevice) -> Result<()> {
        if self.location == BufferLocation::Host || self.location == BufferLocation::Both {
            return Ok(());
        }

        let buffer = self.device_data.as_ref()
            .ok_or_else(|| crate::Error::MemoryError("No device buffer to sync from".to_string()))?;

        let size = self.len * std::mem::size_of::<T>();
        
        let data: Vec<T> = match device {
            ComputeDevice::Gpu(gpu) => {
                cv_hal::gpu_kernels::buffer_utils::read_buffer(
                    gpu.device.clone(),
                    &gpu.queue,
                    buffer,
                    0,
                    size,
                ).await?
            }
            ComputeDevice::Cpu(_) => return Err(crate::Error::MemoryError("Cannot sync to host from CPU device".into())),
        };

        let mut host_guard = self.host_data.lock()
            .map_err(|_| crate::Error::ConcurrencyError("UnifiedBuffer host_data poisoned".to_string()))?;
        *host_guard = data;
        
        self.location = BufferLocation::Both;
        Ok(())
    }

    /// Return the GPU buffer to the pool.
    pub fn release(&mut self, device: &ComputeDevice) -> Result<()> {
        if let Some(buffer) = self.device_data.take() {
            let usages = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
            device.return_buffer(buffer, usages)?;
        }
        Ok(())
    }

    pub fn device_buffer(&self) -> Option<&wgpu::Buffer> {
        self.device_data.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_hal::gpu::GpuContext;

    #[test]
    fn test_unified_buffer_basic() {
        let buf = UnifiedBuffer::<f32>::new(10);
        {
            let mut view = buf.host_view().unwrap();
            view[0] = 42.0;
        }
        assert_eq!(buf.location, BufferLocation::Host);
    }

    #[tokio::test]
    async fn test_unified_buffer_sync() {
        // Only run if GPU is available
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut buf = UnifiedBuffer::<f32>::new(100);
        {
            let mut view = buf.host_view().unwrap();
            for i in 0..100 { view[i] = i as f32; }
        }

        // Sync to device
        buf.sync_to_device(&ctx).unwrap();
        assert_eq!(buf.location, BufferLocation::Both);
        assert!(buf.device_buffer().is_some());

        // Modify device data (not easy to do here without a shader, 
        // so we'll just clear host and sync back)
        {
            let mut view = buf.host_view().unwrap();
            for i in 0..100 { view[i] = 0.0; }
        }
        
        buf.location = BufferLocation::Device; // Force it to think it's only on device
        buf.sync_to_host(&ctx).await.unwrap();

        let view = buf.host_view().unwrap();
        assert_eq!(view[50], 50.0);
    }
}
