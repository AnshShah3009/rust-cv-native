use std::sync::{Arc, Mutex};
use crate::Result;
use cv_hal::DeviceId;
use wgpu::BufferUsages;
use crate::device_registry::{SubmissionIndex, registry};

/// Manages data consistency between host and device.
/// 
/// Uses a versioned state machine to track where the most up-to-date data is.
pub struct UnifiedBuffer<T> {
    host_data: Arc<Mutex<Vec<T>>>,
    device_data: Option<wgpu::Buffer>,
    device_id: Option<DeviceId>,
    
    // Coherence tracking
    host_version: u64,
    device_version: Option<(DeviceId, u64)>,
    
    // Submission tracking for safe retirement
    last_write_submission: SubmissionIndex,
    last_read_submission: SubmissionIndex,
    
    len: usize,
}

impl<T: bytemuck::Pod + Clone + Default + Send + 'static + std::fmt::Debug> UnifiedBuffer<T> {
    pub fn new(len: usize) -> Self {
        Self {
            host_data: Arc::new(Mutex::new(vec![T::default(); len])),
            device_data: None,
            device_id: None,
            host_version: 1,
            device_version: None,
            last_write_submission: SubmissionIndex(0),
            last_read_submission: SubmissionIndex(0),
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn host_view(&self) -> Result<std::sync::MutexGuard<'_, Vec<T>>> {
        // Ensure host is up to date
        if let Some((_, dv)) = self.device_version {
            if dv > self.host_version {
                return Err(crate::Error::MemoryError("Buffer data only on device. Call sync_to_host() first.".to_string()));
            }
        }
        
        self.host_data.lock()
            .map_err(|_| crate::Error::ConcurrencyError("UnifiedBuffer host_data poisoned".to_string()))
    }

    /// Mark that the host data has been modified.
    pub fn mark_host_dirty(&mut self) {
        self.host_version += 1;
    }

    pub fn sync_to_device(&mut self, target_id: DeviceId) -> Result<()> {
        if let Some((did, dv)) = self.device_version {
            if did == target_id && dv >= self.host_version {
                return Ok(());
            }
        }

        let reg = registry()?;
        let runtime = reg.get_device(target_id)
            .ok_or_else(|| crate::Error::RuntimeError(format!("Device {:?} not found", target_id)))?;

        let host_guard = self.host_data.lock()
            .map_err(|_| crate::Error::ConcurrencyError("UnifiedBuffer host_data poisoned".to_string()))?;

        let size = (self.len * std::mem::size_of::<T>()) as u64;
        let usages = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        
        let buffer = if let Some(ref b) = self.device_data {
            if b.size() >= size && self.device_id == Some(target_id) {
                b
            } else {
                // Retire old buffer if any
                if let Some(old_b) = self.device_data.take() {
                    if let Some(old_id) = self.device_id {
                        if let Some(old_runtime) = reg.get_device(old_id) {
                            old_runtime.memory().retire_buffer(old_b, self.last_write_submission.max(self.last_read_submission));
                        }
                    }
                }
                
                let device = match runtime.context() {
                    crate::device_registry::BackendContext::Gpu(ctx) => ctx.device(),
                    _ => return Err(crate::Error::NotSupported("sync_to_device only supports GPU for now".into())),
                };

                let new_b = runtime.memory().get_buffer(device, size, usages);
                self.device_data = Some(new_b);
                self.device_id = Some(target_id);
                self.device_data.as_ref().unwrap()
            }
        } else {
            let device = match runtime.context() {
                crate::device_registry::BackendContext::Gpu(ctx) => ctx.device(),
                _ => return Err(crate::Error::NotSupported("sync_to_device only supports GPU for now".into())),
            };
            
            let new_b = runtime.memory().get_buffer(device, size, usages);
            self.device_data = Some(new_b);
            self.device_id = Some(target_id);
            self.device_data.as_ref().unwrap()
        };

        match runtime.context() {
            crate::device_registry::BackendContext::Gpu(gpu_ctx) => {
                gpu_ctx.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&host_guard));
            }
            _ => return Err(crate::Error::NotSupported(format!("sync_to_device not implemented for {:?}", runtime.backend()))),
        }
        
        self.device_version = Some((target_id, self.host_version));
        Ok(())
    }

    pub async fn sync_to_host(&mut self) -> Result<()> {
        let (did, dv) = self.device_version.ok_or_else(|| crate::Error::MemoryError("No device data to sync from".to_string()))?;
        
        if self.host_version >= dv {
            return Ok(());
        }

        let reg = registry()?;
        let runtime = reg.get_device(did)
            .ok_or_else(|| crate::Error::RuntimeError(format!("Device {:?} not found", did)))?;

        let buffer = self.device_data.as_ref()
            .ok_or_else(|| crate::Error::MemoryError("No device buffer to sync from".to_string()))?;

        let size = self.len * std::mem::size_of::<T>();
        
        let data: Vec<T> = match runtime.context() {
            crate::device_registry::BackendContext::Gpu(gpu_ctx) => {
                cv_hal::gpu_kernels::buffer_utils::read_buffer(
                    gpu_ctx.device.clone(),
                    &gpu_ctx.queue,
                    buffer,
                    0,
                    size,
                ).await?
            }
            _ => return Err(crate::Error::NotSupported(format!("sync_to_host not implemented for {:?}", runtime.backend()))),
        };

        let mut host_guard = self.host_data.lock()
            .map_err(|_| crate::Error::ConcurrencyError("UnifiedBuffer host_data poisoned".to_string()))?;
        *host_guard = data;
        
        self.host_version = dv;
        Ok(())
    }

    /// Record that a GPU operation has written to this buffer.
    pub fn mark_device_write(&mut self, index: SubmissionIndex) {
        self.last_write_submission = index;
        if let Some((did, _)) = self.device_version {
            let next_v = self.host_version.max(self.device_version.map(|v| v.1).unwrap_or(0)) + 1;
            self.device_version = Some((did, next_v));
        }
    }

    /// Record that a GPU operation has read from this buffer.
    pub fn mark_device_read(&mut self, index: SubmissionIndex) {
        self.last_read_submission = index;
    }

    pub fn device_buffer(&self) -> Option<&wgpu::Buffer> {
        self.device_data.as_ref()
    }

    pub fn device_id(&self) -> Option<DeviceId> {
        self.device_id
    }
}

impl<T> Drop for UnifiedBuffer<T> {
    fn drop(&mut self) {
        if let (Some(buffer), Some(id)) = (self.device_data.take(), self.device_id.take()) {
            if let Ok(reg) = registry() {
                if let Some(runtime) = reg.get_device(id) {
                    runtime.memory().retire_buffer(buffer, self.last_write_submission.max(self.last_read_submission));
                }
            }
        }
    }
}
