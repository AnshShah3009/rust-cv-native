use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use cv_hal::DeviceId;
use crate::Result;

/// A device-bound execution engine.
/// 
/// Each Executor owns a private thread pool and is associated with a specific compute device.
#[derive(Debug)]
pub struct Executor {
    device_id: DeviceId,
    pool: rayon::ThreadPool,
    inflight_jobs: Arc<AtomicUsize>,
}

impl Executor {
    pub fn new(device_id: DeviceId, num_threads: usize, name: &str) -> Result<Self> {
        let name_clone = name.to_string();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |idx| format!("{}-{}", name_clone, idx))
            .build()
            .map_err(|e| crate::Error::RuntimeError(format!("Failed to build thread pool: {}", e)))?;
            
        Ok(Self {
            device_id,
            pool,
            inflight_jobs: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let inflight = self.inflight_jobs.clone();
        inflight.fetch_add(1, Ordering::SeqCst);
        
        self.pool.spawn(move || {
            struct JobGuard(Arc<AtomicUsize>);
            impl Drop for JobGuard {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            let _guard = JobGuard(inflight);
            f();
        });
    }

    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.inflight_jobs.fetch_add(1, Ordering::SeqCst);
        let result = self.pool.install(f);
        self.inflight_jobs.fetch_sub(1, Ordering::SeqCst);
        result
    }

    pub fn load(&self) -> usize {
        self.inflight_jobs.load(Ordering::Relaxed)
    }

    /// Spawn an asynchronous task.
    pub fn spawn_async<F, Fut>(&self, f: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let inflight = self.inflight_jobs.clone();
        inflight.fetch_add(1, Ordering::SeqCst);
        
        tokio::spawn(async move {
            struct JobGuard(Arc<AtomicUsize>);
            impl Drop for JobGuard {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            let _guard = JobGuard(inflight);
            f().await;
        });
    }
}

/// A pool of executors for a device, allowing multiple QoS levels.
pub struct ExecutorPool {
    device_id: DeviceId,
    executors: Vec<Arc<Executor>>,
}

impl ExecutorPool {
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            device_id,
            executors: Vec::new(),
        }
    }

    pub fn add_executor(&mut self, executor: Arc<Executor>) {
        assert_eq!(executor.device_id(), self.device_id);
        self.executors.push(executor);
    }

    pub fn best_executor(&self) -> Option<Arc<Executor>> {
        self.executors.iter()
            .min_by_key(|e| e.load())
            .cloned()
    }
}
