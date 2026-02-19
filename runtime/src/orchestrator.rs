use rayon;
use std::sync::{Arc, OnceLock, Mutex};
use std::collections::HashMap;
use crate::Result;
use std::sync::atomic::{AtomicUsize, Ordering};
use cv_hal::compute::{ComputeDevice, get_device};

/// Policy for a resource group
#[derive(Debug, Clone, Copy)]
pub struct GroupPolicy {
    /// If true, this group uses the global thread pool (work stealing enabled)
    pub allow_work_stealing: bool,
    /// If true, the pool can be resized at runtime
    pub allow_dynamic_scaling: bool,
}

impl Default for GroupPolicy {
    fn default() -> Self {
        Self {
            allow_work_stealing: true,
            allow_dynamic_scaling: true,
        }
    }
}

#[derive(Debug)]
pub struct ResourceGroup {
    pub name: String,
    pub policy: GroupPolicy,
    pub num_threads: usize,
    device: ComputeDevice<'static>,
    pool: Arc<rayon::ThreadPool>,
    active_tasks: Arc<AtomicUsize>,
}

impl ResourceGroup {
    pub fn new(_scheduler: &TaskScheduler, name: &str, num_threads: usize, core_ids: Option<Vec<usize>>, policy: GroupPolicy, device: Option<ComputeDevice<'static>>) -> Result<Self> {
        let mut builder = rayon::ThreadPoolBuilder::new().num_threads(num_threads);
        
        if let Some(cores) = core_ids {
            if !cores.is_empty() {
                builder = builder.start_handler(move |thread_idx| {
                    if let Some(&core_id) = cores.get(thread_idx % cores.len()) {
                        if let Some(system_cores) = core_affinity::get_core_ids() {
                            if let Some(target_core) = system_cores.into_iter().find(|c| c.id == core_id) {
                                core_affinity::set_for_current(target_core);
                            }
                        }
                    }
                });
            }
        }

        let pool = Arc::new(builder.build().map_err(|e| crate::Error::RuntimeError(e.to_string()))?);

        Ok(Self {
            name: name.to_string(),
            policy,
            num_threads,
            device: device.unwrap_or_else(get_device),
            pool,
            active_tasks: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn resize(&self, _new_threads: usize) -> Result<()> {
        Err(crate::Error::RuntimeError("Resize not supported in unified architecture".to_string()))
    }

    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let active_tasks_clone = self.active_tasks.clone();
        self.active_tasks.fetch_add(1, Ordering::SeqCst);
        
        self.pool.spawn(move || {
            // Drop guard guarantees decrement even on panic
            struct TaskGuard(Arc<AtomicUsize>);
            impl Drop for TaskGuard {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            let _guard = TaskGuard(active_tasks_clone);
            f();
        });
    }

    /// Execute a parallel job, respecting the group's concurrency limit
    pub fn run<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.active_tasks.fetch_add(1, Ordering::SeqCst);
        let result = self.pool.install(f);
        self.active_tasks.fetch_sub(1, Ordering::SeqCst);
        result
    }

    pub fn current_num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn device(&self) -> ComputeDevice<'static> {
        self.device
    }

    pub fn load(&self) -> usize {
        self.active_tasks.load(Ordering::Relaxed)
    }
}

pub struct TaskScheduler {
    groups: Mutex<HashMap<String, Arc<ResourceGroup>>>,
    total_threads: AtomicUsize,
}

impl TaskScheduler {
    pub fn new() -> Self {
        Self {
            groups: Mutex::new(HashMap::new()),
            total_threads: AtomicUsize::new(0),
        }
    }

    pub fn create_group(&self, name: &str, num_threads: usize, cores: Option<Vec<usize>>, policy: GroupPolicy) -> Result<Arc<ResourceGroup>> {
        self.create_group_with_device(name, num_threads, cores, policy, None)
    }

    pub fn create_group_with_device(&self, name: &str, num_threads: usize, cores: Option<Vec<usize>>, policy: GroupPolicy, device: Option<ComputeDevice<'static>>) -> Result<Arc<ResourceGroup>> {
        let mut groups = match self.groups.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
            
        if groups.contains_key(name) {
            return Err(crate::Error::RuntimeError(format!("Resource group '{}' already exists", name)));
        }

        let group = Arc::new(ResourceGroup::new(self, name, num_threads, cores, policy, device)?);
        groups.insert(name.to_string(), group.clone());
        
        self.total_threads.fetch_add(num_threads, Ordering::SeqCst);
        
        Ok(group)
    }

    pub fn remove_group(&self, name: &str) -> Result<Option<Arc<ResourceGroup>>> {
        let mut groups = match self.groups.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
            
        if let Some(group) = groups.remove(name) {
            self.total_threads.fetch_sub(group.current_num_threads(), Ordering::SeqCst);
            Ok(Some(group))
        } else {
            Ok(None)
        }
    }

    pub fn total_allocated_threads(&self) -> usize {
        self.total_threads.load(Ordering::Relaxed)
    }

    pub fn get_group(&self, name: &str) -> Result<Option<Arc<ResourceGroup>>> {
        let groups = match self.groups.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        Ok(groups.get(name).cloned())
    }

    /// Finds the best available resource group for a given device type.
    /// Prefers groups with the least active tasks.
    pub fn get_best_group(&self, backend_type: cv_hal::BackendType) -> Result<Option<Arc<ResourceGroup>>> {
        let groups = match self.groups.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        
        let mut best_group: Option<Arc<ResourceGroup>> = None;
        let mut min_load = usize::MAX;

        for group in groups.values() {
            let device = group.device();
            let matches = match (backend_type, device) {
                (cv_hal::BackendType::Cpu, ComputeDevice::Cpu(_)) => true,
                // Any GPU backend matches a GPU device for now
                (t, ComputeDevice::Gpu(_)) if t != cv_hal::BackendType::Cpu => true,
                _ => false,
            };

            if matches {
                let load = group.load();
                if load < min_load {
                    min_load = load;
                    best_group = Some(group.clone());
                }
            }
        }

        Ok(best_group)
    }

    pub fn get_default_group(&self) -> Arc<ResourceGroup> {
        self.get_group("default").unwrap().expect("default group must exist")
    }

    pub fn best_gpu_or_cpu(&self) -> Arc<ResourceGroup> {
        // Try WebGPU first (as it's our primary accelerator)
        self.get_best_group(cv_hal::BackendType::WebGPU).unwrap()
            .or_else(|| self.get_best_group(cv_hal::BackendType::Vulkan).unwrap())
            .unwrap_or_else(|| self.get_default_group())
    }

    /// Convenience method to get the best device directly.
    pub fn best_device(&self) -> ComputeDevice<'static> {
        self.best_gpu_or_cpu().device()
    }

    pub fn submit<F>(&self, group_name: &str, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(group) = self.get_group(group_name)? {
            group.spawn(f);
            Ok(())
        } else {
            Err(crate::Error::RuntimeError(format!("Resource group '{}' not found", group_name)))
        }
    }
}

static GLOBAL_SCHEDULER: OnceLock<TaskScheduler> = OnceLock::new();

pub fn scheduler() -> &'static TaskScheduler {
    GLOBAL_SCHEDULER.get_or_init(|| {
        let _ = cv_core::init_global_thread_pool(None);
        let s = TaskScheduler::new();
        s.create_group("default", num_cpus::get(), None, GroupPolicy::default()).unwrap();
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_unified_concurrency() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy::default();
        
        let g1 = s.create_group("g1", 4, None, policy).unwrap();
        let g2 = s.create_group("g2", 2, None, policy).unwrap();
        
        assert_eq!(g1.current_num_threads(), 4);
        assert_eq!(g2.current_num_threads(), 2);
        assert_eq!(s.total_allocated_threads(), 6);
    }

    #[test]
    fn test_duplicate_group_error() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy::default();
        
        s.create_group("same", 2, None, policy).unwrap();
        let res = s.create_group("same", 2, None, policy);
        
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_submit_logic() {
        let s = TaskScheduler::new();
        s.create_group("worker", 2, None, GroupPolicy::default()).unwrap();
        
        let (tx, rx) = std::sync::mpsc::channel();
        s.submit("worker", move || {
            tx.send(42).unwrap();
        }).unwrap();
        
        assert_eq!(rx.recv_timeout(Duration::from_secs(1)).unwrap(), 42);
    }

    #[test]
    fn test_load_aware_steering() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy::default();
        
        let cpu_dev = cv_hal::compute::get_device();
        let g1 = s.create_group_with_device("g1", 2, None, policy, Some(cpu_dev)).unwrap();
        let g2 = s.create_group_with_device("g2", 2, None, policy, Some(cpu_dev)).unwrap();
        
        let backend_type = cpu_dev.backend_type();
        
        assert_eq!(g1.load(), 0);
        assert_eq!(g2.load(), 0);
        
        let (tx, rx) = std::sync::mpsc::channel();
        let g1_cloned = g1.clone();
        std::thread::spawn(move || {
            g1_cloned.run(|| {
                tx.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(200));
            });
        });
        
        rx.recv().unwrap();
        assert_eq!(g1.load(), 1);
        
        let best = s.get_best_group(backend_type).unwrap().unwrap();
        assert_eq!(best.name, "g2");
    }
}


