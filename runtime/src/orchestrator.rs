use rayon::ThreadPool;
use std::sync::{Arc, OnceLock, RwLock};
use std::collections::HashMap;
use std::sync::Mutex;
use crate::Result;
use core_affinity::CoreId;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Policy for a resource group
#[derive(Debug, Clone, Copy)]
pub struct GroupPolicy {
    /// If true, this group uses the global thread pool (work stealing enabled)
    pub allow_work_stealing: bool,
    /// If true, the pool can be resized at runtime (only for isolated groups)
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

#[derive(Debug, Clone)]
enum PoolBackend {
    Global,
    Private(Arc<ThreadPool>),
}

impl PoolBackend {
    fn current_num_threads(&self) -> usize {
        match self {
            PoolBackend::Global => rayon::current_num_threads(),
            PoolBackend::Private(pool) => pool.current_num_threads(),
        }
    }

    fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        match self {
            PoolBackend::Global => rayon::spawn(f),
            PoolBackend::Private(pool) => pool.spawn(f),
        }
    }

    fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match self {
            // For global pool, we are likely already in it or can just run.
            // But to be safe and ensure we use global pool if called from outside:
            PoolBackend::Global => f(), // Rayon operations default to global pool automatically
            PoolBackend::Private(pool) => pool.install(f),
        }
    }
}

#[derive(Debug)]
pub struct ResourceGroup {
    pub name: String,
    backend: RwLock<PoolBackend>,
    pub cores: Vec<usize>,
    pub policy: GroupPolicy,
}

impl ResourceGroup {
    pub fn new(_scheduler: &TaskScheduler, name: &str, num_threads: usize, core_ids: Option<Vec<usize>>, policy: GroupPolicy) -> Result<Self> {
        let name_str = name.to_string();
        
        if let Some(ref cores) = core_ids {
            if cores.is_empty() {
                return Err(crate::Error::RuntimeError("core_ids cannot be empty if provided".to_string()));
            }
        }

        let backend = if policy.allow_work_stealing {
            // Use Global Pool
            PoolBackend::Global
        } else {
            // Isolated groups get their own pool
            let pool = Self::create_isolated_pool(name, num_threads, &core_ids)?;
            PoolBackend::Private(Arc::new(pool))
        };
            
        Ok(Self {
            name: name_str,
            backend: RwLock::new(backend),
            cores: core_ids.clone().unwrap_or_default(),
            policy,
        })
    }

    fn create_isolated_pool(name: &str, num_threads: usize, core_ids: &Option<Vec<usize>>) -> Result<ThreadPool> {
        let thread_name_prefix = format!("cv-{}-", name);
        let core_ids_cloned = core_ids.clone();
        
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |i| format!("{}{}", thread_name_prefix, i))
            .start_handler(move |i| {
                if let Some(ref cores) = core_ids_cloned {
                    if !cores.is_empty() {
                        if let Some(&core_index) = cores.get(i % cores.len()) {
                            core_affinity::set_for_current(CoreId { id: core_index });
                        }
                    }
                }
            })
            .build()
            .map_err(|e| crate::Error::RuntimeError(e.to_string()))
    }

    pub fn resize(&self, new_threads: usize) -> Result<()> {
        if !self.policy.allow_dynamic_scaling {
            return Err(crate::Error::RuntimeError(format!("Dynamic scaling not allowed for group '{}'", self.name)));
        }

        if self.policy.allow_work_stealing {
            // Cannot resize global pool at runtime easily/safely in this architecture
            return Err(crate::Error::RuntimeError("Cannot resize the global shared pool".to_string()));
        }

        let new_pool = Self::create_isolated_pool(&self.name, new_threads, &Some(self.cores.clone()))?;
        let mut backend_guard = self.backend.write()
            .map_err(|_| crate::Error::ConcurrencyError("backend lock poisoned".to_string()))?;
        *backend_guard = PoolBackend::Private(Arc::new(new_pool));
        
        Ok(())
    }

    /// Access the underlying private pool if it exists.
    /// Returns None if using Global pool.
    pub fn private_pool(&self) -> Option<Arc<ThreadPool>> {
        let guard = self.backend.read().unwrap();
        match &*guard {
            PoolBackend::Private(p) => Some(p.clone()),
            PoolBackend::Global => None,
        }
    }

    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.backend.read().unwrap().spawn(f);
    }

    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.backend.read().unwrap().install(f)
    }
    
    pub fn current_num_threads(&self) -> usize {
        self.backend.read().unwrap().current_num_threads()
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
        let mut groups = self.groups.lock()
            .map_err(|_| crate::Error::ConcurrencyError("groups mutex poisoned".to_string()))?;
            
        if groups.contains_key(name) {
            return Err(crate::Error::RuntimeError(format!("Resource group '{}' already exists", name)));
        }

        let current_total = self.total_threads.load(Ordering::Relaxed);
        let max_cores = num_cpus::get();
        
        // Warn if isolated groups oversubscribe (shared global pool doesn't count towards isolated limit check logic usually, or we assume it takes all cores)
        if !policy.allow_work_stealing && current_total + num_threads > max_cores {
            eprintln!("Warning: Isolated resource group '{}' with {} threads will oversubscribe CPU (total isolated threads: {}/{})", 
                name, num_threads, current_total + num_threads, max_cores);
        }

        let group = Arc::new(ResourceGroup::new(self, name, num_threads, cores, policy)?);
        groups.insert(name.to_string(), group.clone());
        
        if !policy.allow_work_stealing {
            self.total_threads.fetch_add(num_threads, Ordering::SeqCst);
        }
        
        Ok(group)
    }

    pub fn remove_group(&self, name: &str) -> Result<Option<Arc<ResourceGroup>>> {
        let mut groups = self.groups.lock()
            .map_err(|_| crate::Error::ConcurrencyError("groups mutex poisoned".to_string()))?;
            
        if let Some(group) = groups.remove(name) {
            if !group.policy.allow_work_stealing {
                self.total_threads.fetch_sub(group.current_num_threads(), Ordering::SeqCst);
            }
            Ok(Some(group))
        } else {
            Ok(None)
        }
    }

    pub fn total_allocated_threads(&self) -> usize {
        self.total_threads.load(Ordering::Relaxed)
    }

    pub fn get_group(&self, name: &str) -> Result<Option<Arc<ResourceGroup>>> {
        Ok(self.groups.lock()
            .map_err(|_| crate::Error::ConcurrencyError("groups mutex poisoned".to_string()))?
            .get(name).cloned())
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
        // Unify: Initialize the global Rayon pool
        let _ = cv_core::init_global_thread_pool(None);
        
        let s = TaskScheduler::new();
        // Create a default group that maps to the global pool
        s.create_group("default", num_cpus::get(), None, GroupPolicy::default()).unwrap();
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_global_pool_unification() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy {
            allow_work_stealing: true,
            allow_dynamic_scaling: true,
        };
        
        let g1 = s.create_group("g1", 4, None, policy).unwrap();
        let g2 = s.create_group("g2", 4, None, policy).unwrap();
        
        // Both use global backend, so they don't have private pools
        assert!(g1.private_pool().is_none());
        assert!(g2.private_pool().is_none());
        
        // Total isolated threads should be 0
        assert_eq!(s.total_allocated_threads(), 0);
    }

    #[test]
    fn test_isolated_pools() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy {
            allow_work_stealing: false,
            allow_dynamic_scaling: false,
        };
        
        let g1 = s.create_group("iso1", 2, None, policy).unwrap();
        let g2 = s.create_group("iso2", 2, None, policy).unwrap();
        
        // Should have different pools
        assert_ne!(Arc::as_ptr(&g1.private_pool().unwrap()), Arc::as_ptr(&g2.private_pool().unwrap()));
        
        // Total threads should be 4
        assert_eq!(s.total_allocated_threads(), 4);
        
        s.remove_group("iso1").unwrap();
        assert_eq!(s.total_allocated_threads(), 2);
    }

    #[test]
    fn test_dynamic_scaling_isolated() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy {
            allow_work_stealing: false,
            allow_dynamic_scaling: true,
        };
        
        let g1 = s.create_group("dynamic", 2, None, policy).unwrap();
        assert_eq!(g1.current_num_threads(), 2);
        
        g1.resize(4).unwrap();
        assert_eq!(g1.current_num_threads(), 4);
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
    fn test_oversubscription_warning() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy {
            allow_work_stealing: false,
            allow_dynamic_scaling: true,
        };
        
        // Create a group with threads > cores to trigger warning (visible in stderr)
        let num_threads = num_cpus::get() + 1;
        let _ = s.create_group("oversubscribed", num_threads, None, policy).unwrap();
    }
}
