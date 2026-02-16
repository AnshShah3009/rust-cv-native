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
    /// If true, this group can share threads with other groups (using a shared pool)
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
    pool: RwLock<Arc<ThreadPool>>,
    pub cores: Vec<usize>,
    pub policy: GroupPolicy,
}

impl ResourceGroup {
    pub fn new(scheduler: &TaskScheduler, name: &str, num_threads: usize, core_ids: Option<Vec<usize>>, policy: GroupPolicy) -> Result<Self> {
        let name_str = name.to_string();
        
        if let Some(ref cores) = core_ids {
            if cores.is_empty() {
                return Err(crate::Error::RuntimeError("core_ids cannot be empty if provided".to_string()));
            }
        }

        let pool = if policy.allow_work_stealing {
            // Shared groups use the global scheduler's shared pool
            scheduler.get_shared_pool()?
        } else {
            // Isolated groups get their own pool
            Arc::new(Self::create_isolated_pool(name, num_threads, &core_ids)?)
        };
            
        Ok(Self {
            name: name_str,
            pool: RwLock::new(pool),
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

    /// Resize the resource group's pool.
    /// If work-stealing is enabled, this might affect other shared groups.
    pub fn resize(&self, new_threads: usize) -> Result<()> {
        if !self.policy.allow_dynamic_scaling {
            return Err(crate::Error::RuntimeError(format!("Dynamic scaling not allowed for group '{}'", self.name)));
        }

        let new_pool = if self.policy.allow_work_stealing {
            // Resizing a shared group actually resizes the shared pool for everyone
            scheduler().resize_shared_pool(new_threads)?
        } else {
            Arc::new(Self::create_isolated_pool(&self.name, new_threads, &Some(self.cores.clone()))?)
        };

        let mut pool_guard = self.pool.write()
            .map_err(|_| crate::Error::ConcurrencyError("pool lock poisoned".to_string()))?;
        *pool_guard = new_pool;
        
        Ok(())
    }

    pub fn pool(&self) -> Arc<ThreadPool> {
        self.pool.read().unwrap().clone()
    }

    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.pool().spawn(f);
    }

    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool().install(f)
    }
}

pub struct TaskScheduler {
    groups: Mutex<HashMap<String, Arc<ResourceGroup>>>,
    total_threads: AtomicUsize,
    shared_pool: RwLock<Option<Arc<ThreadPool>>>,
}

impl TaskScheduler {
    pub fn new() -> Self {
        Self {
            groups: Mutex::new(HashMap::new()),
            total_threads: AtomicUsize::new(0),
            shared_pool: RwLock::new(None),
        }
    }

    pub fn get_shared_pool(&self) -> Result<Arc<ThreadPool>> {
        {
            let pool_guard = self.shared_pool.read()
                .map_err(|_| crate::Error::ConcurrencyError("shared_pool lock poisoned".to_string()))?;
            if let Some(ref pool) = *pool_guard {
                return Ok(pool.clone());
            }
        }

        let mut pool_guard = self.shared_pool.write()
            .map_err(|_| crate::Error::ConcurrencyError("shared_pool lock poisoned".to_string()))?;
        
        if let Some(ref pool) = *pool_guard {
            return Ok(pool.clone());
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .thread_name(|i| format!("cv-shared-{}", i))
            .build()
            .map_err(|e| crate::Error::RuntimeError(e.to_string()))?;
            
        let arc_pool = Arc::new(pool);
        *pool_guard = Some(arc_pool.clone());
        Ok(arc_pool)
    }

    pub fn resize_shared_pool(&self, new_threads: usize) -> Result<Arc<ThreadPool>> {
        let mut pool_guard = self.shared_pool.write()
            .map_err(|_| crate::Error::ConcurrencyError("shared_pool lock poisoned".to_string()))?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(new_threads)
            .thread_name(|i| format!("cv-shared-{}", i))
            .build()
            .map_err(|e| crate::Error::RuntimeError(e.to_string()))?;
            
        let arc_pool = Arc::new(pool);
        *pool_guard = Some(arc_pool.clone());
        
        // Update all shared groups to point to the new pool
        let groups = self.groups.lock()
            .map_err(|_| crate::Error::ConcurrencyError("groups lock poisoned".to_string()))?;
        
        for group in groups.values() {
            if group.policy.allow_work_stealing {
                let mut g_pool = group.pool.write()
                    .map_err(|_| crate::Error::ConcurrencyError("group pool lock poisoned".to_string()))?;
                *g_pool = arc_pool.clone();
            }
        }

        Ok(arc_pool)
    }

    pub fn create_group(&self, name: &str, num_threads: usize, cores: Option<Vec<usize>>, policy: GroupPolicy) -> Result<Arc<ResourceGroup>> {
        let mut groups = self.groups.lock()
            .map_err(|_| crate::Error::ConcurrencyError("groups mutex poisoned".to_string()))?;
            
        if groups.contains_key(name) {
            return Err(crate::Error::RuntimeError(format!("Resource group '{}' already exists", name)));
        }

        let current_total = self.total_threads.load(Ordering::Relaxed);
        let max_cores = num_cpus::get();
        if !policy.allow_work_stealing && current_total + num_threads > max_cores {
            eprintln!("Warning: Isolated resource group '{}' with {} threads will oversubscribe CPU (total threads: {}/{})", 
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
                self.total_threads.fetch_sub(group.pool().current_num_threads(), Ordering::SeqCst);
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
        let s = TaskScheduler::new();
        // Create a default group with work-stealing and scaling enabled
        s.create_group("default", num_cpus::get(), None, GroupPolicy::default()).unwrap();
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_shared_pool_work_stealing() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy {
            allow_work_stealing: true,
            allow_dynamic_scaling: true,
        };
        
        let g1 = s.create_group("g1", 4, None, policy).unwrap();
        let g2 = s.create_group("g2", 4, None, policy).unwrap();
        
        // Both should point to the same underlying shared pool
        assert_eq!(Arc::as_ptr(&g1.pool().clone()), Arc::as_ptr(&g2.pool().clone()));
        
        // Total threads tracked should be 0 because they are shared (global pool not counted in isolated total)
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
        assert_ne!(Arc::as_ptr(&g1.pool().clone()), Arc::as_ptr(&g2.pool().clone()));
        
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
        assert_eq!(g1.pool().current_num_threads(), 2);
        
        g1.resize(4).unwrap();
        assert_eq!(g1.pool().current_num_threads(), 4);
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
        // We use num_cpus + 1 instead of a huge number to avoid killing the system
        let num_threads = num_cpus::get() + 1;
        let _ = s.create_group("oversubscribed", num_threads, None, policy).unwrap();
    }
}
