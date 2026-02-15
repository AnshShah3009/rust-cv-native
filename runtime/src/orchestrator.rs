use rayon::ThreadPool;
use std::sync::{Arc, OnceLock};
use std::collections::HashMap;
use std::sync::Mutex;
use crate::Result;
use core_affinity::CoreId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    RealTime,
}

pub struct ResourceGroup {
    pub name: String,
    pub pool: Arc<ThreadPool>,
    pub cores: Vec<usize>,
}

impl ResourceGroup {
    pub fn new(name: &str, num_threads: usize, core_ids: Option<Vec<usize>>) -> Result<Self> {
        let name_str = name.to_string();
        let thread_name_prefix = format!("cv-{}-", name);
        
        let core_ids_cloned = core_ids.clone();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |i| format!("{}{}", thread_name_prefix, i))
            .start_handler(move |i| {
                if let Some(ref cores) = core_ids_cloned {
                    if let Some(&core_index) = cores.get(i % cores.len()) {
                        core_affinity::set_for_current(CoreId { id: core_index });
                    }
                }
            })
            .build()
            .map_err(|e| crate::Error::RuntimeError(e.to_string()))?;
            
        Ok(Self {
            name: name_str,
            pool: Arc::new(pool),
            cores: core_ids.unwrap_or_default(),
        })
    }

    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.pool.spawn(f);
    }

    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(f)
    }
}

pub struct TaskScheduler {
    groups: Mutex<HashMap<String, Arc<ResourceGroup>>>,
}

impl TaskScheduler {
    pub fn new() -> Self {
        Self {
            groups: Mutex::new(HashMap::new()),
        }
    }

    pub fn create_group(&self, name: &str, num_threads: usize, cores: Option<Vec<usize>>) -> Result<Arc<ResourceGroup>> {
        let group = Arc::new(ResourceGroup::new(name, num_threads, cores)?);
        self.groups.lock().unwrap().insert(name.to_string(), group.clone());
        Ok(group)
    }

    pub fn get_group(&self, name: &str) -> Option<Arc<ResourceGroup>> {
        self.groups.lock().unwrap().get(name).cloned()
    }
}

static GLOBAL_SCHEDULER: OnceLock<TaskScheduler> = OnceLock::new();

pub fn scheduler() -> &'static TaskScheduler {
    GLOBAL_SCHEDULER.get_or_init(|| {
        let s = TaskScheduler::new();
        // Create a default group
        s.create_group("default", rayon::current_num_threads(), None).unwrap();
        s
    })
}
