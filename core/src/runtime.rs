use rayon::ThreadPoolBuilder;
use std::env;
use std::sync::{Mutex, OnceLock};

static THREAD_POOL_INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Initialize the global Rayon thread pool used by CPU-parallel routines.
///
/// Priority:
/// 1. `num_threads` argument
/// 2. `RUSTCV_CPU_THREADS` environment variable
/// 3. Rayon default
pub fn init_global_thread_pool(num_threads: Option<usize>) -> Result<(), String> {
    let res = THREAD_POOL_INIT.get_or_init(|| {
        let configured_threads = match num_threads {
            Some(n) => Some(n),
            None => read_cpu_threads_from_env()?,
        };

        let mut builder = ThreadPoolBuilder::new();
        if let Some(n) = configured_threads {
            if n == 0 {
                return Err("RUSTCV_CPU_THREADS must be >= 1".to_string());
            }
            builder = builder.num_threads(n);
        }

        builder.build_global().map_err(|e| e.to_string())
    });
    res.clone()
}

pub fn current_cpu_threads() -> usize {
    rayon::current_num_threads()
}

fn read_cpu_threads_from_env() -> Result<Option<usize>, String> {
    let raw = match env::var("RUSTCV_CPU_THREADS") {
        Ok(v) => v,
        Err(env::VarError::NotPresent) => return Ok(None),
        Err(e) => return Err(format!("failed to read RUSTCV_CPU_THREADS: {e}")),
    };

    let parsed: usize = raw
        .parse()
        .map_err(|_| format!("RUSTCV_CPU_THREADS must be a positive integer, got '{raw}'"))?;
    if parsed == 0 {
        return Err("RUSTCV_CPU_THREADS must be >= 1".to_string());
    }
    Ok(Some(parsed))
}

/// A simple global buffer pool for reusing memory allocations.
pub struct BufferPool {
    pool: Mutex<Vec<Vec<u8>>>,
}

static GLOBAL_BUFFER_POOL: OnceLock<BufferPool> = OnceLock::new();

impl BufferPool {
    pub fn global() -> &'static self::BufferPool {
        GLOBAL_BUFFER_POOL.get_or_init(|| BufferPool {
            pool: Mutex::new(Vec::new()),
        })
    }

    /// Get a buffer of at least `min_size` bytes.
    pub fn get(&self, min_size: usize) -> Vec<u8> {
        let mut pool = self.pool.lock().unwrap();
        if let Some(mut buf) = pool.pop() {
            if buf.capacity() >= min_size {
                buf.clear();
                return buf;
            }
        }
        Vec::with_capacity(min_size)
    }

    /// Return a buffer to the pool for later reuse.
    pub fn return_buffer(&self, mut buf: Vec<u8>) {
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < 32 {
            // Limit pool size
            buf.clear();
            pool.push(buf);
        }
    }
}
