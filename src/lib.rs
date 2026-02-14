pub use cv_core as core;
pub use cv_features as features;
pub use cv_hal as hal;
pub use cv_imgproc as imgproc;
pub use cv_stereo as stereo;
pub use cv_video as video;

use rayon::ThreadPoolBuilder;
use std::env;
use std::sync::OnceLock;

static THREAD_POOL_INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Initialize a single global Rayon thread pool for all CPU-parallel routines.
///
/// Call this once at application startup before running heavy CV workloads.
/// Repeated calls are idempotent and return the first initialization result.
///
/// Priority order:
/// 1. explicit `num_threads`
/// 2. `RUSTCV_CPU_THREADS` env var
/// 3. Rayon default
pub fn init_thread_pool(num_threads: Option<usize>) -> Result<(), String> {
    let res = THREAD_POOL_INIT.get_or_init(|| {
        let configured_threads = match num_threads {
            Some(n) => Some(n),
            None => read_cpu_threads_from_env()?,
        };
        let mut builder = ThreadPoolBuilder::new();
        if let Some(n) = configured_threads {
            builder = builder.num_threads(n);
        }
        builder.build_global().map_err(|e| e.to_string())
    });
    res.clone()
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
