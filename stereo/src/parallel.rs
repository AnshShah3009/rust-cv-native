use crate::{Result, StereoError};
use std::env;
use std::sync::OnceLock;

static THREAD_POOL_INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

pub fn init_global_thread_pool() -> Result<()> {
    THREAD_POOL_INIT
        .get_or_init(|| {
            let Some(num_threads) = read_cpu_threads_from_env().map_err(|e| e.to_string())? else {
                return Ok(());
            };

            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .map_err(|e| {
                    format!(
                        "Failed to initialize global thread pool with \
                         RUSTCV_CPU_THREADS={num_threads}: {e}"
                    )
                })
        })
        .as_ref()
        .map_err(|e| StereoError::InvalidParameters(e.clone()))?;
    Ok(())
}

fn read_cpu_threads_from_env() -> Result<Option<usize>> {
    let raw = match env::var("RUSTCV_CPU_THREADS") {
        Ok(v) => v,
        Err(env::VarError::NotPresent) => return Ok(None),
        Err(e) => {
            return Err(StereoError::InvalidParameters(format!(
                "Failed to read RUSTCV_CPU_THREADS: {e}"
            )))
        }
    };

    let parsed: usize = raw.parse().map_err(|_| {
        StereoError::InvalidParameters(format!(
            "RUSTCV_CPU_THREADS must be a positive integer, got '{raw}'"
        ))
    })?;
    if parsed == 0 {
        return Err(StereoError::InvalidParameters(
            "RUSTCV_CPU_THREADS must be >= 1".to_string(),
        ));
    }
    Ok(Some(parsed))
}
