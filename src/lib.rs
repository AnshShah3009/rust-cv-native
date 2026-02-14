pub use cv_core as core;
pub use cv_features as features;
pub use cv_hal as hal;
pub use cv_imgproc as imgproc;
pub use cv_stereo as stereo;
pub use cv_video as video;

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
    cv_core::init_global_thread_pool(num_threads)
}
