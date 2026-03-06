//! CubeCL backend — GPU kernels compiled from Rust via `#[cube]` macros.
//!
//! This module replaces all WGSL shaders with CubeCL kernels that JIT-compile to
//! WGSL (WebGPU), CUDA PTX, or HIP depending on the enabled feature.
//!
//! # Entry point
//!
//! Use [`get_client()`] to obtain a lazily-initialized `ComputeClient` singleton.
//!
//! # Feature gates
//!
//! - `cubecl`  — enables this module + WebGPU/Metal backend via `cubecl-wgpu`
//! - `cuda`    — additionally enables NVIDIA CUDA backend via `cubecl-cuda`

use cubecl::client::ComputeClient;
use cubecl::Runtime;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use std::sync::OnceLock;

pub mod kernels;

/// Type alias for the WGPU compute client — avoids repeating long associated-type paths.
pub type WgpuClient = ComputeClient<WgpuRuntime>;

/// Returns a lazily-initialized, process-wide CubeCL WGPU compute client.
///
/// Returns `None` if no GPU adapter is available (e.g., in CI without GPU).
/// The client is created on first call and cloned on subsequent calls
/// (cheap — `ComputeClient` is internally reference-counted).
///
/// # Graceful degradation
///
/// Tests using this function should skip if `None` is returned, rather than
/// panicking. Example:
///
/// ```ignore
/// #[test]
/// fn my_gpu_test() {
///     let Some(client) = get_client() else {
///         eprintln!("GPU unavailable, skipping test");
///         return;
///     };
///     // test code...
/// }
/// ```
pub fn get_client() -> Option<WgpuClient> {
    static CLIENT: OnceLock<Option<WgpuClient>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                WgpuRuntime::client(&WgpuDevice::DefaultDevice)
            })) {
                Ok(client) => Some(client),
                Err(_) => {
                    eprintln!("[CubeCL] No GPU adapter found, GPU tests will be skipped");
                    None
                }
            }
        })
        .clone()
}
