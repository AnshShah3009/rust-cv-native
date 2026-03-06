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
/// The client is created on first call and cloned on subsequent calls
/// (cheap — `ComputeClient` is internally reference-counted).
pub fn get_client() -> WgpuClient {
    static CLIENT: OnceLock<WgpuClient> = OnceLock::new();
    CLIENT
        .get_or_init(|| WgpuRuntime::client(&WgpuDevice::DefaultDevice))
        .clone()
}
