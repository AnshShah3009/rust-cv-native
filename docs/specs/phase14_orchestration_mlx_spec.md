# Technical Specification: Phase 14 Orchestration, Resource Hardening & MLX Support

**Document Status:** Proposed / Active  
**Target Version:** v0.2.1 (Phase 14)  
**Primary Domains:** `cv-runtime`, `cv-core`, `cv-hal`

## 1. Executive Summary
Following the memory safety and type erasure fixes in Phase 13, an audit of the runtime and orchestration layers revealed critical flaws in how `rust-cv-native` handles thread concurrency limits and CPU buffer pooling. Specifically, the thread isolation model is "failing open" (bypassing limits), and the CPU memory pool is inadvertently dropping buffers. 

This phase rectifies these resource-handling bugs by enforcing strict OS-level thread pools, preventing memory fragmentation in hot loops, and introducing a compilation-gated foundation for Apple Silicon (MLX) support. Finally, we formally freeze the runtime execution architecture to guarantee stable integration for downstream Python applications.

---

## 2. Critical Bug Fixes (Orchestration & Resources)

### 2.1. "Fake" Concurrency Limits & Ignored Core Affinity
**Location:** `runtime/src/orchestrator.rs` (`ResourceGroup::spawn` and `ResourceGroup::run`)
**Severity:** High (Performance / Resource Exhaustion)

**Context:** 
Currently, `ResourceGroup` attempts to limit concurrency using a Tokio `Semaphore`. However, if `try_acquire()` fails, the code prints a warning and executes the task anyway on the global Rayon pool. This defeats the purpose of isolation. Furthermore, the `core_ids: Option<Vec<usize>>` parameter in `ResourceGroup::new` is completely ignored, meaning hardware core pinning (essential for heterogenous architectures like big.LITTLE or Apple Silicon) does not work.

**Implementation Plan:**
Replace the semaphore-backed global pool execution with strictly isolated `rayon::ThreadPool` instances per `ResourceGroup`. We will retain the `active_tasks` counter to allow `TaskScheduler::get_best_group()` to continue functioning.

1. **Update Struct Definition:**
   ```rust
   pub struct ResourceGroup {
       pub name: String,
       pub policy: GroupPolicy,
       device: ComputeDevice<'static>,
       pool: Arc<rayon::ThreadPool>, // Replaces Semaphore
       active_tasks: Arc<AtomicUsize>, // Retained for load-balancing heuristics
   }
   ```

2. **True Isolation & Affinity in `ResourceGroup::new`:**
   Configure the Rayon builder to pin threads to specific OS cores using the `core_affinity` crate.
   ```rust
   let mut builder = rayon::ThreadPoolBuilder::new().num_threads(num_threads);
   
   if let Some(cores) = core_ids {
       builder = builder.start_handler(move |thread_idx| {
           // Safely wrap around the provided cores list if num_threads > cores.len()
           if let Some(&core_id) = cores.get(thread_idx % cores.len()) {
               if let Some(system_cores) = core_affinity::get_core_ids() {
                   if let Some(target_core) = system_cores.into_iter().find(|c| c.id == core_id) {
                       core_affinity::set_for_current(target_core);
                   }
               }
           }
       });
   }
   
   let pool = Arc::new(builder.build().map_err(|e| crate::Error::RuntimeError(e.to_string()))?);
   ```

3. **Update Execution Methods:**
   Use the strictly isolated pool for execution and ensure the `active_tasks` counter safely increments/decrements.
   ```rust
   pub fn spawn<F>(&self, f: F) 
   where 
       F: FnOnce() + Send + 'static 
   {
       let counter = self.active_tasks.clone();
       counter.fetch_add(1, Ordering::SeqCst);
       
       self.pool.spawn(move || {
           // Use a drop guard to ensure the counter decrements even if `f()` panics
           struct TaskGuard(Arc<AtomicUsize>);
           impl Drop for TaskGuard {
               fn drop(&mut self) { self.0.fetch_sub(1, Ordering::SeqCst); }
           }
           let _guard = TaskGuard(counter);
           f();
       });
   }

   pub fn run<F, R>(&self, f: F) -> R 
   where 
       F: FnOnce() -> R + Send, 
       R: Send 
   {
       self.active_tasks.fetch_add(1, Ordering::SeqCst);
       let result = self.pool.install(f);
       self.active_tasks.fetch_sub(1, Ordering::SeqCst);
       result
   }
   ```

### 2.2. CPU Buffer Pool Memory Leak (Premature Drop)
**Location:** `core/src/runtime.rs` (`BufferPool::get`)
**Severity:** Medium (Memory Fragmentation / GC Churn)

**Context:** 
In the `BufferPool::get` method, the code pops a buffer from the bucket and checks if its capacity is `>= min_size`. If it is *not*, the code falls through and allocates a new `Vec`. The popped buffer is dropped and permanently removed from the pool. In a 60fps SLAM pipeline, this creates extreme heap fragmentation.

**Implementation Plan:**
Instead of blindly popping, inspect the bucket and use `swap_remove` to extract a suitable buffer in O(1) time without evicting smaller, valid buffers.

```rust
pub fn get(&self, min_size: usize) -> Vec<u8> {
    let idx = Self::get_bucket_index(min_size);
    
    // Check the exact bucket and all larger buckets
    for i in idx..4 {
        if let Ok(mut bucket) = self.buckets[i].lock() {
            // Find the first buffer that meets the capacity requirement
            if let Some(pos) = bucket.iter().position(|b| b.capacity() >= min_size) {
                let mut buf = bucket.swap_remove(pos);
                buf.clear();
                return buf;
            }
        }
    }
    // Only allocate if no suitable buffer was found in any bucket
    Vec::with_capacity(min_size)
}
```

---

## 3. Apple Silicon MLX Support (Untested)

To future-proof the library for MacOS hardware acceleration (avoiding the overhead of WGPU for Matrix-heavy tasks like Bundle Adjustment), we will lay the groundwork for `mlx-rs` support. 

*Warning: As requested, this implementation is a skeleton and remains untested on actual Apple hardware. It is gated behind a feature flag.*

### 3.1. Add Feature Flags
**Location:** `hal/Cargo.toml`
```toml
# existing features...
mlx =[]
```

### 3.2. Extend Backend Definitions
**Location:** `hal/src/backend.rs`
```rust
pub enum BackendType {
    Cpu, Cuda, Vulkan, Metal, DirectML, TensorRT, WebGPU,
    Mlx, // NEW: Apple Silicon MLX
}
```

### 3.3. MLX Context Stub
**Location:** `hal/src/mlx.rs` (New File)
```rust
use crate::{BackendType, DeviceId, Result, Error};
use cv_core::{Tensor, storage::Storage};
use crate::context::{ComputeContext, BorderMode, ThresholdType, MorphologyType, WarpType, ColorConversion, Mog2Params};

/// Experimental MLX Context for Apple Silicon
/// WARNING: Currently untested on actual hardware.
pub struct MlxContext {
    pub device_id: DeviceId,
}

impl MlxContext {
    pub fn new() -> Option<Self> {
        #[cfg(feature = "mlx")]
        {
            // Placeholder for actual mlx-rs initialization
            Some(Self { device_id: DeviceId(0) })
        }
        #[cfg(not(feature = "mlx"))]
        {
            None
        }
    }
}

impl ComputeContext for MlxContext {
    fn backend_type(&self) -> BackendType {
        BackendType::Mlx
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn wait_idle(&self) -> Result<()> {
        Ok(())
    }

    // Delegate all matrix/image operations to Error::NotSupported until implemented
    fn convolve_2d<S: Storage<f32> + 'static>(&self, _i: &Tensor<f32, S>, _k: &Tensor<f32, S>, _b: BorderMode) -> Result<Tensor<f32, S>> {
        Err(Error::NotSupported("MLX convolve_2d not implemented".into()))
    }
    
    // ... Implement the remaining ComputeContext methods returning Err(Error::NotSupported(...)) ...
}
```

### 3.4. Extend ComputeDevice Dispatch
**Location:** `hal/src/compute.rs`
Modify the `ComputeDevice` enum to map the new MLX context. Because `MlxContext` implements `ComputeContext` (even if returning errors), the `match` arms inside `compute.rs` will be clean.

```rust
use crate::mlx::MlxContext;

pub enum ComputeDevice<'a> {
    Cpu(&'a CpuBackend),
    Gpu(&'a GpuContext),
    Mlx(&'a MlxContext), // NEW
}

impl<'a> ComputeDevice<'a> {
    pub fn convolve_2d<S: Storage<f32> + 'static>(...) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.convolve_2d(input, kernel, border_mode),
            ComputeDevice::Gpu(gpu) => gpu.convolve_2d(input, kernel, border_mode),
            ComputeDevice::Mlx(mlx) => mlx.convolve_2d(input, kernel, border_mode),
        }
    }
    // ... update all other dispatch methods similarly ...
}
```

---

## 4. Updates to `FROZEN_ARCHITECTURES.md`

We are finalizing the orchestration and resource models. Append the following to the `FROZEN_ARCHITECTURES.md` document. Modifying these in the future will require a high-level review.

### 28. Isolated ResourceGroup Thread Pools
*   **Status:** Frozen (Phase 14)
*   **Target File:** `cv-runtime/src/orchestrator.rs`
*   **Definition:** `ResourceGroup` utilizes isolated `Arc<rayon::ThreadPool>` instances created via `ThreadPoolBuilder`, mapping to explicit `core_ids`.
*   **Rationale:** Semaphores over a global Rayon pool fail to provide true hardware isolation and lead to false concurrency limits under load. Freezing the isolated thread pool model guarantees that Python annotations (`@resource_group("high_priority", cores=)`) actually result in hard OS-level thread and core steering.

### 29. ComputeDevice Static Dispatch Enum
*   **Status:** Frozen (Phase 14)
*   **Target File:** `cv-hal/src/compute.rs`
*   **Definition:** The `enum ComputeDevice<'a> { Cpu(&'a CpuBackend), Gpu(&'a GpuContext), Mlx(&'a MlxContext) }` pattern.
*   **Rationale:** The library must avoid dynamic dispatch (`Box<dyn ComputeContext>`) in the hot loop. Freezing this enum structure forces all new hardware backends (like MLX or CUDA) to be implemented as variants of this enum, ensuring zero-cost abstraction when passed down to algorithmic components.

### 30. Size-Bucketed Lazy Buffer Pooling
*   **Status:** Frozen (Phase 14)
*   **Target File:** `cv-core/src/runtime.rs`
*   **Definition:** `BufferPool::get` utilizing `iter().position` to selectively `swap_remove` validly sized buffers without draining the pool.
*   **Rationale:** Image processing pipelines generate extreme memory churn. Freezing the strict re-use rules in the `BufferPool` ensures that memory fragmentation remains low during long-running tasks like Video stream processing or SLAM.

---

## 5. Testing and Validation Plan

To ensure the bug fixes are robust, the following tests must be implemented or updated:

1. **Hardware Isolation Test (`runtime/tests/stress_tests.rs`):**
   *   Update `stress_test_heavy_load_mixing` to verify that tasks spawned in an isolated group do not execute on threads belonging to the default global pool. (This can be verified by capturing `std::thread::current().name()`).

2. **Buffer Pool Leak Test (`core/tests/runtime_tests.rs` - *New*):**
   *   Write a unit test that requests a large buffer (e.g., 1MB), returns it, requests a small buffer (e.g., 64KB), returns it, and then requests a 1MB buffer again.
   *   Verify that the 1MB buffer is returned *without* the pool size shrinking unexpectedly (i.e., the 64KB buffer wasn't accidentally popped and dropped).

3. **MLX Compilation Check:**
   *   Run `cargo check --features "mlx" --target aarch64-apple-darwin` (if a cross-compilation toolchain is available, or via CI) to ensure the newly added MLX stubs do not break the HAL contract.

---

## 6. Rollout Strategy

1. **Phase A:** Implement the ThreadPool refactor in `cv-runtime`. Remove the `tokio::sync::Semaphore` from `ResourceGroup`. Wire up the `core_affinity` logic.
2. **Phase B:** Patch the CPU `BufferPool` in `cv-core/src/runtime.rs` to use `iter().position()` and `swap_remove()`.
3. **Phase C:** Create `cv-hal/src/mlx.rs`, update `hal/src/backend.rs`, and wire up the `Mlx` backend placeholders in `hal/src/compute.rs`. Add the `mlx` feature flag to `hal/Cargo.toml`.
4. **Phase D:** Append items 28, 29, and 30 to `FROZEN_ARCHITECTURES.md`.
