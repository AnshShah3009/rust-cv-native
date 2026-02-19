# Technical Specification: Phase 13 Core Hardening & Architecture Freezes

**Document Status:** Proposed / Active  
**Target Version:** v0.2.0 (Phase 13)  
**Primary Domains:** `cv-hal`, `cv-core`, `cv-features`, `cv-scientific`, `cv-runtime`

## 1. Executive Summary
As `rust-cv-native` transitions from algorithm drafting to a production-ready OpenCV replacement, critical memory safety hazards (Undefined Behavior) and architectural leaks have been identified. This specification outlines the mandatory code corrections required to stabilize the library. Additionally, it formally adds five core subsystems to `FROZEN_ARCHITECTURES.md` to prevent downstream breakages during Q3/Q4 2026 development.

---

## 2. Critical Bug Fixes (P0)

### 2.1. Eradicate Undefined Behavior in Tensor Type Erasure
**Location:** `hal/src/cpu/mod.rs` and `features/src/orb.rs`
**Severity:** Critical (Memory Safety / UB)

**Context:** The codebase currently attempts to bypass the borrow checker and generics by performing raw pointer casts between `Tensor<T, S>` and `Tensor<T, GpuStorage<T>>` or `Tensor<T, CpuStorage<T>>`. Furthermore, `std::mem::transmute` is being used on slice references (fat pointers), which is strictly forbidden in Rust.

**Implementation Plan:**
1. **Remove Transmutes:** In `hal/src/cpu/mod.rs` (`subtract` function), replace all `std::mem::transmute` calls on slices with the `bytemuck` crate.
   ```rust
   // REMOVE:
   // let a_f32 = unsafe { std::mem::transmute::<&, &>(src_a) };
   
   // IMPLEMENT:
   let a_f32: &[f32] = bytemuck::cast_slice(src_a);
   let b_f32: &[f32] = bytemuck::cast_slice(src_b);
   let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
   ```
2. **Safe Downcasting:** Replace raw pointer casting (`unsafe { &*input_ptr }`) with safe `Any` trait downcasting utilizing the newly added `Storage::as_any()` method.
   ```rust
   // IMPLEMENT across cv-hal and cv-features:
   if let Some(gpu_storage) = input.storage.as_any().downcast_ref::<GpuStorage<T>>() {
       let input_gpu = Tensor {
           storage: gpu_storage.clone(),
           shape: input.shape,
           dtype: input.dtype,
           _phantom: std::marker::PhantomData,
       };
       // proceed with gpu execution...
   } else {
       return Err(Error::InvalidInput("Expected GpuStorage tensor".into()));
   }
   ```

### 2.2. ORB Pyramid Coordinate Scaling Correction
**Location:** `features/src/orb.rs`
**Severity:** High (Algorithmic Correctness)

**Context:** Keypoints detected in downscaled pyramid levels must be re-scaled to the original image coordinates. The CPU path currently *divides* by the scale factor, shrinking coordinates toward `(0,0)`, while the GPU path correctly *multiplies*.

**Implementation Plan:**
Update the CPU path inside `Orb::detect()` and `Orb::detect_ctx()`:
```rust
// REMOVE:
// let scaled_kp = KeyPoint::new(kp.x / scale as f64, kp.y / scale as f64)
//     .with_size(self.patch_size as f64 * scale as f64)

// IMPLEMENT:
let scaled_kp = KeyPoint::new(kp.x * scale as f64, kp.y * scale as f64)
    .with_size(self.patch_size as f64 * scale as f64)
```

### 2.3. Hungarian Algorithm Matrix Padding Overflow
**Location:** `scientific/src/matching.rs`
**Severity:** Medium (Infinite Loop / Precision Loss Risk)

**Context:** When padding rectangular cost matrices to square matrices, the algorithm uses `max_val * 10.0 + 100.0`. If `max_val` is extremely large (e.g., squared errors without a max threshold), this can push `f64` into precision-loss territory or `INFINITY`, causing the Munkres row-reduction step to fail or loop infinitely.

**Implementation Plan:**
Replace the arbitrary multiplier with a safe upper bound padding.
```rust
// REMOVE:
// matrix = max_val * 10.0 + 100.0;

// IMPLEMENT:
// Add a strictly slightly larger, mathematically safe maximum:
matrix = max_val + 1.0; 
```

---

## 3. Architectural & Performance Refinements (P1)

### 3.1. Thread Pool Escape in `ResourceGroup::join`
**Location:** `runtime/src/orchestrator.rs`
**Problem:** `ResourceGroup::join` calls `rayon::join` directly. Since `ResourceGroup` relies on semaphores to constrain a shared global pool rather than utilizing isolated ThreadPools, `rayon::join` will silently bypass the `concurrency_limit` semaphore, leading to CPU oversubscription.
**Resolution:** Remove the `join` method from `ResourceGroup` entirely *or* document it as unconstrained. Thread steering should rely strictly on `ResourceGroup::run` or `ResourceGroup::spawn` where the `Semaphore::try_acquire` logic lives.

### 3.2. WGPU Async Deadlock in Concurrent Environments
**Location:** `hal/src/gpu_kernels/buffer_utils.rs` (`read_buffer`)
**Problem:** Invoking `device.poll(wgpu::PollType::Wait)` blocks the OS thread. Because Python integrations (`cv_native/jit.py`) use asynchronous execution via Tokio, blocking a Tokio worker thread with `Wait` can cause executor starvation.
**Resolution:** Implement an async-friendly polling loop.
```rust
// IMPLEMENT:
while rx.try_recv().is_err() {
    device.poll(wgpu::Maintain::Poll);
    tokio::task::yield_now().await;
}
let res = rx.await.unwrap();
```

### 3.3. GPU Buffer Pool Memory Leak
**Location:** `hal/src/gpu_kernels/buffer_utils.rs`
**Problem:** `GpuBufferPool` stores up to 8 buffers per bucket. A video processing pipeline with varied sizes will eventually fill all buckets and never release the memory back to the GPU OS driver.
**Resolution:** 
1. Implement a `pub fn clear(&self)` method on `GpuBufferPool`.
2. Automatically invoke `clear()` in the `Drop` implementation of `VideoCapture` and at the end of the `SLAM` processing loop.

---

## 4. Updates to `FROZEN_ARCHITECTURES.md`

The following systems have proven robust and are formally added to the "Frozen" list. Any future modification to these requires a formal design review.

### 23. ComputeDevice Zero-Cost Dispatch
*   **Target File:** `cv-hal/src/compute.rs`
*   **Definition:** The `enum ComputeDevice<'a> { Cpu(&'a CpuBackend), Gpu(&'a GpuContext) }` pattern.
*   **Rationale:** Provides static dispatch across the library without the viral `Box<dyn ComputeContext>` allocation overhead. This prevents the trait-object virtualization penalty in hot loops like FAST detection.

### 24. R-Tree Spatial Indexing
*   **Target File:** `cv-scientific/src/geometry.rs`
*   **Definition:** Wrapping generic geometric queries inside `rstar` using the `IndexedPolygon` AABB wrapper.
*   **Rationale:** Stabilizes the collision and bounding-box evaluation pipeline. Guarantees $O(\log n)$ performance for SLAM clustering and prevents breakages from underlying spatial-engine swaps.

### 25. UnifiedBuffer State Machine
*   **Target File:** `cv-runtime/src/memory.rs`
*   **Definition:** The explicitly managed `BufferLocation` state machine (`Host`, `Device`, `Both`) taking `&mut self` to manually synchronize memory.
*   **Rationale:** Implicit memory transfers are the #1 cause of performance degradation. Freezing this manual synchronization model forces developers to explicitly acknowledge PCI-e bandwidth costs.

### 26. Python JIT Hashing Contract
*   **Target File:** `python_examples/cv_native/jit.py`
*   **Definition:** Cryptographic hashing (`hashlib.sha256`) of function bytecode and first-1024-byte NumPy arrays to derive JIT cache keys.
*   **Rationale:** Modifying the caching heuristic mid-development will cause massive cache invalidations and disk space leaks in the end-user's `~/.cv_native/jit_cache` directory.

### 27. Separable Filter Architecture
*   **Target File:** `cv-imgproc/src/convolve.rs`
*   **Definition:** The separation of 2D convolutions into successive 1D horizontal and vertical passes, allocating intermediate buffers from `cv_core::BufferPool`.
*   **Rationale:** Guarantees $O(K)$ time complexity instead of $O(K^2)$. Freezing this API ensures all future filters (Gaussian, Sobel, Scharr, etc.) are built on top of the performant, SIMD-compatible row-processor rather than naive nested loops.

---

## 5. Rollout Strategy

1. **Phase A (Immediate):** Implement the UB fixes (Section 2.1) and ORB coordinate fixes (Section 2.2). Cut a patch release `0.1.1`.
2. **Phase B:** Implement the WGPU Async polling loop (Section 3.2) and `GpuBufferPool` cleanup logic (Section 3.3). 
3. **Phase C:** Update documentation, append Section 4 to `FROZEN_ARCHITECTURES.md`.
