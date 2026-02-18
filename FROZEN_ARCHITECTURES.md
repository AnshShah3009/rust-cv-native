# Frozen Architectures & Core Decisions

This document tracks the core architectural components that have been stabilized and "frozen." These components represent the foundational design of `rust-cv-native`. 

**Change Policy:** Any modification to these components requires a high-level review. Developers must document:
1. **Why** the change is necessary (e.g., a fundamental limitation discovered).
2. **Impact** on dependent crates and performance.
3. **Alternatives** considered.

---

## 1. Tensor & Storage Abstraction (`cv-core`, `cv-hal`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The `Tensor<T, S: Storage<T>>` model, where `S` is either `CpuStorage` or `GpuStorage`.
*   **Rationale:** Decouples algorithm logic from memory layout and hardware affinity. Explicit migration (e.g., `to_gpu_ctx`) prevents hidden performance costs and ensures type-safety across backends.

## 2. Generic RANSAC Engine (`cv-core::robust`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The trait-based robust estimation engine utilizing `RobustModel` and `Ransac`.
*   **Rationale:** Eliminates redundant, specialized RANSAC loops. Centralizing the engine allows for global optimizations (e.g., parallelizing the sampling or evaluation phase) that benefit all feature matching and calibration tasks.

## 3. CPU Separable Convolution Template (`hal/src/cpu/mod.rs`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The `convolve_separable_u8_to_u8` pattern:
    1.  Acquire intermediate buffer from `BufferPool`.
    2.  Horizontal pass with `wide` SIMD.
    3.  Vertical pass with `wide` SIMD.
    4.  Return buffer to `BufferPool`.
*   **Rationale:** Provides $O(K)$ complexity (vs $O(K^2)$) and minimizes allocation overhead. This pattern is the mandatory standard for all separable linear filters on CPU.

## 4. GPU Multi-Pass Orchestration (`hal/src/gpu_kernels/radix_sort.rs`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The recursive GPU-side orchestration for algorithms requiring global state (e.g., Radix Sort and Prefix Sum).
*   **Rationale:** Establishes a pattern for complex GPU operations that avoids CPU stalls (`pollster::block_on`) and PCIe readbacks. This ensures maximum throughput for large-scale data processing (PointClouds, Morton indexing).

## 5. Unified PointCloud Structure (`cv-core`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** Standardized `PointCloud<T>` (aliased to `PointCloudf32/f64`) located in `cv-core`.
*   **Rationale:** Prevents fragmentation between `cv-stereo`, `cv-3d`, and `cv-registration`. Ensures that output from a depth estimator can be directly consumed by an ICP or TSDF module without conversion.

## 6. SIMD-Accelerated Canny (`imgproc/src/edges.rs`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** Vectorized magnitude calculation and quantized geometric direction checks (avoiding `atan2`) using `wide` f32x8.
*   **Rationale:** Optimized Canny is critical for real-time feature detection and contour analysis. The geometric direction check is a high-performance alternative to traditional trigonometric methods.

## 7. Task Orchestration & Resource Groups (`cv-runtime`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The `TaskScheduler` and `ResourceGroup` model for managing CPU thread pools (via Rayon) and hardware affinity (via `ComputeDevice`).
*   **Rationale:** Provides a unified way to steer compute-intensive tasks to specific hardware backends and isolated thread pools, preventing interference between high-priority real-time tasks and background processing.

## 8. GPU Resource Pooling (`cv-hal`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The `GpuBufferPool` mechanism for reusing `wgpu::Buffer` allocations based on size buckets and usage flags.
*   **Rationale:** Dramatically reduces driver overhead and memory fragmentation in high-frequency GPU operations (e.g., video processing, SLAM).

## 9. Unified Memory Management (`cv-runtime::UnifiedBuffer`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The `UnifiedBuffer` abstraction for synchronized host-device memory access.
*   **Rationale:** Simplifies complex multi-backend pipelines by automating the synchronization of data between CPU and GPU, ensuring that the latest data is always available to the requested compute device.

## 10. GPU ORB Pipeline (`cv-features::orb`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The full end-to-end GPU-accelerated ORB extraction: Pyramid -> FAST -> Extraction -> Orientation -> rBRIEF.
*   **Rationale:** Provides a high-performance baseline for SLAM and tracking systems. The pipeline minimizes CPU-GPU transfers by keeping intermediate score maps and pyramids on the device.

## 11. Recursive GPU Scan & Global Sorting (`cv-hal`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The `gpu_exclusive_scan` implementation that uses a recursive work-efficient Blelloch scan to handle arbitrary data sizes on the GPU.
*   **Rationale:** Essential for many computer vision algorithms (sorting, feature collection, stream compaction). The recursive approach avoids the limitations of single-workgroup scan sizes.
