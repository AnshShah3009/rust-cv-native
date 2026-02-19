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

## 12. SIMD Morphology & Parallel TSDF (`hal/src/cpu/mod.rs`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** 
    - **Morphology:** Vectorized `u8x32` min/max operations for erosion and dilation.
    - **TSDF:** Voxel-centric integration parallelized over volume planes using Rayon.
*   **Rationale:** Ensures that the CPU backend remains a viable high-performance fallback for systems without compatible GPUs, maintaining algorithmic parity with the GPU kernels.

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

## 13. Unified Camera Model & Projection (`cv-core::geometry`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The `CameraModel<T>` trait and its `PinholeModel` implementation.
*   **Rationale:** Standardizes how all 2D/3D algorithms (SFM, SLAM, Calibration) interact with camera hardware. Combining intrinsics and distortion into a single model simplifies APIs and ensures consistent unprojection/undistortion logic.

## 14. Volumetric Raycasting Pattern (`hal/src/gpu_kernels/tsdf.rs`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The zero-crossing raymarch orchestration that transforms volumetric TSDF data into synthetic depth/normal maps.
*   **Rationale:** Bridges the gap between implicit volumes and pixel-space perception. Essential for model-to-frame tracking and real-time AR visualization.

## 15. Iterative GPU Pyramid Propagation (`hal/src/gpu_kernels/optical_flow.rs`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** Coarse-to-fine orchestration where point estimates are propagated and scaled (x2) between pyramid levels using inline GPU scaling kernels.
*   **Rationale:** Enables tracking of large displacements while eliminating PCIe bottlenecks. This pattern is mandatory for all pyramid-based tracking (LK, Patch-based, etc.).

## 16. GPU Atomic Geometry Extraction (`hal/src/gpu_kernels/marching_cubes.rs`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The pattern for extracting non-deterministic triangle meshes from structured volumes using `atomicAdd` on GPU counter buffers.
*   **Rationale:** Provides a scalable, high-throughput way to convert volumetric data to explicit mesh geometry without CPU-side vertex management.

## 17. Spatial Graph Data Model (`cv-optimize::pose_graph`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The SE3 node/edge representation for global consistency, utilizing `Isometry3` and `Matrix6` information matrices.
*   **Rationale:** Establishes the source of truth for the system's pose history and relative constraints. Decouples the graph topology from the specific numerical solver used.

## 18. Dense GPU ICP Tracking (`hal/src/gpu_kernels/icp.rs`)
*   **Status:** Frozen (Feb 19, 2026)
*   **Definition:** The projective point-to-plane ICP orchestration: per-pixel Jacobian/Hessian computation followed by hierarchical parallel reduction on GPU.
*   **Rationale:** Provides the core tracking loop for Dense SLAM. The on-device reduction minimizes data transfer to a single 6x6 matrix per frame, enabling high-frequency tracking against complex volumetric models.

## 19. 3D File I/O Interfaces (`cv-io`)
*   **Status:** Frozen (Feb 20, 2026)
*   **Definition:** The decoupling of 3D data parsing (`ply`, `obj`, `stl`, `pcd`) from the core representation, strictly returning `Result<T, IoError>`.
*   **Rationale:** Keeps the core and math libraries free of string-parsing overhead. Forces all format additions to conform to a standard stream parser approach.

## 20. Video I/O Backend Abstraction (`cv-videoio`)
*   **Status:** Frozen (Feb 20, 2026)
*   **Definition:** The `VideoCapture` and `VideoWriter` traits, strictly decoupled from the underlying `ffmpeg-next` and `v4l` implementations.
*   **Rationale:** Video codec ecosystems are fragile. This interface ensures future implementations (GStreamer, MediaFoundation) can be swapped seamlessly without breaking downstream SLAM modules.

## 21. Python JIT Caching Bridge (`python_examples/cv_native/jit.py`)
*   **Status:** Frozen (Feb 20, 2026)
*   **Definition:** Wrapper logic using `hashlib` to fingerprint function arguments and bytecode to route to cached Rust executions.
*   **Rationale:** Establishes the standard contract between the Python frontend and Rust backend. Prevents unexpected recompilations across different pipeline nodes.

## 22. Epipolar & Camera Mathematical Models (`cv-calib3d`)
*   **Status:** Frozen (Feb 20, 2026)
*   **Definition:** The separation of closed-form solvers (`solve_pnp_dlt`) from iterative refiners (`solve_pnp_refine`) and robust wrappers (`solve_pnp_ransac`) using `nalgebra`.
*   **Rationale:** These algorithms are mathematically absolute. Freezing the signatures ensures a stable geometric bedrock for future multi-camera and Bundle Adjustment features.
