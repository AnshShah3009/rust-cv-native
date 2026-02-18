# Architectural Optimization Plan

## 1. Core Data Structures (`cv-core`)

### Status: Implemented
- `Tensor<T, S: Storage<T>>` refactor is complete.
- `CpuStorage<T>` and `GpuStorage<T>` (in `cv-hal`) are operational.
- Type-safe hardware affinity: `to_cpu()` and `to_gpu_ctx(gpu)` provide explicit data migration.

### Future: Memory Alignment
- Ensure `GpuStorage` follows WGSL 16-byte alignment rules for uniforms.
- Implement zero-copy `as_slice` for CPU pinned memory if supported by backend.

## 2. SIMD Support (`cv-imgproc`, `cv-hal`)

### Status: Operational
- `wide` crate integrated for color conversion, core filters, and descriptors.
- **Implemented:** SIMD acceleration for `FAST` detector, `Sobel`, `Gaussian Blur`, and `Canny` (magnitude + quantized directions) on CPU.
- **Optimized:** `imgproc` filters now leverage asymmetric separable convolution for improved performance.

## 3. Hardware Abstraction (`cv-hal`)

### Status: Implemented
- `ComputeDevice` enum provides a unified dispatch for CPU and GPU.
- `ComputeContext` trait governs operation availability across backends.
- **Implemented:** Global GPU Scan (multi-pass), Global GPU Radix Sort, and `icp_reduction` GPU kernels.
- **Optimized:** `mog2_update`, `pointcloud_normals` (CPU fallback), and `sobel`/`gaussian_blur` (separable CPU implementation).

## 4. Redundancy Consolidation

### Generic RANSAC Engine
- **Status: Completed**
- **Details:** Redundant RANSAC loops have been replaced by a generic implementation in `cv-core::robust` that uses the `RobustModel` trait.
- **Used by:** `cv-features` (homography, fundamental), `cv-calib3d` (PnP), and `cv-registration` (global).

### PointCloud Unification
- **Status: Completed**
- **Details:** Redundant `PointCloud` definitions in `cv-stereo` and `cv-3d` have been consolidated into `cv_core::PointCloud`.
- **Optimization:** Added CPU fallback for normals estimation (via `rstar` and PCA) when GPU is unavailable.

## 5. Large Scale Acceleration & Resource Management

### Integration with Resource Groups
- Add `ComputeDevice` to `ResourceGroup`.
- Each `ResourceGroup` can be configured with a specific backend (CPU, GPU).
- Functions in `imgproc`, `features`, etc., will take an optional `ResourceGroup` or `ComputeDevice`.

### API Propagation Strategy
- Standard API: `gaussian_blur(img, sigma)` -> Uses `GpuContext::global()` or default CPU.
- Advanced API: `gaussian_blur_ctx(img, sigma, device)` -> Explicit control.

### Hardware-Accelerated Primitives Expansion
- **Filtering:** Sobel, Laplacian, Bilateral.
- **Geometric:** Resize (Linear/Nearest), Warp.
- **Morphology:** Erode, Dilate.
- **Features:** FAST, ORB (Pyramid building on GPU).

