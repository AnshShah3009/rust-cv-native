# Architectural Optimization Plan

## 1. Core Data Structures (`cv-core`)

### Current State
- `Tensor` and `ImageBuffer` are tightly coupled to CPU memory (`Vec<T>`).
- No abstraction for data residing on GPU or other accelerators.

### Proposal: `Storage` Trait
Refactor `Tensor<T>` to `Tensor<T, S: Storage<T>>`.

```rust
pub trait Storage<T> {
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
    fn device(&self) -> DeviceType;
    // Future: buffer() -> &wgpu::Buffer
}

pub struct CpuStorage<T>(Vec<T>);
pub struct GpuStorage<T>(wgpu::Buffer); // Future
```

## 2. SIMD Support (`cv-imgproc`)

### Current State
- Operations rely on LLVM auto-vectorization.
- No explicit use of SIMD intrinsics.

### Proposal: `wide` Crate Integration
- Use the `wide` crate for portable SIMD (SSE, AVX, NEON, WASM).
- **Target:** Color conversion (RGB->Gray) and Convolution filters.

## 3. Hardware Abstraction (`cv-hal`)

### Current State
- `ComputeBackend` is a metadata trait.
- GPU kernels are exposed as standalone functions in `hal/src/gpu_kernels`.

### Proposal: `ComputeContext`
Define a unified trait for operations:

```rust
trait ComputeContext {
    fn transform(&self, input: &Tensor, matrix: &Matrix4) -> Result<Tensor>;
    fn filter_2d(&self, input: &Tensor, kernel: &Tensor) -> Result<Tensor>;
}
```

## 4. Batching Strategy

### GPU Batching
- **Current:** One dispatch per operation.
- **Proposed:** `BatchExecutor`.
    - Collects commands (e.g., "transform this point cloud", "filter this image").
    - Submits them in a single `queue.submit()`.
    - Uses `indirect_dispatch` for variable-sized workloads if possible.

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

