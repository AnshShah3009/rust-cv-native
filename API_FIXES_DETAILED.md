# API Consistency Fixes - Detailed Implementation Guide

**Document Status:** Implementation roadmap with before/after code examples
**Created:** 2026-02-24

---

## Table of Contents
1. [Critical Fix #1: Return Type Standardization](#fix-1-return-type-standardization)
2. [Critical Fix #2: Compute Device Parameter Standardization](#fix-2-compute-device-parameter)
3. [Recommended Fix #3: Add Missing _ctx Variants](#fix-3-missing-variants)
4. [Hygiene Fix #4: Add #[must_use] Attributes](#fix-4-must_use)
5. [Implementation Priority & Effort Estimates](#implementation-priority)

---

## FIX #1: Return Type Standardization

### PROBLEM ANALYSIS

Currently, many public compute functions return bare types instead of Result<T>:

```rust
// cv-imgproc/threshold.rs - CURRENT (BAD)
pub fn threshold(src: &GrayImage, thresh: u8, max_value: u8, typ: ThresholdType) -> GrayImage {
    // No error handling possible

// cv-imgproc/edges.rs - CURRENT (BAD)
pub fn sobel_magnitude_ctx(gx: &GrayImage, gy: &GrayImage, group: &RuntimeRunner) -> GrayImage {
    // No error handling possible
```

**Why this is problematic:**
1. Errors are hidden (panic or silent failures)
2. Cannot use `?` operator for error propagation
3. User cannot distinguish between invalid input and computation failure
4. Inconsistent with best practices (functions that can fail should return Result)

### SOLUTION

Convert all compute functions to return `Result<T>`. Use the appropriate error type for each crate:
- cv-imgproc → ImgprocError
- cv-features → FeatureError
- cv-objdetect → CvError or new ObjdetectError
- cv-dnn → CvError
- cv-photo → PhotoError or CvError
- cv-video → VideoError

### AFFECTED FUNCTIONS

#### cv-imgproc/threshold.rs

**BEFORE:**
```rust
pub fn threshold(src: &GrayImage, thresh: u8, max_value: u8, typ: ThresholdType) -> GrayImage {
pub fn threshold_otsu(src: &GrayImage, max_value: u8, typ: ThresholdType) -> (u8, GrayImage) {
```

**AFTER:**
```rust
pub fn threshold(
    src: &GrayImage,
    thresh: u8,
    max_value: u8,
    typ: ThresholdType,
) -> Result<GrayImage> {
    validate_image_size(src.width(), src.height())?;
    // ... existing implementation
    Ok(result)
}

pub fn threshold_otsu(
    src: &GrayImage,
    max_value: u8,
    typ: ThresholdType,
) -> Result<(u8, GrayImage)> {
    validate_image_size(src.width(), src.height())?;
    // ... existing implementation
    Ok((threshold, result))
}
```

#### cv-imgproc/edges.rs

**BEFORE:**
```rust
pub fn sobel_magnitude_ctx(gx: &GrayImage, gy: &GrayImage, group: &RuntimeRunner) -> GrayImage {
pub fn laplacian_ctx(src: &GrayImage, group: &RuntimeRunner) -> GrayImage {
pub fn canny_ctx(src: &GrayImage, low_threshold: u8, high_threshold: u8, group: &RuntimeRunner) -> GrayImage {
```

**AFTER:**
```rust
pub fn sobel_magnitude_ctx(
    gx: &GrayImage,
    gy: &GrayImage,
    group: &RuntimeRunner,
) -> Result<GrayImage> {
    if gx.width() != gy.width() || gx.height() != gy.height() {
        return Err(ImgprocError::DimensionMismatch("Gradient images must have same dimensions".into()));
    }
    // ... existing implementation
    Ok(result)
}

pub fn laplacian_ctx(src: &GrayImage, group: &RuntimeRunner) -> Result<GrayImage> {
    validate_image_size(src.width(), src.height())?;
    // ... existing implementation
    Ok(result)
}

pub fn canny_ctx(
    src: &GrayImage,
    low_threshold: u8,
    high_threshold: u8,
    group: &RuntimeRunner,
) -> Result<GrayImage> {
    validate_image_size(src.width(), src.height())?;
    if low_threshold >= high_threshold {
        return Err(ImgprocError::AlgorithmError(
            "low_threshold must be less than high_threshold".into()
        ));
    }
    // ... existing implementation
    Ok(result)
}
```

#### cv-features/fast.rs

**BEFORE:**
```rust
pub fn fast_detect(image: &GrayImage, threshold: u8, max_keypoints: usize) -> KeyPoints {
pub fn corner_score(image: &GrayImage, x: i32, y: i32, _threshold: u8) -> u8 {
pub fn non_max_suppression(keypoints: KeyPoints, image: &GrayImage, threshold: u8) -> KeyPoints {
```

**AFTER:**
```rust
pub fn fast_detect(
    image: &GrayImage,
    threshold: u8,
    max_keypoints: usize,
) -> Result<KeyPoints> {
    if max_keypoints == 0 {
        return Err(FeatureError::InvalidParameters("max_keypoints must be > 0".into()));
    }
    // ... existing implementation
    Ok(KeyPoints { keypoints: result })
}

pub fn corner_score(image: &GrayImage, x: i32, y: i32, _threshold: u8) -> Result<u8> {
    if x < 0 || y < 0 || x >= image.width() as i32 || y >= image.height() as i32 {
        return Err(FeatureError::InvalidParameters("Coordinates out of bounds".into()));
    }
    // ... existing implementation
    Ok(score)
}

pub fn non_max_suppression(
    keypoints: KeyPoints,
    image: &GrayImage,
    threshold: u8,
) -> Result<KeyPoints> {
    if image.width() == 0 || image.height() == 0 {
        return Err(FeatureError::InvalidParameters("Invalid image dimensions".into()));
    }
    // ... existing implementation
    Ok(KeyPoints { keypoints: result })
}
```

#### cv-features/harris.rs

**BEFORE:**
```rust
pub fn harris_detect(...) -> KeyPoints {
pub fn shi_tomasi_detect(...) -> KeyPoints {
```

**AFTER:**
```rust
pub fn harris_detect(
    image: &GrayImage,
    window_size: usize,
    aperture_size: usize,
    k: f32,
    threshold: f32,
) -> Result<KeyPoints> {
    if window_size == 0 || aperture_size == 0 {
        return Err(FeatureError::InvalidParameters("Window and aperture sizes must be > 0".into()));
    }
    // ... existing implementation
    Ok(KeyPoints { keypoints: result })
}

pub fn shi_tomasi_detect(
    image: &GrayImage,
    window_size: usize,
    aperture_size: usize,
    threshold: f32,
) -> Result<KeyPoints> {
    if window_size == 0 || aperture_size == 0 {
        return Err(FeatureError::InvalidParameters("Window and aperture sizes must be > 0".into()));
    }
    // ... existing implementation
    Ok(KeyPoints { keypoints: result })
}
```

#### cv-objdetect/hog.rs

**BEFORE:**
```rust
pub fn compute(&self, image: &GrayImage) -> Vec<f32> {
```

**AFTER:**
```rust
pub fn compute(&self, image: &GrayImage) -> Result<Vec<f32>> {
    if image.width() == 0 || image.height() == 0 {
        return Err(CvError::InvalidInput("Image dimensions must be non-zero".into()));
    }
    if (image.width() as usize) < self.params.cell_size ||
       (image.height() as usize) < self.params.cell_size {
        return Err(CvError::InvalidInput(
            format!("Image must be at least {}x{}", self.params.cell_size, self.params.cell_size)
        ));
    }
    // ... existing implementation
    Ok(descriptor)
}
```

#### cv-dnn/blob.rs

**BEFORE:**
```rust
pub fn image_to_blob(image: &GrayImage) -> Vec<f32> {
pub fn blob_to_image(blob: &[f32], width: u32, height: u32) -> GrayImage {
```

**AFTER:**
```rust
pub fn image_to_blob(image: &GrayImage) -> Result<Vec<f32>> {
    if image.width() == 0 || image.height() == 0 {
        return Err(CvError::InvalidInput("Image dimensions must be non-zero".into()));
    }
    // ... existing implementation
    Ok(blob)
}

pub fn blob_to_image(blob: &[f32], width: u32, height: u32) -> Result<GrayImage> {
    if width == 0 || height == 0 {
        return Err(CvError::InvalidInput("Dimensions must be non-zero".into()));
    }
    if blob.len() != (width as usize * height as usize) {
        return Err(CvError::InvalidInput(
            format!("Blob size {} doesn't match dimensions {}x{}",
                blob.len(), width, height)
        ));
    }
    // ... existing implementation
    Ok(image)
}
```

#### cv-photo/bilateral.rs

**BEFORE:**
```rust
pub fn bilateral_filter(src: &GrayImage, ...) -> GrayImage {
```

**AFTER:**
```rust
pub fn bilateral_filter(
    src: &GrayImage,
    diameter: i32,
    sigma_color: f32,
    sigma_space: f32,
) -> Result<GrayImage> {
    if src.width() == 0 || src.height() == 0 {
        return Err(CvError::InvalidInput("Image dimensions must be non-zero".into()));
    }
    if diameter <= 0 || sigma_color <= 0.0 || sigma_space <= 0.0 {
        return Err(CvError::InvalidInput("Parameters must be positive".into()));
    }
    // ... existing implementation
    Ok(result)
}
```

#### cv-registration/mod.rs

**BEFORE:**
```rust
pub fn registration_icp_point_to_plane(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f32,
    init_transformation: &Matrix4<f32>,
    max_iterations: usize,
) -> Option<ICPResult> {  // ← Should be Result
```

**AFTER:**
```rust
pub fn registration_icp_point_to_plane(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f32,
    init_transformation: &Matrix4<f32>,
    max_iterations: usize,
) -> Result<ICPResult> {
    if source.points.is_empty() || target.points.is_empty() {
        return Err(CvError::InvalidInput("Point clouds cannot be empty".into()));
    }
    if max_correspondence_distance <= 0.0 {
        return Err(CvError::InvalidInput("max_correspondence_distance must be positive".into()));
    }
    // ... existing implementation
    match result {
        Some(icp_result) => Ok(icp_result),
        None => Err(CvError::ComputationFailed("ICP did not converge".into())),
    }
}
```

---

## FIX #2: Compute Device Parameter Standardization

### PROBLEM ANALYSIS

Different crates use different types for compute device parameters:
- cv-features: `ctx: &ComputeDevice` (from cv-hal)
- cv-imgproc: `group: &RuntimeRunner` (from cv-runtime)

**Decision: Standardize on `ctx: &ComputeDevice`** (aligns with cv-hal, lower coupling)

### AFFECTED FUNCTIONS IN cv-imgproc

#### Change Parameter Type and Name

**BEFORE:**
```rust
pub fn sobel_ex_ctx(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    ksize: usize,
    scale: f32,
    delta: f32,
    border: BorderMode,
    group: &RuntimeRunner,  // ← Should be ctx: &ComputeDevice
) -> GrayImage {
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        // ...
    }
}

pub fn sobel_magnitude_ctx(
    gx: &GrayImage,
    gy: &GrayImage,
    group: &RuntimeRunner,  // ← Should be ctx: &ComputeDevice
) -> GrayImage {
    // ...
}

pub fn laplacian_ctx(
    src: &GrayImage,
    group: &RuntimeRunner,  // ← Should be ctx: &ComputeDevice
) -> GrayImage {
    // ...
}

pub fn canny_ctx(
    src: &GrayImage,
    low_threshold: u8,
    high_threshold: u8,
    group: &RuntimeRunner,  // ← Should be ctx: &ComputeDevice
) -> GrayImage {
    // ...
}
```

**AFTER:**
```rust
pub fn sobel_ex_ctx(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    ksize: usize,
    scale: f32,
    delta: f32,
    border: BorderMode,
    ctx: &ComputeDevice,  // ← Changed from group
) -> Result<GrayImage> {
    match ctx {
        ComputeDevice::Gpu(gpu) => {
            // ... GPU implementation
        }
        ComputeDevice::Cpu(_) => {
            // ... CPU implementation
        }
        ComputeDevice::Mlx(_) => {
            Err(ImgprocError::UnsupportedFormat("MLX not supported".into()))
        }
    }
}

pub fn sobel_magnitude_ctx(
    gx: &GrayImage,
    gy: &GrayImage,
    ctx: &ComputeDevice,  // ← Changed from group
) -> Result<GrayImage> {
    // ... implementation
    Ok(result)
}

pub fn laplacian_ctx(
    src: &GrayImage,
    ctx: &ComputeDevice,  // ← Changed from group
) -> Result<GrayImage> {
    // ... implementation
    Ok(result)
}

pub fn canny_ctx(
    src: &GrayImage,
    low_threshold: u8,
    high_threshold: u8,
    ctx: &ComputeDevice,  // ← Changed from group
) -> Result<GrayImage> {
    // ... implementation
    Ok(result)
}
```

#### Update Call Sites in Tests

**BEFORE:**
```rust
#[test]
fn test_sobel() {
    let runner = cv_runtime::default_runner().unwrap();
    let img = GrayImage::new(100, 100);
    let result = sobel_ex_ctx(&img, 1, 0, 3, 1.0, 0.0, BorderMode::Border, &runner);
}
```

**AFTER:**
```rust
#[test]
fn test_sobel() {
    let cpu = CpuBackend::new().unwrap();
    let ctx = ComputeDevice::Cpu(&cpu);
    let img = GrayImage::new(100, 100);
    let result = sobel_ex_ctx(&img, 1, 0, 3, 1.0, 0.0, BorderMode::Border, &ctx)?;
}
```

---

## FIX #3: Add Missing _ctx Variants

### Problem Analysis

Some compute functions don't have GPU-capable variants:
- cv-dnn blob functions (image_to_blob, blob_to_image)
- cv-objdetect hog compute
- cv-photo bilateral functions

### Solution: Create _ctx Variants

#### cv-dnn/blob.rs

**ADD:**
```rust
/// Convert image to blob with GPU acceleration support
///
/// # Parameters
/// * `image` - Input grayscale image
/// * `ctx` - Compute device (GPU preferred)
///
/// # Returns
/// Result containing normalized float blob
#[must_use]
pub fn image_to_blob_ctx<S: Storage<u8> + 'static>(
    image: &Tensor<u8, S>,
    ctx: &ComputeDevice,
) -> Result<Vec<f32>>
where
    Tensor<u8, S>: TensorToGpu<u8> + TensorToCpu<u8>,
{
    match ctx {
        ComputeDevice::Gpu(gpu) => {
            // GPU path: batch normalize on GPU
            let gpu_tensor = image.to_gpu_ctx(gpu)?;
            let gpu_f32 = <Tensor<u8, _> as TensorCast>::to_f32_ctx(&gpu_tensor, gpu)?;
            // GPU batch normalization
            let cpu_result = gpu_f32.to_cpu_ctx(gpu)?;
            cpu_result.as_slice()
                .ok_or_else(|| CvError::ComputationFailed("Failed to get slice".into()))
                .map(|s| s.iter().map(|&x| x).collect())
        }
        ComputeDevice::Cpu(_) => {
            // CPU path: existing implementation
            image_to_blob(image)
        }
        ComputeDevice::Mlx(_) => {
            Err(CvError::NotSupported("MLX backend not supported".into()))
        }
    }
}

/// Convert blob back to image with GPU acceleration support
///
/// # Parameters
/// * `blob` - Input float blob
/// * `width` - Image width
/// * `height` - Image height
/// * `ctx` - Compute device (GPU preferred)
///
/// # Returns
/// Result containing reconstructed image
#[must_use]
pub fn blob_to_image_ctx<S: Storage<f32> + 'static>(
    blob: &Tensor<f32, S>,
    width: u32,
    height: u32,
    ctx: &ComputeDevice,
) -> Result<GrayImage>
where
    Tensor<f32, S>: TensorToGpu<f32> + TensorToCpu<f32>,
{
    match ctx {
        ComputeDevice::Gpu(gpu) => {
            // GPU path: denormalization on GPU
            let gpu_tensor = blob.to_gpu_ctx(gpu)?;
            // GPU denormalization
            let cpu_result = gpu_tensor.to_cpu_ctx(gpu)?;
            let data = cpu_result.as_slice()
                .ok_or_else(|| CvError::ComputationFailed("Failed to get slice".into()))?;
            blob_to_image(data, width, height)
        }
        ComputeDevice::Cpu(_) => {
            // CPU path: convert to slice and use existing function
            let data: Vec<f32> = blob.iter().copied().collect();
            blob_to_image(&data, width, height)
        }
        ComputeDevice::Mlx(_) => {
            Err(CvError::NotSupported("MLX backend not supported".into()))
        }
    }
}
```

#### cv-objdetect/hog.rs

**ADD:**
```rust
/// Compute HOG descriptor with GPU acceleration support
///
/// # Parameters
/// * `image` - Input image
/// * `ctx` - Compute device (GPU preferred)
///
/// # Returns
/// Result containing HOG descriptor vector
#[must_use]
pub fn compute_ctx<S: Storage<u8> + 'static>(
    &self,
    image: &Tensor<u8, S>,
    ctx: &ComputeDevice,
) -> Result<Vec<f32>>
where
    Tensor<u8, S>: TensorToGpu<u8> + TensorToCpu<u8>,
{
    match ctx {
        ComputeDevice::Gpu(gpu) => {
            // GPU path: compute HOG on GPU (parallelized)
            let gpu_tensor = image.to_gpu_ctx(gpu)?;
            // GPU HOG computation (pseudocode)
            let gpu_result = gpu.hog_compute(&gpu_tensor, self.params.cell_size)?;
            let cpu_result = gpu_result.to_cpu_ctx(gpu)?;
            let data = cpu_result.as_slice()
                .ok_or_else(|| CvError::ComputationFailed("Failed to get slice".into()))?;
            Ok(data.iter().copied().collect())
        }
        ComputeDevice::Cpu(_) => {
            // CPU path: convert to image and use existing implementation
            let cpu_img = image.to_cpu()?;
            self.compute(&cpu_img)
        }
        ComputeDevice::Mlx(_) => {
            Err(CvError::NotSupported("MLX backend not supported".into()))
        }
    }
}
```

#### cv-photo/bilateral.rs

**ADD:**
```rust
/// Apply bilateral filter with GPU acceleration support
///
/// # Parameters
/// * `src` - Source image
/// * `diameter` - Diameter of pixel neighborhood
/// * `sigma_color` - Filter sigma in the color space
/// * `sigma_space` - Filter sigma in the coordinate space
/// * `ctx` - Compute device (GPU preferred)
///
/// # Returns
/// Result containing filtered image
#[must_use]
pub fn bilateral_filter_ctx(
    src: &GrayImage,
    diameter: i32,
    sigma_color: f32,
    sigma_space: f32,
    ctx: &ComputeDevice,
) -> Result<GrayImage> {
    if src.width() == 0 || src.height() == 0 {
        return Err(CvError::InvalidInput("Image dimensions must be non-zero".into()));
    }
    if diameter <= 0 || sigma_color <= 0.0 || sigma_space <= 0.0 {
        return Err(CvError::InvalidInput("Parameters must be positive".into()));
    }

    match ctx {
        ComputeDevice::Gpu(gpu) => {
            // GPU path: bilateral filter on GPU (high performance)
            // Pseudocode for GPU implementation
            let gpu_tensor = src.clone().into();
            let gpu_result = gpu.bilateral_filter(&gpu_tensor, diameter, sigma_color, sigma_space)?;
            let cpu_result = gpu_result.to_cpu_ctx(gpu)?;
            Ok(cpu_result)
        }
        ComputeDevice::Cpu(_) => {
            // CPU path: existing implementation
            bilateral_filter(src, diameter, sigma_color, sigma_space)
        }
        ComputeDevice::Mlx(_) => {
            Err(CvError::NotSupported("MLX backend not supported".into()))
        }
    }
}
```

---

## FIX #4: Add #[must_use] Attributes

### Problem Analysis

Result-returning functions should have `#[must_use]` to prevent silent error ignoring.

### Solution: Add Attribute

**BEFORE:**
```rust
pub fn detect_ctx<S: Storage<u8> + 'static>(
    &self,
    ctx: &ComputeDevice,
    image: &Tensor<u8, S>,
) -> Result<KeyPoints> {
    // ...
}
```

**AFTER:**
```rust
#[must_use]
pub fn detect_ctx<S: Storage<u8> + 'static>(
    &self,
    ctx: &ComputeDevice,
    image: &Tensor<u8, S>,
) -> Result<KeyPoints> {
    // ...
}
```

**Add to all functions in:**
- cv-features: detect_ctx(), detect_and_compute_ctx(), detect_and_refine(), detect_and_compute()
- cv-video: apply_ctx(), compute()
- cv-dnn: load(), forward(), preprocess()
- cv-imgproc: (all Result-returning functions)
- cv-registration: All public functions

**Automation:**
```bash
# Find all Result-returning public functions
grep -n "pub fn.*Result<" features/src/*.rs | cut -d: -f1,2 | sort -u

# Add attribute to each one (manual or with sed)
```

---

## Implementation Priority

### PHASE 1: HIGH-PRIORITY FIXES (Week 1)
- [ ] **FIX #2:** Standardize compute device parameters in cv-imgproc (2-3 days)
  - Impact: HIGH (enables consistency across crates)
  - Complexity: MEDIUM
  - Breaking change: YES (public API change)

- [ ] **FIX #1:** Convert return types to Result<T> (3-4 days)
  - Impact: HIGH (enables error handling)
  - Complexity: HIGH (many functions, testing)
  - Breaking change: YES (public API change)

### PHASE 2: MEDIUM-PRIORITY FIXES (Week 2)
- [ ] **FIX #3:** Add missing _ctx variants (2 days)
  - Impact: MEDIUM (API completeness)
  - Complexity: MEDIUM
  - Breaking change: NO (additions only)

- [ ] **FIX #4:** Add #[must_use] attributes (1 day)
  - Impact: LOW-MEDIUM (hygiene)
  - Complexity: LOW
  - Breaking change: NO

### PHASE 3: DOCUMENTATION (Week 3)
- [ ] Improve doc comments for parameters
- [ ] Add examples for new _ctx variants
- [ ] Update migration guide for breaking changes

---

## Testing Strategy

### Unit Tests Update Required

**For each changed function, update tests:**

```rust
// OLD TEST
#[test]
fn test_threshold() {
    let img = GrayImage::new(100, 100);
    let result = threshold(&img, 128, 255, ThresholdType::Binary);
    assert!(!result.as_raw().is_empty());
}

// NEW TEST
#[test]
fn test_threshold() -> Result<()> {
    let img = GrayImage::new(100, 100);
    let result = threshold(&img, 128, 255, ThresholdType::Binary)?;
    assert!(!result.as_raw().is_empty());
    Ok(())
}

// ADD NEW TEST FOR ERROR CASES
#[test]
fn test_threshold_invalid_image() {
    let img = GrayImage::new(0, 0);  // Invalid
    let result = threshold(&img, 128, 255, ThresholdType::Binary);
    assert!(result.is_err());
}
```

### Integration Tests

```rust
// Test GPU vs CPU consistency
#[test]
fn test_sobel_gpu_cpu_parity() -> Result<()> {
    let img = create_test_image();

    let cpu = CpuBackend::new()?;
    let cpu_result = sobel_ex_ctx(&img, 1, 0, 3, 1.0, 0.0, BorderMode::Border,
                                  &ComputeDevice::Cpu(&cpu))?;

    // Compare with GPU (if available)
    Ok(())
}
```

---

## Migration Guide for Users

### Public API Changes

#### Before (Old API)
```rust
use cv_imgproc::*;

let img = GrayImage::new(100, 100);
let result = threshold(&img, 128, 255, ThresholdType::Binary);
let edge_img = canny(&img, 50, 150);
```

#### After (New API)
```rust
use cv_imgproc::*;
use cv_hal::compute::ComputeDevice;
use cv_hal::cpu::CpuBackend;

let img = GrayImage::new(100, 100);

// With error handling
let result = threshold(&img, 128, 255, ThresholdType::Binary)?;

// With GPU acceleration
let cpu = CpuBackend::new()?;
let ctx = ComputeDevice::Cpu(&cpu);
let edge_img = canny_ctx(&img, 50, 150, &ctx)?;
```

---

## Backward Compatibility Notes

### Breaking Changes (Unavoidable)

1. **Return type changes:** All functions returning bare types now return Result<T>
   - This breaks code that doesn't use `?` operator
   - Migration: Add `?` or `.unwrap()` at call sites

2. **Parameter type changes:** RuntimeRunner → ComputeDevice
   - This breaks code passing group parameter
   - Migration: Create ComputeDevice from backend instead

### Non-Breaking Changes

1. **Adding _ctx variants:** Purely additive, no breaking changes
2. **Adding #[must_use]:** Warning-only, code still compiles

---

## Sign-Off Checklist

Before committing Phase 1-2 fixes:

- [ ] All tests pass locally
- [ ] No new compiler warnings
- [ ] Documentation updated
- [ ] Examples updated
- [ ] Migration guide prepared
- [ ] Breaking changes documented in CHANGELOG
- [ ] Code reviewed by maintainer
- [ ] Git history is clean and logical

---

## Related Issues & PRs

(To be filled in after implementation)

- Issue #xxx: Parameter naming standardization decision
- Issue #yyy: Return type standardization effort
- PR #zzz: Fixes for Phase 1 changes

