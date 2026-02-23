# API Consistency Audit Report

**Date:** 2026-02-24
**Scope:** Comprehensive audit of public API consistency across cv-native codebase
**Priority Crates:** cv-features, cv-video, cv-imgproc, cv-objdetect, cv-registration, cv-dnn, cv-photo

---

## Executive Summary

The codebase demonstrates **inconsistent API patterns** across crates, particularly in:
- **Parameter naming** for GPU/compute device contexts (mixed: `ctx`, `gpu`, `runner`, `group`)
- **Return type patterns** (inconsistent: bare types vs. Result<T>)
- **Function naming conventions** (_ctx variants, lack of standardization)
- **Access patterns** (public variants missing documentation)

**Critical Issues:** 2 High priority
**Moderate Issues:** 5 Medium priority
**Minor Issues:** 3 Low priority

---

## 1. NAMING CONSISTENCY

### 1.1 Function Naming: Excellent ✓

**Finding:** Snake_case is consistently used across all public functions.

**Examples:**
- `detect_and_compute_ctx()`, `gaussian_blur_ctx()`, `bilateral_filter_ctx()`
- `fast_detect()`, `harris_detect()`, `sobel_magnitude_ctx()`
- `registration_icp_point_to_plane()`, `registration_gnc()`

**Status:** COMPLIANT - No changes needed.

---

### 1.2 Type Naming: Excellent ✓

**Finding:** PascalCase is consistently used for all public types.

**Examples:**
- `AkazeParams`, `Mog2`, `HogParams`, `Sift`
- `KeyPoints`, `Descriptors`, `MotionField`, `ComputeDevice`
- `CascadeStage`, `HaarFeature`, `ICPResult`, `GNCOptimizer`

**Status:** COMPLIANT - No changes needed.

---

### 1.3 Function Variant Patterns: INCONSISTENT ⚠️

**Issue:** Mixed naming patterns for compute-device variants.

**Pattern 1: `_ctx` suffix (PREFERRED)**
- cv-features: `detect_ctx()`, `detect_and_compute_ctx()`, `compute_orientations_ctx()`
- cv-video: `apply_ctx()`
- cv-imgproc: `gaussian_blur_ctx()`, `convolve_ctx()`, `bilateral_filter_depth_ctx()`, `gaussian_blur_with_border_into_ctx()`
- cv-registration: (Not used - functions return Option instead of Result)

**Pattern 2: `_group` parameter (Runtime variant)**
- cv-imgproc: `sobel_magnitude_ctx()`, `laplacian_ctx()`, `canny_ctx()` - Use `group: &RuntimeRunner` parameter

**Pattern 3: No variant (CPU-only)**
- cv-objdetect: `compute()` in Hog (no GPU variant)
- cv-dnn: `image_to_blob()`, `blob_to_image()` (CPU only)
- cv-photo: `bilateral_filter()` (CPU only)

**Pattern 4: Private GPU variants**
- cv-imgproc: `gaussian_blur_gpu()`, `convolve_gpu()`, `canny_gpu()` (private helpers)
- cv-features: `detect_aruco_markers_gpu()`, `detect_apriltags_gpu()` (private)

**Examples of inconsistency:**

```rust
// cv-features/akaze.rs - uses 'ctx' parameter
pub fn detect_ctx<S: Storage<u8>>(
    &self,
    ctx: &ComputeDevice,      // ← 'ctx' naming
    image: &Tensor<u8, S>,
) -> Result<KeyPoints>

// cv-imgproc/edges.rs - uses 'group' parameter
pub fn sobel_magnitude_ctx(
    gx: &GrayImage,
    gy: &GrayImage,
    group: &RuntimeRunner,    // ← 'group' naming (different from ctx)
) -> GrayImage

// cv-imgproc/edges.rs - uses 'group' parameter
pub fn canny_ctx(
    src: &GrayImage,
    low_threshold: u8,
    high_threshold: u8,
    group: &RuntimeRunner,    // ← 'group' naming (different from ctx)
) -> GrayImage

// INCONSISTENT: Some _ctx functions don't have _ctx variants
// cv-objdetect/hog.rs
pub fn compute(&self, image: &GrayImage) -> Vec<f32> {
    // No _ctx variant despite this being a compute function
}

// cv-dnn/blob.rs
pub fn image_to_blob(image: &GrayImage) -> Vec<f32> {
    // No _ctx variant
}
```

**Recommendation:**
- Standardize on ONE parameter name across crates
- **Proposal A (Preferred):** Use `ctx: &ComputeDevice` for GPU/compute variants (cv-features style)
- **Proposal B (Alternative):** Use `runner: &RuntimeRunner` for runtime variants (cv-imgproc style)
- Document which compute functions should have variants

**Count:** 15+ functions with mixed naming
**Severity:** MEDIUM

---

## 2. PARAMETER PATTERNS

### 2.1 GPU/Device Parameters: INCONSISTENT ⚠️

**Issue:** Different types and names used for compute device parameters.

**Type 1: `ComputeDevice` (HAL abstraction)**
```rust
// cv-features/akaze.rs
pub fn detect_ctx(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> Result<KeyPoints>

// cv-features/orb.rs
pub fn detect_ctx<S: Storage<u8>>(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> Result<KeyPoints>
```

**Type 2: `RuntimeRunner` (Runtime abstraction)**
```rust
// cv-imgproc/edges.rs
pub fn sobel_magnitude_ctx(gx: &GrayImage, gy: &GrayImage, group: &RuntimeRunner) -> GrayImage

pub fn canny_ctx(src: &GrayImage, low_threshold: u8, high_threshold: u8, group: &RuntimeRunner) -> GrayImage

// cv-imgproc/convolve.rs
pub fn convolve_ctx(input: &GrayImage, kernel: &Kernel, group: &RuntimeRunner) -> Result<GrayImage>
```

**Type 3: No standard parameter (CPU-only functions)**
```rust
// cv-dnn/blob.rs
pub fn image_to_blob(image: &GrayImage) -> Vec<f32> {
    // CPU only, no compute parameter needed
}

// cv-photo/bilateral.rs
pub fn bilateral_filter(src: &GrayImage, ...) -> GrayImage {
    // CPU only
}
```

**Analysis:**
- `ComputeDevice` is from cv-hal (lower-level HAL abstraction)
- `RuntimeRunner` is from cv-runtime (higher-level orchestration)
- **Problem:** Functions using the same compute device have different parameter types

**Parameter Naming Variants:**
- `ctx: &ComputeDevice` (6 functions in cv-features)
- `gpu: &GpuContext` (4 functions in cv-features private helpers)
- `group: &RuntimeRunner` (8+ functions in cv-imgproc)
- `runner: &RuntimeRunner` (mentioned in dnn, not systematically used)

**Recommendation:**
- **Option A (Preferred):** Standardize on `ctx: &ComputeDevice` for HAL-level operations
  - Benefits: Lower-level, more composable, aligns with cv-hal design
  - Action: Migrate cv-imgproc functions to use ComputeDevice instead of RuntimeRunner

- **Option B:** Standardize on `runner: &RuntimeRunner` for all compute variants
  - Benefits: Higher-level orchestration, consistent naming
  - Cost: More dependencies on cv-runtime

**Count:** 20+ functions with inconsistent parameter types
**Severity:** MEDIUM-HIGH

---

### 2.2 Method Parameters: EXCELLENT ✓

**Finding:** Other parameters (thresholds, kernel sizes, etc.) are well-named and consistent.

**Examples:**
- `threshold: u8`, `max_value: u8`, `radius: usize`
- `dx: i32`, `dy: i32`, `ksize: usize`
- `max_iterations: usize`, `learning_rate: f32`
- `window_size: usize`, `pyramid_levels: usize`

**Status:** COMPLIANT - No changes needed.

---

## 3. RETURN TYPE PATTERNS

### 3.1 Result<T> Pattern: INCONSISTENT ⚠️

**Issue:** Inconsistent use of Result<T> vs. bare types vs. Option<T>.

**Pattern 1: Correct Result<T> Usage** ✓
```rust
// cv-features/akaze.rs
pub fn detect_ctx(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> Result<KeyPoints>
pub fn detect_and_compute_ctx(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> Result<(KeyPoints, Descriptors)>

// cv-video/mog2.rs
pub fn apply_ctx<S: Storage<u8>>(&mut self, frame: &Tensor<u8, S>, learning_rate: f32, ctx: &ComputeDevice) -> Result<CpuTensor<u8>>

// cv-dnn/lib.rs
pub fn load<P: AsRef<Path>>(path: P) -> Result<Self>
pub fn forward(&self, input: &Tensor<f32>) -> Result<Vec<Tensor<f32>>>
```

**Pattern 2: Bare Type Returns (No error handling)** ✗
```rust
// cv-imgproc/threshold.rs
pub fn threshold(src: &GrayImage, thresh: u8, max_value: u8, typ: ThresholdType) -> GrayImage {
pub fn threshold_otsu(src: &GrayImage, max_value: u8, typ: ThresholdType) -> (u8, GrayImage) {

// cv-imgproc/edges.rs
pub fn sobel_magnitude_ctx(gx: &GrayImage, gy: &GrayImage, group: &RuntimeRunner) -> GrayImage {
pub fn laplacian_ctx(src: &GrayImage, group: &RuntimeRunner) -> GrayImage {
pub fn canny_ctx(src: &GrayImage, low_threshold: u8, high_threshold: u8, group: &RuntimeRunner) -> GrayImage {

// cv-features/fast.rs
pub fn fast_detect(image: &GrayImage, threshold: u8, max_keypoints: usize) -> KeyPoints {
pub fn corner_score(image: &GrayImage, x: i32, y: i32, threshold: u8) -> u8 {
pub fn non_max_suppression(keypoints: KeyPoints, image: &GrayImage, threshold: u8) -> KeyPoints {

// cv-features/harris.rs
pub fn harris_detect(...) -> KeyPoints {
pub fn shi_tomasi_detect(...) -> KeyPoints {

// cv-objdetect/hog.rs
pub fn compute(&self, image: &GrayImage) -> Vec<f32> {

// cv-dnn/blob.rs
pub fn image_to_blob(image: &GrayImage) -> Vec<f32> {
pub fn blob_to_image(blob: &[f32], width: u32, height: u32) -> GrayImage {

// cv-photo/bilateral.rs
pub fn bilateral_filter(src: &GrayImage, ...) -> GrayImage {

// cv-video/kalman.rs
pub fn predict(&mut self, control: Option<&DVector<f64>>) -> &DVector<f64> {
pub fn correct(&mut self, measurement: &DVector<f64>) -> &DVector<f64> {
```

**Pattern 3: Option<T> Returns** (Partial error handling)
```rust
// cv-registration/mod.rs
pub fn registration_icp_point_to_plane(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f32,
    init_transformation: &Matrix4<f32>,
    max_iterations: usize,
) -> Option<ICPResult> {  // ← Should be Result<ICPResult, RegistrationError>
```

**Analysis:**

| Category | Count | Examples | Issue |
|----------|-------|----------|-------|
| Result<T> | ~15 | akaze, sift, video, dnn | Correct pattern ✓ |
| Bare Type | ~25 | edges, threshold, fast, harris | No error information ✗ |
| Option<T> | ~5 | registration, optical_flow | Loses error context ✗ |

**Problems with bare types:**
1. Cannot distinguish between invalid input and computation failure
2. Panic possibilities hidden in implementations
3. Breaks error handling chain (can't propagate with ?)
4. Example: `threshold()` returns GrayImage - what if dimensions are invalid?

**Problems with Option<T>:**
1. Can't provide error information
2. Registration errors silently become None
3. Hard to debug failures

**Recommendation:**
- **Immediate action:** All public compute functions should return `Result<T, CvError>`
- **Priority 1 (HIGH):** Migrate bare type returns in cv-imgproc, cv-features, cv-objdetect
- **Priority 2 (MEDIUM):** Convert Option<T> to Result<T> in cv-registration, cv-video

**Impacted functions:** ~35 functions
**Severity:** HIGH

---

### 3.2 Result Type Consistency: EXCELLENT ✓

**Finding:** All Result-returning functions use consistent error types from their respective crates.

**Examples:**
```rust
// cv-features
pub type Result<T> = std::result::Result<T, FeatureError>;

// cv-imgproc
pub type Result<T> = std::result::Result<T, ImgprocError>;

// cv-video
pub type Result<T> = cv_core::Result<T>;  // Reuses cv-core

// cv-dnn
pub type Result<T> = cv_core::Result<T>;  // Reuses cv-core
```

**Status:** COMPLIANT - Error types are well-defined and crate-local.

---

## 4. FUNCTION VARIANTS PATTERNS

### 4.1 GPU/CPU Variants: PARTIALLY IMPLEMENTED ⚠️

**Finding:** Some crates provide _ctx variants, others don't.

**Well-implemented:**
- cv-features: akaze.rs, orb.rs have detect_ctx(), detect_and_compute_ctx()
- cv-video: mog2.rs has apply_ctx()
- cv-imgproc: bilateral.rs, convolve.rs have _ctx variants

**Missing variants:**
```rust
// cv-objdetect - NO VARIANTS
pub fn compute(&self, image: &GrayImage) -> Vec<f32> {
    // Should have compute_ctx(&self, image: &GrayImage, ctx: &ComputeDevice) -> Result<Vec<f32>>
}

// cv-dnn - NO VARIANTS
pub fn image_to_blob(image: &GrayImage) -> Vec<f32> {
    // Should have image_to_blob_ctx() variant

pub fn blob_to_image(blob: &[f32], width: u32, height: u32) -> GrayImage {
    // Should have blob_to_image_ctx() variant

// cv-photo - NO VARIANTS
pub fn bilateral_filter(src: &GrayImage, ...) -> GrayImage {
    // Should have bilateral_filter_ctx() variant
```

**Pattern Issues:**
1. Some "simple" functions (threshold, blur) do have _ctx variants
2. Some "simple" functions (HOG, bilateral) don't
3. No clear policy on when to provide variants

**Recommendation:**
- **Policy:** All compute-intensive public functions should provide _ctx variants
- **Definition:** Functions that can benefit from GPU acceleration or runtime scheduling
- **Coverage:** dnn blob functions, objdetect hog, photo bilateral all should have variants

**Severity:** MEDIUM

---

### 4.2 Naming Convention for Variants: INCONSISTENT ⚠️

**Issue:** Private GPU helpers use different naming than public functions.

**Public variants:**
- `{function}_ctx()` - takes ComputeDevice or RuntimeRunner

**Private GPU helpers:**
- `{function}_gpu()` (INCONSISTENT naming)
- Example: `gaussian_blur_gpu()`, `convolve_gpu()`, `canny_gpu()` (cv-imgproc)
- Example: `detect_aruco_markers_gpu()`, `detect_apriltags_gpu()` (cv-features)

**Better approach:** Private helpers should be internal implementation details, not exposed in naming.

**Recommendation:**
- Keep `_ctx` as the ONLY public variant suffix
- Use `impl_gpu()` or similar for private helpers if needed
- Document that _ctx variants automatically select GPU when available

**Severity:** LOW

---

## 5. PUBLIC VS PRIVATE

### 5.1 Function Visibility: GOOD ✓

**Finding:** Public vs private boundary is mostly well-maintained.

**Properly Private:**
- `gaussian_blur_gpu()`, `convolve_gpu()`, `canny_gpu()` in cv-imgproc
- `detect_aruco_markers_gpu()` in cv-features
- `to_cpu_f32()`, `convert_to_f32_cpu()` in cv-features

**Properly Public:**
- User-facing algorithms: detect(), compute(), apply()
- Construction functions: new(), with_*()
- Core operations: threshold(), sobel(), bilateral_filter()

**Status:** COMPLIANT - Visibility is appropriate.

---

### 5.2 pub_crate Items: GOOD ✓

**Finding:** Limited use of pub_crate; not a major concern.

**Status:** COMPLIANT - No issues found.

---

## 6. ERROR HANDLING & MUST_USE ATTRIBUTE

### 6.1 Missing #[must_use] Attributes: ⚠️

**Issue:** Result-returning functions should have #[must_use] to prevent silent failures.

**Missing from:**
- cv-features/akaze.rs: detect_ctx(), detect_and_compute_ctx()
- cv-features/sift.rs: detect_and_compute(), detect_and_refine()
- cv-video/mog2.rs: apply_ctx()
- cv-dnn/lib.rs: load(), forward(), preprocess()

**Recommendation:**
- Add #[must_use] to ALL public functions returning Result<T>
- Example:
```rust
#[must_use]
pub fn detect_ctx(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> Result<KeyPoints>
```

**Severity:** LOW (Hygiene issue)

---

## 7. DOCUMENTATION

### 7.1 Parameter Documentation: INCONSISTENT ⚠️

**Issue:** Parameters like `ctx`, `gpu`, `group` lack inline documentation.

**Missing explanations:**
```rust
// Should document what 'ctx' means:
pub fn detect_ctx(
    &self,
    ctx: &ComputeDevice,  // ← What is this? GPU? CPU? Both?
    image: &Tensor<u8, S>,
) -> Result<KeyPoints>

// Better:
/// Detect features using specified compute device.
///
/// # Parameters
/// * `ctx` - Compute device (GPU preferred for performance, CPU fallback available)
/// * `image` - Input image tensor
pub fn detect_ctx(
    &self,
    ctx: &ComputeDevice,
    image: &Tensor<u8, S>,
) -> Result<KeyPoints>
```

**Severity:** LOW (Documentation issue)

---

## 8. CROSS-CRATE CONSISTENCY

### 8.1 Similar Functions in Different Crates

**Issue:** Same algorithms implemented in different crates with different APIs.

**Example 1: Gaussian Blur**
```rust
// cv-imgproc/convolve.rs
pub fn gaussian_blur_ctx(
    input: &GrayImage,
    sigma: f32,
    ksize: usize,
    group: &RuntimeRunner,
) -> Result<GrayImage>

// cv-features/akaze.rs (internal usage)
ctx.gaussian_blur(image, 1.0, 5)  // Method on ComputeDevice

// INCONSISTENT: Different parameter types (RuntimeRunner vs ComputeDevice)
```

**Example 2: Sobel**
```rust
// cv-imgproc/edges.rs
pub fn sobel_magnitude_ctx(gx: &GrayImage, gy: &GrayImage, group: &RuntimeRunner) -> GrayImage

// cv-hal/compute.rs
pub fn sobel<S: Storage<u8>>(
    &self,
    input: &Tensor<u8, S>,
    dx: i32,
    dy: i32,
    ksize: usize,
) -> Result<(Tensor<u8, S>, Tensor<u8, S>)>

// INCONSISTENT: Return type, parameter structure
```

**Severity:** MEDIUM (Creates confusion about which function to use)

---

## 9. SUMMARY TABLE

| Category | Status | Count | Severity | Priority |
|----------|--------|-------|----------|----------|
| Function naming (snake_case) | ✓ COMPLIANT | - | - | - |
| Type naming (PascalCase) | ✓ COMPLIANT | - | - | - |
| Function variants (_ctx suffix) | ⚠️ INCONSISTENT | 15+ | MEDIUM | MEDIUM |
| Parameter names (ctx/gpu/runner) | ⚠️ INCONSISTENT | 20+ | MEDIUM-HIGH | MEDIUM |
| Return types (Result vs bare) | ⚠️ INCONSISTENT | ~35 | HIGH | HIGH |
| GPU/CPU variants coverage | ⚠️ PARTIAL | 8+ | MEDIUM | MEDIUM |
| Public/Private visibility | ✓ GOOD | - | - | - |
| #[must_use] attributes | ⚠️ MISSING | ~15 | LOW | LOW |
| Documentation | ⚠️ INCOMPLETE | 20+ | LOW | LOW |
| Cross-crate consistency | ⚠️ ISSUES | 5+ | MEDIUM | MEDIUM |

---

## 10. CRITICAL RECOMMENDATIONS

### TIER 1 (HIGH PRIORITY - DO FIRST)

1. **Standardize Return Types**
   - Convert all `pub fn` compute functions returning bare types to `Result<T, CvError>`
   - Files affected: cv-imgproc (threshold, edges, etc.), cv-features (fast, harris), cv-objdetect, cv-photo
   - Effort: HIGH
   - Impact: HIGH (enables error handling, consistency)

2. **Standardize Compute Device Parameter**
   - **Decision:** Use `ctx: &ComputeDevice` consistently (aligns with cv-hal design)
   - Migrate cv-imgproc functions from `group: &RuntimeRunner` to `ctx: &ComputeDevice`
   - OR migrate all to `runner: &RuntimeRunner` (less preferred, higher coupling)
   - Files affected: cv-imgproc (8+ functions)
   - Effort: MEDIUM
   - Impact: HIGH (enables code reuse, clear semantics)

### TIER 2 (MEDIUM PRIORITY - DO AFTER TIER 1)

3. **Add GPU Variants to Missing Functions**
   - Provide _ctx variants for: dnn blob functions, objdetect hog, photo bilateral
   - Benefits: Consistent API surface
   - Effort: MEDIUM
   - Impact: MEDIUM

4. **Add #[must_use] to Result-Returning Functions**
   - Prevents accidental error ignoring
   - Files: akaze, sift, mog2, dnn
   - Effort: LOW
   - Impact: LOW-MEDIUM (hygiene)

### TIER 3 (LOW PRIORITY - DOCUMENTATION/POLISH)

5. **Improve Documentation**
   - Document ctx/gpu/runner parameters clearly
   - Explain when to use _ctx vs non-ctx variants
   - Effort: LOW
   - Impact: LOW (documentation)

---

## 11. IMPLEMENTATION ROADMAP

### Phase 1: Planning & Decision (1 day)
- [ ] Team decision on parameter naming: ctx vs runner
- [ ] Policy document: when to provide _ctx variants
- [ ] Policy document: return type expectations

### Phase 2: Return Types (2-3 days)
- [ ] Update cv-imgproc threshold, edges, morph, bilateral to return Result<T>
- [ ] Update cv-features fast, harris, gftt to return Result<KeyPoints>
- [ ] Update cv-objdetect hog to return Result<Vec<f32>>
- [ ] Update cv-dnn blob functions to return Result<Vec<f32>>
- [ ] Update cv-photo bilateral functions to return Result<GrayImage>

### Phase 3: Parameter Standardization (2-3 days)
- [ ] Migrate cv-imgproc functions to use ComputeDevice (if ctx chosen)
- [ ] Update cv-imgproc function signatures
- [ ] Test GPU/CPU fallback logic

### Phase 4: Missing Variants (1-2 days)
- [ ] Add _ctx variants to dnn, objdetect, photo functions
- [ ] Ensure GPU fallback logic in place

### Phase 5: Attributes & Documentation (1 day)
- [ ] Add #[must_use] to Result-returning functions
- [ ] Improve doc comments for compute device parameters

---

## 12. DETAILED FINDINGS BY CRATE

### cv-features
**Status:** PARTIALLY COMPLIANT
- ✓ Good: akaze, sift, orb have _ctx variants with Result<T>
- ✓ Good: Function naming consistent
- ✗ Bad: fast, harris, gftt return bare KeyPoints (should be Result)
- ✗ Bad: Parameter names use `ctx` (inconsistent with imgproc's `group`)

### cv-video
**Status:** PARTIALLY COMPLIANT
- ✓ Good: mog2.apply_ctx() returns Result<T>
- ✓ Good: kalman has clear API (though returns refs, not Results)
- ✗ Bad: optical_flow lacks _ctx variants

### cv-imgproc
**Status:** PARTIALLY COMPLIANT
- ✗ Bad: threshold, edges, morph return bare types (should be Result)
- ✗ Bad: Functions use `group: &RuntimeRunner` (inconsistent with features' `ctx`)
- ✓ Good: _ctx variants widely provided
- ✓ Good: bilateral_filter_depth_ctx() has proper API

### cv-objdetect
**Status:** NEEDS WORK
- ✗ Bad: hog.compute() returns bare Vec<f32> (should be Result)
- ✗ Bad: No _ctx variants
- ✗ Bad: No compute device parameter

### cv-dnn
**Status:** NEEDS WORK
- ✗ Bad: image_to_blob(), blob_to_image() return bare types
- ✗ Bad: No _ctx variants
- ✗ Bad: No compute device parameter

### cv-photo
**Status:** NEEDS WORK
- ✗ Bad: bilateral_filter() returns bare GrayImage
- ✗ Bad: No _ctx variants
- ✗ Bad: No compute device parameter

### cv-registration
**Status:** NEEDS IMPROVEMENT
- ✗ Bad: registration_icp_point_to_plane() returns Option<ICPResult> (should be Result)
- ✗ Bad: No _ctx variants
- ✗ Bad: Inconsistent error handling

---

## Conclusion

The codebase shows **good naming consistency** (snake_case/PascalCase) but **moderate-to-high inconsistencies** in:
1. **Return type patterns** (bare vs Result vs Option) - CRITICAL
2. **Compute device parameter naming** (ctx vs group) - HIGH PRIORITY
3. **GPU variant coverage** (missing in several crates) - MEDIUM PRIORITY

**Recommended immediate action:** Standardize return types to Result<T> across all public compute functions and establish a clear parameter naming convention for compute devices.
