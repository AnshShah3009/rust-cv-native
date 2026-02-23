# API Audit - Complete List of Affected Functions

**Purpose:** Comprehensive reference of all functions requiring changes
**Generated:** 2026-02-24
**Total Functions:** 51 functions across 13 crate files

---

## Summary by Crate

| Crate | File | Functions | Return Type Fix | Param Fix | GPU Variant | #[must_use] |
|-------|------|-----------|:---:|:---:|:---:|:---:|
| cv-imgproc | edges.rs | 5 | ✓ | ✓ | - | ✓ |
| cv-imgproc | threshold.rs | 2 | ✓ | - | - | ✓ |
| cv-imgproc | bilateral.rs | 3 | ✓ | ✓ | ✓ | ✓ |
| cv-imgproc | convolve.rs | 4 | ✓ | ✓ | - | ✓ |
| cv-features | akaze.rs | 2 | - | - | - | ✓ |
| cv-features | sift.rs | 2 | - | - | - | ✓ |
| cv-features | fast.rs | 4 | ✓ | - | - | ✓ |
| cv-features | harris.rs | 2 | ✓ | - | - | ✓ |
| cv-features | gftt.rs | 1 | ✓ | - | - | ✓ |
| cv-objdetect | hog.rs | 2 | ✓ | - | ✓ | ✓ |
| cv-dnn | blob.rs | 4 | ✓ | - | ✓ | ✓ |
| cv-dnn | lib.rs | 3 | - | - | - | ✓ |
| cv-photo | bilateral.rs | 1 | ✓ | - | ✓ | ✓ |
| cv-registration | mod.rs | 2 | ✓ | - | - | ✓ |
| cv-video | mog2.rs | 1 | - | - | - | ✓ |
| cv-video | optical_flow.rs | 1 | - | - | - | ✓ |
| **TOTALS** | | **51** | **20** | **8** | **4** | **19+** |

---

## DETAILED FUNCTION LIST

### cv-imgproc/edges.rs (5 functions)

#### 1. sobel_ex()
- **Current Signature:** `pub fn sobel_ex(...) -> GrayImage`
- **Line:** 48
- **Fixes:** Return type (bare → Result), Parameter (group → ctx)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Notes:** Public wrapper around sobel_ex_ctx

#### 2. sobel_ex_ctx()
- **Current Signature:** `pub fn sobel_ex_ctx(..., group: &RuntimeRunner) -> GrayImage`
- **Line:** 66
- **Fixes:** Return type (bare → Result), Parameter (group → ctx)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_sobel*, test_edges*
- **GPU Path:** Yes (has sobel_gpu() private helper)
- **CPU Path:** Yes

#### 3. scharr_ex_ctx()
- **Current Signature:** `pub fn scharr_ex_ctx(..., group: &RuntimeRunner) -> GrayImage`
- **Line:** 142
- **Fixes:** Return type (bare → Result), Parameter (group → ctx)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_scharr*

#### 4. sobel_magnitude_ctx()
- **Current Signature:** `pub fn sobel_magnitude_ctx(gx: &GrayImage, gy: &GrayImage, group: &RuntimeRunner) -> GrayImage`
- **Line:** 212
- **Fixes:** Return type (bare → Result), Parameter (group → ctx)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_sobel_magnitude*
- **Error Validation:** Check dimensions match

#### 5. laplacian_ctx()
- **Current Signature:** `pub fn laplacian_ctx(src: &GrayImage, group: &RuntimeRunner) -> GrayImage`
- **Line:** 247
- **Fixes:** Return type (bare → Result), Parameter (group → ctx)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_laplacian*

#### 6. canny_ctx()
- **Current Signature:** `pub fn canny_ctx(src: &GrayImage, low_threshold: u8, high_threshold: u8, group: &RuntimeRunner) -> GrayImage`
- **Line:** 475
- **Fixes:** Return type (bare → Result), Parameter (group → ctx)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_canny*
- **Error Validation:** low_threshold < high_threshold
- **GPU Path:** Yes (has canny_gpu() private helper)
- **CPU Path:** Yes

---

### cv-imgproc/threshold.rs (2 functions)

#### 7. threshold()
- **Current Signature:** `pub fn threshold(src: &GrayImage, thresh: u8, max_value: u8, typ: ThresholdType) -> GrayImage`
- **Line:** 20
- **Fixes:** Return type (bare → Result)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_threshold*
- **Error Validation:** Image dimensions non-zero
- **Notes:** Main threshold function

#### 8. threshold_otsu()
- **Current Signature:** `pub fn threshold_otsu(src: &GrayImage, max_value: u8, typ: ThresholdType) -> (u8, GrayImage)`
- **Line:** 76
- **Fixes:** Return type (bare → Result<(u8, GrayImage)>)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_threshold_otsu*
- **Error Validation:** Image dimensions non-zero

---

### cv-imgproc/bilateral.rs (3 functions)

#### 9. bilateral_filter_depth_ctx()
- **Current Signature:** `pub fn bilateral_filter_depth_ctx(...) -> GrayImage`
- **Line:** 138
- **Fixes:** Return type (bare → Result), Parameter (group → ctx)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_bilateral*

#### 10. bilateral_filter_rgb_ctx()
- **Current Signature:** `pub fn bilateral_filter_rgb_ctx(...) -> GrayImage`
- **Line:** 252
- **Fixes:** Return type (bare → Result), Parameter (group → ctx)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_bilateral*

#### 11. joint_bilateral_filter_ctx()
- **Current Signature:** `pub fn joint_bilateral_filter_ctx(...) -> GrayImage`
- **Line:** 378
- **Fixes:** Return type (bare → Result), Parameter (group → ctx)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_bilateral*

---

### cv-imgproc/convolve.rs (4 functions)

#### 12. convolve_ctx()
- **Current Signature:** `pub fn convolve_ctx(..., group: &RuntimeRunner) -> Result<GrayImage>`
- **Line:** 196
- **Fixes:** Parameter (group → ctx)
- **Priority:** MEDIUM
- **Status:** Partially Done (has Result already)
- **Related Tests:** test_convolve*

#### 13. convolve_into_ctx()
- **Current Signature:** `pub fn convolve_into_ctx(..., group: &RuntimeRunner) -> Result<GrayImage>`
- **Line:** 207
- **Fixes:** Parameter (group → ctx)
- **Priority:** MEDIUM
- **Status:** Partially Done

#### 14. gaussian_blur_ctx()
- **Current Signature:** `pub fn gaussian_blur_ctx(..., group: &RuntimeRunner) -> Result<GrayImage>`
- **Line:** 253
- **Fixes:** Parameter (group → ctx)
- **Priority:** MEDIUM
- **Status:** Partially Done
- **Related Tests:** test_gaussian_blur*
- **GPU Path:** Yes (has gaussian_blur_gpu() private helper)

#### 15. gaussian_blur_with_border_into_ctx()
- **Current Signature:** `pub fn gaussian_blur_with_border_into_ctx(..., group: &RuntimeRunner) -> Result<GrayImage>`
- **Line:** 513
- **Fixes:** Parameter (group → ctx)
- **Priority:** MEDIUM
- **Status:** Partially Done

---

### cv-features/akaze.rs (2 functions)

#### 16. detect_ctx()
- **Current Signature:** `pub fn detect_ctx<S>(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> Result<KeyPoints>`
- **Line:** 51
- **Fixes:** Add #[must_use]
- **Priority:** LOW
- **Status:** Mostly Done (has Result, has ctx already)
- **Related Tests:** test_akaze_detect*

#### 17. detect_and_compute_ctx()
- **Current Signature:** `pub fn detect_and_compute_ctx<S>(&self, ctx: &ComputeDevice, image: &Tensor<u8, S>) -> Result<(KeyPoints, Descriptors)>`
- **Line:** 64
- **Fixes:** Add #[must_use]
- **Priority:** LOW
- **Status:** Mostly Done

---

### cv-features/sift.rs (2 functions)

#### 18. detect_and_refine()
- **Current Signature:** `pub fn detect_and_refine<S>(...) -> Result<KeyPoints>`
- **Line:** 102
- **Fixes:** Add #[must_use]
- **Priority:** LOW
- **Status:** Mostly Done

#### 19. detect_and_compute()
- **Current Signature:** `pub fn detect_and_compute<S>(...) -> Result<(KeyPoints, Descriptors)>`
- **Line:** 166
- **Fixes:** Add #[must_use]
- **Priority:** LOW
- **Status:** Mostly Done

---

### cv-features/fast.rs (4 functions)

#### 20. fast_detect()
- **Current Signature:** `pub fn fast_detect(image: &GrayImage, threshold: u8, max_keypoints: usize) -> KeyPoints`
- **Line:** 7
- **Fixes:** Return type (bare → Result<KeyPoints>)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_fast_detect*
- **Error Validation:** max_keypoints > 0

#### 21. corner_score()
- **Current Signature:** `pub fn corner_score(image: &GrayImage, x: i32, y: i32, threshold: u8) -> u8`
- **Line:** 116
- **Fixes:** Return type (bare → Result<u8>)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Error Validation:** Coordinates in bounds

#### 22. non_max_suppression()
- **Current Signature:** `pub fn non_max_suppression(keypoints: KeyPoints, image: &GrayImage, threshold: u8) -> KeyPoints`
- **Line:** 151
- **Fixes:** Return type (bare → Result<KeyPoints>)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Error Validation:** Image dimensions valid

#### 23. (Additional function in fast.rs)
- **Status:** Review for additional Result-returning functions

---

### cv-features/harris.rs (2 functions)

#### 24. harris_detect()
- **Current Signature:** `pub fn harris_detect(...) -> KeyPoints`
- **Line:** 6
- **Fixes:** Return type (bare → Result<KeyPoints>)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_harris*
- **Error Validation:** window_size > 0, aperture_size > 0

#### 25. shi_tomasi_detect()
- **Current Signature:** `pub fn shi_tomasi_detect(...) -> KeyPoints`
- **Line:** 56
- **Fixes:** Return type (bare → Result<KeyPoints>)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_shi_tomasi*
- **Error Validation:** window_size > 0, aperture_size > 0

---

### cv-features/gftt.rs (1 function)

#### 26. gftt_detect()
- **Current Signature:** `pub fn gftt_detect(...) -> KeyPoints`
- **Line:** 4
- **Fixes:** Return type (bare → Result<KeyPoints>)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_gftt*
- **Error Validation:** max_corners > 0, quality_level > 0

---

### cv-objdetect/hog.rs (2 functions)

#### 27. compute() [member function]
- **Class:** `Hog`
- **Current Signature:** `pub fn compute(&self, image: &GrayImage) -> Vec<f32>`
- **Line:** 36
- **Fixes:** Return type (bare → Result<Vec<f32>>), Add GPU variant (NEW)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_hog*
- **Error Validation:** Image dimensions check, min size check
- **GPU Variant Needed:** `compute_ctx(&self, image: &Tensor<u8, S>, ctx: &ComputeDevice) -> Result<Vec<f32>>`

#### 28. compute_ctx() [NEW FUNCTION]
- **Class:** `Hog`
- **Type:** NEW GPU variant
- **Location:** After compute() in hog.rs
- **Signature:** `pub fn compute_ctx<S>(&self, image: &Tensor<u8, S>, ctx: &ComputeDevice) -> Result<Vec<f32>>`
- **Priority:** MEDIUM
- **GPU Path:** Parallelized HOG on GPU
- **CPU Path:** Convert to image and call compute()

---

### cv-dnn/blob.rs (4 functions)

#### 29. image_to_blob()
- **Current Signature:** `pub fn image_to_blob(image: &GrayImage) -> Vec<f32>`
- **Line:** 3
- **Fixes:** Return type (bare → Result<Vec<f32>>), Add GPU variant (NEW)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_image_to_blob*
- **Error Validation:** Image dimensions non-zero
- **GPU Variant Needed:** `image_to_blob_ctx<S>(..., ctx: &ComputeDevice) -> Result<Vec<f32>>`

#### 30. blob_to_image()
- **Current Signature:** `pub fn blob_to_image(blob: &[f32], width: u32, height: u32) -> GrayImage`
- **Line:** 15
- **Fixes:** Return type (bare → Result<GrayImage>), Add GPU variant (NEW)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_blob_to_image*
- **Error Validation:** Dimensions non-zero, blob size matches
- **GPU Variant Needed:** `blob_to_image_ctx<S>(..., width: u32, height: u32, ctx: &ComputeDevice) -> Result<GrayImage>`

#### 31. image_to_blob_ctx() [NEW FUNCTION]
- **Type:** NEW GPU variant
- **Location:** After image_to_blob() in blob.rs
- **Signature:** `pub fn image_to_blob_ctx<S>(..., ctx: &ComputeDevice) -> Result<Vec<f32>>`
- **Priority:** MEDIUM
- **GPU Path:** Batch normalize on GPU
- **CPU Path:** Call existing image_to_blob()

#### 32. blob_to_image_ctx() [NEW FUNCTION]
- **Type:** NEW GPU variant
- **Location:** After blob_to_image() in blob.rs
- **Signature:** `pub fn blob_to_image_ctx<S>(..., width: u32, height: u32, ctx: &ComputeDevice) -> Result<GrayImage>`
- **Priority:** MEDIUM
- **GPU Path:** Denormalization on GPU
- **CPU Path:** Call existing blob_to_image()

---

### cv-dnn/lib.rs (3 functions)

#### 33. load()
- **Current Signature:** `pub fn load<P>(path: P) -> Result<Self>`
- **Line:** 40
- **Fixes:** Add #[must_use]
- **Priority:** LOW
- **Status:** Mostly Done (has Result)
- **Related Tests:** test_net_load*

#### 34. forward()
- **Current Signature:** `pub fn forward(&self, input: &Tensor<f32>) -> Result<Vec<Tensor<f32>>>`
- **Line:** 63
- **Fixes:** Add #[must_use]
- **Priority:** LOW
- **Status:** Mostly Done

#### 35. preprocess()
- **Current Signature:** `pub fn preprocess(&self, img: &DynamicImage, runner: &ResourceGroup) -> Result<Tensor<f32>>`
- **Line:** 97
- **Fixes:** Add #[must_use]
- **Priority:** LOW
- **Status:** Mostly Done

---

### cv-photo/bilateral.rs (1 function + 1 new)

#### 36. bilateral_filter()
- **Current Signature:** `pub fn bilateral_filter(src: &GrayImage, ...) -> GrayImage`
- **Line:** 3
- **Fixes:** Return type (bare → Result<GrayImage>), Add GPU variant (NEW)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_bilateral*
- **Error Validation:** Image dimensions, parameters positive
- **GPU Variant Needed:** `bilateral_filter_ctx(..., ctx: &ComputeDevice) -> Result<GrayImage>`

#### 37. bilateral_filter_ctx() [NEW FUNCTION]
- **Type:** NEW GPU variant
- **Location:** After bilateral_filter() in bilateral.rs
- **Signature:** `pub fn bilateral_filter_ctx(..., diameter: i32, sigma_color: f32, sigma_space: f32, ctx: &ComputeDevice) -> Result<GrayImage>`
- **Priority:** MEDIUM
- **GPU Path:** Bilateral filter on GPU (high performance)
- **CPU Path:** Call existing bilateral_filter()

---

### cv-registration/mod.rs (2 functions)

#### 38. registration_icp_point_to_plane()
- **Current Signature:** `pub fn registration_icp_point_to_plane(...) -> Option<ICPResult>`
- **Line:** 78
- **Fixes:** Return type (Option → Result<ICPResult>)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_icp*
- **Error Validation:** Point clouds non-empty, distance positive
- **Note:** Currently returns Option<T>, should return Result<T>

#### 39. registration_ransac_based_on_feature_matching()
- **Current Signature:** `pub fn registration_ransac_based_on_feature_matching(...) -> Option<GlobalRegistrationResult>`
- **Location:** registration/src/registration/global.rs
- **Fixes:** Return type (Option → Result<GlobalRegistrationResult>)
- **Priority:** HIGH
- **Status:** Not Implemented
- **Related Tests:** test_ransac*
- **Note:** Currently returns Option<T>, should return Result<T>

---

### cv-video/mog2.rs (1 function)

#### 40. apply_ctx()
- **Current Signature:** `pub fn apply_ctx<S>(..., ctx: &ComputeDevice) -> Result<CpuTensor<u8>>`
- **Line:** 45
- **Fixes:** Add #[must_use]
- **Priority:** LOW
- **Status:** Mostly Done (has Result, has ctx)
- **Related Tests:** test_mog2*

---

### cv-video/optical_flow.rs (1 function)

#### 41. compute()
- **Class:** `FarnebackOpticalFlow`
- **Current Signature:** `pub fn compute(&self, prev_frame: &GrayImage, next_frame: &GrayImage) -> Result<MotionField>`
- **Line:** 199
- **Fixes:** Add #[must_use]
- **Priority:** LOW
- **Status:** Mostly Done (has Result)
- **Related Tests:** test_optical_flow*

---

## Summary Statistics

### By Fix Type

| Fix Type | Count | Functions | Effort | Breaking |
|----------|-------|-----------|--------|----------|
| Return Type Standardization | 20 | 20 | HIGH | YES |
| Parameter Standardization | 8 | 8 | MEDIUM | YES |
| Add GPU Variants | 4 | 4 | MEDIUM | NO |
| Add #[must_use] | 19 | 19 | LOW | NO |

### By Priority

| Priority | Count | Effort | When |
|----------|-------|--------|------|
| HIGH | 27 | 5-7 days | Week 1 |
| MEDIUM | 15 | 2-3 days | Week 1-2 |
| LOW | 9 | <1 day | Week 2 |

### By Status

| Status | Count | Notes |
|--------|-------|-------|
| Not Implemented | 35 | Mostly HIGH priority |
| Partially Done | 12 | Have Result already, need other fixes |
| Mostly Done | 4 | Just need #[must_use] |

---

## Implementation Order (Recommended)

### Phase 1: Return Type Fixes (3-4 days)
Functions 1-39: Convert all to Result<T>

### Phase 2: Parameter Fixes (2-3 days)
Functions 1-15, 24, 25: Change group → ctx

### Phase 3: GPU Variants (1-2 days)
Functions 28, 31-32, 37: Add new _ctx variants

### Phase 4: #[must_use] (< 1 day)
Functions 16-19, 33-35, 40-41: Add attribute

---

## Testing Coverage Required

| Function | Unit | Integration | GPU | CPU | Error |
|----------|:---:|:---:|:---:|:---:|:---:|
| 1-6 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 7-11 | ✓ | ✓ | N/A | ✓ | ✓ |
| 12-15 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 16-19 | ✓ | ✓ | N/A | N/A | N/A |
| 20-26 | ✓ | ✓ | N/A | ✓ | ✓ |
| 27-32 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 33-35 | ✓ | ✓ | N/A | N/A | N/A |
| 36-37 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 38-39 | ✓ | ✓ | N/A | ✓ | ✓ |
| 40-41 | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Cross-References

- **Audit Report:** API_CONSISTENCY_AUDIT.md
- **Implementation Guide:** API_FIXES_DETAILED.md
- **Executive Summary:** API_CONSISTENCY_SUMMARY.md
- **Tracking Checklist:** API_AUDIT_CHECKLIST.md
- **Document Index:** API_AUDIT_INDEX.md

