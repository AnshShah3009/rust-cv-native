# API Consistency Audit - Implementation Checklist

**Created:** 2026-02-24
**Status:** Audit Complete - Implementation Pending
**Related Docs:** API_CONSISTENCY_AUDIT.md, API_FIXES_DETAILED.md, API_CONSISTENCY_SUMMARY.md

---

## Team Decision Points

- [ ] **DECISION 1:** Compute device parameter naming
  - Option A: `ctx: &ComputeDevice` (RECOMMENDED)
  - Option B: `runner: &RuntimeRunner`
  - **Decision:** ___________
  - **Decided By:** ___________
  - **Date:** ___________

- [ ] **DECISION 2:** Breaking change tolerance
  - Can we make breaking API changes in next release?
  - **Decision:** Yes / No
  - **Timeline:** ___________

- [ ] **DECISION 3:** Error type strategy
  - Keep separate error types per crate?
  - Centralize in cv_core::Error?
  - **Decision:** ___________

---

## PHASE 1: RETURN TYPE STANDARDIZATION

### cv-imgproc Functions (8 functions)

- [ ] `threshold()` - Convert to `Result<GrayImage>`
  - File: `imgproc/src/threshold.rs:20`
  - Tests to update: `test_threshold*`
  - Breaking: YES

- [ ] `threshold_otsu()` - Convert to `Result<(u8, GrayImage)>`
  - File: `imgproc/src/threshold.rs:76`
  - Tests to update: `test_threshold_otsu*`
  - Breaking: YES

- [ ] `sobel_magnitude_ctx()` - Convert to `Result<GrayImage>`
  - File: `imgproc/src/edges.rs:212`
  - Tests to update: `test_sobel*`
  - Breaking: YES

- [ ] `laplacian_ctx()` - Convert to `Result<GrayImage>`
  - File: `imgproc/src/edges.rs:247`
  - Tests to update: `test_laplacian*`
  - Breaking: YES

- [ ] `canny_ctx()` - Convert to `Result<GrayImage>`
  - File: `imgproc/src/edges.rs:475`
  - Tests to update: `test_canny*`
  - Breaking: YES

- [ ] `bilateral_filter_depth_ctx()` - Convert to `Result<GrayImage>`
  - File: `imgproc/src/bilateral.rs:138`
  - Tests to update: `test_bilateral*`
  - Breaking: YES

- [ ] `bilateral_filter_rgb_ctx()` - Convert to `Result<GrayImage>`
  - File: `imgproc/src/bilateral.rs:252`
  - Tests to update: `test_bilateral*`
  - Breaking: YES

- [ ] `joint_bilateral_filter_ctx()` - Convert to `Result<GrayImage>`
  - File: `imgproc/src/bilateral.rs:378`
  - Tests to update: `test_bilateral*`
  - Breaking: YES

### cv-features Functions (6 functions)

- [ ] `fast_detect()` - Convert to `Result<KeyPoints>`
  - File: `features/src/fast.rs:7`
  - Tests to update: `test_fast*`
  - Breaking: YES

- [ ] `harris_detect()` - Convert to `Result<KeyPoints>`
  - File: `features/src/harris.rs:6`
  - Tests to update: `test_harris*`
  - Breaking: YES

- [ ] `shi_tomasi_detect()` - Convert to `Result<KeyPoints>`
  - File: `features/src/harris.rs:56`
  - Tests to update: `test_shi_tomasi*`
  - Breaking: YES

- [ ] `gftt_detect()` - Convert to `Result<KeyPoints>`
  - File: `features/src/gftt.rs:4`
  - Tests to update: `test_gftt*`
  - Breaking: YES

- [ ] `corner_score()` - Convert to `Result<u8>`
  - File: `features/src/fast.rs:116`
  - Tests to update: `test_corner_score*`
  - Breaking: YES

- [ ] `non_max_suppression()` - Convert to `Result<KeyPoints>`
  - File: `features/src/fast.rs:151`
  - Tests to update: `test_nms*`
  - Breaking: YES

### cv-objdetect Functions (1 function)

- [ ] `hog.compute()` - Convert to `Result<Vec<f32>>`
  - File: `objdetect/src/hog.rs:36`
  - Class: `Hog`
  - Tests to update: `test_hog*`
  - Breaking: YES

### cv-dnn Functions (2 functions)

- [ ] `image_to_blob()` - Convert to `Result<Vec<f32>>`
  - File: `dnn/src/blob.rs:3`
  - Tests to update: `test_image_to_blob*`
  - Breaking: YES

- [ ] `blob_to_image()` - Convert to `Result<GrayImage>`
  - File: `dnn/src/blob.rs:15`
  - Tests to update: `test_blob_to_image*`
  - Breaking: YES

### cv-photo Functions (1 function)

- [ ] `bilateral_filter()` - Convert to `Result<GrayImage>`
  - File: `photo/src/bilateral.rs:3`
  - Tests to update: `test_bilateral*`
  - Breaking: YES

### cv-registration Functions (2 functions)

- [ ] `registration_icp_point_to_plane()` - Convert from `Option<ICPResult>` to `Result<ICPResult>`
  - File: `registration/src/registration/mod.rs:78`
  - Tests to update: `test_icp*`
  - Breaking: YES

- [ ] `registration_ransac_based_on_feature_matching()` - Convert to `Result<GlobalRegistrationResult>`
  - File: `registration/src/registration/global.rs`
  - Tests to update: `test_ransac*`
  - Breaking: YES

### PHASE 1 COMPLETION CHECKLIST

- [ ] All 20 functions updated in source code
- [ ] All return type signatures changed
- [ ] Error validation added (parameter checks)
- [ ] Ok(result) wrapping added at end of functions
- [ ] All unit tests updated to use `?` operator or `.unwrap()`
- [ ] Error case tests added for each function
- [ ] Code compiles without errors
- [ ] No new compiler warnings introduced
- [ ] Code review completed
- [ ] Merge to feature branch

**Phase 1 Effort Estimate:** 3-4 days
**Phase 1 Owner:** _____________
**Phase 1 Reviewer:** _____________
**Phase 1 Start Date:** _____________
**Phase 1 Target Completion:** _____________

---

## PHASE 2: PARAMETER STANDARDIZATION

### cv-imgproc Functions (8 functions) - Parameter Migration

**Decision Dependency:** Team must decide on `ctx` vs `runner`
**Recommended:** `ctx: &ComputeDevice`

- [ ] `sobel_ex_ctx()` - Change parameter: `group: &RuntimeRunner` → `ctx: &ComputeDevice`
  - File: `imgproc/src/edges.rs:66`
  - Update implementation: Change match statement
  - Tests to update: All calls in tests
  - Breaking: YES

- [ ] `scharr_ex_ctx()` - Change parameter: `group: &RuntimeRunner` → `ctx: &ComputeDevice`
  - File: `imgproc/src/edges.rs:142`
  - Tests to update: All calls in tests
  - Breaking: YES

- [ ] `sobel_magnitude_ctx()` - Change parameter: `group: &RuntimeRunner` → `ctx: &ComputeDevice`
  - File: `imgproc/src/edges.rs:212`
  - Tests to update: All calls in tests
  - Breaking: YES

- [ ] `laplacian_ctx()` - Change parameter: `group: &RuntimeRunner` → `ctx: &ComputeDevice`
  - File: `imgproc/src/edges.rs:247`
  - Tests to update: All calls in tests
  - Breaking: YES

- [ ] `canny_ctx()` - Change parameter: `group: &RuntimeRunner` → `ctx: &ComputeDevice`
  - File: `imgproc/src/edges.rs:475`
  - Tests to update: All calls in tests
  - Breaking: YES

- [ ] `convolve_ctx()` - Change parameter: `group: &RuntimeRunner` → `ctx: &ComputeDevice`
  - File: `imgproc/src/convolve.rs:196`
  - Tests to update: All calls in tests
  - Breaking: YES

- [ ] `gaussian_blur_ctx()` - Change parameter: `group: &RuntimeRunner` → `ctx: &ComputeDevice`
  - File: `imgproc/src/convolve.rs:253`
  - Tests to update: All calls in tests
  - Breaking: YES

- [ ] `gaussian_blur_with_border_into_ctx()` - Change parameter: `group: &RuntimeRunner` → `ctx: &ComputeDevice`
  - File: `imgproc/src/convolve.rs:513`
  - Tests to update: All calls in tests
  - Breaking: YES

### PHASE 2 COMPLETION CHECKLIST

- [ ] Decision made on parameter naming
- [ ] All 8 functions updated in source
- [ ] Parameter name changed in signature
- [ ] Parameter name changed in function body
- [ ] All match statements use new parameter type correctly
- [ ] All test calls updated with new parameter
- [ ] Code compiles without errors
- [ ] GPU/CPU fallback logic works correctly
- [ ] Code review completed
- [ ] Merge to feature branch

**Phase 2 Effort Estimate:** 2-3 days
**Phase 2 Owner:** _____________
**Phase 2 Reviewer:** _____________
**Phase 2 Start Date:** _____________
**Phase 2 Target Completion:** _____________

---

## PHASE 3: ADD MISSING GPU VARIANTS

### cv-dnn Functions (2 new functions)

- [ ] Add `image_to_blob_ctx<S>()` - New GPU variant
  - File: `dnn/src/blob.rs`
  - Location: After `image_to_blob()`
  - Parameters: `image: &Tensor<u8, S>`, `ctx: &ComputeDevice`
  - Returns: `Result<Vec<f32>>`
  - GPU path: Batch normalize on GPU
  - CPU path: Call existing `image_to_blob()`
  - Tests: `test_image_to_blob_ctx_gpu` and `test_image_to_blob_ctx_cpu`
  - Breaking: NO (addition only)

- [ ] Add `blob_to_image_ctx<S>()` - New GPU variant
  - File: `dnn/src/blob.rs`
  - Location: After `blob_to_image()`
  - Parameters: `blob: &Tensor<f32, S>`, `width: u32`, `height: u32`, `ctx: &ComputeDevice`
  - Returns: `Result<GrayImage>`
  - GPU path: Denormalization on GPU
  - CPU path: Call existing `blob_to_image()`
  - Tests: `test_blob_to_image_ctx_gpu` and `test_blob_to_image_ctx_cpu`
  - Breaking: NO (addition only)

### cv-objdetect Functions (1 new function)

- [ ] Add `compute_ctx<S>()` - New GPU variant
  - File: `objdetect/src/hog.rs`
  - Location: After existing `compute()`
  - Parameters: `image: &Tensor<u8, S>`, `ctx: &ComputeDevice`
  - Returns: `Result<Vec<f32>>`
  - GPU path: HOG computation on GPU (parallelized)
  - CPU path: Convert to image and call existing `compute()`
  - Tests: `test_hog_compute_ctx_gpu` and `test_hog_compute_ctx_cpu`
  - Breaking: NO (addition only)

### cv-photo Functions (1 new function)

- [ ] Add `bilateral_filter_ctx()` - New GPU variant
  - File: `photo/src/bilateral.rs`
  - Location: After existing `bilateral_filter()`
  - Parameters: `src: &GrayImage`, `diameter: i32`, `sigma_color: f32`, `sigma_space: f32`, `ctx: &ComputeDevice`
  - Returns: `Result<GrayImage>`
  - GPU path: Bilateral filter on GPU (high performance)
  - CPU path: Call existing `bilateral_filter()`
  - Tests: `test_bilateral_filter_ctx_gpu` and `test_bilateral_filter_ctx_cpu`
  - Breaking: NO (addition only)

### PHASE 3 COMPLETION CHECKLIST

- [ ] All 4 new functions added to source
- [ ] Function signatures match pattern of existing _ctx variants
- [ ] GPU paths implemented (or documented as TODO)
- [ ] CPU fallback paths implemented
- [ ] All functions have `#[must_use]` attribute
- [ ] All functions have proper doc comments
- [ ] Unit tests added (GPU, CPU, error cases)
- [ ] Code compiles without errors
- [ ] Code review completed
- [ ] Merge to feature branch

**Phase 3 Effort Estimate:** 1-2 days
**Phase 3 Owner:** _____________
**Phase 3 Reviewer:** _____________
**Phase 3 Start Date:** _____________
**Phase 3 Target Completion:** _____________

---

## PHASE 4: ADD #[must_use] ATTRIBUTES

### cv-features Functions (4 functions)

- [ ] Add `#[must_use]` to `detect_ctx()` in akaze.rs
- [ ] Add `#[must_use]` to `detect_and_compute_ctx()` in akaze.rs
- [ ] Add `#[must_use]` to `detect_and_compute()` in sift.rs
- [ ] Add `#[must_use]` to `detect_and_refine()` in sift.rs

### cv-video Functions (2 functions)

- [ ] Add `#[must_use]` to `apply_ctx()` in mog2.rs
- [ ] Add `#[must_use]` to `compute()` in optical_flow.rs

### cv-dnn Functions (3 functions)

- [ ] Add `#[must_use]` to `load()` in lib.rs
- [ ] Add `#[must_use]` to `forward()` in lib.rs
- [ ] Add `#[must_use]` to `preprocess()` in lib.rs

### cv-imgproc Functions (8 functions from Phase 1)

- [ ] Add `#[must_use]` to all newly Result-returning functions

### cv-registration Functions (2 functions)

- [ ] Add `#[must_use]` to `registration_icp_point_to_plane()`
- [ ] Add `#[must_use]` to `registration_ransac_based_on_feature_matching()`

### PHASE 4 COMPLETION CHECKLIST

- [ ] All 19 functions have `#[must_use]` attribute
- [ ] Code compiles without errors
- [ ] No new warnings introduced
- [ ] Code review completed
- [ ] Merge to feature branch

**Phase 4 Effort Estimate:** Few hours
**Phase 4 Owner:** _____________
**Phase 4 Start Date:** _____________
**Phase 4 Target Completion:** _____________

---

## PHASE 5: DOCUMENTATION & TESTING

### Documentation Updates

- [ ] Update function doc comments for all changed functions
- [ ] Document parameter meanings (especially `ctx`)
- [ ] Add examples for new _ctx variants
- [ ] Create migration guide for breaking changes
- [ ] Update CHANGELOG with all breaking changes
- [ ] Update API documentation website

### Integration Testing

- [ ] Test GPU vs CPU consistency for all _ctx functions
- [ ] Test error paths for all functions returning Result
- [ ] Test parameter validation (invalid sizes, etc.)
- [ ] Performance benchmarks (ensure no regressions)
- [ ] Test with actual GPU hardware (if available)

### User Facing

- [ ] Update examples/ directory with new API usage
- [ ] Update README.md with breaking change notice
- [ ] Prepare migration guide document
- [ ] Add deprecation notices if keeping old functions

### PHASE 5 COMPLETION CHECKLIST

- [ ] All doc comments updated
- [ ] Migration guide created and reviewed
- [ ] All integration tests pass
- [ ] Benchmarks show no performance regression
- [ ] Examples updated and tested
- [ ] CHANGELOG updated with all changes
- [ ] Release notes prepared

**Phase 5 Effort Estimate:** 1-2 days
**Phase 5 Owner:** _____________
**Phase 5 Start Date:** _____________
**Phase 5 Target Completion:** _____________

---

## CROSS-PHASE VALIDATION

### Code Quality Metrics

- [ ] No new compiler warnings
- [ ] All tests pass (unit + integration)
- [ ] Code coverage maintained or improved
- [ ] No clippy warnings
- [ ] Format with rustfmt: `cargo fmt`
- [ ] Lint with clippy: `cargo clippy --all-targets`

### Review Checklist

- [ ] Naming conventions followed (snake_case/PascalCase)
- [ ] Error handling is consistent
- [ ] Parameter naming is standardized
- [ ] Return types follow pattern
- [ ] Documentation is complete
- [ ] Tests cover error cases
- [ ] GPU/CPU fallback logic works

### Testing Matrix

| Function | Unit Tests | Integration Tests | GPU Test | CPU Test | Error Cases |
|----------|:---:|:---:|:---:|:---:|:---:|
| `threshold()` | [ ] | [ ] | N/A | [ ] | [ ] |
| `sobel_magnitude_ctx()` | [ ] | [ ] | [ ] | [ ] | [ ] |
| `canny_ctx()` | [ ] | [ ] | [ ] | [ ] | [ ] |
| `fast_detect()` | [ ] | [ ] | N/A | [ ] | [ ] |
| `image_to_blob_ctx()` | [ ] | [ ] | [ ] | [ ] | [ ] |
| `bilateral_filter_ctx()` | [ ] | [ ] | [ ] | [ ] | [ ] |
| `hog.compute_ctx()` | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## FINAL REVIEW & RELEASE

### Pre-Release Checklist

- [ ] All phases complete
- [ ] All tests passing
- [ ] All code reviewed and approved
- [ ] All documentation updated
- [ ] Migration guide is clear and complete
- [ ] CHANGELOG is accurate
- [ ] Release notes prepared
- [ ] Version number decided (major/minor bump for breaking changes)

### Release Steps

- [ ] Create release branch: `git checkout -b release/v0.X.0`
- [ ] Update version in Cargo.toml files
- [ ] Update CHANGELOG
- [ ] Create release commit: `git commit -m "Release v0.X.0"`
- [ ] Create git tag: `git tag v0.X.0`
- [ ] Publish to crates.io: `cargo publish`
- [ ] Create GitHub release with notes
- [ ] Update documentation site
- [ ] Announce breaking changes in appropriate channels

### Post-Release

- [ ] Monitor for user feedback
- [ ] Address any issues discovered
- [ ] Plan follow-up documentation improvements
- [ ] Update examples based on user questions

---

## Summary Statistics

| Phase | Files | Functions | Breaking Changes | Days | Status |
|-------|-------|-----------|---|---|---|
| 1: Return Types | 6 | 20 | YES | 3-4 | [ ] |
| 2: Parameters | 1 | 8 | YES | 2-3 | [ ] |
| 3: Variants | 3 | 4 | NO | 1-2 | [ ] |
| 4: #[must_use] | Multiple | 19 | NO | <1 | [ ] |
| 5: Documentation | Multiple | All | NO | 1-2 | [ ] |
| **TOTALS** | **13** | **51+** | **YES** | **9-13** | |

---

## Contact & Escalation

**Audit Lead:** _____________
**Implementation Lead:** _____________
**Review Lead:** _____________

**Escalation Contact:** _____________
**Escalation Email:** _____________

---

## Notes & Additional Context

(Space for team notes, decisions made, issues discovered)

_____________________________________________________________________________

_____________________________________________________________________________

_____________________________________________________________________________

---

**Last Updated:** 2026-02-24
**Document Version:** 1.0
**Status:** Ready for Team Review
