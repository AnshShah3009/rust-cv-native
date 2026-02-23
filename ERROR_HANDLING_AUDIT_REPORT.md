# Error Handling Audit Report
## rust-cv-native Error Standardization Project

**Date:** February 24, 2026
**Status:** Audit Complete - Standardization Partially Complete
**Scope:** All 26 workspace members

---

## Executive Summary

This audit identifies all custom error types across the rust-cv-native workspace and evaluates their status against the standardized cv_core::Error approach.

**Key Findings:**
- **6 crates** use custom error enums (beyond cv_core)
- **3 crates** have deprecated aliases pointing to cv_core::Error (already standardized)
- **2 crates** use cv_core::Error exclusively (best practice)
- **15 crates** do not explicitly define error types
- **cv_core::Error** is the canonical error type with 21 variants covering all domains

---

## Detailed Crate-by-Crate Analysis

### Crates with Custom Error Types (Non-Standardized)

#### 1. **cv-videoio** (`videoio/src/lib.rs`)
**Status:** Custom error enum - NEEDS STANDARDIZATION

```rust
pub type Result<T> = std::result::Result<T, VideoError>;

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Capture failed: {0}")]
    CaptureFailed(String),
}
```

**Mapping to cv_core::Error:**
- `Io()` → `cv_core::Error::IoError()`
- `Backend()` → `cv_core::Error::RuntimeError()` or `cv_core::Error::DeviceError()`
- `InvalidParameters()` → `cv_core::Error::InvalidInput()`
- `CaptureFailed()` → `cv_core::Error::VideoError()`

**Action Required:** Migrate public API to return `cv_core::Result<T>` with conversion From implementation.

---

#### 2. **cv-sfm** (`sfm/src/triangulation.rs`)
**Status:** Custom error enum - NEEDS STANDARDIZATION

```rust
#[derive(Debug, thiserror::Error)]
pub enum SfmError {
    #[error("Triangulation failed: {0}")]
    TriangulationFailed(String),
}

pub type Result<T> = std::result::Result<T, SfmError>;
```

**Mapping to cv_core::Error:**
- `TriangulationFailed()` → `cv_core::Error::SfMError()`

**Action Required:** Replace with cv_core::Result and cv_core::Error::SfMError().

---

#### 3. **cv-imgproc** (`imgproc/src/lib.rs`)
**Status:** Custom error enum - NEEDS STANDARDIZATION

```rust
pub type Result<T> = std::result::Result<T, ImgprocError>;

#[derive(Debug, thiserror::Error)]
pub enum ImgprocError {
    #[error("Image error: {0}")]
    ImageError(String),

    #[error("Algorithm error: {0}")]
    AlgorithmError(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}
```

**Mapping to cv_core::Error:**
- `ImageError()` → `cv_core::Error::ImgprocError()`
- `AlgorithmError()` → `cv_core::Error::AlgorithmError()`
- `UnsupportedFormat()` → `cv_core::Error::ParseError()`
- `DimensionMismatch()` → `cv_core::Error::DimensionMismatch()`

**Action Required:** Replace with cv_core equivalents.

---

#### 4. **cv-calib3d** (`calib3d/src/lib.rs`)
**Status:** Custom error enum - NEEDS STANDARDIZATION

```rust
pub type Result<T> = std::result::Result<T, CalibError>;

#[derive(Debug, thiserror::Error)]
pub enum CalibError {
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("SVD failed: {0}")]
    SvdFailed(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
}
```

**Mapping to cv_core::Error:**
- `InvalidParameters()` → `cv_core::Error::InvalidInput()`
- `SvdFailed()` → `cv_core::Error::AlgorithmError()`
- `NumericalError()` → `cv_core::Error::CalibrationError()`
- `Io()` → `cv_core::Error::IoError()`
- `Image()` → `cv_core::Error::ImgprocError()`

**Action Required:** Replace with cv_core variants.

---

#### 5. **cv-plot** (`plot/src/lib.rs`)
**Status:** Custom error enum - NEEDS STANDARDIZATION

```rust
#[derive(Debug, thiserror::Error)]
pub enum PlotError {
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Export error: {0}")]
    Export(String),
}
```

**Mapping to cv_core::Error:**
- `InvalidData()` → `cv_core::Error::InvalidInput()`
- `Io()` → `cv_core::Error::IoError()`
- `Export()` → `cv_core::Error::Other()`

**Action Required:** Replace with cv_core variants.

---

### Crates with Deprecated Error Aliases (Already Standardized)

These crates maintain backward compatibility aliases while standardized to cv_core:

#### 1. **cv-features** (`features/src/lib.rs`)
**Status:** ✅ STANDARDIZED (Deprecated aliases present)

```rust
pub use cv_core::{KeyPoint, KeyPoints, Error, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(since = "0.1.0", note = "Use cv_core::Error instead")]
pub type FeatureError = cv_core::Error;

#[deprecated(since = "0.1.0", note = "Use cv_core::Result instead")]
pub type FeatureResult<T> = cv_core::Result<T>;
```

**Status:** Public API uses cv_core::Result<T>. Backward compatibility maintained.

---

#### 2. **cv-video** (`video/src/lib.rs`)
**Status:** ✅ STANDARDIZED (Deprecated aliases present)

```rust
pub use cv_core::{Error, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(since = "0.1.0", note = "Use cv_core::Error instead")]
pub type VideoError = cv_core::Error;

#[deprecated(since = "0.1.0", note = "Use cv_core::Result instead")]
pub type VideoResult<T> = cv_core::Result<T>;
```

**Status:** Public API uses cv_core::Result<T>. Backward compatibility maintained.

---

#### 3. **cv-stereo** (`stereo/src/lib.rs`)
**Status:** ✅ STANDARDIZED (Deprecated aliases present)

```rust
pub use cv_core::{Error, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(since = "0.1.0", note = "Use cv_core::Error instead")]
pub type StereoError = cv_core::Error;

#[deprecated(since = "0.1.0", note = "Use cv_core::Result instead")]
pub type StereoResult<T> = cv_core::Result<T>;
```

**Status:** Public API uses cv_core::Result<T>. Backward compatibility maintained.

---

### Crates with Custom Non-Standard Errors

#### 1. **cv-hal** (`hal/src/lib.rs`)
**Status:** Custom error enum with cv_core bridging

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Queue error: {0}")]
    QueueError(String),

    #[error("Kernel error: {0}")]
    KernelError(String),

    #[error("Not supported: {0}")]
    NotSupported(String),

    #[error("Initialization error: {0}")]
    InitError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Core error: {0}")]
    CoreError(#[from] cv_core::Error),
}
```

**Analysis:** HAL has its own error type as a lower-level abstraction layer. This is acceptable as it has a `CoreError` variant for wrapping cv_core errors. However, public APIs should still prefer returning cv_core::Error for consistency.

**Recommendation:**
- Keep internal hal::Error for lower-level operations
- Implement From<hal::Error> for cv_core::Error
- Public APIs should map to cv_core::Error where possible

---

#### 2. **cv-runtime** (`runtime/src/lib.rs`)
**Status:** Custom error enum with cv_core bridging

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),

    #[error("Not supported: {0}")]
    NotSupported(String),

    #[error("HAL error: {0}")]
    HalError(#[from] cv_hal::Error),

    #[error("Core error: {0}")]
    CoreError(#[from] cv_core::Error),

    #[error("Initialization error: {0}")]
    InitError(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

**Analysis:** Runtime has its own error type as it sits between cv_core and specific modules. This is acceptable as it wraps both cv_hal and cv_core errors. However, public scheduler/executor APIs should preferentially return cv_core::Error.

**Recommendation:**
- Implement From<runtime::Error> for cv_core::Error
- Public APIs should map runtime errors to cv_core equivalents

---

#### 3. **cv-scientific** (`scientific/src/lib.rs`)
**Status:** Custom error enum with cv_core bridging

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Decomposition failed: {0}")]
    DecompositionError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Mathematical error: {0}")]
    MathError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Core error: {0}")]
    CoreError(#[from] cv_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
```

**Analysis:** Scientific has its own error type for domain-specific errors. Wraps cv_core::Error. This is acceptable.

**Recommendation:**
- Implement From<scientific::Error> for cv_core::Error
- Consider mapping domain errors to cv_core variants

---

#### 4. **cv-3d** (`3d/src/lib.rs`)
**Status:** Custom error enum with minimal coverage

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Async task join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

**Analysis:** Minimal custom error - mainly for async operations. Should bridge to cv_core.

**Recommendation:**
- Implement From<3d::Error> for cv_core::Error
- Map errors to cv_core::RuntimeError or DeviceError

---

### Crates with Module-Level Custom Errors

#### **cv-calib3d** (`calib3d/src/pnp.rs`)
**Status:** Module-level Result alias

```rust
pub type Result<T> = std::result::Result<T, CalibError>;
```

**Analysis:** Reuses CalibError from parent module. Consistent with parent.

---

## cv_core::Error Canonical Definition

**Location:** `/home/prathana/RUST/rust-cv-native/core/src/lib.rs`

**Current Variants (21 total):**

```rust
pub enum Error {
    RuntimeError(String),           // Generic runtime errors
    ConcurrencyError(String),       // Thread/concurrency errors
    InvalidInput(String),           // Invalid input parameters
    MemoryError(String),            // Memory allocation/access
    DimensionMismatch(String),      // Tensor/array dimension errors
    IoError(String),                // File I/O operations
    FeatureError(String),           // Feature detection/extraction
    VideoError(String),             // Video processing
    ImgprocError(String),           // Image processing
    RegistrationError(String),      // 3D registration
    StereoError(String),            // Stereo vision
    ObjectDetectionError(String),   // Object detection
    DnnError(String),               // Deep neural networks
    PhotoError(String),             // Photography/image enhancement
    CalibrationError(String),       // Camera calibration
    SfMError(String),               // Structure from Motion
    DeviceError(String),            // Hardware/device errors
    GpuError(String),               // GPU computation
    ParseError(String),             // Parse/format errors
    AlgorithmError(String),         // Algorithm failure/convergence
    Other(String),                  // Generic fallback
}
```

**Assessment:** ✅ cv_core::Error is well-designed and covers all identified error domains.

---

## Standardization Status Summary

| Crate | Status | Type | Action Required |
|-------|--------|------|-----------------|
| cv-core | ✅ | Canonical | None |
| cv-features | ✅ | Standardized | None (deprecated aliases) |
| cv-video | ✅ | Standardized | None (deprecated aliases) |
| cv-stereo | ✅ | Standardized | None (deprecated aliases) |
| **cv-videoio** | ❌ | Custom | Migrate to cv_core::Error |
| **cv-sfm** | ❌ | Custom | Migrate to cv_core::Error |
| **cv-imgproc** | ❌ | Custom | Migrate to cv_core::Error |
| **cv-calib3d** | ❌ | Custom | Migrate to cv_core::Error |
| **cv-plot** | ❌ | Custom | Migrate to cv_core::Error |
| cv-hal | ⚠️ | Custom + Bridge | Acceptable for HAL layer |
| cv-runtime | ⚠️ | Custom + Bridge | Acceptable for runtime layer |
| cv-scientific | ⚠️ | Custom + Bridge | Provide From impl for cv_core::Error |
| cv-3d | ⚠️ | Custom + Bridge | Provide From impl for cv_core::Error |
| cv-registration | ✅ | Uses cv_core | None |
| cv-objdetect | ✅ | Uses cv_core | None |
| cv-io | ✅ | Uses cv_core | None |
| cv-dnn | ✅ | Uses cv_core | None |
| cv-photo | ✅ | Uses cv_core | None |
| cv-slam | ✅ | Uses cv_core | None |
| cv-optimize | ✅ | Uses cv_core | None |
| cv-point-cloud | ✅ | Uses cv_core | None |
| cv-python | ✅ | N/A | None |
| cv-viewer | ✅ | N/A | None |
| cv-rendering | ✅ | N/A | None |
| cv-examples | ✅ | N/A | None |

---

## Detailed Migration Plan

### Phase 1: Immediate Standardization Required (5 crates)

#### 1.1 cv-imgproc (`imgproc/src/lib.rs`)
**Priority:** HIGH (used by many crates)

**Changes:**
1. Keep ImgprocError as deprecated type alias
2. Update all public function signatures to return `cv_core::Result<T>`
3. Add From<ImgprocError> impl for cv_core::Error
4. Update error creation to use cv_core::Error variants

**Example Migration:**
```rust
// Before
pub fn validate_image_size(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(ImgprocError::DimensionMismatch(...));
    }
    Ok(())
}

// After
pub fn validate_image_size(width: u32, height: u32) -> cv_core::Result<()> {
    if width == 0 || height == 0 {
        return Err(cv_core::Error::DimensionMismatch(...));
    }
    Ok(())
}
```

---

#### 1.2 cv-calib3d (`calib3d/src/lib.rs`)
**Priority:** HIGH (camera calibration is foundational)

**Changes:**
1. Deprecate CalibError type alias
2. Update all public APIs to return cv_core::Result<T>
3. Map CalibError variants to cv_core::Error
4. Update calibration module functions

---

#### 1.3 cv-sfm (`sfm/src/triangulation.rs`)
**Priority:** MEDIUM-HIGH (Structure from Motion)

**Changes:**
1. Remove SfmError enum
2. Update triangulation functions to return cv_core::Result<T>
3. Use cv_core::Error::SfMError for errors

**Affected Functions:**
- `triangulate_point_dlt()`
- `triangulate_points()`
- `triangulate_points_ctx()`

---

#### 1.4 cv-videoio (`videoio/src/lib.rs`)
**Priority:** MEDIUM (Video I/O)

**Changes:**
1. Deprecate VideoError
2. Add From<VideoError> for cv_core::Error
3. Update trait methods to return cv_core::Result<T>
4. Update open_video(), open_camera() functions

**Affected Traits/Functions:**
- VideoCapture trait (grab, retrieve, read)
- VideoWriter trait (write)
- open_video()
- open_camera()

---

#### 1.5 cv-plot (`plot/src/lib.rs`)
**Priority:** LOW-MEDIUM (Visualization)

**Changes:**
1. Deprecate PlotError
2. Add From<PlotError> for cv_core::Error
3. Update public API to use cv_core::Result<T>
4. Update save_* functions

---

### Phase 2: Add From Implementations (4 crates)

#### 2.1 cv-hal
Add implementation:
```rust
impl From<hal::Error> for cv_core::Error {
    fn from(err: hal::Error) -> Self {
        match err {
            hal::Error::BackendNotAvailable(s) => cv_core::Error::DeviceError(s),
            hal::Error::DeviceError(s) => cv_core::Error::DeviceError(s),
            hal::Error::MemoryError(s) => cv_core::Error::MemoryError(s),
            hal::Error::QueueError(s) => cv_core::Error::RuntimeError(s),
            hal::Error::KernelError(s) => cv_core::Error::GpuError(s),
            hal::Error::NotSupported(s) => cv_core::Error::Other(s),
            hal::Error::InitError(s) => cv_core::Error::RuntimeError(s),
            hal::Error::InvalidInput(s) => cv_core::Error::InvalidInput(s),
            hal::Error::RuntimeError(s) => cv_core::Error::RuntimeError(s),
            hal::Error::CoreError(e) => e,
        }
    }
}
```

#### 2.2 cv-runtime
Add implementation similar to hal::Error

#### 2.3 cv-scientific
Add implementation for scientific domain errors

#### 2.4 cv-3d
Add implementation for 3D-specific errors

---

### Phase 3: Compilation Verification

After each phase, run:
```bash
cargo check --all-features
cargo test --all-features
cargo clippy --all
```

---

## Backward Compatibility Strategy

1. **Deprecation Path:**
   - Add `#[deprecated]` attributes to custom error types
   - Provide type aliases pointing to cv_core equivalents
   - Maintain From implementations for existing code

2. **Semver:**
   - Mark breaking API changes with new major version
   - Maintain deprecated types for one full minor version cycle

3. **Migration Guide:**
   - Document in CHANGELOG
   - Provide code examples
   - Update user-facing docs

---

## Benefits of Standardization

1. **Reduced Complexity:** Single error type across entire codebase
2. **Better Error Handling:** Unified error matching patterns
3. **Improved Interoperability:** Crates can easily combine operations
4. **Easier Debugging:** Consistent error messages
5. **API Clarity:** Clear what errors a function can return
6. **Testing:** Simplified error testing across crates
7. **Documentation:** Single error type to document

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Breaking user code | High | Medium | Deprecated aliases, long transition |
| Compilation errors | Medium | Low | Comprehensive testing, phase approach |
| Performance impact | Low | Low | Error types are thin wrappers |
| Missed error variants | Medium | Medium | Thorough audit (already done) |

---

## Implementation Checklist

- [ ] Phase 1: Migrate 5 core crates (cv-imgproc, cv-calib3d, cv-sfm, cv-videoio, cv-plot)
  - [ ] Update cv-imgproc error handling
  - [ ] Update cv-calib3d error handling
  - [ ] Update cv-sfm error handling
  - [ ] Update cv-videoio error handling
  - [ ] Update cv-plot error handling
  - [ ] Compilation check after each migration

- [ ] Phase 2: Add From implementations (cv-hal, cv-runtime, cv-scientific, cv-3d)
  - [ ] Implement From<hal::Error> for cv_core::Error
  - [ ] Implement From<runtime::Error> for cv_core::Error
  - [ ] Implement From<scientific::Error> for cv_core::Error
  - [ ] Implement From<3d::Error> for cv_core::Error
  - [ ] Compilation check

- [ ] Phase 3: Testing & Verification
  - [ ] `cargo check --all-features`
  - [ ] `cargo test --all-features`
  - [ ] `cargo clippy --all`
  - [ ] Manual testing of error paths
  - [ ] Documentation update

- [ ] Phase 4: Release & Communication
  - [ ] Update CHANGELOG
  - [ ] Create migration guide
  - [ ] Update examples
  - [ ] Release as new minor version (0.2.0)

---

## Affected Public APIs

### cv-imgproc
- validate_image_size()
- All function signatures returning ImgprocError

### cv-calib3d
- solve_pnp_dlt()
- solve_pnp_ransac()
- solve_pnp_refine()
- calibrate_camera_*()
- stereo_calibrate_*()
- find_chessboard_corners()
- essential_from_extrinsics()
- find_essential_mat*()
- find_fundamental_mat*()
- triangulate_points()
- And ~20 more functions

### cv-sfm
- triangulate_point_dlt()
- triangulate_points()
- triangulate_points_ctx()

### cv-videoio
- open_video()
- open_camera()
- VideoCapture trait (grab, retrieve, read)
- VideoWriter trait (write)

### cv-plot
- save_svg()
- save_html()
- save_png()
- to_svg()
- to_html()
- Plot methods

---

## Conclusion

The rust-cv-native codebase is well-structured with cv_core::Error already providing comprehensive coverage of all error domains. The remaining work involves consolidating 5 crates that use custom error types, adding From implementations for lower-level layers, and maintaining backward compatibility through deprecated type aliases.

**Estimated effort:** 2-3 days for Phase 1 migrations + 1 day for Phase 2 + 1 day for testing/verification.

**Risk level:** LOW - cv_core::Error variants are already defined; this is primarily refactoring existing error handling.

