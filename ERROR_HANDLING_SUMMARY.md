# Error Handling Standardization Summary

## Quick Reference

### Current Status: 5/26 Crates Need Standardization

```
STANDARDIZED (3 crates) ✅
├── cv-features: Uses cv_core::Result + deprecated FeatureError alias
├── cv-video: Uses cv_core::Result + deprecated VideoError alias
└── cv-stereo: Uses cv_core::Result + deprecated StereoError alias

NEEDS STANDARDIZATION (5 crates) ❌
├── cv-imgproc: ImgprocError enum (HIGH priority - widely used)
├── cv-calib3d: CalibError enum (HIGH priority - foundational)
├── cv-sfm: SfmError enum (MEDIUM-HIGH priority)
├── cv-videoio: VideoError enum (MEDIUM priority)
└── cv-plot: PlotError enum (LOW-MEDIUM priority)

LOWER-LEVEL LAYERS (4 crates) - Acceptable with From impls ⚠️
├── cv-hal: Error enum (hardware abstraction layer)
├── cv-runtime: Error enum (runtime orchestration)
├── cv-scientific: Error enum (domain-specific + cv_core wrap)
└── cv-3d: Error enum (async/3D-specific + minimal)

ALREADY STANDARD (14+ crates) ✅
├── cv-registration, cv-objdetect, cv-io, cv-dnn, cv-photo, etc.
└── All use cv_core::Error or cv_core::Result directly
```

---

## Error Variant Coverage in cv_core::Error

The canonical error type has 21 variants:

```
Core Layer (7):
├── RuntimeError(String)      ← Runtime failures
├── ConcurrencyError(String)  ← Thread/concurrency issues
├── InvalidInput(String)      ← Invalid parameters
├── MemoryError(String)       ← Memory allocation/access
├── DimensionMismatch(String) ← Tensor dimension errors
├── IoError(String)           ← File I/O operations
└── Other(String)             ← Generic fallback

Domain Layer (14):
├── FeatureError(String)         ← Features (AKAZE, SIFT, ORB, etc.)
├── VideoError(String)           ← Video processing (MOG2, Optical Flow)
├── ImgprocError(String)         ← Image processing (filters, transforms)
├── RegistrationError(String)    ← Point cloud registration (ICP, GNC)
├── StereoError(String)          ← Stereo matching (SGM, Block matching)
├── ObjectDetectionError(String) ← Detection (HOG, Haar, Yolo)
├── DnnError(String)             ← Deep learning (DNN module)
├── PhotoError(String)           ← Photo enhancement (bilateral, etc.)
├── CalibrationError(String)     ← Camera calibration (chess, planar)
├── SfMError(String)             ← Structure from Motion (triangulation)
├── DeviceError(String)          ← Hardware/device issues
├── GpuError(String)             ← GPU computation
├── ParseError(String)           ← Format/file parsing
└── AlgorithmError(String)       ← Convergence/algorithm failure
```

---

## Migration Mapping Reference

### cv-imgproc Error Mapping
```rust
ImgprocError::ImageError(s)        → Error::ImgprocError(s)
ImgprocError::AlgorithmError(s)    → Error::AlgorithmError(s)
ImgprocError::UnsupportedFormat(s) → Error::ParseError(s)
ImgprocError::DimensionMismatch(s) → Error::DimensionMismatch(s)
```

### cv-calib3d Error Mapping
```rust
CalibError::InvalidParameters(s) → Error::InvalidInput(s)
CalibError::SvdFailed(s)         → Error::AlgorithmError(s)
CalibError::NumericalError(s)    → Error::CalibrationError(s)
CalibError::Io(e)               → Error::IoError(format!("{}", e))
CalibError::Image(e)            → Error::ImgprocError(format!("{}", e))
```

### cv-sfm Error Mapping
```rust
SfmError::TriangulationFailed(s) → Error::SfMError(s)
```

### cv-videoio Error Mapping
```rust
VideoError::Io(e)              → Error::IoError(format!("{}", e))
VideoError::Backend(s)         → Error::RuntimeError(s) or Error::DeviceError(s)
VideoError::InvalidParameters(s) → Error::InvalidInput(s)
VideoError::CaptureFailed(s)   → Error::VideoError(s)
```

### cv-plot Error Mapping
```rust
PlotError::InvalidData(s) → Error::InvalidInput(s)
PlotError::Io(e)         → Error::IoError(format!("{}", e))
PlotError::Export(s)     → Error::Other(s)
```

---

## Files Affected by Standardization

### Phase 1 Files to Modify (5 crates, ~15 files)

**cv-imgproc:**
- `imgproc/src/lib.rs` - Define Result type alias, add deprecated marker
- `imgproc/src/threshold.rs` - Update error returns
- `imgproc/src/edges.rs` - Update error returns
- `imgproc/src/morph.rs` - Update error returns
- `imgproc/src/bilateral.rs` - Update error returns
- All modules using ImgprocError

**cv-calib3d:**
- `calib3d/src/lib.rs` - Replace CalibError
- `calib3d/src/pnp.rs` - Update error handling
- `calib3d/src/calibration.rs` - Update error handling
- `calib3d/src/pattern.rs` - Update error handling
- ~10 module files

**cv-sfm:**
- `sfm/src/lib.rs` - Update lib.rs
- `sfm/src/triangulation.rs` - Remove SfmError enum, use Error variant
- `sfm/src/bundle_adjustment.rs` - Update error handling

**cv-videoio:**
- `videoio/src/lib.rs` - Replace VideoError enum with alias
- `videoio/src/backends/*.rs` - Update error returns

**cv-plot:**
- `plot/src/lib.rs` - Replace PlotError enum with alias
- `plot/src/export.rs` - Update error handling
- `plot/src/chart.rs` - Update error handling

---

## Code Examples

### Before Standardization (cv-imgproc example)
```rust
// In imgproc/src/lib.rs
pub type Result<T> = std::result::Result<T, ImgprocError>;

#[derive(Debug, thiserror::Error)]
pub enum ImgprocError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

pub fn validate_image_size(width: u32, height: u32) -> Result<()> {
    if width == 0 {
        return Err(ImgprocError::DimensionMismatch("width must be non-zero".into()));
    }
    Ok(())
}

// Usage in calling code
fn process(width: u32) -> Result<()> {
    validate_image_size(width, 480)?;
    // ...
}
```

### After Standardization (cv-imgproc example)
```rust
// In imgproc/src/lib.rs
pub use cv_core::{Error, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(since = "0.1.0", note = "Use cv_core::Error instead")]
pub type ImgprocError = cv_core::Error;

pub fn validate_image_size(width: u32, height: u32) -> Result<()> {
    if width == 0 {
        return Err(Error::DimensionMismatch("width must be non-zero".into()));
    }
    Ok(())
}

// Usage in calling code - SAME
fn process(width: u32) -> Result<()> {
    validate_image_size(width, 480)?;
    // ...
}
```

---

## Implementation Order

**Recommended sequence for compilation safety:**

1. **Step 1:** Update cv-sfm (smallest, fewest dependencies)
2. **Step 2:** Update cv-videoio (independent module)
3. **Step 3:** Update cv-plot (visualization only)
4. **Step 4:** Update cv-calib3d (many dependents but high priority)
5. **Step 5:** Update cv-imgproc (most dependents - do last)
6. **Step 6:** Add From implementations for HAL/runtime/scientific/3d
7. **Step 7:** Full test suite

**After each step:** Run `cargo check --all-features`

---

## Transition Timeline

| Phase | Duration | Activity |
|-------|----------|----------|
| Planning | ✅ Complete | Audit, create this document |
| Implementation | 2-3 days | Modify 5 crates, test locally |
| Verification | 1 day | Full test suite, CI/CD |
| Integration | 1 day | Merge to develop, resolve conflicts |
| Release | 1 day | Changelog, version bump, PR review |

---

## Testing Strategy

### Unit Tests
- Run existing tests with new error types
- `cargo test --all-features`

### Error Path Tests
- Test each error variant is reachable
- Example: `test_dimension_mismatch_error()`

### Integration Tests
- Test error propagation across crate boundaries
- Example: imgproc → calib3d → features chain

### Backward Compatibility Tests
- Use deprecated error types
- Verify they still work (trigger deprecation warning)

### Command Sequence
```bash
# Check for compilation errors
cargo check --all-features

# Run all tests
cargo test --all-features

# Check for clippy warnings
cargo clippy --all -- -D warnings

# Check documentation
cargo doc --no-deps --all-features

# Final verification
cargo build --release
```

---

## Deprecation Warning Reference

When standardization is complete, users will see:

```
warning: use of deprecated type alias `ImgprocError`
  --> user_code.rs:10:14
   |
10 | impl From<ImgprocError> for CustomError { ... }
   |         ^^^^^^^^^^^^^^
   |
   = note: Use cv_core::Error instead. This type alias exists only for backward compatibility.
   = note: `#[warn(deprecated)]` on by default
```

This is intentional - it guides users to migrate their code.

---

## Success Criteria

✅ All 5 crates updated to use cv_core::Result<T>
✅ Deprecated type aliases in place for backward compatibility
✅ From implementations added for bridging layers
✅ Zero compilation errors: `cargo check --all-features` passes
✅ All tests pass: `cargo test --all-features` succeeds
✅ No clippy warnings introduced
✅ Documentation builds without warnings
✅ Existing user code still compiles (with deprecation warnings)

---

## Resource Files

- **Full Audit:** `/home/prathana/RUST/rust-cv-native/ERROR_HANDLING_AUDIT_REPORT.md`
- **This Document:** `/home/prathana/RUST/rust-cv-native/ERROR_HANDLING_SUMMARY.md`

---

## Next Steps

1. Review this document with team
2. Approve standardization plan
3. Create feature branch: `feature/standardize-error-handling`
4. Begin Phase 1 migrations (start with cv-sfm)
5. After each crate: run `cargo check --all-features`
6. Create PR with all changes
7. Final verification in CI/CD
8. Release as v0.2.0

