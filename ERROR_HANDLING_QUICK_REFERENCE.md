# Error Handling Quick Reference Card

## Quick Status Check

| Crate | Status | Fix Required | Effort |
|-------|--------|--------------|--------|
| cv-sfm | ❌ Custom | Yes | 1-2 hrs |
| cv-videoio | ❌ Custom | Yes | 2-3 hrs |
| cv-plot | ❌ Custom | Yes | 1-2 hrs |
| cv-calib3d | ❌ Custom | Yes | 4-6 hrs |
| cv-imgproc | ❌ Custom | Yes | 6-8 hrs |
| cv-hal | ⚠️ Bridge | Add From impl | 0.5 hrs |
| cv-runtime | ⚠️ Bridge | Add From impl | 0.5 hrs |
| cv-scientific | ⚠️ Bridge | Add From impl | 0.5 hrs |
| cv-3d | ⚠️ Bridge | Add From impl | 0.5 hrs |
| cv-features | ✅ Done | None | — |
| cv-video | ✅ Done | None | — |
| cv-stereo | ✅ Done | None | — |

**Total effort:** 2-4 days

---

## Error Variant Cheat Sheet

### Use When...

```rust
Error::InvalidInput(_)       // Parameter validation fails (fn(0) where 0 forbidden)
Error::DimensionMismatch(_)  // Array/tensor shape mismatch (3x4 + 2x2)
Error::MemoryError(_)        // Allocation or access failure
Error::IoError(_)            // File read/write fails
Error::RuntimeError(_)       // General unexpected failure
Error::AlgorithmError(_)     // Convergence failure, optimization failed
Error::SfMError(_)           // Triangulation fails, essential matrix bad
Error::CalibrationError(_)   // Camera calibration invalid
Error::ImgprocError(_)       // Image processing algorithm failed
Error::VideoError(_)         // Video capture/processing failed
Error::FeatureError(_)       // Feature detection/matching failed
Error::RegistrationError(_)  // Point cloud registration failed
Error::StereoError(_)        // Stereo matching failed
Error::ObjectDetectionError(_) // Detection/classification failed
Error::DnnError(_)           // Neural network inference failed
Error::PhotoError(_)         // Photo enhancement failed
Error::DeviceError(_)        // Hardware not available
Error::GpuError(_)           // GPU computation failed
Error::ParseError(_)         // File format invalid
Error::ConcurrencyError(_)   // Thread/sync failure
Error::Other(_)              // Anything else
```

---

## Migration Template (Copy/Paste)

### Step 1: Update lib.rs

```rust
// REMOVE THIS:
// pub type Result<T> = std::result::Result<T, CustomError>;
// #[derive(Debug, thiserror::Error)]
// pub enum CustomError { ... }

// ADD THIS:
pub use cv_core::{Error, Result};

#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Error instead. This type alias exists only for backward compatibility."
)]
pub type CustomError = cv_core::Error;
```

### Step 2: Replace Error Constructors

```rust
// BEFORE:
return Err(CustomError::SomeVariant("message".into()));

// AFTER (use appropriate variant):
return Err(Error::SpecificVariant("message".into()));
```

### Step 3: Update Function Signatures

```rust
// BEFORE:
pub fn do_something() -> crate::Result<Output> { ... }

// AFTER:
pub fn do_something() -> Result<Output> { ... }
```

### Step 4: Verify

```bash
cargo check -p {crate_name}
cargo test -p {crate_name}
```

---

## Error Mapping Reference

### ImgprocError → cv_core::Error

```rust
ImgprocError::ImageError(s)        → Error::ImgprocError(s)
ImgprocError::AlgorithmError(s)    → Error::AlgorithmError(s)
ImgprocError::UnsupportedFormat(s) → Error::ParseError(s)
ImgprocError::DimensionMismatch(s) → Error::DimensionMismatch(s)
```

### CalibError → cv_core::Error

```rust
CalibError::InvalidParameters(s) → Error::InvalidInput(s)
CalibError::SvdFailed(s)         → Error::AlgorithmError(s)
CalibError::NumericalError(s)    → Error::CalibrationError(s)
CalibError::Io(e)               → Error::IoError(e.to_string())
CalibError::Image(e)            → Error::ImgprocError(e.to_string())
```

### SfmError → cv_core::Error

```rust
SfmError::TriangulationFailed(s) → Error::SfMError(s)
```

### VideoError → cv_core::Error

```rust
VideoError::Io(e)              → Error::IoError(e.to_string())
VideoError::Backend(s)         → Error::DeviceError(s)
VideoError::InvalidParameters(s) → Error::InvalidInput(s)
VideoError::CaptureFailed(s)   → Error::VideoError(s)
```

### PlotError → cv_core::Error

```rust
PlotError::InvalidData(s) → Error::InvalidInput(s)
PlotError::Io(e)         → Error::IoError(e.to_string())
PlotError::Export(s)     → Error::Other(s)
```

---

## Find & Replace Commands

For each crate, use these to find what needs changing:

```bash
# Find all error enum definitions
grep -n "pub enum.*Error" {crate}/src/lib.rs

# Find all error constructors
grep -r "Error::" {crate}/src/ | head -20

# Find all custom error uses
grep -r "CustomError::" {crate}/src/

# Verify migration complete
grep -r "CustomError::" {crate}/src/
# Should be zero results
```

---

## Compilation Verification Checklist

After each crate migration:

```bash
# Single crate check (quick)
cargo check -p cv-{crate}

# All crates check (comprehensive)
cargo check --all-features

# Run crate tests
cargo test -p cv-{crate} --all-features

# Check for warnings
cargo clippy -p cv-{crate} --all-features
```

---

## Common Mistakes & Fixes

### Mistake 1: Incomplete Migration
```rust
// WRONG - Still using old error
Err(ImgprocError::DimensionMismatch("..."))

// RIGHT - Use cv_core::Error
Err(Error::DimensionMismatch("..."))
```

### Mistake 2: Wrong Error Variant
```rust
// WRONG - Generic RuntimeError for domain-specific case
SfmError::TriangulationFailed(s) → Error::RuntimeError(s)

// RIGHT - Use domain-specific variant
SfmError::TriangulationFailed(s) → Error::SfMError(s)
```

### Mistake 3: Missing Deprecation Attribute
```rust
// WRONG - No indication this is deprecated
pub type ImgprocError = cv_core::Error;

// RIGHT - Guides users to migrate
#[deprecated(since = "0.1.0", note = "Use cv_core::Error instead")]
pub type ImgprocError = cv_core::Error;
```

### Mistake 4: Leftover Custom Error Usage
```rust
// WRONG - Custom error still defined after alias
pub enum ImgprocError {
    Dimension(String),
}

// RIGHT - Remove custom enum entirely, use alias
#[deprecated(...)]
pub type ImgprocError = cv_core::Error;
```

---

## Testing Pattern

```rust
#[test]
fn test_error_propagates() {
    let result = some_function_that_fails();
    assert!(matches!(result, Err(Error::DimensionMismatch(_))));
}

#[test]
#[allow(deprecated)]
fn test_deprecated_alias_works() {
    let err: crate::ImgprocError =
        Error::DimensionMismatch("test".into());
    // Should compile with deprecation warning
}
```

---

## When to Use Each Error Type

```
InvalidInput
├─ Parameter out of valid range
├─ Array/vector empty when size > 0 required
├─ Null pointer where not allowed
└─ Invalid enum variant value

DimensionMismatch
├─ Matrix shape wrong for operation
├─ Image height ≠ expected
└─ Vector sizes don't match

MemoryError
├─ Allocation failed
├─ Out of memory
└─ Memory access violation

IoError
├─ File not found
├─ File read/write failed
├─ Path invalid
└─ Permission denied

RuntimeError
├─ Unexpected null
├─ Scheduler initialization
├─ Queue submission failed
└─ Generic "something went wrong"

AlgorithmError
├─ Convergence failed
├─ Optimization diverged
├─ SVD failed
└─ Numerical instability

SfMError
├─ Triangulation failed (point at infinity)
├─ Essential matrix invalid
└─ Bundle adjustment failed

CalibrationError
├─ Chessboard detection failed
├─ Calibration board not found
└─ Camera model incompatible

ImgprocError
├─ Filter operation invalid for image type
├─ Color space conversion failed
└─ Interpolation mode unsupported

VideoError
├─ Video file not found
├─ Codec unsupported
├─ Frame grab failed
└─ Camera not opened

FeatureError
├─ Feature detection found 0 features
├─ Descriptor extraction failed
└─ Matcher initialization failed

... (etc. for other domains)
```

---

## File Modification Checklist

For cv-sfm (example):

```
☐ sfm/src/lib.rs
   ☐ Remove: pub enum SfmError { ... }
   ☐ Remove: pub type Result<T> = Result<T, SfmError>
   ☐ Add: pub use cv_core::{Error, Result};
   ☐ Add: #[deprecated] pub type SfmError = Error;

☐ sfm/src/triangulation.rs
   ☐ Search for all SfmError:: occurrences
   ☐ Replace with Error::SfMError(...)
   ☐ Remove: pub enum SfmError
   ☐ Remove: pub type Result<T>
   ☐ Verify: cargo check -p cv-sfm

☐ sfm/src/bundle_adjustment.rs
   ☐ Similar changes as triangulation.rs
   ☐ Verify: cargo check -p cv-sfm

☐ Tests
   ☐ Update error matches to use cv_core::Error
   ☐ Verify: cargo test -p cv-sfm
```

---

## One-Liner Commands

```bash
# Find all custom error types to migrate
find . -name "*.rs" -exec grep -l "pub enum.*Error" {} \;

# Count functions using custom errors (before)
grep -r "fn.*->.*Error" imgproc/src/ | wc -l

# Check migration progress
grep -r "pub enum ImgprocError" .  # Should be 0 after migration
grep -r "pub type ImgprocError" .  # Should be 1 (the alias)

# Run all tests
cargo test --all-features --lib

# Full verification
cargo check && cargo test && cargo clippy --all -- -D warnings
```

---

## Phase Summary

| Phase | Tasks | Time | Status |
|-------|-------|------|--------|
| Audit | ✅ Find custom errors | Complete | ✅ |
| | ✅ Map variants | Complete | ✅ |
| | ✅ Plan migration | Complete | ✅ |
| | ✅ Create guides | Complete | ✅ |
| Implement | ⏳ cv-sfm | 1-2h | Ready |
| | ⏳ cv-videoio | 2-3h | Queued |
| | ⏳ cv-plot | 1-2h | Queued |
| | ⏳ cv-calib3d | 4-6h | Queued |
| | ⏳ cv-imgproc | 6-8h | Queued |
| Bridge | ⏳ Add From impls | 2-3h | Queued |
| Test | ⏳ Full suite | 4h | Queued |
| Release | ⏳ Version, docs, PR | 2h | Queued |

---

## Success Indicators

✅ All five crates compile: `cargo check --all-features`
✅ All tests pass: `cargo test --all-features`
✅ No clippy warnings: `cargo clippy --all`
✅ Docs build cleanly: `cargo doc --no-deps`
✅ Each Error variant has clear semantic meaning
✅ Deprecated aliases trigger appropriate warnings
✅ No breaking changes for users of deprecated types

---

## Keep This Handy

Print this page and keep it visible while implementing:
- Error mapping table (top reference)
- Migration template (for copy/paste)
- Compilation verification checklist (run after each file)
- Common mistakes (avoid these)

---

## Questions?

Refer to:
- **Full details:** ERROR_HANDLING_AUDIT_REPORT.md
- **Step-by-step:** ERROR_HANDLING_IMPLEMENTATION_GUIDE.md
- **Overview:** ERROR_HANDLING_SUMMARY.md
- **Executive view:** ERROR_STANDARDIZATION_EXECUTIVE_SUMMARY.md

