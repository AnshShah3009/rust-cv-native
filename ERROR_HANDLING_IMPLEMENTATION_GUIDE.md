# Error Handling Standardization - Implementation Guide

## Overview

This guide provides step-by-step instructions for standardizing error handling in rust-cv-native, migrating from crate-specific error types to the unified `cv_core::Error` type.

---

## Architecture Decision

### Why Unified Error Types?

1. **Consistency:** Single error type across entire codebase
2. **Interoperability:** Easy error propagation between crates
3. **Maintainability:** One error enum to update instead of many
4. **API Clarity:** Users understand all possible error cases
5. **Rust Idioms:** Aligns with standard library patterns

### Error Hierarchy

```
cv_core::Error (canonical)
├── Used by default in all public APIs
├── 21 variants covering all domains
└── Referenced by all specialized layers

cv_hal::Error (lower-level)
├── HAL-specific concerns
├── Wraps cv_core::Error
├── Implements From<cv_hal::Error> for cv_core::Error

cv_runtime::Error (orchestration)
├── Runtime-specific concerns
├── Wraps both cv_hal and cv_core errors
├── Implements From<cv_runtime::Error> for cv_core::Error

Domain errors (imgproc, calib3d, sfm, etc.)
└── DEPRECATED in favor of cv_core::Error
    └── Kept as type aliases for backward compatibility
```

---

## Phase 1: Standardize High-Priority Crates (5 crates)

### Template for Each Crate Migration

#### Step 1: Update lib.rs

**Location:** `{crate}/src/lib.rs`

**Pattern:**

```rust
// BEFORE
pub type Result<T> = std::result::Result<T, CustomError>;

#[derive(Debug, thiserror::Error)]
pub enum CustomError {
    #[error("Error type 1: {0}")]
    Type1(String),
    #[error("Error type 2: {0}")]
    Type2(String),
}

// AFTER
pub use cv_core::{Error, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Error instead. This type exists only for backward compatibility."
)]
pub type CustomError = cv_core::Error;
```

**Why this approach:**
- No breaking changes to existing code
- Deprecation warnings guide users to migrate
- Both old and new code work transitionally
- Full forward compatibility

---

#### Step 2: Update Module Error Returns

**For each module returning custom error:**

```rust
// BEFORE
use crate::CustomError;

pub fn do_something() -> Result<Output> {
    if some_condition {
        return Err(CustomError::Type1("message".into()));
    }
    Ok(result)
}

// AFTER
use cv_core::Error;

pub fn do_something() -> Result<Output> {
    if some_condition {
        return Err(Error::DomainVariant("message".into()));
    }
    Ok(result)
}
```

**Key rule:** Replace each CustomError variant with appropriate cv_core::Error variant.

---

#### Step 3: Update Error Conversions

**For each From implementation:**

```rust
// BEFORE (if present)
impl From<std::io::Error> for CustomError {
    fn from(err: std::io::Error) -> Self {
        CustomError::Type1(err.to_string())
    }
}

// AFTER (optional - cv_core likely already has this)
impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err.to_string())
    }
}
```

**Note:** Most standard type conversions already exist in cv_core.

---

### Crate 1: cv-sfm (Smallest, Easiest)

**Files to modify:**
- `sfm/src/lib.rs`
- `sfm/src/triangulation.rs`
- `sfm/src/bundle_adjustment.rs` (if it has custom errors)

**Step-by-step:**

1. **Read current structure:**
   ```bash
   grep -n "SfmError" sfm/src/triangulation.rs
   ```

2. **Update lib.rs:**
   ```rust
   // Before
   pub use triangulation::SfmError;
   pub type Result<T> = std::result::Result<T, SfmError>;

   // After
   pub use cv_core::{Error, Result};

   #[deprecated(since = "0.1.0", note = "Use cv_core::Error instead")]
   pub type SfmError = cv_core::Error;
   ```

3. **Update triangulation.rs:**
   ```rust
   // Remove this:
   pub enum SfmError { ... }
   pub type Result<T> = std::result::Result<T, SfmError>;

   // Replace all SfmError::TriangulationFailed(msg) with:
   cv_core::Error::SfMError(msg)
   ```

4. **Update bundle_adjustment.rs:**
   ```rust
   // Update error returns similarly
   ```

5. **Test:**
   ```bash
   cd sfm
   cargo check
   cargo test
   ```

---

### Crate 2: cv-videoio (Medium, Independent)

**Files to modify:**
- `videoio/src/lib.rs`
- `videoio/src/backends/*.rs`

**Variant Mapping:**
```rust
VideoError::Io(err)              → Error::IoError(err.to_string())
VideoError::Backend(msg)         → Error::RuntimeError(msg) or Error::DeviceError(msg)
VideoError::InvalidParameters(m) → Error::InvalidInput(m)
VideoError::CaptureFailed(msg)   → Error::VideoError(msg)
```

**Process:**
1. Update lib.rs with Result and deprecated type alias
2. Update each backend implementation file
3. Replace error constructors
4. Test with: `cargo check --all-features`

---

### Crate 3: cv-plot (Small, Visualization Only)

**Files to modify:**
- `plot/src/lib.rs`
- `plot/src/export.rs`
- `plot/src/chart.rs`

**Variant Mapping:**
```rust
PlotError::InvalidData(msg) → Error::InvalidInput(msg)
PlotError::Io(err)         → Error::IoError(err.to_string())
PlotError::Export(msg)     → Error::Other(msg)
```

**Process:**
1. Update lib.rs with Result and deprecated alias
2. Update export.rs and chart.rs
3. Test individually, then with: `cargo check`

---

### Crate 4: cv-calib3d (Large, Many Dependents)

**Files to modify:**
- `calib3d/src/lib.rs`
- `calib3d/src/pnp.rs`
- `calib3d/src/calibration.rs`
- `calib3d/src/pattern.rs`
- `calib3d/src/distortion.rs`
- `calib3d/src/essential_fundamental.rs`
- `calib3d/src/project.rs`
- `calib3d/src/stereo.rs`
- `calib3d/src/triangulation.rs`

**Variant Mapping:**
```rust
CalibError::InvalidParameters(m) → Error::InvalidInput(m)
CalibError::SvdFailed(msg)       → Error::AlgorithmError(msg)
CalibError::NumericalError(msg)  → Error::CalibrationError(msg)
CalibError::Io(err)             → Error::IoError(err.to_string())
CalibError::Image(err)          → Error::ImgprocError(err.to_string())
```

**Process:**
1. Update lib.rs with deprecated alias
2. Update pnp.rs (most critical, used in triangulation)
3. Update calibration.rs, pattern.rs, etc.
4. Verify with: `cargo check --all-features`
5. Run tests: `cargo test calib3d`

**Critical:** Do not proceed to imgproc until calib3d compiles cleanly.

---

### Crate 5: cv-imgproc (Largest, Most Dependents)

**Files to modify:**
- `imgproc/src/lib.rs`
- `imgproc/src/bilateral.rs`
- `imgproc/src/color.rs`
- `imgproc/src/contours.rs`
- `imgproc/src/convolve.rs`
- `imgproc/src/edges.rs`
- `imgproc/src/histogram.rs`
- `imgproc/src/hough.rs`
- `imgproc/src/local_threshold.rs`
- `imgproc/src/morph.rs`
- `imgproc/src/resize.rs`
- `imgproc/src/simd.rs`
- `imgproc/src/template_matching.rs`
- `imgproc/src/threshold.rs`
- And more...

**Variant Mapping:**
```rust
ImgprocError::ImageError(m)        → Error::ImgprocError(m)
ImgprocError::AlgorithmError(m)    → Error::AlgorithmError(m)
ImgprocError::UnsupportedFormat(m) → Error::ParseError(m)
ImgprocError::DimensionMismatch(m) → Error::DimensionMismatch(m)
```

**Process:**
1. Update lib.rs FIRST
2. Use script to find all custom error constructors:
   ```bash
   grep -r "ImgprocError::" imgproc/src/
   ```
3. Update each file systematically
4. Verify: `cargo check --all-features`
5. Run full test suite

---

## Phase 2: Add From Implementations (4 crates)

### For Lower-Level Layers

These crates need From implementations to bridge to cv_core::Error.

#### cv-hal Error Bridging

**File:** `hal/src/lib.rs`

Add after Error enum definition:

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

**Verification:**
```bash
cargo check -p cv-hal
```

---

#### cv-runtime Error Bridging

**File:** `runtime/src/lib.rs`

Add similar From implementation:

```rust
impl From<runtime::Error> for cv_core::Error {
    fn from(err: runtime::Error) -> Self {
        match err {
            runtime::Error::RuntimeError(s) => cv_core::Error::RuntimeError(s),
            runtime::Error::MemoryError(s) => cv_core::Error::MemoryError(s),
            runtime::Error::ConcurrencyError(s) => cv_core::Error::ConcurrencyError(s),
            runtime::Error::NotSupported(s) => cv_core::Error::Other(s),
            runtime::Error::HalError(e) => e.into(), // Uses From<hal::Error>
            runtime::Error::CoreError(e) => e,
            runtime::Error::InitError(s) => cv_core::Error::RuntimeError(s),
        }
    }
}
```

---

#### cv-scientific Error Bridging

**File:** `scientific/src/lib.rs`

```rust
impl From<scientific::Error> for cv_core::Error {
    fn from(err: scientific::Error) -> Self {
        match err {
            scientific::Error::DecompositionError(s) => cv_core::Error::AlgorithmError(s),
            scientific::Error::InvalidInput(s) => cv_core::Error::InvalidInput(s),
            scientific::Error::MathError(s) => cv_core::Error::AlgorithmError(s),
            scientific::Error::IoError(e) => cv_core::Error::IoError(e.to_string()),
            scientific::Error::CoreError(e) => e,
        }
    }
}
```

---

#### cv-3d Error Bridging

**File:** `3d/src/lib.rs`

```rust
impl From<cv_3d::Error> for cv_core::Error {
    fn from(err: cv_3d::Error) -> Self {
        match err {
            cv_3d::Error::JoinError(e) => cv_core::Error::RuntimeError(e.to_string()),
            cv_3d::Error::RuntimeError(s) => cv_core::Error::RuntimeError(s),
            cv_3d::Error::InternalError(s) => cv_core::Error::Other(s),
        }
    }
}
```

---

## Phase 3: Testing & Verification

### Comprehensive Testing Checklist

#### Unit Test Verification
```bash
# Test each crate individually
cargo test -p cv-sfm
cargo test -p cv-videoio
cargo test -p cv-plot
cargo test -p cv-calib3d
cargo test -p cv-imgproc

# Test integration
cargo test --all-features
```

#### Compilation Verification
```bash
# Full workspace check
cargo check --all-features

# Clippy warnings
cargo clippy --all -- -D warnings

# Documentation
cargo doc --no-deps --all-features
```

#### Error Path Testing

Create integration test file: `tests/error_handling_integration.rs`

```rust
#[test]
fn test_imgproc_error_dimension_mismatch() {
    let result = cv_imgproc::validate_image_size(0, 480);
    match result {
        Err(cv_core::Error::DimensionMismatch(_)) => {},
        _ => panic!("Expected DimensionMismatch error"),
    }
}

#[test]
fn test_calib3d_error_invalid_input() {
    let result = cv_calib3d::solve_pnp_dlt(
        &[], // empty array - should fail
        &[],
        &Default::default(),
    );
    match result {
        Err(cv_core::Error::InvalidInput(_)) => {},
        _ => panic!("Expected InvalidInput error"),
    }
}

#[test]
fn test_error_propagation() {
    fn call_chain() -> cv_core::Result<()> {
        cv_calib3d::solve_pnp_dlt(&[], &[], &Default::default())?;
        Ok(())
    }

    let result = call_chain();
    assert!(result.is_err());
}
```

#### Backward Compatibility Testing

```rust
#[test]
#[allow(deprecated)]
fn test_deprecated_imgproc_error_alias() {
    let _alias: cv_imgproc::ImgprocError =
        cv_core::Error::DimensionMismatch("test".into());
    // Compiles with deprecation warning - exactly what we want
}
```

---

## Phase 4: Comprehensive Testing Commands

### Pre-Release Testing Sequence

Run these in order - stop if any fail:

```bash
# 1. Check code compiles
cargo check --all-features
echo "✓ Compilation check passed"

# 2. Run all unit tests
cargo test --all-features
echo "✓ All tests passed"

# 3. Check code style
cargo clippy --all -- -D warnings
echo "✓ Clippy analysis passed"

# 4. Build documentation
cargo doc --no-deps --all-features
echo "✓ Documentation builds"

# 5. Build in release mode
cargo build --release
echo "✓ Release build succeeded"

# 6. Run specific error path tests
cargo test error_handling_integration
echo "✓ Error handling integration tests passed"
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Circular Dependency
**Problem:** cv_core tries to use hal::Error before hal defines its Error
**Solution:** Error types should not depend on each other, use impl From<> instead

### Pitfall 2: Lost Error Context
**Problem:** Converting CustomError to Error loses domain information
**Solution:** Always use the most specific cv_core::Error variant
```rust
// WRONG
SfmError::TriangulationFailed(msg) → Error::RuntimeError(msg)

// RIGHT
SfmError::TriangulationFailed(msg) → Error::SfMError(msg)
```

### Pitfall 3: Incomplete Migration
**Problem:** Module still uses old error type in some functions
**Solution:** Search entire crate for custom error type before closing
```bash
grep -r "CustomError" {crate}/src/
# Should return 0 results (except in type alias)
```

### Pitfall 4: Test Failures
**Problem:** Tests use deprecated error types
**Solution:** Update tests to use cv_core::Error variants
```rust
// Old test
#[test]
fn test_error() {
    assert!(matches!(result, Err(ImgprocError::DimensionMismatch(_))));
}

// New test
#[test]
fn test_error() {
    assert!(matches!(result, Err(cv_core::Error::DimensionMismatch(_))));
}
```

---

## Rollback Plan

If anything goes wrong during migration:

```bash
# 1. Identify problem
cargo check --all-features  # See what breaks

# 2. Quick rollback (git)
git checkout -- {affected_files}

# 3. Verify rollback
cargo check --all-features

# 4. Root cause analysis before retrying
```

**Key principle:** Never merge incomplete migrations. All 5 crates in Phase 1 should be standardized before commit.

---

## Performance Impact

**Migration is zero-cost:**
- Error types are thin wrappers (single enum)
- No runtime overhead change
- Same error propagation mechanism
- Result<T> type remains identical

No performance testing needed - this is purely API refactoring.

---

## Documentation Updates

After standardization, update:

1. **Crate README.md:**
   ```markdown
   ## Error Handling

   Functions return `cv_core::Result<T>` with `cv_core::Error` variants:
   - `InvalidInput` - Invalid parameters
   - `AlgorithmError` - Convergence or algorithm failure
   - ... (relevant variants)
   ```

2. **Function documentation:**
   ```rust
   /// Validates image dimensions.
   ///
   /// # Errors
   /// Returns `Error::DimensionMismatch` if width or height is 0.
   pub fn validate_image_size(width: u32, height: u32) -> Result<()>
   ```

3. **CHANGELOG.md:**
   ```
   ## [0.2.0] - 2026-02-24

   ### Changed
   - BREAKING: Standardized error handling to use `cv_core::Error` across all crates
   - Deprecated `ImgprocError`, `CalibError`, `SfmError`, `VideoError`, `PlotError`
   - All public APIs now return `cv_core::Result<T>`

   ### Migration Guide
   - Replace `ImgprocError` with `cv_core::Error`
   - Replace `your_crate::Result<T>` with `cv_core::Result<T>`
   - See ERROR_HANDLING_MIGRATION_GUIDE.md for details
   ```

---

## Validation Checklist

Before declaring standardization complete:

- [ ] All 5 crates use cv_core::Result<T> for public APIs
- [ ] Deprecated type aliases exist in all 5 crates
- [ ] From implementations added for hal, runtime, scientific, 3d
- [ ] `cargo check --all-features` passes cleanly
- [ ] `cargo test --all-features` passes all tests
- [ ] `cargo clippy --all` reports no warnings
- [ ] `cargo doc` builds without warnings
- [ ] Error path integration tests added and passing
- [ ] Backward compatibility tests confirm deprecation warnings work
- [ ] All files updated (no stray CustomError references)
- [ ] Documentation updated with new error variants
- [ ] CHANGELOG.md updated with breaking changes

---

## Success Metrics

✅ Implementation complete when:
- 0 compilation errors
- 100% test pass rate
- 0 clippy warnings
- 0 doc build warnings
- All deprecated aliases trigger warnings (as intended)
- Error propagation works across crate boundaries
- Users can still compile old code (with deprecation warnings)

---

## Next Session Checklist

To continue work after this audit:

1. Create branch: `git checkout -b feature/standardize-error-handling`
2. Start with cv-sfm (smallest crate)
3. After each crate: `cargo check --all-features`
4. Keep error mapping reference handy
5. Use this implementation guide for exact patterns
6. Test frequently - don't batch changes
7. Commit each crate standardization separately

