# Storage Redesign Migration Guide

**Phase 2 Storage Redesign**
**Version:** 1.0
**Date:** March 2026

## Overview

This document guides library developers through migrating from the legacy slice-based tensor storage to the new handle-based storage system introduced in Phase 2 of the rust-cv-native storage redesign.

### Key Changes

- **Before:** Tensors held `Vec<T>` directly; slices provided data access
- **After:** Tensors use `Storage<T>` trait with handle-based identification; supports both CPU and GPU storage
- **Benefit:** Unified GPU/CPU interface, zero-copy GPU operations, better device tracking

### Timeline

- **Phase 2 (Current):** CPU storage with `CpuStorage<T>`, introduction of `Storage<T>` trait
- **Phase 2.5:** Deprecation warnings for legacy slice-based APIs
- **Phase 3:** GPU storage (`GpuStorage`), unified projection API
- **Phase 4:** Full migration complete, legacy APIs removed

## Migration Patterns

### Pattern 1: CPU Tensors (Recommended)

For operations working exclusively with CPU data, use the `CpuStorageMarker` trait for direct memory access.

#### Before (Legacy)

```rust
use cv_core::Tensor;

let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], shape)?;

// Direct slice access
if let Some(slice) = tensor.as_slice() {
    println!("Data: {:?}", slice);
}

// Mutable access
if let Some(slice_mut) = tensor.as_mut_slice() {
    slice_mut[0] = 99.0;
}
```

#### After (Phase 2)

```rust
use cv_core::{Tensor, CpuStorageMarker};

let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], shape)?;

// Access storage directly
use cv_core::CpuStorageMarker;

// Get immutable slice
let slice = CpuStorageMarker::as_slice(&tensor.storage());
println!("Data: {:?}", slice);

// Get mutable slice
let slice_mut = CpuStorageMarker::as_mut_slice(&mut tensor.storage());
slice_mut[0] = 99.0;
```

**Migration Steps:**
1. Import `CpuStorageMarker` from `cv_core::storage`
2. Call `.storage()` on your Tensor to get the storage reference
3. Use `CpuStorageMarker::as_slice()` for immutable access
4. Use `CpuStorageMarker::as_mut_slice()` for mutable access
5. Implement the `CpuStorageMarker` trait if creating custom storage types

#### Advantage

- Direct memory access with zero overhead
- Type-safe - compiler verifies you're using CPU storage
- Works with generic `Storage<T>` implementations

#### Full Example

```rust
use cv_core::{CpuStorage, storage::CpuStorageMarker};

// Create storage
let storage = CpuStorage::from_vec(vec![1.0f32, 2.0, 3.0])?;

// Access as slice
let data = CpuStorageMarker::as_slice(&storage);
println!("First element: {}", data[0]);

// Verify handle uniqueness for tracking
let handle = storage.handle();
println!("Storage handle: {:?}", handle);

// Modify data
let mut storage_mut = storage.clone();
let data_mut = CpuStorageMarker::as_mut_slice(&mut storage_mut);
data_mut[0] = 42.0;
```

### Pattern 2: Abstract Storage Downcasting

For functions accepting generic `Storage<T>` trait objects, use `as_any()` for runtime type checking.

#### Before (Legacy - Not Applicable)

The legacy system didn't have trait objects for storage.

#### After (Phase 2)

```rust
use cv_core::storage::{Storage, CpuStorageMarker};
use cv_core::CpuStorage;

fn process_storage<T: Clone + Debug + 'static>(storage: &dyn Storage<T>) -> Result<(), String> {
    // Check what kind of storage we have
    if let Some(cpu_storage) = storage.as_any().downcast_ref::<CpuStorage<T>>() {
        // It's CPU storage - we can access directly
        println!("Device: {}", match storage.device() {
            cv_core::storage::DeviceType::Cpu => "CPU",
            _ => "Other",
        });
        Ok(())
    } else {
        // Unknown storage type
        Err("Unsupported storage type".to_string())
    }
}

// Usage
let cpu_storage = CpuStorage::from_vec(vec![1.0f32, 2.0, 3.0])?;
process_storage::<f32>(&cpu_storage)?;
```

**Migration Steps:**

1. Change function signature to accept `&dyn Storage<T>`
2. Call `.as_any()` on the storage reference
3. Use `.downcast_ref::<CpuStorage<T>>()` to check type
4. For other storage types, implement similar pattern
5. Handle the `None` case for unknown types

#### Advantages

- Write generic code that works with different storage backends
- Runtime type checking for flexibility
- Enables future GPU storage support without code changes

#### Full Example with Pattern Matching

```rust
use cv_core::storage::{Storage, CpuStorageMarker};
use cv_core::CpuStorage;

fn print_storage_info<T: Clone + Debug + 'static>(storage: &dyn Storage<T>) {
    let any = storage.as_any();

    if any.downcast_ref::<CpuStorage<T>>().is_some() {
        println!("CPU Storage:");
        println!("  Handle: {:?}", storage.handle());
        println!("  Size: {}", storage.len());
        println!("  Device: CPU");
    } else {
        println!("Unknown storage type: {}", storage.data_type_name());
    }
}
```

### Pattern 3: GPU Operations (Phase 3 Preview)

For code that will eventually support GPU operations, structure your code now for future compatibility.

#### Current (Phase 2 - CPU Only)

```rust
use cv_core::storage::Storage;

fn compute_on_storage<S: Storage<f32>>(storage: &S) -> Result<f32, String> {
    // Generic over storage type - works with any implementation
    match storage.device() {
        cv_core::storage::DeviceType::Cpu => {
            // CPU path - use CpuStorageMarker for direct access
            // Code path for Phase 2
            Ok(0.0)
        }
        cv_core::storage::DeviceType::Cuda => {
            // GPU path - to be implemented in Phase 3
            Err("CUDA not yet supported".to_string())
        }
        _ => Err("Unsupported device".to_string()),
    }
}
```

#### Future (Phase 3 - With GPU Support)

```rust
use cv_core::storage::{Storage, CpuStorageMarker, GpuStorageMarker};
use cv_core::CpuStorage;
use cv_hal::GpuStorage;

fn compute_on_storage<S: Storage<f32>>(storage: &S) -> Result<f32, String> {
    // Check device type and route accordingly
    if let Some(cpu_storage) = storage.as_any().downcast_ref::<CpuStorage<f32>>() {
        // CPU path
        let data = CpuStorageMarker::as_slice(cpu_storage);
        Ok(data.iter().sum())
    } else if let Some(_gpu_storage) = storage.as_any().downcast_ref::<GpuStorage>() {
        // GPU path - Phase 3
        // Use GpuStorageMarker to transfer data
        Err("GPU not yet implemented".to_string())
    } else {
        Err("Unknown storage".to_string())
    }
}
```

**Why Structure This Way:**

1. Code compiles today but uses conditional compilation
2. When Phase 3 arrives, just remove the error return for GPU
3. Handle types aren't tightly coupled to implementation
4. Easy to add support for new accelerators later

## Breaking Changes

### Storage Trait Object Support

**Change:** Storage is no longer a trait object due to `Self: Sized` requirement.

**Impact:** You cannot write `Box<dyn Storage<T>>` directly.

**Migration:** Use `as_any()` and `downcast_ref()` for type checking instead.

```rust
// ❌ This won't work
let storage_obj: Box<dyn Storage<f32>> = Box::new(cpu_storage);

// ✓ This works
let any_ref = cpu_storage.as_any();
if let Some(downcast) = any_ref.downcast_ref::<CpuStorage<f32>>() {
    // Use downcast
}
```

### Handle-Based Identification

**Change:** Tensors no longer use implicit vector identity; use explicit `BufferHandle`.

**Impact:** Handle-based operations enable GPU memory management.

**Migration:** Store `handle` from storage when you need to track tensor identity.

```rust
// ❌ Old approach - vector pointer
let ptr = storage.data.as_ptr();

// ✓ New approach - stable handle
let handle = storage.handle();

// Use handle as map key for tracking
use std::collections::HashMap;
let mut tensor_map = HashMap::new();
tensor_map.insert(handle, tensor_data);
```

### Slice Access Changes

**Change:** Direct `.as_slice()` on Tensor now requires method disambiguation.

**Impact:** Need to specify trait method explicitly for clarity.

**Migration:**

```rust
// ❌ Ambiguous (which trait?)
let slice = tensor.as_slice();

// ✓ Clear
let slice = CpuStorageMarker::as_slice(&tensor.storage());
```

## Deprecation Timeline

### Phase 2 (Current)

- New `Storage<T>` trait available alongside legacy APIs
- `Tensor` accepts both old and new storage patterns
- No deprecation warnings

### Phase 2.5 (Q2 2026)

- Deprecation warnings added to legacy slice access patterns
- Documentation updated to recommend new APIs
- Migration guide released (this document)

### Phase 3 (Q3 2026)

- GPU storage fully integrated
- Unified projection API released
- Performance optimizations enabled

### Phase 4 (Q4 2026)

- Legacy APIs removed
- Performance improvements documented
- Migration guide archived

## Library Update Checklist

When updating your library to Phase 2 storage:

- [ ] Identify all code using `.as_slice()` on tensors
- [ ] Check if operations are CPU-only or need GPU support
- [ ] Replace with `CpuStorageMarker::as_slice()` for CPU ops
- [ ] Add device checking for future GPU compatibility
- [ ] Update function signatures to accept `Storage<T>` where applicable
- [ ] Test with both `CpuStorage` and legacy storage types
- [ ] Update documentation/examples
- [ ] Run full test suite
- [ ] File any issues with migration difficulties

## Common Patterns

### Reading CPU Tensor Data

```rust
use cv_core::{CpuStorageMarker};

fn read_tensor<T: Clone>(storage: &impl CpuStorageMarker<Element = T>) -> Vec<T> {
    CpuStorageMarker::to_vec_cpu(storage)
}
```

### Creating Tensor from Factory

```rust
use cv_core::storage::StorageFactory;
use cv_core::CpuStorage;

fn create_from_vec<T: Clone + Default + Debug>(data: Vec<T>) -> Result<CpuStorage<T>, String> {
    <CpuStorage<T> as StorageFactory<T>>::from_vec(data)
}
```

### Handle-Based Tensor Tracking

```rust
use cv_core::BufferHandle;
use std::collections::HashMap;

struct TensorRegistry {
    tensors: HashMap<BufferHandle, String>,
}

impl TensorRegistry {
    fn register(&mut self, handle: BufferHandle, name: &str) {
        self.tensors.insert(handle, name.to_string());
    }

    fn lookup(&self, handle: BufferHandle) -> Option<&str> {
        self.tensors.get(&handle).map(|s| s.as_str())
    }
}
```

## Troubleshooting

### "method `as_slice` is ambiguous"

**Problem:**
```rust
let slice = tensor.as_slice();
// error: multiple `as_slice` found
```

**Solution:**
Disambiguate by specifying the trait:
```rust
use cv_core::storage::CpuStorageMarker;
let slice = CpuStorageMarker::as_slice(&tensor.storage());
```

### "Storage<T> is not object safe"

**Problem:**
```rust
let obj: Box<dyn Storage<f32>> = ...
// error: `Storage` is not dyn compatible
```

**Solution:**
Use `as_any()` and `downcast_ref()` instead:
```rust
if let Some(cpu) = storage.as_any().downcast_ref::<CpuStorage<f32>>() {
    // Work with cpu_storage
}
```

### "StorageFactory not implemented"

**Problem:**
```rust
let storage = MyStorage::from_vec(data)?;
// error: no implementation for `StorageFactory`
```

**Solution:**
Implement the trait for your custom storage:
```rust
impl<T> StorageFactory<T> for MyStorage<T> {
    fn from_vec(data: Vec<T>) -> Result<Self, String> {
        // Your implementation
    }

    fn new(size: usize, default: T) -> Result<Self, String> {
        // Your implementation
    }
}
```

## Backward Compatibility

### Deprecated APIs (Still Working)

These APIs still work in Phase 2 but will be deprecated in Phase 2.5:

- Direct `Tensor.as_slice()` without trait specification
- `Storage<T>` as trait object (use `as_any()` instead)
- `PoseF32` type alias (use `Pose` with `from()`/`into()`)

### Migration Path

Legacy code will continue to work but should be updated to use:

1. Explicit trait method calls
2. Handle-based tracking instead of pointer identity
3. Device type checking before operations

## FAQ

**Q: Do I need to change all my code at once?**
A: No. Legacy patterns still work in Phase 2. You can migrate incrementally. Start with critical paths and public APIs.

**Q: Will my code break when Phase 3 arrives?**
A: Only if you're using deprecated APIs without updating. Follow the deprecation timeline warnings.

**Q: Can I use CPU and GPU storage in the same codebase?**
A: Yes! Phase 3 will provide `GpuStorage` with same `Storage<T>` interface. Code written with device checking now will work unchanged.

**Q: Should I store handles or storage references?**
A: Store handles for identity tracking (e.g., in caches/maps). Store references for data access.

**Q: What about performance?**
A: Phase 2 has zero runtime overhead vs. legacy. Phase 3 gains efficiency from GPU memory management.

## Related Documentation

- [ARCHITECTURE_STORAGE.md](./ARCHITECTURE_STORAGE.md) - Design rationale and internal architecture
- [PHASE2_VERIFICATION.md](../PHASE2_VERIFICATION.md) - Build verification and test results
- cv-core API docs: `cargo doc -p cv-core --open`

## Support

For migration questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review [Common Patterns](#common-patterns)
3. Read [ARCHITECTURE_STORAGE.md](./ARCHITECTURE_STORAGE.md) for design details
4. File an issue with reproduction steps
