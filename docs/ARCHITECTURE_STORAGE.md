# Storage Architecture - Design Rationale and Internal Design

**Phase 2 Storage Redesign**
**Version:** 1.0
**Date:** March 2026

## Executive Summary

The Phase 2 storage redesign introduces a unified `Storage<T>` trait that abstracts tensor data across different compute devices (CPU, GPU) while maintaining zero-cost abstractions and enabling efficient device-specific optimizations. This document explains the design decisions, architecture, and invariants that enable this unification.

## Design Goals

### 1. Unified GPU/CPU Interface

**Goal:** Single `Storage<T>` trait supports both CPU (RAM) and GPU (VRAM) storage.

**Rationale:**
- Algorithms need to work transparently on either device
- Reduces code duplication for device-agnostic operations
- Enables future support for TPUs, specialized accelerators

**Implementation:**
- `Storage<T>` defines common interface (handle, capacity, shape)
- Device-specific traits (`CpuStorageMarker`, `GpuStorageMarker`) provide specialized access
- Implementations provide efficient access patterns for their device

### 2. Type Safety with Flexibility

**Goal:** Compile-time guarantees with runtime flexibility.

**Rationale:**
- Monomorphic code for performance-critical paths
- Polymorphic code for generic operations
- No runtime type errors for well-typed code

**Implementation:**
- `Storage<T>` bounds ensure T is managed correctly
- `as_any()` enables safe downcasting to concrete types
- Trait system ensures only valid combinations compile

### 3. Zero-Copy Design

**Goal:** GPU operations never need intermediate CPU copies.

**Rationale:**
- PCI-e bandwidth is expensive (~8 GB/s)
- Large tensors (megabytes) cost milliseconds to copy
- GPU-to-GPU operations should not touch CPU RAM

**Implementation:**
- Handles identify data location without copying
- Device-specific storage types manage their own memory
- Runtime tracks handle→device mapping

### 4. Handle-Based Identification

**Goal:** Stable, lightweight identifiers for tensor data.

**Rationale:**
- Vector pointers change if vector reallocates
- GPU buffers have no pointer equivalents
- Handles enable caching, deduplication, memory tracking

**Implementation:**
- `BufferHandle(u64)` - atomic counter for unique IDs
- Copy/Clone semantics - cheap to pass around
- Hashable - can be used in maps for tracking

## Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Storage<T> Trait                       │
│  (Device-agnostic interface for tensor storage)          │
├─────────────────────────────────────────────────────────┤
│ • handle() → BufferHandle                               │
│ • capacity() → usize                                    │
│ • shape() → &[usize]                                    │
│ • len() → usize                                         │
│ • data_type_name() → &'static str                      │
│ • as_any() → &dyn Any  (for downcasting)               │
│ • device() → DeviceType                                │
│ • as_slice()/as_mut_slice()  (backward compat)        │
└─────────────────────────────────────────────────────────┘
         ▲                ▲                    ▲
         │                │                    │
    ┌────┴────┐      ┌────┴────┐      ┌──────┴──────┐
    │          │      │         │      │             │
 CpuStorage  GpuStorage WgpuGpuStorage (future)  (future)
 (Vec<T>)   (handle+id) (wgpu buffer)  MLXStorage   DMLStorage
    │          │      │         │      │             │
    └────┬──────┴──────┴─────────┴──────┴─────────────┘
         │
    Implements Storage<T>
    + device-specific traits
```

## Components

### 1. BufferHandle

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u64);
```

**Purpose:** Lightweight, unique identifier for storage buffers.

**Properties:**
- **Copy:** Cheap to pass around (8 bytes)
- **Hashable:** Can be used as map key
- **Unique:** Generated from atomic counter, guaranteed unique
- **Stable:** Never changes during storage lifetime

**Invariants:**
- IDs never repeat (monotonically increasing)
- Two different storages always have different handles
- Clone preserves the same handle

**Use Cases:**
- Storage registry: `HashMap<BufferHandle, StorageInfo>`
- Tensor identity: "Is this the same underlying buffer?"
- Memory management: "Which device owns this handle?"
- Deduplication: "Do we already have this data cached?"

### 2. Storage<T> Trait

```rust
pub trait Storage<T: 'static>: Debug + Clone {
    fn handle(&self) -> BufferHandle;
    fn capacity(&self) -> usize;
    fn shape(&self) -> &[usize];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn data_type_name(&self) -> &'static str;
    fn as_any(&self) -> &dyn Any;
    fn device(&self) -> DeviceType { DeviceType::Cpu }
    fn as_slice(&self) -> Option<&[T]> { None }
    fn as_mut_slice(&mut self) -> Option<&mut [T]> { None }
}
```

**Design Decisions:**

1. **Clone instead of Rc/Arc:**
   - Implementations manage their own sharing
   - CpuStorage clones the Vec (cheap)
   - GpuStorage clones the handle (free)
   - Avoids Arc overhead for types that don't need it

2. **Self: Sized (not object-safe):**
   - Allows Clone, Copy bounds in generic code
   - Prevents accidental trait object overhead
   - Forces explicit `as_any()` for polymorphism

3. **Optional Methods:**
   - `as_slice()`: May be None for GPU storage
   - `as_mut_slice()`: May be None for GPU storage
   - `device()`: Defaults to CPU for backward compat
   - Backward compat without overhead

4. **as_any() for Downcasting:**
   - Safe runtime type checking
   - Standard Rust pattern
   - Used by migration shim and device-specific code

**Invariants:**
- `len() <= capacity()`
- `shape.product() == len()` (if shape is not empty)
- Cloning produces a storage with the same handle
- `data_type_name()` matches the concrete T

### 3. CpuStorage<T>

```rust
#[derive(Debug, Clone)]
pub struct CpuStorage<T> {
    handle: BufferHandle,
    shape: Vec<usize>,
    data: Vec<T>,
}
```

**Purpose:** CPU-resident tensor storage backed by `Vec<T>`.

**Properties:**
- **Direct Access:** as_slice() returns actual data
- **Mutable:** as_mut_slice() for in-place operations
- **Shaped:** Supports multi-dimensional tensors (2D, 3D, etc.)
- **Cloneable:** Clones the Vec, preserves handle

**Performance Characteristics:**
- Creation: O(n) for allocation
- Clone: O(n) for data copying
- Access: O(1) - just pointer dereference
- Memory: Same as Vec<T>

**Methods:**

```rust
impl<T: Clone + Debug + 'static> CpuStorage<T> {
    pub fn from_vec(data: Vec<T>) -> Result<Self, String>;
    pub fn from_vec_with_shape(data: Vec<T>, shape: Vec<usize>) -> Result<Self, String>;
    pub fn new_with_shape(shape: Vec<usize>) -> Result<Self, String> where T: Default;
    pub fn new(size: usize, default: T) -> Result<Self, String>;
    pub fn to_vec(&self) -> Vec<T> where T: Clone;
}
```

**Invariants:**
- `data.len()` always matches `len()`
- If shape is not empty: `shape.product() == data.len()`
- Mutable operations maintain these invariants

### 4. CpuStorageMarker Trait

```rust
pub trait CpuStorageMarker {
    type Element;
    fn as_slice(&self) -> &[Self::Element];
    fn as_mut_slice(&mut self) -> &mut [Self::Element];
    fn to_vec_cpu(&self) -> Vec<Self::Element> where Self::Element: Clone;
}
```

**Purpose:** Marker trait for CPU-resident storage.

**Benefits:**
- Compile-time assertion that storage is on CPU
- Type-safe slice access without Option
- Enables generic CPU algorithms

**Implementation:**
```rust
impl<T: Clone + Debug + 'static> CpuStorageMarker for CpuStorage<T> {
    type Element = T;
    fn as_slice(&self) -> &[T] { &self.data }
    fn as_mut_slice(&mut self) -> &mut [T] { &mut self.data }
    fn to_vec_cpu(&self) -> Vec<T> { self.data.clone() }
}
```

### 5. GpuStorage

```rust
#[derive(Debug, Clone)]
pub struct GpuStorage {
    handle: BufferHandle,
    device_id: u32,
    shape: Vec<usize>,
}
```

**Purpose:** GPU-resident tensor storage with lightweight handle-based design.

**Design Rationale:**
- **No Data:** Actual GPU buffer managed by runtime
- **Handle:** Identifies data in GPU memory
- **Device ID:** Tracks which GPU owns this data
- **Shape:** Supports multi-dimensional tensors on GPU

**Performance Characteristics:**
- Creation: O(1) - just creates struct
- Clone: O(1) - copies handle and device_id
- Access: Only via GpuStorageMarker (transfers data)
- Memory: Only 24 bytes on CPU side

**Methods:**

```rust
impl GpuStorage {
    pub fn new(handle: BufferHandle, device_id: u32, shape: Vec<usize>) -> Self;
    pub fn device_id(&self) -> u32;
}
```

### 6. GpuStorageMarker Trait

```rust
pub trait GpuStorageMarker {
    type Element;
    fn to_cpu(&self) -> Result<Vec<Self::Element>, String> where Self::Element: Clone;
}
```

**Purpose:** Marker trait for GPU-resident storage.

**Benefits:**
- Explicit GPU-to-CPU transfer (users can't accidentally do this)
- Type-safe access to GPU data
- Enables GPU algorithms

**Implementation:**
```rust
impl GpuStorageMarker for GpuStorage {
    type Element = f32;
    fn to_cpu(&self) -> Result<Vec<f32>, String> {
        // Call GPU runtime to transfer data
        // Phase 3: Implement actual transfer
    }
}
```

### 7. StorageFactory Trait

```rust
pub trait StorageFactory<T: 'static>: Storage<T> + Sized {
    fn from_vec(data: Vec<T>) -> Result<Self, String>;
    fn new(size: usize, default: T) -> Result<Self, String>;
}
```

**Purpose:** Factory pattern for creating storage instances.

**Benefits:**
- Generic factory methods for any storage type
- Enables polymorphic storage creation
- Works with generic code

**Implementation:**
```rust
impl<T: Clone + Debug + 'static> StorageFactory<T> for CpuStorage<T> {
    fn from_vec(data: Vec<T>) -> Result<Self, String> {
        Ok(CpuStorage { handle: generate_handle_id(), shape: vec![], data })
    }
    fn new(size: usize, default: T) -> Result<Self, String> {
        Ok(CpuStorage { handle: generate_handle_id(), shape: vec![], data: vec![default; size] })
    }
}
```

### 8. DeviceType Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
    Dml,
    TensorRT,
}
```

**Purpose:** Device type identification and routing.

**Use Cases:**
- Conditional branching based on device
- Logging/profiling device information
- Memory management policies

**Invariants:**
- `DeviceType::Cpu` for CPU storage
- Different GPU types for GPU storage
- Matches actual storage location

## Data Flow Examples

### CPU Tensor Creation and Use

```
User creates tensor:
  Tensor::from_vec(vec![1.0, 2.0, 3.0])
         ↓
    CpuStorage::from_vec()
         ↓
    Generate BufferHandle (e.g., #42)
         ↓
    Create CpuStorage {
        handle: #42,
        shape: [],
        data: vec![1.0, 2.0, 3.0]
    }
         ↓
    Tensor wraps CpuStorage
         ↓
User accesses data:
  storage.as_slice()
         ↓
    Returns &[1.0, 2.0, 3.0]
         ↓
    O(1) - just pointer dereference
```

### Device-Agnostic Operation

```
Generic function:
  fn sum_storage<S: Storage<f32>>(s: &S) -> f32

User calls with CPU storage:
  let result = sum_storage(&cpu_storage)
         ↓
  Inside sum_storage:
    match storage.device() {
        DeviceType::Cpu => {
            // Cast to CpuStorage<f32>
            let cpu = storage.as_any().downcast_ref::<CpuStorage<f32>>()?;
            // Use CpuStorageMarker for direct access
            return CpuStorageMarker::as_slice(cpu).iter().sum();
        }
        DeviceType::Cuda => {
            // Phase 3: GPU path
            return Err("GPU not implemented");
        }
        _ => return Err("Unknown device");
    }
         ↓
  Result: 6.0
```

### Multi-Device Operation (Phase 3 Preview)

```
User creates two tensors on different devices:
  let cpu_tensor = Tensor::from_vec(...);  // DeviceType::Cpu
  let gpu_tensor = create_gpu_tensor(...);  // DeviceType::Cuda

User processes both:
  let result = unified_operation(&cpu_tensor, &gpu_tensor)
         ↓
  Inside unified_operation:
    // Transfer GPU data to CPU if needed
    let gpu_data = gpu_storage.to_cpu()?;  // Explicit transfer
    let result = cpu_operation(&cpu_data, &gpu_data);
         ↓
  Result computed on CPU

Future optimization (Phase 4):
    // Or keep data on GPU
    let gpu_result = gpu_operation(&cpu_tensor_on_gpu, &gpu_tensor)?;
         ↓
  Result stays on GPU
```

## Error Handling

### Storage Creation

```rust
impl<T: Clone + Debug + 'static> CpuStorage<T> {
    pub fn from_vec_with_shape(
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(format!("Shape mismatch: expected {}, got {}", expected, data.len()));
        }
        // ... create storage
    }
}
```

**Invariant Checked:** `shape.product() == data.len()`

### Device Access

```rust
fn cpu_operation(storage: &impl CpuStorageMarker) -> Result<(), String> {
    // Type system guarantees this is CPU storage
    let slice = CpuStorageMarker::as_slice(storage);
    // Use slice safely
}

// If you have generic Storage<T>:
fn generic_operation(storage: &dyn Storage<f32>) -> Result<(), String> {
    if storage.device() != DeviceType::Cpu {
        return Err("Only CPU storage supported".to_string());
    }
    // Now safe to downcast
}
```

**Invariant Checked:** Storage device type matches requested operations

## Invariants Maintained

### Storage Consistency

1. **Handle Uniqueness:** Each storage instance has a unique handle
   - Generated from atomic counter
   - Never reused
   - Enables caching and deduplication

2. **Shape Validity:** Shape dimensions multiply to equal length
   - Checked on construction
   - Empty shape for flat buffers
   - Multi-dimensional for tensors

3. **Data Consistency:** Cloned storages share handle but not data (for CPU)
   - CPU clone: new handle, cloned Vec
   - GPU clone: same handle, same VRAM location
   - This is by design (CPU clones isolate, GPU clones share)

4. **Type Consistency:** data_type_name() matches concrete T
   - Only CpuStorage<T> can claim T type
   - Downcasting verifies this
   - Type system ensures at compile time

### Device Consistency

1. **Device Type Accuracy:** DeviceType matches storage location
   - DeviceType::Cpu for CpuStorage
   - DeviceType::Cuda for GpuStorage on CUDA device
   - Used for routing and logging

2. **Device Access Validity:** Can only access data on its device
   - CpuStorage.as_slice() always works
   - GpuStorage.as_slice() returns None
   - as_any() + downcast enforces this

## Efficiency Analysis

### CPU Operations

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Create storage | O(n) | O(n) | allocate Vec |
| Clone storage | O(n) | O(n) | copy Vec |
| Access data | O(1) | O(1) | just pointer |
| Shape validation | O(1) | O(1) | multiply dimensions |
| Get handle | O(1) | O(1) | return u64 |

### GPU Operations (Phase 3)

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Create storage | O(1) | O(1) | just handle+ID |
| Clone storage | O(1) | O(1) | clone handle |
| Transfer to CPU | O(n) | O(n) | GPU→CPU copy |
| Get handle | O(1) | O(1) | return u64 |
| Device routing | O(1) | O(1) | enum match |

## Extension Points

### Adding a New Storage Type

To add support for a new device (e.g., TPU):

1. **Implement Storage<T>:**
   ```rust
   #[derive(Debug, Clone)]
   pub struct TpuStorage {
       handle: BufferHandle,
       device_id: u32,
       shape: Vec<usize>,
   }

   impl<T> Storage<T> for TpuStorage { ... }
   ```

2. **Implement Device Marker:**
   ```rust
   pub trait TpuStorageMarker {
       type Element;
       fn transfer_to_cpu(&self) -> Result<Vec<Self::Element>, String>;
   }

   impl TpuStorageMarker for TpuStorage { ... }
   ```

3. **Add to DeviceType:**
   ```rust
   pub enum DeviceType {
       // ...
       Tpu,
   }
   ```

4. **Update Runtime:**
   - Register TpuStorage in device registry
   - Implement transfer protocols
   - Add device-specific optimizations

5. **Update User Code:**
   ```rust
   match storage.device() {
       DeviceType::Tpu => {
           let tpu = storage.as_any().downcast_ref::<TpuStorage>()?;
           // Use TpuStorageMarker
       }
       // ...
   }
   ```

## Performance Considerations

### Zero-Cost Abstractions

- **Generic monomorphization:** No virtual call overhead
- **Inlining:** Simple methods inline to single instruction
- **Memory layout:** Storage<T> has no indirection layers
- **Handle generation:** Atomic operation, amortized O(1)

### Optimization Opportunities

1. **Device-Local Operations:** Keep data on device, avoid transfers
2. **Batch Transfers:** Group multiple small transfers
3. **Prefetching:** Move data before computation
4. **Memory Pooling:** Reuse allocations via handles

## Security Considerations

### Memory Safety

- **Bounds Checking:** Vec provides safety
- **Lifetime Management:** Rust's borrow checker enforces
- **Type Safety:** Downcasting is typechecked

### Device Safety (Phase 3)

- **Handle Validation:** Runtime checks handle exists
- **Access Control:** Only allow access on correct device
- **Resource Cleanup:** RAII patterns for GPU memory

## Future Extensions (Phase 3+)

### Unified Projection API

```rust
// Instead of scattered implementations:
pub fn project_point(
    point: &Point3<f64>,
    intrinsics: &CameraIntrinsics,
    extrinsics: Option<&Pose>,
    distortion: Option<&Distortion>,
) -> Point2<f64> { ... }
```

### Pose Graph Consolidation

```rust
// Single implementation in cv-optimize
pub fn optimize_pose_graph<S: Storage<f32>>(
    poses: &[Pose],
    storage: &S,
) -> Result<Vec<Pose>, Error> { ... }
```

### Advanced Features

- Device memory pools with handle→allocation mapping
- Automatic device selection (CPU/GPU)
- Tensor fusion and operator graphs
- Distributed computation support

## Related Documentation

- [STORAGE_MIGRATION.md](./STORAGE_MIGRATION.md) - Migration patterns for library updates
- [PHASE2_VERIFICATION.md](../PHASE2_VERIFICATION.md) - Implementation verification and test results
- cv-core source code: `/core/src/storage.rs`, `/core/src/buffer_handle.rs`

## Conclusion

The Phase 2 storage architecture provides a solid foundation for unified GPU/CPU tensor operations while maintaining Rust's safety guarantees and zero-cost abstractions. The design enables seamless addition of new device types and GPU-specific optimizations in Phase 3 without requiring changes to user code written with device-agnostic patterns.

Key achievements:
- **Unified Interface:** Single Storage<T> trait for all devices
- **Type Safety:** Compile-time checks with runtime flexibility
- **Zero-Copy:** GPU operations don't touch CPU RAM
- **Extensible:** Easy to add new device types
- **Efficient:** No overhead vs. direct Vec usage
