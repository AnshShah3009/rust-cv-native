# Crate Dependency Hierarchy

## Overview
The workspace follows a layered architecture with clear dependencies:

```
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                         │
│  rust-cv-native (main crate)                                    │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   cv-python   │      │  cv-viewer    │      │   cv-slam     │
│  (Python FFI) │      │  (GUI/Vis)    │      │   (SLAM)      │
└───────────────┘      └───────────────┘      └───────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  cv-features  │      │  cv-stereo    │      │    cv-sfm     │
│  (Keypoints)  │      │    (Depth)    │      │    (SFM)      │
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ cv-imgproc  │  │  cv-3d      │  │cv-calib3d   │  │cv-photo │ │
│  │ (2D Image)  │  │ (3D/Mesh)   │  │(Calibration)│  │(Stitch) │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RUNTIME LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ cv-runtime  │  │   cv-hal    │  │cv-optimize  │              │
│  │ (Memory/    │  │ (CPU/GPU    │  │(Solvers)    │              │
│  │  Scheduler) │  │  Backend)   │  │              │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORE LAYER (Foundation)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │   cv-core   │  │   cv-io     │  │ cv-videoio  │  │cv-dnn   │ │
│  │ (Types/     │  │ (File I/O)  │  │ (Capture)   │  │(ONNX)   │ │
│  │  PointCloud)│  │             │  │             │  │         │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Dependency Count by Crate

| Crate | # Dependents | Role |
|-------|--------------|------|
| **cv-core** | 18 | Foundation types (PointCloud, Image, etc.) |
| **cv-hal** | 6 | Hardware abstraction (CPU/GPU backends) |
| **cv-imgproc** | 6 | Image processing algorithms |
| **cv-runtime** | 3 | Memory management, task scheduling |
| **cv-features** | 2 | Feature detection (SIFT, ORB, etc.) |
| **cv-calib3d** | 2 | Camera calibration, PnP |

## Dependency Rules

✅ **Valid Dependencies:**
- Higher layers can depend on lower layers
- Same-layer crates can depend on each other (carefully)
- All crates depend on `cv-core`

❌ **Invalid/Circular Dependencies:**
- Core layer cannot depend on processing layer
- cv-hal cannot depend on cv-imgproc
- No circular dependencies between crates

## Current Crate Independence

### Fully Independent (only cv-core)
- `cv-io` - File I/O (new!)
- `cv-videoio` - Video capture
- `cv-dnn` - Deep learning (ONNX)
- `cv-scientific` - Scientific computing

### Processing Layer (cv-core + cv-hal)
- `cv-imgproc` - Image processing
- `cv-3d` - 3D geometry processing
- `cv-calib3d` - Camera calibration
- `cv-photo` - Computational photography

### Algorithm Layer (processing + runtime)
- `cv-features` - Feature detection
- `cv-stereo` - Stereo vision
- `cv-sfm` - Structure from Motion
- `cv-slam` - SLAM

### Application Layer (all above)
- `cv-python` - Python bindings
- `cv-viewer` - Visualization GUI
- `rust-cv-native` - Main library

## External Dependencies by Layer

### Core Layer
- `nalgebra` - Linear algebra (essential)
- `image` - Image loading (essential)
- `thiserror` - Error handling
- `rayon` - Threading

### Processing Layer
- `wide`, `pulp` - SIMD
- `wgpu` - GPU compute

### Algorithm Layer
- (Mostly internal dependencies)

### Application Layer
- `pyo3`, `numpy` - Python FFI
- `egui`, `eframe` - GUI

## Design Principles

1. **cv-core is the foundation** - All crates depend on it
2. **cv-hal abstracts hardware** - CPU/GPU backends
3. **cv-runtime manages resources** - Memory pools, task scheduling
4. **Processing crates are independent** - Can be used standalone
5. **Algorithm crates compose** - Features + Stereo = SLAM
6. **Application crates integrate** - Python, GUI, main lib

## Advantages

✅ **Modular** - Use only what you need
✅ **Testable** - Each crate can be tested independently
✅ **Parallel Development** - Teams can work on different crates
✅ **Clear Boundaries** - No spaghetti dependencies
✅ **Optimized Builds** - Only compile what you use

## When to Add a New Crate?

**Create new crate if:**
- New domain (e.g., cv-nlp, cv-robotics)
- Large feature set (>5k lines)
- Different release cycle
- External API (e.g., cv-python)

**Add to existing crate if:**
- Related functionality
- Shares types heavily
- Similar performance characteristics
