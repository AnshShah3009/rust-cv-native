# rust-cv-native

A comprehensive native Rust computer vision library with full in-house implementations (no external C/C++ dependencies), GPU acceleration using wgpu/WebGPU, and Python bindings with PyO3.

## Features

- **Core** - Basic types, image containers, camera models, frame conventions, tensors
- **Imgproc** - Image processing (filters, morphology, color conversion)
- **Features** - Feature detection and descriptors (ORB, Harris, FAST, BRIEF, HOG, SIFT)
- **Stereo** - Stereo vision, disparity matching, depth estimation
- **Calib3d** - Camera calibration, pose estimation, chessboard detection
- **3D** - Point clouds, ICP registration, triangulation, mesh reconstruction (Poisson, BPA)
- **SFM** - Structure from Motion, triangulation, bundle adjustment
- **SLAM** - ISAM2 incremental optimization, keyframe management
- **Registration** - ICP, global registration, SE(3) transforms
- **Rendering** - Gaussian splatting, mesh processing
- **Plot** - 2D/3D visualization (cv-plot)
- **Video** - MOG2 background subtraction, Kalman filtering, optical flow, tracking
- **Videoio** - Video capture (FFmpeg-next), platform-specific backends
- **Optimize** - Factor graphs, sparse solvers, ISAM2

## Architecture

```
rust-cv-native/
├── core/          # Core types, camera models, frame conventions
├── hal/           # Hardware abstraction layer (CPU/GPU)
├── imgproc/       # Image processing
├── features/      # Feature detection and matching
├── stereo/        # Stereo vision
├── calib3d/       # Camera calibration
├── registration/  # ICP, global registration
├── 3d/            # Point clouds, triangulation, mesh reconstruction
├── sfm/           # Structure from Motion
├── slam/          # SLAM with ISAM2
├── optimize/      # Optimization (ISAM2, sparse solvers)
├── rendering/     # Gaussian splatting
├── plot/          # Visualization
├── video/         # Video processing (MOG2, optical flow)
├── videoio/       # Video I/O (FFmpeg backend)
├── python/        # Python bindings (PyO3)
└── examples/      # Usage examples
```

## Installation

### Rust

```bash
cargo build --workspace
```

### Python

```bash
cd python
pip install maturin
maturin develop
```

## Testing

**291+ tests** across all crates including:
- cv-core: geometry, robust estimation, tensor operations
- cv-features: Harris, FAST, BRIEF, GFTT, HOG
- cv-sfm: triangulation
- cv-3d: mesh reconstruction (Poisson, BPA, Alpha Shapes)
- cv-optimize: ISAM2
- cv-registration: SE(3) transforms

```bash
cargo test --workspace
```

## Requirements

- Rust 1.70+
- Python 3.10+ (for Python bindings)

## License

MIT
