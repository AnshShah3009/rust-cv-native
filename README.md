# rust-cv-native

A comprehensive native Rust computer vision library with full in-house implementations (no external C/C++ dependencies), GPU acceleration using wgpu/WebGPU, and Python bindings with PyO3.

## Features

- **Core** - Basic types, image containers, camera models, frame conventions
- **Imgproc** - Image processing (filters, morphology, color conversion)
- **Features** - Feature detection and descriptors (ORB, Harris, etc.)
- **Stereo** - Stereo vision, disparity matching, depth estimation
- **Calib3d** - Camera calibration, pose estimation, chessboard detection
- **3D** - Point clouds, ICP registration, triangulation
- **Rendering** - Gaussian splatting, mesh processing
- **Plot** - 2D/3D visualization (cv-plot)
- **Video** - MOG2 background subtraction, Kalman filtering, optical flow, tracking
- **Videoio** - Video capture (FFmpeg-next), platform-specific backends

## Architecture

```
rust-cv-native/
├── core/          # Core types, camera models, frame conventions
├── imgproc/       # Image processing
├── features/      # Feature detection and matching
├── stereo/        # Stereo vision
├── calib3d/       # Camera calibration
├── registration/  # ICP, global registration
├── 3d/            # Point clouds, triangulation
├── rendering/     # Gaussian splatting
├── plot/          # Visualization
├── video/         # Video processing (MOG2, optical flow)
├── videoio/       # Video I/O (FFmpeg backend)
├── hal/           # Hardware abstraction layer (CPU/GPU)
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

## Requirements

- Rust 1.70+
- Python 3.10+ (for Python bindings)

## License

MIT
