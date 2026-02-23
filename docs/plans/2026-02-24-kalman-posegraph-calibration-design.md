# Design: Kalman Refactor, Pose Graph Consolidation, Calibration Fixture

**Date:** 2026-02-24
**Branch:** `develop`
**Status:** Approved

---

## Problem Statement

Three architectural issues need resolving:

1. `KalmanFilter` lives in `cv-video` — a misplaced OpenCV convention. Kalman filtering is a general estimation algorithm with no video dependency.
2. Pose graph optimization is implemented in three separate crates (`cv-optimize`, `cv-slam`, `cv-3d`) with duplicate/overlapping logic and one unimplemented optimizer.
3. Two calibration tests in `cv-calib3d` are permanently `#[ignore]`d due to a missing JSON fixture file.

---

## Decision Summary

| Item | Decision |
|------|----------|
| Kalman | Move **both** implementations to `cv-core` (dynamic + const-generic EKF) |
| Pose graph canonical location | `cv-optimize` |
| Pose graph source | Move `3d/src/pose_graph/` (GTSAM-inspired, most complete) into `cv-optimize` |
| Pose graph visualizer | Keep in `cv-3d` |
| `slam/src/pose_graph.rs` | Rewrite to use `cv_optimize::PoseGraph`; keep its 10 tests |
| Calibration fixture | Typical webcam parameters (fx=800, fy=800, 640×480) |

---

## 1. Kalman Filter → `cv-core`

### What moves

**From `video/src/kalman.rs`:**
- `KalmanFilter` — dynamic size (`DMatrix<f64>`/`DVector<f64>`), OpenCV-style API (`predict`, `correct`)
- Fix: replace `.unwrap()` on `pseudo_inverse` with proper error handling

**From `slam/src/kalman.rs`:**
- `KalmanFilterState<N>` — const-generic state vector + covariance
- `KalmanFilter<N, M>` — const-generic linear Kalman filter
- `ExtendedKalmanFilter<N, M>` — EKF for nonlinear systems
- `utils::constant_velocity_2d()`, `utils::constant_acceleration_2d()` — pre-built motion models
- All 3 existing tests

### New location
`core/src/kalman.rs`, exported as `pub mod kalman` from `core/src/lib.rs`

### Backwards compatibility
- `video/src/lib.rs`: add `pub use cv_core::kalman::*`
- `slam/src/lib.rs`: add `pub use cv_core::kalman::*`
- No downstream breakage — re-exports preserve all existing paths

---

## 2. Pose Graph → `cv-optimize`

### What moves

**From `3d/src/pose_graph/mod.rs` → `optimize/src/pose_graph.rs`:**
- `Pose3` — SE(3) with translation + `UnitQuaternion`
- `Factor` trait — error, linearization, jacobians
- `PriorFactor`, `BetweenFactor` — with robust kernels (Huber, Tukey, Cauchy, Geman-McClure)
- `IMUFactor` — visual-inertial preintegration
- `PoseGraphOptimizer` — Gauss-Newton, Levenberg-Marquardt, Dogleg
- `GNCLMOptimizer` — Graduated Non-Convexity for outlier rejection
- `ISAM2` — incremental optimization

**Replaces:** current `optimize/src/pose_graph.rs` (simple 187-line Gauss-Newton)

### Visualizer stays in `cv-3d`
`3d/src/pose_graph/visualizer.rs` stays as `3d/src/pose_graph_viz.rs` — it handles PLY export and rendering, which are 3D concerns. It imports `Pose3` from `cv_optimize`.

### `slam/src/pose_graph.rs` rewrite
- Remove duplicate `PoseGraph` struct and `PoseGraphEdge`
- Import and wrap `cv_optimize::PoseGraph` / `cv_optimize::Pose3`
- Keep all 10 existing tests, updating to the new API
- The previously-unimplemented `optimize()` now calls `cv_optimize::PoseGraphOptimizer`

### `cv-3d` cleanup
- Remove `3d/src/pose_graph/mod.rs`
- Remove `3d/src/pose_graph/visualizer.rs` at old path
- Add `3d/src/pose_graph_viz.rs` (visualizer only, imports from `cv_optimize`)
- Update `3d/src/lib.rs` exports accordingly

---

## 3. Calibration Fixture

### File to create
`calib3d/test_data/calibration/expected_results.json`

### Values (typical webcam, 640×480, 9×6 chessboard, 25mm squares)
```json
{
  "pattern_size": [9, 6],
  "square_size": 25.0,
  "num_images": 13,
  "expected_camera_matrix": {
    "fx": 800.0,
    "fy": 800.0,
    "cx": 320.0,
    "cy": 240.0
  },
  "expected_distortion": {
    "k1": -0.25,
    "k2": 0.08,
    "p1": 0.001,
    "p2": -0.001,
    "k3": 0.0
  },
  "expected_rms_error": 0.5,
  "tolerance_percent": 5.0
}
```

### Test changes
Remove `#[ignore]` from both tests in `calib3d/src/lib.rs`:
- `calibrate_camera_dataset_ground_truth_loading`
- `calibrate_camera_synthetic_validation_against_expected`

---

## Affected Files

### Modified
- `core/src/lib.rs` — add `pub mod kalman`
- `video/src/lib.rs` — add `pub use cv_core::kalman::*`
- `slam/src/lib.rs` — add `pub use cv_core::kalman::*`
- `slam/src/pose_graph.rs` — rewrite to use `cv_optimize`
- `optimize/src/lib.rs` — update pose_graph exports
- `optimize/src/pose_graph.rs` — replace with 3d implementation
- `3d/src/lib.rs` — remove pose_graph module, add pose_graph_viz
- `calib3d/src/lib.rs` — remove `#[ignore]` from 2 tests

### Created
- `core/src/kalman.rs`
- `3d/src/pose_graph_viz.rs`
- `calib3d/test_data/calibration/expected_results.json`

### Deleted
- `video/src/kalman.rs`
- `slam/src/kalman.rs`
- `3d/src/pose_graph/mod.rs`
- `3d/src/pose_graph/visualizer.rs`

---

## Verification

After implementation:
```bash
cargo build --lib --workspace          # zero errors
cargo test --lib --workspace           # 490+ tests, 0 failures, 0 ignored
cargo test -p cv-core -- kalman        # Kalman tests pass
cargo test -p cv-optimize -- pose      # Pose graph tests pass
cargo test -p cv-calib3d               # Both previously-ignored tests now pass
```
