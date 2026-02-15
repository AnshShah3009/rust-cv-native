## Marker Detection GPU Plan

### Goal
Design a wgpu compute shader pipeline that replaces the inner loop of `features::markers::detect_*` so the binary grid sampling, bitmask generation, and confidence scoring run on the GPU while keeping the existing CPU fallback for robustness and comparison.

### Requirements
- Input: candidate bounding rectangles (min/max x/y) already computed by the CPU candidate finder.
- Output: per-candidate bitmask (up to 8×8 bits), rotation index, and confidence score.
- Shader must support configurable grid sizes (`payload_bits` + border), optional normalization of coordinates, and output aligned to the detection buffers.
- CPU fallback must still work with the same API so detection callers can toggle GPU vs CPU via a runtime flag.

### Buffer Layout
1. **Candidate buffer (read-only):** `u32` entries `[min_x, min_y, max_x, max_y, grid_size, payload_bits]`.
2. **Image texture (read-only):** `r8unorm` texture containing the gray image. Shared between shader runs.
3. **Result buffer (write):** struct per-candidate with `u64 bitmask`, `u32 rotation`, `f32 confidence`, `u32 status`.

### Shader Operation
1. Each workgroup handles one candidate.
2. Sample the grid at evenly spaced coordinates with bilinear filtering (or manual weighting) to produce `grid_size²` intensity values.
3. Decide bit = 1 when intensity < threshold (threshold can be uniform or derived from border).
4. Evaluate border bits to ensure strong black border across ring cells.
5. Encode payload bits and compare against dictionary table (pre-uploaded as uniform buffer) to find best id/rotation.
6. Emit confidence (e.g., ratio of matching bits) and status (valid, too noisy, no match).

### Integration Steps
1. Add `features::markers::gpu::ShaderContext` that compiles the shader at runtime (or includes precompiled WGSL/SPIRV).
2. Provide a helper `run_candidate_scan_gpu(image, candidates)` that returns decoded bitmasks and confidence values.
3. Update `detect_aruco_markers`, `detect_apriltags`, and `detect_charuco_corners` to query `features::markers::gpu::run_candidate_scan` before falling back to the CPU bit sampler.
4. Expose `features::markers::use_gpu(bool)` or runtime detection of wgpu availability (via existing `StereoError` or new helper).

### Testing
- Add unit test that uses a synthetic marker image, runs the GPU pipeline (when wgpu is available), and ensures the decoded IDs match the CPU path.
- Keep CPU tests unchanged so the shader can be verified via output parity and fallback.

### Documentation
- Document the new shader path in `docs/marker_gpu_plan.md` (this file).
- Expand `PROJECT_STATUS.md` note about GPU work once the shader is implemented.

### Handoff Notes
- Candidate generation remains CPU-only; the shader just accelerates the grid sampling + bit decoding.
- Keep `features/src/markers.rs` CPU helpers intact to allow the shader results to be validated/tested against the existing logic.
