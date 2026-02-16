# cv-runtime + HAL Threading/Low-Level Scout Review

## Scope

This review is a static scout only (no compile/run), focused on:

- `runtime/` (`cv-runtime`)
- `hal/` (`cv-hal`)
- `core/src/runtime.rs` (global thread pool + buffer pool)
- Selected related threading code:
  - `3d/src/async_ops/mod.rs`
  - `stereo/src/parallel.rs`
  - integration call sites in `python/src/lib.rs` and `3d/src/point_cloud.rs`

## High-Priority Findings

1. Empty core affinity list can panic worker thread startup
- File: `runtime/src/orchestrator.rs:33`
- `cores.get(i % cores.len())` will divide by zero when `cores = Some(vec![])`.
- Impact: runtime panic during pool worker startup.
- Recommendation:
  - Reject empty core lists in `create_group`/`ResourceGroup::new` with a clear error.
  - Keep `None` as "no affinity".

2. Public GPU APIs can panic at runtime due to `todo!()`
- File: `hal/src/gpu_kernels/mod.rs:252`
- Many public entry points are `todo!()` (point cloud, TSDF, ICP, spatial, mesh, raycasting, odometry, unified GPU dispatch).
- Impact: hard panic if any caller reaches these functions.
- Recommendation:
  - Replace `todo!()` in public APIs with typed errors (`Result<T, cv_hal::Error>`).
  - Gate unfinished modules behind feature flags or make experimental APIs explicitly internal.

3. GPU readback helper returns silent wrong results
- File: `hal/src/gpu_kernels/mod.rs:203`
- `read_buffer` currently returns `vec![]` placeholder.
- Impact: callers may treat empty output as valid data.
- Recommendation:
  - Return `Result<Vec<T>>` and an explicit `NotSupported`/`NotImplemented` error until implemented.

4. Sparse matrix CSR conversion lacks bounds validation
- File: `hal/src/gpu_sparse.rs:87`
- `triplets_to_csr` does not validate `row < rows` / `col < cols`.
- Impact:
  - Out-of-bounds indexing panic for invalid `row`.
  - silent truncation risk on `usize -> u32` casts.
- Recommendation:
  - Validate all triplets and return `Result<Self>`.
  - Check `rows`, `cols`, `nnz` against `u32::MAX` before casting.

## Medium-Priority Findings

1. Lock poisoning causes panic via `unwrap()`
- Files:
  - `runtime/src/orchestrator.rs:77`
  - `runtime/src/memory.rs:29`
  - `core/src/runtime.rs:69`
- Impact: one panic in a lock holder can cascade into future panics.
- Recommendation:
  - Replace raw `unwrap()` on locks with error mapping or poison recovery strategy.
  - At minimum, convert to crate error types.

2. `submit_to!` silently drops work if group is missing
- File: `runtime/src/lib.rs:24`
- If group does not exist, macro does nothing.
- Impact: difficult-to-debug task loss.
- Recommendation:
  - Return `Result<()>` variant of submit API/macros.
  - Optionally log warning in debug builds.

3. Resource group replacement semantics are implicit
- File: `runtime/src/orchestrator.rs:75`
- `create_group` with same name overwrites existing entry without warning.
- Impact: surprising behavior and possible accidental pool replacement.
- Recommendation:
  - Add `create_group_if_absent` or explicit `replace_group`.
  - Return an error on duplicate by default.

4. `GpuContext::new` loses useful diagnostics
- File: `hal/src/gpu.rs:15`
- Returns `Option<Self>`, discarding adapter/device creation errors.
- Impact: poor operability and debugging.
- Recommendation:
  - Return `Result<Self, cv_hal::Error>` with source error message.

5. `block_on` in GPU init can block async runtimes
- File: `hal/src/gpu.rs:23`
- `futures::executor::block_on` is used directly.
- Impact: potential performance/latency issues if called on async executor threads.
- Recommendation:
  - Provide an async constructor (`async fn new_async`) and keep sync wrapper for non-async callers.

6. Runtime/core thread-pool lifecycle can conflict
- Files:
  - `core/src/runtime.rs:13`
  - `runtime/src/orchestrator.rs:92`
- `cv-core` uses Rayon global pool; `cv-runtime` creates dedicated pools. Without policy, users can oversubscribe CPU by mixing both.
- Recommendation:
  - Document intended ownership model:
    - either global pool first-class, or resource groups first-class.
  - Add guidance in README/API docs for mixed usage.

## Low-Priority Findings / Design Gaps

1. `TaskPriority` exists but is unused
- File: `runtime/src/orchestrator.rs:9`
- No scheduling policy currently consumes priority.
- Suggestion: remove for now or introduce a priority queue/executor semantics.

2. `UnifiedBuffer` device side is placeholder-only
- File: `runtime/src/memory.rs:14`
- `device_data` is type-erased and never concretely synchronized.
- Suggestion: either narrow this to explicit typed backend buffers or mark API experimental.

3. Buffer pool policy is simplistic and can miss reuse
- File: `core/src/runtime.rs:68`
- Pool pops one buffer and discards it if too small.
- Suggestion: size-bucketed pool or scan for sufficient capacity before allocating.

4. `3d` async wrappers may panic on join failures
- File: `3d/src/async_ops/mod.rs:74`
- Extensive `.await.unwrap()` after `spawn_blocking`.
- Suggestion: return `Result` and propagate `JoinError`.

5. `3d` async config appears mostly unused
- File: `3d/src/async_ops/mod.rs:13`
- `AsyncConfig` is defined but not wired into the execution path.
- Suggestion: either integrate into API behavior (concurrency limits/pool choice) or remove.

6. Dead conditional in async transform wrapper
- File: `3d/src/async_ops/mod.rs:66`
- `std::mem::size_of::<Vec<Point3<f32>>>() > 0` is always true.
- Suggestion: replace with real size/work-threshold heuristic or remove branch.

## Suggested Improvement Plan

1. Safety pass (first)
- Validate `cores` non-empty when provided.
- Validate sparse triplet bounds and integer conversion limits.
- Replace panic stubs (`todo!`) in public HAL APIs with `Result` errors.

2. Reliability pass
- Remove `unwrap()` from lock and task-join paths.
- Change silent-failure submission to explicit result-returning API.
- Upgrade GPU context creation from `Option` to `Result`.

3. API clarity pass
- Define threading model across `cv-core` global pool vs `cv-runtime` groups.
- Clarify experimental vs stable HAL GPU surfaces.
- Decide fate of currently unused types (`TaskPriority`, dormant config knobs).

## Overall Assessment

- `cv-runtime` is a useful scaffold for grouped execution and affinity, but currently has panic edges and ambiguous failure behavior.
- `cv-hal` has solid structure but many GPU surfaces are still placeholders; current API shape can expose callers to runtime panics or silent incorrect outputs.
- For production-readiness, the biggest wins are to eliminate panic paths, make failure explicit (`Result`-first), and lock down validation on low-level memory/index conversions.
