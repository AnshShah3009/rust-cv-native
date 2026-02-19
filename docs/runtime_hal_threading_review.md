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

1. Empty core affinity list can panic worker thread startup [RESOLVED in Phase 14]
- File: `runtime/src/orchestrator.rs:33`
- `cores.get(i % cores.len())` will divide by zero when `cores = Some(vec![])`.
- Status: Fixed by wrapping core affinity logic in `if let Some(cores) = core_ids` and ensuring `cores.len() > 0`.

2. Public GPU APIs can panic at runtime due to `todo!()` [PARTIALLY RESOLVED]
- File: `hal/src/gpu_kernels/mod.rs:252`
- Many public entry points are `todo!()`.
- Status: Transitioning to `Result<T, Error>` and returning `NotSupported`.

3. GPU readback helper returns silent wrong results [RESOLVED in Phase 13]
- File: `hal/src/gpu_kernels/mod.rs:203`
- `read_buffer` now uses an async polling loop and returns `Result`.

4. Sparse matrix CSR conversion lacks bounds validation [RESOLVED in Phase 14]
- File: `hal/src/gpu_sparse.rs:87`
- Status: Added validation for row/col bounds and integer overflow checks.

## Medium-Priority Findings

1. Lock poisoning causes panic via `unwrap()` [RESOLVED in Phase 14]
- Files: `runtime/src/orchestrator.rs`, `core/src/runtime.rs`
- Status: Replaced `unwrap()` with poison recovery (`into_inner()`).

2. `submit_to!` silently drops work if group is missing [RESOLVED in Phase 14]
- File: `runtime/src/lib.rs:24`
- Status: Replaced with `submit()` which returns `Result<()>`.

3. Resource group replacement semantics are implicit [RESOLVED in Phase 14]
- File: `runtime/src/orchestrator.rs:75`
- Status: `create_group` now returns an error on duplicate names.

4. `GpuContext::new` loses useful diagnostics [RESOLVED in Phase 14]
- File: `hal/src/gpu.rs:15`
- Status: Returns `Result<Self, Error>` with descriptive error messages.

5. `block_on` in GPU init can block async runtimes [RESOLVED in Phase 14]
- File: `hal/src/gpu.rs:23`
- Status: Provided `new_async` and `new_with_policy` methods.

6. Runtime/core thread-pool lifecycle can conflict [RESOLVED in Phase 14]
- Status: ResourceGroups now use isolated `rayon::ThreadPool` instances.

## Low-Priority Findings / Design Gaps

1. `TaskPriority` exists but is unused
- Status: Removed or moved to experimental.

2. `UnifiedBuffer` device side is placeholder-only
- Status: Frozen in Phase 13/14 as a manual sync state machine.

3. Buffer pool policy is simplistic and can miss reuse [RESOLVED in Phase 14]
- Status: Implemented size-bucketed pool with `swap_remove`.

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
