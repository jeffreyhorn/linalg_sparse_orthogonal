# Sprint 61 Day 11 - Compatibility Layer & Regression Sweep

Date: 2026-06-09
Branch: sprint-61

## Objective

Tighten the landed Phase 1 compatibility story after the Day 6-Day 10 typed
analysis/reorder work:

- re-read the full precedence model
- prove stable default behavior explicitly
- remove stale env-only wording from the remaining internal seams
- close the batch from the full reviewed baseline

## Landed Scope

Touched files:

- `src/sparse_graph_internal.h`
- `src/sparse_graph_coarsen.c`
- `tests/test_reorder_nd.c`

No new public API fields were added on Day 11. This was a compatibility,
regression, and wording sweep over the already-landed Phase 1 typed surface.

## Main Result

### 1. Internal wording now matches the shipped precedence model

The stale “env-var-only” commentary remaining in the ND/coarsening internals
was updated so the code reads the way it now behaves:

1. explicit typed option when provided
2. legacy env-var compatibility override when the typed field is unspecified
3. internal default policy

That wording cleanup landed in:

- `src/sparse_graph_internal.h`
- `src/sparse_graph_coarsen.c`

This matters because the Day 6-Day 10 code already routed those controls
through the resolved analysis/reorder policy bridge; the comments were the part
still lagging behind.

### 2. Stable default behavior is now explicitly proven

The new Day 11 proof additions in `tests/test_reorder_nd.c` are:

- `test_analysis_default_nd_coarsen_floor_ratio_matches_internal_default`
- `test_analysis_nd_coarsening_cv_fallthrough_default_matches_compat_value`

What they prove:

- leaving `nd_coarsen_floor_ratio` unspecified behaves the same as explicitly
  selecting the shipped internal default of `100`
- leaving `SPARSE_ND_COARSENING_CV_FALLTHROUGH` unset behaves the same as the
  shipped compatibility/default value of `0.30`

The result is that the Phase 1 control-plane story is now proven at all three
levels:

- typed precedence over env
- compatibility fallback when typed is unspecified
- stable default behavior when nothing is overridden

### 3. The proof burden stayed bounded

Day 11 did not need to widen into:

- `tests/test_graph.c`
- `tests/test_integration.c`

The selected reorder/ND proof home remained sufficient. That keeps Sprint 61
inside the planned Phase 1 fence rather than turning the compatibility sweep
into a broader graph-test rewrite.

## Post-Landing Compatibility State

The landed Phase 1 ND/reorder model is now explicit and test-backed across the
highest-value controls:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`
- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`

Still compatibility-only or explicitly deferred:

- legacy `SPARSE_ND_SUPERNODAL_POSTORDER` alias
- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`
- all `SPARSE_FM_*`

## Validation

Because `*.c` / `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 348.47 sec`

Focused retained proof points:

- `test_reorder_nd` passed with the new default/compatibility proofs
- the graph/reorder-sensitive reviewed surface stayed clean:
  - `test_graph`
  - `test_graph_fm_buckets`
  - `test_reorder_nd`
  - `test_reorder_amd_qg`

Non-blocking note:

- the reviewed CMake rebuild again emitted ordinary compiler warnings while
  rebuilding `bench_eigs_reuse`, but the path still completed cleanly and
  passed all parity gates

## Close

Sprint 61 Day 11 closes the planned compatibility sweep:

- the Phase 1 precedence story is explicitly proven
- stable defaults are now anchored by regression tests
- the remaining env-var behavior is bounded and intentional
- no stale wording still implies a broader or older control model than the
  shipped implementation
