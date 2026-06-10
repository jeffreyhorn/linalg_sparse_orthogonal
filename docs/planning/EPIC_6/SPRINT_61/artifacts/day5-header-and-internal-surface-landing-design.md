# Sprint 61 Day 5: Header and Internal-Surface Landing Design

Date: 2026-06-09
Branch: `sprint-61`


## Purpose

Turn the Day 4 typed-options contract into an exact touched-file and
API/implementation boundary plan so the Day 6-7 code landing stays bounded and
does not expand into a reorder-API rewrite or a broad graph-subsystem churn
batch.

## Minimum Viable Public API Addition

### Public surfaces to touch

- `include/sparse_analysis.h`

### Public surfaces to keep untouched

- `include/sparse_reorder.h`
- one-shot direct-family option headers
- graph-internal headers staying non-public

### Public design rule

The minimum viable public addition is:

- a bounded widening of `sparse_analysis_opts_t`
- one nested advanced/reorder sub-struct
- zero-init-safe enums and numeric fields

This keeps the public repeated-run direct lifecycle as the configuration front
door without inventing a second public configuration object.

## Internal Bridge Design

### Core bridge decision

The smallest viable implementation bridge is:

- keep `sparse_reorder_nd(...)` as the public compatibility wrapper
- add one internal policy-aware ND entry point used by `sparse_analyze(...)`

Why this matters:

- `sparse_analyze(...)` needs to honor typed controls
- the current public `sparse_reorder_nd(A, perm)` signature cannot receive them
- a public reorder-API redesign is too large for Sprint 61 Phase 1

### Internal ownership lanes

- resolve public typed options plus compatibility overrides in:
  - `src/sparse_analysis.c`
- declare or hold the internal policy structs in:
  - `src/sparse_graph_internal.h`
- declare the internal ND entry point in:
  - `src/sparse_reorder_nd_internal.h`

## Day 6 vs Day 7 Split

### Day 6: Bridge and first selected controls

Touch:

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `src/sparse_graph_internal.h`
- `src/sparse_reorder_nd_internal.h`
- `src/sparse_reorder_nd.c`

Selected controls:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`

Reason:

- this proves the full public-to-internal precedence path on the smallest
  high-value slice
- it stays closest to `sparse_analyze(...)` and the ND recursion seam

### Day 7: Remaining selected analysis/reorder controls

Touch:

- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_separator.c`
- `src/sparse_reorder_nd.c`
- `tests/test_reorder_nd.c`
- `tests/test_graph.c`
- optional bounded `tests/test_integration.c`

Remaining selected controls:

- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`

Reason:

- these are still Phase 1 public candidates, but they belong deeper in the
  graph/reorder consumer set than the Day 6 bridge slice

## Exact Touched-File Plan

### Expected Day 6-7 touched files

- public:
  - `include/sparse_analysis.h`
- internal headers:
  - `src/sparse_graph_internal.h`
  - `src/sparse_reorder_nd_internal.h`
- implementation:
  - `src/sparse_analysis.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_separator.c`
- likely proof follow-through:
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - optional bounded `tests/test_integration.c`
- likely docs follow-through after code lands:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/maintainer_guide.md`

### Expected non-touch set for the first landing

- `include/sparse_reorder.h`
- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_svd.c`

These files should stay untouched unless the code landing proves they are
strictly necessary for compilation or contract correctness.

## Operational Non-Goals

Do not widen the first code landing into:

- public FM tuning controls
- debug/profile option migration
- compile-time threshold policy cleanup
- one-shot direct-family option widening
- generic repo-wide configuration helpers
- packaging/platform work
- broad docs simplification outside touched-surface truthfulness updates

## Day 5 Exit State

Sprint 61 now has a precise implementation boundary:

- the minimum viable public addition is fixed
- the internal policy bridge is fixed
- the Day 6 versus Day 7 split is fixed
- the touched-file and non-touch sets are fixed
- the first code landing can proceed without reopening surface-area scope
