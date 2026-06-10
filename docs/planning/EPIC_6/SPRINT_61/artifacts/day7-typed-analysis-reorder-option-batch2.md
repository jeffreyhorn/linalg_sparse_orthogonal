# Sprint 61 Day 7: Typed Analysis/Reorder Option Batch 2

Date: 2026-06-09
Branch: `sprint-61`


## Purpose

Complete the bounded Sprint 61 Phase 1 configuration-modernization batch by
adding the remaining selected caller-facing typed analysis/reorder controls,
translating them through the resolved internal ND policy seam, preserving
legacy env-var compatibility for unspecified fields, and proving typed-over-env
precedence on the live coarsening, coarsest-bisection, separator-lift, and
separator-lift-weight paths.

## Scope

### Touched files

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `src/sparse_graph_internal.h`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_separator.c`
- `src/sparse_reorder_nd.c`
- `tests/test_reorder_nd.c`

### Selected controls in this batch

- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`

### Explicit non-goals

- public FM tuning controls
- debug/profile option migration
- `include/sparse_reorder.h` widening
- generic repo-wide configuration helpers
- backend/AUTO policy work
- packaging/platform work

## Landed Public Surface

`include/sparse_analysis.h` now completes the selected Sprint 61 Phase 1
analysis/reorder widening on `sparse_analysis_reorder_opts_t`:

- `nd_coarsening`
- `nd_coarsest_bisection`
- `nd_sep_lift_strategy`
- `nd_sep_lift_weight`

The new public typed enums are:

- `sparse_analysis_nd_coarsening_t`
  - `SPARSE_ANALYSIS_ND_COARSENING_DEFAULT`
  - `SPARSE_ANALYSIS_ND_COARSENING_HEAVY_EDGE`
  - `SPARSE_ANALYSIS_ND_COARSENING_HCC`
- `sparse_analysis_nd_coarsest_bisection_t`
  - `SPARSE_ANALYSIS_ND_COARSEST_BISECTION_DEFAULT`
  - `SPARSE_ANALYSIS_ND_COARSEST_BISECTION_DEFAULT_ROUTING`
  - `SPARSE_ANALYSIS_ND_COARSEST_BISECTION_SPECTRAL`
  - `SPARSE_ANALYSIS_ND_COARSEST_BISECTION_GGGP`
  - `SPARSE_ANALYSIS_ND_COARSEST_BISECTION_BRUTE`
- `sparse_analysis_nd_sep_lift_strategy_t`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_STRATEGY_DEFAULT`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_STRATEGY_SMALLER_WEIGHT`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_STRATEGY_BALANCED_BOUNDARY`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_STRATEGY_PER_VERTEX`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_STRATEGY_PER_VERTEX_BALANCE`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_STRATEGY_PER_VERTEX_DEGREE`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_STRATEGY_PER_VERTEX_FIXED_K`
- `sparse_analysis_nd_sep_lift_weight_t`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_WEIGHT_DEFAULT`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_WEIGHT_HYBRID`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_WEIGHT_BALANCE`
  - `SPARSE_ANALYSIS_ND_SEP_LIFT_WEIGHT_DEGREE`

The widened API remains zero-init safe:

- `DEFAULT` / `0` leaves the field unspecified
- unspecified fields continue to resolve through compatibility overrides and
  then internal defaults

## Internal Policy Bridge

`src/sparse_graph_internal.h` now extends the resolved internal ND policy seam
used by `sparse_analyze(...)` and the internal ND consumer path:

- `sparse_graph_nd_policy_t`
  - `nd_coarsening`
  - `nd_coarsest_bisection`
  - `nd_sep_lift_strategy`
  - `nd_sep_lift_weight`

The new internal typed policy enums are:

- `sparse_graph_nd_coarsest_bisection_mode_t`
- `sparse_graph_nd_sep_lift_strategy_mode_t`
- `sparse_graph_nd_sep_lift_weight_mode_t`

The graph/ND override bridge is now explicit through:

- `sparse_graph_coarsening_override_begin(...)`
- `sparse_graph_coarsening_override_end(...)`
- `sparse_graph_coarsest_bisection_override_begin(...)`
- `sparse_graph_coarsest_bisection_override_end(...)`
- `sparse_graph_sep_lift_override_begin(...)`
- `sparse_graph_sep_lift_override_end(...)`

`src/sparse_analysis.c` now resolves the selected Day 7 controls through the
same explicit precedence chain used on Day 6:

1. explicit typed option
2. legacy compatibility override, only when the typed field is left
   unspecified/default
3. internal typed policy default

Current internal defaults remain:

- ND coarsening: `HCC`
- coarsest bisection: `DEFAULT_ROUTING`
- separator lift strategy: `SMALLER_WEIGHT`
- separator lift weight: `HYBRID`

Validation in the resolver is explicit:

- invalid public enum values return `SPARSE_ERR_BADARG`

## ND Consumer Integration

`src/sparse_reorder_nd.c` now threads the extended resolved
`sparse_graph_nd_policy_t` through the actual graph/ND consumer path by
bracketing the recursion with internal typed-policy overrides for:

- coarsening strategy
- coarsest-bisection strategy
- separator-lift strategy
- separator-lift weight

This keeps both lanes intact:

- public compatibility wrapper:
  - `sparse_reorder_nd(...)`
- internal policy-aware entry:
  - `sparse_reorder_nd_with_policy(...)`

Important bounded behavior was preserved:

- the internal forced heavy-edge recovery path still wins when the separator
  degenerates and the implementation needs that fallback
- legacy env-var compatibility still applies when typed fields remain
  unspecified
- `sparse_reorder_nd(...)` itself stays public-API stable

## Proof Surface

`tests/test_reorder_nd.c` now carries four new typed-over-env precedence tests
for the remaining selected Sprint 61 Phase 1 controls:

- typed `HEAVY_EDGE` overriding env `SPARSE_ND_COARSENING=hcc`
- typed `DEFAULT_ROUTING` overriding env
  `SPARSE_ND_COARSEST_BISECTION=spectral`
- typed `PER_VERTEX` overriding env
  `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k`
- typed `HYBRID` overriding env
  `SPARSE_ND_SEP_LIFT_WEIGHT=balance`

These tests are behavior-level proofs:

- they drive `sparse_analyze(...)`
- they compare actual fill-sensitive symbolic output
- they only skip when the chosen matrix/data path cannot distinguish the
  compared strategies meaningfully

Representative reviewed-path proof points:

- `bcsstk14`: HEM `nnz(L) = 129576`, HCC+Kuu-safe `nnz(L) = 130422`
- `bcsstk04` fixed-K weight path:
  - `hybrid = 3679`
  - `balance = 4469`
  - `degree = 4613`

## Validation

Required gate after a clean rebuild:

- `make clean`
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 320.27 sec`

Representative retained proof points:

- `test_reorder_nd` passed with the new Day 7 typed-over-env precedence tests
- the reviewed path rebuilt and passed the graph/reorder proof surface:
  - `test_graph`
  - `test_graph_fm_buckets`
  - `test_reorder_nd`
  - `test_reorder_amd_qg`

One non-blocking validation note is explicit:

- the reviewed CMake rebuild emitted ordinary compiler warnings while
  rebuilding `bench_eigs_reuse`, but the full reviewed path still completed
  cleanly and passed all parity gates

## Exit State

Sprint 61 Day 7 now completes the selected Phase 1 typed analysis/reorder
landing:

- the remaining selected public typed controls are shipped
- the full selected ND policy bridge is shipped
- the graph/ND consumers now honor typed policy first without a public
  reorder-API redesign
- legacy env-var behavior remains intact for unspecified typed fields
- the deeper typed-over-env precedence proof is shipped
- the remaining Epic 6 configuration queue is now narrower than the original
  env-var backlog
