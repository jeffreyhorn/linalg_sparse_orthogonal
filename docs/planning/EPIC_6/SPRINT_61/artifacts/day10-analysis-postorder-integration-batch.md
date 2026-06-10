# Sprint 61 Day 10: Analysis/Postorder Integration Batch

Date: 2026-06-09
Branch: `sprint-61`


## Purpose

Land the remaining bounded analysis-time control batch by moving
`SPARSE_ND_COARSEN_FLOOR_RATIO` onto the typed
`sparse_analysis_reorder_opts_t` surface, resolving
`SPARSE_ND_COARSENING_CV_FALLTHROUGH` through the internal ND policy seam, and
proving the preserved precedence/compatibility rules on the live ND analysis
path.

## Inputs

- Sprint 61 Day 8 post-landing analysis audit
- Sprint 61 Day 9 analysis/postorder integration design
- live `sparse_analysis` public option surface
- live `sparse_graph_nd_policy_t` internal bridge
- live ND/coarsening implementation seams

## Landed Batch

### Public typed-option widening

Added one new scalar field to `sparse_analysis_reorder_opts_t`:

- `idx_t nd_coarsen_floor_ratio`

Public contract:

- `0` remains unspecified/default
- positive values request an explicit typed override
- negative values are rejected
- out-of-range positive values are rejected

### Internal ND policy completion

Extended `sparse_graph_nd_policy_t` with:

- `idx_t nd_coarsen_floor_ratio`
- `double nd_coarsening_cv_fallthrough`

Internal defaults:

- `nd_coarsen_floor_ratio = 100`
- `nd_coarsening_cv_fallthrough = 0.30`

### Resolution/precedence behavior

Public floor-ratio precedence:

1. explicit typed `nd_coarsen_floor_ratio`
2. legacy compatibility override from `SPARSE_ND_COARSEN_FLOOR_RATIO` when the
   typed field remains unspecified
3. internal default `100`

Internal HCC CV fallthrough precedence:

1. resolved internal policy value
2. legacy compatibility override from `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
   when no explicit internal override is active
3. internal default `0.30`

### Internal override plumbing

The coarsening implementation now has dedicated begin/end override plumbing for:

- coarsening floor-ratio divisor
- HCC CV fallthrough threshold

The policy-aware ND path now brackets `nd_recurse(...)` with:

- coarsening strategy override
- coarsening floor-ratio override
- HCC CV fallthrough override
- coarsest-bisection override
- separator-lift override

## Touched Files

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `src/sparse_graph_internal.h`
- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `tests/test_reorder_nd.c`

## Proof Additions

Added bounded Day 10 proofs in `tests/test_reorder_nd.c`:

- `test_analysis_typed_nd_coarsen_floor_ratio_overrides_env`
- `test_analysis_nd_coarsening_cv_fallthrough_env_affects_policy_path`

These sit alongside the existing Day 6-7 precedence proofs for:

- root bisection
- root bisection max-N
- coarsening strategy
- coarsest bisection
- separator-lift strategy
- separator-lift weight
- supernodal postorder

## Validation

Because `*.c` / `*.h` changed, I ran:

- `make clean`
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 332.33 sec`

Focused retained proof points:

- `test_reorder_nd` passed with the two new Day 10 proofs
- the full graph/reorder-sensitive reviewed surface stayed clean:
  - `test_graph`
  - `test_graph_fm_buckets`
  - `test_reorder_nd`
  - `test_reorder_amd_qg`

## Preserved Non-Goals

The batch did not widen into:

- public FM tuning controls
- debug/profile option migration
- repo-wide configuration helper layers
- public `sparse_reorder_nd(...)` API changes
- backend/AUTO policy work
- packaging/platform work

## Exit State

After Day 10:

- the last justified public analysis-time coarsening scalar is typed
- the residual HCC fallthrough threshold is part of the internal resolved ND
  policy seam for the explicit analysis lifecycle
- the ND driver carries both residual controls through one coherent override
  path
- the Day 8-Day 9 residual queue is reduced to explicit deferred FM/debug
  compatibility work rather than more Phase 1 analysis-surface debt
