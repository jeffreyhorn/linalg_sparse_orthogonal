# Sprint 61 Day 9: Analysis/Postorder Integration Design

Date: 2026-06-09
Branch: `sprint-61`


## Purpose

Convert the Day 8 residual coarsening-policy queue into one exact Day 10
implementation fence by defining the public/internal field split, the
precedence and compatibility rules, the touched-file set, and the explicit
deferred-control list before more code lands.

## Inputs

- Sprint 61 Day 8 post-landing audit
- live `sparse_analysis_reorder_opts_t` surface
- live coarsening implementation seam
- live `sparse_graph_nd_policy_t` internal bridge
- existing Sprint 61 precedence pattern from Days 6-7

## Exact Control Subset To Move

### Move publicly in Day 10

- `SPARSE_ND_COARSEN_FLOOR_RATIO`

### Move internally in Day 10

- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`

### Do not move in Day 10

- legacy `SPARSE_ND_SUPERNODAL_POSTORDER` alias
- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`
- all `SPARSE_FM_*`

## Public/Internal Plumbing Design

### Public field addition

Add one scalar field to `sparse_analysis_reorder_opts_t`:

- `idx_t nd_coarsen_floor_ratio`

Public semantics:

- `0` means unspecified/default
- positive values request an explicit typed override
- negative values are invalid

Recommended public comment:

- "Optional ND coarsening floor ratio divisor. Use 0 to leave unspecified."

### Internal policy additions

Extend `sparse_graph_nd_policy_t` with:

- `idx_t nd_coarsen_floor_ratio`
- `double nd_coarsening_cv_fallthrough`

Internal defaults:

- `nd_coarsen_floor_ratio = 100`
- `nd_coarsening_cv_fallthrough = 0.30`

Internal semantic ranges:

- `nd_coarsen_floor_ratio`: `1..100000`
- `nd_coarsening_cv_fallthrough`: `0.0..100.0`

`0.0` on the internal CV threshold continues to mean:

- disable the HCC fallthrough threshold check

## Precedence and Compatibility Contract

### Public floor-ratio control

1. explicit typed `nd_coarsen_floor_ratio` when > 0
2. legacy compatibility override from `SPARSE_ND_COARSEN_FLOOR_RATIO` when the
   typed field is `0`
3. internal typed default = `100`

### Internal HCC CV fallthrough control

1. internal resolved-policy value
2. compatibility override from `SPARSE_ND_COARSENING_CV_FALLTHROUGH` only when
   the internal field remains unset by the caller path
3. internal typed default = `0.30`

### Preserved compatibility rule

Public `sparse_reorder_nd(...)` remains a compatibility wrapper:

- env-var behavior stays live there
- Day 10 does not widen the reorder API itself

### Explicit non-rule

Do not add a public typed field for the HCC CV threshold in Day 10.

## Day 10 Touched-File Fence

### Required touched files

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `src/sparse_graph_internal.h`
- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `tests/test_reorder_nd.c`

### Optional only if proof burden forces it

- `tests/test_graph.c`

### Explicit non-touch set

- `src/sparse_graph_bisect.c`
- `src/sparse_graph_separator.c`
- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_reorder_amd_qg.c`
- `README.md`
- `docs/tutorial.md`
- `docs/maintainer_guide.md`

## Regression Obligations

Required Day 10 proof additions:

- typed-over-env precedence proof for `nd_coarsen_floor_ratio`
- stable-default proof when the typed field remains unspecified
- bounded proof that the `sparse_analyze(...)` path now resolves the internal
  HCC fallthrough threshold through the same resolved-policy seam

Preferred proof home:

- `tests/test_reorder_nd.c`

Only widen `tests/test_graph.c` if a direct partition-shape proof is required
and cannot be expressed cleanly through the analysis path.

## Deferred-Control List

### Stay compatibility-only for now

- legacy `SPARSE_ND_SUPERNODAL_POSTORDER` alias

### Explicitly defer

- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`
- all `SPARSE_FM_*`
- any widening of `sparse_reorder_nd(...)`
- any repo-wide configuration helper layer

## Exit State

After Day 9, the next code batch is now exact rather than generic:

- one public coarsening-threshold field
- one internal HCC fallthrough policy field
- explicit precedence and compatibility behavior
- a fixed touched-file fence
- bounded proof obligations
- a named deferred queue
