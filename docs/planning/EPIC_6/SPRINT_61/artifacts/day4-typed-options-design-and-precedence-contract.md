# Sprint 61 Day 4: Typed Options Design and Precedence Contract

Date: 2026-06-09
Branch: `sprint-61`


## Purpose

Define the Phase 1 typed configuration model and the exact precedence rules
before Sprint 61 starts code changes, so the first implementation batch lands
against a real public/internal ownership contract instead of a generic cleanup
goal.

## Design Decision

### 1. Public Phase 1 options belong on `sparse_analysis_opts_t`

The strongest Phase 1 controls all sit on the direct-analysis / reorder path,
so the public surface should be a bounded extension of `sparse_analysis_opts_t`
rather than a brand-new top-level configuration object.

Recommended direction:

- keep `sparse_analysis_opts_t` as the public entry point
- add one nested advanced/reorder sub-struct
- keep every new field zero-init safe

### 2. Public Phase 1 controls stay narrow and caller-meaningful

Recommended public typed-option set:

- one reusable tri-state switch enum:
  - `SPARSE_OPTION_DEFAULT = 0`
  - `SPARSE_OPTION_OFF = 1`
  - `SPARSE_OPTION_ON = 2`
- bounded analysis/reorder enums for:
  - ND coarsening strategy
  - ND coarsest bisection strategy
  - ND separator-lift strategy
  - ND separator-lift weight
- nested public analysis/reorder fields for:
  - `supernodal_postorder`
  - `nd_coarsening`
  - `nd_coarsest_bisection`
  - `nd_root_bisect`
  - `nd_root_bisect_max_n`
  - `nd_sep_lift_strategy`
  - `nd_sep_lift_weight`

Representation rule:

- enum `DEFAULT = 0` means “unspecified; resolve normally”
- integer threshold `0` means “unspecified; resolve normally”

This preserves the zero-init contract while making the strongest analysis-time
controls explicit.

## Internal Ownership Split

### 1. Lower-level ND/FM tuning becomes internal typed policy first

These controls should move away from raw process-global parsing, but not become
public API in Phase 1:

- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_ANNEALING_SCHEDULE`
- `SPARSE_FM_THICK_RESTART_PERTURB`
- `SPARSE_FM_GAIN_NOISE_SCHEDULE`

Recommended direction:

- one internal resolved-policy struct
- resolved once near the analysis/reorder front door
- passed down as typed values instead of repeatedly parsing env vars in deep
  implementation files

### 2. Debug/profile controls stay internal

The following remain out of the Phase 1 public design:

- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`
- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_ANNEALING_DEBUG`
- `SPARSE_FM_THICK_RESTART_DEBUG`
- `SPARSE_FM_GAIN_NOISE_DEBUG`

These are instrumentation or maintainer-only surfaces, not stable caller
workflow controls.

## Precedence Contract

The Phase 1 precedence rule is:

1. explicit typed option value
2. legacy compatibility override, but only when the typed field is left
   unspecified/default
3. internal typed policy default

The “default typed option value” rule is intentionally constrained:

- a zero-initialized public options struct does not eagerly stamp concrete
  values into every new field
- `DEFAULT` / `0` means “unspecified”
- if a helper is added later to materialize recommended defaults, it must map
  to the same resolved values as the unspecified path when no compatibility
  override is present

This keeps the precedence model simple:

- explicit typed values always win
- env vars remain bounded backward-compatibility inputs
- internal policy remains the final fallback source of truth

## Compatibility Contract

### 1. What remains compatible in Phase 1

- canonical env-var names keep working when the new typed field is unspecified
- the existing legacy alias remains accepted where it already exists:
  - `SPARSE_ND_SUPERNODAL_POSTORDER` as a compatibility alias for
    `SPARSE_SUPERNODAL_POSTORDER`

### 2. What does not widen

- no new legacy aliases
- no public promotion of debug/profile controls
- no Phase 1 widening of adjacent seams like `SPARSE_SVD_LOWRANK_OUTER`

### 3. Translation ownership

Compatibility parsing should be centralized behind one or a few translation
helpers instead of staying distributed across deep implementation files.

## Landing Fence for the First Code Batch

### Public and front-door surfaces

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`

### Main graph/reorder consumer surfaces

- `src/sparse_graph.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_reorder_nd.c`

### Later proof follow-through

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`
- `tests/test_integration.c`

## Explicit Non-Goals

- no public FM tuning explosion
- no generic repo-wide configuration object
- no migration of debug/profile-only controls into the public API
- no packaging/platform widening
- no backend/AUTO-policy rewrite in the same batch

## Day 4 Exit State

Sprint 61 now has a concrete configuration-surface contract:

- the public Phase 1 surface is bounded
- the internal policy lane is bounded
- the precedence order is explicit
- the compatibility story is explicit
- Day 5 can now define the exact touched-file plan against this contract
