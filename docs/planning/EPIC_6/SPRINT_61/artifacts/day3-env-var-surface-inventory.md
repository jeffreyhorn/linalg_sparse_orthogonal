# Sprint 61 Day 3: Env-Var Surface Inventory

Date: 2026-06-09
Branch: `sprint-61`


## Purpose

Reduce the broad Epic 6 “advanced tuning is too env-var driven” claim to a
concrete ranked Phase 1 control map before Sprint 61 moves into typed-option
design.

## Rechecked Surfaces

- public front-door analysis surface:
  - `include/sparse_analysis.h`
- direct analysis/control translation seam:
  - `src/sparse_analysis.c`
- graph/reorder orchestration and runtime-control seams:
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_amd_qg.c`
- adjacent non-Phase-1 surface:
  - `src/sparse_svd.c`
- proof surfaces:
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_reorder_amd_qg.c`
- user-facing reference surfaces:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/maintainer_guide.md`

## Ranked Control Map

### 1. Strongest Phase 1 public typed-option candidates

These controls already behave most like caller-meaningful analysis/reorder
choices:

- `SPARSE_SUPERNODAL_POSTORDER`
- legacy `SPARSE_ND_SUPERNODAL_POSTORDER`
- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`

Why they rank first:

- they sit close to the `sparse_analyze(...)` / `SPARSE_REORDER_ND` front door
- they affect ordering or symbolic-analysis behavior that a caller can
  reasonably care about
- they already show up in tests and in the shipped README as more than
  maintainer-only internals

### 2. Strongest internal typed-policy candidates

These controls are real, but they read more like internal multilevel-search
policy than stable public API knobs:

- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_ANNEALING_SCHEDULE`
- `SPARSE_FM_THICK_RESTART_PERTURB`
- `SPARSE_FM_GAIN_NOISE_SCHEDULE`

Why they rank second:

- they are tightly coupled to FM or coarsening runtime orchestration
- they interact with `_Thread_local` FM state and runtime save/restore helpers
- their meaning is strongest inside the implementation, not at the main
  analysis front door

### 3. Compatibility-only or legacy-translation candidates

The strongest explicit compatibility seam is:

- canonical:
  - `SPARSE_SUPERNODAL_POSTORDER`
- legacy accepted alias:
  - `SPARSE_ND_SUPERNODAL_POSTORDER`

This is the clearest Day 3 example of where Sprint 61 needs a bounded
legacy-override story instead of pretending compatibility can disappear in one
batch.

### 4. Stay-internal diagnostic/debug controls

These should stay out of the Phase 1 public typed-option design:

- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`
- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_ANNEALING_DEBUG`
- `SPARSE_FM_THICK_RESTART_DEBUG`
- `SPARSE_FM_GAIN_NOISE_DEBUG`

They are tied to instrumentation, stderr tracing, or maintainer-oriented debug
flows rather than stable user-facing workflow selection.

### 5. Adjacent but out-of-scope for Sprint 61 Phase 1

- `SPARSE_SVD_LOWRANK_OUTER`

This remains a real env-var seam, but it is not part of the strongest direct
analysis/reorder modernization cut.

## Architectural Consequences

### 1. The strongest Epic 6 configuration gap is now clearly inside the direct-analysis / graph-ordering lane

The live repo does not have a generic “missing typed options everywhere”
problem. It has a concentrated mixed control plane:

- public typed options already exist for the top-level repeated-run direct
  lifecycle
- but key ND/FM and postorder behavior still parses from process-global env vars

### 2. FM tuning is more tightly coupled than the public review summary implied

The FM controls are reinforced by runtime state, not just parsing helpers:

- `_Thread_local` FM state in `src/sparse_graph_refine.c`
- orchestration save/restore helpers:
  - `sparse_graph_fm_runtime_get(...)`
  - `sparse_graph_fm_runtime_set(...)`
- forced-HEM retry plumbing in `src/sparse_graph_coarsen.c`

That means Sprint 61 should not treat FM migration as a simple string-to-enum
rewrite.

### 3. The proof surface already treats several env vars as contract-bearing

The strongest current proof surfaces:

- `tests/test_graph.c` = `2900`
- `tests/test_reorder_nd.c` = `1594`
- `tests/test_reorder_amd_qg.c` = `273`

They already pin behavior under:

- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_SUPERNODAL_POSTORDER`

So Phase 1 migration must plan for proof-surface updates, not just implementation
changes.

## Strongest Sprint 61 Phase 1 Cut Line

- public typed-option candidates:
  - `SPARSE_SUPERNODAL_POSTORDER`
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSEST_BISECTION`
  - `SPARSE_ND_ROOT_BISECT`
  - `SPARSE_ND_ROOT_BISECT_MAX_N`
  - `SPARSE_ND_SEP_LIFT_STRATEGY`
  - `SPARSE_ND_SEP_LIFT_WEIGHT`
- internal typed-policy candidates:
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
  - selected `SPARSE_FM_*` budget/strategy/schedule controls
- explicit defer:
  - debug/profile-only controls
  - `SPARSE_SVD_LOWRANK_OUTER`
  - broader compile-time threshold policy

## Day 3 Exit State

Sprint 61 now has a ranked live control inventory rather than a generic env-var
cleanup goal:

- the strongest public typed-option candidates are explicit
- the strongest internal typed-policy candidates are explicit
- the compatibility-only and debug/profile sets are explicit
- Day 4 can now design typed options and precedence rules against a real Phase 1
  target instead of a vague modernization backlog
