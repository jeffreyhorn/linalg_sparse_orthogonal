# Sprint 61 Day 8: Post-Landing Analysis/Postorder Audit

Date: 2026-06-09
Branch: `sprint-61`


## Purpose

Re-audit the remaining analysis-time and postorder-adjacent env-var controls
after the Day 6-7 typed-option landing, separate what still justifies Sprint
61 movement from what should stay compatibility-only or explicitly defer, and
fix the exact bounded Day 9 landing target before more code moves.

## Scope

### Inputs re-read

- Sprint 61 Day 6 artifact
- Sprint 61 Day 7 artifact
- Sprint 61 working notes
- Sprint 61 plan Day 8-Day 10 section
- live implementation seams in:
  - `src/sparse_analysis.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_separator.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_amd_qg.c`
  - `src/sparse_graph_internal.h`

### Audit question

After the Day 6-7 landing completed the selected public analysis/reorder typed
surface, which remaining analysis-time and postorder-adjacent env-var controls
still justify Sprint 61 movement, and which should remain compatibility-only
or defer?

## Findings

### 1. The strongest residual Sprint 61 queue is now concentrated in the coarsening-policy seam

The Day 6-7 landing already moved the major public analysis/reorder controls:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`

The strongest residual analysis-time env vars are now:

- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`

Both sit inside the same coarsening-policy seam:

- hierarchy stopping threshold in `sparse_graph_hierarchy_build(...)`
- HCC-to-HEM fallthrough threshold in `graph_coarsen_with_strategy(...)`

Conclusion:

- the next justified Sprint 61 move is not another broad ND sweep
- it is a bounded coarsening-policy follow-through slice

### 2. The supernodal-postorder legacy alias is now compatibility-only

The live postorder env behavior is:

- canonical:
  - `SPARSE_SUPERNODAL_POSTORDER`
- compatibility alias:
  - `SPARSE_ND_SUPERNODAL_POSTORDER`

This alias still matters for back-compat, but it does not justify new public
typed work:

- the canonical control is already typed on `sparse_analysis_opts_t`
- the alias is already subordinate to the canonical name
- widening or removing it now would create churn without user-facing value

Conclusion:

- no further postorder-specific integration batch is justified in Sprint 61
- the alias should remain compatibility-only for now

### 3. The debug/profile env vars are real but explicitly not Sprint 61 Phase 1 product controls

Still-live instrumentation/debug env vars include:

- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`

These remain internal tooling/support surfaces:

- they emit profiling/debug output
- they are implementation-coupled
- they do not represent stable caller intent in the same way the landed
  reorder/ND controls do

Conclusion:

- they should not move into the public typed option surface in Sprint 61
- they should remain explicitly deferred

### 4. The FM family remains adjacent but still not the strongest Sprint 61 slice

`SPARSE_FM_*` remains live in the graph/refinement implementation, but it is
still broader and riskier than the residual coarsening-policy seam:

- more tightly coupled to refinement internals
- more strategy-heavy
- more debug/advisory oriented
- more likely to reopen the Sprint 61 non-goal fence

Conclusion:

- the FM family stays explicitly deferred after Day 8

### 5. The strongest Day 9 target is now one mixed public/internal coarsening-policy batch

The best next slice is:

- public typed candidate:
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
- internal typed-policy candidate:
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`

This gives a narrow and coherent Day 9-10 target:

- public widening only where the caller-facing story is strong enough
- internal typed policy where the knob is still too implementation-shaped
- no widening into debug/profile or FM strategy controls

Likely touched surfaces for the next batch:

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `src/sparse_graph_internal.h`
- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `tests/test_reorder_nd.c`
- optional bounded `tests/test_graph.c` only if the proof burden requires it

### 6. The main post-Day-7 risk is now default/compatibility drift in a small number of coarsening-policy seams

The Day 6-7 design intentionally preserved:

- public typed resolution in `src/sparse_analysis.c`
- env-compat behavior for direct graph entry surfaces inside graph modules

That keeps the compatibility story honest, but it concentrates the residual
risk:

- if remaining defaults or compat parsers drift, the drift will now happen in
  a small number of coarsening-policy seams instead of across the full ND
  pipeline

Conclusion:

- the best response is a narrow Day 9-10 batch that closes the strongest
  coarsening-policy gap cleanly
- broadening into multiple unrelated residual env vars would raise risk rather
  than reduce it

## Ranked Residual Queue

### Move in Sprint 61

1. `SPARSE_ND_COARSEN_FLOOR_RATIO`
2. `SPARSE_ND_COARSENING_CV_FALLTHROUGH`

### Stay compatibility-only for now

1. legacy `SPARSE_ND_SUPERNODAL_POSTORDER` alias

### Explicitly defer

1. `SPARSE_ND_PROFILE`
2. `SPARSE_QG_PROFILE`
3. `SPARSE_HCC_DEBUG`
4. `SPARSE_FM_*`

## Day 9 Landing Target

The exact bounded Day 9 target should be:

- a mixed public/internal coarsening-policy batch
- public typed candidate:
  - `nd_coarsen_floor_ratio`
- internal typed-policy candidate:
  - HCC CV fallthrough threshold
- preserved precedence rule:
  1. explicit typed option
  2. legacy compatibility override when the typed field remains unspecified
  3. internal typed policy default
- explicit non-goals:
  - no FM-family widening
  - no debug/profile migration
  - no new repo-wide configuration helper layer
  - no packaging/platform spillover

## Exit State

After Day 8, the Sprint 61 queue is smaller and more concrete than the Day 3
inventory implied:

- the strongest residual analysis-time work is a single coarsening-policy seam
- the postorder residual is compatibility-only
- the debug/profile and FM families are explicitly deferred
- the Day 9 target is precise enough to design without reopening Sprint 61
  scope
