# Sprint 73 Day 8: Debug/Profile Rationalization Design

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Define the bounded second Sprint 73 implementation batch around the strongest
remaining developer-only/profile spill after the Day 7 rerank.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day7-post-landing-audit-and-rerank.md`
- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_graph_internal.h`

## Day 8 Design Conclusions

### 1. The second batch center is graph/coarsen + ND profile ownership

The strongest Day 9 implementation center is now:

- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`

Likely support only if the batch needs a shared internal runtime seam:

- `src/sparse_graph_internal.h`
- `src/sparse_reorder_amd_qg.c`

The key narrowing from Day 7 is:

- `src/sparse_reorder_amd_qg.c` remains part of the same developer-only
  profile story, but it now reads as support-only follow-through instead of
  the main second-batch center

### 2. The exact second-batch goal is now fixed

The Day 9 batch should preserve the real maintained compatibility controls:

- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- `SPARSE_ND_COARSEN_FLOOR_RATIO`

It should narrow the strongest developer-only/profile spill:

- `SPARSE_HCC_DEBUG` in `src/sparse_graph_coarsen.c`
- `SPARSE_ND_PROFILE` in `src/sparse_reorder_nd.c`

Likely support-only follow-through:

- `SPARSE_QG_PROFILE` in `src/sparse_reorder_amd_qg.c`

The implementation direction should be:

- keep maintained routing/default policy controls where they are truly part of
  the shipped contract
- move debug/profile activation into a clearer internal runtime or
  entry-boundary ownership model
- avoid making developer instrumentation look like a peer public option family

### 3. The preserved compatibility checklist is explicit

The second batch must preserve:

- default routing behavior when the coarsening compatibility env surface is
  unset
- current recognized behavior for:
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
- current opt-in behavior for developer-only instrumentation when enabled
- the narrow interpretation of profile/debug surfaces as diagnostics or
  benchmark aids rather than production-facing policy promises

The second batch must not:

- reopen the Day 6 FM policy convergence batch
- create a new public typed debug/profile option surface
- widen into SVD-routing, separator-lift, or broader public docs/header work

### 4. Support and non-touch sets are now fixed

Support only if the Day 9 implementation truly forces it:

- `src/sparse_graph_internal.h`
- `src/sparse_reorder_amd_qg.c`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`
- `tests/test_integration.c`
- `docs/maintainer_guide.md`

Explicit non-touch set:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_analysis.c`
- `include/sparse_analysis.h`
- `src/sparse_svd.c`
- `src/sparse_graph_separator.c`
- public README/tutorial/example/benchmark surfaces
- capability/type/platform/workflow files

## Exit State

Sprint 73 Day 8 closes with:

1. one exact second-batch center around graph/coarsen and ND profile
2. one likely support-only follow-through path for QG profile
3. one preserved compatibility checklist for the maintained coarsening controls
4. one explicit non-touch set before Day 9 implementation
