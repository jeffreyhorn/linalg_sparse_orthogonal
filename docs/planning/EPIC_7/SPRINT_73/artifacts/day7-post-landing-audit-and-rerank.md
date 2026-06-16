# Sprint 73 Day 7: Post-Landing Audit and Rerank

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Reassess the residual configuration queue after the Day 6 FM/graph policy
landing and choose the strongest exact Day 8 target from the live
post-landing state.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day6-fm-graph-policy-integration-batch1.md`
- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_svd.c`

## Day 7 Rerank

### 1. The Day 6 landing closed the strongest first-lane contradiction

The FM/graph lane no longer carries the same ownership blur it had on Day 5.

What changed:

- `src/sparse_graph.c` now owns compatibility parsing for the FM strategy/pass
  and schedule lane
- `src/sparse_graph_refine.c` now consumes lowered runtime state instead of
  acting like a second parser

That means the strongest Day 5 contradiction is now gone:

- the next batch should not be another generic FM follow-through pass by
  default

### 2. The strongest remaining contradiction has shifted to debug/profile controls

The best next Sprint 73 lane is now the developer-only/profile surface across:

- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`

Why this lane now ranks first:

- `src/sparse_graph_coarsen.c` still mixes real routing/default behavior with
  developer-only `SPARSE_HCC_DEBUG` and residual compatibility override reads
- `src/sparse_reorder_nd.c` still activates `SPARSE_ND_PROFILE` directly from
  process-global state
- `src/sparse_reorder_amd_qg.c` still does the same for
  `SPARSE_QG_PROFILE`

So the strongest post-Day-6 contradiction is no longer "two FM parsers"; it is
now "multiple graph/reorder families still carry ad hoc process-global
instrumentation and developer-only control reads."

### 3. The weaker lanes are now explicit

FM/graph follow-through is no longer the strongest next center:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`

Later or narrower residual lanes remain valid, but rank lower:

- `src/sparse_svd.c`
- `src/sparse_analysis.c`
- `src/sparse_graph_separator.c`

Reason:

- `SPARSE_SVD_LOWRANK_OUTER` is still only one narrow advisory-routing seam
- the separator-lift compatibility surface is real but less cross-family than
  the profile/debug lane
- `src/sparse_analysis.c` remains an authority surface, but it is not the
  strongest immediate contradiction after the Day 6 landing

## Exact Day 8 Target Fence

Required design center:

- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`

Support only if the design truly forces it:

- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`
- `tests/test_integration.c`
- `docs/maintainer_guide.md`

Explicit deferred set:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_analysis.c`
- `include/sparse_analysis.h`
- `src/sparse_svd.c`
- `src/sparse_graph_separator.c`
- public README/tutorial/example/benchmark surfaces
- capability/type/platform/workflow files

## Exit State

Sprint 73 Day 7 closes with:

1. the Day 6 FM lane reranked downward to support/deferred context
2. the developer-only/profile lane promoted to the strongest remaining seam
3. one exact Day 8 target fence around coarsening + ND/profile ownership
