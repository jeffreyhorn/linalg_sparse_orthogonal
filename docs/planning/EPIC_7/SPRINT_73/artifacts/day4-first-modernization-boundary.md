# Sprint 73 Day 4: First Modernization Boundary

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Refine the Day 3 residual-control ranking and freeze the first bounded Sprint
73 modernization fence before implementation design begins.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day3-residual-control-inventory-audit.md`
- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`
- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_reorder_nd.c`
- `tests/test_integration.c`
- `include/sparse_analysis.h`
- `docs/maintainer_guide.md`

## Day 4 Boundary Conclusions

### 1. The strongest first Sprint 73 fence is graph/FM policy convergence

The Day 4 rerank confirms the best first bounded lane is:

- graph/FM strategy and pass-count policy convergence

This lane has the strongest combination of:

- public process-global surprise
- implementation ownership blur
- bounded cleanup payoff
- acceptable first-pass compatibility risk

The graph/FM lane still exposes the densest residual process-global story
across `src/sparse_graph.c` and `src/sparse_graph_refine.c`, so it is the
right first target before broader ND or debug/profile work.

### 2. The ND compatibility/default-policy seam is support context for later
work, not the first landing

The ND lane remains the strongest second contradiction center.

But the rerank shows it should be treated as:

- the strongest second batch
- not the first landing

because:

- the typed-precedence story there is already stronger than in the FM lane
- the residual cost is denser in compatibility follow-through than in the
  broad public process-global story
- it widens quickly into more policy surfaces than the graph/FM first batch

### 3. The first-batch landing surfaces are now explicit

Required first landing:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`

Likely support only if the first landing forces it:

- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_integration.c`
- `include/sparse_analysis.h`
- `docs/maintainer_guide.md`

Deferred or explicitly later:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`
- `tests/test_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_svd.c`
- broader public/doc surfaces
- capability/type work
- packaging/platform/workflow surfaces

### 4. The first batch is a control-surface convergence lane, not a generic
graph-family cleanup

The first batch should clarify only:

- FM strategy ownership
- FM pass-count ownership
- FM schedule and perturbation ownership
- the line between typed/internal policy and compatibility-only process-global
  overrides

It should not widen into:

- general graph algorithm redesign
- ND/coarsening default-policy redesign
- broad debug/profile cleanup everywhere at once
- proof-owner churn beyond the immediate FM lane

## Exit State

Sprint 73 Day 4 closes with:

1. one explicit first modernization boundary around graph/FM policy
   convergence
2. one fixed support-only map for proof and maintained-surface follow-through
3. one explicit deferred map for ND, debug/profile, and SVD-routing cleanup
4. one clear starting point for Day 5 implementation design
