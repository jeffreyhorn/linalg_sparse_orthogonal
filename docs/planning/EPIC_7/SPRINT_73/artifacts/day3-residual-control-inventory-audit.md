# Sprint 73 Day 3: Residual Control Inventory Audit

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Re-rank the remaining configuration surfaces by live ownership cost instead of
by historical familiarity, so Sprint 73 can work from one concrete
contradiction map rather than one generic env-var-cleanup story.

## Main Result

Sprint 73’s broad configuration-modernization problem is now reduced to one
ranked live contradiction map:

- strongest first target:
  - graph/FM strategy and pass-count policy
- strongest second target:
  - ND compatibility/default-policy overrides
- strongest third target:
  - developer-only debug/profile surfaces
- strongest later target:
  - residual SVD-routing and advanced compatibility controls

## Ranked Contradiction Map

### 1. Graph/FM strategy and pass-count policy

Strongest first target across:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`

Why it ranks first:

- it carries the densest residual process-global public control surface
- orchestration-level parsing and refinement-level parsing are still split
  across two files
- the FM lane still over-relies on env parsing for advanced strategy choice,
  pass counts, and debug behavior
- `tests/test_graph.c` is already the strongest permanent proof owner for this
  lane

Main controls:

- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_THICK_RESTART_DEBUG`
- `SPARSE_FM_ANNEALING_SCHEDULE`
- `SPARSE_FM_THICK_RESTART_PERTURB`
- `SPARSE_FM_GAIN_NOISE_SCHEDULE`
- `SPARSE_FM_ANNEALING_DEBUG`
- `SPARSE_FM_GAIN_NOISE_DEBUG`

### 2. ND compatibility/default-policy overrides

Strongest second target across:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`

Why it ranks second:

- the typed-precedence story is better than it used to be
- `sparse_reorder_nd_default_policy()` already centralizes more default policy
  ownership than earlier sprints
- but the lane still carries a dense compatibility parser bundle and residual
  legacy override story
- `tests/test_reorder_nd.c` already owns the strongest proof cost for this
  lane

Main controls:

- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`
- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_SUPERNODAL_POSTORDER`

### 3. Developer-only debug/profile surfaces

Strongest third target across:

- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_coarsen.c`

Why it ranks third:

- it still leaks operational or developer-only controls into permanent code
  paths
- but it carries lower public correctness/behavior cost than the graph/FM and
  ND policy lanes
- it reads more like a second-batch rationalization target than the best first
  ownership-convergence center

Main controls:

- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`
- FM debug flags

### 4. Residual SVD-routing and advanced compatibility controls

Later target:

- `src/sparse_svd.c`

Why it ranks later:

- narrower surface
- lower public confusion cost
- more isolated proof and ownership story than graph/reorder policy

Main control:

- `SPARSE_SVD_LOWRANK_OUTER`

## Interpretation

The live tree says Sprint 73 should not start by touching every residual
`getenv(...)` call.

It should start where:

- the process-global public story is still densest
- the control parsing is still split across multiple ownership centers
- the strongest proof owners already exist

That fixes the Day 4 rerank direction:

- graph/FM policy convergence is the best first landing
- ND compatibility/default-policy convergence is the best second landing
- debug/profile rationalization is better as the second batch than the first
- SVD-routing cleanup stays later unless a stronger queue collapse occurs

## Exit State

Sprint 73 Day 3 closes with:

1. one ranked residual-control contradiction map
2. one strongest first target fixed to graph/FM policy convergence
3. one strongest second target fixed to ND compatibility/default-policy
   overrides
4. one bounded later queue for debug/profile and SVD-routing cleanup
