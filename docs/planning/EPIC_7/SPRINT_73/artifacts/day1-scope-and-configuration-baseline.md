# Sprint 73 Day 1: Scope and Configuration Baseline

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Turn the Sprint 73 project-plan scope plus the Sprint 70 and Sprint 72
handoff into one bounded configuration-modernization sprint, with the
strongest live control surfaces and non-goal fence fixed before deeper audit
begins.

## Main Result

Sprint 73 now starts from a precise residual-control queue, not from another
planning reset and not from another product-model or public-surface cleanup
wave.

The strongest next Epic 7 queue is explicitly:

- residual env-var inventory and rerank
- typed versus internal-policy ownership design
- FM/graph policy integration
- debug/profile and residual advanced-control rationalization
- proof and docs follow-through only where the landed configuration work truly
  moves the maintained contract

## Preserved Fence

The Sprint 70 architecture and truthfulness fence remains explicit:

- no generic env-var purge detached from real ownership cost
- no fake product widening through undocumented internal controls
- no platform/install/reviewed-contract widening
- no broad backend or product-model rewrite hidden inside configuration work

## Live Baseline Anchors

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

## Strongest Likely Sprint 73 Touch Surfaces

Raw Day 1 `wc -l` counts from the live tree:

### Maintained public/policy surfaces

- `docs/maintainer_guide.md` = `585`
- `include/sparse_analysis.h` = `499`

### Configuration-modernization implementation seams

- `src/sparse_graph.c` = `821`
- `src/sparse_graph_internal.h` = `850`
- `src/sparse_reorder_nd.c` = `739`
- `src/sparse_graph_refine.c` = `629`
- `src/sparse_graph_coarsen.c` = `641`
- `src/sparse_graph_separator.c` = `297`
- `src/sparse_graph_bisect.c` = `528`
- `src/sparse_reorder_amd_qg.c` = `611`
- `src/sparse_svd.c` = `1319`

### Strongest proof and reporting surfaces

- `tests/test_graph.c` = `2900`
- `tests/test_reorder_nd.c` = `2262`
- `tests/test_integration.c` = `2448`
- `tests/test_fuzz.c` = `651`
- `examples/example_analysis.c` = `210`
- `benchmarks/bench_reorder.c` = `321`

## Residual Control Families

The live residual control map is concentrated in these families:

### Graph/FM strategy and pass-count controls

Centered on `src/sparse_graph.c` and `src/sparse_graph_refine.c`:

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

### ND/coarsening and reorder/profile controls

Centered on `src/sparse_reorder_nd.c`, `src/sparse_graph_coarsen.c`,
`src/sparse_graph_separator.c`, `src/sparse_graph_bisect.c`, and
`src/sparse_reorder_amd_qg.c`:

- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`
- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`

### Residual compatibility and advanced-routing controls

Centered on `src/sparse_analysis.c` and `src/sparse_svd.c`:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_SUPERNODAL_POSTORDER`
- `SPARSE_SVD_LOWRANK_OUTER`

## Interpretation

The live tree says Sprint 73 should start here:

- graph/FM and ND policy convergence is the strongest first modernization lane
- debug/profile and residual compatibility spill is the strongest second lane
- broader capability, platform, and product-model work should remain out of
  scope
- proof cost remains concentrated in the permanent graph/reorder and
  integration test owners rather than in new generic proof surfaces

## Exit State

Sprint 73 Day 1 closes with:

1. one configuration-modernization starting queue
2. one explicit non-goal fence
3. one live reviewed baseline anchor
4. one ranked live residual-control hotspot map
