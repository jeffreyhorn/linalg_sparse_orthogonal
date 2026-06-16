# Sprint 73 Day 5: Typed/Internal Policy Design

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Define the bounded implementation contract for the first Sprint 73
configuration-modernization landing before code edits begin.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day4-first-modernization-boundary.md`
- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`

## Day 5 Design Conclusions

### 1. The first Sprint 73 batch is internal-policy-first, not public-option-first

The first graph/FM modernization batch should not add a new public typed FM
option surface.

It should instead converge the graph/FM lane behind one clearer internal typed
policy owner, because the strongest current pain is:

- split process-global parsing across `src/sparse_graph.c` and
  `src/sparse_graph_refine.c`
- not the absence of a broad new public FM option model

### 2. The ownership split is now explicit

Public typed options in the first batch:

- none required

Internal typed policy owner in the first batch:

- finest FM strategy
- ensemble member list
- finest-level pass count
- intermediate-level pass count
- annealing schedule choice
- thick-restart perturbation choice
- gain-noise schedule choice
- debug flags only as internal/runtime fields if they still need to exist

Compatibility-only env overrides in the first batch:

- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_ANNEALING_SCHEDULE`
- `SPARSE_FM_THICK_RESTART_PERTURB`
- `SPARSE_FM_GAIN_NOISE_SCHEDULE`

Narrowed developer-only or debug-only behavior:

- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_THICK_RESTART_DEBUG`
- `SPARSE_FM_ANNEALING_DEBUG`
- `SPARSE_FM_GAIN_NOISE_DEBUG`

### 3. The first-batch precedence rules are fixed

The first batch must preserve:

- existing default behavior when no compatibility env var is set
- the current effective behavior of recognized compatibility env values
- current safe-default fallback on malformed or unrecognized compatibility env
  inputs
- debug-only flags remaining outside any broadened public typed contract

The implementation goal is therefore:

- parse compatibility envs once at the orchestration boundary
- lower them into one internal FM policy/runtime contract
- stop treating the refinement subsystem as a second independent public parser

### 4. The first-batch touch and non-touch sets are now explicit

Required first implementation center:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`

Support only if the implementation truly forces it:

- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_integration.c`

Explicit non-touch set:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`
- `tests/test_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_svd.c`
- `include/sparse_analysis.h`
- `docs/maintainer_guide.md`
- broader README/tutorial/example/benchmark surfaces
- capability/type surfaces
- packaging/platform/workflow files

## Exit State

Sprint 73 Day 5 closes with:

1. one explicit internal-policy-first design for the FM lane
2. one preserved compatibility-precedence checklist
3. one exact first-batch touch set
4. one explicit non-touch set before Day 6 implementation begins
