# Sprint 73 Day 6: FM/Graph Policy Integration Batch 1

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Land the first bounded Sprint 73 implementation batch by converging the
graph/FM compatibility env surface behind one internal policy owner, without
widening the public option surface.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day5-typed-internal-policy-design.md`
- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_internal.h`

## Landed Batch

### 1. One internal FM policy object now owns the compatibility control surface

The Day 6 landing introduces `sparse_graph_fm_policy_t` in
`src/sparse_graph_internal.h` and makes `src/sparse_graph.c` the one place
that resolves the FM compatibility env surface into that internal policy.

That policy now owns:

- finest FM strategy
- ensemble strategy list and count
- finest-level pass count
- intermediate-level pass count
- annealing schedule choice
- thick-restart perturbation choice
- gain-noise schedule choice
- retained debug/runtime flags

Compatibility envs still supported through that boundary:

- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_ANNEALING_SCHEDULE`
- `SPARSE_FM_THICK_RESTART_PERTURB`
- `SPARSE_FM_GAIN_NOISE_SCHEDULE`
- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_THICK_RESTART_DEBUG`
- `SPARSE_FM_ANNEALING_DEBUG`
- `SPARSE_FM_GAIN_NOISE_DEBUG`

### 2. The refinement subsystem now consumes lowered runtime state

Before the Day 6 batch, `src/sparse_graph_refine.c` still carried direct FM
parser helpers and direct `getenv(...)`-driven debug decisions for parts of
the annealing / thick-restart / gain-noise lane.

After the landing:

- schedule / perturbation parser helpers now live only at the orchestration
  boundary in `src/sparse_graph.c`
- the graph layer lowers policy to runtime with
  `graph_uncoarsen_runtime_for_level(...)`
- `sparse_graph_fm_runtime_set(...)` transfers the resolved state into the
  refinement subsystem
- `graph_refine_fm(...)` now consumes runtime flags instead of independently
  consulting process-global env state for the retained annealing/gain-noise
  debug paths

This is the real Day 6 ownership win:

- refinement is no longer acting like a second public parser

### 3. The batch stayed bounded

Touched files:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_internal.h`

Untouched deferred surfaces:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`
- `tests/test_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_svd.c`
- `include/sparse_analysis.h`
- `docs/maintainer_guide.md`

No extra proof-surface widening was needed:

- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_integration.c`

## One Validation-Driven Correction

The first cut stored the finest-FM strategy in the new policy object as `int`.

That was accepted by the ordinary compile, but the strict `make lint` path
failed under `-Werror -Wsign-conversion`.

The landed correction was to:

- move `finest_fm_strategy_t` into `src/sparse_graph_internal.h`
- use that enum type directly in `sparse_graph_fm_policy_t`

That keeps the Day 6 ownership seam typed instead of muting the warning with a
cast.

## Validation

Because `*.c` and `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 337.63 sec`

## Exit State

Sprint 73 Day 6 closes with:

1. one internal FM policy ownership center in the graph/orchestration layer
2. one narrower runtime handoff into refinement
3. preserved FM compatibility behavior without a new public FM option model
4. a fully validated first Sprint 73 implementation landing
