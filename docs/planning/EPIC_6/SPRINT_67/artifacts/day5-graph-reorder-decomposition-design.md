# Sprint 67 Day 5: Graph/Reorder Decomposition Design

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Turn the Day 4 first-landing boundary into one explicit ownership and
extraction contract so the first Sprint 67 implementation batch stays bounded
to the remaining graph/reorder orchestration shells.

## First-Landing Ownership Contract

The first landing remains fixed to:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`

But Day 5 now makes their intended durable ownership more explicit.

`src/sparse_graph.c` should converge toward:

- graph partition top-level orchestration
- coarsest-level seed selection and retry ownership
- uncoarsening orchestration

`src/sparse_reorder_nd.c` should converge toward:

- ND policy normalization at the public boundary
- ND recursion and top-level orchestration
- ND profiling publication at the public boundary

So the first landing is not a broad graph-family cleanup. It is a bounded
ownership extraction inside the two remaining orchestration shells.

## `src/sparse_graph.c`: Keep Orchestration, Question Support Helpers

The live `src/sparse_graph.c` still owns both durable orchestration and weaker
support/helper surfaces.

Durable orchestration that belongs there:

- `graph_uncoarsen(...)`
- `graph_hierarchy_coarsest(...)`
- `graph_partition_seed_coarsest(...)`
- `graph_partition_should_retry_with_forced_hem(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`

Weaker long-term owners if the Day 6 landing needs extraction:

- `graph_parse_env_int_range(...)`
- `graph_parse_finest_strategy(...)`
- `graph_parse_ensemble_strategy_list(...)`
- `graph_env_flag_enabled(...)`
- `graph_uncoarsen_level_passes(...)`
- `graph_uncoarsen_runtime_for_level(...)`

Design consequence:

- keep graph partition and uncoarsening orchestration in `src/sparse_graph.c`
- move support parsing/runtime-accounting helpers only where that materially
  clarifies the orchestration shell

## `src/sparse_reorder_nd.c`: Keep ND Orchestration, Question Compatibility Parsers

The live `src/sparse_reorder_nd.c` still mixes durable ND owners with policy
compatibility parsing and support helpers.

Durable ND owners:

- `nd_recurse(...)`
- `sparse_reorder_nd_with_policy(...)`
- `sparse_reorder_nd(...)`

Likely separable support helpers:

- `nd_emit_natural(...)`
- `nd_subgraph_to_sparse(...)`

Compatibility/policy parsing and local profiling helpers:

- `parse_nd_root_bisect_strategy_compat_override(...)`
- `parse_nd_coarsening_compat_override(...)`
- `parse_nd_coarsest_bisection_compat_override(...)`
- `parse_nd_root_bisect_max_n_compat_override(...)`
- `parse_nd_coarsen_floor_ratio_compat_override(...)`
- `parse_nd_coarsening_cv_fallthrough_compat_override(...)`
- `parse_nd_sep_lift_strategy_compat_override(...)`
- `parse_nd_sep_lift_weight_compat_override(...)`
- `sparse_reorder_nd_default_policy(...)`

Design consequence:

- keep ND public-boundary normalization and recursive orchestration in
  `src/sparse_reorder_nd.c`
- move compatibility parsers, leaf/base-case helpers, or profiling helpers
  only where that materially reduces mixed ownership and stale chronology

## Exact Day 6-7 Touched-File Fence

Required first-batch implementation surfaces:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`

Likely proof home:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

Support only if the landed extraction truly needs it:

- `src/sparse_graph_internal.h`
- `tests/test_integration.c`

This keeps the first implementation batch family-local by default and leaves
header/integration widening explicitly conditional.

## Explicit Non-Touch Set

The first graph/reorder landing should not widen into:

- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- public coordination headers unless the landed extraction truly forces it
- packaging/platform/build churn

That non-touch set matters because Sprint 67 still has real later lanes after
the graph/reorder landing, and the first success condition is cleaner ownership
inside the two remaining orchestration shells rather than broader source churn.

## Exit State

Sprint 67 Day 5 closes with one exact first implementation contract:

1. required first batch:
   - `src/sparse_graph.c`
   - `src/sparse_reorder_nd.c`
2. likely proof home:
   - `tests/test_graph.c`
   - `tests/test_reorder_nd.c`
3. support only if needed:
   - `src/sparse_graph_internal.h`
   - `tests/test_integration.c`
4. explicit non-touch set:
   - already-extracted graph subsystem files
   - `src/sparse_reorder_amd_qg.c`
   - CSC/analysis residuals
   - iterative/eigensolver residuals

That gives Day 6 one exact job:

- land one bounded graph/reorder ownership extraction batch without widening
  into graph-family redesign
