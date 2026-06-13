# Sprint 67 Day 6: Graph/Reorder Ownership Extraction Batch 1

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Land the first bounded ownership extraction batch inside the Day 5 fence by
making the two remaining graph/reorder orchestration shells read more like
durable owners and less like mixed helper buckets.

## Landed Batch

Touched implementation surfaces:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`

No proof/header/support widening was required for this first batch.

## `src/sparse_graph.c`: Centralized Uncoarsen Control-Plane Setup

The landed batch introduced:

- `graph_uncoarsen_options_t`
- `graph_uncoarsen_options_from_env(...)`

This centralizes the uncoarsening control-plane setup that previously sat
inline in `graph_uncoarsen(...)`:

- finest/intermediate pass counts
- finest FM strategy
- annealing / thick-restart / gain-noise schedule choices
- ensemble strategy selection and debug flag

That means `graph_uncoarsen(...)` now spends more of its visible body on:

- level-walk orchestration
- per-level runtime setup
- ping-pong buffer movement
- FM / thick-restart / ensemble sequencing

The env/runtime selection still lives locally in the file, so the landing stays
bounded, but it no longer competes as directly with the orchestration shell.

## `src/sparse_reorder_nd.c`: Extracted Three Inline Support Responsibilities

The landed batch introduced:

- `nd_emit_leaf_amd(...)`
- `nd_partition_current_graph(...)`
- `nd_recurse_side(...)`

Those helpers pull three mixed responsibilities out of the middle of
`nd_recurse(...)`:

1. leaf AMD materialization/splice
2. root spectral-versus-multilevel partition dispatch
3. repeated side-subgraph build/map/recurse glue

This leaves `nd_recurse(...)` reading more directly as the durable ND recursive
driver:

- base-case decision
- partition-result ownership
- recursive left/right descent
- separator-last emission

Again, the helpers stay in the same file, so the batch remains inside the Day 5
two-file fence while still landing a real ownership extraction.

## Explicit Non-Widening Result

The first landed batch did not widen into:

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
- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`
- `tests/test_integration.c`

That matters because Sprint 67 still has real later maintainability lanes, and
the Day 6 success condition was a bounded graph/reorder ownership extraction,
not broader churn.

## Validation

Because `*.c` changed, the required validation set was run:

- `make format`
- `make lint`
- `make test`

And because this was substantial decomposition work on orchestration-heavy
files, the stronger reviewed path was also run:

- `make quality-review-full`

All passed.

## Exit State

Sprint 67 Day 6 now hands off one concrete first landing result:

1. `src/sparse_graph.c`
   - uncoarsen env/runtime setup is centralized into one local options seam
2. `src/sparse_reorder_nd.c`
   - leaf handling, partition dispatch, and side recursion glue are extracted
     from the recursive driver
3. the batch stayed inside the exact two-file landing fence
4. reviewed validation passed from the landed state

That gives Day 7 one exact follow-through job:

- rerank the residual graph/reorder ownership seam after the first landed
  extraction and decide whether a second bounded graph/reorder batch is still
  justified
