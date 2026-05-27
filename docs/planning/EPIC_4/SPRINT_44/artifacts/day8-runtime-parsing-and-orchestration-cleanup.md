# Sprint 44 Day 8 Artifact: Runtime Parsing and Orchestration Cleanup

## Purpose

Land one bounded cleanup pass in the residual `src/sparse_graph.c` after the
FM and separator extractions so the remaining file reads like orchestration-
owned runtime composition rather than another hidden subsystem, without
changing retry/fallback or FM strategy behavior.

## 1. What Changed

Day 8 kept the residual graph queue in place and simplified its local runtime
setup.

The main changes were:

- stale residual section wording was corrected
- repeated env-var parsing blocks were consolidated into small local helpers
- per-level FM runtime selection was moved behind clearer helper seams

New local residual helpers:

- `graph_parse_env_int_range(...)`
- `graph_parse_finest_strategy(...)`
- `graph_parse_ensemble_strategy_list(...)`
- `graph_env_flag_enabled(...)`
- `graph_uncoarsen_level_passes(...)`
- `graph_uncoarsen_runtime_for_level(...)`

## 2. What This Cleanup Now Owns

The residual file still owns:

- `graph_uncoarsen(...)`
- `graph_hierarchy_coarsest(...)`
- `graph_partition_seed_coarsest(...)`
- `graph_partition_count_separator_vertices(...)`
- `graph_partition_should_retry_with_forced_hem(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`

The important difference is that `graph_uncoarsen(...)` now expresses its
runtime/config setup more directly:

- parse bounded integer env vars once
- parse finest-level strategy once
- parse ensemble selector lists once
- select per-level pass counts through one helper
- derive FM runtime settings through one helper

## 3. What Stayed Intentionally Local

Day 8 did **not** try to peel residual orchestration into another module.

The following stayed local by design:

- runtime snapshot / restore choreography
- ensemble winner-selection flow
- thick-restart anchor replay flow
- top-level partition entry and retry composition

This matches the Day 7 audit:

- the residual file no longer had another clean extraction seam
- the right move was readability/ownership cleanup in place

## 4. What Did Not Change

The cleanup was structural only.

It did **not** change:

- finest/intermediate FM strategy semantics
- ensemble/thick-restart enablement rules
- forced-HEM retry semantics
- public APIs
- shared graph internal header shape
- the extracted ownership model from:
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`

## 5. Residual Ownership Wording

The stale section banner:

- `Uncoarsening + vertex-separator extraction`

was replaced with:

- `Uncoarsening + residual orchestration runtime`

That matters because separator extraction is already complete in
`src/sparse_graph_separator.c`, so the residual file should describe the live
ownership model rather than preserve pre-extraction wording.

## 6. Validation

Because `src/sparse_graph.c` changed, the full required validation gate was
run:

- `make format`
- `make lint`
- `make test`

All passed.

## Bottom Line

Day 8 closed the residual graph cleanup seam the right way:

- no new module
- no behavior change
- clearer orchestration-owned runtime setup
- cleaner residual ownership wording
- preserved retry/fallback structure
