# Sprint 44 Day 7 Artifact: Runtime Strategy and Orchestration Audit

## Purpose

Audit the residual `src/sparse_graph.c` after the FM and separator extractions
so the remaining runtime/config parsing, uncoarsening composition, and
retry/fallback glue are classified concretely before the Day 8 cleanup batch.

## 1. Residual Graph File Shape

After Day 5 and Day 6, the residual graph file now contains two real seam
classes:

- uncoarsening and finest/intermediate FM pass composition
- top-level partition orchestration and forced-HEM retry glue

The remaining helper set is:

- `graph_uncoarsen(...)`
- `graph_hierarchy_coarsest(...)`
- `graph_partition_seed_coarsest(...)`
- `graph_partition_count_separator_vertices(...)`
- `graph_partition_should_retry_with_forced_hem(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`

This is no longer a generic leftover monolith.

## 2. Remaining Runtime / Config Parsing Ownership

The live residual env-var parsing is now fully concentrated in
`graph_uncoarsen(...)`.

### Direct config blocks still in the file

- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_THICK_RESTART_DEBUG`

### Interpretation

This parsing is no longer FM-owned or separator-owned. Those parser seams
already moved with:

- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`

What remains is orchestration-scoped configuration: it selects how many passes
run, how finest-level strategy dispatch is composed, and how ensemble/thick-
restart orchestration is instrumented.

## 3. What Is Ready for Day 8 Consolidation

The bounded cleanup-ready blocks are:

- finest-level strategy enum + parse block
- ensemble selector-list parsing block
- finest-pass count parsing block
- intermediate-pass count parsing block
- ensemble debug flag read
- thick-restart debug flag read

These are good Day 8 targets because they are:

- local to the residual file
- repetitive enough to simplify
- clearly orchestration-scoped

## 4. What Should Stay Local to Orchestration

The following logic is still tightly coupled to live orchestration and should
not be treated like another extraction candidate:

- per-level pass-count selection
- runtime snapshot / restore choreography
- thick-restart anchor allocation / replay
- ensemble buffer allocation / winner selection
- the `partition_once(...)` / retry wrapper sequence

Interpretation:

- Day 8 should simplify this code path in place
- Day 8 should not invent a generic parser file or another graph module

## 5. Retry / Fallback Contract

The retry/fallback seam is already tight:

- `graph_partition_should_retry_with_forced_hem(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`

This glue now clearly composes:

- hierarchy build from `src/sparse_graph_coarsen.c`
- coarsest split from `src/sparse_graph_bisect.c`
- FM refinement from `src/sparse_graph_refine.c`
- separator lifting from `src/sparse_graph_separator.c`
- forced-HEM retry through the coarsening override seam

Interpretation:

- retry policy is now obviously orchestration-owned
- Day 8 should preserve it and only improve readability/structure

## 6. Internal Header / Comment Cleanup Notes

Day 7 did not surface a major internal-header redesign need.

The useful cleanup notes are small and local:

- `src/sparse_graph_internal.h` already reflects the module split well
- `src/sparse_graph.c` still has one stale section banner:
  - `Uncoarsening + vertex-separator extraction`
  - separator extraction now lives in `src/sparse_graph_separator.c`
- residual comment wording should emphasize:
  - uncoarsening
  - orchestration
  - retry glue

## 7. Concrete Day 8 Target List

Do on Day 8:

- consolidate the remaining config parsing blocks in `graph_uncoarsen(...)`
- simplify the finest/intermediate dispatch scaffolding without changing
  behavior
- tighten residual orchestration comments and section ownership wording
- keep `partition_once(...)` and retry glue behavior unchanged

Do not do on Day 8:

- no new graph module
- no public API change
- no retry-policy semantic change
- no second FM or separator redesign

## Bottom Line

Day 7 confirms that Sprint 44’s remaining graph queue is now concrete:

- there is no hidden fourth extraction seam
- the remaining parsing is orchestration-owned
- retry/fallback glue should stay local
- Day 8 can now deliver one bounded residual cleanup pass instead of another
  exploratory refactor
