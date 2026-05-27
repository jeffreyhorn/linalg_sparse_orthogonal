# Sprint 43 Day 10: Runtime Strategy / Glue Reconciliation

## Summary

Day 10 reconciled the remaining top-level graph orchestration after the
Phase-1 graph splits. It did **not** extract another subsystem file. Instead,
it made the residual ownership in `src/sparse_graph.c` read more clearly as a
composition layer over the already-extracted modules.

The touched code stayed deliberately narrow:

- `src/sparse_graph.c`
- `src/sparse_graph_internal.h`

The main result is that the remaining graph monolith now more explicitly owns:

- FM refinement
- uncoarsening
- separator lifting
- sep=0 retry/orchestration policy

while the extracted modules continue to own:

- construction/ownership (`src/sparse_graph_core.c`)
- hierarchy/coarsening (`src/sparse_graph_coarsen.c`)
- coarse bisection (`src/sparse_graph_bisect.c`)

## Files Changed

- `src/sparse_graph.c`
- `src/sparse_graph_internal.h`

## What Changed

### 1. The remaining orchestration layer now has explicit local glue helpers

Day 10 added four small helpers in `src/sparse_graph.c`:

- `graph_hierarchy_coarsest(...)`
- `graph_partition_seed_coarsest(...)`
- `graph_partition_count_separator_vertices(...)`
- `graph_partition_should_retry_with_forced_hem(...)`

These helpers do not change algorithmic behavior. They make the remaining
file's role clearer:

- get the coarsest graph from the hierarchy
- seed the coarsest split + immediate FM refinement
- count separator vertices after separator lifting
- decide whether the top-level orchestration should retry with forced HEM

That is the right Day 10 landing because it clarifies the composition seam
without reopening FM or separator extraction.

### 2. Sep=0 retry policy now has a clearer home

Before Day 10, the sep=0 retry rule was embedded directly inside
`sparse_graph_partition(...)`.

After Day 10, the orchestration layer owns that decision more explicitly via:

- `graph_partition_should_retry_with_forced_hem(...)`

The underlying coarsening override implementation still lives where it should:

- `sparse_graph_coarsening_strategy_current()`
- `sparse_graph_force_hem_override_begin()`
- `sparse_graph_force_hem_override_end()`

So the boundary is now clearer:

- coarsening module owns strategy/override implementation
- orchestration layer owns the retry policy that composes it

### 3. The coarsest split + first FM step now reads as one cross-module seam

Day 10 also consolidated the coarsest partition seed flow into:

- `graph_partition_seed_coarsest(...)`

This helper makes the remaining orchestration shape more legible:

- `graph_bisect_coarsest(...)` comes from `src/sparse_graph_bisect.c`
- `graph_refine_fm(...)` still lives in `src/sparse_graph.c`

That reduces the drift created by earlier splits while keeping runtime
strategy semantics unchanged.

### 4. Shared ownership notes were reconciled with the live split

The internal graph header now states the post-Day-9/Day-10 ownership model
more directly:

- `src/sparse_graph_coarsen.c`
  - hierarchy build
  - coarsening strategy ownership
  - HEM retry override seam
- `src/sparse_graph_bisect.c`
  - coarsest-level split routing
  - spectral / brute / GGGP behavior
- `src/sparse_graph.c`
  - FM refinement
  - uncoarsening
  - separator lifting
  - final retry/orchestration glue

This is the real reconciliation value of Day 10: the shared contract now
matches the current Phase-1 module layout.

## Deliberate Deferrals

Day 10 intentionally did **not** pull forward:

- FM refinement extraction
- separator lifting extraction
- broader environment-parser consolidation
- deeper runtime-strategy cleanup beyond the retry/orchestration seam

Those remain the correct later-phase graph tasks.

## Validation

Because `*.c` and `*.h` files changed, the full required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

The authoritative `make test` sweep again covered the graph/ND surfaces most
relevant to this cleanup:

- `test_graph`
- `test_graph_fm_buckets`
- `test_reorder_nd`
- `test_reorder_amd_qg`

All remained green.

## Day 10 Outcome

Day 10 did not make the graph subsystem *smaller* by another file. It made the
remaining `src/sparse_graph.c` **truer** to its current ownership role.

That is the right mid-sprint result:

- extracted modules now integrate through clearer local glue
- top-level retry/orchestration ownership is easier to read
- the Phase-2 FM/separator queue remains explicitly deferred
- runtime behavior stayed stable
