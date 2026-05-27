# Sprint 43 Day 6: Hierarchy / Coarsening Extraction Batch I

## Summary

Day 6 landed the first hierarchy/coarsening extraction batch by moving the
multilevel coarsening core and hierarchy lifecycle out of
`src/sparse_graph.c` into:

- `src/sparse_graph_coarsen.c`

This batch also carried the small internal HEM/HCC strategy seam that the
remaining graph orchestration still depends on for the sep=0 retry path.

## Files Changed

- `src/sparse_graph_coarsen.c`
- `src/sparse_graph.c`
- `src/sparse_graph_internal.h`
- `Makefile`
- `CMakeLists.txt`

## What Moved

### 1. Coarsening strategy ownership

The new module now owns:

- `coarsening_strategy_t`
- the internal coarsening-strategy parser
- the HEM-override thread-local state
- the small internal helpers:
  - `sparse_graph_coarsening_strategy_current(...)`
  - `sparse_graph_force_hem_override_begin(...)`
  - `sparse_graph_force_hem_override_end(...)`

That is the right ownership boundary because the sep=0 retry path is still a
top-level orchestration concern, but the strategy interpretation and override
mechanics belong to the coarsening subsystem.

### 2. Heavy-edge / HCC coarsening core

`src/sparse_graph_coarsen.c` now owns:

- `splitmix64_next(...)`
- `fisher_yates_shuffle(...)`
- `coarse_edge_t`
- `cmp_coarse_edge(...)`
- `graph_coarsen_with_strategy(...)`
- `graph_coarsen_heavy_edge_matching(...)`
- `graph_coarsen_hcc(...)`

This is the central Day 6 seam:

- matching-loop scoring and tie-break behavior
- adaptive HCC-to-HEM fall-through
- coarse-graph construction
- weight aggregation
- sort/merge dedup
- debug cmap emission

all now live together instead of being embedded in the main graph monolith.

### 3. Hierarchy lifecycle

The new module also now owns:

- `sparse_graph_hierarchy_free(...)`
- `sparse_graph_hierarchy_build(...)`

This keeps:

- coarse-graph ownership transitions
- `cmap` ownership
- stop-condition logic
- per-level coarsening dispatch

inside the same subsystem as the coarsening core itself.

## What Stayed in `src/sparse_graph.c`

Day 6 intentionally did **not** move:

- coarse bisection
- FM refinement
- separator lifting
- top-level uncoarsening / partition orchestration

That preserves the Sprint 43 Phase-1 order:

1. graph ownership / construction
2. hierarchy / coarsening
3. coarse bisection
4. later FM / separator cleanup

## Internal Header Changes

`src/sparse_graph_internal.h` was expanded only where the new file boundary
required stable cross-file declarations:

- `coarsening_strategy_t`
- the internal strategy/override helper declarations

The header still stayed compact:

- no public API changes
- no new public headers
- no FM bucket leakage into the new module

## Boundary Quality

The Day 6 extraction preserved the intended subsystem split:

- `src/sparse_graph_coarsen.c` now owns hierarchy/coarsening behavior and
  lifecycle
- remaining `src/sparse_graph.c` still owns FM, separator lifting, and
  orchestration
- the sep=0 retry path now talks to the coarsening subsystem through a small
  explicit seam instead of sharing raw thread-local state in the same file

That is a better Phase-1 architecture without changing user-visible behavior.

## Validation

Because `*.c` and `*.h` files changed, the full required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

The authoritative `make test` sweep also covered the graph-focused regression
surface affected by this extraction:

- `test_graph`
- `test_reorder_nd`
- `test_reorder_amd_qg`

All remained green.

## Day 6 Outcome

Sprint 43 now has the second real Phase-1 subsystem file in place:

- graph ownership / construction is already split into
  `src/sparse_graph_core.c`
- hierarchy / coarsening is now split into `src/sparse_graph_coarsen.c`
- the remaining monolith is materially smaller and more focused on the still-
  deferred bisection/FM/separator/orchestration seams

This is the right Day 6 result: a meaningful structural reduction in the graph
hotspot without dragging FM or separator work into the batch.
