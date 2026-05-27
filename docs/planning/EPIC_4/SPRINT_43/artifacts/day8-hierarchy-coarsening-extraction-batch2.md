# Sprint 43 Day 8: Hierarchy / Coarsening Extraction Batch II

## Summary

Day 8 completed the planned first-phase hierarchy/coarsening extraction as a
bounded interface/ownership cleanup batch rather than another broad code move.

The key change is that the shared graph internal contract now reflects the
real post-Day-6 ownership split more clearly:

- the coarsening section explicitly owns hierarchy/coarsening behavior and
  points to `src/sparse_graph_coarsen.c`
- `graph_build_laplacian(...)` is no longer grouped under coarsening
  declarations
- the coarse-bisection / FM section now explicitly owns that Laplacian helper
  and the remaining `src/sparse_graph.c` monolith seam
- the top-of-file ownership note in `src/sparse_graph.c` now matches the
  actual Sprint 43 Phase-1 split

This is the right Day 8 result because Day 7 showed that the implementation
extraction itself was already materially complete. The real residual risk was
interface drift that could blur the Day 9 coarse-bisection seam.

## Files Changed

- `src/sparse_graph_internal.h`
- `src/sparse_graph.c`

## What Changed

### 1. The shared internal header now groups declarations by real Phase-1 ownership

The coarsening banner in `src/sparse_graph_internal.h` now reads as:

- multilevel coarsening + hierarchy lifecycle
- implementation lives in `src/sparse_graph_coarsen.c`
- new coarsening helpers should not be added back to the remaining monolith

That makes the Phase-1 extraction boundary explicit where future work will
actually look first.

### 2. `graph_build_laplacian(...)` was moved to the bisection section

Before Day 8, the Laplacian builder still sat under the coarsening block even
though it only supports spectral coarse bisection.

Day 8 moved its declaration and contract comment so it now lives with:

- `graph_bisect_coarsest(...)`
- the coarse-bisection support surface
- the FM/uncoarsening section that still remains in `src/sparse_graph.c`

That is the most meaningful shared-header cleanup before the planned Day 9
extraction.

### 3. The remaining monolith now documents its real ownership more honestly

The top-level file note in `src/sparse_graph.c` now says directly that this
file is the remaining:

- coarse-bisection
- FM refinement
- uncoarsening
- separator lifting
- top-level orchestration

slice of the original Sprint 22 graph partitioner, while:

- graph construction / ownership lives in `src/sparse_graph_core.c`
- hierarchy / coarsening lives in `src/sparse_graph_coarsen.c`

That removes the stale implication that the file still owns the full graph
pipeline.

## Boundary Quality

Day 8 intentionally did **not** move:

- coarse-bisection implementation
- `graph_uncoarsen(...)`
- FM refinement
- separator lifting
- top-level retry/orchestration glue

That preserves the intended Sprint 43 order:

1. graph ownership / construction
2. hierarchy / coarsening
3. coarse bisection
4. later FM / separator cleanup

## Validation

Because `*.c` and `*.h` files changed, the full required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

The authoritative `make test` sweep also covered the graph-focused regression
surface most relevant to the current subsystem boundary:

- `test_graph`
- `test_graph_fm_buckets`
- `test_reorder_nd`
- `test_reorder_amd_qg`

All remained green.

## Day 8 Outcome

Sprint 43's first-phase hierarchy/coarsening extraction is now complete in the
way Day 7 argued it should be:

- implementation moved out on Day 6
- interface ownership tightened on Day 8
- the remaining monolith is clearer and narrower
- Day 9 now has a cleaner coarse-bisection extraction seam without dragging
  FM or orchestration work forward
