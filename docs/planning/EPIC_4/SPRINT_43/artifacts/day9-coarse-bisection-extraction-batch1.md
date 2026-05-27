# Sprint 43 Day 9: Coarse-Bisection Extraction Batch I

## Summary

Day 9 landed the first real coarse-bisection extraction by moving the bounded
coarse-level split logic out of `src/sparse_graph.c` into:

- `src/sparse_graph_bisect.c`

The moved seam includes:

- brute-force coarse bisection
- GGGP coarse bisection
- Laplacian construction for spectral bisection
- spectral coarse-bisection support
- coarsest-bisection strategy parsing and dispatch

The key boundary held:

- `compute_cut_weight(...)` stayed in `src/sparse_graph.c`
- FM refinement, uncoarsening, separator lifting, and top-level orchestration
  stayed in `src/sparse_graph.c`

That keeps the batch faithful to the Sprint 43 plan: extract the coarse-level
split logic without dragging later graph phases forward.

## Files Changed

- `src/sparse_graph_bisect.c`
- `src/sparse_graph.c`
- `src/sparse_graph_internal.h`
- `Makefile`
- `CMakeLists.txt`

## What Moved

### 1. Coarse search and dispatch now have their own implementation unit

`src/sparse_graph_bisect.c` now owns:

- `bisect_brute_force(...)`
- `bfs_distances(...)`
- `bisect_gggp(...)`
- `graph_build_laplacian(...)`
- `cmp_double_asc(...)`
- `graph_bisect_coarsest_spectral(...)`
- `coarsest_bisect_strategy_t`
- `parse_coarsest_bisect_strategy(...)`
- `graph_bisect_coarsest(...)`

This is the right seam because these routines form one coherent coarse-level
partitioning layer:

- exact or heuristic initial coarse split
- optional spectral path
- env-var strategy routing

without owning any FM or projection behavior.

### 2. The remaining monolith is now more honestly an FM/uncoarsening/orchestration file

After Day 9, `src/sparse_graph.c` no longer owns the coarse-bisection
implementation block.

It now retains:

- `compute_cut_weight(...)`
- FM bucket + refinement implementation
- `graph_uncoarsen(...)`
- separator lifting
- top-level `partition_once(...)` / `sparse_graph_partition(...)`

Keeping `compute_cut_weight(...)` local was intentional because it is still a
real shared helper for FM/uncoarsening behavior and not just a coarse-level
detail.

### 3. The shared contract and build wiring were updated together

Day 9 also:

- added `src/sparse_graph_bisect.c` to `Makefile`
- added `src/sparse_graph_bisect.c` to `CMakeLists.txt`
- updated the internal-header ownership note so the bisection seam now points
  to `src/sparse_graph_bisect.c`
- updated `src/sparse_graph.c`'s top note so it no longer claims coarse
  bisection ownership

That keeps the code split, shared declarations, and build surfaces aligned.

## Boundary Quality

Day 9 intentionally did **not** move:

- `compute_cut_weight(...)`
- FM refinement
- `graph_uncoarsen(...)`
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
surface most relevant to this extraction:

- `test_graph`
- `test_graph_fm_buckets`
- `test_reorder_nd`
- `test_reorder_amd_qg`

All remained green.

## Day 9 Outcome

Sprint 43 now has a third real Phase-1 subsystem file:

- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`

The remaining `src/sparse_graph.c` is materially narrower and more focused on
the still-deferred FM/uncoarsening/orchestration seam. That is the right Day 9
result: a meaningful structural reduction in the graph hotspot while keeping
runtime behavior unchanged.
