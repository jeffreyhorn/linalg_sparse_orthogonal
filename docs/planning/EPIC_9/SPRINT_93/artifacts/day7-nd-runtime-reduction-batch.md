# Sprint 93 Day 7: ND Runtime Reduction Batch

## Purpose

Land one bounded recursion-side runtime reduction inside the reviewed ND owner
without changing ND policy semantics, widening graph-policy work, or reopening
proof topology beyond directly forced validation.

## Main Result

The Day 7 landing stayed inside the exact Day 6 contract:

- the only code owner touched was `src/sparse_reorder_nd.c`
- no directly forced edits were needed in:
  - `src/sparse_graph.c`
  - `src/sparse_graph_refine.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `benchmarks/bench_reorder.c`

The landed runtime reduction is now explicit:

- ND no longer allocates two separate side arrays (`vs0`, `vs1`) per
  non-leaf recursion frame
- ND no longer performs a separate full post-recursion scan over `part[]` to
  emit separators
- one `scratch` buffer now carries:
  - side 0 vertices
  - side 1 vertices
  - separator vertices
- the recursive side calls and final separator-last emission both consume that
  same packed layout

## Preserved Semantics

The batch preserved the bounded Day 6 invariants:

- per-side vertex order is still ascending because packing still walks
  `part[]` left-to-right
- `perm[new_i] = old_i` stays unchanged
- separator-last behavior stays unchanged
- current threshold, policy, and env/control interpretation stay unchanged

## Validation

The required code-day validation queue passed cleanly:

- `make format`
- `make lint`
- `make test`

## Exit State

- Sprint 93 now has one landed ND recursion-side runtime reduction batch.
- The first implementation seam reduced heap churn and post-partition scan
  cost without changing ND ordering semantics.
- The remaining runtime debt can now be reranked from the post-landing tree
  before any broader runtime-control or proof-surface widening is considered.
