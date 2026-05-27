# Sprint 44 Day 10 Artifact: Focused Graph Seam Tests

## Purpose

Add the smallest useful graph regression batch after the Sprint 44 Phase-2
split so the extracted separator-policy seam and the residual orchestration
composition path both gain explicit protection.

## 1. Tests Added

Day 10 added two tests in `tests/test_graph.c`.

### Direct separator-policy contract

- `test_edge_to_vertex_separator_balanced_boundary_prefers_smaller_boundary`

What it covers:

- a crafted graph/partition where:
  - the smaller side has the larger boundary
  - the larger side has the smaller boundary
- `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary`
- direct call to `graph_edge_separator_to_vertex_separator(...)`

What it asserts:

- the conversion succeeds
- the resulting partition invariant holds
- exactly one separator vertex is chosen
- the chosen separator is the unique smaller-boundary-side vertex

Why it matters:

- it directly protects the extracted separator module
- it goes beyond the old default smaller-side test
- it remains behavior-level rather than implementation-level

### Post-split orchestration smoke

- `test_partition_fifo_balanced_boundary_smoke`

What it covers:

- full `sparse_graph_partition(...)` path on a 10×10 grid
- non-default configuration:
  - `SPARSE_ND_COARSENING=heavy_edge`
  - `SPARSE_FM_FINEST_STRATEGY=fifo`
  - `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary`

What it asserts:

- partition succeeds
- invariant holds
- separator count is nondegenerate but broadly bounded
- both interior sides remain populated

Why it matters:

- it protects the interaction boundary between:
  - extracted coarsening
  - extracted FM refinement
  - extracted separator lifting
  - residual uncoarsening / orchestration glue
- it avoids pinning private `graph_uncoarsen(...)` structure

## 2. What Day 10 Did Not Do

The batch stayed intentionally small.

It did **not**:

- add a new graph-focused test binary
- add direct tests for static/local parser helpers
- add FM-private unit tests just because FM moved files
- add bisection-private tests just because bisection moved files
- change any production `src/` code

That matches the Day 9 audit:

- FM and bisection were already well protected
- the separator policy seam was the clearest direct gap
- residual orchestration should stay protected end-to-end

## 3. Validation

Because `tests/test_graph.c` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Inside the authoritative `make test` sweep, `test_graph` reported both new
cases as passing:

- `test_edge_to_vertex_separator_balanced_boundary_prefers_smaller_boundary`
- `test_partition_fifo_balanced_boundary_smoke`

## Bottom Line

Day 10 closed the intended graph-test gap without broadening scope:

- the extracted separator seam now has a direct non-default policy contract
- the post-split orchestration path now has one compact non-default smoke
- the graph regression surface stayed behavior-oriented and bounded
