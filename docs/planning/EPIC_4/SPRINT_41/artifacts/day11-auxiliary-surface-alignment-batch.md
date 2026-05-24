# Sprint 41 Day 11 Artifact: Auxiliary-Surface Alignment Batch

## Purpose

Land the bounded auxiliary batch chosen on Day 10 by:

- aligning the highest-value small public examples with Sprint 41's allocation
  safety conventions
- preserving the internal-first / public-boundary rule from Sprint 40
- recording the residual auxiliary work that still belongs to later sprints

## Landed Batch

### New shared example-only helper surface

Added:

- `examples/example_alloc_helpers.h`

This helper header mirrors the **conventions** of the Sprint 41 internal
helper layer without exposing `src/` private headers to public examples.

It provides:

- `example_check_array_bytes(...)`
- `example_malloc_array(...)`
- `example_calloc_array(...)`

Design intent:

- keep public examples on public headers only
- reuse one bounded auxiliary helper seam instead of reintroducing ad hoc
  `malloc((size_t)n * sizeof(T))` / `calloc((size_t)n, sizeof(T))` drift
- preserve the readability of the examples by keeping the helper interface
  small and direct

### Touched examples

Primary Day 10 targets landed:

- `examples/example_iterative.c`
- `examples/example_matrix_free.c`

Optional bounded add-on also landed:

- `examples/example_colamd.c`

## What Changed

### `examples/example_iterative.c`

Replaced:

- raw `calloc((size_t)n, sizeof(double))` for:
  - `b`
  - `x`
  - `ones`

With:

- `example_calloc_array(...)`

Preserved exactly:

- GMRES / ILU example flow
- allocation-failure messaging
- solve semantics
- printed output structure

### `examples/example_matrix_free.c`

Replaced:

- raw `calloc((size_t)n, sizeof(double))` for:
  - `b`
  - `x`
  - `x_exact`

With:

- `example_calloc_array(...)`

Preserved exactly:

- matrix-free/operator-teaching semantics
- diagonal preconditioner example flow
- output/reporting structure

### `examples/example_colamd.c`

Replaced:

- raw `malloc((size_t)n * sizeof(idx_t))` for:
  - `perm`
  - `id_perm`

With:

- `example_malloc_array(...)`

Preserved exactly:

- COLAMD permutation flow
- LU fill-in comparison logic
- QR+COLAMD demonstration

## Why This Batch Stayed Bounded

Day 11 deliberately did **not**:

- include `src/sparse_alloc_internal.h` in public examples
- widen into `example_eigs.c`, `example_ic_minres.c`, or `example_analysis.c`
- touch benchmark harnesses
- touch scripts
- widen into broader public-doc or usability rewrites

That keeps the batch inside the Day 10 contract:

- auxiliary alignment only where semantics are truly parallel
- no broad public-facing cleanup batch

## Residual Auxiliary Follow-On List

Still deferred from the routine Sprint 41 auxiliary path:

- public-teaching examples with larger lifecycle/composition significance:
  - `examples/example_eigs.c`
  - `examples/example_ic_minres.c`
  - `examples/example_analysis.c`
- benchmark harnesses:
  - `benchmarks/bench_main.c`
  - `benchmarks/bench_eigs.c`
  - broader benchmark cluster from Day 10
- script maintainability surfaces:
  - `scripts/deadcode_report.py`
  - `scripts/deadcode_workflow.sh`

## Validation

Because `*.c` and `*.h` files changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Targeted auxiliary checks also passed:

- `./build/example_iterative`
- `./build/example_matrix_free`
- `./build/example_colamd`

Representative live outputs:

- `example_iterative`:
  - unpreconditioned GMRES converged in `25` iterations
  - ILU(0)-preconditioned GMRES converged in `9` iterations
- `example_matrix_free`:
  - both runs converged in `3` iterations
  - computed solution matched `x_exact` to `~1e-13`
- `example_colamd`:
  - ran end to end
  - QR+COLAMD residual printed `0.00e+00`

## Day 11 Conclusion

Sprint 41 now has a real auxiliary proof point for its shared-allocation
cleanup, but the batch stayed honest:

- public examples were aligned without leaking private internals
- the smallest high-value surfaces were handled first
- larger public-teaching, benchmark, and script surfaces remain explicitly
  deferred instead of being pulled into a mixed cleanup pass
