# Sprint 65 Day 6: Canonical Performance Surface and Implementation Fence

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Convert the Day 5 taxonomy and normalization design into one exact canonical
maintained performance surface, one explicit non-canonical queue, and one
bounded implementation fence before the first benchmark/output or
solver-efficiency edits land.

## Canonical Maintained Performance Surface

The maintained canonical Sprint 65 surface should be:

- `bench_refactor_csc`
- `bench_chol_csc`
- `bench_iterative_reuse`
- `bench_eigs_reuse`

What each proves:

- `bench_refactor_csc`
  - repeated-run direct throughput and CSC follow-through on the maintained
    analyze-once / factor-many lane
- `bench_chol_csc`
  - backend/path identity plus bounded Cholesky CSC throughput signal
- `bench_iterative_reuse`
  - repeated-run iterative public-handle efficiency signal
- `bench_eigs_reuse`
  - repeated-run eigensolver public-handle efficiency signal

## Important but Non-Canonical Surfaces

These remain valuable benchmark-side proof surfaces without defining the first
canonical batch:

- `bench_refactor`
- `bench_ldlt_csc`

These remain explicit non-canonical sets:

- regression-sensitive runtime:
  - `bench_scaling`
  - `bench_fillin`
  - `bench_colamd`
  - `bench_reorder --skip-factor`
  - maybe `bench_amd_qg`
- exploratory or later:
  - `bench_main`
  - `bench_convergence`
  - `bench_svd`
  - `bench_bicgstab`
  - `bench_eigs`
  - broader `bench_reorder`

## Day 7-10 Implementation Fence

### Required first-batch benchmark/doc surfaces

- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `benchmarks/README.md`
- `README.md`
- `docs/maintainer_guide.md`

### Likely proof surfaces

- `tests/test_chol_csc.c`
- `tests/test_integration.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`

### Likely solver/hotspot surfaces if the first efficiency target lands on the direct repeated-run lane

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`

### Conditional-only solver surfaces

- `src/sparse_iterative.c`
- `src/sparse_iterative_workspace_internal.c`
- `src/sparse_eigs.c`
- `src/sparse_eigs_workspace_internal.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_supernodal.c`

## Initial Solver-Efficiency Shortlist

### 1. Direct repeated-run CSC/Cholesky follow-through

Strongest evidence surfaces:

- `bench_refactor_csc`
- `bench_chol_csc`

Likely solver seams:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`

Likely proof homes:

- `tests/test_integration.c`
- `tests/test_chol_csc.c`

### 2. Iterative public-handle reuse follow-through

Evidence surface:

- `bench_iterative_reuse`

Likely solver seams:

- `src/sparse_iterative.c`
- `src/sparse_iterative_workspace_internal.c`

Proof home:

- `tests/test_iterative.c`

### 3. Eigensolver public-handle reuse follow-through

Evidence surface:

- `bench_eigs_reuse`

Likely solver seams:

- `src/sparse_eigs.c`
- `src/sparse_eigs_workspace_internal.c`

Proof home:

- `tests/test_eigs.c`

## Day 6 Exit State

Sprint 65 now has:

- one exact four-binary canonical maintained performance surface
- one explicit non-canonical proof/runtime/exploratory split
- one bounded Day 7-10 implementation fence
- one ranked solver-efficiency shortlist led by the direct repeated-run
  CSC/Cholesky lane
