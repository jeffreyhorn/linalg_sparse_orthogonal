# Sprint 67 Day 13: Full Validation Sweep

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Run the full Sprint 67 validation sweep from the landed large-source
maintainability state and reconfirm the highest-signal proof surfaces touched by
the graph/reorder extraction, shared ND policy convergence, and large-`n`
Cholesky CSC handoff work.

## Core Validation

The full validation sweep passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real)` = `418.98 sec`

## Touched Proof Surfaces

The highest-signal Sprint 67 proof reruns also passed:

- `./build/test_integration` = `47 / 47`
- `./build/test_graph` = `60 / 60`
- `./build/test_reorder_nd` = `34 / 34`
- `./build/test_chol_csc` = `145 / 145`

Representative retained proof points:

- `test_integration` still carries the public one-shot vs explicit repeated-run
  parity and failure-preservation lane without regression
- `test_graph` still carries the partition/uncoarsen/runtime-policy graph lane
  without regression
- `test_reorder_nd` still carries the shared ND compatibility/default-policy
  convergence lane and retained:
  - `Pres_Poisson`: `ND/AMD = 0.923`
  - `bcsstk14`: `ND/AMD = 1.124`
  - supernodal-postorder corpus `nnz(L)` invariants
- `test_chol_csc` still carries the family-local large-`n` analysis-backed
  Cholesky CSC handoff lane, including the Sprint 67 Day 11 helper-route proof

## Representative Example and Benchmark Signals

Representative examples and maintained benchmark/reporting surfaces also reran
cleanly:

- `./build/example_analysis`
- `./build/example_basic_solve`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Retained outputs:

- `example_analysis` residual stayed `4.44e-16`
- `example_basic_solve` residual stayed `0.00e+00`
- `bench_refactor_csc,proof,nos4.mtx,chol_spd,...,speedup_refactor=1.25,...`
- `bench_chol_csc,proof,nos4.mtx,chol_backend_compare,...,scalar,supernodal,builtin,...`
- `bench_iterative_reuse` kept bounded reuse deltas:
  - `cg-tridiag-300` = `1.02x`
  - `gmres-unsym-220` = `0.98x`
  - `minres-kkt-42` = `1.12x`
- `bench_eigs_reuse` kept stable reuse and agreement:
  - `growm-nos4-k5` = `1.06x`
  - `thick-bcsstk14-k5` = `1.00x`
  - `lobpcg-diag40-k3` = `1.04x`
  - `lambda_max_diff = 0.000e+00`

This confirms that the maintainability-oriented ownership moves did not disturb
the shipped workflow, residual, or benchmark/reporting contracts.

## Notes

One non-blocking reviewed-path note remained explicit:

- reviewed CMake `test_reorder_nd` still dominated the total runtime at
  `291.93 sec` out of the `418.98 sec` full reviewed-CMake `ctest` total

That is an existing runtime characteristic of the maintained reviewed baseline,
not a new Sprint 67 regression.

## Exit State

Sprint 67 Day 13 closes with:

- one revalidated strongest reviewed baseline
- one revalidated graph/reorder maintainability proof lane
- one revalidated large-`n` Cholesky CSC handoff proof lane
- one revalidated example and benchmark/reporting signal set
- one clean Day 14 closeout starting point
