# Sprint 38 Day 3 Compile-Only Regression Surface Audit

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Audit the named Sprint 34 exclusion-list binaries directly so Sprint 38 can
separate true compile-only regression gaps from the narrower dead-code
compile-db/reporting gap that still remains open.

## Named Sprint 34 Exclusion List

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

## Current Ground Truth

### Makefile compile-only protection

Current Makefile truth:

- `BENCH_SRCS` explicitly includes `bench_svd.c`
- `EX_SRCS = $(wildcard examples/*.c)` includes all six named example files
- `bench-build` compiles all benchmark binaries
- `examples-build` compiles all example binaries
- `tooling-build = bench-build + examples-build`
- `lint` depends on `tooling-build`

Observed current counts:

- benchmarks built by `tooling-build` = `14`
- examples built by `tooling-build` = `12`

Interpretation:

- all seven named exclusion-list binaries are already under the maintained
  Makefile compile-only protection path

### Dead-code compile-db / reporting coverage

Current `build/deadcode-cmake/compile_commands.json` membership:

- `bench_svd.c` = missing
- `example_basic_solve.c` = missing
- `example_condition.c` = missing
- `example_iterative.c` = missing
- `example_least_squares.c` = missing
- `example_matrix_free.c` = missing
- `example_svd_lowrank.c` = missing

Current `build/deadcode/coverage-notes.txt` summary:

- benchmarks covered = `13`
- examples covered = `6`
- missing benchmark:
  - `bench_svd`
- missing examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Interpretation:

- the named list is still a real open exclusion queue
- but it is now specifically a dead-code compile-db/reporting exclusion queue

### CMake registration surface

Current CMake example registration includes only:

- `example_ldlt`
- `example_ic_minres`
- `example_analysis`
- `example_minnorm`
- `example_colamd`
- `example_eigs`

The six named missing examples are not registered in the dead-code CMake tree.

Current benchmark registration includes many bench binaries, but the named
`bench_svd` target is still absent from that dead-code compile-db tree.

Interpretation:

- the root cause is bounded and structural:
  - partial benchmark/example CMake registration for the dead-code tree
- this is narrower than a generic "compile_commands coverage is flaky" claim

## Status Map For The Seven Named Surfaces

| Surface | Makefile compile-only protection | Dead-code compile-db/reporting | Current status |
|---|---|---|---|
| `bench_svd` | yes | no | compile-protected; dead-code-excluded |
| `example_basic_solve` | yes | no | compile-protected; dead-code-excluded |
| `example_condition` | yes | no | compile-protected; dead-code-excluded |
| `example_iterative` | yes | no | compile-protected; dead-code-excluded |
| `example_least_squares` | yes | no | compile-protected; dead-code-excluded |
| `example_matrix_free` | yes | no | compile-protected; dead-code-excluded |
| `example_svd_lowrank` | yes | no | compile-protected; dead-code-excluded |

## Close / Re-Document / Defer

### Already closed elsewhere

These are no longer generic compile-only regression gaps:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

Reason:

- the maintained Makefile compile-only path already builds them through
  `tooling-build`

### Still open and should be described more precisely

These remain open as dead-code/reporting exclusions:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

Reason:

- the dead-code CMake compile database and `coverage-notes.txt` still omit them

### Defer unless later Day 6 implementation chooses to close them directly

- routine execution of these binaries as part of the reviewed baseline
- broader CMake parity ownership of all benchmark/example compile-only surfaces

## Day 6 Implementation Direction

The Sprint 38 compile-only batch should now be framed more precisely:

1. keep the Makefile compile-only story intact
2. stop describing the seven files as generic compile-only drift
3. choose one of two honest follow-through paths:
   - implementation path:
     - broaden the dead-code CMake compile-db to include some or all of the
       seven excluded files
   - documentation path:
     - re-document the remaining list explicitly as a dead-code/reporting
       limitation rather than a missing Makefile compile-only protection surface

The important Day 3 result is the vocabulary correction: the exclusion list is
still real, but it now belongs primarily to dead-code/reporting maturity, not to
the already-shipped Makefile compile-only gate.
