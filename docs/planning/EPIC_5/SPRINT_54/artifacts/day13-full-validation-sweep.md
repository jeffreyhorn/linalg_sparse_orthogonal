# Sprint 54 Day 13: Full Validation Sweep

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Day 13 runs the full validated closeout from the landed Sprint 54 state.

The goal is not another design or compatibility pass. The goal is to confirm
that the full required gate, the reviewed Makefile/CMake truthfulness anchors,
and the targeted Sprint 54 repeated-run solver follow-ons all pass together
from the same branch state.

## Main Day 13 Conclusion

Sprint 54 has a real measured validation close state:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `make quality-review-full` reviewed CMake total time = `144.25 sec`

## Required Gate

### `make format`

- passed

### `make lint`

- passed

### `make test`

- passed

### `make quality-review-full`

- passed

This included:

- reviewed Makefile path
- dead-code report-completeness closeout
- reviewed CMake rebuild/parity path
- full reviewed CMake `ctest`

## Truthfulness Anchors

The maintained reviewed truthfulness anchors remained exact:

- reviewed CMake test discovery:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity:
  - `53 vs 53`
- reviewed CMake execution through `make quality-review-full`:
  - `53 / 53`
- reviewed CMake total time from `make quality-review-full`:
  - `144.25 sec`

## Targeted Sprint 54 Follow-Ons

### Supported iterative repeated-run proof

- `./build/test_iterative`
  - `79 / 79` passed
- `./build/test_minres`
  - `43 / 43` passed

These reruns keep the full supported iterative handle set green:

- `CG`
- `GMRES`
- `MINRES`

### Iterative example surfaces

- `./build/example_iterative`
  - unpreconditioned GMRES:
    - `25` iterations
    - residual `9.56e-11`
  - ILU(0)-preconditioned GMRES:
    - `9` iterations
    - residual `3.14e-11`

- `./build/example_ic_minres`
  - MINRES on the `42x42` KKT system:
    - `39` iterations
    - residual `3.87e-11`
  - Jacobi-MINRES:
    - `26` iterations
    - residual `4.16e-11`
  - block MINRES on the `28x28` KKT system:
    - residual `8.06e-16`

### Iterative reuse benchmark proof

- `./build/bench_iterative_reuse`
  - `cg-tridiag-300`
    - `speedup = 1.12x`
  - `gmres-unsym-220`
    - `speedup = 0.85x`
  - `minres-kkt-42`
    - `speedup = 1.28x`

The benchmark still matches the final supported public handle set and keeps
iteration/residual parity between one-shot and reuse paths.

### Supported eigensolver repeated-run proof

- `./build/test_eigs`
  - `30 / 30` passed
- `./build/test_eigs_lobpcg`
  - `26 / 26` passed

These reruns keep the full supported eigensolver handle backend set green:

- grow-m Lanczos
- thick-restart Lanczos
- explicit `LOBPCG`

### Eigensolver example surface

- `./build/example_eigs`
  - nos4 largest-eigenvalue case:
    - `5 / 5` pairs
    - `115` Lanczos iterations
  - KKT nearest-`sigma` case:
    - `3 / 3` pairs
    - `6` Lanczos iterations
  - explicit LOBPCG on `bcsstk04`:
    - `3 / 3` smallest eigenpairs
    - `62` outer iterations
    - reported residual `8.808e-09`

### Eigensolver reuse benchmark proof

- `./build/bench_eigs_reuse`
  - `growm-nos4-k5`
    - `speedup = 1.00x`
  - `thick-bcsstk14-k5`
    - `speedup = 0.99x`
  - `lobpcg-diag40-k3`
    - `speedup = 1.00x`
  - all three kept exact eigenvalue parity:
    - `|lambda|max diff = 0.000e+00`

## Operational Result

Sprint 54 now closes from a validated measured baseline instead of from
inference:

1. the full required gate passed
2. the reviewed Makefile/CMake truthfulness anchors remained exact
3. the supported iterative and eigensolver repeated-run proof surfaces stayed
   green together

No new reconciliation queue surfaced during validation. Day 14 can therefore
focus on closeout and Sprint 55 handoff rather than post-validation repair.
