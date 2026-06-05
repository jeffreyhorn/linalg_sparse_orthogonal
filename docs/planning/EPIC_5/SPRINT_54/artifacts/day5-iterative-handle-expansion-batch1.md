# Sprint 54 Day 5 - iterative handle expansion batch 1

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Land the only new iterative repeated-run public-handle family allowed by the
Day 4 boundary: MINRES.

## Landed public surface

Sprint 54 Day 5 adds bounded MINRES support to the existing public iterative
handle model:

- `sparse_iter_handle_prepare_minres(...)`
- `sparse_solve_minres_with_handle(...)`

This keeps the public repeated-run iterative handle set aligned around one
shared owner:

- `CG`
- `GMRES`
- `MINRES`

## Implementation shape

The landing stayed inside the existing handle model rather than inventing a
new MINRES-specific owner.

### Header surface

`include/sparse_iterative.h` now:

- exposes an explicit MINRES prepare helper
- exposes an explicit MINRES handle-backed solve entry
- keeps the one-shot MINRES surface intact

### Source integration

`src/sparse_iterative.c` now:

- routes MINRES through a shared workspace-backed execution seam
- lets the one-shot path allocate and free a private workspace wrapper
- lets the handle-backed path reuse the existing `sparse_iter_handle_t`
  workspace owner
- preserves zero-init on-demand handle growth

Interpretation:

- the repeated-run handle path is real reuse, not a parallel implementation
- one-shot and handle-backed MINRES now differ mainly in workspace ownership

## Regression proof added

`tests/test_iterative.c` now proves the public MINRES handle surface directly.

The new coverage checks:

- null prepare / null handle validation
- explicit prepare followed by repeated handle-backed solves
- zero-init handle growth on first use
- parity of convergence, solution quality, and iteration counts across reuse

## Validation

Required Day 5 gates all passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 123.22 sec`

Focused follow-ons also passed:

- `./build/test_iterative` -> `79 / 79`
- `./build/test_minres` -> `43 / 43`
- `./build/example_ic_minres`
- `./build/bench_iterative_reuse`

Representative direct outputs:

- `example_ic_minres`:
  - MINRES on the `42x42` KKT demo converged in `39` iterations
  - Jacobi-preconditioned MINRES converged in `26` iterations
- `bench_iterative_reuse`:
  - `cg-tridiag-300`: `1.00x`
  - `gmres-unsym-220`: `1.05x`

## Boundary preserved

The Day 5 landing did not broaden Sprint 54 beyond the Day 4 support line.

Still explicitly out of scope after this batch:

- BiCGSTAB public repeated-run handle exposure
- block iterative public-handle exposure
- backend-specific eigensolver API expansion

## Conclusion

Day 5 closes the highest-value public iterative repeated-run asymmetry without
changing the underlying public handle model:

- MINRES is now part of the supported public repeated-run iterative-handle set
- one-shot MINRES remains first-class
- the remaining Sprint 54 work can focus on proof/alignment and eigensolver
  tightening rather than more iterative handle invention
