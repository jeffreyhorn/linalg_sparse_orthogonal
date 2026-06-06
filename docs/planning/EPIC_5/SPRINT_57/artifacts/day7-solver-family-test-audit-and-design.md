# Sprint 57 Day 7 - solver-family test audit and design

Date: 2026-06-06
Branch: `sprint-57`

## Scope

Reduce the iterative/eigensolver giant-test queue to explicit seam classes and
freeze the first bounded solver-family refactor boundary before any Day 8 code
movement lands.

## Live hotspot ranking

Current solver-family giant-test sizes:

- `tests/test_svd.c` = `3746`
- `tests/test_qr.c` = `3197`
- `tests/test_iterative.c` = `2993`

All three are still real maintainability hotspots, but their ownership shapes
are different enough that line count alone would pick the wrong first target.

## Ownership audit

### `tests/test_iterative.c`

Main seam classes:

- CG proof
- GMRES proof
- right-preconditioning proof
- matrix-free proof
- public repeated-run handle proof
- shared matrix builders and callback/preconditioner shims

Assessment:

- large and important
- but already organized around a shared public front-door model
- remaining helper seams are smaller and more generic than the file size first
  suggests

### `tests/test_qr.c`

Main seam classes:

- Householder / reconstruction
- least-squares solve
- SuiteSparse validation
- rank / null-space
- economy mode
- sparse-mode
- refinement

Assessment:

- large, but its common helper layer is already fairly consolidated
- likely closer to a consciously dense proof file than a cluttered ownership
  file
- not the highest-value first Sprint 57 solver-family move

### `tests/test_svd.c`

Main seam classes:

- Golub-Kahan extraction / validation
- bidiagonal and full-SVD convergence
- full-SVD driver proof
- partial-SVD backend proof
- partial-SVD vector proof
- low-rank / pseudoinverse / condition-number applications
- Sprint 29 outer-product / full-mode follow-through

Assessment:

- the largest remaining solver-family test
- also the cleanest backend-owned split
- partial-SVD proof is large, contiguous, and behaviorally cohesive
- best first maintainability target

## Ranked target order

1. `tests/test_svd.c`
2. `tests/test_iterative.c`
3. `tests/test_qr.c`

This ranking is based on seam quality and ownership clarity, not only size.

## Selected first refactor boundary

### First target

- `tests/test_svd.c`

### First owned seam

- the partial-SVD family

Specifically:

- partial-SVD backend proof (`test_partial_svd_*`)
- partial-SVD vector proof (`test_partial_svd_vectors_*`)

### Preferred Day 8 landing style

- build-neutral include-style seam
- recommended new local file:
  - `tests/test_svd_partial_helpers.h`

This mirrors the successful Day 5 pattern:

- create a real owned seam
- keep the test binary and runner shape stable
- avoid unnecessary Makefile/CMake churn in the first solver-family batch

## Exact Day 8 ownership split

### Move into `tests/test_svd_partial_helpers.h`

- partial-SVD-family local helpers
- any narrow support routines used only by:
  - `test_partial_svd_*`
  - `test_partial_svd_vectors_*`

### Keep in `tests/test_svd.c`

- Golub-Kahan extraction / validation
- bidiagonal/full-SVD convergence groups
- full-SVD driver proof
- low-rank / pseudoinverse / condition-number groups
- Sprint 29 outer-product / full-mode follow-through
- `main()` and current `RUN_TEST(...)` ordering

## Explicit non-goal fence

Day 8 should not:

- create a new test target
- edit `Makefile`
- edit `CMakeLists.txt`
- widen generic test-helper infrastructure broadly
- rewrite low-rank or full-SVD proof meaning
- treat QR or iterative tests as collateral cleanup

## Preserved invariants

The first solver-family refactor must preserve:

- the `test_svd` binary shape
- `main()` ownership in `tests/test_svd.c`
- test names and proof intent
- partial-SVD output truthfulness
- low-rank / full-mode output truthfulness
- fixture coverage, especially:
  - `nos4`
  - `west0067`
  - `bcsstk04`
  - `steam1`
  - `orsirr_1`

This is an ownership/readability change, not an SVD-behavior change.

## Conclusion

Day 7 turns the solver-family maintainability queue into a concrete plan:

- first target:
  - `tests/test_svd.c`
- first seam:
  - partial-SVD backend + vector proof family
- preferred Day 8 landing:
  - build-neutral local include seam via
    `tests/test_svd_partial_helpers.h`

That gives Sprint 57 a precise solver-family implementation boundary instead
of a generic "refactor a giant solver test" placeholder.
