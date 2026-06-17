# Sprint 74 Day 9: Scalar Surface Preparation Batch

Date: 2026-06-16
Branch: `sprint-74`

## Purpose

Land the first bounded scalar-surface preparation seam for later capability
widening, keeping the work inside the Day 8 fence and out of any fake
scalar-generic or broader algorithm-expansion story.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/artifacts/day8-scalar-surface-preparation-design.md`
- `include/sparse_types.h`
- `src/sparse_types.c`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `tests/test_iterative.c`
- `tests/test_eigs.c`

## Day 9 Implementation Conclusions

### 1. The landed scalar owner is explicit and still truthful

The batch added one deliberate public scalar owner:

- `sparse_scalar_t`
- `SPARSE_SCALAR_BITS`
- `sparse_scalar_bits()`

in `include/sparse_types.h` / `src/sparse_types.c`.

That changes the public capability reading in one bounded way:

- iterative and eigensolver dense callback/buffer/result contracts now read
  through one named scalar owner
- the shipped contract is still explicitly real-only and `double`-backed
- later widening now has a cleaner public seam than raw repeated `double`
  spelling in every touched callback and result surface

### 2. The iterative and eigs public seams now use the same scalar owner

The strongest Day 8 target set landed directly:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

The iterative public seam now uses `sparse_scalar_t` for:

- progress and result residual fields
- tolerance fields
- residual-history buffers
- preconditioner callback vectors
- matrix-free matvec callback vectors
- one-shot, block, handle, and matrix-free RHS/result vectors

The eigs public seam now uses `sparse_scalar_t` for:

- usage examples
- `sigma`
- `tol`
- caller-owned eigenvalue/eigenvector buffers
- reported residual norm
- peak-basis byte interpretation comments

### 3. The proof stayed narrow and public-contract-local

The focused proof stayed inside the touched scalar seam:

- `tests/test_iterative.c` now proves a matrix-free CG callback and
  caller-owned vectors can use `sparse_scalar_t` directly through the public
  iterative contract
- `tests/test_eigs.c` now proves caller-owned eigensolver result buffers and
  option fields can use `sparse_scalar_t` directly through the public eigs
  contract

That is enough proof for this batch because the landing is contract-owner
preparation, not a numeric-behavior redesign.

### 4. The preserved fence stayed intact

The batch did not widen into:

- repo-wide scalar-generic conversion
- fake complex-readiness or broader precision-product claims
- `include/sparse_svd.h` / `src/sparse_svd.c`
- unsymmetric eigensolver expansion
- another width-contract or matrix-shell batch

## Validation

Because `*.c` and `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 438.00 sec`

## Exit State

Sprint 74 Day 9 exits with:

1. one explicit public scalar owner added without widening the shipped numeric
   claim
2. one bounded iterative/eigs callback and result seam converged onto that
   owner
3. one focused proof pair confirming the touched public scalar contracts
4. one fully validated capability-boundary landing inside the Sprint 74 fence
