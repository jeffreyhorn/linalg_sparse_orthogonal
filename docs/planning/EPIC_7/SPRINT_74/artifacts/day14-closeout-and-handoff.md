# Sprint 74 Day 14: Closeout and Handoff

Date: 2026-06-16
Branch: `sprint-74`

## Purpose

Close Sprint 74 from the Day 13 validated baseline with one explicit
first-phase capability-modernization package and a ranked carry-forward queue
for Sprint 75 and beyond.

## Sprint 74 Package Summary

Sprint 74 now closes as one coherent first-phase capability package across:

- capability audit and rerank
- bounded index/scalar architecture contract
- first index-width modernization seam
- bounded scalar-surface preparation
- docs/proof follow-through
- validated close state

## What Landed

### 1. The width ceiling now has one deliberate compile-time seam

- `include/sparse_types.h` / `src/sparse_types.c` now define:
  - `SPARSE_IDX_BITS`
  - `SPARSE_PRIDX`
  - `SPARSE_SCNIDX`
  - `sparse_idx_bits()`
- reviewed builds still default to the shipped `idx_t == int32_t` lane
- wider indices now read as one bounded compile-time seam instead of a manual
  typedef-edit story

### 2. The matrix shell now consumes the width bridge more coherently

- `src/sparse_alloc_internal.*` is now the clearer checked bridge between
  `idx_t` counts and `size_t` byte math
- `include/sparse_matrix.h` / `src/sparse_matrix.c` now use that bridge more
  consistently for allocation, byte sizing, memory-usage accounting, and
  Matrix Market formatting on the touched paths
- `tests/test_sparse_matrix.c` owns the focused width-contract proof

### 3. The strongest public real-only seams now point at one scalar owner

- `include/sparse_types.h` / `src/sparse_types.c` now define:
  - `sparse_scalar_t`
  - `SPARSE_SCALAR_BITS`
  - `sparse_scalar_bits()`
- `include/sparse_iterative.h` now routes the touched iterative public
  callback/result and dense-buffer contracts through `sparse_scalar_t`
- `include/sparse_eigs.h` now does the same for the touched eigensolver public
  callback/result seam
- `tests/test_iterative.c` and `tests/test_eigs.c` own the focused scalar-seam
  proof

### 4. The public and policy docs now tell the smaller truthful story

- `README.md` now says directly:
  - reviewed builds still default to the 32-bit `idx_t` lane
  - wider indices are the bounded compile-time `SPARSE_IDX_BITS=64` seam
  - shipped scalar support is still real-only
  - `sparse_scalar_t` is bounded public preparation, not a broader generic or
    complex-support claim
- `docs/maintainer_guide.md` now owns the narrower Sprint 74 capability
  interpretation and names the focused proof owners

## Preserved Fence

Sprint 74 preserved the intended Epic 7 boundary:

- no repo-wide 64-bit conversion claim
- no repo-wide scalar-type genericity claim
- no fake complex-readiness or broader precision-product claim
- no unsymmetric eigensolver expansion in the first capability lane
- no widened reviewed/install/platform story beyond maintained evidence

## Validated Close State

Sprint 74 closes from the Day 13 validated baseline:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- reviewed CMake parity `53`
- Makefile/CMake parity `53 vs 53`
- reviewed CMake `ctest` `53 / 53`
- `Total Test time (real) = 372.56 sec`

Representative retained anchors:

- `test_sparse_matrix`: `test_idx_width_contract`
- `test_iterative`: `test_iterative_public_scalar_alias`
- `test_eigs`: `test_eigs_public_scalar_alias`
- `example_analysis`: residual `4.44e-16`
- `example_basic_solve`: residual `0.00e+00`
- `bench_refactor_csc nos4`: `speedup_refactor = 1.41`
- `bench_chol_csc nos4`: `speedup_csc = 0.65`, `speedup_csc_sn = 0.69`
- install/package regressions: installed `pkg-config` version `2.2.0`

## Ranked Carry-Forward Queue

1. scalar-breadth modernization beyond the landed public seam, starting where
   the real-only contract is still strongest and most reused
2. later algorithm-family widening only where the capability contract and proof
   justify it, rather than as a broad “more algorithms” batch
3. backend/performance maturity only where benchmark-governed ownership seams
   justify movement
4. later permanent-surface cleanup only after the higher-value capability lanes
   move

## Project Plan Recheck

`docs/planning/EPIC_7/PROJECT_PLAN.md` does not need a Sprint 74 correction.
The landed work still matches the sprint contract.

## Exit State

Sprint 74 closes with:

1. one real compile-time width seam instead of a permanent manual width ceiling
2. one clearer public scalar owner for the strongest touched real-only seams
3. one truthful docs/proof interpretation of the bounded capability landing
4. one validated handoff package for Sprint 75
