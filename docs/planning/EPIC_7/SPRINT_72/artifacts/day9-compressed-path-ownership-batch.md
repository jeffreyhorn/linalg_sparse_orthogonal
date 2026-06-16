# Sprint 72 Day 9: Compressed-Path Ownership Batch

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Land the bounded second Sprint 72 implementation batch so the Cholesky CSC
publish-back seam separates internal ownership more clearly without widening
the public one-shot contract or the broader direct-family design.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day8-compressed-path-ownership-design.md`
- `src/sparse_chol_csc.c`
- `tests/test_chol_csc.c`

## Day 9 Implementation Results

### 1. The publish-back seam now has explicit internal owners

The Day 9 implementation lands in the exact Day 8 design center:

- `src/sparse_chol_csc.c`

`chol_csc_writeback_to_sparse(...)` now reads as one bounded publication
pipeline rather than one mixed helper body. The landed helper split separates:

- reorder-permutation payload copying via
  `chol_csc_copy_reorder_perm(...)`
- CSC-factor to temporary linked-list shell materialization via
  `chol_csc_materialize_sparse_factor(...)`
- caller-shell storage transplant via
  `chol_csc_transplant_materialized_factor(...)`
- factor and reorder compatibility publication via
  `chol_csc_publish_materialized_factor(...)`

This keeps the existing public one-shot behavior intact while reducing the
internal ownership blur between compressed factor materialization and caller
matrix publication.

### 2. The bounded compatibility contract stays unchanged

The Day 9 batch preserves the exact contract Sprint 72 needed to keep stable:

- successful one-shot Cholesky factorization still publishes a solve-ready
  matrix shell
- reordered one-shot attempts still publish only after success
- `used_csc_path` semantics stay unchanged
- linked-list and CSC solve-result parity stays intact
- the Day 6 matrix-shell reset rule remains valid

The batch intentionally does not widen into:

- LDL^T or LU CSR follow-through
- `SparseMatrix` redesign
- new family-local factor types
- threshold or backend-policy changes
- broader public API redesign

### 3. The new proof closes the exact writeback publication claim

The Day 9 regression lands in the planned strongest proof home:

- `tests/test_chol_csc.c`

The new test proves that a CSC factor written back through
`chol_csc_writeback_to_sparse(...)` leaves the caller shell:

- factored and solve-ready
- carrying the published reorder permutation payload
- carrying identity internal row and column permutation shells
- able to solve the original SPD system correctly

That is the right proof shape because it validates the exact family-local
publication seam without widening into unrelated public-path integration work.

## Validation

The full Day 9 gate passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity stayed `53 vs 53`
- full reviewed CMake `ctest` passed `53 / 53`
- `test_reorder_nd` remained the dominant reviewed long-tail test at
  `227.22 sec`
- `Total Test time (real) = 325.88 sec`

## Exit State

Sprint 72 Day 9 closes with:

1. one landed Cholesky CSC publish-back ownership split
2. one focused family-local regression proving the writeback-produced shell is
   published and solve-ready
3. one preserved one-shot Cholesky compatibility contract
4. one full reviewed validation pass with exact parity preserved
