# Sprint 72 Day 14: Closeout and Handoff

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Close Sprint 72 with one explicit first-phase product-model convergence
package and a ranked carry-forward queue for Sprint 73 and later Epic 7 work.

## Main Result

Sprint 72 now closes as one coherent first-phase product-model convergence
package rather than as a loose set of matrix-shell, CSC, and header changes.

The sprint hands off:

- a cleaner direct-workflow matrix-shell ownership boundary
- a cleaner Cholesky CSC publish-back ownership boundary
- cleaner public contract wording around one-shot compatibility versus
  repeated-run factor ownership
- explicit proof-owner alignment for the landed boundaries
- a fully validated close state from the strongest reviewed baseline

## First-Phase Product-Model Package Summary

### Direct-workflow ownership

- `SparseMatrix` now reads more clearly as the mutable construction and
  one-shot compatibility shell
- copied factored matrix shells now keep one-shot solve compatibility only
  until later matrix-shell mutation or `sparse_reset_perms()`
- `sparse_reset_perms()` now recovers a plain matrix shell instead of leaving
  stale reordered/factored solve compatibility behind
- repeated-run analysis/factor surfaces read more clearly as the long-lived
  reuse owner

### Compressed-path ownership

- the Cholesky CSC publish-back lane now reads as a bounded ownership pipeline
- CSC-factor materialization, caller-shell transplant, and factor/reorder
  publication are separated more clearly inside
  `chol_csc_writeback_to_sparse(...)`
- the one-shot Cholesky compatibility shell still remains solve-ready and
  behaviorally aligned with the linked-list path

### Public contract follow-through

- `include/sparse_matrix.h` now states the short-lived one-shot solve contract
  for copied factored shells directly
- `include/sparse_cholesky.h` now states directly that CSC publish-back
  preserves the same solve-ready compatibility shell while keeping long-lived
  factor ownership in `sparse_analysis.h`
- Sprint 72 did not need to reopen broader README/tutorial/example wording

### Proof/reference alignment

- `tests/test_integration.c` owns the matrix-shell reset boundary
- `tests/test_chol_csc.c` owns the Cholesky CSC publish-back ownership
  boundary
- `docs/maintainer_guide.md` now names those owners directly

### Validated close state

Sprint 72 closes from the Day 13 validated baseline:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed
- reviewed CMake parity stayed exact at `53`
- Makefile/CMake parity stayed `53 vs 53`
- reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 334.55 sec`

Focused follow-ons also stayed clean:

- `test_sparse_matrix` -> `56 / 56`
- `test_integration` -> `48 / 48`
- `test_chol_csc` -> `146 / 146`
- `example_analysis` residual stayed `4.44e-16`
- `example_basic_solve` residual stayed `0.00e+00`
- install/package regressions stayed `11 / 11` and `13 / 13`

## Ranked Carry-Forward Queue

1. Sprint 73 should continue product-model convergence from the next strongest
   compressed and matrix-state seams, without widening into a broad
   `SparseMatrix` rewrite.
2. Configuration modernization should follow only where the remaining
   env-var/default-policy seams still carry real ownership cost.
3. Capability modernization should stay led by index width, with scalar
   breadth second and unsymmetric eigensolver expansion later.
4. Backend/performance maturity should remain benchmark-governed and bounded,
   without widening product or platform claims.
5. Later permanent-surface cleanup should happen only where future
   implementation work actually moves ownership again.

## Project-Plan Recheck

The Sprint 72 section of `docs/planning/EPIC_7/PROJECT_PLAN.md` still matches
the landed package. No Sprint 72 correction is needed.

## Exit State

Sprint 72 closes from a clean validated branch state:

1. the first-phase product-model package is materially cleaner and more
   coherent
2. the Day 13 validated baseline remains the authoritative close state
3. the Sprint 73 carry-forward queue is ranked explicitly
4. no project-plan repair is required before handoff
