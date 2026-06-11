# Sprint 63 Day 5: Header and Internal Landing Design

Date: 2026-06-10
Branch: sprint-63

## Purpose

Convert the Day 4 lifecycle-uniformity contract into an exact Day 6-10
touched-file plan that keeps the first LU and CSC landings narrow, reviewable,
and aligned to the real lifecycle seams.

## Minimum Viable Public Surface

Required public/header lane:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`

Conditional later follow-through only if implementation needs it:

- `README.md`
- `docs/tutorial.md`
- `docs/maintainer_guide.md`

Implication:

- Sprint 63 is implementation-first
- public-surface changes should be small truthfulness edits, not another broad
  docs reduction pass

## LU First Landing

### Required files

- `src/sparse_lu.c`
- `tests/test_integration.c`

### Likely companion file

- `include/sparse_lu.h`

### Optional support lane only if proven necessary

- `src/sparse_factor_state_internal.c`
- `src/sparse_matrix_internal.h`
- `src/sparse_matrix_state_internal.h`
- `tests/test_sparse_lu.c`

### Intended seam

- factor publication semantics
- rejection/preservation semantics on wrapper re-entry
- solve/refactor-style result-state coherence where the one-shot wrapper and
  shared lifecycle already meet

### Explicit non-goals for the first LU batch

- no `src/sparse_analysis.c` by default
- no other direct families
- no broad docs/examples/benchmark work

## Cholesky / CSC Second Landing

### Required files

- `src/sparse_cholesky.c`
- `src/sparse_chol_csc.c`
- `tests/test_integration.c`
- `tests/test_chol_csc.c`

### Likely companion file

- `include/sparse_cholesky.h`

### Conditional later seam only if the landing proves it necessary

- `src/sparse_analysis.c`
- `include/sparse_analysis.h`

### Intended seam

- CSC publication/write-back discipline
- CSC dispatch/state-retention coherence
- repeated-run solve/refactor semantics behind the existing explicit public
  lifecycle

### Explicit non-goals for the first CSC batch

- no broad rework of the Sprint 62 one-shot preservation story
- no linked-list cancellation redesign
- no LDL^T widening for cosmetic symmetry

## Proof Surface Plan

Primary proof homes:

- `tests/test_integration.c`
- `tests/test_chol_csc.c`

Secondary/optional proof home:

- `tests/test_sparse_lu.c`

Not part of the first landing by default:

- `tests/test_ldlt.c`
- `tests/test_ldlt_csc.c`
- any new bespoke lifecycle harness

## Day 6-10 Order

1. Day 6:
   - first LU lifecycle follow-through slice
2. Day 7:
   - first Cholesky/CSC repeated-run uniformity slice
3. Day 8:
   - post-landing audit
4. Day 9-10:
   - second bounded follow-through slice only if a real contradiction remains

## Explicit Non-Goals

- `src/sparse_ldlt.c`
- `src/sparse_ldlt_csc.c`
- `include/sparse_ldlt.h`
- `src/sparse_qr.c`
- `include/sparse_qr.h`
- broad `README.md` / tutorial rewrite
- benchmark-governance work
- packaging/platform work
- configuration-surface work

## Exit State

Sprint 63 now has an exact touched-file map instead of a generic lifecycle
cleanup plan:

- the first LU landing is fixed
- the second Cholesky/CSC landing is fixed
- helper/state and `sparse_analysis` widening are conditional rather than
  assumed
- the first implementation fence is narrow enough to preserve momentum and
  reviewability
