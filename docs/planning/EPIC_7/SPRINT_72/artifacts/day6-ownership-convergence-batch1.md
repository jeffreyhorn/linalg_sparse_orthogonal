# Sprint 72 Day 6: Ownership Convergence Batch 1

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Land the first bounded Sprint 72 implementation batch so the public
direct-workflow ownership split is explicit and the matrix shell no longer
retains stale one-shot solve compatibility after permutation reset.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day5-ownership-convergence-design.md`
- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `src/sparse_matrix.c`
- `tests/test_integration.c`

## Landed Batch

### 1. Public ownership wording now states the intended product-model split directly

The Day 6 batch updates the first public ownership surfaces so they read more
coherently:

- `include/sparse_matrix.h` now says `SparseMatrix` is the mutable sparse
  construction and one-shot direct-workflow compatibility shell rather than the
  long-lived repeated-run owner
- `include/sparse_analysis.h` now states more directly that the shared
  repeated-run lifecycle is the clearer long-lived owner of reusable symbolic
  and factor/workspace state
- `include/sparse_lu.h`, `include/sparse_cholesky.h`, and
  `include/sparse_ldlt.h` now point back to that repeated-run lifecycle as the
  clearer reuse owner while preserving the supported one-shot matrix-shell lane

This did not redesign the public API. It clarified the ownership reading that
Sprint 72 wants callers and maintainers to carry forward.

### 2. `sparse_reset_perms()` now recovers a plain matrix shell instead of leaving stale one-shot solve compatibility behind

The Day 6 mechanics fix lands in `src/sparse_matrix.c`.

Before the batch:

- a copied matrix that had gone through one-shot direct factorization could
  keep factor / reorder compatibility state
- `sparse_reset_perms()` restored visible row and column permutation shells to
  identity
- but the old one-shot compatibility could still survive even though the
  permutation shell it depended on had been rewritten

After the batch:

- `sparse_reset_perms()` detects when the matrix shell carries:
  - a stored reorder permutation, or
  - non-identity row / column permutation shells
- after restoring row and column permutation arrays to identity, it clears the
  reorder permutation compatibility state
- when the shell had one-shot factored/reordered compatibility tied to the old
  permutation shell, it clears that factor compatibility as well

The result is the intended Sprint 72 reading:

- permutation reset means recovery of a plain matrix shell
- the one-shot shell must be factorized again before later solve calls
- the repeated-run analysis/factor lane remains the clearer owner of reusable
  long-lived state

### 3. The new regression proves the bounded ownership rule on the live public seam

The batch adds a focused integration regression in `tests/test_integration.c`.

The new test proves:

- a copied matrix can still factor and solve through the one-shot LU lane
- the factored shell initially carries a non-identity row permutation
- after `sparse_reset_perms()`:
  - row and column permutation shells return to identity
  - the old one-shot LU solve contract is rejected with `SPARSE_ERR_BADARG`

That keeps the Day 6 proof tight and public-facing rather than relying on
internal-only helper behavior.

## Preserved Fence

The first ownership batch stayed inside the Day 5 non-touch set:

- no CSC / CSR conversion redesign
- no compressed-path publication or writeback redesign
- no new family-local factor types
- no capability, packaging, or platform contract changes
- no broad proof-surface redesign

Touched implementation/proof surfaces were limited to:

- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `src/sparse_matrix.c`
- `tests/test_integration.c`

## Validation

The full Day 6 validation gate passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity stayed `53 vs 53`
- full reviewed CMake `ctest` passed `53 / 53`
- `test_reorder_nd` remained the dominant reviewed long-tail test at
  `229.99 sec`
- `Total Test time (real) = 324.07 sec`

## Exit State

Sprint 72 Day 6 closes with:

1. one landed ownership wording batch across the first public direct-workflow
   surfaces
2. one bounded matrix-shell state fix that drops stale permuted one-shot solve
   compatibility on permutation reset
3. one focused integration regression proving that rule on the live public seam
4. one full reviewed validation pass with exact Makefile/CMake parity preserved
