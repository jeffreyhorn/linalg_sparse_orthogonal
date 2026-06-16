# Sprint 72 Day 7: Post-Landing Audit and Rerank

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Re-rank the remaining Sprint 72 product-model seams after the Day 6 ownership
batch so the second implementation lane follows the strongest live ownership
contradiction instead of forcing a second generic matrix-shell cleanup pass.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day5-ownership-convergence-design.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day6-ownership-convergence-batch1.md`
- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `src/sparse_matrix.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`
- `tests/test_chol_csc.c`
- `tests/test_integration.c`

## Day 7 Rerank Conclusions

### 1. The Day 6 batch closed the strongest first direct-workflow contradiction

The Day 6 landing materially reduced the highest-ranked Day 3-5 seam:

- `SparseMatrix` now reads more directly as the mutable sparse construction and
  one-shot compatibility shell
- the repeated-run analysis/factor lane now reads more directly as the
  long-lived owner of reusable symbolic and factor/workspace state
- `sparse_reset_perms()` no longer leaves stale one-shot solve compatibility
  behind after recovering an identity permutation shell

That means Sprint 72 no longer needs a second matrix-shell batch just to keep
working the same first contradiction.

### 2. Another generic matrix-shell batch would now be lower-yield than the deferred compressed-path seam

The matrix shell still carries real residual pressure:

- mixed logical versus physical matrix-state semantics
- long-term compatibility-shell accumulation
- generic state-density and chronology cleanup

But those are no longer the strongest next bounded landing. They now read more
like later support or Phase 2 cleanup than the best immediate Sprint 72 move.

### 3. The strongest remaining seam is the transparent Cholesky CSC publish-back contract

The rerank now fixes the strongest remaining ownership seam to the Cholesky CSC
publication path:

- required design center:
  - `src/sparse_chol_csc.c`
- likely proof homes:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
- support only if needed:
  - `include/sparse_cholesky.h`

Why this seam is now strongest:

- it is the clearest live place where a compressed working-format factor is
  still transparently published back into the public matrix shell
- `chol_csc_writeback_to_sparse(...)` owns real product-model work:
  conversion, filtering, storage transplant, factor-state publication, and
  permutation publication
- the public Cholesky contract still documents temporary reordered working
  copies and later publish-back, so this seam is not merely internal

This makes the Cholesky CSC lane the best second Sprint 72 landing: it still
ties compressed-path ownership directly to the public one-shot shell contract.

### 4. LDL^T and LU are real later lanes, but they are weaker Day 8 targets

The rerank also clarifies why the other deferred compressed files are not the
best Day 8 center:

- `src/sparse_ldlt_csc.c` is large and important, but its strongest writeback
  seam lands in a separately-owned `sparse_ldlt_t` result struct rather than
  overwriting the caller matrix shell, so the public product-model
  contradiction is weaker than on the Cholesky side
- `src/sparse_lu_csr.c` remains important support context, but its strongest
  residual pressure is still more internal CSR conversion/update ownership than
  public matrix-shell publication ownership

### 5. The Day 8 target is now fixed

Sprint 72 Day 8 should define the second bounded ownership batch around:

- `src/sparse_chol_csc.c`

With:

- `tests/test_chol_csc.c`
- `tests/test_integration.c`

as the likely proof homes, and:

- `include/sparse_cholesky.h`

as support only if the exact design truly moves caller-facing wording.

## Exit State

Sprint 72 Day 7 closes with:

1. the Day 6 matrix-shell batch confirmed as having closed the strongest first
   direct-workflow contradiction
2. the strongest remaining seam reranked to the Cholesky CSC publish-back and
   publication contract
3. the Day 8 design target fixed to `src/sparse_chol_csc.c`
4. the likely proof-home map fixed to `tests/test_chol_csc.c` and
   `tests/test_integration.c`
