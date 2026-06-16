# Sprint 72 Day 8: Compressed-Path Ownership Design

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Define the bounded second Sprint 72 implementation batch around the strongest
remaining Cholesky CSC publish-back seam so Day 9 can reduce compressed-path
ownership blur without widening into a broader backend redesign.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day7-post-landing-audit-and-rerank.md`
- `src/sparse_chol_csc.c`
- `include/sparse_cholesky.h`
- `tests/test_chol_csc.c`
- `tests/test_integration.c`

## Day 8 Design Conclusions

### 1. The second batch should target publish-back ownership, not generic CSC internals

The strongest remaining Sprint 72 seam is now fixed to the transparent
Cholesky CSC publish-back path in `chol_csc_writeback_to_sparse(...)`.

That helper currently bundles together:

- caller-shell precondition validation
- permutation payload copying
- CSC-factor to temporary-shell materialization
- caller-shell storage transplant
- factor and reorder compatibility publication

That is the exact remaining ownership bundle Sprint 72 should shrink next.

### 2. The right cleanup is a publication-phase separation

The best bounded Day 9 design is to make the publish-back phases read more
cleanly:

- phase 1: materialize a temporary linked-list shell from the CSC factor
- phase 2: transplant that materialized shell into the caller matrix
- phase 3: publish factor and reorder compatibility state onto the caller shell

This preserves the public one-shot contract while reducing the internal blur
between compressed working ownership and matrix-shell publication ownership.

### 3. The exact touched-file fence is fixed

Required design center:

- `src/sparse_chol_csc.c`

Likely proof homes:

- `tests/test_chol_csc.c`
- `tests/test_integration.c`

Support only if the exact code batch truly moves caller-facing wording:

- `include/sparse_cholesky.h`

Explicit non-touch set:

- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`
- `src/sparse_matrix.c`
- `include/sparse_matrix.h`
- capability/type surfaces
- packaging/platform/docs truth surfaces
- broad benchmark/example spill

### 4. The proof-home map stays narrow

The second batch should keep proof in the existing strongest homes:

- `tests/test_chol_csc.c` for family-local writeback preconditions, round-trip
  expectations, and CSC versus linked-list path behavior
- `tests/test_integration.c` for public-path parity against the shared
  repeated-run analysis/factor lane and the exposed dispatch telemetry

No new proof owner should be introduced unless the exact Day 9 mechanics force
it.

### 5. The preserved compatibility checklist is explicit

Day 9 must preserve:

- successful one-shot Cholesky factorization still publishing a solve-ready
  matrix shell
- reordered one-shot attempts publishing only after success
- current `used_csc_path` semantics
- linked-list and CSC solve-result parity
- the Day 6 matrix-shell reset rule for stale one-shot compatibility

It must not widen into:

- public API redesign
- backend-threshold or dispatch-policy changes
- new family-local factor types
- broad compressed-path cleanup across every direct family

## Exit State

Sprint 72 Day 8 closes with:

1. one exact second-batch design centered on the Cholesky CSC publish-back seam
2. one fixed Day 9 touched-file fence
3. one narrow proof-home map anchored to `tests/test_chol_csc.c` and
   `tests/test_integration.c`
4. one explicit compatibility checklist that keeps the batch bounded to
   compressed-path publication ownership cleanup
