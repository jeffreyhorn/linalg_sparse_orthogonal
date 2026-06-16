# Sprint 72 Day 10: Public Contract and Example Adoption Design

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Define the exact public-header, doc, and example follow-through required by
the Sprint 72 Day 6 and Day 9 ownership landings, while explicitly keeping
already-coherent public surfaces out of scope.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day6-ownership-convergence-batch1.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day9-compressed-path-ownership-batch.md`
- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `README.md`
- `docs/tutorial.md`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

## Day 10 Design Conclusions

### 1. The strongest remaining public contract center is header-local

The Day 6 and Day 9 implementation batches changed ownership mechanics that
are primarily expressed in the public headers, not in the broad front-door or
tutorial surfaces.

The strongest remaining follow-through center is therefore:

- `include/sparse_matrix.h`
- `include/sparse_cholesky.h`

Support-first, but likely non-moving unless exact wording proves necessary:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_ldlt.h`

This keeps Sprint 72 aligned with the Day 6 and Day 9 implementation lanes
instead of reopening the wider Sprint 71 docs-cleanup surface.

### 2. README, tutorial, and the shipped direct-workflow examples are already broadly coherent

The reread against the current public adoption surfaces shows that the broad
story already matches the landed ownership split:

- one-shot direct APIs remain first-class entry points
- callers still use a fresh matrix or fresh `sparse_copy()` when they need the
  original coefficient view later
- the explicit repeated-run direct lifecycle remains the clearer owner of
  reusable symbolic and factor/workspace state
- `example_basic_solve.c` still demonstrates the one-shot copy-then-factor
  discipline
- `example_analysis.c` still demonstrates the same-pattern repeated-run lane
- `docs/tutorial.md` already keeps the one-shot versus repeated-run split
  readable

That means Sprint 72 should avoid README/tutorial/example churn unless a
specific contradiction appears during the final Day 11 pass.

### 3. The exact Day 11 follow-through fence is now fixed

Required Day 11 follow-through center:

- `include/sparse_matrix.h`
- `include/sparse_cholesky.h`

Support only if the exact wording really forces it:

- `README.md`
- `docs/tutorial.md`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

Likely non-moving support:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_ldlt.h`

Explicit non-touch set:

- implementation `src/` files
- proof-owner tests
- capability/type surfaces
- packaging/platform/install/workflow surfaces

### 4. The preserved truthfulness checklist is explicit

Any Day 11 follow-through must preserve:

- `SparseMatrix` as the mutable construction and one-shot compatibility shell
- the explicit repeated-run direct lifecycle as the clearer long-lived owner
  of reusable symbolic and factor/workspace state
- one-shot direct APIs as first-class peer entry points
- the copy-first discipline for one-shot factorization when callers still need
  the original coefficient view later
- Cholesky CSC writeback as an internal publish-back path that still returns a
  standard solve-ready matrix shell

It must avoid:

- reopening Sprint 71 front-door cleanup
- generic tutorial or example rewrites
- capability/platform/install spill
- new ownership claims that the shipped APIs do not actually guarantee

## Exit State

Sprint 72 Day 10 closes with:

1. one exact Day 11 follow-through fence centered on bounded
   direct-workflow contract wording
2. one explicit separation between actually-required follow-through and
   already-coherent public adoption surfaces
3. one preserved truthfulness checklist that keeps Sprint 72 out of generic
   docs spill
