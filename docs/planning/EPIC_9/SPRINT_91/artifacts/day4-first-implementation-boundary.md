# Sprint 91 Day 4: First Implementation Boundary

## Purpose

Fix one bounded first implementation fence so Sprint 91 starts with the
highest-value compressed-first seam instead of generic product churn.

## Main Result

Sprint 91 now has one explicit first implementation fence:

- required first landing:
  - `include/sparse_csr.h`
  - the matching import/construction implementation seam behind the public
    matrix-shell owner

- directly forced support surfaces only if the first landing truly needs them:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `README.md`
  - `docs/maintainer_guide.md`

- explicitly later unless the first landing truly forces movement:
  - publication/export reinterpretation on `sparse_to_csr()` / `sparse_to_csc()`
  - one-shot vs repeated-run lifecycle wording beyond the touched seam
  - `include/sparse_analysis.h`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - examples and install/export surfaces

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- Sprint 91 should start by improving how compressed inputs enter the public
  product model
- it should not begin by trying to demote or remove the linked-list shell
  broadly
- it should not reopen publication/export ownership or broad lifecycle wording
  in the first batch unless the construction/import landing actually forces it

## Deferred From The First Landing

The first batch now explicitly defers:

- broad shell removal
- family-wide direct-API rewrites
- public publication/export contract widening as a first-batch center
- repeated-run direct-owner rewrites centered on `sparse_analysis.h`
- backend, runtime, capability, or package widening
- examples, install/export, and workflow churn detached from the first product
  seam

## Exit State

- Sprint 91 has one explicit first implementation boundary.
- The first code landing is fixed to compressed-first construction/import on the
  public matrix-shell story.
- Day 5 can define the architecture contract without reopening the ranked
  first-center choice.
