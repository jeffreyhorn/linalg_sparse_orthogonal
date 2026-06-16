# Sprint 72 Day 11: Public Contract and Example Adoption Batch

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Land the exact public-facing follow-through still required by the Sprint 72
Day 6 and Day 9 ownership batches, while explicitly keeping already-coherent
README, tutorial, and example surfaces out of scope.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day10-public-contract-and-example-adoption-design.md`
- `include/sparse_matrix.h`
- `include/sparse_cholesky.h`
- `README.md`
- `docs/tutorial.md`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

## Day 11 Implementation Results

### 1. The batch stayed exactly header-local

The landed Day 11 follow-through touched only:

- `include/sparse_matrix.h`
- `include/sparse_cholesky.h`

That is the right bounded result because the Day 6 and Day 9 implementation
changes were ownership clarifications in the matrix shell and Cholesky CSC
publish-back seam, not a new broad adoption story.

### 2. `SparseMatrix` copy semantics now state the reset boundary directly

The Day 11 wording in `include/sparse_matrix.h` now makes the Day 6 matrix-
shell rule explicit:

- copying a factored matrix preserves the one-shot matrix-shell solve contract
- later matrix-shell mutation or `sparse_reset_perms()` drops that
  compatibility again

This keeps the one-shot copy discipline truthful without overstating copied
matrix shells as long-lived factor owners.

### 3. Cholesky CSC publish-back now states the Day 9 ownership boundary directly

The Day 11 wording in `include/sparse_cholesky.h` now makes the Day 9 CSC
publish-back rule explicit:

- the CSC lane returns the same solve-ready `SparseMatrix` compatibility shell
  as the linked-list path
- that internal publish-back step does not move long-lived factor ownership
  away from the explicit repeated-run direct lifecycle in `sparse_analysis.h`

That is the smallest useful clarification of the transparent CSC lane.

### 4. README, tutorial, and shipped examples did not need edits

The reread confirmed that these support surfaces already match the landed
ownership story and therefore remained untouched:

- `README.md`
- `docs/tutorial.md`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

This preserved the Day 10 non-goal fence against generic documentation spill
and unnecessary example churn.

## Validation

The required Day 11 gate passed:

- `make format`
- `make lint`
- `make test`

Touched-surface raw `wc -l` counts:

- `include/sparse_matrix.h` = `604`
- `include/sparse_cholesky.h` = `220`

## Exit State

Sprint 72 Day 11 closes with:

1. one bounded public-header follow-through batch
2. one explicit restatement of the Day 6 matrix-shell reset rule
3. one explicit restatement of the Day 9 Cholesky CSC publish-back ownership
   rule
4. one confirmed non-move of the broader README/tutorial/example adoption
   surfaces
5. one clean Day 11 validation pass
