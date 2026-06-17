# Sprint 74 Day 5: Index / Scalar Architecture Design

Date: 2026-06-16
Branch: `sprint-74`

## Purpose

Define the bounded implementation contract for the first Sprint 74 capability
modernization landing before code edits begin.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/artifacts/day4-first-capability-boundary.md`
- `include/sparse_types.h`
- `src/sparse_types.c`
- `include/sparse_matrix.h`
- `src/sparse_matrix.c`
- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`
- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`

## Day 5 Design Conclusions

### 1. The first Sprint 74 batch is width-contract-first, not full-conversion-first

The first bounded Sprint 74 landing should not attempt to ship full
repo-wide `int64_t` mode.

It should instead converge the width lane behind one clearer contract:

- the public width surface should read as one deliberate bounded
  modernization path
- the internal allocation and overflow helpers should be the width bridge
  between `idx_t`-counted public dimensions and `size_t`-based byte math
- the matrix shell should consume that bridge more consistently on the
  highest-value touched seams

### 2. The ownership split is now explicit

Public width contract owner in the first batch:

- `include/sparse_types.h`

Internal width-bridge owner in the first batch:

- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`
- the existing checked `idx_t` <-> `size_t` helper path used by the matrix
  shell

Highest-value matrix-shell follow-through in the first batch:

- `include/sparse_matrix.h`
- `src/sparse_matrix.c`

Support-only proof and wording surfaces:

- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`
- `INSTALL.md`

### 3. The first-batch compatibility rules are fixed

The first batch must preserve:

- current shipped behavior with `idx_t == int32_t`
- current `IDX_MAX`-based width reading for downstream callers
- current allocation and overflow failure behavior on impossible or
  out-of-range counts
- current one-shot and repeated-run user-facing matrix-shell semantics

The implementation goal is therefore:

- make the width contract more explicit and less "edit typedef by hand and
  hope the rest follows"
- tighten the internal checked bridge between public `idx_t` counts and
  `size_t` allocation math
- improve the highest-value matrix-shell seams without promising a full
  alternate-width build matrix yet

### 4. The first-batch touch and non-touch sets are now explicit

Required first implementation center:

- `include/sparse_types.h`
- `src/sparse_types.c`
- `include/sparse_matrix.h`
- `src/sparse_matrix.c`

Support only if the implementation truly forces it:

- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`
- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`
- `INSTALL.md`

Explicit non-touch set:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_svd.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_svd.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- package/install workflow files beyond truthful width-contract wording
- benchmark-governance or backend/performance files
- broad product-model or configuration follow-through

## Exit State

Sprint 74 Day 5 closes with:

1. one explicit width-contract-first design for the first capability lane
2. one preserved compatibility checklist
3. one exact first-batch touch set
4. one explicit non-touch set before Day 6 implementation begins
