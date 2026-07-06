# Sprint 108 Retrospective

**Sprint:** 108 - Residual Proof-Owner & Source Boundary Follow-Through
**Duration:** 14 days (Days 1-14 landed on branch `sprint-108`)
**Status:** Complete

## Definition of Done Checklist

- [x] Sprint 108 started from Sprint 107 residual deferred debt and excluded
      completed Sprint 107 work from duplicate cleanup.
- [x] remaining proof owners were re-ranked from live repository evidence
      before implementation work started.
- [x] `tests/test_ldlt_csc.c` received one bounded residual assertion helper
      after a dedicated LDLT CSC oracle-boundary artifact.
- [x] `tests/test_qr.c` received one narrow tall diagonal-dominant fixture
      helper while preserving rank, solve, reconstruction, sparse-mode, and
      refinement proof logic at call sites.
- [x] `tests/test_iterative.c` received one bounded diagonal-preconditioner
      fixture helper while preserving solver options, restart values,
      preconditioner semantics, convergence assertions, and residual checks at
      call sites.
- [x] `tests/test_svd.c` received one full-SVD fixture helper for approved
      call sites only, while keeping singular-triplet, orthogonality,
      reconstruction, and economy/full parity proof logic visible.
- [x] `src/sparse_eigs.c` received an updated feasibility boundary and
      closeout handoff without landing a risky source split.
- [x] `src/sparse_matrix.c` received a public-behavior and private-dependency
      review before any future shell extraction.
- [x] no public API, install-header, implementation source, Makefile, CMake,
      source-list, helper-target, or CTest registration drift was introduced.
- [x] final validation passed:
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scan over `docs/planning/EPIC_10/SPRINT_108`
- [x] residual source-boundary and proof-owner work is explicitly handed
      forward with prerequisites.

## What Went Well

1. **The sprint stayed boundary-first.**
   Day 1 and Day 2 separated completed Sprint 107 work from Sprint 108
   residuals, then ranked the remaining cleanup by reviewability,
   failure-localization value, validation cost, and dependency order.

2. **Proof-owner cleanup remained narrow.**
   The LDLT CSC, QR, iterative, and SVD changes moved repeated setup or
   assertion ceremony into local helpers without hiding solver behavior,
   tolerance values, residual checks, rank interpretation, or oracle
   comparisons.

3. **Source-boundary work did not overclaim.**
   The eigensolver review identified `s21_dense_sym_jacobi` as a plausible
   future private source seam, but kept `src/sparse_eigs.c` intact because
   even that movement crosses thick-restart and LOBPCG validation. The matrix
   shell review likewise kept `src/sparse_matrix.c` intact because it owns
   public lifecycle, mutation, permutation, factor-state, I/O, and solver
   compatibility behavior.

4. **Public and reviewed support surfaces stayed stable.**
   Sprint 108 changed no public headers, installed headers, implementation
   sources, build membership, helper targets, or CTest registration. This kept
   the sprint focused on proof-owner cleanup and future extraction contracts.

5. **Validation matched the touched surface.**
   Because four `.c` test files changed, Day 14 ran the full
   `make format && make lint && make test` gate and follow-up diff hygiene
   checks.

## What Didn't Go Well

1. **Some proof-owner files still grew.**
   `tests/test_ldlt_csc.c` and `tests/test_iterative.c` both grew slightly
   because the helpers added failure cleanup and naming clarity. The tradeoff
   is acceptable, but it reinforces that this work is about review locality
   and proof clarity rather than raw line-count reduction.

2. **The largest test owners remain large.**
   Sprint 108 improved bounded local seams, but `tests/test_ldlt_csc.c`,
   `tests/test_qr.c`, `tests/test_iterative.c`, and `tests/test_svd.c` remain
   large proof owners that still need future one-helper-family cleanup passes.

3. **No implementation source extraction landed.**
   The eigensolver and matrix-shell reviews both found real future candidates,
   but neither was low-risk enough to move in Sprint 108 without broad
   validation and source-list follow-through.

4. **Validation was necessarily expensive.**
   The branch only changed tests and planning artifacts, but because `.c`
   files changed, the correct closeout still required the full formatting,
   lint, and test gate.

## Final Metrics

### Validation

| Metric | Sprint 108 close state |
|---|---:|
| full branch-level gate | `make format && make lint && make test` passed |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on `docs/planning/EPIC_10/SPRINT_108` |
| public/install header drift | 0 files |
| implementation `src/*.c` drift | 0 files |
| Make/CMake/source-list drift | 0 files |
| helper-target drift | 0 targets |
| CTest registration drift | 0 tests |

### Test Owner Movement

| owner | Sprint 108 baseline | final | delta |
|---|---:|---:|---:|
| `tests/test_ldlt_csc.c` | 3,887 | 3,896 | +9 |
| `tests/test_qr.c` | 3,213 | 3,210 | -3 |
| `tests/test_iterative.c` | 2,828 | 2,849 | +21 |
| `tests/test_svd.c` | 2,897 | 2,896 | -1 |

### Source Owner Close State

| owner | final lines | Sprint 108 disposition |
|---|---:|---|
| `src/sparse_eigs.c` | 1,538 | no split; future dense Jacobi movement must start with Make/CMake/manifest parity and cross-backend spectral validation |
| `src/sparse_matrix.c` | 1,359 | no split; future movement must start with one public-behavior owner, private-header dependency plan, and focused tests |

### Build and Review Surfaces

| surface | Sprint 108 close state |
|---|---:|
| public/install headers changed | 0 |
| internal headers changed | 0 |
| implementation `src/*.c` files changed | 0 |
| Make/CMake source membership changes | 0 |
| source-list manifest changes | 0 |
| GitHub workflow changes | 0 |
| reviewed test registration changes | 0 |
| new compiled helper targets | 0 |

### Sprint 108 Artifact Package

| Metric | Sprint 108 close state |
|---|---:|
| artifact files under `SPRINT_108/artifacts/` | 14 |
| planning and working-note files | 2 |
| retrospective files | 1 |

Notes:

- scope, ranking, and boundary artifacts:
  - `day1-carry-forward-intake.md`
  - `day2-residual-proof-owner-boundary-refresh.md`
  - `day3-ldlt-csc-oracle-boundary.md`
  - `day5-qr-residual-fixture-boundary.md`
  - `day7-iterative-convergence-boundary.md`
  - `day9-svd-validation-lane-boundary.md`
- implementation and cleanup artifacts:
  - `day4-ldlt-csc-helper-follow-through.md`
  - `day6-qr-fixture-follow-through.md`
  - `day8-iterative-convergence-cleanup.md`
  - `day10-svd-oracle-reconstruction-cleanup.md`
- source-boundary and closeout artifacts:
  - `day11-eigensolver-source-feasibility-boundary.md`
  - `day12-eigensolver-feasibility-closeout.md`
  - `day13-matrix-shell-public-behavior-review.md`
  - `day14-validation-metrics-closeout.md`

## Residual Deferred Debt

Most important carry-forward work:

- `src/sparse_eigs.c` remains a source-owner risk. The first plausible future
  movement is a private dense Jacobi helper owner for `s21_dense_sym_jacobi`,
  gated by Make/CMake/manifest parity, `make source-list-check`, and focused
  `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, and
  `test_sprint29_integration` validation.
- eigensolver grow-m refinement, dispatch/defaults, handle/workspace glue, and
  shared Lanczos kernels remain behavior-sensitive and should not move without
  stronger residual, shift-invert, backend-selection, handle-workspace, and
  cross-backend numerical evidence.
- `src/sparse_matrix.c` remains central public matrix-shell territory. Future
  extraction must start with one named public-behavior owner, a private-header
  dependency plan, source-list parity, focused public behavior tests, and
  solver smoke tests for touched semantics.
- `tests/test_ldlt_csc.c`, `tests/test_qr.c`, `tests/test_iterative.c`, and
  `tests/test_svd.c` remain large proof owners. Future cleanup should continue
  one helper family at a time with proof logic visible at call sites.

Still consciously constrained rather than silently solved:

- no public API or install-header change;
- no implementation source extraction;
- no new compiled test helper target;
- no reviewed test-count change;
- no broad eigensolver or matrix-shell redesign;
- no broad solver-family rewrite from fixture cleanup.

Not carried forward as unresolved Sprint 108 debt:

- Sprint 107 residual intake and exclusion list;
- live residual proof-owner re-rank;
- LDLT CSC residual assertion helper follow-through;
- QR tall diagonal-dominant fixture helper;
- iterative diagonal-preconditioner fixture helper;
- SVD full-UV fixture helper;
- eigensolver feasibility boundary and closeout handoff;
- matrix-shell public-behavior review;
- final validation, metrics, and drift checks.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-carry-forward-intake.md](./artifacts/day1-carry-forward-intake.md)
- [day2-residual-proof-owner-boundary-refresh.md](./artifacts/day2-residual-proof-owner-boundary-refresh.md)
- [day3-ldlt-csc-oracle-boundary.md](./artifacts/day3-ldlt-csc-oracle-boundary.md)
- [day4-ldlt-csc-helper-follow-through.md](./artifacts/day4-ldlt-csc-helper-follow-through.md)
- [day5-qr-residual-fixture-boundary.md](./artifacts/day5-qr-residual-fixture-boundary.md)
- [day6-qr-fixture-follow-through.md](./artifacts/day6-qr-fixture-follow-through.md)
- [day7-iterative-convergence-boundary.md](./artifacts/day7-iterative-convergence-boundary.md)
- [day8-iterative-convergence-cleanup.md](./artifacts/day8-iterative-convergence-cleanup.md)
- [day9-svd-validation-lane-boundary.md](./artifacts/day9-svd-validation-lane-boundary.md)
- [day10-svd-oracle-reconstruction-cleanup.md](./artifacts/day10-svd-oracle-reconstruction-cleanup.md)
- [day11-eigensolver-source-feasibility-boundary.md](./artifacts/day11-eigensolver-source-feasibility-boundary.md)
- [day12-eigensolver-feasibility-closeout.md](./artifacts/day12-eigensolver-feasibility-closeout.md)
- [day13-matrix-shell-public-behavior-review.md](./artifacts/day13-matrix-shell-public-behavior-review.md)
- [day14-validation-metrics-closeout.md](./artifacts/day14-validation-metrics-closeout.md)
