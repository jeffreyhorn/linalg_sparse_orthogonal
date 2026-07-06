# Sprint 107 Retrospective

**Sprint:** 107 - Residual Maintainability Debt & Proof-Owner Cleanup
**Duration:** 14 days (Days 1-14 landed on branch `sprint-107`)
**Status:** Complete

## Definition of Done Checklist

- [x] Sprint 107 started from Sprint 106 residual deferred debt and the updated
      Epic 10 project plan.
- [x] The residual owner queue was re-ranked from live repository evidence
      before extraction work started.
- [x] `tests/test_ldlt_csc.c` received one narrow row-adjacency proof helper
      after a dedicated boundary artifact.
- [x] `tests/test_qr.c` fixture cleanup reduced repeated small 4x3 builders
      while keeping rank, solve, residual, reconstruction, sparse-mode, and
      refinement proof intent inline.
- [x] `tests/test_iterative.c` matrix-free fixture cleanup reused existing
      tridiagonal builders and added one sequential RHS helper without moving
      convergence behavior.
- [x] `tests/test_svd.c` fixture cleanup added diagonal and rank-1 matrix
      builders while leaving rank/oracle/reconstruction interpretation inline.
- [x] `src/sparse_eigs.c` received a fresh source boundary and explicit
      no-split deferral tied to Sprint 103 comparison surfaces.
- [x] `src/sparse_matrix.c` received a central matrix shell deferral contract
      tied to Sprint 101 compressed-first compatibility rules.
- [x] no public API, install-header, Makefile, CMake, workflow, CTest
      registration, or helper-target drift was introduced.
- [x] final validation passed:
  - `make build/test_ldlt_csc && ./build/test_ldlt_csc`
  - `make build/test_qr && ./build/test_qr`
  - `make build/test_iterative && ./build/test_iterative`
  - `make build/test_svd && ./build/test_svd`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scans
- [x] residual maintainability debt is explicitly handed forward with
      prerequisites instead of being silently claimed as solved.

## What Went Well

1. **The sprint converted deferred debt into bounded work.**
   Sprint 106 left six residual owners. Sprint 107 re-ranked them, selected
   narrow test cleanup seams, and documented source-owner deferrals rather
   than treating line count as the only signal.

2. **Proof-owner cleanup stayed readable.**
   The LDLT CSC, QR, iterative, and SVD edits moved repeated setup into local
   helpers while leaving solver expectations, rank interpretation, convergence
   behavior, residual checks, and oracle comparisons visible at the test sites.

3. **The QR and iterative owners got smaller.**
   `tests/test_qr.c` dropped by 24 lines and `tests/test_iterative.c` dropped
   by 13 lines while keeping reviewed test registration unchanged.

4. **The source-owner decisions were explicit.**
   `src/sparse_eigs.c` and `src/sparse_matrix.c` were not split
   opportunistically. Both now have boundary artifacts that explain why future
   extraction needs stronger build-system, comparison, or public-behavior
   preparation.

5. **Validation matched the touched surface.**
   The branch touched `.c` test files, so Day 14 reran focused affected tests
   and the full `make format && make lint && make test` gate.

## What Didn't Go Well

1. **The largest proof owners remain large.**
   Sprint 107 reduced repeated setup, but `tests/test_ldlt_csc.c`,
   `tests/test_qr.c`, `tests/test_iterative.c`, and `tests/test_svd.c` remain
   substantial files that still need careful future boundaries.

2. **Some helpers increased local line count.**
   The LDLT CSC and SVD helpers improved naming and review locality but had
   small net line increases. That is acceptable, but it reinforces that line
   count alone is not the right maintainability metric.

3. **No implementation source extraction landed.**
   The eigensolver and matrix-shell boundaries showed that a source split would
   have crossed comparison-critical or public compatibility behavior. The
   correct outcome was deferral, but it means the source-owner debt remains.

4. **No implementation source extraction landed.**
   The sprint deliberately documented source-owner deferrals instead of moving
   code whose behavior was still tied to comparison evidence or public
   compatibility contracts.

## Final Metrics

### Validation

| Metric | Sprint 107 close state |
|---|---:|
| focused LDLT CSC test | 100 tests, 0 failures |
| focused QR test | 73 tests, 0 failures |
| focused iterative test | 80 tests, 0 failures |
| focused SVD test | 98 tests, 0 failures |
| full branch-level gate | `make format && make lint && make test` passed |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scans | passed on touched docs and `.c` files |

### Test Owner Movement

| owner | Sprint 107 baseline | final | delta |
|---|---:|---:|---:|
| `tests/test_ldlt_csc.c` | 3,884 | 3,885 | +1 |
| `tests/test_qr.c` | 3,234 | 3,210 | -24 |
| `tests/test_iterative.c` | 2,841 | 2,828 | -13 |
| `tests/test_svd.c` | 2,879 | 2,885 | +6 |

### Source Owner Close State

| owner | final lines | Sprint 107 disposition |
|---|---:|---|
| `src/sparse_eigs.c` | 1,538 | no split; future dense Jacobi or refinement split must start with comparison and build parity proof |
| `src/sparse_matrix.c` | 1,359 | no split; future movement must start with public-behavior and private-header boundary review |

### Build and Review Surfaces

| surface | Sprint 107 close state |
|---|---:|
| public/install headers changed | 0 |
| internal headers changed | 0 |
| implementation `src/*.c` files changed | 0 |
| Make/CMake source membership changes | 0 |
| GitHub workflow changes | 0 |
| reviewed test registration changes | 0 |
| new compiled helper targets | 0 |

### Sprint 107 Artifact Package

| Metric | Sprint 107 close state |
|---|---:|
| artifact files under `SPRINT_107/artifacts/` | 14 |
| planning and working-note files | 2 |
| project-plan update files | 1 |

Notes:

- scope, ranking, and proof-boundary artifacts:
  - `day1-residual-debt-intake.md`
  - `day2-residual-boundary-rerank.md`
  - `day3-ldlt-csc-proof-boundary.md`
  - `day5-qr-fixture-boundary.md`
  - `day7-iterative-fixture-boundary.md`
  - `day9-svd-proof-owner-boundary.md`
- implementation and cleanup artifacts:
  - `day4-ldlt-csc-proof-helper-extraction.md`
  - `day6-qr-fixture-cleanup.md`
  - `day8-iterative-fixture-cleanup.md`
  - `day10-svd-proof-owner-cleanup.md`
- source deferral and closeout artifacts:
  - `day11-eigensolver-source-boundary.md`
  - `day12-eigensolver-source-deferral.md`
  - `day13-central-matrix-shell-deferral-contract.md`
  - `day14-validation-metrics-closeout.md`

## Residual Deferred Debt

Most important carry-forward work:

- `tests/test_ldlt_csc.c` still contains broad direct-solver proof and oracle
  logic; future cleanup should extract only another named proof helper after a
  fresh boundary.
- `tests/test_qr.c` still has generated fixtures, tall/economy builders,
  diagonal/singleton setup, and SuiteSparse exact-RHS setup that may be
  extracted only if assertions stay visible.
- `tests/test_iterative.c` still contains convergence-sensitive setup; future
  cleanup should avoid hiding solver options, restarts, preconditioners,
  convergence results, or direct comparisons.
- `tests/test_svd.c` still has rank, oracle, reconstruction, pseudoinverse,
  low-rank, partial-SVD, and condition-number proof logic that should not move
  without a dedicated validation lane.
- `src/sparse_eigs.c` remains a source-owner risk; future extraction should
  start with dense Jacobi feasibility or a grow-m refinement boundary only
  after Make/CMake/source-list and cross-backend spectral validation are
  planned.
- `src/sparse_matrix.c` remains central API/compatibility territory; future
  extraction should begin with a public-behavior review and private-header
  dependency plan.

Still consciously constrained rather than silently solved:

- no public API or install-header change;
- no new compiled test helper target;
- no reviewed test-count change;
- no implementation source extraction;
- no broad solver-family redesign from fixture cleanup;
- no central sparse matrix shell extraction.

Not carried forward as unresolved Sprint 107 debt:

- Sprint 106 deferred debt intake;
- residual owner re-rank;
- LDLT CSC row-adjacency helper extraction;
- QR small-fixture builder cleanup;
- iterative matrix-free fixture cleanup;
- SVD diagonal and rank-1 fixture cleanup;
- eigensolver source boundary and deferral;
- matrix shell deferral contract;
- final validation and drift checks.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-residual-debt-intake.md](./artifacts/day1-residual-debt-intake.md)
- [day2-residual-boundary-rerank.md](./artifacts/day2-residual-boundary-rerank.md)
- [day3-ldlt-csc-proof-boundary.md](./artifacts/day3-ldlt-csc-proof-boundary.md)
- [day4-ldlt-csc-proof-helper-extraction.md](./artifacts/day4-ldlt-csc-proof-helper-extraction.md)
- [day5-qr-fixture-boundary.md](./artifacts/day5-qr-fixture-boundary.md)
- [day6-qr-fixture-cleanup.md](./artifacts/day6-qr-fixture-cleanup.md)
- [day7-iterative-fixture-boundary.md](./artifacts/day7-iterative-fixture-boundary.md)
- [day8-iterative-fixture-cleanup.md](./artifacts/day8-iterative-fixture-cleanup.md)
- [day9-svd-proof-owner-boundary.md](./artifacts/day9-svd-proof-owner-boundary.md)
- [day10-svd-proof-owner-cleanup.md](./artifacts/day10-svd-proof-owner-cleanup.md)
- [day11-eigensolver-source-boundary.md](./artifacts/day11-eigensolver-source-boundary.md)
- [day12-eigensolver-source-deferral.md](./artifacts/day12-eigensolver-source-deferral.md)
- [day13-central-matrix-shell-deferral-contract.md](./artifacts/day13-central-matrix-shell-deferral-contract.md)
- [day14-validation-metrics-closeout.md](./artifacts/day14-validation-metrics-closeout.md)
