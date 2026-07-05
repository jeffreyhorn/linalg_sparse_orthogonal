# Day 14 Validation, Metrics & Closeout

## Purpose

Day 14 closes Sprint 107 by validating all touched surfaces, publishing
maintainability metrics, checking for public API/install-header/test-count
drift, and handing unresolved residual debt to future work with explicit
rationale.

## Touched Surfaces

Sprint 107 touched:

- planning documentation:
  - `docs/planning/EPIC_10/PROJECT_PLAN.md`
  - `docs/planning/EPIC_10/SPRINT_107/PLAN.md`
  - `docs/planning/EPIC_10/SPRINT_107/WORKING_NOTES.md`
  - Day 1 through Day 14 Sprint 107 artifacts
- test proof owners:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_iterative.c`
  - `tests/test_svd.c`

Sprint 107 did not touch:

- public headers;
- install headers;
- implementation sources under `src/`;
- Makefile or CMake source membership;
- GitHub workflow files;
- CTest registration files.

## Implemented Cleanup Summary

| Area | Outcome |
|---|---|
| Epic 10 project plan | Integrated Sprint 106 deferred debt into Sprint 107 and shifted later sprint scope while preserving the 168-hour sprint cap. |
| LDLT CSC proof owner | Added one local row-adjacency assertion helper and kept direct CSC proof intent visible. |
| QR proof owner | Added local 4x3 fixture builders for repeated banded, duplicate-column, and near-duplicate-column setup. |
| Iterative proof owner | Reused tridiagonal builders and added a local sequential RHS helper for matrix-free cases. |
| SVD proof owner | Added local diagonal and rank-1 row-progression fixture builders while leaving rank/oracle interpretation inline. |
| Eigensolver source owner | Documented a no-split source boundary and deferral path. |
| Central matrix shell owner | Documented a non-extraction contract and future split prerequisites. |

## Metrics

### Tracked File Line Deltas

| File | Before | After | Delta |
|---|---:|---:|---:|
| `docs/planning/EPIC_10/PROJECT_PLAN.md` | 380 | 420 | +40 |
| `tests/test_ldlt_csc.c` | 3,884 | 3,885 | +1 |
| `tests/test_qr.c` | 3,234 | 3,210 | -24 |
| `tests/test_iterative.c` | 2,841 | 2,828 | -13 |
| `tests/test_svd.c` | 2,879 | 2,885 | +6 |

### Current Residual Owner Sizes

| Owner | Current Lines | Sprint 107 Disposition |
|---|---:|---|
| `tests/test_ldlt_csc.c` | 3,885 | One narrow helper extracted; broader solve/oracle proof remains inline. |
| `tests/test_qr.c` | 3,210 | Repeated fixture builders reduced; solve/reconstruction proof remains inline. |
| `tests/test_iterative.c` | 2,828 | Matrix/RHS setup reduced; convergence behavior remains inline. |
| `tests/test_svd.c` | 2,885 | Diagonal/rank-1 fixtures reduced; rank/oracle interpretation remains inline. |
| `src/sparse_eigs.c` | 1,538 | No source split; future split gated by comparison and build-system proof. |
| `src/sparse_matrix.c` | 1,359 | No source split; future split gated by public behavior review. |

### Sprint 107 Artifact Set

Sprint 107 produced:

- one 14-day plan;
- one working-notes file;
- fourteen day artifacts;
- one project-plan update that folds Sprint 106 residual debt into Sprint 107
  and later Epic 10 sequencing.

## Drift Checks

| Surface | Result |
|---|---|
| Public API headers | No tracked diff under `include/`. |
| Internal headers | No tracked diff under `src/*.h`. |
| Install-header surface | No header or install metadata changed. |
| Make/CMake source membership | No tracked diff in `Makefile`, `CMakeLists.txt`, or CMake files. |
| GitHub workflows | No tracked diff in `.github`. |
| CTest registration | No `RUN_TEST` registration lines added or removed in touched test files. |
| Helper targets | No new compiled helper target added. |

## Validation Plan

Because Sprint 107 changed `.c` test files, the closeout validation must run:

```sh
make build/test_ldlt_csc && ./build/test_ldlt_csc
make build/test_qr && ./build/test_qr
make build/test_iterative && ./build/test_iterative
make build/test_svd && ./build/test_svd
make format && make lint && make test
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md tests/test_ldlt_csc.c tests/test_qr.c tests/test_iterative.c tests/test_svd.c
```

## Validation Results

- `make build/test_ldlt_csc && ./build/test_ldlt_csc`: passed; focused
  suite ran 100 tests with 0 failures.
- `make build/test_qr && ./build/test_qr`: passed; focused suite ran 73
  tests with 0 failures.
- `make build/test_iterative && ./build/test_iterative`: passed; focused
  suite ran 80 tests with 0 failures.
- `make build/test_svd && ./build/test_svd`: passed; focused suite ran 98
  tests with 0 failures.
- `make format && make lint && make test`: passed.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md tests/test_ldlt_csc.c tests/test_qr.c tests/test_iterative.c tests/test_svd.c`:
  passed; no matches.

## Residual Handoff

Sprint 107 intentionally leaves the following work for future sprints:

- Any broader LDLT CSC proof-owner extraction beyond the selected
  row-adjacency helper.
- QR assertion/proof movement beyond fixture setup.
- Iterative solver convergence helper movement beyond matrix/RHS setup.
- SVD rank, oracle, reconstruction, pseudoinverse, and condition-number proof
  helper movement.
- `src/sparse_eigs.c` source extraction, starting with dense Jacobi only if a
  future sprint prepares source-list/CMake parity and cross-backend spectral
  validation.
- `src/sparse_matrix.c` source extraction, starting only after a
  public-behavior boundary review and private-header dependency plan.

## Closeout Position

Sprint 107 completed the residual maintainability debt cleanup that was safe
within the sprint constraints. It reduced repeated fixture setup in four large
test owners and converted two source-owner risks into explicit future
contracts rather than forcing risky extraction. No public API, install-header,
build-system, or reviewed test registration change was introduced.
