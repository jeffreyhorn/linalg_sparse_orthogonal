# Day 2 Residual Proof-Owner Boundary Refresh

## Purpose

Day 2 refreshes the Sprint 108 proof-owner boundary from the live tree after
Sprint 107 merged. The goal is to rank the remaining cleanup work without
duplicating completed Sprint 107 helper extractions and without hiding proof
assertions behind overly broad fixture or oracle helpers.

## Live Inventory

| Owner | Lines | Static Functions | Tests | Assertions | Sparse Creates | Sparse Inserts |
|---|---:|---:|---:|---:|---:|---:|
| `tests/test_ldlt_csc.c` | 3,887 | 132 | 100 | 489 | 56 | 170 |
| `tests/test_qr.c` | 3,213 | 86 | 73 | 337 | 56 | 224 |
| `tests/test_iterative.c` | 2,828 | 89 | 77 | 316 | 26 | 103 |
| `tests/test_svd.c` | 2,897 | 81 | 75 | 363 | 55 | 160 |

The line counts and assertion/setup counts show that all four proof owners
still justify careful follow-through, but they do not justify broad extraction.
Each owner has dense proof logic where fixture construction and behavioral
assertions are intentionally close together.

## Completed Sprint 107 Work Excluded

Sprint 108 must not repeat these completed changes:

- `tests/test_ldlt_csc.c`: row-adjacency exact-set helper extraction and
  duplicate-entry assertion refinement.
- `tests/test_qr.c`: small 4x3 banded, duplicate-column, and near-duplicate
  fixture builders with checked insert handling.
- `tests/test_iterative.c`: matrix-free tridiagonal builder reuse and
  sequential RHS helper.
- `tests/test_svd.c`: diagonal matrix and rank-1 row-progression fixture
  builders with checked insert handling.
- `src/sparse_eigs.c`: Sprint 107 boundary and no-split deferral record.
- `src/sparse_matrix.c`: Sprint 107 central matrix shell deferral contract.

These exclusions prevent Sprint 108 from improving already-settled helper
names or moving the same setup a second time.

## Residual Owner Assessment

### `tests/test_ldlt_csc.c`

Remaining risk:

- broad direct-solver proof and oracle logic;
- external dense reference checks;
- multiple residual and solve-agreement patterns;
- supernodal, native, wrapper, and linked-list comparison surfaces.

Sprint 108 opportunity:

- extract at most one additional named assertion, residual, or oracle helper;
- preserve direct CSC proof intent at call sites;
- keep all focused validation in `test_ldlt_csc`.

Day 2 disposition: proceed to Day 3 boundary selection. This is the highest
value because the file is the largest proof owner and has the highest assertion
count, but the change must remain narrow.

### `tests/test_qr.c`

Remaining risk:

- generated fixtures outside the Sprint 107 4x3 builders;
- tall/economy builders;
- diagonal and singleton setup;
- SuiteSparse exact-RHS setup;
- dense/sparse QR parity and reconstruction surfaces.

Sprint 108 opportunity:

- extract only fixture construction or RHS setup whose assertions stay visible;
- avoid hiding rank, solve, reconstruction, refinement, or residual proof.

Day 2 disposition: schedule after LDLT CSC. QR has high setup repetition, but
the proof readability risk is lower once fixture construction is clearly
separated from assertions.

### `tests/test_iterative.c`

Remaining risk:

- convergence-sensitive setup;
- solver options and restart values;
- preconditioner setup;
- direct solver comparisons;
- result/convergence assertions.

Sprint 108 opportunity:

- clean one bounded setup pattern only after a convergence-boundary artifact;
- keep solver configuration and convergence evidence visible.

Day 2 disposition: schedule after QR. Iterative cleanup is valuable but more
semantically fragile because helper movement can hide why a solver converged or
failed to converge.

### `tests/test_svd.c`

Remaining risk:

- rank and oracle assertions;
- reconstruction checks;
- pseudoinverse proof;
- low-rank and partial-SVD comparisons;
- condition-number behavior;
- SuiteSparse and dense-reference expectations.

Sprint 108 opportunity:

- create a dedicated validation lane before moving any remaining helper family;
- extract only one safe helper family if the lane proves reviewable.

Day 2 disposition: schedule after iterative cleanup. SVD has broad proof logic
and must not move rank/oracle meaning without a validation lane.

### `src/sparse_eigs.c`

Remaining risk:

- source-owner size and shared spectral kernels;
- dense Jacobi feasibility;
- grow-m refinement boundaries;
- source-list, Make/CMake, and cross-backend validation impact.

Sprint 108 opportunity:

- prepare a feasibility and validation plan only;
- defer source movement unless a future sprint proves the split low risk.

Day 2 disposition: schedule after test proof-owner cleanup so source feasibility
can use final proof-owner guardrails and comparison evidence.

### `src/sparse_matrix.c`

Remaining risk:

- central public behavior and compatibility;
- private-header dependencies;
- allocation, mutation, permutation, factor-state, Matrix Market, and matvec
  behavior;
- Sprint 101 compressed-first product contract.

Sprint 108 opportunity:

- conduct public-behavior and private-header dependency review;
- define future extraction prerequisites.

Day 2 disposition: schedule after eigensolver feasibility. Matrix shell work
is review/planning only in Sprint 108 and should not precede source feasibility
unless earlier work reveals a dependency.

## Ranked Sprint 108 Cleanup Queue

| Rank | Candidate | Reason | Validation Cost | Decision |
|---:|---|---|---|---|
| 1 | LDLT CSC oracle helper follow-through | Largest proof owner, highest assertion count, direct value for failure localization. | Medium: focused `test_ldlt_csc`, full gate if `.c` changes. | Proceed to Day 3 boundary. |
| 2 | QR residual fixture follow-through | High fixture/setup repetition with relatively clear builder boundaries. | Medium: focused `test_qr`, full gate if `.c` changes. | Proceed after LDLT boundary. |
| 3 | Iterative convergence-sensitive cleanup | Valuable setup cleanup but solver behavior can be obscured. | Medium-high: focused iterative solver tests and full gate if `.c` changes. | Proceed after explicit Day 7 boundary. |
| 4 | SVD validation-lane cleanup | Broad proof surface requires a lane before movement. | High: focused `test_svd`, likely full gate if `.c` changes. | Proceed only after Day 9 validation lane. |
| 5 | Eigensolver source feasibility plan | Source split risk remains, but Sprint 108 should plan rather than move code. | Documentation-first unless source changes occur. | Plan-only unless later evidence changes. |
| 6 | Matrix shell public-behavior review | Central compatibility territory; prerequisites must precede any split. | Documentation-first unless source changes occur. | Review-only in Sprint 108. |

## Validation Cost Map

| Surface | Focused Validation | Full Gate Trigger |
|---|---|---|
| `tests/test_ldlt_csc.c` | `make build/test_ldlt_csc && ./build/test_ldlt_csc` | Any `.c` change requires `make format && make lint && make test`. |
| `tests/test_qr.c` | `make build/test_qr && ./build/test_qr` | Any `.c` change requires `make format && make lint && make test`. |
| `tests/test_iterative.c` | `make build/test_iterative && ./build/test_iterative` | Any `.c` change requires `make format && make lint && make test`. |
| `tests/test_svd.c` | `make build/test_svd && ./build/test_svd` | Any `.c` change requires `make format && make lint && make test`. |
| `src/sparse_eigs.c` | focused eigensolver suites, thick restart, LOBPCG, and comparison smoke tests | Any source/header/build membership change requires full gate and source-list/CMake checks. |
| `src/sparse_matrix.c` | sparse matrix, sparse IO, CSR/CSC, arithmetic, matmul, and solver-entry smoke tests | Any source/header/build membership change requires full gate and public behavior review. |

## Day 2 Decision

Sprint 108 remains boundary-first. The next step is Day 3 LDLT CSC oracle
boundary selection, with a hard cap of one additional proof helper unless the
boundary artifact explicitly defers all candidates. QR, iterative, and SVD
cleanup stay ordered behind their own boundary artifacts. Eigensolver and
matrix shell work remain planning/review surfaces until build-system and
behavioral guardrails are explicit.

## Completion Criteria Status

- All Sprint 107 residual proof-owner items have a Sprint 108 disposition.
- Completed Sprint 107 helper extractions are excluded from Sprint 108 work.
- Cleanup order is explicit and dependency-safe.

