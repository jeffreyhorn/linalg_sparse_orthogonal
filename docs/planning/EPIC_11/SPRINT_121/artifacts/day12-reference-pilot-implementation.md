# Sprint 121 Day 12 - Reference Pilot Implementation

## Purpose

Day 12 implemented the bounded SVD external dense-reference pilot designed on
Day 11. The pilot compares this library's full SVD singular values against a
small pure-Python dense reference for one deterministic rectangular
rank-deficient fixture.

## Scope

| Field | Value |
|---|---|
| Sprint/day | Sprint 121 Day 12 |
| Artifact owner | Sprint 121 SVD external dense-reference pilot |
| Solver or behavior family | SVD singular-value external dense-reference comparison |
| Touched surfaces | `tests/test_svd.c`, `tests/svd_external_dense_reference.py`, Sprint 121 planning artifacts |
| Explicitly out of scope | LAPACK/SciPy/NumPy parity, QR external comparison, partial-SVD external comparison, low-rank optimality, singular-vector/subspace comparison, SuiteSparse fixtures, benchmark timing, Makefile/CMake/CTest membership changes, public API expansion, package/install lanes, and platform support claims. |

## Implementation

Added `tests/svd_external_dense_reference.py`.

- Uses only the Python standard library.
- Builds the `svd_rect_fullrank_6x4` fixture.
- Computes `A^T A`.
- Diagonalizes the small symmetric Gram matrix with a bounded Jacobi
  iteration.
- Clamps tiny negative roundoff to zero.
- Emits sorted singular values through the existing external-reference output
  protocol.

Updated `tests/test_svd.c`.

- Enabled the existing `tf_read_external_reference_vector` helper for
  `test_svd`.
- Added a local 6x4 fixture builder for `svd_rect_fullrank_6x4`.
- Added `read_svd_external_reference_singular_values`.
- Added `test_svd_external_dense_reference_rect_fullrank_6x4`.
- Registered the test in the existing `test_svd` executable.

No Makefile, CMake, CTest registration, workflow, package, benchmark, public
API, or production source surfaces were changed.

## Fixture

| Fixture | Symmetry | Definiteness | Rank | Conditioning/scaling | Sparsity pattern | Expected behavior |
|---|---|---|---|---|---|---|
| `svd_rect_fullrank_6x4` | Nonsymmetric rectangular | Not applicable | 4 | Moderate scale, no extreme conditioning | Dense 6x4 | Library full SVD and pure-Python dense reference agree on the four singular values within `1e-8`. |

Matrix:

```text
[ 3.0  -1.0   0.0   2.0 ]
[ 0.0   4.0   1.0  -1.0 ]
[ 2.0   0.0   3.0   0.5 ]
[ 5.0   3.0   4.0   1.5 ]
[-1.0   5.0   4.0  -0.5 ]
[ 3.0   4.0   7.0   2.5 ]
```

## Oracle Or Reference Source

| Oracle/reference | Invocation | Trust boundary | Skip/error handling |
|---|---|---|---|
| External pure-Python dense SVD reference | `python3 tests/svd_external_dense_reference.py svd_rect_fullrank_6x4` | Independent dense arithmetic path for one small fixed fixture; not a LAPACK/SciPy/NumPy oracle and not a broad SVD correctness proof. | Missing `python3` skips through `tf_read_external_reference_vector`; helper `ERROR` output is a test failure; Windows skips explicitly. |
| Library full SVD | `sparse_svd_compute(A, NULL, &svd)` | Product behavior under test. | Allocation or SVD failure is a test failure. |
| Singular-value comparison | Max absolute difference across four singular values. | Bounded singular-value comparison only. | Difference above `1e-8` is a test failure. |

## Observed Evidence

Focused `test_svd` output included:

```text
external SVD dense ref rect_fullrank_6x4: max |sigma-sigma_ref| = 6.217e-15
```

`test_svd` result:

- 104 tests
- 0 failures
- 0 skips
- 1685 assertions

## Validation Commands

| Command | Required because | Reviewed/supplemental/local | Result |
|---|---|---|---|
| `python3 tests/svd_external_dense_reference.py svd_rect_fullrank_6x4` | Direct helper smoke check | Local | Passed; emitted 4 singular values. |
| `make format` | `.c` file changed | Reviewed quality gate piece | Passed. |
| `make build/test_svd && ./build/test_svd` | Focused SVD pilot proof | Reviewed focused validation | Passed. |
| `make lint` | `.c` file changed | Reviewed quality gate piece | Passed. |
| `make test` | `.c` file changed | Reviewed quality gate piece | Passed. |

## Unsupported Or Expected-Failure Cases

| Case | Disposition | Reason |
|---|---|---|
| NumPy/SciPy/LAPACK invocation | Unsupported | The pilot intentionally avoids external numerical package dependencies and parity claims. |
| Singular-vector or subspace comparison | Unsupported | Rank-deficient singular-vector bases are not unique. |
| Partial-SVD external comparison | Deferred | The current sprint owns one bounded full-SVD external singular-value pilot only. |
| QR external comparison | Deferred | Day 9 already expanded deterministic QR least-squares proof coverage. |
| SuiteSparse fixtures | Unsupported | The pilot stays small, deterministic, and fast. |
| Windows external execution | Skipped | Matches existing external-reference test policy. |
| Performance comparison | Unsupported | Helper runtime is not product evidence. |

## Drift Check

| Public/support surface | Impact | Action |
|---|---|---|
| README | None | No update. |
| Solver-selection docs | None | No update. |
| Examples/tutorial | None | No update. |
| Benchmark/performance wording | None | No update. |
| Package/platform docs | None | No update. |

## Non-Claims Preserved

- The pilot does not prove LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos,
  Eigen, or broad external dense-library parity.
- The pilot does not prove singular-vector, subspace, partial-SVD, low-rank,
  pseudoinverse, least-squares, or QR parity.
- The pilot does not prove performance, scalability, platform support, package
  support, ABI stability, or state-of-the-art behavior.
- The pilot does not add or change public API.

## Residual Handoff

| Residual | Next owner | Evidence link |
|---|---|---|
| Decide whether additional SVD external fixtures are worth adding | Future SVD oracle owner | This artifact and Sprint 121 retrospective queue. |
| Decide whether QR should receive a separate external dense-reference lane | Future QR oracle owner | Day 9 least-squares artifact and Day 11 design artifact. |
| Keep partial-SVD external parity out of Sprint 121 unless separately designed | Future partial-SVD oracle owner | Day 10 partial-SVD artifact. |

## Completion Check

| Criterion | Status |
|---|---|
| Day 11 selected pilot is implemented. | Complete. |
| Fixture and external-reference trust boundary are explicit. | Complete. |
| Tolerance and unsupported cases are explicit. | Complete. |
| Focused SVD validation passes. | Complete. |
| Required full C quality gate passes. | Complete. |
| Drift and non-claims are recorded. | Complete. |
| Residual handoff is recorded. | Complete. |
