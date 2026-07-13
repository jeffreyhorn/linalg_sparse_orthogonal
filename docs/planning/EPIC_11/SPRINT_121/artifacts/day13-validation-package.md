# Sprint 121 Day 13 - Validation Package

## Purpose

Day 13 packaged the Sprint 121 QR, rank-deficient, least-squares,
pseudoinverse, low-rank, partial-SVD, and SVD external-reference evidence into
one validation and trust-boundary record. The day did not add new product
claims; it verified that the accumulated proof-owner changes still pass the
focused solver suite and the required full C quality gate.

## Scope

| Field | Value |
|---|---|
| Sprint/day | Sprint 121 Day 13 |
| Artifact owner | Sprint 121 validation and trust-boundary closeout |
| Solver or behavior family | QR, QR solve, rank, SVD, pseudoinverse, low-rank, partial SVD, and bounded SVD external reference |
| Touched validation surfaces | `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_svd.c`, `tests/test_qr_helpers.h`, `tests/test_svd_helpers.h`, `tests/test_svd_partial_helpers.h`, `tests/svd_external_dense_reference.py`, Sprint 121 planning artifacts |
| Explicitly out of scope | LAPACK/SciPy/NumPy/SuiteSparse/PETSc/Trilinos/Eigen parity, singular-vector/subspace external comparison, partial-SVD external comparison, QR external comparison, broad dense-library parity, performance/scalability claims, package/install lanes, ABI claims, CMake/CTest membership changes, and public API expansion. |

## Focused Validation Package

Command:

```sh
python3 tests/svd_external_dense_reference.py svd_rect_rank3_6x4 && \
make build/test_qr build/test_qr_solve build/test_svd && \
./build/test_qr && ./build/test_qr_solve && ./build/test_svd
```

Results:

| Surface | Result |
|---|---|
| `tests/svd_external_dense_reference.py svd_rect_rank3_6x4` | Passed; emitted 4 singular values. |
| `test_qr` | 65 tests, 0 failures, 0 skips, 603 assertions. |
| `test_qr_solve` | 13 tests, 0 failures, 0 skips, 1014 assertions. |
| `test_svd` | 104 tests, 0 failures, 0 skips, 1685 assertions. |
| SVD external-reference pilot | Passed with max `|sigma-sigma_ref| = 6.217e-15`. |

The Python helper emitted:

```text
OK 4
12.936234553653524
6.5735839300926795
2.777297940451561
1.3147037829304029
```

## Full Quality Gate

Because Sprint 121 has `.c` and `.h` test changes, Day 13 ran the required
full quality gate:

```sh
make format && make lint && make test
```

Result: passed.

`make test` re-ran the touched QR/SVD proof-owner executables and preserved the
same focused outcomes:

- `test_qr`: 65 tests, 0 failures, 0 skips, 603 assertions.
- `test_qr_solve`: 13 tests, 0 failures, 0 skips, 1014 assertions.
- `test_svd`: 104 tests, 0 failures, 0 skips, 1685 assertions.
- SVD external-reference pilot max `|sigma-sigma_ref| = 6.217e-15`.

## Build Membership

No Makefile, CMake, CTest registration, workflow, package, benchmark, public
API, or production source surfaces were changed on Day 13.

No source-list, CTest-count, package, install, or workflow membership check was
required for Day 13 because the validation package did not add or remove any
test executable.

## Trust Boundary And Non-Claim Register

The Sprint 121 evidence supports deterministic in-repository proof-owner
fixtures and one bounded SVD external-reference singular-value pilot. It does
not support broader product claims.

Preserved non-claims:

- No LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos, Eigen, or broad
  external dense-library parity claim.
- No singular-vector or subspace external parity claim.
- No partial-SVD external parity claim.
- No QR external parity claim.
- No low-rank global optimality claim beyond the deterministic fixtures.
- No pseudoinverse or minimum-norm global optimality claim beyond the
  deterministic fixtures.
- No performance, scalability, package, platform, ABI, or state-of-the-art
  claim.
- No public API expansion claim.

## Guidance And Documentation Impact

The accumulated Sprint 121 artifacts already document the trust boundaries for
the helper extraction, rank-deficient fixtures, least-squares/pseudoinverse
fixtures, low-rank/partial-SVD fixtures, and SVD external-reference pilot.

No public README, solver-selection, package, or install documentation was
updated on Day 13 because the new evidence remains internal validation
coverage, not a user-facing capability or support-level expansion.

## Deferred Validation Queue

| Deferred validation | Reason for deferral | Future owner |
|---|---|---|
| Additional SVD external fixtures | Current pilot intentionally covers one fixed 6x4 full-SVD singular-value fixture. | Future SVD oracle owner. |
| QR external dense-reference lane | Sprint 121 expanded deterministic QR/rank/least-squares fixtures but did not design a QR external helper. | Future QR oracle owner. |
| Partial-SVD external parity | Current partial-SVD evidence compares against in-repository full-SVD behavior and deterministic vector residuals only. | Future partial-SVD oracle owner. |
| Public solver-selection wording | Current evidence is internal validation; public guidance should change only after broader external or support-level evidence lands. | Future docs/product owner. |

## Completion Check

| Criterion | Status |
|---|---|
| Focused SVD, QR, rank, least-squares, and reference-pilot validation ran. | Complete. |
| Required full quality gate ran because `.c` and `.h` files changed. | Complete. |
| Trust-boundary and non-claim register is explicit. | Complete. |
| Public documentation drift was assessed. | Complete. |
| Residual validation queue is recorded. | Complete. |
