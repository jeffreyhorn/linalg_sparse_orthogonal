# Sprint 123 Day 4 SVD External Fixture Implementation

## Purpose

Day 4 executes the Day 3 decision to add one bounded full-SVD external fixture:

`svd_wide_fullrank_4x6`

The fixture adds wide-shape full-SVD singular-value evidence while keeping the
external-reference lane limited to a small fixed matrix and Python
standard-library arithmetic.

## Implemented Fixture Contract

| Field | Implemented Decision |
| --- | --- |
| Fixture key | `svd_wide_fullrank_4x6` |
| Matrix shape | 4x6 dense rectangular |
| Rank model | Full row rank |
| Reference path | `tests/svd_external_dense_reference.py` computes `A^T A`, runs bounded Jacobi eigenvalue iteration, sorts singular values, and emits exactly `min(m,n)` values. |
| Product path | `tests/test_svd.c` runs full SVD through `sparse_svd_compute`. |
| Compared quantity | Singular values only. |
| Expected output count | Exactly 4. |
| Tolerance | Max absolute difference below `1e-8`. |
| Skip behavior | Missing `python3` skips through the existing external-reference helper; Windows skips explicitly like the existing SVD external lanes. |
| Failure semantics | Reference helper `ERROR` output fails; output-count mismatch fails through the vector reader; SVD compute failure fails; singular-value mismatch fails as fixture-local disagreement. |
| Build membership impact | None. The test is registered inside existing `test_svd`; no new CTest target, Makefile test target, or CMake test member was added. |

## Implemented Surfaces

| Surface | Change |
| --- | --- |
| `tests/svd_external_dense_reference.py` | Added `build_svd_wide_fullrank_4x6`, fixture-key dispatch, and `min(m,n)` singular-value emission for all fixtures. |
| `tests/test_svd.c` | Added wide-fixture dimensions, sparse matrix builder, fixture-key allow-list entry, `test_svd_external_dense_reference_wide_fullrank_4x6`, and existing-suite registration. |
| `tests/test_svd_partial_helpers.h` | No change. |
| Makefile | No change. |
| CMake / CTest | No change. |
| Public docs / API | No change. |

## Reference Helper Evidence

```text
$ python3 tests/svd_external_dense_reference.py svd_wide_fullrank_4x6
OK 4
6.4515834893439248
4.7012584520796645
4.4421185646585881
3.2852430814613958
```

The `OK 4` header confirms the wide fixture emits exactly `min(4,6)` values.

## Focused Test Evidence

`make build/test_svd && ./build/test_svd` passed.

The focused SVD output included:

```text
external SVD dense ref wide_fullrank_4x6: max |sigma-sigma_ref| = 4.441e-15
```

`test_svd` summary:

- 107 tests
- 0 failures
- 0 skips
- 1755 assertions

## Full Validation Evidence

| Command | Result |
| --- | --- |
| `python3 tests/svd_external_dense_reference.py svd_wide_fullrank_4x6` | Passed; emitted `OK 4`. |
| `make build/test_svd && ./build/test_svd` | Passed; wide fixture max difference was `4.441e-15`. |
| `make format` | Passed. |
| `make lint` | Passed. |
| `make test` | Passed. |
| `git diff --check` | Passed. |
| Focused trailing-whitespace scan over Sprint 123 docs and touched files | Passed. |

## Deferred SVD Work

Day 4 does not implement:

- near-dependent threshold external singular-value evidence;
- repeated-spectrum external value or subspace evidence;
- low-rank tail-energy evidence;
- pseudoinverse or minimum-norm external evidence;
- SuiteSparse or broad corpus external SVD evidence;
- singular-vector, orientation, or subspace external parity.

Those lanes remain deferred under the Day 3 promotion gates.

## Non-Claim Register

This Day 4 implementation does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad ecosystem parity;
- broad external dense-library SVD correctness;
- singular-vector, subspace, repeated-spectrum basis, or orientation parity;
- partial-SVD external vector, subspace, convergence, or residual parity;
- pseudoinverse, Moore-Penrose, or minimum-norm correctness;
- low-rank global optimality or approximation quality;
- rank-threshold policy correctness;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- portable performance, scalability, memory behavior, or state-of-the-art
  behavior.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Day 3 decision is fully executed. | Complete | `svd_wide_fullrank_4x6` is implemented in the helper and `test_svd`. |
| Any code/script change has focused validation evidence. | Complete | Helper output and focused `test_svd` run passed. |
| No unsupported public SVD claim is added. | Complete | No public docs or API changed; non-claims remain explicit. |
