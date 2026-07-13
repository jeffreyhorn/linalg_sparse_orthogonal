# Sprint 122 Day 4 SVD External Fixture Decision

## Purpose

Day 4 completes Sprint 122 Item 2 by deciding whether additional SVD external
fixtures should be added beyond the Sprint 121 `svd_rect_fullrank_6x4` pilot.
The Day 3 inventory identified one bounded fixture that adds evidence without
turning the external-reference lane into a broad dense-library parity claim.

## Decision

Accepted and implemented one additional bounded full-SVD singular-value fixture:

`svd_rankdef_duplicate_5x4`

No other Day 3 SVD candidates were accepted for Sprint 122.

## Accepted Fixture Contract

| Field | Decision |
| --- | --- |
| Fixture key | `svd_rankdef_duplicate_5x4` |
| Matrix shape | 5x4 dense rectangular |
| Rank model | Exact rank deficient; the third column is the sum of the first two columns |
| Reference path | `tests/svd_external_dense_reference.py` computes `A^T A`, runs bounded Jacobi eigenvalue iteration, clamps tiny negative roundoff, and emits sorted singular values. |
| Product path | `tests/test_svd.c` runs full SVD through `sparse_svd_compute`. |
| Compared quantity | Singular values only. |
| Positive singular-value tolerance | Max absolute difference below `1e-8`. |
| Zero-tail tolerance | Library and reference smallest singular values must both be below `1e-8`. |
| Dependency policy | Python standard library only; no NumPy, SciPy, LAPACK, BLAS, or external data dependency. |
| Skip behavior | Missing `python3` skips through the existing external-reference helper; Windows skips explicitly like the existing SVD external pilot. |
| Failure semantics | Reference helper `ERROR` output fails the test; SVD compute failure fails the test; positive singular-value mismatch and zero-tail mismatch fail separately. |
| Build membership impact | None. The test remains inside existing `test_svd`. |

## Implemented Surfaces

| Surface | Change |
| --- | --- |
| `tests/svd_external_dense_reference.py` | Added `build_svd_rankdef_duplicate_5x4` and fixture-key routing. |
| `tests/test_svd.c` | Added rank-deficient fixture builder, allowed the new fixture key, added `test_svd_external_dense_reference_rankdef_duplicate_5x4`, and registered it in existing `test_svd`. |
| Makefile | No change. |
| CMake / CTest | No change. |
| Public docs / API | No change. |

## Observed Focused Evidence

Python helper output:

```text
OK 4
5.8492807379668301
5.0896447694773794
2.6232481714435254
0
```

Focused `test_svd` output included:

```text
external SVD dense ref rankdef_duplicate_5x4: max positive diff = 3.553e-15, smallest sigma/ref = 0.000e+00/0.000e+00
```

`test_svd` result:

- 105 tests
- 0 failures
- 0 skips
- 1716 assertions

## Rejected or Deferred SVD Candidates

| Candidate | Disposition | Reason |
| --- | --- | --- |
| `svd_wide_fullrank_4x6_external_sigma` | Deferred | Wide-shape external output semantics need a separate decision around `min(m,n)` values versus padded zero singular values. |
| `svd_near_dependent_5x4_external_sigma` | Deferred | Threshold-sensitive singular values overlap rank-policy and condition-number semantics. |
| `svd_diag_repeated_5x5_external_sigma` | Rejected | Analytical diagonal repeated-spectrum behavior is already deterministic and would duplicate existing internal proof. |
| `svd_lowrank_outer_product_external_sigma` | Rejected for Day 4 | Singular-value evidence would not prove low-rank output optimality and risks claim drift. |
| `svd_suite_sparse_external_sigma` | Rejected for Sprint 122 | Optional corpus fixtures would add platform, runtime, and broad-corpus interpretation risk. |
| `svd_vector_subspace_external_check` | Rejected for Day 4 | Vector and subspace parity require separate sign, basis, and repeated-spectrum semantics. |

## Non-Claim Register

This Day 4 fixture does not claim:

- LAPACK, SciPy, NumPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, or broad
  dense-library parity;
- singular-vector or subspace external parity;
- partial-SVD external parity;
- low-rank or pseudoinverse global optimality;
- QR or least-squares external parity;
- performance, scalability, package, platform, ABI, public API, or
  state-of-the-art behavior.

## Validation Plan

Because `.c` changed, the required validation gate for the branch is:

1. `python3 tests/svd_external_dense_reference.py svd_rankdef_duplicate_5x4`
2. `make format`
3. `make build/test_svd && ./build/test_svd`
4. `make lint`
5. `make test`
6. `git diff --check`
7. Focused trailing-whitespace scan over Sprint 122 docs and touched files

## Validation Results

| Command | Result |
| --- | --- |
| `python3 tests/svd_external_dense_reference.py svd_rankdef_duplicate_5x4` | Passed; emitted four singular values with a zero tail. |
| `make format` | Passed. |
| `make build/test_svd && ./build/test_svd` | Passed: 105 tests, 0 failures, 0 skips, 1716 assertions. |
| `make lint` | Passed after Day 6. |
| `make test` | Passed after Day 6. |

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 2 is complete. | Complete | Accepted and implemented one bounded additional SVD fixture; other candidates are rejected or deferred. |
| Decision is reproducible from named evidence. | Complete | Day 3 inventory, Sprint 121 pilot artifacts, and this fixture contract document the choice. |
| Accepted work has enough detail for implementation or deferral owner. | Complete | Implemented surfaces, tolerance, skip, failure semantics, and non-claims are explicit. |
