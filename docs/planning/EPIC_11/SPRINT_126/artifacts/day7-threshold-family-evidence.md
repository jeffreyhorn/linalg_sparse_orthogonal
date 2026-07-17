# Sprint 126 Day 7 Threshold Family Evidence

## Decision

Day 7 accepts one bounded expanded QR threshold-family fixture:
`qr_rank_threshold_diag4_scaled_family`.

The fixture extends Sprint 125's completed unscaled diagonal threshold ladder
with three named scale values while preserving the same relative-threshold
rank expectations. It is rank-threshold evidence only. It does not assert
residual, nullspace, minimum-norm, pseudoinverse, Q-basis, economy,
sparse-mode, reorder, backend, SuiteSparse corpus, platform, performance,
dense-library parity, or global QR rank-threshold behavior.

## Fixture Contract

| Field | Value |
| --- | --- |
| Fixture key | `qr_rank_threshold_diag4_scaled_family` |
| Matrix family | 4x4 diagonal |
| Base diagonal | `[1, 1e-8, 1e-12, 0]` |
| Scale values | `1e-6`, `1`, `1e6` |
| Scaled diagonal | `[s, s*1e-8, s*1e-12, 0]` |
| Thresholds | `1e-14`, `1e-10`, `1e-6` |
| Expected ranks per scale | `3`, `2`, `1` |
| External helper output | `OK 27` |
| Output semantics | `scale`, `threshold`, `expected_rank` triples |
| Product owner | `tests/test_qr.c` |
| Helper owner | `tests/qr_external_dense_reference.py` |

## Implemented Evidence

- Extended `threshold_rank_reference()` in
  `tests/qr_external_dense_reference.py` to emit
  `qr_rank_threshold_diag4_scaled_family`.
- Routed the new fixture key through the helper `main()` dispatch.
- Added the fixture key to `read_qr_threshold_external_reference()` in
  `tests/test_qr.c`.
- Added
  `test_qr_external_dense_reference_rank_threshold_diag4_scaled_family()` to
  `tests/test_qr.c`.
- Registered the test next to the completed unscaled threshold fixture.

The C test checks every scale/threshold record by:

1. Building the scaled diagonal matrix.
2. Factoring with QR.
3. Reading R diagonal magnitudes.
4. Computing the product rank with `sparse_qr_rank()`.
5. Computing rank-info rank with `sparse_qr_rank_info()`.
6. Comparing both ranks against the external expected rank.
7. Printing scale, relative threshold, absolute threshold, expected rank,
   product rank, rank-info rank, and R diagonal magnitudes.

## Focused Validation

```text
$ python3 -m py_compile tests/qr_external_dense_reference.py
$ python3 tests/qr_external_dense_reference.py qr_rank_threshold_diag4_scaled_family
OK 27
...

$ make build/test_qr && ./build/test_qr
external QR dense ref rank_threshold_diag4_scaled_family:
scale=1e-06 tol=1e-14 abs_tol=1.000e-20 expected=3 product=3 info=3
scale=1e-06 tol=1e-10 abs_tol=1.000e-16 expected=2 product=2 info=2
scale=1e-06 tol=1e-06 abs_tol=1.000e-12 expected=1 product=1 info=1
scale=1e+00 tol=1e-14 abs_tol=1.000e-14 expected=3 product=3 info=3
scale=1e+00 tol=1e-10 abs_tol=1.000e-10 expected=2 product=2 info=2
scale=1e+00 tol=1e-06 abs_tol=1.000e-06 expected=1 product=1 info=1
scale=1e+06 tol=1e-14 abs_tol=1.000e-08 expected=3 product=3 info=3
scale=1e+06 tol=1e-10 abs_tol=1.000e-04 expected=2 product=2 info=2
scale=1e+06 tol=1e-06 abs_tol=1.000e+00 expected=1 product=1 info=1
Tests run:    70
Tests failed: 0
Tests skipped: 0
ALL TESTS PASSED
```

## Proof Boundary

Day 7 proves only this bounded statement:

> For the accepted scaled diagonal 4x4 family at scale values `1e-6`, `1`,
> and `1e6`, product QR rank and rank-info rank match the fixture-local
> expected ranks `3`, `2`, and `1` at relative thresholds `1e-14`, `1e-10`,
> and `1e-6`.

Day 7 does not prove:

- a global QR rank-threshold policy;
- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package threshold
  parity;
- broad QR factorization, QR solve, rank-deficient solve, numerical-rank,
  residual, nullspace, subspace, Q-basis, economy, sparse-mode, reorder,
  backend, corpus, platform, or performance parity;
- minimum-norm optimality, solution uniqueness, pseudoinverse behavior,
  QR-vs-SVD oracle behavior, COLAMD behavior, fallback behavior, or refinement
  behavior;
- package, ABI, public API, CMake, Makefile, CI, CTest, scalability, memory,
  or state-of-the-art behavior.

## Deferred Threshold Families

| Deferred Family | Reason | Future Gate |
| --- | --- | --- |
| Perturbed duplicate-column threshold | QR pivoting and roundoff still require exact perturbation metadata and margins. | Define perturbation values, expected ranks, strict comparison behavior, and roundoff margin. |
| Dependent-row threshold | Mixes rank, residual, and nullspace interpretations unless the primary claim is explicit. | Define primary claim and prove residual/nullspace evidence is not being folded into threshold evidence. |
| Wide threshold | Rank changes imply nullity changes and can be misread as underdetermined minimum-norm evidence. | Pin rank/nullity per threshold and preserve minimum-norm non-claims. |
| SuiteSparse threshold | Requires corpus support tier, expected-rank metadata, optional-data behavior, and platform diagnostics. | Defer to Days 8-9 SuiteSparse corpus gate. |
| Default-threshold evidence | Uses product default tolerance, not explicit fixture thresholds. | Define default-threshold claim separately from explicit-threshold family evidence. |

## Validation Notes

Day 7 changed C test code and the Python external-reference helper, so required
validation is:

1. `python3 -m py_compile tests/qr_external_dense_reference.py`
2. `python3 tests/qr_external_dense_reference.py qr_rank_threshold_diag4_scaled_family`
3. `make build/test_qr && ./build/test_qr`
4. `make format && make lint && make test`
5. `git diff --check`

Focused validation passed as recorded above. Full quality gate completed:

```text
$ make format && make lint && make test
All tests passed.
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 4 is complete or explicitly deferred. | Complete | Scaled diagonal threshold family implemented; lower-priority families deferred with gates. |
| Accepted threshold fixtures have pinned metadata and diagnostics. | Complete | Fixture contract, helper triples, and product diagnostics record scale, threshold, expected rank, product rank, rank-info rank, absolute threshold, and R diagonal magnitudes. |
| Broad external parity and global threshold claims remain fenced. | Complete | See proof boundary and deferred threshold families. |
