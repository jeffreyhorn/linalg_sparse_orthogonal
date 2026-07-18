# Sprint 127 Day 7 Threshold Family Evidence

## Decision

Day 7 accepts one bounded perturbed duplicate-column QR threshold fixture:
`qr_rank_threshold_duplicate_5x4_perturbed_family`.

The fixture starts from the completed duplicate-column 5 x 4 rank-deficient
matrix and changes only row `0`, column `3` from `0.0` to `6e-8`. It is
threshold-rank evidence only. It does not assert residual, nullspace,
subspace, minimum-norm, pseudoinverse, Q-basis, economy, sparse-mode, reorder,
backend, SuiteSparse corpus, optional-data, platform, performance,
dense-library parity, default-threshold behavior, or global QR
rank-threshold policy.

## Fixture Contract

| Field | Value |
| --- | --- |
| Fixture key | `qr_rank_threshold_duplicate_5x4_perturbed_family` |
| Owner | `tests/test_qr.c` and `tests/qr_external_dense_reference.py` |
| Baseline matrix | Existing `tf_qr_make_rankdef_duplicate_5x4()` fixture |
| Matrix shape | 5 rows by 4 columns |
| Perturbation | Insert `6e-8` at row `0`, column `3` |
| Thresholds | `1e-10`, `1e-6` |
| Expected ranks | `4` at `1e-10`; `3` at `1e-6` |
| External helper output | `OK 6` |
| Output semantics | perturbation/threshold/expected-rank triples |
| Product diagnostics | relative threshold, absolute threshold, product rank, rank-info rank, pivot ratio, and R diagonal magnitudes |

## Implemented Evidence

- Added `build_qr_rank_threshold_duplicate_5x4_perturbed_family()` to
  `tests/qr_external_dense_reference.py`.
- Extended `threshold_rank_reference()` so the helper emits the new
  perturbation/threshold/rank triples.
- Routed the new key through the helper `main()` dispatch.
- Added the new key to `read_qr_threshold_external_reference()` in
  `tests/test_qr.c`.
- Added
  `test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family()`
  to `tests/test_qr.c`.
- Registered the test next to the existing unscaled and scaled threshold
  fixtures.

The C test:

1. Reads the helper triples and checks the fixture-local perturbation,
   thresholds, and expected ranks.
2. Builds the duplicate-column 5 x 4 fixture.
3. Inserts the accepted perturbation at row `0`, column `3`.
4. Factors with QR.
5. Reads R diagonal magnitudes.
6. Compares `sparse_qr_rank()` and `sparse_qr_rank_info()` against the
   expected rank for each threshold.
7. Prints the perturbation, relative threshold, absolute threshold, expected
   rank, product rank, rank-info rank, small-pivot ratio, and R diagonal
   magnitudes.

## Focused Validation

```text
$ python3 -m py_compile tests/qr_external_dense_reference.py
$ python3 tests/qr_external_dense_reference.py qr_rank_threshold_duplicate_5x4_perturbed_family
OK 6
5.9999999999999995e-08
1e-10
4
5.9999999999999995e-08
9.9999999999999995e-07
3

$ make build/test_qr && ./build/test_qr
external QR dense ref rank_threshold_duplicate_5x4_perturbed_family:
perturb=6.0e-08 tol=1e-10 abs_tol=3.873e-10 expected=4 product=4 info=4
pivot_ratio=9.968e-09 |Rdiag|=[3.873e+00, 3.066e+00, 1.711e+00, 3.861e-08]
external QR dense ref rank_threshold_duplicate_5x4_perturbed_family:
perturb=6.0e-08 tol=1e-06 abs_tol=3.873e-06 expected=3 product=3 info=3
pivot_ratio=9.968e-09 |Rdiag|=[3.873e+00, 3.066e+00, 1.711e+00, 3.861e-08]
Tests run:    72
Tests failed: 0
Tests skipped: 0
ALL TESTS PASSED
```

## Proof Boundary

Day 7 proves only this bounded statement:

> For the accepted perturbed duplicate-column 5 x 4 fixture, product QR rank
> and rank-info rank match fixture-local expected ranks `4` and `3` at
> explicit relative thresholds `1e-10` and `1e-6`.

Day 7 does not prove:

- a global QR rank-threshold, default-threshold, or numerical-rank policy;
- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package threshold
  parity;
- broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, residual, nullspace, subspace, Q-basis, economy,
  sparse-mode, reorder, backend, corpus, platform, or performance behavior;
- minimum-norm optimality, solution uniqueness, solution-selection policy,
  pseudoinverse behavior, QR-vs-SVD oracle behavior, COLAMD behavior,
  fallback behavior, or refinement behavior;
- SuiteSparse corpus correctness, optional-data behavior, optional-large
  support, platform support, or runtime behavior;
- package, ABI, public API, CMake, Makefile, CI, CTest, scalability, memory,
  or state-of-the-art behavior.

## Deferred Threshold Families

| Deferred Family | Reason | Future Gate |
| --- | --- | --- |
| Dependent-row threshold family | Still mixes rank-threshold, residual, and subspace interpretation unless the primary claim is narrowed further. | Define controlled perturbation source, expected ranks, and evidence wording that cannot reuse residual/projector metrics as threshold proof. |
| Wide threshold family | Rank changes imply nullity changes and can be misread as underdetermined solution-selection or minimum-norm evidence. | Pin rank and nullity for each threshold plus explicit minimum-norm and wide-solve non-claims. |
| Default-threshold evidence | `tol <= 0` behavior is product-local and easy to overclaim as public policy. | Define exact product-local claim, platform/compiler stability, and no-global-policy wording. |
| SuiteSparse threshold family | Requires support tier, expected-rank metadata, optional-data behavior, runtime budget, platform diagnostics, skip/fail behavior, and validation. | Defer to Days 8-9 SuiteSparse corpus gate. |
| Near-threshold nullspace/subspace | Requires threshold-specific rank and nullity first, then projection metrics. | Complete threshold-family and subspace gates before implementation. |

## Validation Notes

Day 7 changed C test code and the Python external-reference helper, so required
validation is:

1. `python3 -m py_compile tests/qr_external_dense_reference.py`
2. `python3 tests/qr_external_dense_reference.py qr_rank_threshold_duplicate_5x4_perturbed_family`
3. `make build/test_qr && ./build/test_qr`
4. `make format && make lint && make test`
5. `git diff --check`

Focused validation passed as recorded above. Full quality validation is
complete:

```text
$ make format && make lint && make test
All tests passed.
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 4 is complete or explicitly deferred. | Complete | One perturbed duplicate-column threshold fixture implemented; remaining families deferred with gates. |
| Accepted evidence has clear fixture-local threshold interpretation. | Complete | Fixture contract, helper triples, diagnostics, and proof boundary record perturbation, thresholds, and expected ranks. |
| Broad default-threshold and external parity claims remain absent. | Complete | See proof boundary, deferred families, and non-claim text. |
