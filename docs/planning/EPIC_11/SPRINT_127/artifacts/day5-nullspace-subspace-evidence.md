# Sprint 127 Day 5 Nullspace/Subspace Evidence

## Decision

Day 5 accepts one bounded dependent-row QR nullspace/subspace fixture:
`qr_rankdef_dependent_row_4x3_nullspace_projector`.

This fixture complements Sprint 126's
`qr_rankdef_dependent_row_4x3_residual_only` lane with external projector
evidence. It compares subspaces through a projector metric and explicitly
avoids raw basis-vector equality, basis ordering, basis orientation,
minimum-norm, pseudoinverse, Q-basis, economy, sparse-mode, backend,
SuiteSparse, optional-data, platform, performance, and broad QR claims.

## Fixture Contract

| Field | Value |
| --- | --- |
| Fixture key | `qr_rankdef_dependent_row_4x3_nullspace_projector` |
| Owner | `tests/test_qr.c` and `tests/qr_external_dense_reference.py` |
| Matrix shape | 4 rows by 3 columns |
| Matrix source | Existing `tf_qr_make_dependent_row_4x3()` helper |
| Matrix rows | `[1,0,1]`, `[0,1,2]`, `[1,1,3]`, `[2,-1,0]` |
| Structural model | Rank-deficient dependent-row fixture with nullspace vector proportional to `[-1, -2, 1]` |
| Expected rank | 2 |
| Expected nullity | 1 |
| Threshold | 0.0 |
| External helper output | `OK 13` |
| Metadata entries | `n`, `rank`, `nullity`, `threshold` |
| Projector entries | 9 column-major entries for `z z^T`, where `z = [-1, -2, 1] / sqrt(6)` |
| Product metric | Max absolute projector difference after product-basis normalization |
| Tolerance | Projector diff `< 1e-8`; null residual `< 1e-10`; norm error `< 1e-10` |

## Implemented Evidence

- Extended `tests/qr_external_dense_reference.py` so
  `nullspace_projector_reference()` emits the exact 3 x 3 nullspace projector
  for `qr_rankdef_dependent_row_4x3_nullspace_projector`.
- Routed the new fixture key through the helper `main()` dispatch.
- Added the new fixture key to `read_qr_basis_external_reference()` in
  `tests/test_qr.c`.
- Added
  `test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector()`
  to `tests/test_qr.c`.
- Registered the test beside the existing nullspace projector evidence.

The C test:

1. Checks helper metadata: `n = 3`, expected rank `2`, expected nullity `1`,
   and threshold `0.0`.
2. Builds the existing dependent-row 4 x 3 fixture locally.
3. Factors with QR and checks product rank/nullity against pinned metadata.
4. Reads the product nullspace basis.
5. Normalizes the product basis vector.
6. Computes `z z^T` and compares it to the external projector.
7. Reports projector diff, null residual, and norm error.

## Focused Validation

```text
$ python3 -m py_compile tests/qr_external_dense_reference.py
$ python3 tests/qr_external_dense_reference.py qr_rankdef_dependent_row_4x3_nullspace_projector
OK 13
3
2
1
0
0.16666666666666671
0.33333333333333343
-0.16666666666666671
0.33333333333333343
0.66666666666666685
-0.33333333333333343
-0.16666666666666671
-0.33333333333333343
0.16666666666666671

$ make build/test_qr && ./build/test_qr
external QR dense ref rankdef_dependent_row_4x3_nullspace_projector:
projector diff = 5.551e-17, null residual = 2.544e-16, norm err = 4.441e-16
Tests run:    71
Tests failed: 0
Tests skipped: 0
ALL TESTS PASSED
```

## Proof Boundary

Day 5 proves only this bounded statement:

> For the accepted dependent-row 4 x 3 fixture at threshold `0.0`, the QR
> product nullspace spans the same one-dimensional nullspace as the exact
> standard-library reference projector within `1e-8`.

Day 5 does not prove:

- broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, nullspace, subspace, or dense-library parity;
- raw nullspace vector equality, basis ordering, basis orientation, unique
  basis behavior, sign convention, or principal-angle parity;
- minimum-norm optimality, solution uniqueness, solution-selection policy,
  pseudoinverse behavior, or QR-vs-SVD oracle behavior;
- Q-basis, economy, sparse-mode, reorder, backend, SuiteSparse corpus,
  optional-data, platform, or performance behavior;
- global QR rank-threshold, default-threshold, or numerical-rank behavior;
- public API, package, ABI, CMake, Makefile, CI, CTest, scalability, memory,
  or state-of-the-art behavior.

## Deferred Lanes

| Deferred Lane | Reason | Future Gate |
| --- | --- | --- |
| Wide-shape nullspace/subspace | Still overlaps underdetermined solution-selection and minimum-norm interpretation. | Pin rank/nullity, define projection metric, and prove wording cannot imply minimum-norm behavior. |
| Near-threshold nullspace/subspace | Depends on threshold-family policy and threshold-specific expected nullity. | Complete Days 6-7 threshold metadata first. |
| SuiteSparse nullspace/subspace | Requires corpus support tier, optional-data skips, expected-rank/nullity metadata, runtime budget, platform diagnostics, and validation. | Complete Days 8-9 SuiteSparse corpus gate first. |
| Sparse-mode nullspace/subspace | Belongs to Q/economy/sparse-mode output semantics. | Defer to Sprint 128 owner. |
| Principal-angle metric | Adds helper complexity without current need. | Justify value beyond projector/two-way projection metrics. |
| Raw basis equality | Invalid for sign-sensitive and rotated nullspaces. | Prove deterministic ordering, sign convention, normalization, and a reason projection metrics are insufficient. |

## Validation Notes

Day 5 changed C test code and the Python external-reference helper, so required
validation is:

1. `python3 -m py_compile tests/qr_external_dense_reference.py`
2. `python3 tests/qr_external_dense_reference.py qr_rankdef_dependent_row_4x3_nullspace_projector`
3. `make build/test_qr && ./build/test_qr`
4. `make format && make lint && make test`
5. `git diff --check`

Focused validation passed as recorded above. Full quality validation is
required before Day 5 closeout because `.c` and Python helper files changed.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 3 is complete or explicitly deferred. | Complete | One dependent-row projector fixture implemented; remaining candidate lanes deferred with gates. |
| Accepted evidence uses only approved projection-style metrics. | Complete | Product basis is normalized and compared by projector difference. |
| Broad raw Q-basis, orientation, and unique-nullspace claims remain absent. | Complete | See proof boundary, deferred lanes, and non-claim text. |
