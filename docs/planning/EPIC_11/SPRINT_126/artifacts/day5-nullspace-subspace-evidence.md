# Sprint 126 Day 5 Nullspace/Subspace Evidence

## Decision

Day 5 accepts one bounded expanded QR nullspace/subspace fixture:
`qr_rank1_4x3_nullspace_projector`.

This fixture adds the first multi-dimensional external nullspace projector
evidence after Sprint 125's nullity-1 duplicate-column projector. It compares
subspaces through a projector metric and explicitly avoids raw basis-vector
equality, basis ordering, basis orientation, minimum-norm, pseudoinverse,
Q-basis, economy, sparse-mode, backend, SuiteSparse, and broad QR claims.

## Fixture Contract

| Field | Value |
| --- | --- |
| Fixture key | `qr_rank1_4x3_nullspace_projector` |
| Matrix shape | 4 rows by 3 columns |
| Matrix entries | `A[i,j] = i + 1` for all columns |
| Structural model | Rank-1 outer product with column vector `[1, 2, 3, 4]^T` and row vector `[1, 1, 1]` |
| Expected rank | 1 |
| Expected nullity | 2 |
| Threshold | 0.0 |
| External helper output | `OK 13` |
| Metadata entries | `n`, `rank`, `nullity`, `threshold` |
| Projector entries | 9 column-major entries for `P_null = I - 11^T / 3` |
| Product metric | Max absolute projector difference after local product-basis orthonormalization |
| Tolerance | Projector diff `< 1e-8`; null residual `< 1e-10`; orthogonality error `< 1e-10` |

## Implemented Evidence

- Extended `tests/qr_external_dense_reference.py` so
  `nullspace_projector_reference()` emits the exact 3x3 nullspace projector for
  `qr_rank1_4x3_nullspace_projector`.
- Routed the new key through the helper `main()` dispatch.
- Added the new fixture key to `read_qr_basis_external_reference()` in
  `tests/test_qr.c`.
- Added
  `test_qr_external_dense_reference_rank1_4x3_nullspace_projector()` to
  `tests/test_qr.c`.
- Registered the test beside the existing nullspace projector evidence.

The C test:

1. Checks helper metadata: `n = 3`, expected rank `1`, expected nullity `2`,
   and threshold `0.0`.
2. Builds the rank-1 4x3 fixture locally.
3. Factors with QR and checks product rank/nullity against the pinned
   metadata.
4. Reads the product nullspace basis.
5. Locally orthonormalizes the two product basis vectors.
6. Computes `Z Z^T` and compares it to the external projector.
7. Reports projector diff, max null residual, and orthogonality error.

## Focused Validation

```text
$ python3 -m py_compile tests/qr_external_dense_reference.py
$ python3 tests/qr_external_dense_reference.py qr_rank1_4x3_nullspace_projector
OK 13
3
1
2
0
0.66666666666666674
-0.33333333333333331
-0.33333333333333331
-0.33333333333333331
0.66666666666666674
-0.33333333333333331
-0.33333333333333331
-0.33333333333333331
0.66666666666666674

$ make build/test_qr && ./build/test_qr
external QR dense ref rank1_4x3_nullspace_projector:
projector diff = 2.220e-16, null residual = 5.088e-16,
orthogonality err = 2.220e-16
Tests run:    69
Tests failed: 0
Tests skipped: 0
ALL TESTS PASSED
```

## Proof Boundary

Day 5 proves only this bounded statement:

> For the accepted rank-1 4x3 fixture at threshold `0.0`, the QR product
> nullspace spans the same two-dimensional nullspace as the exact
> standard-library reference projector within `1e-8`.

Day 5 does not prove:

- broad QR factorization, QR solve, rank-deficient solve, nullspace, subspace,
  or dense-library parity;
- raw nullspace vector equality, basis ordering, basis orientation, unique
  basis behavior, sign convention, or principal-angle parity;
- minimum-norm optimality, solution uniqueness, pseudoinverse behavior, or
  QR-vs-SVD oracle behavior;
- Q-basis, economy, sparse-mode, reorder, backend, SuiteSparse corpus,
  optional-data, platform, or performance behavior;
- global QR rank-threshold behavior;
- public API, package, ABI, CMake, Makefile, CI, CTest, scalability, memory,
  or state-of-the-art behavior.

## Deferred Lanes

| Deferred Lane | Reason | Future Gate |
| --- | --- | --- |
| Dependent-row projector | The rank-1 multi-dimensional lane adds more distinct trust for Day 5. | Add only if a dependent-row projector proves value beyond deterministic dependent-row rank/null residual evidence and Day 3 residual evidence. |
| Wide-shape nullspace/subspace | Still overlaps underdetermined solution-selection and minimum-norm interpretation. | Pin rank/nullity, define projection metric, and prove wording cannot imply minimum-norm behavior. |
| Near-threshold nullspace/subspace | Depends on threshold-family policy and threshold-specific expected nullity. | Complete Days 6-7 threshold metadata first. |
| SuiteSparse nullspace/subspace | Requires corpus support tier, optional-data skips, expected-rank/nullity metadata, and platform diagnostics. | Complete Days 8-9 SuiteSparse corpus gate first. |
| Sparse-mode nullspace/subspace | Belongs to Q/economy/sparse-mode semantics. | Defer to Sprint 127 owner. |
| Principal-angle metric | Adds helper complexity without current need. | Justify value beyond projector/two-way projection metrics. |
| Raw basis equality | Invalid for rotated multi-dimensional nullspaces. | Prove deterministic ordering, sign convention, normalization, and a reason projection metrics are insufficient. |

## Validation Notes

Day 5 changed C test code and the Python external-reference helper, so required
validation is:

1. `python3 -m py_compile tests/qr_external_dense_reference.py`
2. `python3 tests/qr_external_dense_reference.py qr_rank1_4x3_nullspace_projector`
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
| Project-plan Item 3 is complete or explicitly deferred. | Complete | One multi-dimensional projector fixture implemented; remaining candidate lanes deferred with gates. |
| Accepted evidence uses only approved projection-style metrics. | Complete | Product basis is orthonormalized locally and compared by projector difference. |
| Broad raw Q-basis, orientation, and unique-nullspace claims remain absent. | Complete | See proof boundary, deferred lanes, and non-claim text. |
