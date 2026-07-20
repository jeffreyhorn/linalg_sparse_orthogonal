# Sprint 128 Day 5 Wide and Near-Threshold Subspace Evidence

## Decision

Day 5 accepts one bounded wide-shape QR nullspace/subspace fixture:
`qr_rankdef_wide_3x5_nullspace_subspace`.

This fixture complements the existing deterministic `test_rank_rect_deficient`
wide 3 x 5 lane with external projector evidence. It compares subspaces
through a projection-style projector metric and explicitly avoids raw
basis-vector equality, basis ordering, basis orientation, minimum-norm,
pseudoinverse, Q-basis, economy, sparse-mode, backend, SuiteSparse,
optional-data, platform, performance, and broad QR claims.

## Fixture Contract

| Field | Value |
| --- | --- |
| Fixture key | `qr_rankdef_wide_3x5_nullspace_subspace` |
| Owner | `tests/test_qr.c` and `tests/qr_external_dense_reference.py` |
| Matrix shape | 3 rows by 5 columns |
| Matrix source | Existing deterministic matrix from `test_rank_rect_deficient` |
| Matrix rows | `[1,2,0,1,0]`, `[0,3,1,0,2]`, `[1,5,1,1,2]` |
| Structural model | Wide rank-deficient fixture where row 2 is row 0 plus row 1 |
| Expected rank | 2 |
| Expected nullity | 3 |
| Threshold | 0.0 |
| External helper output | `OK 29` |
| Metadata entries | `n`, `rank`, `nullity`, `threshold` |
| Projector entries | 25 column-major entries for the exact nullspace projector |
| Reference span | `[-2,1,-3,0,0]`, `[-1,0,0,1,0]`, `[0,0,-2,0,1]` |
| Product metric | Max absolute projector difference after product-basis Gram-Schmidt |
| Tolerance | Projector diff `< 1e-8`; null residual `< 1e-10`; orthogonality error `< 1e-10` |

## Implemented Evidence

- Extended `tests/qr_external_dense_reference.py` so
  `nullspace_projector_reference()` emits the exact 5 x 5 nullspace projector
  for `qr_rankdef_wide_3x5_nullspace_subspace`.
- Routed the new fixture key through the helper `main()` dispatch.
- Added the new fixture key to `read_qr_basis_external_reference()` in
  `tests/test_qr.c`.
- Added
  `test_qr_external_dense_reference_rankdef_wide_3x5_nullspace_subspace()`
  to `tests/test_qr.c`.
- Registered the test beside the existing nullspace projector evidence.
- Updated `docs/maintainer_guide.md` to include the bounded wide subspace
  fixture in the maintained QR evidence row.

The C test:

1. Checks helper metadata: `n = 5`, expected rank `2`, expected nullity `3`,
   and threshold `0.0`.
2. Builds the exact wide 3 x 5 matrix locally.
3. Factors with QR and checks product rank/nullity against pinned metadata.
4. Reads the product nullspace basis.
5. Orthonormalizes the product basis with local Gram-Schmidt.
6. Computes `Z Z^T` and compares it to the external projector.
7. Reports projector diff, null residual, and orthogonality error.

## Focused Validation

```text
$ python3 -m py_compile tests/qr_external_dense_reference.py
$ python3 tests/qr_external_dense_reference.py qr_rankdef_wide_3x5_nullspace_subspace
OK 29
5
2
3
0
...

$ make build/test_qr && ./build/test_qr
external QR dense ref rankdef_wide_3x5_nullspace_subspace:
projector diff = 4.441e-16, null residual = 8.671e-16,
orthogonality err = 4.441e-16
Tests run:    73
Tests failed: 0
Tests skipped: 0
ALL TESTS PASSED
```

## Proof Boundary

Day 5 proves only this bounded statement:

> For the accepted wide 3 x 5 fixture at threshold `0.0`, the QR product
> nullspace spans the same three-dimensional nullspace as the exact
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
| Near-threshold nullspace/subspace | Depends on threshold-family policy and threshold-specific expected nullity. | Complete Days 6-7 threshold metadata first. |
| SuiteSparse nullspace/subspace | Requires corpus support tier, optional-data skips, independent expected-rank/nullity metadata, runtime budget, platform diagnostics, and validation. | Complete Days 8-9 SuiteSparse corpus gate first. |
| Sparse-mode nullspace/subspace | Belongs to Q/economy/sparse-mode output semantics. | Defer to Sprint 129 owner. |
| Economy-mode nullspace/subspace | Requires named economy output-shape semantics and projection boundaries. | Defer to Sprint 129 owner. |
| Principal-angle metric | Adds helper complexity without current need. | Justify value beyond projector/two-way projection metrics. |
| Raw basis equality | Invalid for sign-sensitive and rotated nullspaces. | Prove deterministic ordering, sign convention, normalization, and a reason projection metrics are insufficient. |

## Validation Notes

Day 5 changed C test code, the Python external-reference helper, and
maintainer documentation, so required validation is:

1. `python3 -m py_compile tests/qr_external_dense_reference.py`
2. `python3 tests/qr_external_dense_reference.py qr_rankdef_wide_3x5_nullspace_subspace`
3. `make build/test_qr && ./build/test_qr`
4. `make format && make lint && make test`
5. `git diff --check`

Focused validation passed as recorded above. Full quality validation is
required before Day 5 closeout because `.c` and Python helper files changed.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 3 is complete or explicitly deferred. | Complete | One wide projector fixture implemented; remaining candidate lanes deferred with gates. |
| Accepted evidence uses approved projection-style metrics. | Complete | Product basis is locally orthonormalized and compared by projector difference. |
| Broad raw Q-basis, orientation, and unique-nullspace claims remain absent. | Complete | See proof boundary, deferred lanes, and non-claim text. |
