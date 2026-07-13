# Sprint 121 Day 10 - Low-Rank and Partial-SVD Expansion

## Purpose

Day 10 expanded bounded low-rank and partial-SVD evidence using deterministic
rectangular fixtures. The changes remain in test code and do not change public
API, production source, build registration, package metadata, or workflow
surfaces.

## Code Surfaces

- `tests/test_svd.c`
- `tests/test_svd_partial_helpers.h`

## Partial-SVD Evidence

Added `test_partial_svd_vectors_rectangular_lowrank_recon` in
`tests/test_svd_partial_helpers.h`.

Fixture:

- 6x4 diagonal rectangular matrix with singular values `{9, 6, 3, 1}`.
- Partial SVD rank `k = 2`.

Assertions:

- Partial singular values are exactly `{9, 6}` within `1e-10`.
- Returned dimensions and vector buffers are populated.
- `A*v ~= sigma*u` residual is below `1e-10`.
- Rank-2 reconstruction Frobenius error is `sqrt(3^2 + 1^2) = sqrt(10)`.

## Low-Rank Evidence

Added `test_lowrank_rectangular_dense_sparse_consistency` in
`tests/test_svd.c`.

Fixture:

- 5x7 diagonal rectangular matrix with diagonal values `{8, 4, 2, 1, 0}`.
- Low-rank target `k = 3`.

Assertions:

- Dense low-rank reconstruction error is exactly `1.0` within `1e-10`.
- Sparse low-rank output with `drop_tol = 0.0` matches dense low-rank output
  with Frobenius difference `0.0` within `1e-10`.
- Sparse output dimensions remain 5x7.
- Kept diagonal entries `{8, 4, 2}` are present and the omitted fourth
  diagonal entry is zero.

## Non-Claims

- These are deterministic fixture proofs, not broad numerical-optimality
  claims.
- The tests do not claim external library parity.
- Dense-vs-sparse low-rank comparison is internal and only covers the
  zero-drop rectangular fixture.
- Partial-SVD reconstruction error is checked against the fixture's known
  omitted singular spectrum, not against arbitrary rank-k behavior.

## Focused Validation

Command:

```sh
make build/test_svd && ./build/test_svd
```

Result:

- `test_svd`: 103 tests, 0 failures, 0 skips, 1659 assertions.

## Full Quality Gate

Command:

```sh
make format && make lint && make test
```

Result:

- Passed.
- `make test` included the new rectangular partial-SVD and low-rank
  dense/sparse consistency tests inside `test_svd`.

## Deferred Queue

- Days 11-12 should design and optionally pilot one bounded dense-reference
  or external comparison lane.
- Broader partial-SVD helper extraction remains out of this sprint unless a
  later item needs it for shared evidence ownership.
