# Day 10 Partial-SVD Evidence Package

## Purpose

Execute the Day 9 partial-SVD semantics decision by adding one bounded external value-only lane and explicitly deferring vector, subspace, convergence-budget, repeated/clustered spectrum, rank-threshold, and low-rank optimality evidence.

## Accepted Lane

| Field | Value |
| --- | --- |
| Fixture key | `partial_svd_tall_diag_8x5_k3` |
| Semantic class | Rectangular partial-SVD top-k singular values |
| Matrix shape | 8x5 |
| Matrix values | Diagonal entries `[8.0, 5.0, 3.0, 1.0, 0.25]` with three trailing zero rows |
| `k` | 3 |
| Expected helper output | `OK 3`, followed by `8`, `5`, `3` |
| Tolerance | Maximum absolute singular-value difference below `1e-8` |
| Explicit non-semantics | No vector, subspace, convergence-budget, low-rank, rank-threshold, performance, platform, or public API claim |

## Implementation Summary

- `tests/svd_external_dense_reference.py`
  - Added `build_partial_svd_tall_diag_8x5_k3`.
  - Added fixture dispatch for `partial_svd_tall_diag_8x5_k3`.
  - Limited the fixture output to the top three singular values.
- `tests/test_svd.c`
  - Added `partial_svd_tall_diag_8x5_k3` to the bounded SVD external-reference allow-list.
  - Registered the new partial-SVD external test in the existing `test_svd` suite.
- `tests/test_svd_partial_helpers.h`
  - Added `test_partial_svd_external_dense_reference_tall_diag_8x5_k3`.
  - Kept the test value-only by checking `partial.sigma`, dimensions, and `partial.U == NULL` / `partial.Vt == NULL`.

## Surfaces Not Changed

- No public API changes.
- No Makefile, CMake, or CTest membership changes.
- No package, platform, ABI, benchmark, or public documentation changes.
- No vector/subspace helper extraction.
- No low-rank helper or low-rank optimality changes.

## Helper Evidence

```text
$ python3 tests/svd_external_dense_reference.py partial_svd_tall_diag_8x5_k3
OK 3
8
5
3
```

## Focused Test Evidence

```text
$ make build/test_svd && ./build/test_svd
external partial-SVD dense ref tall_diag_8x5_k3: max |sigma-sigma_ref| = 2.665e-15
Tests run:    108
Tests failed: 0
Tests skipped: 0
Assertions:   1769
ALL TESTS PASSED
```

## Full Validation

```text
$ make format && make lint && make test
All tests passed.
```

Additional local hygiene checks:

- `git diff --check` passed.
- Focused trailing-whitespace scan over Sprint 123 artifacts and touched SVD/QR files passed.

## Deferrals Preserved

- Singular-vector external parity remains deferred until a sign-invariant residual protocol or projection metric is selected.
- Subspace external parity remains deferred until projection/principal-angle helpers exist.
- Repeated-spectrum and clustered-spectrum external evidence remains deferred because value ordering and vector identity are ambiguous without subspace semantics.
- Rank-deficient partial-SVD external evidence remains deferred until zero/near-zero threshold behavior is defined as the primary fixture claim.
- Convergence-budget external evidence remains deferred until options, iteration cap, tolerance, and failure interpretation are explicit.
- Low-rank optimality remains deferred to low-rank-specific proof ownership.

## Completion Criteria Status

- Partial-SVD evidence lane selected: complete.
- Accepted lane has value/vector/subspace/convergence meaning stated: complete; value-only.
- Bounded implementation landed with focused validation: complete.
- Broad partial-SVD external parity remains unsupported: complete.
