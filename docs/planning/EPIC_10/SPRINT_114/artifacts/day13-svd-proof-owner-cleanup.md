# Sprint 114 Day 13: SVD Proof-Owner Cleanup

## Purpose

Day 13 cleans a bounded batch of SVD proof-owner tests without creating a
broad SVD proof abstraction. The cleanup reduces repeated loop mechanics for
reconstruction, orthogonality, Moore-Penrose products, low-rank residuals, and
condition-number checks while preserving the dimensions, storage conventions,
fixture values, tolerances, and expected claims at the call sites.

## Implemented Cleanup

| Area | File | Change |
|---|---|---|
| Reconstruction by storage contract | `tests/test_svd.c` | Added storage-explicit reconstruction helpers for max error and relative Frobenius residual, then used them in economy and full-mode reconstruction tests. |
| U/Vt orthogonality by leading dimension | `tests/test_svd.c` | Added a Vt row-orthogonality helper that requires explicit row count, column count, and leading dimension. Existing U orthogonality remains through `orthogonality_error`. |
| Moore-Penrose product dimensions | `tests/test_svd.c` | Added `svd_pinv_first_moore_penrose_error` for the first identity `A * A+ * A ≈ A`, with `m` and `n_cols` still visible in each test. |
| Dense low-rank proof loops | `tests/test_svd.c` | Added a dense low-rank Frobenius residual helper and used it for diagonal and theoretical-bound low-rank tests. |
| Sparse low-rank proof loops | `tests/test_svd.c` | Added sparse-vs-dense and sparse-vs-sparse Frobenius helpers for dense baseline comparisons and outer-product corpus checks. |
| Condition-number proof logic | `tests/test_svd.c` | Added finite and infinite condition-number assertion helpers while leaving each fixture and expected value visible at the call site. |

## Proof Values Preserved

- Economy SVD reconstruction keeps the `U` leading dimension and `Vt` leading
  dimension visible at call sites.
- Full SVD reconstruction and orthogonality keep the full-mode `U` and `Vt`
  leading dimensions explicit.
- Moore-Penrose tests still state the product dimensions:
  `A(m x n_cols) * A+(n_cols x m) * A(m x n_cols) -> A(m x n_cols)`.
- Dense low-rank tests still expose rank, diagonal values, theoretical
  residuals, and Frobenius tolerances.
- Sparse low-rank tests still expose fixture names, ranks, drop tolerances,
  and corpus dimensions.
- Condition-number tests still expose finite expected values, infinite
  singular cases, and rectangular interpretation at the call sites.

## Non-Claims

- No public SVD API changed.
- No install header, source-list, helper-target, Make, CMake, or reviewed CTest
  membership changed.
- No SVD helper moved out of `tests/test_svd.c`.
- No broad SVD proof abstraction is claimed; the helpers remain file-local and
  storage-contract-specific.

## Focused Validation

Focused validation passed:

```text
make build/test_svd && ./build/test_svd
```

Observed focused summary:

- `test_svd`: `98` tests, `0` failures, `1580` assertions.

## Required Full Gate

Day 13 modifies `.c` tests, so the required final gate is:

```text
make format && make lint && make test
```

## Completion Criteria

- SVD cleanup preserves all storage and dimension conventions.
- Focused SVD tests pass.
- No broad SVD proof abstraction is claimed.
- Full quality gate is required before the day is complete.
