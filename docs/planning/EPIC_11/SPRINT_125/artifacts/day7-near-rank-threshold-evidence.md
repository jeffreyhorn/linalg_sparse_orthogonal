# Sprint 125 Day 7 Near-Rank-Deficient Threshold Evidence

## Decision

Accepted one bounded near-rank-deficient QR threshold fixture:
`qr_rank_threshold_diag4_family`.

The fixture is rank-only and threshold-only.  It checks a 4x4 diagonal bucket
ladder at named relative tolerances and does not assert residual, nullspace,
minimum-norm, pseudoinverse, economy, sparse-mode, reorder, backend, corpus,
performance, or global rank-threshold behavior.

## Accepted Fixture

| Field | Value |
| --- | --- |
| Fixture key | `qr_rank_threshold_diag4_family` |
| Matrix | 4x4 diagonal |
| Diagonal values | `[1, 1e-8, 1e-12, 0]` |
| Thresholds | `1e-14`, `1e-10`, `1e-6` |
| Expected ranks | `3`, `2`, `1` |
| External helper output | `OK 6` |
| Output semantics | threshold/rank pairs |
| Product owner | `tests/test_qr.c` |
| Helper owner | `tests/qr_external_dense_reference.py` |

## Implemented Evidence

- Added `build_qr_rank_threshold_diag4_family()` to
  `tests/qr_external_dense_reference.py`.
- Added `threshold_rank_reference()` to emit the threshold/rank pairs using
  standard-library arithmetic and the strict `abs(value) > threshold` rule.
- Added `read_qr_threshold_external_reference()` to `tests/test_qr.c`.
- Added `test_qr_external_dense_reference_rank_threshold_diag4_family()` to
  compare `sparse_qr_rank()` and `sparse_qr_rank_info()` against the external
  threshold/rank pairs.
- Printed threshold diagnostics for each row: fixture key, relative
  threshold, absolute threshold, expected rank, product rank, rank-info rank,
  and `R` diagonal magnitudes.
- Updated `docs/maintainer_guide.md` to name the bounded threshold-rank
  fixture while preserving the no-global-rank-threshold non-claim.

## Deferred Threshold Families

| Deferred family | Reason | Promotion gate |
| --- | --- | --- |
| Scaled diagonal bucket ladder | The unscaled ladder now proves the core threshold/rank protocol; scale evidence should remain separate. | Add scale metadata and prove ranks remain unchanged under named scales. |
| Perturbed duplicate-column family | QR pivoting and roundoff make threshold diagnostics harder than the diagonal ladder. | Define perturbation sizes separated from thresholds by at least two orders of magnitude. |
| Dependent-row near-threshold family | Mixes row-space perturbation with residual/nullspace interpretation. | Define whether rank, residual, or nullspace is the primary claim before implementation. |
| Wide near-threshold family | Nullity changes create subspace policy risk. | Accept rank-threshold evidence first, then add fixture-local nullity/subspace gates. |
| SuiteSparse near-threshold corpus | Depends on optional-corpus support-tier and platform skip policy. | Defer to Day 8-9 SuiteSparse rank-deficient QR corpus policy. |

## Non-Claims Preserved

- No global QR rank-threshold policy.
- No LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  or dense-library threshold parity.
- No broad rank-deficient QR parity.
- No residual correctness claim.
- No raw nullspace basis or Q-basis equality claim.
- No minimum-norm or pseudoinverse behavior claim.
- No economy-mode, sparse-mode, reorder, backend, corpus, platform,
  performance, package, ABI, public API, or CI behavior claim.

## Validation

Focused validation passed:

```text
python3 -m py_compile tests/qr_external_dense_reference.py
python3 tests/qr_external_dense_reference.py qr_rank_threshold_diag4_family
make build/test_qr && ./build/test_qr
```

Full required quality validation passed:

```text
make format && make lint && make test
```

The focused QR run passed 68 tests, 0 failures, 0 skips, and 669 assertions.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 4 is complete or explicitly deferred. | Complete | Accepted the diagonal threshold fixture and deferred lower-priority families. |
| Accepted code/helper changes have focused validation evidence. | Complete | See focused validation commands and QR summary. |
| Threshold proof boundaries remain documented. | Complete | See accepted fixture, deferred families, and non-claims. |

