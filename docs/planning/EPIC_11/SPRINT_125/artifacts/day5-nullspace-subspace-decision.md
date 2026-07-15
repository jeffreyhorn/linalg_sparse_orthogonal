# Sprint 125 Day 5 Nullspace/Subspace Decision

## Decision

Accepted one bounded rank-deficient QR nullspace/subspace fixture:
`qr_rankdef_duplicate_5x4_nullspace_projector`.

The fixture reuses the 5x4 duplicate-column matrix from the completed
`qr_rankdef_duplicate_5x4_rank_only` and
`qr_rankdef_duplicate_5x4_residual_only` lanes.  It pins expected rank 3,
nullity 1, and threshold 0.0, then compares a normalized product nullspace
projector against an external standard-library reference projector.

## Fixture Contract

| Field | Value |
| --- | --- |
| Fixture key | `qr_rankdef_duplicate_5x4_nullspace_projector` |
| Matrix shape | 5 rows by 4 columns |
| Expected rank | 3 |
| Expected nullity | 1 |
| Threshold | 0.0 |
| External helper output | `OK 20` |
| Metadata entries | `n`, `rank`, `nullity`, `threshold` |
| Projector entries | 16 column-major entries for `P = z z^T` |
| Reference null vector | `[0, -1/sqrt(2), 0, 1/sqrt(2)]` |

## Implemented Evidence

- Added `nullspace_projector_reference()` to
  `tests/qr_external_dense_reference.py`.
- Added external-helper dispatch for
  `qr_rankdef_duplicate_5x4_nullspace_projector`.
- Added `test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector()`
  to `tests/test_qr.c`.
- Normalized the product nullspace vector before computing the projector.
- Compared projector entries against the external reference with
  `max_projector_diff < 1e-8`.
- Kept `||A*v|| < 1e-10` as a diagnostic and secondary correctness check.
- Updated `docs/maintainer_guide.md` to name the bounded nullspace projector
  fixture without broadening the QR public claim.

## Non-Claims Preserved

- No raw nullspace vector equality claim.
- No Q-basis, Q-sign, Q-orientation, or basis-ordering claim.
- No broad QR nullspace parity claim.
- No minimum-norm, pseudoinverse, or QR-vs-SVD-pseudoinverse claim.
- No economy-mode, sparse-mode, reorder, SuiteSparse corpus, backend,
  performance, platform, LAPACK, NumPy, or SciPy parity claim.

## Deferred Nullspace/Subspace Work

| Deferred lane | Reason | Future gate |
| --- | --- | --- |
| Dependent-row projector | Needs an independent projector fixture rather than reusing residual-only evidence. | Add a named fixture with rank/nullity metadata, external projector, and null residual diagnostics. |
| 3x5 multi-dimensional nullspace | Requires subspace metric for nullity greater than 1. | Compare full projectors or two-way projection residuals, not ordered basis vectors. |
| Wide rank-deficient nullspace | Needs separate wide-shape policy and failure interpretation. | Prove expected rank/nullity and projector tolerance before accepting evidence. |
| Near-rank threshold nullspace | Depends on Day 6-7 threshold-family policy. | Pin thresholds and expected rank/nullity per fixture. |
| SuiteSparse nullspace/subspace | Depends on Day 8-9 corpus policy. | Establish skip semantics, fixture ownership, and support-tier wording. |

## Day 6 Inputs

- Use projector/subspace metrics for any threshold-family nullspace evidence.
- Keep threshold claims fixture-local; do not introduce a global QR rank
  threshold.
- If a near-rank fixture changes expected nullity across thresholds, publish
  separate threshold metadata and expected projector/nullity outcomes for each
  accepted threshold.

## Validation

Focused validation passed:

```text
python3 -m py_compile tests/qr_external_dense_reference.py
python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_nullspace_projector
make build/test_qr && ./build/test_qr
```

Full required quality validation passed:

```text
make format && make lint && make test
```

