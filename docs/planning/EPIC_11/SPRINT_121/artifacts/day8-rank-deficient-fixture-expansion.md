# Sprint 121 Day 8 - Rank-Deficient Fixture Expansion

## Purpose

Day 8 expanded deterministic rank-deficient evidence for the QR and SVD proof
lanes without changing production code, build registration, package metadata,
or workflow surfaces.

## Code Surfaces

- `tests/test_qr_helpers.h`
- `tests/test_qr.c`
- `tests/test_svd_helpers.h`
- `tests/test_svd.c`

## Fixture Additions

### QR

- Added `tf_qr_make_dependent_row_4x3`, a deterministic 4x3 fixture whose
  rows are combinations of two independent row vectors.
- Added `tf_qr_make_diag_matrix`, a shared QR diagonal fixture builder for
  threshold-sensitive rank evidence.
- Added `test_qr_rank_dependent_row_fixture` to assert:
  - QR rank is exactly 2.
  - `sparse_qr_rank(..., 0.0)` is exactly 2.
  - QR reconstruction remains below `1e-10`.
  - Null-space dimension is exactly 1 and the returned basis satisfies
    `||A*v||_2 < 1e-10`.
- Added `test_qr_rank_diagonal_threshold_fixture` to assert exact rank
  thresholds for diagonal values `{1.0, 1e-8, 1e-12, 0.0}`:
  - `rank(1e-14) = 3`
  - `rank(1e-10) = 2`
  - `rank(1e-6) = 1`

### SVD

- Added `tf_svd_make_dependent_row_4x3`, matching the QR dependent-row fixture
  shape and exact rank.
- Added `test_svd_rank_diagonal_threshold_fixture` to assert the same diagonal
  threshold-rank behavior through `sparse_svd_rank`.
- Added `test_svd_qr_rank_dependent_row_fixture` to assert SVD and QR agree
  that the dependent-row fixture has rank 2 at tolerance `1e-10`.

## Tolerance Rationale

- The diagonal-threshold fixtures use exact diagonal entries so the expected
  ranks are owned by the explicit tolerance cutoffs, not by incidental
  conditioning or corpus behavior.
- The dependent-row fixture uses exact linear dependencies and keeps
  null-space residual assertions at `1e-10`, matching nearby QR reconstruction
  and null-space evidence.
- The SVD/QR cross-check uses `1e-10` to keep the proof lane explicit and
  avoid treating default tolerance behavior as a hidden oracle.

## Focused Validation

Command:

```sh
make build/test_qr build/test_svd && ./build/test_qr && ./build/test_svd
```

Result:

- `test_qr`: 65 tests, 0 failures, 0 skips, 603 assertions.
- `test_svd`: 100 tests, 0 failures, 0 skips, 1605 assertions.

## Full Quality Gate

Command:

```sh
make format && make lint && make test
```

Result: passed.

Additional checks:

- `git diff --check`: passed.
- Focused trailing-whitespace scan over Sprint 121 artifacts and modified
  QR/SVD test surfaces: passed.

## Deferred Queue

- Day 9 should apply the rank fixture taxonomy to compatible, incompatible,
  and minimum-norm least-squares proof lanes.
- Day 10 should decide whether partial-SVD vector helpers or partial-rank
  fixture coverage can be expanded without broadening public claims.
- Dense-reference comparison lanes remain queued for Days 11-12.
