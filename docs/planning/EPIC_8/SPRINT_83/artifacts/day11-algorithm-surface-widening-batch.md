# Sprint 83 Day 11: Algorithm-Surface Widening Batch

## Goal

Land the focused QR public-header follow-through justified by the Day 6
matrix-shell scalar widening and the Day 9 shared-vocabulary reconciliation,
without reopening SVD or direct-family public-family work.

## Required Landing Center

- `include/sparse_qr.h`

## Support-Only Surfaces Moved

- `tests/test_qr.c`
- `docs/maintainer_guide.md`

## Non-Touch Surfaces Preserved

- `include/sparse_svd.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `src/sparse_qr.c`
- `tests/test_svd.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`
- `README.md`

## Batch Outcome

The Day 11 landing stayed bounded and capability-truthful:

- `include/sparse_qr.h` no longer publishes the strongest caller-owned QR
  vectors and dense helper outputs purely as raw `double`
- the QR public seam now routes the highest-value caller-owned buffers and
  helper outputs through `sparse_scalar_t`
- `tests/test_qr.c` now proves that widened public scalar seam directly
- `docs/maintainer_guide.md` now names the QR proof-owner surface explicitly
- the shipped scalar contract still remains real-only `double`

## Important Clarifications

- This batch widens the public QR owner vocabulary, not the underlying numeric
  capability claim.
- Sprint 83 still does not claim complex QR, broad mixed precision, or a QR
  implementation rewrite.
- Tolerances, condition estimation, and rank-threshold interpretation remain
  real-valued diagnostics.
- No `src/sparse_qr.c` implementation churn was required because
  `sparse_scalar_t` still aliases the shipped `double` lane.

## Validation

Because public `*.h` and `*.c` proof-owner surfaces changed, I ran:

- `make format`
- `make lint`
- `make test`

All passed.
