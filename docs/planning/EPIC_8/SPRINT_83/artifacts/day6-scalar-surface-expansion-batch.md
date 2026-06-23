# Sprint 83 Day 6: Scalar-Surface Expansion Batch

## Purpose

Land the first bounded Sprint 83 capability batch on the shared matrix-shell
public seam without widening the shipped scalar contract beyond real-only
`double`.

## Main Result

Sprint 83 Day 6 landed one bounded shared-surface batch:

- required implementation center:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- strongest support-only follow-through that was truly needed:
  - `tests/test_sparse_matrix.c`
  - `docs/maintainer_guide.md`
- not needed in the batch:
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
  - `README.md`

## Landed Surface

The landed shared matrix-shell public seam now uses `sparse_scalar_t` across:

- insert/get/set helpers
- symmetry tolerance input
- norm output
- matvec and block-matvec dense vectors
- scale and add helpers

This is a vocabulary and ownership widening batch, not a behavior-widening
batch. The shipped scalar contract remains real-only `double` because
`sparse_scalar_t` is still that exact underlying type.

## Proof and Support Follow-Through

The strongest direct proof now lives in `tests/test_sparse_matrix.c`:

- `sparse_scalar_bits()` remains coherent with `sizeof(sparse_scalar_t)`
- `sparse_insert(...)` accepts the widened shared public scalar vocabulary
- `sparse_matvec(...)` and `sparse_norminf(...)` preserve expected numeric
  results on the widened seam
- `sparse_scale(...)` preserves expected numeric behavior on the widened seam

The authoritative maintainer-policy reading now matches the landing:

- `docs/maintainer_guide.md` treats `sparse_scalar_t` as the dense-scalar
  owner on the shared matrix-shell helper seam in addition to the already-real
  iterative/eigs seam

`README.md` did not require movement because its broader capability wording
remained truthful after the batch.

## Strongest Clarification

The useful Day 6 clarification is explicit:

- the first Sprint 83 batch was the shared matrix-shell scalar/public-owner
  seam, not family-local algorithm widening
- compatibility-preserving behavior stayed centered in `src/sparse_matrix.c`
- QR, SVD, Cholesky, LDL^T, complex support, mixed precision, and broader
  package/platform maturity remain outside this batch

## Validation

The Day 6 batch was validated with:

- `make format`
- `make lint`
- `make test`

## Exit State

- Sprint 83 now has one landed shared scalar-surface expansion batch.
- The highest-value shared matrix-shell seam no longer reads as a raw `double`
  outlier relative to the already-real `sparse_scalar_t` vocabulary.
- Later sprint work can stay focused on touched-path width/ABI follow-through
  and bounded family-local capability widening instead of reopening the shared
  matrix-shell scalar owner.
