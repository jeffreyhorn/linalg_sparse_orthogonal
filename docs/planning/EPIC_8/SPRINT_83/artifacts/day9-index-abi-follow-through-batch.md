# Sprint 83 Day 9: Index / ABI Follow-Through Batch

## Purpose

Reconcile the shared scalar/width vocabulary owner with the Day 6 landed
matrix-shell scalar seam so the public contract reads consistently without
reopening matrix-shell implementation work or broader family-level capability
surfaces.

## Main Result

Sprint 83 Day 9 landed one bounded shared-vocabulary batch:

- required implementation center:
  - `include/sparse_types.h`
- strongest support-only surfaces that were rechecked:
  - `tests/test_sparse_matrix.c`
  - `README.md`
  - `docs/maintainer_guide.md`
- not needed in the batch:
  - `include/sparse_matrix.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `src/sparse_matrix.c`

## Landed Contract

The landed reconciliation is explicit now:

- `include/sparse_types.h` no longer describes `sparse_scalar_t` primarily as
  the iterative/eigs dense-scalar seam
- it now treats the shared matrix-shell helper seam plus the iterative/eigs
  public scalar seams as the active public-owner surface
- `SPARSE_SCALAR_BITS` and `sparse_scalar_bits()` now describe that widened
  shared owner truthfully

This remains a public contract and vocabulary reconciliation batch, not a
numeric behavior widening batch. The shipped scalar contract still remains
real-only `double`.

## Proof and Support Follow-Through

The strongest support-only follow-through was smaller than the Day 8 design
fence:

- `tests/test_sparse_matrix.c` already owned the strongest direct proof for
  the landed shared matrix-shell scalar seam and did not need movement
- `docs/maintainer_guide.md` already matched the post-Day-6 owner split and
  did not need movement
- `README.md` already remained broadly truthful and did not need movement

## Strongest Clarification

The useful Day 9 clarification is explicit:

- the residual contradiction after Day 6 was shared-vocabulary reading, not
  matrix-shell implementation behavior
- the shared scalar owner now reads consistently across the common vocabulary
  header and the already-landed matrix-shell helper seam
- QR/SVD and other family-level capability widening remain later Sprint 83
  work, not part of this batch

## Validation

The Day 9 batch was validated with:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

## Exit State

- Sprint 83 now has one landed shared-vocabulary reconciliation batch.
- The strongest residual contradiction after Day 6 is closed without
  reopening matrix-shell implementation work.
- Later Sprint 83 work can move to bounded algorithm-family widening instead
  of circling back on the shared scalar owner.
