# Sprint 84 Day 6: Direct-Family Differential Batch

## Purpose

Land the first maintained external differential proof batch on the bounded
direct-family SPD Cholesky lane.

## Main Result

Sprint 84 Day 6 landed one bounded maintained direct-family external
differential batch:

- required implementation center:
  - `tests/test_chol_csc.c`
- strongest support-only follow-through that was truly needed:
  - `tests/chol_external_dense_reference.py`
  - `docs/maintainer_guide.md`
- not needed in the batch:
  - `tests/test_chol_csc_supernodal_helpers.h`
  - `tests/test_framework.h`
  - `tests/test_fuzz.c`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `README.md`

## Landed Surface

The landed maintained external differential seam is explicit now:

- `test_external_dense_reference_nos4_csc`
- `test_external_dense_reference_bcsstk04_amd_csc`
- external dense reference helper invoked through:
  - `python3 tests/chol_external_dense_reference.py`

This is a proof-surface widening batch, not a production algorithm batch. It
keeps the first maintained external lane:

- test-owned
- fixture-backed
- family-local to the direct-family SPD Cholesky CSC path
- bounded to a pure-stdlib Python external process instead of a mandatory
  SciPy/CHOLMOD dependency stack

## Proof and Support Follow-Through

The strongest direct proof now lives in `tests/test_chol_csc.c`:

- `nos4` runs through the forced CSC Cholesky path and compares the solved
  vector against the external dense reference result
- `bcsstk04` runs through the AMD-reordered forced CSC Cholesky path and
  compares the solved vector against the external dense reference result
- both checks still assert a tight retained relative residual on the in-repo
  Cholesky solve itself

The authoritative maintainer-policy reading now matches the landing:

- `docs/maintainer_guide.md` treats `tests/test_chol_csc.c` as the bounded
  direct-family maintained external differential owner on the SPD Cholesky lane
- it keeps the lane explicit as family-local rather than implying repo-wide
  external-proof adoption

`README.md` did not require movement because its broader proof-surface wording
remained truthful after the batch.

## Strongest Clarification

The useful Day 6 clarification is explicit:

- Sprint 84 now has one real maintained external differential lane
- that lane is still bounded to the direct-family SPD Cholesky proof owner
- seeded-property widening remains separate follow-through work
- failure-path numerical proof remains separate follow-through work
- iterative/eigs external adoption remains later work
- benchmarks and examples still do not become oracle owners

## Validation

The Day 6 batch was validated with:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Representative retained outputs from the landed external differential tests:

- `nos4`: `max|x-x_ref| = 4.690e-13`, `rel_residual = 3.907e-15`
- `bcsstk04`: `max|x-x_ref| = 3.224e-11`, `rel_residual = 3.010e-16`

## Exit State

- Sprint 84 now has one landed bounded direct-family maintained external
  differential batch.
- The strongest missing assurance seam is no longer whether any maintained
  external proof exists on the direct-family SPD lane.
- Later sprint work can stay focused on reranking seeded-property expansion,
  failure-path numerical proof, and later-family external follow-through.
