# Sprint 84 Day 9: Seeded Property Expansion Batch

## Purpose

Land the bounded deterministic seeded-property expansion batch on the retained
large-`n` direct-family lifecycle owner.

## Main Result

Sprint 84 Day 9 landed one bounded deterministic property-expansion batch:

- required implementation center:
  - `tests/test_fuzz.c`
- strongest support-only follow-through that was not needed:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `docs/maintainer_guide.md`
  - `README.md`

## Landed Surface

The landed deterministic property widening is explicit now:

- helper:
  - `property_assert_rel_residual_small`
- new Cholesky property:
  - `test_property_large_n_cholesky_csc_reorder_repeat_solve_agreement`
- new LDL^T property:
  - `test_property_large_n_ldlt_csc_reorder_repeat_solve_agreement`

This is a proof-surface widening batch, not a production algorithm batch. It
keeps the Day 9 move:

- test-owned
- deterministic and seed-backed
- bounded to the retained large-`n` direct-family lifecycle proof owner
- explicit about reorder agreement, repeated-solve invariance, and residual
  smallness

## Proof Widening

The new Cholesky property proves, on retained large-`n` CSC-backed SPD public
lifecycle flows:

- `SPARSE_REORDER_NONE` and `SPARSE_REORDER_AMD` agree on the solved vector
- repeated solves on the same analyzed/factored state stay numerically aligned
- same-pattern refactor followed by repeated solve preserves that agreement
- retained relative residuals remain small on both reorder lanes

The new LDL^T property proves the same bounded invariants on retained large-`n`
CSC-backed indefinite public lifecycle flows:

- reorder agreement across `NONE` vs `AMD`
- repeated-solve invariance on the same factor state
- same-pattern refactor agreement
- retained relative residual smallness on both lanes

Together with the pre-existing Day 8 owner surface, `tests/test_fuzz.c` now
covers:

- small random LU / Cholesky / QR / SVD properties
- large-`n` Cholesky CSC public lifecycle same-pattern alignment
- large-`n` LDL^T CSC public lifecycle same-pattern alignment
- large-`n` Cholesky reorder / repeated-solve / residual invariants
- large-`n` LDL^T reorder / repeated-solve / residual invariants

## Strongest Clarification

The useful Day 9 clarification is explicit:

- Sprint 84 now has deeper deterministic lifecycle/property coverage on the
  retained direct-family large-`n` public lifecycle seam
- the strongest next assurance seam is no longer "add more direct-family
  property depth inside `tests/test_fuzz.c`"
- later failure-path numerical proof remains separate work
- later iterative/eigs external adoption remains separate work
- support surfaces did not need to move just because the property owner grew

## Validation

The Day 9 batch was validated with:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`

Representative retained outputs:

- `test_fuzz` = `28 / 28`, `20544` assertions
- reviewed CMake `test_fuzz` passed in `29.71 sec`
- reviewed CMake `Total Test time (real)` = `454.57 sec`

## Exit State

- Sprint 84 now has one landed bounded deterministic seeded-property expansion
  batch.
- The retained large-`n` direct-family lifecycle owner now proves reorder
  agreement, repeated-solve invariance, and residual smallness in addition to
  the earlier same-pattern public-vs-one-shot alignment.
- Later failure-path numerical proof remains the next assurance seam.
