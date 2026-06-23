# Sprint 85 Day 9: Direct-Family Hotspot Batch

## Purpose

Land the bounded mixed-ownership cleanup fixed on Day 8 by moving the dense
LDL^T/backend seam out of `src/sparse_chol_csc.c` and into the LDL^T CSC owner.

## Main Result

The Day 9 landing stayed inside the Day 8 fence:

- required implementation center:
  - `src/sparse_chol_csc.c`
- directly forced support surfaces actually needed:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
- not needed in the batch:
  - `docs/maintainer_guide.md`
  - `README.md`
  - giant-test architecture cleanup
  - adjacent source cleanups in `src/sparse_qr.c` or `src/sparse_ldlt.c`

## Landed Surface

The landed batch moved the dense LDL^T primitive and bounded backend-selection
seam from the Cholesky CSC owner to the LDL^T CSC owner.

Moved implementation ownership:

- `ldlt_dense_factor`
- `ldlt_dense_factor_selected`
- `ldlt_dense_factor_backend_name`
- the associated Accelerate probe and backend-selection helpers

Moved declaration ownership:

- from `src/sparse_chol_csc_internal.h`
- to `src/sparse_ldlt_csc_internal.h`

Minimal proof-owner follow-through:

- `tests/test_chol_csc.c` now includes `sparse_ldlt_csc_internal.h` for the
  retained dense-LDL^T cross-checks it already owns
- `tests/test_ldlt.c` now depends on `sparse_ldlt_csc_internal.h` instead of
  the unrelated Cholesky internal header

## Strongest Clarification

The useful Day 9 clarification is explicit now:

- this was a real direct-family maintainability reduction, not a generic
  Cholesky refactor
- the batch reduced the `src/sparse_chol_csc.c` hotspot by removing a
  non-Cholesky ownership block rather than redistributing arbitrary helpers
- the LDL^T CSC owner now directly owns the dense primitive/backend seam that
  its supernodal path and env-contract tests already treat as family-local
- giant-test architecture cleanup remains later Sprint 85 work

## Preserved Non-Goal Fence

The preserved bounded-cleanup reading held:

- no Cholesky CSC elimination rewrite
- no family-wide public/API widening
- no giant-test registration cleanup
- no maintainer-guide or README movement
- no benchmark/example ownership drift

## Validation

The landed batch passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed parity remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`

Reviewed runtime note:

- `test_reorder_nd` remained the long tail at `254.28 sec`
- reviewed CMake `Total Test time (real)` = `351.72 sec`

## Exit State

- Sprint 85 now has one landed bounded direct-family source cleanup batch.
- The Cholesky CSC hotspot no longer owns the LDL^T dense/backend seam.
- Giant-test architecture cleanup remains the strongest later Sprint 85 seam.
