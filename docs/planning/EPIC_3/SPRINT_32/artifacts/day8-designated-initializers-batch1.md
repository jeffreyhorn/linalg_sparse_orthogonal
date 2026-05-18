# Sprint 32 Day 8 Designated Initializers Batch I

**Date:** 2026-05-18  
**Branch:** `sprint-32`

## Objective

Remove the first half of the Sprint 32 test-side designated-initializer warning queue by converting the LDLT-family positional `sparse_ldlt_opts_t` initializers in:

- `tests/test_ldlt.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint20_integration.c`

## Files Updated

- `tests/test_ldlt.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint20_integration.c`

## Changes

### `tests/test_ldlt.c`

Converted every remaining LDLT options initializer to designated form.

That covered:

- reorder/tolerance-only forms such as:
  - AMD reorder cases
  - RCM reorder cases
- backend-routing forms such as:
  - AUTO with `used_csc_path`
  - forced LINKED_LIST
  - forced CSC
  - invalid backend rejection
  - no-telemetry `used_csc_path == NULL`

This matters because the file exercised both major struct-growth eras:

- early positional `{reorder, tol}` forms that now warn on missing `backend`
- later positional backend/telemetry forms that now warn on missing `progress_cb`

### `tests/test_sprint12_integration.c`

Converted the LDLT integration options to designated form for:

- KKT pipeline with AMD
- reorder equivalence with AMD and RCM
- SuiteSparse `nos4` with AMD
- large KKT pipeline with AMD

These tests only intend to control reorder mode plus the explicit zero/default tolerance, so designated initialization makes that intent visible without binding the test to newer trailing fields.

### `tests/test_sprint20_integration.c`

Converted the backend-routing options to designated form for:

- AUTO below threshold
- AUTO above threshold on SPD
- AUTO above threshold on indefinite KKT
- forced LINKED_LIST on large matrix
- forced CSC on small matrix

These tests explicitly care about:

- `.backend`
- `.used_csc_path`

and do not care about the newer callback/context tail. The designated form now reflects that contract directly.

## Chosen Day 8 Style

Day 8 reuses the Sprint 31 designated-init rule:

- name only the fields the test intentionally overrides
- keep explicit `.used_csc_path` only where the test asserts routing telemetry
- leave callback/context fields at default `NULL`
- avoid positional mirroring of `sparse_ldlt_opts_t`

Representative Day 8 forms now look like:

- `{.reorder = SPARSE_REORDER_AMD, .tol = 0.0}`
- `{.reorder = SPARSE_REORDER_NONE, .tol = 0.0, .backend = SPARSE_LDLT_BACKEND_AUTO, .used_csc_path = &used_csc}`

## Validation

Validation commands:

- `make format`
- `make build/test_ldlt build/test_sprint12_integration build/test_sprint20_integration`
- `./build/test_ldlt`
- `./build/test_sprint12_integration`
- `./build/test_sprint20_integration`
- `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first`

Observed results:

- targeted build passed
- `test_ldlt` passed: `83` tests, `0` failed
- `test_sprint12_integration` passed: `8` tests, `0` failed
- `test_sprint20_integration` passed: `20` tests, `0` failed
- clean serialized CMake rebuild passed

## Warning Delta

Relative to the Day 7 baseline:

- full-tree warnings: `91 -> 63`
- `tests` warnings: `91 -> 63`
- `-Wmissing-field-initializers`: `58 -> 30`
- `-Wdouble-promotion`: unchanged at `33`

Per-file reduction:

- `tests/test_ldlt.c`: `18 -> 0`
- `tests/test_sprint12_integration.c`: `5 -> 0`
- `tests/test_sprint20_integration.c`: `9 -> 4`

Interpretation of `test_sprint20_integration.c`:

- all `5` LDLT initializer warnings are gone
- the file’s remaining `4` warnings are the already-planned double-promotion sites and belong to the later Sprint 32 promotion-cleanup pass

## Remaining Initializer Queue

After Day 8, the remaining designated-initializer queue is:

- `tests/test_chol_csc.c`: `8`
- `tests/test_colamd.c`: `7`
- `tests/test_cholesky.c`: `5`
- `tests/test_reorder.c`: `4`
- `tests/test_sprint18_integration.c`: `3`
- `tests/test_sprint19_integration.c`: `2`
- `tests/test_etree.c`: `1`

Total remaining `-Wmissing-field-initializers`: `30`

That matches the Day 7 plan exactly, so Day 9 can close the remaining initializer work without reopening the LDLT family.

## Conclusion

Day 8 cleanly closed the LDLT-family initializer batch:

- the touched LDLT tests now express only the fields they actually mean to override
- all three targeted binaries still pass
- the clean-build warning queue dropped by exactly the expected `28` initializer sites

Sprint 32 is now ready for the Day 9 companion batch on the remaining Cholesky, QR, and LU test files.
