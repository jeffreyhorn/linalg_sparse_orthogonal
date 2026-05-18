# Sprint 32 Day 9 Designated Initializers Batch II

**Date:** 2026-05-18  
**Branch:** `sprint-32`

## Objective

Finish the Sprint 32 designated-initializer cleanup by converting the remaining Cholesky, QR, and LU option-struct sites in:

- `tests/test_chol_csc.c`
- `tests/test_cholesky.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_colamd.c`
- `tests/test_reorder.c`
- `tests/test_etree.c`

## Files Updated

- `tests/test_chol_csc.c`
- `tests/test_cholesky.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_colamd.c`
- `tests/test_reorder.c`
- `tests/test_etree.c`

## Changes

### Cholesky family

Files:

- `tests/test_chol_csc.c`
- `tests/test_cholesky.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`

Converted the remaining Cholesky options forms to designated initialization.

That covered:

- reorder-only forms
- AUTO backend routing with telemetry
- forced LINKED_LIST / CSC path comparisons
- the legacy Day 11 dispatch-compatibility check

The highest-signal cleanup here was `tests/test_chol_csc.c`:

- the old positional `{use_amd ? SPARSE_REORDER_AMD : SPARSE_REORDER_NONE, 0.0}` form no longer depends on `0` mapping to `SPARSE_CHOL_BACKEND_AUTO`
- the test now says what it actually means: override only `.reorder` and leave backend/telemetry/callback fields at default values

### QR family

File:

- `tests/test_colamd.c`

Converted all remaining QR options to designated form.

Representative cases:

- reorder-only forms now set only `.reorder`
- sparse-mode coverage now sets:
  - `.reorder = SPARSE_REORDER_COLAMD`
  - `.sparse_mode = 1`

This leaves callback/context fields at default `NULL` while keeping the sparse-mode contract explicit.

### LU family

Files:

- `tests/test_reorder.c`
- `tests/test_etree.c`

Converted the remaining LU options to designated form:

- `.pivot`
- `.reorder`
- `.tol`

These tests only intend to control pivoting, reorder mode, and tolerance. The designated form now states that directly instead of depending on the current trailing callback/context layout.

## Validation

Validation commands:

- `make format`
- `make build/test_chol_csc build/test_cholesky build/test_sprint18_integration build/test_sprint19_integration build/test_colamd build/test_reorder build/test_etree`
- `./build/test_chol_csc`
- `./build/test_cholesky`
- `./build/test_sprint18_integration`
- `./build/test_sprint19_integration`
- `./build/test_colamd`
- `./build/test_reorder`
- `./build/test_etree`
- `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first`

Observed results:

- targeted build passed
- all seven touched binaries passed
- clean serialized Apple Clang CMake rebuild passed

Direct-run summaries:

- `test_chol_csc`: `137` tests, `0` failed
- `test_cholesky`: `21` tests, `0` failed
- `test_sprint18_integration`: `10` tests, `0` failed
- `test_sprint19_integration`: `8` tests, `0` failed
- `test_colamd`: `70` tests, `0` failed
- `test_reorder`: `38` tests, `0` failed
- `test_etree`: `94` tests, `0` failed

## Warning Delta

Relative to the Day 8 baseline:

- full-tree warnings: `63 -> 33`
- `tests` warnings: `63 -> 33`
- `-Wmissing-field-initializers`: `30 -> 0`
- `-Wdouble-promotion`: unchanged at `33`

Per-file reduction:

- `tests/test_chol_csc.c`: `8 -> 0`
- `tests/test_colamd.c`: `8 -> 1`
- `tests/test_cholesky.c`: `5 -> 0`
- `tests/test_reorder.c`: `4 -> 0`
- `tests/test_sprint18_integration.c`: `6 -> 3`
- `tests/test_sprint19_integration.c`: `3 -> 1`
- `tests/test_etree.c`: `1 -> 0`

Interpretation of the mixed files:

- `test_colamd.c` now retains only its single planned `-Wdouble-promotion` site
- `test_sprint18_integration.c` now retains only its three planned `-Wdouble-promotion` sites
- `test_sprint19_integration.c` now retains only its single planned `-Wdouble-promotion` site

## End State

Sprint 32’s designated-initializer cleanup is now complete.

Compared to the Day 1 baseline:

- `-Wmissing-field-initializers`: `62 -> 0`
- full-tree warnings: `98 -> 33`

The remaining warning queue is now pure double-promotion debt:

- `tests/test_sprint6_integration.c`: `6`
- `tests/test_svd.c`: `6`
- `tests/test_sprint20_integration.c`: `4`
- `tests/test_bidiag.c`: `3`
- `tests/test_sprint18_integration.c`: `3`
- `tests/test_qr.c`: `2`
- `tests/test_sprint10_integration.c`: `2`
- `tests/test_bicgstab.c`: `1`
- `tests/test_block_solvers.c`: `1`
- `tests/test_colamd.c`: `1`
- `tests/test_ilu.c`: `1`
- `tests/test_lu_csr.c`: `1`
- `tests/test_sprint19_integration.c`: `1`
- `tests/test_sprint5_integration.c`: `1`

## Conclusion

Day 9 closed the remaining initializer queue exactly as planned:

- all residual `-Wmissing-field-initializers` sites are gone
- the touched Cholesky, QR, LU, and companion integration binaries still pass
- Sprint 32 can now move cleanly to the double-promotion cleanup phase with no initializer debt left behind
