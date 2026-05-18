# Sprint 32 Day 11 Double-Promotion Cleanup Batch I

**Date:** 2026-05-18  
**Branch:** `sprint-32`

## Objective

Remove the non-SVD Sprint 32 `-Wdouble-promotion` queue by replacing helper-level `INFINITY` sentinels in `double` contexts with `HUGE_VAL`.

## Files Updated

- `tests/test_ilu.c`
- `tests/test_sprint5_integration.c`
- `tests/test_qr.c`
- `tests/test_sprint6_integration.c`
- `tests/test_bidiag.c`
- `tests/test_lu_csr.c`
- `tests/test_block_solvers.c`
- `tests/test_sprint10_integration.c`
- `tests/test_colamd.c`
- `tests/test_bicgstab.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_sprint20_integration.c`

## Changes

Day 11 applied the Day 10 Batch I cleanup rule directly:

- `return INFINITY;` -> `return HUGE_VAL;`
- `double ... = INFINITY;` -> `double ... = HUGE_VAL;`

That covered three shapes:

### Helper return sentinels

Residual and reconstruction helpers that return `double` on success and a sentinel on allocation or intermediate failure now use `HUGE_VAL`.

Representative files:

- `tests/test_ilu.c`
- `tests/test_qr.c`
- `tests/test_bidiag.c`
- `tests/test_lu_csr.c`
- `tests/test_block_solvers.c`
- `tests/test_colamd.c`
- `tests/test_bicgstab.c`
- `tests/test_sprint20_integration.c`

### Local `double` sentinel initialization

Integration tests that initialize local residual placeholders now use `HUGE_VAL`.

Representative files:

- `tests/test_sprint6_integration.c`
- `tests/test_sprint18_integration.c`

### Comment truthfulness cleanup

Two touched comments were updated so the source now describes the actual sentinel strategy:

- `tests/test_sprint20_integration.c`
  - helper comment now says `Returns HUGE_VAL`
- `tests/test_bicgstab.c`
  - helper comment now states the `HUGE_VAL` sentinel purpose directly

No numeric thresholds, solver options, or test expectations changed.

## Validation

Validation commands:

- `make format`
- `make build/test_ilu build/test_sprint5_integration build/test_qr build/test_sprint6_integration build/test_bidiag build/test_lu_csr build/test_block_solvers build/test_sprint10_integration build/test_colamd build/test_bicgstab build/test_sprint18_integration build/test_sprint19_integration build/test_sprint20_integration`
- `./build/test_ilu`
- `./build/test_sprint5_integration`
- `./build/test_qr`
- `./build/test_sprint6_integration`
- `./build/test_bidiag`
- `./build/test_lu_csr`
- `./build/test_block_solvers`
- `./build/test_sprint10_integration`
- `./build/test_colamd`
- `./build/test_bicgstab`
- `./build/test_sprint18_integration`
- `./build/test_sprint19_integration`
- `./build/test_sprint20_integration`
- `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first`

Observed results:

- targeted build passed
- all `13` touched binaries passed
- clean serialized Apple Clang CMake rebuild passed

Direct-run summaries:

- `test_ilu`: `34` tests, `0` failed
- `test_sprint5_integration`: `14` tests, `0` failed
- `test_qr`: `71` tests, `0` failed
- `test_sprint6_integration`: `7` tests, `0` failed
- `test_bidiag`: `12` tests, `0` failed
- `test_lu_csr`: `53` tests, `0` failed
- `test_block_solvers`: `15` tests, `0` failed
- `test_sprint10_integration`: `14` tests, `0` failed
- `test_colamd`: `70` tests, `0` failed
- `test_bicgstab`: `58` tests, `0` failed
- `test_sprint18_integration`: `10` tests, `0` failed
- `test_sprint19_integration`: `8` tests, `0` failed
- `test_sprint20_integration`: `20` tests, `0` failed

## Warning Delta

Relative to the Day 10 starting state:

- full-tree warnings: `33 -> 6`
- `tests` warnings: `33 -> 6`
- `-Wdouble-promotion`: `33 -> 6`

Per-file reduction:

- `tests/test_sprint6_integration.c`: `6 -> 0`
- `tests/test_sprint20_integration.c`: `4 -> 0`
- `tests/test_bidiag.c`: `3 -> 0`
- `tests/test_sprint18_integration.c`: `3 -> 0`
- `tests/test_qr.c`: `2 -> 0`
- `tests/test_sprint10_integration.c`: `2 -> 0`
- `tests/test_bicgstab.c`: `1 -> 0`
- `tests/test_block_solvers.c`: `1 -> 0`
- `tests/test_colamd.c`: `1 -> 0`
- `tests/test_ilu.c`: `1 -> 0`
- `tests/test_lu_csr.c`: `1 -> 0`
- `tests/test_sprint19_integration.c`: `1 -> 0`
- `tests/test_sprint5_integration.c`: `1 -> 0`

Residual queue after Day 11:

- `tests/test_svd.c`: `6`

## Conclusion

Day 11 closed the helper-sentinel cohort exactly as planned:

- all non-SVD Sprint 32 `-Wdouble-promotion` sites are gone
- touched integration, QR, LU CSR, BiCGSTAB, COLAMD, ILU, and helper-heavy tests still pass
- Sprint 32 now has a single-file Day 12 finish:
  - `tests/test_svd.c`
  - `6` remaining warning sites
