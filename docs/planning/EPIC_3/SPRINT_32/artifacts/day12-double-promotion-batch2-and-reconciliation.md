# Sprint 32 Day 12 Double-Promotion Cleanup Batch II and Reconciliation

**Date:** 2026-05-18  
**Branch:** `sprint-32`

## Objective

Finish the remaining Sprint 32 `-Wdouble-promotion` queue in `tests/test_svd.c` and reconcile the final warning state against the Day 1 baseline.

## Files Updated

- `tests/test_svd.c`

## Changes

Day 12 closed the final six Sprint 32 warning sites using the exact SVD-specific rules chosen on Day 10.

### Helper return sentinel

The reconstruction helper now uses the correct `double` sentinel:

- `return INFINITY;` -> `return HUGE_VAL;`

### NaN placeholder

The transposed-bidiag placeholder now uses the `double` NaN constructor:

- `double recon = NAN;` -> `double recon = nan("");`

This preserves the existing output behavior for the wide-path log line:

- `GK 5x10: recon=nan, U_orth=..., V_orth=...`

### Infinity assertions

The condition-number coverage now asserts positive infinity semantically:

- `ASSERT_TRUE(c == INFINITY);` -> `ASSERT_TRUE(isinf(c) && c > 0.0);`

That change landed in:

- rank-deficient singular condition-number coverage
- `1x1` zero-matrix condition-number coverage
- NULL-input condition-number coverage with and without the `err` out-pointer

No algorithm, tolerance, or user-facing condition-number behavior changed.

## Validation

Validation commands:

- `make format`
- `make build/test_svd`
- `./build/test_svd`
- `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first`

Observed results:

- targeted build passed
- `test_svd` passed
- clean serialized Apple Clang CMake rebuild passed

Direct-run summary:

- `test_svd`: `96` tests, `0` failed, `0` skipped

## Warning Delta

Relative to the Day 11 starting state:

- full-tree warnings: `6 -> 0`
- `tests` warnings: `6 -> 0`
- `-Wdouble-promotion`: `6 -> 0`

Per-file reduction:

- `tests/test_svd.c`: `6 -> 0`

## Final Sprint 32 Reconciliation

Relative to the Day 1 baseline:

- full-tree warnings: `98 -> 0`
- `src`: `0 -> 0`
- `tests`: `98 -> 0`
- `benchmarks`: `0 -> 0`
- `examples`: `0 -> 0`

Warning-class closure:

- `-Wmissing-field-initializers`: `62 -> 0`
- `-Wdouble-promotion`: `33 -> 0`
- `-Wunused-function`: `3 -> 0`

Residual warning queue after Day 12:

- none

## Conclusion

Sprint 32's warning-cleanup track is now fully closed.

The final state is stronger than the initial Day 1 target framing:

- there are no remaining test warnings
- there is no residual mixed-class warning debt
- later Sprint 32 validation can focus on proving the cleaned suite still passes rather than carrying forward unnamed cleanup work
