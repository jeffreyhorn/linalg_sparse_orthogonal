# Day 8 QR Evidence Implementation

## Purpose

Implement the Day 6 accepted compatible tall QR external reference fixture while preserving the Day 7 deferrals for rank-deficient QR, underdetermined minimum-norm QR, and Q/economy evidence.

## Implemented Fixture Contract

- Fixture key: `qr_overdetermined_compatible_5x3`
- Matrix shape: 5x3, full column rank, overdetermined and compatible
- Expected solution: `[1.0, -2.0, 0.5]`
- Right-hand side: `[2.0, -2.5, 4.0, -0.5, 2.0]`
- Helper output shape: `OK 4`, followed by three solution entries and one residual norm
- Acceptance thresholds:
  - Maximum solution-entry difference below `1e-8`
  - Residual-norm difference below `1e-8`
  - Computed residual below `1e-8`

## Surfaces Changed

- `tests/qr_external_dense_reference.py`
  - Added `qr_overdetermined_compatible_5x3`.
  - Generalized the normal-equation reference solve from the prior 2x2-only helper to a bounded dense Gaussian-elimination helper for small reference systems.
- `tests/test_qr_solve.c`
  - Added the fixture key to the QR external reference allow-list.
  - Added `test_qr_external_dense_reference_overdetermined_compatible_5x3`.
  - Registered the new test in the existing `test_qr_solve` suite.

## Surfaces Not Changed

- No `tests/test_qr.c` changes.
- No Makefile, CMake, or CTest surface changes.
- No public API, public documentation, package, platform, ABI, or performance-claim changes.
- No claim that rank-deficient QR, minimum-norm QR, or Q/economy external oracle coverage is complete.

## Helper Evidence

```text
$ python3 tests/qr_external_dense_reference.py qr_overdetermined_compatible_5x3
OK 4
1
-2
0.5
0
```

## Focused Test Evidence

```text
$ make build/test_qr_solve && ./build/test_qr_solve
external QR dense ref overdetermined_compatible_5x3: solution diff = 4.441e-16, residual diff = 9.930e-16
Tests run:    15
Tests failed: 0
Tests skipped: 0
Assertions:   1042
ALL TESTS PASSED
```

## Full Validation

```text
$ make format && make lint && make test
All tests passed.
```

Additional local hygiene checks:

- `git diff --check` passed.
- Focused trailing-whitespace scan over Sprint 123 artifacts and touched QR/SVD files passed.

## Explicit Deferrals Preserved

- Rank-deficient QR external reference evidence remains deferred because the current QR external helper contract is least-squares residual oriented and does not yet encode rank-threshold, nullspace, or pseudoinverse policy.
- Underdetermined minimum-norm external evidence remains deferred to a future QR solve / minimum-norm oracle owner.
- Q/economy external evidence remains deferred to a future QR basis/economy owner that can define sign, orientation, projection, subspace, and economy-shape semantics.

## Completion Criteria Status

- Compatible tall QR external fixture implemented: complete.
- Helper and C test use the same fixture key and expected output shape: complete.
- Focused `test_qr_solve` evidence captured: complete.
- Full required code gate passed after `.c` and helper changes: complete.
