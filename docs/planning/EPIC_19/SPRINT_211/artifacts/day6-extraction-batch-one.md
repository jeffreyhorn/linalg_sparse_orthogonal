# Sprint 211 Day 6: Extraction Batch One

## Summary

Day 6 implements the first selected LDLT CSC native-parity extraction batch.
The Sprint 18 Day 3 1x1 native parity test bodies moved from
`tests/test_ldlt_csc.c` into the new header-only family-local helper
`tests/test_ldlt_csc_native_parity_helpers.h`.

The proof-owner executable remains `tests/test_ldlt_csc.c`; all selected
`RUN_TEST(...)` registrations remain there.

## Changed Files

| File | Change |
| --- | --- |
| `tests/test_ldlt_csc.c` | Added `#include "test_ldlt_csc_native_parity_helpers.h"` with the selected LDLT CSC helper includes and removed the moved 1x1 native parity test bodies. |
| `tests/test_ldlt_csc_native_parity_helpers.h` | New selected helper header for the 1x1 native parity test bodies. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Added Day 6 implementation notes. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day6-extraction-batch-one.md` | Added this artifact. |

## Moved Symbols

| Symbol | New owner |
| --- | --- |
| `test_native_1x1_diagonal_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_1x1_tridiagonal_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_1x1_mixed_indefinite_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_1x1_with_swap_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_1x1_tridiag_large_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_detects_near_zero_1x1_pivot` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_1x1_identity_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |

## Preserved Ownership

| Surface | Status |
| --- | --- |
| `RUN_TEST(...)` registrations | Still owned by `tests/test_ldlt_csc.c`. |
| `check_native_matches_wrapper()` | Still owned by `tests/test_ldlt_csc_oracle_helpers.h`. |
| Dense oracle comparison helpers | Still owned by `tests/test_ldlt_csc_oracle_helpers.h`. |
| 2x2 native parity tests | Still owned by `tests/test_ldlt_csc.c` until Day 7. |
| Production LDLT CSC source | Unchanged. |
| Makefile/CMake test registration | Unchanged. |

## Include State

`tests/test_ldlt_csc.c` now includes:

```c
#include "test_ldlt_csc_fixtures.h"
#include "test_ldlt_csc_native_parity_helpers.h"
#include "test_ldlt_csc_oracle_helpers.h"
#include "test_ldlt_csc_supernode_helpers.h"
```

The new helper is header-only and has include guard
`TEST_LDLT_CSC_NATIVE_PARITY_HELPERS_H`.

## Metrics

| File | Day 5 baseline | Day 6 after batch one | Delta |
| --- | ---: | ---: | ---: |
| `tests/test_ldlt_csc.c` | 3469 | 3346 | -123 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 0 | 133 | +133 |
| `tests/test_ldlt_csc_fixtures.h` | 145 | 145 | 0 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 151 | 151 | 0 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | 140 | 0 |

The proof-owner file shrank by 123 lines. The net selected test surface grew by
10 lines due to the new helper guard, includes, and owner comment.

## Focused Validation

| Command | Result |
| --- | --- |
| `make build/test_ldlt_csc` | Passed. |
| `./build/test_ldlt_csc` | Passed: `100` tests, `0` failed, `0` skipped, `3556` assertions. |
| `make ldlt-csc-helper-guard` | Passed with the existing guard. |

The existing helper guard does not yet enforce the new helper. That work is
explicitly deferred to the guard/wiring days.

## Required Full Gate

Day 6 modifies `.c` and `.h` files, so the sprint-required quality gate is:

```sh
make format && make lint && make test
```

Result: passed. The command completed formatting, lint, and the full test
suite. The full suite included `./build/test_ldlt_csc`, which passed with
`100` tests, `0` failed, `0` skipped, and `3556` assertions.

## Item Evidence

Item 211.4 has started with a behavior-preserving extraction. The first
selected batch builds and passes the focused proof-owner test, and the original
proof-owner file is measurably smaller.
