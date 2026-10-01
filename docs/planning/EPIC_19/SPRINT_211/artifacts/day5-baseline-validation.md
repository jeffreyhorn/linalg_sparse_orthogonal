# Sprint 211 Day 5: Baseline Validation

## Summary

Day 5 captures the pre-extraction behavior and review-surface metrics for the
selected LDLT CSC native-parity cluster. Days 6-8 should compare against this
baseline after creating `tests/test_ldlt_csc_native_parity_helpers.h` and
moving the selected test bodies.

## Baseline Context

| Field | Value |
| --- | --- |
| Branch | `sprint-211` |
| Baseline commit | `64cb32cc` |
| Proof owner | `tests/test_ldlt_csc.c` |
| Selected cluster | Sprint 18 Day 3/Day 4 native 1x1 and 2x2 parity tests. |
| Selected code span | Lines 2640-2934, `295` lines. |
| Registration span | Lines 3439-3453. |

## Commands And Results

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed. |
| `./build/test_ldlt_csc` | Passed: `100` tests, `0` failed, `0` skipped, `3556` assertions, `ALL TESTS PASSED`. |
| `ctest -N --test-dir build/quality-review-cmake` | Available; `Total Tests: 60`. |
| `ctest -N --test-dir build` | Not useful locally; `Total Tests: 0`. |

The focused proof-owner executable is the primary Day 5 behavior baseline. The
quality-review CMake tree is the CTest-count baseline for this local checkout.

## Guard Baseline

`make ldlt-csc-helper-guard` reported:

```text
ldlt-csc-helper-guard: proof-owner registration ok
ldlt-csc-helper-guard: helper headers ok
ldlt-csc-helper-guard: header-only registration ok
ldlt-csc-helper-guard: passed
```

The guard currently covers the existing LDLT CSC helper headers:

- `tests/test_ldlt_csc_fixtures.h`;
- `tests/test_ldlt_csc_oracle_helpers.h`;
- `tests/test_ldlt_csc_supernode_helpers.h`.

It does not yet cover the planned native-parity helper because that file does
not exist before Day 6.

## Line-Count Baseline

| File | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3469 |
| `tests/test_ldlt_csc_fixtures.h` | 145 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 151 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 139 |

Expected Day 6/7 metric: `tests/test_ldlt_csc.c` should shrink while the new
native-parity helper grows by the moved selected test bodies. The total test
surface may stay similar; the review-surface win is ownership and focused
readability, not behavior change.

## Selected Registration Baseline

| Line | Registration |
| ---: | --- |
| 3439 | `RUN_TEST(test_native_1x1_diagonal_matches_wrapper);` |
| 3440 | `RUN_TEST(test_native_1x1_tridiagonal_matches_wrapper);` |
| 3441 | `RUN_TEST(test_native_1x1_mixed_indefinite_matches_wrapper);` |
| 3442 | `RUN_TEST(test_native_1x1_with_swap_matches_wrapper);` |
| 3443 | `RUN_TEST(test_native_1x1_tridiag_large_matches_wrapper);` |
| 3444 | `RUN_TEST(test_native_detects_near_zero_1x1_pivot);` |
| 3445 | `RUN_TEST(test_native_1x1_identity_matches_wrapper);` |
| 3448 | `RUN_TEST(test_native_2x2_forced_matches_wrapper);` |
| 3449 | `RUN_TEST(test_native_2x2_nonadjacent_partner_matches_wrapper);` |
| 3450 | `RUN_TEST(test_native_mixed_pivots_matches_wrapper);` |
| 3451 | `RUN_TEST(test_native_mixed_pivots_larger_matches_wrapper);` |
| 3452 | `RUN_TEST(test_native_2x2_solve_matches_linked_list);` |
| 3453 | `RUN_TEST(test_native_2x2_inertia_matches_wrapper);` |

Guard hardening should assert active registration lines and ordering instead
of raw substring counts.

## Include And Registration Baseline

Current helper includes in `tests/test_ldlt_csc.c`:

```c
#include "test_ldlt_csc_fixtures.h"
#include "test_ldlt_csc_oracle_helpers.h"
#include "test_ldlt_csc_supernode_helpers.h"
```

Current proof-owner build registrations:

- `Makefile`: `$(TESTDIR)/test_ldlt_csc.c`;
- `CMakeLists.txt`: `add_sparse_test(test_ldlt_csc)`.

## Risks And Fallbacks

| Risk | Baseline handling |
| --- | --- |
| Normal `build` tree has no CTest registration. | Use `./build/test_ldlt_csc` and `build/quality-review-cmake` CTest surface. |
| New helper is not yet guarded. | Expected before implementation; Day 8 must add helper and active-registration checks. |
| Selected 2x2 solve test uses `rel_residual()` defined later. | Keep a forward declaration or record an alternate solve-helper owner during implementation. |
| C/header implementation edits will require full gate. | Day 6+ must run `make format && make lint && make test` if `.c` or `.h` files change. |

## Item Evidence

Day 5 completes the baseline required before implementation. The selected
behavior passes, current helper ownership passes, registration order is
recorded, CTest availability is characterized, and before-state line counts are
available for Days 6-8.

## Validation

Day 5 changed planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

`git diff --check` is the Day 5 validation command.
