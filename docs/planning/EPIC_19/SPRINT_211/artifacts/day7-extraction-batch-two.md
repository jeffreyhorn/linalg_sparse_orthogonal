# Sprint 211 Day 7: Extraction Batch Two

## Summary

Day 7 completes the selected LDLT CSC native-parity extraction. The Sprint 18
Day 4 2x2/native solve parity test bodies moved from `tests/test_ldlt_csc.c`
into `tests/test_ldlt_csc_native_parity_helpers.h`.

The proof-owner executable remains `tests/test_ldlt_csc.c`; all selected
`RUN_TEST(...)` registrations remain there.

## Changed Files

| File | Change |
| --- | --- |
| `tests/test_ldlt_csc.c` | Removed the selected Sprint 18 Day 4 2x2/native solve parity test bodies while keeping their registrations. |
| `tests/test_ldlt_csc_native_parity_helpers.h` | Added the selected 2x2/native solve parity test bodies, `<math.h>`, and a forward declaration for `rel_residual()`. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Added Day 7 implementation notes. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day7-extraction-batch-two.md` | Added this artifact. |

## Moved Symbols

| Symbol | New owner |
| --- | --- |
| `test_native_2x2_forced_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_2x2_nonadjacent_partner_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_mixed_pivots_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_mixed_pivots_larger_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_2x2_solve_matches_linked_list` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_2x2_inertia_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |

## Preserved Ownership

| Surface | Status |
| --- | --- |
| `RUN_TEST(...)` registrations | Still owned by `tests/test_ldlt_csc.c`. |
| `check_native_matches_wrapper()` | Still owned by `tests/test_ldlt_csc_oracle_helpers.h`. |
| Dense oracle comparison helpers | Still owned by `tests/test_ldlt_csc_oracle_helpers.h`. |
| `rel_residual()` implementation | Still owned by `tests/test_ldlt_csc.c` in the Day 9 solve block. |
| Production LDLT CSC source | Unchanged. |
| Makefile/CMake test registration | Unchanged. |

## Registration State

The proof-owner file keeps the selected registrations in order:

| Registration group | Lines |
| --- | ---: |
| 1x1 native parity | 3144-3150 |
| 2x2/native solve parity | 3153-3158 |

## Metrics

| File | Day 6 after batch one | Day 7 after batch two | Delta |
| --- | ---: | ---: | ---: |
| `tests/test_ldlt_csc.c` | 3346 | 3174 | -172 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 133 | 303 | +170 |
| `tests/test_ldlt_csc_fixtures.h` | 145 | 145 | 0 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 151 | 151 | 0 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | 140 | 0 |

The proof-owner file is now 295 lines smaller than the Day 5 baseline. The net
selected test surface is 8 lines larger than the Day 5 baseline due to the new
helper guard, local includes, and owner comments.

## Focused Validation

| Command | Result |
| --- | --- |
| `make build/test_ldlt_csc` | Passed. |
| `./build/test_ldlt_csc` | Passed: `100` tests, `0` failed, `0` skipped, `3556` assertions. |
| `make ldlt-csc-helper-guard` | Passed with the existing guard. |

The existing helper guard does not yet enforce the new native parity helper.
That work remains explicitly deferred to Day 8 or Day 9.

## Required Full Gate

Day 7 modifies `.c` and `.h` files, so the sprint-required quality gate ran:

```sh
make format && make lint && make test
```

Result: passed. The command completed formatting, strict warning compile,
lint, cppcheck, and the full test suite. The full suite included
`./build/test_ldlt_csc`, which passed with `100` tests, `0` failed, `0`
skipped, and `3556` assertions.

## Item Evidence

Item 211.4 is complete for the selected native-parity test-body extraction.
The selected native parity test bodies now live in the planned helper owner,
the original proof-owner file is measurably smaller, and behavior remains
covered by the same proof-owner registrations.
