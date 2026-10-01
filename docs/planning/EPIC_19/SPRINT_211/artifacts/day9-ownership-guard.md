# Sprint 211 Day 9: Ownership Guard

## Summary

Day 9 completes the ownership guard layer for the selected LDLT CSC
native-parity extraction. The helper guard now proves that the moved test-body
definitions actively live in `tests/test_ldlt_csc_native_parity_helpers.h` and
do not drift back into `tests/test_ldlt_csc.c` or another LDLT CSC helper.

## Changed Files

| File | Change |
| --- | --- |
| `scripts/check_ldlt_csc_helper_guard.sh` | Added moved-definition markers and active-code ownership checks. |
| `tests/test_ldlt_csc_helper_guard.py` | Added fixture definitions and negative ownership-drift mutations. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Added Day 9 implementation notes. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day9-ownership-guard.md` | Added this artifact. |

## Guarded Definitions

The guard now checks these moved definitions:

| Definition | Required owner |
| --- | --- |
| `test_native_1x1_diagonal_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_1x1_tridiagonal_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_1x1_mixed_indefinite_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_1x1_with_swap_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_1x1_tridiag_large_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_detects_near_zero_1x1_pivot` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_1x1_identity_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_2x2_forced_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_2x2_nonadjacent_partner_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_mixed_pivots_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_mixed_pivots_larger_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_2x2_solve_matches_linked_list` | `tests/test_ldlt_csc_native_parity_helpers.h` |
| `test_native_2x2_inertia_matches_wrapper` | `tests/test_ldlt_csc_native_parity_helpers.h` |

For each definition, the guard requires exactly one active-code occurrence in
the native parity helper and zero active-code occurrences in the proof-owner
file or the other LDLT CSC helper headers.

## Comment Handling

The guard strips both line comments and block comments before matching moved
definitions and selected `RUN_TEST(...)` registrations. A commented-out
definition or registration cannot satisfy the guard.

## Regression Fixture

`tests/test_ldlt_csc_helper_guard.py` now proves clear failures for:

- missing moved definition in the native parity helper;
- block-commented moved definition in the native parity helper;
- moved definition duplicated into `tests/test_ldlt_csc.c`;
- moved definition duplicated into `tests/test_ldlt_csc_oracle_helpers.h`.

These checks extend the Day 8 fixture coverage for missing helper includes,
missing Makefile prerequisites, commented registrations, reordered
registrations, standalone CMake registration, and library-source registration.

## Validation

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |
| `./build/test_ldlt_csc` | Passed: `100` tests, `0` failed, `0` skipped, `3556` assertions. |

Day 9 changed shell, Python, and planning artifacts only. No additional `.c`
or `.h` edits were made on Day 9, so the latest full C gate remains the Day 7
`make format && make lint && make test` pass.

## Item Evidence

Item 211.5 is complete for ownership guard coverage. The selected native parity
cluster is guarded against missing registrations, commented inert
registrations, missing helper ownership, commented inert definitions, wrong
definition owners, standalone helper test registration, and library-source
registration.
