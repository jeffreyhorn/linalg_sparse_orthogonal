# Sprint 211 Day 10: Behavior Regression Coverage

## Summary

Day 10 adds focused behavior-regression coverage for the extracted LDLT CSC
native-parity cluster. Ownership and build wiring are guarded by Days 8-9; this
day verifies the proof-owner executable still runs the selected native parity
tests in order and still reports the Day 5 baseline summary counts.

## Changed Files

| File | Change |
| --- | --- |
| `tests/test_ldlt_csc_native_parity_behavior.py` | Added focused behavior-regression runner. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Added Day 10 implementation notes. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day10-behavior-regression.md` | Added this artifact. |

## Behavior Runner

The new runner performs:

```sh
make build/test_ldlt_csc
./build/test_ldlt_csc
```

It then asserts:

- all 13 selected native parity `[PASS]` markers are present;
- the selected markers appear in the expected order;
- `Tests run` remains `100`;
- `Tests failed` remains `0`;
- `Tests skipped` remains `0`;
- `Assertions` remains `3556`;
- `ALL TESTS PASSED` is present.

## Selected Native Parity Markers

The behavior runner checks:

| Order | Marker |
| ---: | --- |
| 1 | `test_native_1x1_diagonal_matches_wrapper` |
| 2 | `test_native_1x1_tridiagonal_matches_wrapper` |
| 3 | `test_native_1x1_mixed_indefinite_matches_wrapper` |
| 4 | `test_native_1x1_with_swap_matches_wrapper` |
| 5 | `test_native_1x1_tridiag_large_matches_wrapper` |
| 6 | `test_native_detects_near_zero_1x1_pivot` |
| 7 | `test_native_1x1_identity_matches_wrapper` |
| 8 | `test_native_2x2_forced_matches_wrapper` |
| 9 | `test_native_2x2_nonadjacent_partner_matches_wrapper` |
| 10 | `test_native_mixed_pivots_matches_wrapper` |
| 11 | `test_native_mixed_pivots_larger_matches_wrapper` |
| 12 | `test_native_2x2_solve_matches_linked_list` |
| 13 | `test_native_2x2_inertia_matches_wrapper` |

## Baseline Comparison

| Surface | Day 5 baseline | Day 10 observed | Status |
| --- | ---: | ---: | --- |
| `test_ldlt_csc` tests run | 100 | 100 | Unchanged. |
| `test_ldlt_csc` failures | 0 | 0 | Unchanged. |
| `test_ldlt_csc` skipped | 0 | 0 | Unchanged. |
| `test_ldlt_csc` assertions | 3556 | 3556 | Unchanged. |
| Selected native parity pass markers | 13 | 13 | Unchanged and ordered. |

No selected-output or assertion-count drift was observed.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_ldlt_csc_native_parity_behavior.py` | Passed. |
| `make ldlt-csc-helper-guard` | Passed. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |

Day 10 changed Python and planning artifacts only. No additional `.c` or `.h`
edits were made on Day 10, so the latest full C gate remains the Day 7
`make format && make lint && make test` pass.

## Item Evidence

Item 211.5 now has behavior-regression coverage beyond ownership-only checks.
The extracted native parity cluster remains behaviorally reachable through the
same proof-owner executable with unchanged focused summary counts.
