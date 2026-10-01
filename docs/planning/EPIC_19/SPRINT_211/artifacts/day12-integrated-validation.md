# Sprint 211 Day 12: Integrated Validation

## Purpose

Day 12 ran the integrated validation suite for the Sprint 211 LDLT CSC
native-parity helper extraction. Because the branch changes include
`tests/test_ldlt_csc.c` and the new
`tests/test_ldlt_csc_native_parity_helpers.h`, the required full C quality
chain was rerun.

## Changed Surface

| Surface | Files |
| --- | --- |
| Build wiring | `Makefile` |
| Maintainer documentation | `docs/maintainer_guide.md` |
| Helper ownership guard | `scripts/check_ldlt_csc_helper_guard.sh` |
| Proof-owner C test | `tests/test_ldlt_csc.c` |
| New helper/test validation files | `tests/test_ldlt_csc_native_parity_helpers.h`, `tests/test_ldlt_csc_helper_guard.py`, `tests/test_ldlt_csc_native_parity_behavior.py` |
| Sprint planning artifacts | `docs/planning/EPIC_19/SPRINT_211/PLAN.md`, `WORKING_NOTES.md`, and Day 1-12 artifacts |

## Line-Count Snapshot

| File | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3174 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 303 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 375 |
| `tests/test_ldlt_csc_helper_guard.py` | 317 |
| `tests/test_ldlt_csc_native_parity_behavior.py` | 92 |
| `Makefile` | 1104 |
| `docs/maintainer_guide.md` | 2206 |

The proof-owner test remains reduced by 295 lines from the Day 5 baseline of
3469 lines.

## Validation Results

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed; proof-owner registrations, helper headers, header-only boundaries, selected `RUN_TEST(...)` registrations, and moved-definition ownership all passed. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |
| `python3 tests/test_ldlt_csc_native_parity_behavior.py` | Passed. |
| `make source-list-check` | Passed; reported `49` library sources. |
| `make docs-check` | Passed; Doxygen coverage checked `18` public headers and generated `18` reference pages plus `18` source pages. |
| `make format && make lint && make test` | Passed. |
| `make quality-review-cmake-compile` | Passed; CMake configured, clean rebuilt, `ctest -N` reported `60` tests, and Makefile/CMake parity matched `59` Makefile tests plus `1` focused CTest-only selector. |

## CMake and Source-List Evidence

`make quality-review-cmake-compile` rebuilt the CMake tree and compiled the
changed `test_ldlt_csc` target successfully. The CTest registration surface
remains `60` tests: the Makefile test count is `59`, with `1` focused
CTest-only selector.

`make source-list-check` confirmed the production library source list remains
unchanged at `49` entries. The new helper remains test-local and header-only;
it is not a library source or standalone CMake test.

## Residual Risks

- The CMake validation run used compile and CTest-registration parity only; it
  did not run the full CMake `ctest` suite on Day 12.
- Platform-hosted validation remains outside the local Day 12 evidence.
- Day 13 should review the artifact inventory and guard wording for missed
  owner-boundary cases before closeout.

## Non-Claims

This validation does not claim new LDLT CSC behavior, public API or ABI
support, package support, platform support, performance improvement, release
support, or state-of-the-art evidence.
