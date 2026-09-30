# Sprint 211 Day 11: Documentation Calibration

## Purpose

Day 11 calibrated maintainer and sprint documentation for the Sprint 211 LDLT
CSC native-parity helper extraction. The update documents the current ownership
model without claiming solver, API, ABI, platform, performance, package,
release, or state-of-the-art changes.

## Documentation Updates

| File | Update |
| --- | --- |
| `docs/maintainer_guide.md` | Added `tests/test_ldlt_csc_native_parity_helpers.h` to the LDLT CSC helper-owner list, documented that `tests/test_ldlt_csc.c` remains the proof-owner/registration owner, and recorded focused guard and behavior validation commands. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Added the Day 11 changed-surface inventory, line-count snapshot, validation results, and non-goal wording. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day11-documentation-calibration.md` | Added this artifact as the Day 11 provenance record. |

README and INSTALL were left unchanged because the Sprint 211 work is an
internal test review-surface reduction, not a user-facing install, support,
API, package, or solver-behavior change.

## Current Ownership Model

| Surface | Owner |
| --- | --- |
| Proof-owner executable, `main`, and selected `RUN_TEST(...)` registrations | `tests/test_ldlt_csc.c` |
| Selected Sprint 211 native parity test bodies | `tests/test_ldlt_csc_native_parity_helpers.h` |
| Dense/native comparison helpers | `tests/test_ldlt_csc_oracle_helpers.h` |
| Family-local KKT and analysis fixtures | `tests/test_ldlt_csc_fixtures.h` |
| Supernode fixtures and factor-state comparison helpers | `tests/test_ldlt_csc_supernode_helpers.h` |
| Helper ownership and registration guard | `scripts/check_ldlt_csc_helper_guard.sh` and `make ldlt-csc-helper-guard` |
| Guard regression fixture | `tests/test_ldlt_csc_helper_guard.py` |
| Focused behavior regression | `tests/test_ldlt_csc_native_parity_behavior.py` |

## Line-Count Snapshot

| File | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3174 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 303 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 644 after PR #234 guard hardening |
| `tests/test_ldlt_csc_helper_guard.py` | 920 after PR #234 guard hardening |
| `tests/test_ldlt_csc_native_parity_behavior.py` | 92 |
| `docs/maintainer_guide.md` | 2206 after Day 11 edits |

The proof-owner test remains reduced by 295 lines from the Day 5 baseline of
3469 lines while keeping the same selected execution behavior recorded on Day
10.

## Non-Goals

Day 11 does not introduce or claim:

- new LDLT CSC solver behavior;
- new public API or ABI support;
- new numerical tolerance behavior;
- new package-manager or install support;
- new platform support;
- new performance, release, or state-of-the-art evidence.

## Validation

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed; proof-owner registrations, helper headers, header-only boundaries, selected `RUN_TEST(...)` registrations, and moved-definition ownership all passed. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |
| `python3 tests/test_ldlt_csc_native_parity_behavior.py` | Passed. |
| `git diff --check` | Passed. |

Day 11 changed documentation only. No `.c` or `.h` files were edited on Day
11, so the sprint-required full C quality gate is not rerun for this
documentation-only calibration. The latest full C gate remains the Day 7
`make format && make lint && make test` pass after the extraction.
