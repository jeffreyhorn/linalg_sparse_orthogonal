# Sprint 211 Day 13: Review Hardening

## Purpose

Day 13 reviewed the Sprint 211 diff as a reviewer and closed one concrete
guard-coverage gap before final closeout. The hardening stayed scoped to the
selected LDLT CSC native-parity helper extraction and its evidence.

## Finding and Fix

| Finding | Fix |
| --- | --- |
| The LDLT CSC helper guard checked helper includes with a raw fixed-string count. A commented-out `#include "test_ldlt_csc_native_parity_helpers.h"` line could satisfy helper presence while the proof-owner test stopped including the helper. | `scripts/check_ldlt_csc_helper_guard.sh` now strips line/block comments before counting helper includes and requires exactly one active include for each LDLT CSC helper header. |
| PR #234 review identified additional guard gaps for single translation-unit ownership, complete Makefile boundary enforcement, conditional-preprocessor awareness, path-qualified includes, selected-native block boundary before the Day 9 solve block, duplicated scanner predicates, zero-valued preprocessor expressions, Makefile target wiring, multiline Makefile prerequisite rules, and commented-out target commands. | The guard now rejects extra active includes of the native helper outside `tests/test_ldlt_csc.c` across repository C/header trees, compares quoted include basenames so path-qualified includes are counted, counts every literal helper basename occurrence in `Makefile`, uses one shared branch-aware `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif` AWK prelude for active-code scans, treats `#if 00`, `#if 0x0`, and `#if (0)` as inactive zero expressions, rejects ambiguous non-constant `#if`/`#elif` ownership, accepts logical Makefile prerequisite rules split with continuations, verifies `ldlt-csc-helper-guard` runs both Python suites, verifies `quality-review-compile` runs the helper guard, ignores commented-out recipe commands when validating target wiring, and requires the selected native registrations to remain before `RUN_TEST(test_solve_null_args);`. |

## Regression Coverage Added

| Test | Coverage |
| --- | --- |
| `test_line_commented_native_helper_include_fails_clearly()` | Proves a line-commented native helper include no longer satisfies the guard. |
| `test_block_commented_native_helper_include_fails_clearly()` | Proves a block-commented native helper include no longer satisfies the guard. |
| `test_native_helper_extra_makefile_registration_fails_clearly()` | Proves the helper cannot be added to an extra Makefile registration surface. |
| `test_native_helper_duplicate_same_line_makefile_registration_fails_clearly()` | Proves duplicate helper prerequisites on one Makefile line are counted as duplicate occurrences. |
| `test_native_helper_bare_makefile_registration_fails_clearly()` | Proves a bare helper basename registration is still counted as an extra Makefile occurrence. |
| `test_native_helper_second_translation_unit_include_fails_clearly()` | Proves the native helper cannot be included by a second translation unit. |
| `test_native_helper_examples_translation_unit_include_fails_clearly()` | Proves the native helper cannot be included by a second translation unit outside `tests`, `src`, or `include`. |
| `test_native_helper_path_qualified_examples_include_fails_clearly()` | Proves path-qualified helper includes outside the proof owner are counted by basename and rejected. |
| `test_if_zero_run_test_registration_fails_clearly()` | Proves inactive selected registrations under `#if 0` do not satisfy the guard. |
| `test_parenthesized_zero_native_helper_include_fails_clearly()` | Proves an inactive helper include under `#if (0)` does not satisfy include ownership. |
| `test_if_zero_else_native_helper_include_passes_guard()` | Proves a helper include in the active `#else` branch of `#if 0` is accepted. |
| `test_path_qualified_native_helper_include_passes_guard()` | Proves path-qualified helper includes in the proof owner are counted by basename. |
| `test_ifdef_native_helper_include_passes_guard()` | Proves primary `#ifdef` branches remain scannable for helper includes. |
| `test_multiline_makefile_prerequisite_rule_passes_guard()` | Proves continued Makefile prerequisites are parsed as one logical rule. |
| `test_makefile_guard_target_runs_python_suite_fails_clearly()` | Proves the Makefile helper target must invoke the guard regression suite. |
| `test_makefile_guard_target_runs_behavior_suite_fails_clearly()` | Proves the Makefile helper target must invoke the native parity behavior suite. |
| `test_quality_review_compile_runs_helper_guard_fails_clearly()` | Proves the reviewed compile-quality path must invoke the helper guard target. |
| `test_makefile_guard_target_ignores_commented_python_suite_fails_clearly()` | Proves a commented-out guard-suite command does not satisfy helper target wiring. |
| `test_quality_review_compile_ignores_commented_helper_guard_fails_clearly()` | Proves a commented-out recursive helper-guard command does not satisfy reviewed compile wiring. |
| `test_octal_zero_run_test_registration_fails_clearly()` | Proves inactive selected registrations under `#if 00` do not satisfy the guard. |
| `test_if_zero_else_run_test_registration_passes_guard()` | Proves a selected registration in the active `#else` branch of `#if 0` is accepted. |
| `test_if_zero_elif_run_test_registration_passes_guard()` | Proves a selected registration in an active `#elif` branch after `#if 0` is accepted. |
| `test_if_one_run_test_registration_passes_guard()` | Proves explicit `#if 1` selected registrations remain accepted. |
| `test_unknown_primary_if_run_test_registration_fails_closed()` | Proves ambiguous non-constant primary `#if` selected registrations fail closed. |
| `test_if_zero_unknown_elif_run_test_registration_fails_closed()` | Proves ambiguous non-constant `#elif` ownership is rejected. |
| `test_selected_registration_after_solve_block_fails_clearly()` | Proves selected native registrations must remain before the Day 9 solve block. |
| `test_if_zero_moved_definition_fails_clearly()` | Proves inactive moved definitions under `#if 0` do not satisfy helper ownership. |
| `test_hex_zero_moved_definition_fails_clearly()` | Proves inactive moved definitions under `#if 0x0` do not satisfy helper ownership. |
| `test_if_zero_else_moved_definition_passes_guard()` | Proves a moved definition in the active `#else` branch of `#if 0` is accepted. |
| `test_ifndef_moved_definition_passes_guard()` | Proves primary `#ifndef` branches remain scannable for moved definitions. |

## Changed Surface

| Surface | Files |
| --- | --- |
| Guard implementation | `scripts/check_ldlt_csc_helper_guard.sh` |
| Guard regression fixture | `tests/test_ldlt_csc_helper_guard.py` |
| Sprint evidence | `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md`, `docs/planning/EPIC_19/SPRINT_211/artifacts/day13-review-hardening.md` |

Day 13 did not edit `.c` or `.h` files. The latest required full C quality
chain remains the Day 12 `make format && make lint && make test` pass.

## Current Line-Count Snapshot

| File | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3174 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 303 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 576 |
| `tests/test_ldlt_csc_helper_guard.py` | 812 |
| `tests/test_ldlt_csc_native_parity_behavior.py` | 92 |

## Validation

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |
| `python3 tests/test_ldlt_csc_native_parity_behavior.py` | Passed. |
| `git diff --check` | Passed. |

## Residual Risks

- Day 13 did not rerun the full C gate because no `.c` or `.h` files changed
  during hardening.
- Day 13 did not run the full CMake `ctest` suite; Day 12 covered CMake
  compile and `ctest -N` registration parity.
- Platform-hosted validation remains outside local sprint evidence.

## Non-Claims

The hardening does not claim new LDLT CSC behavior, public API or ABI support,
package support, platform support, performance improvement, release support, or
state-of-the-art evidence.
