# Sprint 211 Day 13: Review Hardening

## Purpose

Day 13 reviewed the Sprint 211 diff as a reviewer and closed one concrete
guard-coverage gap before final closeout. The hardening stayed scoped to the
selected LDLT CSC native-parity helper extraction and its evidence.

## Finding and Fix

| Finding | Fix |
| --- | --- |
| The LDLT CSC helper guard checked helper includes with a raw fixed-string count. A commented-out `#include "test_ldlt_csc_native_parity_helpers.h"` line could satisfy helper presence while the proof-owner test stopped including the helper. | `scripts/check_ldlt_csc_helper_guard.sh` now strips line/block comments before counting helper includes and requires exactly one active include for each LDLT CSC helper header. |
| PR #234 review identified additional guard gaps for single translation-unit ownership, complete Makefile boundary enforcement, conditional-preprocessor awareness, path-qualified includes, angle-bracket includes, repeated same-line `RUN_TEST(...)` registrations, repeated same-line moved-definition markers, selected-native block boundary before the Day 9 solve block, missing Day 9 solve registration diagnostics, duplicated scanner predicates, zero-valued preprocessor expressions, Makefile target wiring, full-gate build scope, multiline Makefile prerequisite rules, exact prerequisite-token matching, commented-out prerequisites, commented-out target commands, echoed command text, bare helper-basename manifest entries, inline-comment mentions, failure diagnostics, conditional forbidden includes, unknown-conditional duplicate ownership, repository-wide moved-definition duplicates, behavior-suite contract drift, comment-only behavior contract bypasses, and diagnostic-output preservation. | The guard now rejects extra active includes of the native helper outside `tests/test_ldlt_csc.c` across repository C/header trees, compares quoted or angle-bracket include basenames so path-qualified includes are counted, treats possible-active includes under unknown conditionals as forbidden ownership outside the proof owner, compares active and possible-active counts for required proof-owner includes, selected `RUN_TEST(...)` registrations, and moved definitions so unknown-conditional duplicates cannot hide, counts every literal helper basename occurrence in active Makefile text after stripping comments, counts every selected active `RUN_TEST(...)` marker occurrence on a line, counts every active moved-definition marker occurrence on a line, scans repository C/header files for duplicate moved-definition ownership outside the native helper, uses one shared branch-aware `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif` AWK prelude for active-code scans, treats `#if 00`, `#if 0x0`, and `#if (0)` as inactive zero expressions, rejects ambiguous non-constant `#if`/`#elif` and unknown `#ifdef`/`#ifndef` required ownership while whitelisting each file's own include guard, accepts only exact Makefile prerequisite tokens after comments and continuations are handled, reports the exact missing `$(TESTDIR)/...` prerequisite token, verifies `ldlt-csc-helper-guard` runs both Python suites, verifies `quality-review-compile` runs the helper guard, verifies `quality-review-full` runs `quality-review-compile`, rejects helper basenames in `build-metadata/library_sources.txt`, matches parsed recipe commands at a command boundary after stripping inline comments, requires the selected native registrations to remain before `RUN_TEST(test_solve_null_args);`, reports the missing Day 9 solve registration from the numeric sentinel, validates the behavior suite's executable `SELECTED_TESTS` and `BASELINE_SUMMARY` assignments with Python AST parsing, verifies the behavior suite checks pass-marker order and all-tests marker, and requires bounded command-output excerpts in behavior-regression failures. |

## Regression Coverage Added

| Test | Coverage |
| --- | --- |
| `test_line_commented_native_helper_include_fails_clearly()` | Proves a line-commented native helper include no longer satisfies the guard. |
| `test_block_commented_native_helper_include_fails_clearly()` | Proves a block-commented native helper include no longer satisfies the guard. |
| `test_native_helper_extra_makefile_registration_fails_clearly()` | Proves the helper cannot be added to an extra Makefile registration surface. |
| `test_native_helper_duplicate_same_line_makefile_registration_fails_clearly()` | Proves duplicate helper prerequisites on one Makefile line are counted as duplicate occurrences. |
| `test_native_helper_bare_makefile_registration_fails_clearly()` | Proves a bare helper basename registration is still counted as an extra Makefile occurrence. |
| `test_commented_native_helper_makefile_prerequisite_fails_clearly()` | Proves a commented-out helper prerequisite does not satisfy Makefile ownership. |
| `test_suffix_native_helper_makefile_prerequisite_fails_clearly()` | Proves a suffix-sharing Makefile token does not satisfy the required helper prerequisite. |
| `test_native_helper_second_translation_unit_include_fails_clearly()` | Proves the native helper cannot be included by a second translation unit. |
| `test_native_helper_angle_bracket_second_translation_unit_include_fails_clearly()` | Proves the native helper cannot be included by a second translation unit with angle brackets. |
| `test_native_helper_examples_translation_unit_include_fails_clearly()` | Proves the native helper cannot be included by a second translation unit outside `tests`, `src`, or `include`. |
| `test_native_helper_path_qualified_examples_include_fails_clearly()` | Proves path-qualified helper includes outside the proof owner are counted by basename and rejected. |
| `test_native_helper_unknown_ifdef_second_translation_unit_include_fails_closed()` | Proves possible-active helper includes under unknown conditionals outside the proof owner are rejected. |
| `test_native_helper_unknown_ifdef_duplicate_owner_include_fails_closed()` | Proves a duplicate helper include hidden under an unknown conditional in the proof owner is rejected. |
| `test_if_zero_run_test_registration_fails_clearly()` | Proves inactive selected registrations under `#if 0` do not satisfy the guard. |
| `test_duplicate_same_line_run_test_registration_fails_clearly()` | Proves repeated selected registrations on one active line are counted as duplicates. |
| `test_unknown_ifdef_duplicate_run_test_registration_fails_closed()` | Proves a duplicate selected registration hidden under an unknown conditional is rejected. |
| `test_parenthesized_zero_native_helper_include_fails_clearly()` | Proves an inactive helper include under `#if (0)` does not satisfy include ownership. |
| `test_if_zero_else_native_helper_include_passes_guard()` | Proves a helper include in the active `#else` branch of `#if 0` is accepted. |
| `test_path_qualified_native_helper_include_passes_guard()` | Proves path-qualified helper includes in the proof owner are counted by basename. |
| `test_angle_bracket_native_helper_include_passes_guard()` | Proves angle-bracket helper includes in the proof owner are counted by basename. |
| `test_ifdef_native_helper_include_fails_closed()` | Proves helper includes under unknown primary `#ifdef` branches fail closed. |
| `test_ifndef_native_helper_include_fails_closed()` | Proves helper includes under unknown primary `#ifndef` branches fail closed. |
| `test_multiline_makefile_prerequisite_rule_passes_guard()` | Proves continued Makefile prerequisites are parsed as one logical rule. |
| `test_makefile_guard_target_runs_python_suite_fails_clearly()` | Proves the Makefile helper target must invoke the guard regression suite. |
| `test_makefile_guard_target_runs_behavior_suite_fails_clearly()` | Proves the Makefile helper target must invoke the native parity behavior suite. |
| `test_quality_review_compile_runs_helper_guard_fails_clearly()` | Proves the reviewed compile-quality path must invoke the helper guard target. |
| `test_quality_review_full_runs_compile_gate_fails_clearly()` | Proves the strongest reviewed quality path must invoke the compile-quality gate. |
| `test_makefile_guard_target_ignores_commented_python_suite_fails_clearly()` | Proves a commented-out guard-suite command does not satisfy helper target wiring. |
| `test_quality_review_compile_ignores_commented_helper_guard_fails_clearly()` | Proves a commented-out recursive helper-guard command does not satisfy reviewed compile wiring. |
| `test_makefile_guard_target_ignores_echoed_python_suite_fails_clearly()` | Proves echoed guard-suite command text does not satisfy helper target wiring. |
| `test_quality_review_compile_ignores_inline_comment_helper_guard_fails_clearly()` | Proves helper-guard text after an inline recipe comment does not satisfy reviewed compile wiring. |
| `test_octal_zero_run_test_registration_fails_clearly()` | Proves inactive selected registrations under `#if 00` do not satisfy the guard. |
| `test_if_zero_else_run_test_registration_passes_guard()` | Proves a selected registration in the active `#else` branch of `#if 0` is accepted. |
| `test_if_zero_elif_run_test_registration_passes_guard()` | Proves a selected registration in an active `#elif` branch after `#if 0` is accepted. |
| `test_if_one_run_test_registration_passes_guard()` | Proves explicit `#if 1` selected registrations remain accepted. |
| `test_unknown_primary_if_run_test_registration_fails_closed()` | Proves ambiguous non-constant primary `#if` selected registrations fail closed. |
| `test_if_zero_unknown_elif_run_test_registration_fails_closed()` | Proves ambiguous non-constant `#elif` ownership is rejected. |
| `test_selected_registration_after_solve_block_fails_clearly()` | Proves selected native registrations must remain before the Day 9 solve block. |
| `test_missing_solve_registration_fails_clearly()` | Proves removing the Day 9 solve registration reports the missing-registration diagnostic instead of an ordering fallback. |
| `test_duplicate_same_line_moved_definition_fails_clearly()` | Proves repeated moved-definition markers on one active line are counted as duplicates. |
| `test_unknown_ifdef_duplicate_moved_definition_fails_closed()` | Proves a duplicate moved definition hidden under an unknown conditional in the native helper is rejected. |
| `test_if_zero_moved_definition_fails_clearly()` | Proves inactive moved definitions under `#if 0` do not satisfy helper ownership. |
| `test_hex_zero_moved_definition_fails_clearly()` | Proves inactive moved definitions under `#if 0x0` do not satisfy helper ownership. |
| `test_if_zero_else_moved_definition_passes_guard()` | Proves a moved definition in the active `#else` branch of `#if 0` is accepted. |
| `test_ifndef_moved_definition_fails_closed()` | Proves moved definitions under unknown primary `#ifndef` branches fail closed. |
| `test_ifdef_moved_definition_fails_closed()` | Proves moved definitions under unknown primary `#ifdef` branches fail closed. |
| `test_moved_definition_in_examples_translation_unit_fails_clearly()` | Proves moved-definition markers are rejected outside the native helper across repository C files. |
| `test_unknown_ifdef_moved_definition_outside_native_helper_fails_closed()` | Proves possible-active moved definitions under unknown conditionals outside the native helper are rejected. |
| `test_native_helper_bare_library_source_registration_fails_clearly()` | Proves a bare helper basename cannot enter the library source manifest. |
| `test_behavior_suite_missing_selected_marker_fails_clearly()` | Proves the guard fails if the behavior suite stops pinning a selected native parity pass marker. |
| `test_behavior_suite_commented_selected_marker_fails_clearly()` | Proves a selected marker left only in a Python comment does not satisfy the behavior-suite contract. |
| `test_behavior_suite_summary_contract_fails_clearly()` | Proves the guard fails if the behavior suite weakens the exact focused summary contract. |
| `test_behavior_suite_commented_summary_contract_fails_clearly()` | Proves a baseline summary value left only in a Python comment does not satisfy the behavior-suite contract. |
| `test_behavior_suite_output_diagnostics_contract_fails_clearly()` | Proves the guard fails if behavior-regression failures stop preserving command-output excerpts. |

## Changed Surface

| Surface | Files |
| --- | --- |
| Guard implementation | `scripts/check_ldlt_csc_helper_guard.sh` |
| Guard regression fixture | `tests/test_ldlt_csc_helper_guard.py`, `tests/test_ldlt_csc_native_parity_behavior.py` |
| Quality gate wiring | `Makefile` |
| Sprint evidence | `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md`, `docs/planning/EPIC_19/SPRINT_211/artifacts/day13-review-hardening.md` |

Day 13 did not edit `.c` or `.h` files. The latest required full C quality
chain remains the Day 12 `make format && make lint && make test` pass.

## Current Line-Count Snapshot

| File | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3174 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 303 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 790 |
| `tests/test_ldlt_csc_helper_guard.py` | 1211 |
| `tests/test_ldlt_csc_native_parity_behavior.py` | 108 |

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
