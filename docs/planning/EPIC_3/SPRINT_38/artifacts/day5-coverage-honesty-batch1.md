# Sprint 38 Day 5 Coverage-Honesty Batch I

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Apply the narrowest high-value truthfulness fixes from the Day 2 audit by
correcting the top-level README coverage/testing language without changing the
underlying test or coverage machinery.

## Changes Made

### 1. Replaced the stale top-level testing/coverage claim

Updated the README testing intro so it no longer says:

- `1453 unit tests`
- `42 test suites`
- `>=95% line coverage (CI-enforced)`

The new wording now reflects the current maintained truth:

- default regression surface is `53` registered CTest test binaries
- coverage is a separate supplemental signal
- Linux coverage enforcement is an `80%` line-coverage threshold on `src/`
  for the default instrumented run path

### 2. Corrected the quick command-map coverage description

Updated the short command summary so `make coverage` is no longer described as
an lcov-only path. It now says:

- default line-coverage report on the active test surface
- `80%` threshold
- backend auto-selected

That matches the current Makefile behavior:

- GCC / Linux -> `coverage-lcov`
- Apple Clang / macOS -> `coverage-gcovr`

### 3. Corrected the testing command block threshold wording

Updated the testing section's `make coverage` note from:

- fails if `< 95%`

to:

- fails if `< 80%`

### 4. Expanded the test-category policy to include the large-matrix opt-in path

Added the missing live opt-in surface:

- `SPARSE_TEST_LARGE=1 make test`

This now sits alongside the existing wrapper-based opt-in categories:

- `SPARSE_TEST_SLOW=1 make test`
- `SPARSE_TEST_EXPERIMENTAL=1 make test`

The README now states directly that the large-matrix SuiteSparse path is live
supported coverage when enabled, but intentionally excluded from the default run
because of fixture/runtime cost.

## What This Batch Intentionally Did Not Change

- no test execution behavior
- no coverage threshold calibration
- no cross-platform coverage expansion
- no automatic enabling of slow / experimental / large opt-in paths inside
  `make coverage`
- no change to the reviewed baseline wrappers

## Residual Coverage-Honesty Queue

Narrowed after Day 5:

- check whether any nearby docs outside the README still casually flatten
  default regression execution and coverage instrumentation into one concept
- later Sprint 38 readiness/report wording can reinforce that coverage is a
  supplemental signal rather than part of the reviewed baseline
- no new gating work is justified yet from the Day 2 audit alone

## Validation

This was a docs-only batch. Validation was a targeted wording sanity pass
against:

- `README.md`
- `Makefile`
- `tests/test_framework.h`
- `tests/test_suitesparse.c`

No `*.c` or `*.h` behavior changed.
