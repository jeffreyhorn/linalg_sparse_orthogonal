# Sprint 38 Day 2 Coverage-Honesty Audit

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Audit coverage-related docs, target names, summaries, and artifact wording
against the actual active/opt-in test contract so Sprint 38 can fix truthfulness
drift without inventing fake new coverage or blurring the reviewed baseline.

## Current Ground Truth

### Default active regression surface

The default executed regression surface is:

- plain `RUN_TEST(...)` registrations in each `tests/test_*.c` binary
- executed by:
  - `make test`
  - `ctest`

Current suite-count facts:

- `tests/test_*.c` count = `53`
- `add_sparse_test(...)` count in `CMakeLists.txt` = `53`
- `ctest -N --test-dir build/quality-review-cmake` = `53`

### Explicit opt-in live test categories

The framework-level opt-in wrappers are:

- `RUN_TEST_SLOW(...)`
  - enabled by `SPARSE_TEST_SLOW=1`
- `RUN_TEST_EXPERIMENTAL(...)`
  - enabled by `SPARSE_TEST_EXPERIMENTAL=1`

There is also a separate non-wrapper opt-in live surface already present in the
suite:

- large-matrix SuiteSparse coverage in `tests/test_suitesparse.c`
  - enabled by `SPARSE_TEST_LARGE=1`

### Current line-coverage instrumentation surface

The maintained coverage targets are:

- `make coverage`
- `make coverage-lcov`
- `make coverage-gcovr`

Current threshold truth source:

- `Makefile` sets `COV_THRESHOLD = 80`
- the Sprint 29 calibration docs record the explicit decision to lower the
  inherited `95` threshold to `80`

Current CI truth:

- Linux CI has a supplemental coverage job
- coverage is not part of the reviewed baseline wrappers

## Ranked Coverage-Honesty Mismatches

### 1. Stale README threshold and suite-count language

Current README wording still says:

- `1453 unit tests`
- `42 test suites`
- `>=95% line coverage (CI-enforced)`

Why this is wrong:

- the current maintained test-binary / CTest count is `53`
- the current maintained coverage threshold is `80`, not `95`
- coverage is enforced in a supplemental Linux CI job, not as part of the
  reviewed baseline contract

Classification:

- `fix`

Priority:

- highest

### 2. README test-category policy is incomplete about live opt-in surfaces

Current README wording explains:

- `RUN_TEST_SLOW(...)`
- `RUN_TEST_EXPERIMENTAL(...)`

But it does not also name:

- `SPARSE_TEST_LARGE=1` for the large-matrix SuiteSparse path

Why this matters:

- the current text reads too much like the two wrapper-based categories are the
  entire non-default live surface
- that over-flattens the actual contract

Classification:

- `fix`

Priority:

- high

### 3. Coverage wording still blurs default executed tests vs opt-in executed tests vs line-coverage reports

Current repo behavior distinguishes three real concepts:

- default executed regression surface
- opt-in executed regression surface
- instrumented line coverage on the default run path

But some user-facing wording still compresses them into one "coverage" story.

Why this matters:

- it can make `make coverage` sound like it measures every live optional path
- it can make `make test` sound like it executes every live optional path

Classification:

- `fix`

Priority:

- high

### 4. Coverage is supplemental CI signal, not reviewed-baseline parity signal

Current contract is already mostly honest:

- README cross-platform CI table classifies coverage as supplemental on Linux
- reviewed wrappers do not include coverage

Residual issue:

- older broader README testing language can still be read as if coverage is part
  of the same baseline as format/lint/test/dead-code

Classification:

- `fix`

Priority:

- medium

### 5. Backend/platform coverage wording is mostly already correct

Current truthful surfaces:

- `INSTALL.md` explains:
  - GCC + lcov path
  - Apple Clang + gcovr path
- `Makefile` comments explain the Sprint 29 calibration and backend split
- Linux CI coverage job explicitly runs `make coverage`

Classification:

- `keep`

Priority:

- low

## Wording Problems vs Real Gating Gaps

### Wording problems

- stale `95%` threshold claim
- stale `42 test suites` claim
- incomplete README opt-in category explanation
- language that overstates what default `make test` / `make coverage` means

### Real gating gaps

- none required to complete Day 2

Notes:

- this audit did not find evidence that Sprint 38 first needs a new coverage
  execution target
- the first cleanup batch can stay documentation/reporting-focused

## Keep / Fix / Defer

### Keep

- `tests/test_framework.h` slow/experimental wrapper model
- `tests/test_framework_optin.c` self-check coverage
- `Makefile` `COV_THRESHOLD = 80`
- Linux supplemental coverage CI job
- `INSTALL.md` backend/platform coverage notes

### Fix

- README testing intro sentence with stale count/threshold claims
- README test-category policy to include the large-matrix opt-in surface
- README/coverage wording so default regression execution, opt-in execution,
  and line coverage are described separately
- any nearby wording that implies coverage is part of the reviewed baseline

### Defer

- automatically enabling slow/experimental/large tests in `make coverage`
- cross-platform coverage parity expansion
- reopening the Sprint 29 threshold calibration absent new measured evidence

## Day 5 Cleanup Direction

The safest first coverage-honesty batch is now clear:

1. fix the high-visibility README testing/coverage intro
2. keep the test-framework policy text, but extend it to mention the large-test
   opt-in surface
3. make the reviewed-baseline vs supplemental-coverage distinction explicit in
   the README coverage/readiness wording

That gives Sprint 38 a narrow, low-risk Day 5 implementation slice with high
truthfulness payoff.
