# Sprint 33 Day 10 Cleanup Batch I

**Date:** 2026-05-18  
**Branch:** `sprint-33`

## Objective

Remove the only approved definitely-unused internal candidate from the Sprint 33 queue and validate both the touched CSC-Cholesky subsystem and the full project gate afterward.

## What Changed

Removed `chol_csc_dump_supernodes`:

- deleted the private declaration from `src/sparse_chol_csc_internal.h`
- deleted the debug-only implementation from `src/sparse_chol_csc.c`

Why this was safe:

- internal-only declaration surface
- compiled only under `#ifndef NDEBUG`
- no remaining in-repo callers
- no public-header or documented CLI/API surface involved

## Focused Validation

Focused commands:

1. `make build/test_chol_csc`
2. `./build/test_chol_csc`
3. `make deadcode-report`
4. `make deadcode-check`

Results:

- `test_chol_csc`: passed
- `deadcode-report`: passed
- `deadcode-check`: passed

Post-removal report bucket counts:

- `coverage-gap`: `7`
- `definitely-unused-internal-candidate`: `0`
- `public-surface-review`: `4`
- `secondary-candidate-signal`: `35`
- `non-deadcode-static-analysis-noise`: `6`

Most important outcome:

- the first-pass internal deletion queue is now empty

## Required Full Validation

Because Batch I touched `*.c` and `*.h`, the required full validation was run:

1. `make format`
2. `make lint`
3. `make test`

Results:

- all passed

## Report Delta

Before Batch I:

- `definitely-unused-internal-candidate`: `1`
  - `chol_csc_dump_supernodes`

After Batch I:

- `definitely-unused-internal-candidate`: `0`

Secondary signal note:

- `src/sparse_chol_csc.c` remains present in the aggregated `cppcheck` secondary bucket, but the high-confidence `xunused` cleanup candidate is gone

## Workflow Lesson

One parallel pair of dead-code target invocations exposed a tooling race:

- concurrent `make deadcode-report` / `make deadcode-check` attempts can fight over the shared `build/deadcode-cmake` configure tree
- one such run produced a transient `configure_file(...)` failure during CMake configure
- a serial rerun passed cleanly

Interpretation:

- this is a dead-code workflow concurrency limitation, not a code regression
- Sprint 33 should treat the `deadcode*` targets as serial operator tooling unless the build-tree isolation is improved later

## Day 10 Conclusion

Batch I completed the only approved first-pass internal deletion work from Sprint 33’s current queue.

That sets up Day 11 correctly as report reconciliation unless the rerun surfaces a new equally strong internal candidate.
