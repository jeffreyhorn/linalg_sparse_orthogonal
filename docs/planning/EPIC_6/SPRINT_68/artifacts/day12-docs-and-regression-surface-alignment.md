# Sprint 68 Day 12: Docs & Regression-Surface Alignment

## Goal

Finish the maintained docs/regression-surface wording so the landed Sprint 68
test and assurance boundaries read consistently across the caller-facing truth
surfaces before the validation sweep.

## Why Day 12 Was Needed

By the end of Day 11, the branch already said who owns the new Sprint 68
assurance lanes:

- `tests/test_integration.c`
- `tests/test_fuzz.c`
- `tests/test_chol_csc.c`

But one smaller contradiction remained on the maintained docs side:

- the test owners were named clearly
- the example and benchmark surfaces still under-stated what they do **not**
  own

That mattered because Sprint 68 deliberately kept three different proof roles
separate:

1. examples teach workflow adoption
2. benchmarks prove retained workflow/performance behavior
3. tests own the regression, oracle, and bounded property guarantees

Day 12 closes that wording gap without widening into new implementation,
workflow, or benchmark behavior.

## Landed Surfaces

- `README.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`

## What Changed

### 1. README now states the non-ownership split directly

The top-level workflow summary now says plainly that the stronger numerical
oracle/property guarantees for the large-`n` CSC-backed Cholesky lifecycle stay
test-owned, not example- or benchmark-owned.

The repeated-run CSC interpretation section also now says directly that:

- `example_analysis` teaches the repeated-run workflow
- `bench_refactor` / `bench_refactor_csc` prove retained workflow/performance
  behavior
- neither replaces the test-owned oracle/property lanes

### 2. `examples/README.md` now keeps `example_analysis` in the adoption lane

`example_analysis` already said it was not the full error-path owner. Day 12
tightens that wording so it now points to the exact retained regression owners:

- staged public one-shot vs repeated-run parity and failed-refactor
  preservation:
  - `tests/test_integration.c`
- bounded seeded generative lifecycle follow-through:
  - `tests/test_fuzz.c`

That keeps examples in the teaching lane instead of drifting into proof-owner
language.

### 3. `benchmarks/README.md` now keeps the benchmark surfaces benchmark-side

`bench_refactor_csc` already said it was not the failed-refactor owner and not
the family-local helper-parity owner. Day 12 adds the missing Sprint 68 note:

- the bounded seeded generative large-`n` lifecycle follow-through stays owned
  by `tests/test_fuzz.c`

`bench_chol_csc` now also says directly that it is **not** the owner of:

- the Sprint 68 staged public-path oracle/parity lane
- the bounded seeded lifecycle property lane

Those remain test-owned in:

- `tests/test_integration.c`
- `tests/test_fuzz.c`

### 4. The maintainer guide now owns the final pre-validation ownership split

`docs/maintainer_guide.md` now makes the separation explicit across all three
surface types:

- tests own the regression/oracle/property guarantees
- `examples/example_analysis.c` stays example-side and teaches the repeated-run
  lifecycle
- benchmark surfaces stay benchmark-side and do not replace the family-local,
  public oracle, or property owners

That gives one maintained policy owner for the final Sprint 68 proof split
before validation.

## Preserved Fence

This Day 12 batch did not:

- touch any `src/` implementation file
- change any test binary, benchmark binary, or example binary behavior
- widen platform claims beyond the Day 11 contract
- add new benchmark or example promises
- reopen the Sprint 68 giant-test or property/fuzz code paths

## Validation

This was a docs-only alignment batch, so no `*.c` / `*.h` validation was
required.

Targeted Day 12 sanity checks were sufficient:

- touched-doc diff review
- terminology/alignment `rg`
- touched-surface `wc -l`
- branch-status recheck

## Bottom Line

Sprint 68 Day 12 closes the remaining maintained-docs contradiction from the
Day 9-11 assurance work:

1. the repo now says clearly which tests own the new large-`n` CSC-backed
   Cholesky assurance lanes
2. the example and benchmark surfaces now say clearly that they do **not** own
   those regression guarantees
3. the branch is now aligned for the Day 13 full validation sweep without
   stale ownership wording on the touched surfaces
