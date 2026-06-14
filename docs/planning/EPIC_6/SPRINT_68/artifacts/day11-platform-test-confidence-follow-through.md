# Sprint 68 Day 11: Platform-Test Confidence Follow-Through

Date: 2026-06-13
Branch: `sprint-68`

## Purpose

Tighten the platform-confidence wording that Sprint 68 actually moved after the
Day 6 giant-test extraction, Day 9 public-path oracle, and Day 10 property/fuzz
expansion, without widening into new CI behavior or fake platform closure.

## Why Day 11 Was Needed

Sprint 68 changed proof ownership in two concrete ways:

1. `tests/test_integration.c` now owns a stronger large-`n` CSC-backed
   Cholesky public-path oracle lane.
2. `tests/test_fuzz.c` now owns a bounded seeded generative large-`n`
   CSC-backed lifecycle parity lane.

That second change matters to platform confidence because Windows already keeps
`test_fuzz` outside its reviewed CMake subset.

If the docs and workflow comments did not say that plainly, the repo would
start implying broader reviewed Windows confidence than it actually has.

## Landed Surfaces

The follow-through stayed bounded to touched contract surfaces only:

- `README.md`
- `docs/maintainer_guide.md`
- `.github/workflows/windows-ci.yml`

No build logic, implementation files, benchmark semantics, or test lists moved.

## What Changed

### 1. README proof ownership is now explicit for the Sprint 68 additions

The large-`n` CSC-backed Cholesky proof split now states all three relevant
owners directly:

- `tests/test_chol_csc.c`
  - family-local analysis-backed/helper route
- `tests/test_integration.c`
  - public one-shot vs repeated-run parity/error-path contract
- `tests/test_fuzz.c`
  - bounded seeded generative lifecycle follow-through

The default test inventory was also updated:

- fuzz and property-based tests:
  - `24 -> 25`

### 2. The cross-platform CI contract now says what Windows does not prove

The Windows row in `README.md` now keeps the same staged exclusions, but makes
the implication explicit:

- `test_fuzz` remains excluded
- therefore the bounded Sprint 68 property/fuzz lifecycle lane is outside the
  reviewed Windows subset

This tightens interpretation without changing the CI surface itself.

### 3. The maintainer guide now owns the Sprint 68 platform-confidence meaning

`docs/maintainer_guide.md` now records:

- the post-Sprint-68 maintained proof ownership split
- that Linux/macOS still exercise the full `test_fuzz` binary in the relevant
  direct/reviewed paths
- that Windows still excludes `test_fuzz`, so the new property lane must not be
  read as reviewed Windows evidence

That keeps the narrower confidence story in the policy owner rather than
scattered only in top-level docs.

### 4. The Windows workflow comment/job output now matches the docs

The workflow comment and staged-exclusion output now say directly that:

- `test_fuzz` remains excluded
- this includes Sprint 68's bounded lifecycle property lane

That keeps workflow logs aligned with the maintained documentation story.

## Non-Widening Fence Preserved

This Day 11 batch did not:

- change any implementation or test logic
- add new Windows jobs or remove staged exclusions
- widen the reviewed Windows claim
- move benchmark or install/package platform ownership

It is a wording/truthfulness batch only.

## Validation

This was a docs/workflow-comment-only batch, so no `*.c` / `*.h` validation was
required and no reviewed baseline rerun was needed.

Targeted Day 11 sanity checks were sufficient:

- touched-surface diff review
- terminology/alignment `rg`
- branch-status recheck

## Exit State

Sprint 68 Day 11 closes with the platform-confidence story tightened exactly
where Sprint 68 moved proof ownership:

1. the large-`n` CSC-backed proof split is now explicit across:
   - `tests/test_chol_csc.c`
   - `tests/test_integration.c`
   - `tests/test_fuzz.c`
2. Windows still excludes `test_fuzz`, and the docs/workflow now say plainly
   that the new bounded property lane is therefore outside the reviewed Windows
   subset
3. no new CI promises or platform closure claims were introduced

That gives Day 12 a cleaner alignment target:

- finish the maintained docs/regression-surface wording so the landed Sprint 68
  proof ownership reads consistently across the remaining truth surfaces.
