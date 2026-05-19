# Sprint 33 Day 9 Cleanup Batch Design

**Date:** 2026-05-18  
**Branch:** `sprint-33`

## Objective

Turn the post-Day-8 refined queue into a realistic cleanup plan for Days 10 and 11 without overstating how much first-pass deletion work Sprint 33 actually has left.

## Queue State Entering Day 9

After the Day 8 public-surface audit:

- cleanup-ready internal candidates: `1`
  - `chol_csc_dump_supernodes`
- public-surface review queue:
  - closed as `keep`
- compile-db blind spots:
  - unchanged
- `cppcheck` secondary signals:
  - still supporting evidence only

## Main Design Decision

Sprint 33 no longer has enough approved deletion work to justify two separate implementation batches across multiple areas.

The truthful plan is:

- Day 10 removes the only approved internal candidate
- Day 11 reconciles the regenerated report and only removes more code if a new high-confidence internal candidate appears after rerunning the tooling

That is a better fit for the actual queue than inventing a second deletion batch just to match the original sprint-plan shape.

## Batch I Scope

Day 10 Batch I should remove:

- `chol_csc_dump_supernodes`

Touched files:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_internal.h`

Why this is low risk:

- internal-only declaration surface
- `#ifndef NDEBUG` debug helper only
- no in-repo callers remain
- zero public header or CLI surface involved

## Day 11 Reconciliation Scope

Day 11 should:

1. regenerate `make deadcode-report`
2. compare the new report against the Day 7 baseline
3. record whether the internal-candidate bucket reached zero
4. classify the remaining rows as:
   - coverage gaps
   - intentional public keeps
   - secondary evidence
   - static-analysis noise

Only if the rerun produces a new high-confidence internal candidate should Day 11 perform another code removal.

## Focused Validation Plan

Because Day 10 will edit `*.c` and `*.h`, the required project-wide validation is:

1. `make format`
2. `make lint`
3. `make test`

Focused subsystem checks:

1. `make build/test_chol_csc`
2. `./build/test_chol_csc`
3. `make deadcode-report`
4. `make deadcode-check`

Why this set:

- `test_chol_csc` directly exercises the touched CSC-Cholesky internals
- the dead-code workflow proves the candidate actually leaves the actionable queue
- the full project gate preserves the repo’s normal code-edit standard

## Expected Outcomes

Expected Day 10 outcome:

- `chol_csc_dump_supernodes` removed
- internal-candidate bucket drops from `1` to `0`

Expected Day 11 outcome:

- no remaining first-pass internal deletion queue
- report reduced to:
  - coverage gaps
  - public-surface keeps
  - secondary `cppcheck` evidence
  - static-analysis noise

## Day 9 Conclusion

The Day 9 plan is intentionally small because the approved queue is intentionally small.

That is the correct outcome of the Sprint 33 policy/reporting work:

- exported symbols were filtered out on Day 8
- broad `cppcheck` noise stayed out of the deletion queue
- one internal candidate remains for Day 10
- Day 11 becomes reconciliation unless the rerun surfaces another equally strong internal candidate
