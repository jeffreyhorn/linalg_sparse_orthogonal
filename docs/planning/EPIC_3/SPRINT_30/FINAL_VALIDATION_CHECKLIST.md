# Sprint 30 Final Validation Checklist

**Status:** Prepared on Day 12  
**Sprint:** 30  
**Scope:** Day 13 validation pass and Day 14 closeout sanity checks

## Day 13 Validation Pass

### Primary workflow run

- Run `make warning-workflow WARNING_WORKFLOW_LABEL=day13-final`
- Confirm the workflow completes successfully
- Confirm serialized warning capture remains in effect (`WARNING_WORKFLOW_JOBS=1` unless intentionally overridden and documented)

### Expected warning-count state

Authoritative Apple Clang CMake full-tree path:

- configure warnings: `0`
- full-tree warnings: `112`
- by area:
  - `src`: `0`
  - `tests`: `98`
  - `benchmarks`: `13`
  - `examples`: `1`
- by class:
  - `-Wmissing-field-initializers`: `72`
  - `-Wdouble-promotion`: `34`
  - `-Wunused-function`: `3`
  - `-Wimplicit-function-declaration`: `2`
  - `-Wswitch`: `1`

Makefile `all` path:

- warnings: `0`

### Regression expectations

- Confirm the warning counts match the post-Day-9 state exactly, unless a deliberate Day 13 or Day 14 change is made and documented
- Confirm there is no reappearance of `src/` warnings
- Confirm there is no new strict-only `src/` warning debt if any stricter pass is replayed

### Test and quality gates

- Run `make format`
- Run `make lint`
- Run `make test`
- Confirm all three complete successfully

### Artifact checks

- Confirm the workflow writes the expected warning logs and summary files under `docs/planning/EPIC_3/SPRINT_30/artifacts`
- Confirm the final warning-count artifacts reflect the current branch state rather than an older intermediate build
- Record the Day 13 validation outputs in `WORKING_NOTES.md`

## Day 14 Closeout Checks

### Consistency checks

- Reconfirm the Day 1 baseline and post-core-fix counts are still stated consistently across:
  - `artifacts/day1-warning-baseline.md`
  - `COMPILE_HYGIENE_PLAYBOOK.md`
  - `artifacts/day12-baseline-reconciliation.md`
  - `WORKING_NOTES.md`
- Reconfirm the follow-up queues for Sprint 31 and later remain consistent across Day 10, Day 11, and Day 12 artifacts

### Closeout summary content

- Summarize:
  - baseline established
  - core-library warnings reduced from `11` to `0`
  - full-tree warnings reduced from `123` to `112`
  - rebuild workflow documented
  - tests triaged
  - benchmarks/examples triaged
  - strict-pass findings reconciled

### Branch state

- Ensure end-of-day validation has been rerun before the final Day 14 commit:
  - `make format`
  - `make lint`
  - `make test`
- Ensure the worktree is clean after commit

## Failure Handling

If Day 13 or Day 14 diverges from the expected counts:

1. Identify whether the delta is in `src/`, `tests/`, `benchmarks/`, or `examples`
2. Determine whether the delta is a real regression, a tooling-capture change, or an intentional warning cleanup
3. Update the counts and narrative only after the cause is understood
4. Do not silently overwrite the expected `112`-warning post-Day-9 state without documenting why it changed
