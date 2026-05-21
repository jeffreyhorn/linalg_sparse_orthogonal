# Sprint 38 Day 8 Dead-Code Workflow Maturation Batch I

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Improve the routine dead-code report/check signal for the current zero-gap
state without changing classifier semantics, adding content-based failure
rules, or implying concurrent-safe workflow execution.

## Changes Made

### 1. Tightened zero-gap coverage wording in the markdown report

Updated `scripts/deadcode_report.py` so the `## Coverage Gaps` section now says
both:

- no current benchmark/example compile-db coverage gaps are recorded
- the dead-code compile database currently covers the maintained
  benchmark/example tooling surface

This keeps the report truthful after the Day 6 compile-db expansion and makes
the closure state explicit instead of merely omitting gap rows.

### 2. Reframed the empty internal-candidate bucket as an empty cleanup queue

Updated the `## Definitely-Unused Internal Candidates` empty-state wording to:

- `No current definitely-unused internal cleanup batch is classified in this run.`

This is a more useful operator-facing summary than the older generic empty
bucket wording.

### 3. Tightened the wording around audited keeps and supporting evidence

Updated the markdown report so it now says more directly:

- public-surface rows are audited keeps, not cleanup
- `cppcheck` secondary rows are supporting evidence only
- `cppcheck` secondary rows are not direct removal instructions or current
  pass/fail criteria

### 4. Refined the next-action queue for the zero-gap state

Updated the report next-action queue so it now points to:

- no current definitely-unused internal queue in the current report
- public audited keeps remaining visible for context, not cleanup
- `cppcheck` supporting evidence staying summarized for later review work
- serialized authoritative validation through:
  - `make deadcode-report`
  - `make deadcode-check`

### 5. Tightened the `deadcode-check` success message

Updated `Makefile` so `make deadcode-check` now says:

- report completeness checks passed
- this is not a zero-findings gate
- authoritative execution remains serialized

That aligns the command output with the actual staged contract more directly.

## Validation

Authoritative serial validation:

- `python3 -m py_compile scripts/deadcode_report.py`
- `make deadcode-report`
- `make deadcode-check`

Observed post-batch report state:

- `coverage-gap = 0`
- `definitely-unused-internal-candidate = 0`
- `public-surface-review = 4`
- `secondary-candidate-signal = 35`
- `non-deadcode-static-analysis-noise = 6`

Observed `deadcode-check` output:

- `deadcode-check: report completeness checks passed (not a zero-findings gate).`
- `deadcode-check: authoritative execution remains serialized; inspect build/deadcode/report.md and build/deadcode/report.tsv.`

## What This Batch Did Not Change

- no bucket classifications
- no `xunused` classification rules
- no `cppcheck` content-based gating
- no shared-path isolation
- no multi-platform dead-code expansion

## Residual Queue After Day 8

Still remaining for later work:

- shared-path serialization / execution-model maturity
- future review of supporting `cppcheck` evidence and summarized noise
- later readiness/reporting integration work

Closed by this batch:

- zero-gap report wording drift
- generic empty internal-queue wording drift
- ambiguous `deadcode-check` success wording
