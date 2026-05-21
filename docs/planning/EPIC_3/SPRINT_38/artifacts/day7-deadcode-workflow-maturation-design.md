# Sprint 38 Day 7 Dead-Code Workflow Maturation Design

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Define the safest next-stage dead-code workflow refinement now that the Sprint
34 compile-db exclusion list is closed and the report has reached a zero-gap
state.

## Current Ground Truth After Day 6

Current live dead-code bucket counts from `build/deadcode/report.tsv`:

- `coverage-gap` = `0`
- `definitely-unused-internal-candidate` = `0`
- `public-surface-review` = `4`
- `secondary-candidate-signal` = `35`
- `non-deadcode-static-analysis-noise` = `6`

Current report meaning:

- no current benchmark/example compile-db coverage gaps are recorded
- no cleanup-ready internal dead-code queue is currently classified
- public-surface rows remain audited keeps
- `cppcheck` secondary/noise rows remain supporting or explanatory data only

Current enforced `deadcode-check` boundary:

- report/check generation succeeds
- `xunused` findings are categorized
- report completeness invariants hold

It still does **not** mean:

- zero findings
- zero static-analysis noise
- cleanup-ready code exists
- concurrent-safe execution

## Chosen Sprint 38 Dead-Code Maturity Step

The next batch should improve routine operator signal without changing the
staged contract.

Chosen scope for Day 8:

1. refine dead-code report wording for the zero-gap state
2. make the "no internal cleanup queue" outcome more direct
3. keep audited public-surface keeps visible but clearly non-actionable
4. keep `cppcheck` supporting evidence visible but clearly outside direct
   cleanup instructions
5. preserve the current completeness-based `deadcode-check` model

## What Belongs In The Day 8 Batch

### Report structure / wording refinements

High-value candidates:

- make the coverage section read as closure, not lingering staged debt
- make the internal-candidate section read as an explicit empty queue
- tighten the next-action queue so it points to:
  - no current internal cleanup batch
  - public audited keeps
  - deferred future review of supporting `cppcheck` evidence

### Optional check/message refinements

Allowed only if needed to match the report wording:

- clarify that `deadcode-check` is a completeness gate, not a "zero findings"
  gate
- clarify that the authoritative path remains serialized

## What Explicitly Does Not Belong In This Batch

- content-based failure logic over `secondary-candidate-signal`
- content-based failure logic over `non-deadcode-static-analysis-noise`
- removal or hiding of public audited keeps
- any claim of concurrent-safe `deadcode*` execution
- any claim of broader multi-platform dead-code enforcement
- shared-path topology changes

## Residual Limitations That Remain Staged

### 1. Serialized execution model

The workflow still depends on shared paths:

- `build/deadcode-cmake`
- `build/deadcode/`

Meaning:

- authoritative validation remains serial
- concurrent `deadcode*` calls are still not part of the supported contract

### 2. Supporting-evidence noise remains intentionally non-gating

Residual `cppcheck` evidence:

- `secondary-candidate-signal = 35`
- `non-deadcode-static-analysis-noise = 6`

Meaning:

- these rows remain useful for future prioritization
- they are still not trustworthy enough for direct cleanup or pass/fail logic

## Day 8 Implementation Contract

Day 8 should ship a small report/check refinement batch with this validation
path:

- `python3 -m py_compile scripts/deadcode_report.py`
- `make deadcode-report`
- `make deadcode-check`

Success means:

- dead-code output becomes easier to read in the zero-gap state
- the completeness gate remains truthful
- the staged serialized contract remains explicit

That is the safest meaningful dead-code maturity step available after the Day 6
compile-db closure.
