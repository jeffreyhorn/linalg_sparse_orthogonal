# Sprint 38 Day 11 Release/Readiness Checklist Design

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Choose the canonical scope, landing surface, and surrounding support-doc
boundaries for a concise quality-readiness checklist.

## Chosen Landing Surface

The checklist should live in `README.md`.

Why:

- `README.md` already owns the maintained quality command map
- `README.md` already owns the cross-platform CI contract
- `INSTALL.md` already points back to the README for the canonical operator
  path instead of competing with it

Rejected alternatives:

- a sprint-local planning artifact only
- a separate standalone checklist doc with duplicated command explanations
- `INSTALL.md` as the primary landing surface

## Chosen Checklist Scope

The checklist should stay compact and criterion-based.

Chosen criteria:

1. strongest local reviewed baseline passes:
   - `make quality-review-full`
2. dead-code completeness/report path remains truthful:
   - `make deadcode-report`
   - `make deadcode-check`
3. active test-surface / CMake parity remains truthful:
   - current `ctest -N` surface
   - current reviewed CMake parity contract
4. coverage wording stays truthful:
   - supplemental signal
   - `80%` threshold on `src/` in the Linux coverage path
5. docs/examples/header usage stays consistent with shipped behavior
6. cross-platform enforced/staged/excluded boundaries stay named honestly

## What The Checklist Should Not Become

The checklist should **not** become:

- a duplicate of the full command map
- a duplicate of the dead-code report explainer
- a duplicate of the cross-platform CI contract table
- a false claim that supplemental or staged paths are part of the reviewed
  baseline

## Keep / Link / Defer Classification

### Keep in the checklist

- short pass/fail-style readiness criteria
- one-line clarifications where staged/supplemental distinctions matter

### Link or reference from the checklist

- reviewed command map
- dead-code explanation section
- cross-platform CI contract
- coverage section

### Defer outside the checklist

- broad README restructuring
- new CI or target semantics
- detailed troubleshooting prose

## Day 12 Implementation Contract

Day 12 should ship:

- one concise README checklist section
- only the smallest adjacent wording polish needed so the checklist reads
  cleanly

Validation should stay docs/report-surface-focused rather than reopening the
full validation matrix early.
