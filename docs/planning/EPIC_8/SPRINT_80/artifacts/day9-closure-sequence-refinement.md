# Sprint 80 Day 9: Closure Sequence Refinement

## Purpose

Confirm that the Epic 8 gap-closure todo still closes the major review
findings in a credible order after Sprint 80 fixed the baseline, external,
benchmark, and non-goal contracts.

## Refinements Made

Updated directly in `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md`:

- added a Sprint 80 alignment preface so the todo is now read against:
  - the refreshed reviewed baseline
  - the bounded first external-oracle lane
  - the threshold-free benchmark contract
  - the explicit Epic 8 non-goal fence
- tightened Stage 1 external-reference wording:
  - CHOLMOD-class SPD direct-solver comparison first
  - BLAS/LAPACK-class references as performance-reference support
- tightened Stage 3 benchmark wording so benchmark measurability does not
  silently reopen timing-gate interpretations
- tightened Stage 5 reviewed cross-platform proof wording so Linux remains the
  strongest reviewed truth unless later evidence changes that
- tightened Stage 8 packaging wording so shared-library work remains optional
  and proof-backed rather than assumed
- tightened Stage 10 comparison wording so final claim calibration stays
  bounded instead of widening into ecosystem-comparison theater

## Closure Sequence Cross-check

The todo still closes the major gaps in the right order:

1. baseline and competitive/measurement contract
2. storage/workflow ceiling
3. dense/backend ceiling
4. capability breadth
5. assurance expansion
6. maintainability concentration
7. runtime long-pole reduction
8. packaging/platform convergence
9. front-door usability simplification
10. final comparison and claim calibration

## Deferred / Non-goal Reconciliation

The refined todo no longer implies:

- automatic broad external sparse-solver comparison
- portable benchmark-threshold verdicts
- automatic cross-platform parity expansion
- assumed shared-library maturity
- broad scalar-family genericity by default
- whole-library rewrite pressure

Those remain deferred or explicitly outside the first Epic 8 contract unless
later proof-backed work justifies them.

## Day 9 Exit State

The Epic 8 todo still represents the closure sequence for the review findings,
but it now reads from the landed Sprint 80 contract package and no longer
quietly reopens disallowed non-goals.
