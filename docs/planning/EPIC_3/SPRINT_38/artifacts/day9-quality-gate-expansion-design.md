# Sprint 38 Day 9 Quality-Gate Expansion Design

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Choose the safest next-tier quality-gate expansion after the Sprint 38 coverage
truthfulness, compile-db closure, and dead-code report maturation work.

## Current Gate Ground Truth

### Maintained reviewed local primitives already exist

Current local reviewed operator paths:

- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`
- `make deadcode-report`
- `make deadcode-check`

Meaning:

- the repo does not need another primitive compile/test/dead-code gate before
  expanding quality signaling
- it needs a clearer next-tier aggregate over the primitives that are already
  stable

### Coverage remains supplemental by design

After the Day 5 wording cleanup:

- coverage is a supplemental signal
- Linux coverage still enforces the `80%` threshold
- coverage is not part of the reviewed cross-platform baseline

Meaning:

- quality-gate expansion should not fold `make coverage` into the main reviewed
  baseline
- doing so would undo the Sprint 38 coverage-honesty work

### Cross-platform staged/excluded boundaries remain honest and should stay that way

Still staged or excluded:

- macOS dead-code
- Windows local Makefile reviewed-wrapper parity
- Windows dead-code

Meaning:

- Day 10 should not try to promote staged/excluded paths by naming them as
  enforced
- the next safe expansion should stay local and platform-neutral

## Chosen Expansion Batch

The safest next expansion is a single local aggregate reviewed baseline target.

Chosen shape for Day 10:

1. add one serial top-level wrapper that runs:
   - `make quality-review`
   - `make quality-review-cmake`
2. banner it clearly so failure attribution stays obvious
3. document it as the strongest local reviewed baseline command
4. keep existing reviewed primitives unchanged

## Why This Is The Right Batch

### It improves regression-proofing without inventing new gate semantics

The aggregate wrapper would not create new lower-level checks. It would simply
make the full maintained local reviewed baseline easier to run routinely.

### It preserves the Sprint 36 platform contract

The wrapper is local and platform-neutral:

- it does not imply macOS dead-code enforcement
- it does not imply Windows local Makefile wrapper enforcement
- it does not change the current CI enforced/staged/excluded contract

### It sets up later readiness-checklist work cleanly

A single named "run the strongest local reviewed baseline" target is easier to
reference in a concise readiness checklist than two separate top-level
commands.

## What Day 10 Should Not Do

- add `make coverage` to the reviewed aggregate
- add `make wall-check` or `make sanitize` to the reviewed aggregate
- reclassify staged cross-platform paths as enforced
- change dead-code bucket semantics
- change Linux/macOS/Windows CI job ownership

## Residual Staged/Excluded Surfaces After The Planned Batch

These remain outside the planned Day 10 expansion:

- Linux supplemental coverage job
- Linux supplemental benchmark/sanitizer jobs
- macOS dead-code
- Windows local Makefile reviewed-wrapper parity
- Windows dead-code

## Day 10 Implementation Contract

Day 10 should ship:

- one new serial aggregate reviewed wrapper
- concise operator-facing messaging for that wrapper
- README command-map update for the new strongest local reviewed baseline

Validation should stay direct and attributable:

- `make -n <new-wrapper>`
- live run of the new wrapper

That is the smallest meaningful quality-gate expansion available after the
Sprint 38 Days 5-8 groundwork.
