# Sprint 38 Day 10 Quality-Gate Expansion Batch I

**Date:** 2026-05-21  
**Branch:** `sprint-38`

## Objective

Implement the smallest meaningful next-tier quality-gate expansion from the
Day 9 design by adding one explicit local aggregate wrapper over the maintained
reviewed Makefile and reviewed CMake paths.

## Changes Made

### 1. Added a new top-level local reviewed aggregate wrapper

Added:

- `make quality-review-full`

to `Makefile`.

Behavior:

- runs `make quality-review`
- then runs `make quality-review-cmake`
- remains serial and bannered
- gives direct rerun guidance for whichever half fails

### 2. Kept the aggregate wrapper out of the staged/supplemental surfaces

The new wrapper intentionally does **not** include:

- `make coverage`
- `make wall-check`
- `make sanitize`
- any dead-code promotion beyond the existing `quality-review` path
- any cross-platform staged/excluded path promotion

That preserves the Sprint 36 enforced/staged/excluded contract and the Sprint
38 coverage-honesty cleanup.

### 3. Updated the README command map

Updated `README.md` so it now says more directly:

- `make quality-review-full` is the strongest local reviewed baseline command
- `quality-review-full` runs:
  - `make quality-review`
  - `make quality-review-cmake`

## Validation

Direct validation:

- `make -n quality-review-full`
- `make quality-review-full`

Observed end state:

- `quality-review-full` ran the reviewed Makefile path first
- then ran the reviewed CMake parity path
- `ctest -N` remained `53`
- full reviewed CMake `ctest` passed `53 / 53`
- final wrapper output ended with:
  - `quality-review-full: passed (quality-review + quality-review-cmake)`

## What This Batch Did Not Change

- no lower-level reviewed primitive semantics
- no Linux/macOS/Windows CI job ownership
- no macOS dead-code staging status
- no Windows local Makefile reviewed-wrapper staging status
- no Windows dead-code exclusion status
- no coverage classification

## Residual Queue After Day 10

Still remaining for later Sprint 38 work:

- readiness-checklist design and implementation
- targeted CI/reporting polish
- later staged/excluded cross-platform follow-through already owned by future
  sprint work

Closed by this batch:

- lack of a single named strongest local reviewed baseline command
