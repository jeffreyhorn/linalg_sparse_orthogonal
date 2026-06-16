# Sprint 71 Day 6: Front-Door & Install Cleanup Batch

Date: 2026-06-16
Branch: `sprint-71`

## Purpose

Land the first bounded Sprint 71 cleanup batch on `README.md` and
`INSTALL.md` without widening claims or dragging in support/reference
surfaces.

## Landed Batch

### `README.md`

The landed front-door cleanup tightened:

- repeated-run direct-workflow handoff
- examples / benchmarks / tests ownership wording
- canonical maintained benchmark-surface wording
- install/package summary wording

The cleanup stayed bounded:

- `example_analysis` remains the strongest small adoption entry point
- the tutorial remains the fuller step-by-step teaching flow
- benchmark surfaces remain the retained workflow/performance proof surfaces
- tests remain the regression/oracle/property owners

### `INSTALL.md`

The landed install cleanup tightened:

- quick-start compile-quality wrapper wording
- the split between the front-door command map and the install guide
- the focused install-proof interpretation

The install contract itself stayed fixed:

- the maintained release shape remains static-first
- local install/package proof scripts remain explicit
- Windows remains the reviewed CMake-first consumer story

## Preserved Truth Checklist

The Day 6 batch preserved:

- the orthogonal linked-list public-center reading
- examples / benchmarks / tests ownership
- `make bench-canonical-report` as threshold-free artifact reporting
- the static-first install/export contract
- reviewed Linux/macOS/Windows asymmetry
- Windows as the reviewed CMake-first consumer story rather than a reviewed
  install-validation lane

## Non-Widening Result

The batch did not widen into:

- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- public headers
- implementation files
- proof-owner tests
- workflow files

## Day 6 Sanity Recheck

Touched-surface raw `wc -l` counts after the landing:

- `README.md` = `1037`
- `INSTALL.md` = `237`

Targeted terminology checks still read correctly:

- examples = workflow/adoption teaching
- benchmarks = retained workflow/performance proof
- tests = regression/oracle/property ownership
- install scripts = maintained supplemental local proof
- Windows = reviewed CMake-first consumer story

## Exit State

Sprint 71 Day 6 closes with the first bounded public cleanup batch landed:

1. `README.md` is materially tighter as the front door
2. `INSTALL.md` is materially tighter as the install/operator surface
3. the landed wording stays inside the Day 5 fence
4. the support/reference queue remains untouched for the post-landing audit
