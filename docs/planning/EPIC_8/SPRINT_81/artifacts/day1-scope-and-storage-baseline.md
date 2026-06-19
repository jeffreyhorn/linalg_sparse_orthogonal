# Sprint 81 Day 1: Scope and Storage Baseline

## Purpose

Turn the Sprint 81 project-plan section and Sprint 80 closeout into one bounded
product/storage execution package before implementation work begins.

## Starting State

Sprint 80 closed with:

- a refreshed Epic 8 reviewed baseline
- a ranked contradiction map
- a bounded external-oracle contract
- a bounded benchmark/performance contract
- an explicit non-goal and risk fence

Sprint 81 therefore starts from a stronger place than a generic “next sprint”
handoff. The strongest remaining first-tier contradiction is still the
linked-list-first product/storage ceiling.

One practical Day 1 correction was also required:

- the current `master` branch did not carry `docs/planning/EPIC_8/`
- the Epic 8 planning tree was restored from `origin/sprint-80` so Sprint 81
  could execute against the actual Epic 8 source plan and Sprint 80 closeout
  record

## Live Validation Anchor

The strongest local reviewed baseline remains:

- `make quality-review-full`

The reviewed CMake parity anchor was re-materialized live:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`

## Sprint 81 Workstreams

Sprint 81 is now fixed around one bounded product/storage modernization lane:

- storage/conversion audit
- compressed-first architecture design
- construction/import landing
- repeated-run workflow convergence
- focused proof and benchmark follow-through
- docs/examples/header alignment only where implementation truly forces it

## Strongest Likely Touch Surfaces

The highest-signal product/storage and proof surfaces are explicit from the
live tree:

- `include/sparse_matrix.h` = `610`
- `src/sparse_matrix.c` = `1125`
- `src/sparse_cholesky.c` = `615`
- `src/sparse_ldlt.c` = `1535`
- `src/sparse_qr.c` = `1563`
- `tests/test_sparse_matrix.c` = `1071`
- `tests/test_integration.c` = `2689`
- `benchmarks/bench_refactor_csc.c` = `611`
- `benchmarks/README.md` = `393`
- `README.md` = `1050`
- `docs/planning/EPIC_8/PROJECT_PLAN.md` = `351`

## First-Day Clarifications

The strongest Day 1 clarifications are now explicit:

- Sprint 81 should not reopen Sprint 80’s baseline/oracle/benchmark contract
  package
- Sprint 81 should not widen into backend, capability, or package/platform
  work
- the first bounded implementation center should come from the
  linked-list-first product/storage seam, not from a broad architecture rewrite

## Preserved Non-Goal Fence

Sprint 81 starts with these preserved non-goals fixed directly:

- no backend/performance lane spill
- no capability-surface widening
- no broad package/platform reopening
- no generic whole-library workflow rewrite
- no broad public API redesign hidden inside storage cleanup

## Day 1 Exit State

Sprint 81 now starts from one explicit validated baseline, one explicit
product/storage workstream map, and one preserved non-goal fence. Day 2 can
move on to the validation/proof recheck without ambiguity.
