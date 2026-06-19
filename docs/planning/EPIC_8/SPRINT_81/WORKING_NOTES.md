# Sprint 81 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 81 baseline for Epic 8 by grounding the sprint in
the validated Sprint 80 close state, the live Epic 8 project-plan section, and
the current permanent validation, product/storage, benchmark, and proof-owner
surfaces rather than another generic implementation start.

### Actions
- Re-read the Sprint 81 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 81 day-by-day plan in `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the strongest Sprint 80 closeout context:
  - `docs/planning/EPIC_8/SPRINT_80/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_80/RETROSPECTIVE.md`
- Restored the Epic 8 planning tree from `origin/sprint-80` because the
  current `master` branch did not carry `docs/planning/EPIC_8/`.
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 81
  touch surfaces across product/storage, direct-workflow, proof-owner, and
  support surfaces.
- Opened Sprint 81 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 81 starts from the same strongest local reviewed baseline Sprint 80
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 81 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 81 is not a broad “storage rewrite” sprint. Its highest value is one
  bounded product/storage modernization package centered on:
  - storage/conversion audit
  - compressed-first architecture design
  - construction/import landing
  - repeated-run workflow convergence
  - focused proof and benchmark follow-through
  - docs/examples/header alignment only where the implementation truly moves
    the contract
- The strongest likely Sprint 81 product/storage and proof surfaces are
  explicit from the live tree:
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
- The strongest Day 1 clarification is now fixed:
  - Sprint 81 should not reopen Sprint 80’s baseline, oracle, benchmark, or
    non-goal contract package
  - it should first reduce the linked-list-first product/storage ceiling on the
    highest-value seams only
- The preserved Sprint 81 non-goal pressure is explicit before Day 2:
  - no backend/performance lane spill
  - no capability-surface widening
  - no broad package/platform reopening
  - no generic whole-library workflow rewrite
  - no broad public API redesign hidden inside storage cleanup

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live product/storage, proof-owner, and support-surface hotspot
  map from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 81 no longer starts from generic Epic 8 planning prose.
- The baseline, storage/conversion audit, compressed-first design,
  construction/import landing, workflow-convergence, and focused proof
  follow-through workstreams are fixed in writing.
- The strongest likely Sprint 81 touch surfaces are explicit before the
  validation/proof recheck begins.
