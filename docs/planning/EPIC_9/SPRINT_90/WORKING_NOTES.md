# Sprint 90 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 90 baseline for Epic 9 by grounding the sprint in
the validated Epic 8 close state, the live Sprint 90 project-plan section, and
the strongest current product-model, comparison, support-surface, build, and
hotspot signals rather than another generic "new epic" reset.

### Actions
- Re-read the Sprint 90 section of `docs/planning/EPIC_9/PROJECT_PLAN.md` and
  the full Sprint 90 day-by-day plan in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the strongest Epic 8 closeout context:
  - `docs/planning/EPIC_8/EPIC_8_RETROSPECTIVE.md`
  - `docs/planning/EPIC_8/SPRINT_89/artifacts/day14-closeout-and-handoff.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor from the Day 1 parity rebuild:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw line-count hotspot map for the strongest likely Sprint
  90 touch surfaces across the Epic 9 planning surfaces, prior Epic 8 review
  inputs, support/build/workflow owners, and the highest-value residual code
  and proof hotspots that the Epic 9 review is likely to discuss.
- Opened Sprint 90 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 90 starts from the same strongest local reviewed baseline Epic 8
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 90 review or
  planning work widens:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 90 is not a generic "start the next epic" sprint. Its highest value
  is one bounded Epic 9 baseline package centered on:
  - baseline recheck
  - target-state freeze
  - product-model audit
  - comparison/measurement contract
  - non-goal and risk fence
  - review/todo/project-plan package
- The validated Epic 8 close state already fixed the strongest handoff truth
  entering Sprint 90:
  - the project now has a real bounded external SPD comparison lane
  - the package/install/export contract is sharper and explicitly static-first
  - the front-door adoption/support split is materially cleaner
  - the residual queue is smaller and better calibrated than at Epic 8 start
- The strongest current contradiction entering Epic 9 is no longer basic
  engineering hygiene. It is the structural product gap between:
  - a highly disciplined and well-proven sparse library
  - and a truly state-of-the-art sparse numerical product
- The strongest likely Sprint 90 review and planning surfaces are explicit
  from the live tree:
  - Epic 9 planning and prior-epic interpretation owners:
    - `docs/planning/EPIC_9/PROJECT_PLAN.md` = `350`
    - `docs/planning/EPIC_8/EPIC_8_RETROSPECTIVE.md` = `292`
    - `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md` = `501`
    - `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md` = `361`
  - strongest support, package, and workflow owners:
    - `README.md` = `1113`
    - `INSTALL.md` = `315`
    - `docs/maintainer_guide.md` = `727`
    - `Makefile` = `908`
    - `CMakeLists.txt` = `416`
    - `.github/workflows/ci.yml` = `223`
    - `.github/workflows/macos-ci.yml` = `104`
    - `.github/workflows/windows-ci.yml` = `63`
    - `tests/test_install.sh` = `195`
    - `tests/test_cmake_install.sh` = `208`
    - `scripts/bench_canonical_report.sh` = `101`
  - strongest residual implementation and proof hotspots likely to matter in
    the Epic 9 review:
    - `src/sparse_matrix.c` = `1297`
    - `src/sparse_dense.c` = `862`
    - `src/sparse_iterative.c` = `1854`
    - `src/sparse_ldlt_csc.c` = `2694`
    - `tests/test_reorder_nd.c` = `2340`
    - `tests/test_graph.c` = `2925`
    - `tests/test_chol_csc.c` = `4987`
    - `tests/test_ldlt_csc.c` = `3680`
- The strongest Day 1 clarification is now fixed:
  - Sprint 90 should begin with one evidence-first baseline and target audit
    rather than with assumptions inherited loosely from Epic 8
  - the Epic 9 planning package should be grounded in live post-Epic-8 truth
    surfaces, not just in prior planning prose
  - anti-sprawl and anti-overclaim work belongs at the start of the epic, not
    later after implementation has already widened
- The preserved Sprint 90 non-goal pressure is explicit before Day 2:
  - no generic "review everything" prose detached from repo evidence
  - no speculative implementation batch before the target state and ranking are
    fixed
  - no fake platform, package, or capability broadening in the review wording
  - no benchmark or workflow reinterpretation that outruns the maintained
    proof-owner surfaces
  - no Epic 9 summary writing before the review, todo, and project-plan
    package is evidence-backed

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live planning/support/hotspot map from direct line-count
  measurement.

### Day 1 Exit State
- Sprint 90 no longer starts from generic Epic 9 ambition prose.
- The baseline recheck, target-state freeze, product-model audit,
  comparison/measurement contract, non-goal/risk fence, and review/todo/plan
  workstreams are fixed in writing.
- The strongest likely touch surfaces and preserved non-goals are explicit
  before the validation and maintained-surface recheck begins.
