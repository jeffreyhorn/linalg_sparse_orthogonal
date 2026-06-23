# Sprint 86 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 86 baseline for Epic 8 by grounding the sprint in
the validated Sprint 85 close state, the live Sprint 86 project-plan section,
and the current reviewed-runtime, reorder, nested-dissection, benchmark, and
support-surface hotspots rather than another generic “optimize tests” reset.

### Actions
- Re-read the Sprint 86 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 86 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_86/PLAN.md`.
- Re-read the strongest Sprint 85 closeout context:
  - `docs/planning/EPIC_8/SPRINT_85/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_85/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 86
  touch surfaces across reorder proof owners, reorder/graph implementation
  owners, benchmark surfaces, and support surfaces.
- Opened Sprint 86 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 86 starts from the same strongest local reviewed baseline Sprint 85
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 86 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 86 is not a generic “speed up tests” sprint. Its highest value is one
  bounded reviewed-runtime and reordering-scalability package centered on:
  - reviewed runtime audit
  - algorithm / proof runtime design
  - ND runtime reduction
  - proof-surface rebalancing
  - benchmark / comparison follow-through
  - CI / reviewed-path alignment
  - validation and closeout
- The validated Sprint 85 close baseline already fixed the strongest runtime
  contradiction entering Sprint 86:
  - reviewed CMake `Total Test time (real)` = `404.15 sec`
  - reviewed `test_reorder_nd` time = `283.53 sec`
  - the strongest runtime long pole is therefore concentrated on the reorder /
    ND reviewed proof lane, not on a generic whole-suite slowdown
- The strongest likely Sprint 86 implementation, proof, and support surfaces
  are explicit from the live tree:
  - strongest reviewed proof and runtime owner:
    - `tests/test_reorder_nd.c` = `2287`
  - adjacent reorder proof owners:
    - `tests/test_reorder.c` = `1082`
    - `tests/test_reorder_amd_qg.c` = `273`
  - strongest reorder and ND implementation owners:
    - `src/sparse_graph.c` = `841`
    - `src/sparse_reorder_nd.c` = `757`
    - `src/sparse_graph_coarsen.c` = `659`
    - `src/sparse_reorder_amd_qg.c` = `611`
    - `src/sparse_graph_refine.c` = `602`
    - `src/sparse_graph_bisect.c` = `528`
    - `src/sparse_reorder.c` = `419`
    - `src/sparse_graph_separator.c` = `297`
  - strongest measurement and support surfaces:
    - `benchmarks/bench_reorder.c` = `321`
    - `benchmarks/bench_fillin.c` = `178`
    - `README.md` = `1050`
    - `docs/maintainer_guide.md` = `726`
- The strongest Day 1 clarification is now fixed:
  - Sprint 86 should not reopen Sprint 85’s source-decomposition package as
    its first implementation center
  - Sprint 86 should first reduce reviewed runtime concentration and improve
    reorder / ND scalability on the strongest current long pole
  - it should preserve correctness-proof quality while deciding how much of
    the fix is algorithmic, fixture-organization, or reviewed-path
    architecture
- The preserved Sprint 86 non-goal pressure is explicit before Day 2:
  - no generic maintainability decomposition restart
  - no weakening of correctness proof quality to buy runtime wins
  - no benchmark-governance or example-governance drift into correctness
    ownership
  - no package/platform maturity claim widening
  - no support-surface churn detached from a real landed runtime seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Carried forward the validated Sprint 85 close runtime anchors:
  - reviewed CMake `Total Test time (real)` = `404.15 sec`
  - reviewed `test_reorder_nd` time = `283.53 sec`
- Captured the live reorder, ND, benchmark, and support-surface hotspot map
  from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 86 no longer starts from generic Epic 8 runtime prose.
- The reviewed runtime audit, algorithm/proof design, ND runtime reduction,
  proof-surface rebalancing, benchmark follow-through, CI alignment, and
  validation workstreams are fixed in writing.
- The strongest likely Sprint 86 touch surfaces are explicit before the
  validation/proof recheck begins.
