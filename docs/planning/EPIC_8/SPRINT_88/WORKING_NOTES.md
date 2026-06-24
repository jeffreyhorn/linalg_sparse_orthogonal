# Sprint 88 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 88 baseline for Epic 8 by grounding the sprint in
the validated Sprint 87 close state, the live Sprint 88 project-plan section,
and the current front-door, example, install, benchmark-reference, and
public-narrative hotspots rather than another generic "improve docs" reset.

### Actions
- Re-read the Sprint 88 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 88 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_88/PLAN.md`.
- Re-read the strongest Sprint 87 closeout context:
  - `docs/planning/EPIC_8/SPRINT_87/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_87/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly through the Day 1 parity
  rebuild:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 88
  touch surfaces across front-door docs, support references, example surfaces,
  maintained proof scripts, workflows, and highest-signal public headers.
- Opened Sprint 88 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 88 starts from the same strongest local reviewed baseline Sprint 87
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 88 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 88 is not a generic "improve docs" sprint. Its highest value is one
  bounded front-door usability and workflow-simplification package centered
  on:
  - user-journey audit
  - workflow-simplification design
  - README / tutorial batch
  - examples / workflow batch
  - support-surface consolidation
  - header / API narrative cleanup
  - validation and closeout
- The validated Sprint 87 close state already fixed the strongest handoff
  truth entering Sprint 88:
  - the repo now has a sharper static-first package/export contract
  - the maintained local consumer proof is stronger
  - the next first-tier Epic 8 contradiction is front-door usability and
    audience-boundary clarity rather than install/export truthfulness
- The strongest current maintained front-door contract is more truthful than
  earlier Epic 8 phases, but it still leaves adoption friction:
  - README, install, example references, and support surfaces still carry too
    much policy density for first adoption
  - example, benchmark, install, and maintainer references still need a
    clearer audience split
  - some highest-signal public headers still leak more internal policy than an
    adoption-focused public narrative needs
- The strongest likely Sprint 88 implementation, proof, and support surfaces
  are explicit from the live tree:
  - strongest front-door and support owners:
    - `README.md` = `1051`
    - `docs/maintainer_guide.md` = `727`
    - `INSTALL.md` = `266`
    - `benchmarks/README.md` = `399`
    - `CMakeLists.txt` = `416`
    - `Makefile` = `908`
  - strongest maintained proof and workflow surfaces:
    - `tests/test_cmake_install.sh` = `208`
    - `tests/test_install.sh` = `195`
    - `.github/workflows/ci.yml` = `223`
    - `.github/workflows/macos-ci.yml` = `104`
    - `.github/workflows/windows-ci.yml` = `63`
  - strongest example and high-signal public narrative surfaces:
    - `examples/cmake_example/main.c` = `50`
    - `examples/cmake_example/CMakeLists.txt` = `10`
    - `include/sparse_iterative.h` = `773`
    - `include/sparse_eigs.h` = `651`
    - `include/sparse_matrix.h` = `622`
    - `include/sparse_types.h` = `313`
- The strongest Day 1 clarification is now fixed:
  - Sprint 88 should not reopen package/platform semantics already stabilized
    in Sprint 87
  - Sprint 88 should first re-rank adoption-friction contradictions, then
    define one explicit front-door and support-layering contract
  - any widened usability wording must stay tied to maintained proof,
    realistic audience boundaries, and the preserved correctness contract
- The preserved Sprint 88 non-goal pressure is explicit before Day 2:
  - no package/platform contract rewrite
  - no correctness-ownership redistribution
  - no benchmark-policy rewrite detached from adoption guidance
  - no internal architectural rewrite disguised as usability cleanup
  - no support-surface churn detached from a real landed front-door seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live front-door / support / public-header hotspot map from
  direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 88 no longer starts from generic Epic 8 usability prose.
- The user-journey audit, workflow-simplification design, README/tutorial
  batch, examples/workflow batch, support-surface consolidation,
  header/API-narrative cleanup, and validation workstreams are fixed in
  writing.
- The strongest likely Sprint 88 touch surfaces and preserved non-goals are
  explicit before the validation / maintained-surface recheck begins.
