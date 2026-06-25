# Sprint 89 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 89 baseline for Epic 8 by grounding the sprint in
the validated Sprint 88 close state, the live Sprint 89 project-plan section,
and the current end-state review, comparison, proof, reporting, and closeout
hotspots rather than another generic "final polish" reset.

### Actions
- Re-read the Sprint 89 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 89 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_89/PLAN.md`.
- Re-read the strongest Sprint 88 closeout context:
  - `docs/planning/EPIC_8/SPRINT_88/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_88/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly through the Day 1 parity
  rebuild:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw line-count hotspot map for the strongest likely Sprint
  89 touch surfaces across planning/closeout docs, maintained install/export
  proof, benchmark/reporting surfaces, workflows, and the highest-value
  reviewed runtime and graph/reorder owners still likely to matter in a final
  fix batch.
- Opened Sprint 89 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 89 starts from the same strongest local reviewed baseline Sprint 88
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 89 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 89 is not a generic "close things out" sprint. Its highest value is
  one bounded final-integration and Epic 8 closeout package centered on:
  - end-state re-audit
  - external comparison sweep
  - final cross-surface fix batch
  - full validation and reporting sweep
  - residual queue finalization
  - retrospective, handoff, and final project-summary closeout
- The validated Sprint 88 close state already fixed the strongest handoff
  truth entering Sprint 89:
  - the front-door adoption path is clearer
  - example and install guidance now has a cleaner audience split
  - the static-first package and consumer contract remain bounded and
    truthful
  - Epic 8 now has one explicit final sprint queue instead of a broad
    residual bucket
- The strongest current end-state contradiction is no longer a single product,
  runtime, or usability seam. It is the final evidence problem:
  - the project needs one fresh live re-audit against the original Epic 8
    concerns
  - it needs one bounded external comparison package rather than only
    internally generated proof
  - it needs one final calibrated residual queue that distinguishes real carry
    forward work from deliberate non-claims
- The strongest likely Sprint 89 implementation, proof, reporting, and
  closeout surfaces are explicit from the live tree:
  - planning and closeout owners:
    - `docs/planning/EPIC_8/PROJECT_PLAN.md` = `351`
    - `docs/planning/EPIC_8/SPRINT_88/RETROSPECTIVE.md` = `267`
    - `docs/planning/EPIC_8/SPRINT_88/artifacts/day14-closeout-and-handoff.md`
      = `73`
  - strongest support, install/export, and reporting owners:
    - `README.md` = `1113`
    - `INSTALL.md` = `315`
    - `docs/maintainer_guide.md` = `727`
    - `tests/test_install.sh` = `195`
    - `tests/test_cmake_install.sh` = `208`
    - `benchmarks/README.md` = `399`
    - `scripts/bench_canonical_report.sh` = `101`
  - strongest workflow/build/package evidence surfaces:
    - `.github/workflows/ci.yml` = `223`
    - `.github/workflows/macos-ci.yml` = `104`
    - `.github/workflows/windows-ci.yml` = `63`
    - `CMakeLists.txt` = `416`
    - `Makefile` = `908`
  - strongest reviewed-runtime and reorder/graph proof surfaces still likely
    to matter if the re-audit forces a final batch:
    - `tests/test_reorder_nd.c` = `2340`
    - `tests/test_graph.c` = `2925`
    - `tests/test_reorder.c` = `1082`
    - `tests/test_reorder_amd_qg.c` = `273`
    - `benchmarks/bench_reorder.c` = `338`
    - `benchmarks/bench_fillin.c` = `178`
- The strongest Day 1 clarification is now fixed:
  - Sprint 89 should begin with one evidence-first end-state review rather
    than with a speculative fix batch
  - external comparison belongs before any last-mile implementation widening
  - final closeout writing should come only after the strongest reviewed,
    install/export, and reporting anchors are refreshed from the live tree
- The preserved Sprint 89 non-goal pressure is explicit before Day 2:
  - no broad reopening of earlier sprint scope
  - no speculative optimization or capability widening without fresh evidence
  - no support-surface churn detached from a live end-state contradiction
  - no benchmark or workflow rewriting that outclaims the maintained proof
    owners
  - no Epic 8 summary writing before the final validation/reporting baseline
    is rebuilt

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live final-integration / proof / reporting hotspot map from
  direct line-count measurement.

### Day 1 Exit State
- Sprint 89 no longer starts from generic Epic 8 closeout prose.
- The end-state re-audit, external comparison sweep, final cross-surface fix
  batch, validation/reporting, residual-queue calibration, and closeout
  workstreams are fixed in writing.
- The strongest likely Sprint 89 touch surfaces and preserved non-goals are
  explicit before the validation and maintained cross-surface recheck begins.

## Day 2 - Validation and Maintained Cross-Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live maintained
reviewed, install/export, benchmark-reporting, example, and workflow truth
split before Sprint 89 changes any final-integration, comparison, or closeout
surface.

### Actions
- Re-read the Day 2 validation-baseline expectations from
  `docs/planning/EPIC_8/SPRINT_89/PLAN.md`.
- Re-read the strongest recent validation/surface templates from:
  - `docs/planning/EPIC_8/SPRINT_88/artifacts/day2-validation-baseline-and-maintained-support-surface-recheck.md`
  - `docs/planning/EPIC_8/SPRINT_87/artifacts/day2-validation-baseline-and-maintained-consumer-surface-recheck.md`
- Reconfirmed reviewed CMake parity directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the presence of the strongest reviewed representative binaries and
  examples that remain the main executable truth surfaces entering Sprint 89:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the maintained canonical reporting command surface with:
  - `make -n bench-canonical-report`
- Rechecked the maintained reporting and consumer-proof owners:
  - `scripts/bench_canonical_report.sh`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
- Re-read the CI, macOS, and Windows workflow surfaces that constrain the
  current reviewed, supplemental, and staged platform truth:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

### Findings
- Sprint 89 continues to inherit the strongest local reviewed baseline:
  - `make quality-review-full`
- The code-day and docs-day split is now fixed explicitly for this sprint:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial final-integration, comparison, residual-calibration, or
    closeout-support batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The reviewed CMake tree currently remains the strongest shared executable
  truth surface entering Sprint 89:
  - reviewed representative proof owners:
    - `./build/quality-review-cmake/test_reorder_nd`
    - `./build/quality-review-cmake/test_reorder`
    - `./build/quality-review-cmake/test_reorder_amd_qg`
    - `./build/quality-review-cmake/test_graph`
  - representative examples:
    - `./build/quality-review-cmake/example_analysis`
    - `./build/quality-review-cmake/example_basic_solve`
- Canonical benchmark reporting remains command- and script-owned rather than
  reviewed-binary-owned:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - root `build/` canonical emitters:
    - `build/bench_refactor_csc`
    - `build/bench_chol_csc`
    - `build/bench_iterative_reuse`
    - `build/bench_eigs_reuse`
- Maintained install/export proof remains script- and fixture-owned:
  - `bash tests/test_install.sh` proves the local Unix-side Make
    install/uninstall + `pkg-config` path
  - `bash tests/test_cmake_install.sh` proves the local Unix-side CMake
    install/export + `find_package(Sparse)` path
  - `examples/cmake_example/CMakeLists.txt` remains the representative
    downstream CMake consumer surface used by the CMake install/export proof
- Workflow-side truth remains intentionally layered rather than flattened into
  one broad parity claim:
  - Linux remains the strongest reviewed source of truth through the enforced
    reviewed Makefile, reviewed CMake, and dead-code lanes
  - macOS carries a narrower enforced reviewed Apple Clang lane plus a
    supplemental static-first Make install/`pkg-config` confidence lane
  - Windows remains the reviewed CMake-first consumer subset and does not
    claim a reviewed Makefile or separate reviewed install-validation lane
  - Windows still fixes its reviewed `ctest -N` expectation at `50` and keeps
    staged exclusions explicit in job output
- The strongest Day 2 clarification is now fixed:
  - reviewed CMake binaries remain the main executable truth anchor
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned
  - downstream consumer proof remains local and bounded
  - workflow lanes remain support evidence rather than broad cross-platform
    parity claims

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked the presence of the strongest reviewed representative binaries and
  examples.
- Rechecked `make -n bench-canonical-report`.
- Rechecked `scripts/bench_canonical_report.sh`,
  `tests/test_install.sh`, `tests/test_cmake_install.sh`,
  `examples/cmake_example/CMakeLists.txt`, and the CI/macOS/Windows workflow
  surfaces.

### Day 2 Exit State
- Sprint 89 now has one explicit validation and maintained cross-surface
  contract before the end-state re-audit begins.
- Reviewed CMake binaries remain the main executable truth anchor.
- Canonical benchmark reporting remains command/script owned.
- Install/export proof remains script owned.
- Workflow lanes remain support evidence rather than broad parity claims.
