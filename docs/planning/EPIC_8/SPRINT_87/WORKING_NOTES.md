# Sprint 87 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 87 baseline for Epic 8 by grounding the sprint in
the validated Sprint 86 close state, the live Sprint 87 project-plan section,
and the current install/export, ABI, workflow, and support-surface hotspots
rather than another generic "improve packaging" reset.

### Actions
- Re-read the Sprint 87 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 87 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_87/PLAN.md`.
- Re-read the strongest Sprint 86 closeout context:
  - `docs/planning/EPIC_8/SPRINT_86/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_86/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly through the Day 1 parity
  rebuild:
  - `ctest -N --test-dir build/quality-review-cmake`
- Re-read the strongest current package-contract wording and maintained proof
  owners in:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 87
  touch surfaces across build/package owners, export/config files, proof
  scripts, workflow surfaces, and support docs.
- Opened Sprint 87 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 87 starts from the same strongest local reviewed baseline Sprint 86
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 87 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 87 is not a generic "improve packaging" sprint. Its highest value is
  one bounded package / ABI / consumer modernization package centered on:
  - release / package gap audit
  - product-matrix design
  - packaging batch
  - consumer-proof expansion
  - workflow / platform follow-through
  - support-surface alignment
  - validation and closeout
- The validated Sprint 86 close state already fixed the strongest handoff
  truth entering Sprint 87:
  - the repo now has a materially smaller reviewed-runtime long pole
  - the next first-tier Epic 8 contradiction is package / install-export /
    consumer truthfulness rather than reviewed runtime concentration
- The strongest current maintained package contract is narrower and more
  explicit than a broad shared-library product claim:
  - the shipped install/export surface is real and maintained
  - the maintained release shape remains static-first
  - `pkg-config` and `find_package(Sparse)` both describe the same installed
    static archive surface
  - version metadata remains single-sourced from `VERSION`
  - current packaging wording explicitly does not claim a broad shared-library
    or dynamic-ABI guarantee
- The strongest likely Sprint 87 implementation, proof, and support surfaces
  are explicit from the live tree:
  - strongest build/package owners:
    - `README.md` = `1050`
    - `Makefile` = `908`
    - `docs/maintainer_guide.md` = `726`
    - `CMakeLists.txt` = `413`
    - `benchmarks/README.md` = `399`
    - `INSTALL.md` = `265`
  - strongest maintained proof and workflow surfaces:
    - `.github/workflows/ci.yml` = `223`
    - `tests/test_install.sh` = `172`
    - `tests/test_cmake_install.sh` = `146`
    - `.github/workflows/macos-ci.yml` = `117`
    - `.github/workflows/windows-ci.yml` = `63`
  - strongest export/config and downstream-consumer surfaces:
    - `sparse.pc.in` = `10`
    - `examples/cmake_example/CMakeLists.txt` = `10`
    - `cmake/SparseConfig.cmake.in` = `5`
- The strongest Day 1 clarification is now fixed:
  - Sprint 87 should not begin by promising a broad shared-library or ABI
    surface the repo does not yet maintain
  - Sprint 87 should first re-rank package and consumer contradictions, then
    define whether the repo remains permanently static-first or earns one
    bounded shared lane
  - any widened package or platform claim must stay tied to maintained local
    proof and realistic workflow ownership
- The preserved Sprint 87 non-goal pressure is explicit before Day 2:
  - no broad shared-library product claim without bounded proof
  - no generic build-system rewrite detached from a chosen product contract
  - no broad ABI-compatibility promise without real validation ownership
  - no cross-platform widening that outruns maintained workflow evidence
  - no support-surface churn detached from a real landed packaging seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Re-read the maintained package contract wording in `README.md`,
  `INSTALL.md`, and `docs/maintainer_guide.md`.
- Re-read the maintained local install/export proof owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- Captured the live build/package/workflow hotspot map from direct `wc -l`
  measurement.

### Day 1 Exit State
- Sprint 87 no longer starts from generic Epic 8 packaging prose.
- The package-gap audit, product-matrix design, packaging batch,
  consumer-proof expansion, workflow/platform follow-through,
  support-surface alignment, and validation workstreams are fixed in writing.
- The strongest likely Sprint 87 touch surfaces and preserved non-goals are
  explicit before the validation / maintained-surface recheck begins.
