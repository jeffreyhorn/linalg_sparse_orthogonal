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

## Day 2 - Validation and Maintained Consumer-Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live maintained
install/export, downstream-consumer, workflow, and reviewed-surface split
before Sprint 87 changes any package, export, or product-contract surface.

### Actions
- Re-read the Day 2 validation-baseline expectations from
  `docs/planning/EPIC_8/SPRINT_87/PLAN.md`.
- Re-read the strongest recent validation/proof-surface template from
  `docs/planning/EPIC_8/SPRINT_86/artifacts/day2-validation-baseline-and-reviewed-surface-recheck.md`.
- Reconfirmed reviewed CMake parity directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the presence of the strongest reviewed representative binaries and
  examples that remain the main executable truth surfaces entering Sprint 87:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the maintained canonical reporting command surface with:
  - `make -n bench-canonical-report`
- Rechecked the maintained package-proof and downstream-consumer surfaces:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `scripts/bench_canonical_report.sh`
- Re-read the macOS and Windows workflow wording that constrains the current
  package/platform truth:
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

### Findings
- Sprint 87 continues to inherit the strongest local reviewed baseline:
  - `make quality-review-full`
- The code-day and docs-day split is now fixed explicitly for this sprint:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial package, consumer-proof, workflow, or support-surface batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The reviewed CMake tree currently remains the strongest shared executable
  truth surface entering Sprint 87:
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
  - `bash tests/test_install.sh` is the local Unix-side Make
    install/uninstall + `pkg-config` proof
  - `bash tests/test_cmake_install.sh` is the local Unix-side CMake
    install/export + `find_package(Sparse)` proof
  - `examples/cmake_example/CMakeLists.txt` remains the representative
    downstream CMake consumer surface used by that proof lane
- The workflow-side package/platform split remains intentionally narrower than
  a broad cross-platform package parity claim:
  - macOS carries a supplemental static-first Make install/`pkg-config`
    confidence lane only
  - Windows remains the reviewed CMake-first consumer subset and does not
    claim a separate reviewed install-validation lane
- The strongest Day 2 clarification is now fixed:
  - reviewed CMake binaries remain the main executable truth anchor
  - install/export proof remains script owned
  - downstream consumer proof remains local and bounded
  - workflow lanes are support evidence and remain narrower than a broad
    install/export parity claim

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked the presence of the strongest reviewed representative binaries and
  examples.
- Rechecked `make -n bench-canonical-report`.
- Rechecked `scripts/bench_canonical_report.sh`,
  `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
  `examples/cmake_example/CMakeLists.txt`.
- Re-read `.github/workflows/macos-ci.yml` and
  `.github/workflows/windows-ci.yml` for the current bounded package/platform
  truth model.

### Day 2 Exit State
- Sprint 87 now has one explicit validation and maintained-consumer-surface
  contract before the package-gap audit begins.
- The live split across reviewed binaries, command-owned canonical reporting,
  script-owned install/export proof, and narrower workflow-side package
  evidence is fixed in writing.
- The highest-signal rerun set is explicit before the first package /
  consumer contradiction rerank.

## Day 3 - Release / Package Gap Audit

### Goal
Reduce Sprint 87's broad packaging and ABI problem to one ranked live
contradiction map so the sprint can choose one bounded product-contract lane
instead of another generic build or release bucket.

### Actions
- Re-read the Day 3 package-gap expectations from
  `docs/planning/EPIC_8/SPRINT_87/PLAN.md`.
- Re-read the strongest recent rerank template from
  `docs/planning/EPIC_8/SPRINT_86/artifacts/day3-reviewed-runtime-long-pole-audit.md`.
- Re-read the current authoritative package and ABI wording in:
  - `docs/maintainer_guide.md`
  - `README.md`
  - `INSTALL.md`
- Re-read the live build/export implementation surfaces:
  - `CMakeLists.txt`
  - `Makefile`
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`
  - `examples/cmake_example/CMakeLists.txt`
- Re-read the workflow-side package/platform contract surfaces:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Reconciled the package rerank against the Sprint 80 packaging/platform
  direction and the Sprint 86 close handoff.

### Findings
- Sprint 87's broad package / ABI / consumer problem is now reduced to one
  ranked live contradiction map:
  - strongest first target:
    - bounded product-matrix design centered on the static/shared and ABI
      contract currently implemented by `CMakeLists.txt`,
      `cmake/SparseConfig.cmake.in`, `sparse.pc.in`, and the matching package
      wording in `README.md`, `INSTALL.md`, and
      `docs/maintainer_guide.md`
  - strongest second target:
    - bounded consumer-proof expansion centered on
      `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
      `examples/cmake_example/CMakeLists.txt`
  - strongest third target:
    - bounded workflow / platform follow-through centered on
      `.github/workflows/macos-ci.yml` and `.github/workflows/windows-ci.yml`
      after the product contract is explicit
  - strongest support-only but real target:
    - support-surface alignment across `README.md`, `INSTALL.md`,
      `docs/maintainer_guide.md`, and narrow benchmark/docs wording only where
      landed package work truly changes the contract
- The strongest current contradiction is now explicit:
  - the repo's maintained docs repeatedly say the package surface is
    intentionally static-first and not a broad shared-library or dynamic-ABI
    guarantee
  - but the live CMake install/export surface already emits package-version
    metadata through `SparseConfigVersion.cmake` with `SameMajorVersion`
    compatibility semantics
  - and the configure path accepts `BUILD_SHARED_LIBS=ON` only to continue
    producing a static target
  - this means the strongest first Sprint 87 move is not "add shared now"; it
    is to define the exact product matrix the repo is willing to support and
    make the build/export language match it cleanly
- The strongest second contradiction is downstream consumer asymmetry:
  - the local proof story is real on Unix:
    - `tests/test_install.sh` proves Make install/uninstall + `pkg-config`
    - `tests/test_cmake_install.sh` proves CMake install/export +
      `find_package(Sparse)`
  - but the installed surfaces are still asymmetric:
    - Make installs headers, archive, and `sparse.pc`
    - CMake installs headers, archive, exported targets, and package config
  - and the representative downstream consumer example is CMake-only
  - that makes consumer-proof expansion real Sprint 87 work, but still second
    after the product-matrix contract is explicit
- The strongest third contradiction is workflow/platform asymmetry:
  - Linux remains the strongest reviewed source of truth, but its package proof
    stays developer-side rather than a separate reviewed CI lane
  - macOS carries only a narrower supplemental Make install/`pkg-config`
    confidence path
  - Windows keeps the reviewed CMake-first consumer subset and explicitly does
    not claim a separate reviewed install-validation lane
  - that means workflow follow-through is real Sprint 87 work, but it remains
    bounded and must stay behind a truthful product contract
- The strongest support-only follow-through remains bounded:
  - `README.md` = `1050`
  - `INSTALL.md` = `265`
  - `docs/maintainer_guide.md` = `726`
  - `benchmarks/README.md` = `399`
  - these remain support-only unless the landed package work truly changes the
    contract, local proof interpretation, or workflow reading
- The Sprint 80 and Sprint 86 carry-forward reading is now fixed:
  - Sprint 80 already pushed the repo toward a static-first maintained package
    truth rather than an unbounded platform promise
  - Sprint 86 already removed reviewed-runtime as the strongest first-tier Epic
    8 contradiction
  - Sprint 87 therefore begins with package-contract truthfulness, not another
    runtime or generic maintainability lane
- Broad product and ABI widening remains lower-value first work:
  - no broad shared-library product claim without bounded proof
  - no dynamic-ABI promise detached from explicit validation ownership
  - no generic build-system rewrite detached from the chosen product contract
  - no workflow widening that outruns maintained local proof
  - no support-surface churn detached from a real landed packaging seam

### Validation
- Re-read the authoritative package and ABI wording in `README.md`,
  `INSTALL.md`, and `docs/maintainer_guide.md`.
- Re-read the live build/export implementation surfaces in `CMakeLists.txt`,
  `Makefile`, `cmake/SparseConfig.cmake.in`, `sparse.pc.in`, and
  `examples/cmake_example/CMakeLists.txt`.
- Re-read `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, and
  `.github/workflows/windows-ci.yml`.
- Reconciled the package rerank against the validated Sprint 86 handoff and
  Sprint 80 packaging direction.

### Day 3 Exit State
- Sprint 87 now has one ranked live package / ABI / consumer contradiction map
  grounded in the current tree and maintained package contract.
- The first implementation center is fixed to bounded product-matrix design,
  not immediate shared-library widening.
- Later consumer-proof expansion, workflow/platform follow-through, and
  support-surface alignment are explicitly ordered behind that first lane.

## Day 4 - First Packaging and ABI Boundary Freeze

### Goal
Fix the first bounded Sprint 87 packaging / ABI implementation fence so the
next design pass can define one real product-matrix contract instead of
another broad release rewrite.

### Actions
- Re-read the Day 4 boundary expectations from
  `docs/planning/EPIC_8/SPRINT_87/PLAN.md`.
- Re-read the strongest recent Day 4 boundary template from
  `docs/planning/EPIC_8/SPRINT_86/artifacts/day4-first-runtime-scalability-boundary.md`.
- Re-read the Day 3 package-gap rerank in
  `docs/planning/EPIC_8/SPRINT_87/artifacts/day3-release-package-gap-audit.md`.
- Reconciled the required first landing against the current package contract
  implemented by:
  - `CMakeLists.txt`
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
- Re-froze which support surfaces remain allowed only if the first landing
  truly forces them:
  - install/export proof scripts
  - representative downstream-consumer surfaces
  - workflow files
  - support-surface docs

### Findings
- Sprint 87 now has one explicit first implementation fence:
  - required first landing:
    - `CMakeLists.txt`
  - directly forced support surfaces only if the first landing truly needs
    them:
    - `cmake/SparseConfig.cmake.in`
    - `sparse.pc.in`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
  - support-only proof and workflow surfaces that stay later unless the first
    landing truly forces movement:
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
    - `examples/cmake_example/CMakeLists.txt`
    - `.github/workflows/ci.yml`
    - `.github/workflows/macos-ci.yml`
    - `.github/workflows/windows-ci.yml`
  - explicitly deferred from the first landing:
    - consumer-proof expansion as a first-batch center
    - workflow/platform follow-through as a first-batch center
    - broad docs alignment detached from a real package-contract change
    - immediate shared-library product widening
    - broad ABI-compatibility promise widening
    - generic build-system rewrite detached from the chosen product contract
- The strongest Day 4 clarification is now fixed:
  - the best first Sprint 87 move is one bounded product-matrix and
    build/export contract pass centered on `CMakeLists.txt`
  - the first landing should decide how the repo wants its static/shared and
    package-version/export semantics to read before proof scripts or workflow
    widening move
  - `cmake/SparseConfig.cmake.in` and `sparse.pc.in` remain directly allowed
    support surfaces only if that contract truly forces them to move
  - install/export proof, downstream-consumer proof, and workflow surfaces
    remain later work unless the product-contract landing truly changes their
    obligations
- The preserved first-batch non-goal fence is explicit now:
  - no platform claims without maintained proof
  - no broad shared-library product claim without bounded evidence
  - no dynamic-ABI promise detached from explicit validation ownership
  - no generic build-system rewrite detached from the chosen product contract
  - no support-surface churn detached from a real landed packaging seam
  - no workflow widening that outruns maintained local proof

### Validation
- Re-read the Day 3 rerank and the authoritative package/build surfaces.
- Reconfirmed the first-vs-later touch split against the Sprint 87 plan and
  the current maintained package contract.

### Day 4 Exit State
- Sprint 87 now has one bounded first packaging/ABI landing center.
- Day 5 can design one explicit product-matrix contract inside that fence.
- Later consumer-proof expansion, workflow/platform follow-through, and broad
  support-surface alignment are held back until later lanes.

## Day 5 - Product-Matrix Design

### Goal
Define the bounded static/shared, ABI, and downstream-consumer contract that
Sprint 87 will actually support on its first packaging lane.

### Actions
- Re-read the Day 5 design expectations from
  `docs/planning/EPIC_8/SPRINT_87/PLAN.md`.
- Re-read the strongest recent Day 5 design template from
  `docs/planning/EPIC_8/SPRINT_86/artifacts/day5-algorithm-proof-runtime-architecture-design.md`.
- Re-read the Day 4 boundary in
  `docs/planning/EPIC_8/SPRINT_87/artifacts/day4-first-packaging-abi-boundary.md`.
- Re-read the current version/export implementation details in:
  - `CMakeLists.txt`
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`
  - `VERSION`
- Reconciled the product decision against the maintained package wording in:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`

### Findings
- Sprint 87 now has one explicit first implementation contract:
  - required implementation center:
    - `CMakeLists.txt`
  - directly forced support surfaces only if the first batch truly needs them:
    - `cmake/SparseConfig.cmake.in`
    - `sparse.pc.in`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
  - consumer-proof and workflow surfaces remain later owners unless the first
    batch truly changes their obligations:
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
    - `examples/cmake_example/CMakeLists.txt`
    - `.github/workflows/ci.yml`
    - `.github/workflows/macos-ci.yml`
    - `.github/workflows/windows-ci.yml`
- The Day 5 ownership split is now fixed:
  - product-matrix contract owner:
    - `CMakeLists.txt`
  - retained CMake export/config owner if the product contract truly changes
    installed package semantics:
    - `cmake/SparseConfig.cmake.in`
  - retained pkg-config contract owner if the product contract truly changes
    Make-installed consumer semantics:
    - `sparse.pc.in`
  - retained local install/export proof owners after the first landing:
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
  - retained downstream-consumer proof owner after the first landing:
    - `examples/cmake_example/CMakeLists.txt`
  - retained workflow/platform evidence owners after the first landing:
    - `.github/workflows/ci.yml`
    - `.github/workflows/macos-ci.yml`
    - `.github/workflows/windows-ci.yml`
  - support-surface wording owners only if implementation truly changes the
    package contract reading:
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
- The strongest product decision is now explicit:
  - Sprint 87 remains static-first only
  - it does not open a bounded shared-library product lane in this sprint
  - the maintained contract should instead become sharper about:
    - static archive output as the only shipped product shape
    - `pkg-config` and `find_package(Sparse)` describing that same static
      package surface
    - version metadata being real package metadata rather than a broad
      dynamic-ABI guarantee
    - platform truth remaining narrower on macOS and Windows than on Linux
- The useful Day 5 clarification is explicit now:
  - Day 6 should not try to "add shared"
  - it should tighten the live build/export semantics so they read exactly
    like the maintained static-first product contract
  - it should preserve downstream proof ownership and workflow evidence as
    later lanes rather than folding them into the first packaging batch
  - it should treat package-version/export metadata as part of the contract
    language problem, not as evidence for a broader ABI promise
- The preserved first-batch fence is explicit:
  - no shared-library product claim without bounded proof
  - no dynamic-ABI promise detached from explicit validation ownership
  - no workflow widening folded into the first batch unless the product
    contract truly forces it
  - no support-surface churn detached from the landed package seam
  - no generic build-system rewrite detached from the chosen product contract

### Validation
- Re-read the Day 4 boundary and the current version/export implementation
  details in `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, `sparse.pc.in`,
  and `VERSION`.
- Reconciled the product decision against the maintained package wording in
  `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md`.

### Day 5 Exit State
- Sprint 87 now has one bounded static-first product-matrix contract.
- Ownership between the first package/build lane, retained consumer-proof
  owners, retained workflow evidence, and later support-surface alignment is
  fixed before Day 6 begins.
- Consumer-proof expansion, workflow/platform follow-through, and broader
  docs alignment remain explicitly outside the first packaging batch.

## Day 6 - Packaging Batch

### Goal
Land one bounded packaging/export modernization batch that makes the live build
and install surface read exactly like Sprint 87's maintained static-first
product contract.

### Actions
- Landed the first package-contract change in `CMakeLists.txt` by tightening
  the generated CMake package version semantics from
  `SameMajorVersion` to `ExactVersion`.
- Added an inline contract comment at the `write_basic_package_version_file`
  callsite so the static-first/no-broad-ABI rationale stays attached to the
  actual generated package-version surface.
- Updated directly forced support-surface wording only where the new contract
  would otherwise remain implicit:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
- Extended the retained CMake install/export proof owner
  `tests/test_cmake_install.sh` so it now validates:
  - `find_package(Sparse ${EXPECTED_VERSION} EXACT REQUIRED)` succeeds
  - `find_package(Sparse ${MISMATCH_VERSION} REQUIRED)` is rejected
- Re-ran the maintained package-proof surfaces:
  - `bash tests/test_cmake_install.sh`
  - `bash tests/test_install.sh`

### Findings
- Sprint 87's first implementation landing stayed inside the Day 5 fence:
  - required implementation center:
    - `CMakeLists.txt`
  - directly forced support follow-through actually needed:
    - `tests/test_cmake_install.sh`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
  - not needed in the batch:
    - `cmake/SparseConfig.cmake.in`
    - `sparse.pc.in`
    - `tests/test_install.sh` logic changes
    - `examples/cmake_example/CMakeLists.txt`
    - `.github/workflows/ci.yml`
    - `.github/workflows/macos-ci.yml`
    - `.github/workflows/windows-ci.yml`
- The kept packaging win is explicit:
  - the generated `SparseConfigVersion.cmake` no longer advertises
    same-major-version compatibility
  - the installed CMake package now matches the maintained docs reading:
    exact package-version identity is real package metadata, not a broad
    dynamic-ABI compatibility promise
- The directly forced proof follow-through was limited and product-owned:
  - `tests/test_cmake_install.sh` now proves the exact-version CMake consumer
    contract end to end
  - `tests/test_install.sh` remained valid unchanged because the Make/pkg-config
    side of the static-first contract did not change
- The strongest Day 6 clarification is now explicit:
  - the first Sprint 87 packaging win does not require opening a shared lane
  - it comes from making the existing static-first install/export semantics
    stricter and more truthful
  - downstream-consumer expansion and workflow follow-through remain later
    lanes, not part of the first batch

### Validation
- `bash tests/test_cmake_install.sh` passed
  - including exact-version success and mismatched-version rejection
- `bash tests/test_install.sh` passed
- Because no `*.c` or `*.h` files changed, `make format`, `make lint`, and
  `make test` were not required for this batch.

### Day 6 Exit State
- Sprint 87 now has one landed bounded packaging/export batch.
- The live CMake package-version semantics now match the maintained static-first
  and no-broad-ABI contract.
- Later Sprint 87 work remains centered on consumer-proof expansion,
  workflow/platform follow-through, and broader support-surface alignment.
