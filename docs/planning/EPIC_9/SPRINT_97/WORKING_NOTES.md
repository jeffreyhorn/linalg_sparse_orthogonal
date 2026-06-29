# Sprint 97 Working Notes

## Day 1 - Sprint 97 Scope & Build Topology Baseline

### Goal

Open Sprint 97 with a current build, package, workflow, and platform proof
baseline from the merged Sprint 96 tree. Day 1 does not change build topology.
It identifies the surfaces and validation rules that Day 2 can audit and rank.

### Actions

- Re-read the Sprint 97 section of
  `docs/planning/EPIC_9/PROJECT_PLAN.md`.
- Re-read the Sprint 97 plan Day 1 scope.
- Re-read the prior package/platform inputs from Sprints 90, 92, 95, and 96
  at the planning-artifact level.
- Inventoried current build and workflow topology:
  - `Makefile`
  - `CMakeLists.txt`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - README and install package-claim surfaces
- Measured current file-family counts from the live tree:
  - 42 library source files under `src/*.c`
  - 18 internal headers under `src/*.h`
  - 54 test sources under `tests/test_*.c`
  - 16 benchmark sources under `benchmarks/*.c`
  - 12 example sources under `examples/*.c`
  - 18 public headers under `include/*.h`
- Identified initial duplication candidates across Make, CMake, workflows,
  install/export proof, and platform claim surfaces.
- Recorded authoritative inputs in
  `artifacts/day1-authoritative-inputs.txt`.
- Recorded the Day 1 build-topology baseline in
  `artifacts/day1-build-topology-baseline.md`.

### Findings

- Make and CMake both carry explicit library source lists, test registration
  lists, benchmark lists, and example lists. The library and test lists are the
  most important Day 2 duplication candidates because they gate correctness and
  CI proof.
- `CMakeLists.txt`, `README.md`, and `INSTALL.md` currently preserve the
  static-first package story. `BUILD_SHARED_LIBS=ON` is explicitly treated as
  not earning a shared-library package contract.
- Windows CI currently enforces the reviewed CMake-first consumer subset with
  an expected CTest count of 51 and explicit exclusions for pthread/fuzz lanes.
- macOS CI currently has an enforced Apple Clang reviewed path, a supplemental
  Homebrew GCC path, and a supplemental Make install/pkg-config proof job.
- `tests/test_install.sh` validates Make install plus pkg-config consumers.
  `tests/test_cmake_install.sh` validates CMake install, `find_package`, export
  files, and version compatibility.
- Day 2 should distinguish harmful duplication from intentionally independent
  proof. Some repeated CI messages and expected counts are proof assertions,
  not necessarily reduction targets.

### Validation

- Day 1 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, or script files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only baseline pass.
- Follow-up hygiene checks should include `git diff --check` and a
  trailing-whitespace scan on Sprint 97 planning files.

### Day 1 Exit State

Sprint 97 now has a current build-topology baseline, authoritative inputs, a
starting duplication candidate list, and validation expectations for later
docs, build, workflow, CMake, and code-touch days. Day 2 can rank duplication
from current evidence instead of treating all repeated build data as equally
harmful.

## Day 2 - Build-Topology Duplication Audit

### Goal

Rank the highest-cost duplication between Make, CMake, CI, and consumer proof
surfaces while preserving intentionally independent proof.

### Actions

- Re-read the Day 1 build-topology baseline and Day 2 plan.
- Mechanically compared Make and CMake registrations for:
  - library sources
  - test executables
  - benchmark executables
  - example executables
- Compared Linux, macOS, and Windows workflows against local reviewed Make and
  CMake validation gates.
- Compared install/export proof scripts with CMake install/export rules and
  package-claim docs.
- Separated duplicated build topology from product-story repetition and
  platform-specific proof assertions.
- Recorded the ranked Day 2 audit in
  `artifacts/day2-build-topology-duplication-audit.md`.

### Findings

- Make/CMake source, test, benchmark, and example registrations are currently
  in parity.
- The highest-cost duplication is still the library source list: 42 source
  files are repeated in `Makefile` and `CMakeLists.txt`, and every new source
  split must touch both surfaces.
- The second-highest duplication is the test registration list: 54 tests are
  repeated across Make and CMake, with additional platform-specific exclusions
  in CMake and Windows CI.
- Benchmark registration is lower risk than library/test registration, but it
  still duplicates 16 benchmark programs across Make and CMake.
- Example registration is a lower-cost candidate because Make uses a wildcard
  while CMake explicitly registers 12 examples.
- Windows expected CTest count, staged exclusions, and workflow messages are
  proof assertions. They create maintenance cost but should not be removed
  unless Day 3 designs an equivalent assertion.
- Static-first package language appears in CMake, README, INSTALL, workflows,
  and install proof scripts. This is product-story consistency work, not the
  same problem as source-list duplication.

### Validation

- Day 2 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, or script files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only audit pass.
- Required hygiene checks: `git diff --check` and a trailing-whitespace scan on
  Sprint 97 planning files.

### Day 2 Exit State

Sprint 97 now has an authoritative duplication ranking. Day 3 should design a
bounded convergence path around the library source list first, with test
registration as the second candidate only if proof assertions and platform
exclusions remain explicit.

## Day 3 - Convergence Architecture Design

### Goal

Define a bounded convergence architecture for the highest-value duplicated
build topology before editing Make, CMake, scripts, or workflows.

### Actions

- Re-read the Day 2 duplication audit and Day 3 plan.
- Inspected current helper/script structure under `scripts/` and `cmake/`.
- Re-inspected Make's `LIB_SRCS`, `TEST_SRCS`, and reviewed CMake parity
  targets.
- Re-inspected CMake's `add_library(...)`, `add_sparse_test(...)`, benchmark,
  example, install/export, and package-config ownership.
- Evaluated practical convergence mechanisms:
  - generated source lists
  - checked manifests
  - shared include fragments
  - CMake target-property extraction
  - Make variables
  - workflow-local assertions
  - documentation-only alignment
- Selected the library source list as the first convergence target.
- Recorded the architecture in
  `artifacts/day3-convergence-architecture-design.md`.

### Findings

- The best first landing path is a source manifest plus a small checker or
  generator, not an immediate full replacement of Make and CMake source lists.
- Full hidden generation would reduce duplication but also make reviewed build
  membership harder to read in diffs.
- A proof-preserving Day 4/5 batch should first add an explicit library-source
  manifest and enforce parity against Make and CMake. A later batch can decide
  whether to generate one side from the manifest.
- Test registration should remain a follow-up design topic. It carries
  platform exclusions, CTest count assertions, and reviewed-scope messages that
  are more proof-sensitive than library source membership.
- Benchmark and example registration should remain residual unless the library
  manifest pattern proves simple and reviewable.
- Package/install/export and platform workflow language should stay in the
  Day 7-12 lanes rather than being folded into source-list convergence.

### Validation

- Day 3 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, or script files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only design pass.
- Required hygiene checks: `git diff --check` and a trailing-whitespace scan on
  Sprint 97 planning files.

### Day 3 Exit State

Sprint 97 now has a build-convergence architecture. Day 4 should freeze a
small implementation boundary around a library-source manifest and parity
checker, while explicitly deferring test registration centralization until the
platform and proof assertions can be preserved.

## Day 4 - Source-List Reduction Boundary Freeze

### Goal

Freeze the first source-list convergence implementation batch before touching
Make, CMake, scripts, or build metadata.

### Actions

- Re-read the Day 3 convergence architecture.
- Inspected the current tracked support directories and `.gitignore`.
- Chose a tracked metadata location for durable build membership:
  `build-metadata/library_sources.txt`.
- Chose a checker implementation location:
  `scripts/check_library_sources.py`.
- Froze a minimal Make integration:
  - add `source-list-check`
  - call it from `quality-review-compile`
- Explicitly deferred CMake configure-time hooks until the checker proves
  stable.
- Recorded retained proof surfaces, validation commands, and rollback notes in
  `artifacts/day4-source-list-boundary-freeze.md`.

### Findings

- `build/` is ignored, so it should not hold durable source-list metadata.
- `cmake/` is available but is too CMake-specific for a shared Make/CMake
  source-membership manifest.
- A new `build-metadata/` directory is the clearest tracked home for shared
  build topology metadata.
- The first implementation batch should check parity rather than generate
  Make or CMake source lists.
- The first hook should be local and reviewed through Make's compile-quality
  path. A CMake configure-time hook can remain a residual until Day 6 or later.

### Validation

- Day 4 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, or script files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only boundary-freeze pass.
- Required hygiene checks: `git diff --check` and a trailing-whitespace scan on
  Sprint 97 planning files.

### Day 4 Exit State

Sprint 97 now has an exact Day 5 implementation boundary for the first
source-list convergence batch. Day 5 should add a manifest, a parser/checker,
and a Make reviewed-path hook without changing library behavior, package
claims, test registration, or platform workflow assertions.

## Day 5 - Source-List/Workflow Reduction Batch 1

### Goal

Land the first bounded source-list convergence batch by adding a library-source
manifest, a parity checker, and a reviewed Makefile hook.

### Actions

- Re-read the Day 4 boundary-freeze artifact.
- Added `build-metadata/library_sources.txt` with the current 42 library
  sources in reviewed Makefile order.
- Added `scripts/check_library_sources.py`.
- Implemented checker parsing for:
  - the manifest
  - `Makefile` `LIB_SRCS`
  - `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`
- Added `make source-list-check`.
- Hooked `source-list-check` into `quality-review-compile` between
  `format-check` and `lint`.
- Preserved existing Make and CMake source lists, test registrations,
  package/install surfaces, and platform workflow assertions.
- Recorded the implementation in
  `artifacts/day5-source-list-reduction-batch1.md`.

### Findings

- The manifest/checker path caught no current drift: manifest, Make, and CMake
  all report the same 42 library sources in the same order.
- The Makefile hook adds an inexpensive early reviewed check before lint.
- The implementation does not hide source membership; Make and CMake remain
  directly reviewable.
- CMake configure-time hook integration remains a Day 6 or later residual.
- Test, benchmark, and example registration convergence remains deferred.

### Validation

- `python3 scripts/check_library_sources.py` passed.
- `make source-list-check` passed.
- `make quality-review-compile` passed.
- No `.c` or `.h` files were modified.
- Full `make format && make lint && make test` is not required for this
  manifest/script/Makefile-only batch under the Day 4 validation plan.
- Required final hygiene: `git diff --check` and a trailing-whitespace scan on
  Sprint 97 planning files.

### Day 5 Exit State

Sprint 97 now has an enforced library-source parity check across the durable
manifest, Makefile, and CMake. Day 6 can decide whether to strengthen the
checker with CMake integration, keep it as the reviewed Makefile gate, or move
on to residual source-list/workflow cleanup.

## Day 6 - Source-List/Workflow Reduction Batch 2

### Goal

Close the selected source-list reduction by reconciling validation ownership
and checking that Make, CMake, and CI surfaces do not contradict the Day 5
guard.

### Actions

- Re-read the Day 5 implementation artifact and Day 6 plan.
- Re-inspected the current source-list validation surfaces:
  - `build-metadata/library_sources.txt`
  - `scripts/check_library_sources.py`
  - `Makefile` `source-list-check`
  - `Makefile` `quality-review-compile`
  - `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`
  - Linux, macOS, and Windows workflow proof lanes
- Confirmed that `source-list-check` remains owned by the reviewed Make
  compile-quality path and reaches Linux/macOS CI through existing
  `make quality-review-compile` workflow steps.
- Confirmed no Day 6 workflow label, package claim, or CMake configure-time
  hook edit was needed.
- Deferred CMake configure-time checking, `quality-review-cmake-compile`
  wiring, generated source lists, and test/benchmark/example registration
  centralization.
- Recorded the closeout in
  `artifacts/day6-source-list-reduction-closeout.md`.

### Findings

- The Day 5 implementation already completed the selected Day 4 batch:
  manifest, checker, direct Make target, and reviewed Makefile hook.
- Keeping the checker in `quality-review-compile` preserves a low-friction
  local and CI proof path without adding a CMake configure-time Python
  dependency.
- Make and CMake still intentionally duplicate readable source-list text, but
  they no longer carry silent-drift risk.
- Linux and macOS reviewed compile-quality workflows now inherit the
  source-list parity guard through their existing `make quality-review-compile`
  commands.
- Windows remains a CMake-first consumer subset proof and should not gain a new
  source-list-count assertion unless Sprint 97 later changes Windows validation
  ownership explicitly.

### Validation

- `python3 scripts/check_library_sources.py` passed.
- `make source-list-check` passed.
- `git diff --check` passed.
- Trailing-whitespace scan across Sprint 97 docs, build metadata, the checker,
  and `Makefile` passed with no matches.
- No `.c` or `.h` files were modified during Day 6.
- Full `make format && make lint && make test` is not required for this
  documentation/reconciliation closeout.

### Day 6 Exit State

Sprint 97 has closed the first source-list reduction batch. The selected
duplicated surface is lower-cost to maintain, validation ownership is explicit,
and the remaining build-topology duplication is documented as residual work for
later Sprint 97 or Sprint 98 decisions.

## Day 7 - Package-Surface Decision Audit

### Goal

Audit current install, export, package, README, CMake consumer, and workflow
claims before deciding whether Sprint 97 preserves static-first packaging or
earns one bounded shared-library lane.

### Actions

- Re-read the Day 7 plan and Day 6 closeout.
- Audited package and consumer claims in:
  - `README.md`
  - `INSTALL.md`
  - `CMakeLists.txt`
  - `Makefile`
  - `sparse.pc.in`
  - `cmake/SparseConfig.cmake.in`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - Linux, macOS, and Windows workflows
- Ran a Make install dry run to confirm the installed Make package shape.
- Ran a CMake configure probe with `BUILD_SHARED_LIBS=ON` to confirm the
  static-first configure behavior remains explicit.
- Compared three package decision options:
  - preserve static-first
  - earn one bounded shared-library lane
  - document shared-library work as deferred
- Recorded the audit in
  `artifacts/day7-package-surface-decision-audit.md`.

### Findings

- The live package evidence currently supports a maintained static-first
  contract.
- `make install` installs `libsparse_lu_ortho.a`, public headers, generated
  version metadata, and `sparse.pc`.
- CMake still declares `sparse_lu_ortho` as `STATIC`; configuring with
  `BUILD_SHARED_LIBS=ON` prints the explicit static-first message and does not
  earn a shared-library package contract.
- `tests/test_install.sh` and `tests/test_cmake_install.sh` validate the
  static-first install/export surface from different consumer paths.
- A bounded shared-library lane would require real new proof: build behavior,
  package metadata, installed consumer tests, runtime-loader handling, and
  platform-specific interpretation.
- `INSTALL.md` has stale workflow-wrapper wording for
  `quality-review-compile`; it still says `format-check + lint` after Day 5
  added `source-list-check`. This is a Day 8-10 docs/workflow coherence
  residual, not a package-contract contradiction.

### Validation

- `make -n install PREFIX=/tmp/sparse-sprint97-day7` confirmed the Make install
  shape without writing to the prefix.
- `cmake -S . -B build/sprint97-day7-shared-probe -DBUILD_SHARED_LIBS=ON
  -DCMAKE_INSTALL_PREFIX=/tmp/sparse-sprint97-day7` passed and printed the
  static-first status message.
- Day 7 changed planning documentation only.
- No `.c` or `.h` files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only package audit.
- Required final hygiene: `git diff --check` and a trailing-whitespace scan on
  Sprint 97 planning files.

### Day 7 Exit State

Sprint 97 now has a package-surface decision audit. Day 8 should preserve the
static-first contract unless it deliberately funds the larger proof burden for
one bounded shared-library lane.

## Day 8 - Package-Surface Decision Batch

### Goal

Land the Sprint 97 package decision and remove ambiguity around shared-library
packaging without widening unsupported platform or install/export claims.

### Actions

- Re-read the Day 7 package-surface decision audit and Day 8 plan.
- Chose the static-first package contract for Sprint 97.
- Updated `README.md` to make the static install summary explicit that
  shared-library packaging is deferred.
- Updated `INSTALL.md` to state the proof required before shared-library
  packaging can become maintained.
- Updated the stale `INSTALL.md` description of `quality-review-compile` so it
  includes the Day 5 `source-list-check` phase.
- Updated the `CMakeLists.txt` static-first explanatory comment to remove
  stale sprint-history wording while preserving behavior.
- Recorded the package decision in
  `artifacts/day8-package-surface-decision.md`.

### Findings

- The Day 7 audit did not expose a contradiction in the static-first package
  contract.
- A shared-library lane would require new build behavior, package metadata,
  installed-consumer proof, runtime-loader handling, and platform-specific
  interpretation.
- The package decision did not require changing Make install rules, CMake
  install/export rules, package metadata templates, install scripts, or CI.
- The only adjacent stale wording found during the Day 7 audit was the
  `quality-review-compile` phase summary in `INSTALL.md`.

### Validation

- `cmake -S . -B build/sprint97-day8-shared-probe -DBUILD_SHARED_LIBS=ON
  -DCMAKE_INSTALL_PREFIX=/tmp/sparse-sprint97-day8` passed and printed the
  static-first status message.
- `bash tests/test_install.sh` passed: 13 passed, 0 failed.
- `bash tests/test_cmake_install.sh` passed: 15 passed, 0 failed, 0 skipped.
- `python3 scripts/check_library_sources.py` passed.
- `git diff --check` passed.
- Trailing-whitespace scan across the touched docs, CMake, and Sprint 97
  planning files passed with no matches.
- No `.c` or `.h` files were modified.
- Full `make format && make lint && make test` is not required for this
  docs/CMake-comment package-decision batch.

### Day 8 Exit State

Sprint 97 now has an explicit package decision: preserve the maintained
static-first package surface and defer shared-library packaging until a future
change adds the required proof. Day 9 can align consumer proof from that stable
contract.

## Day 9 - Consumer Proof Follow-Through

### Goal

Align install/export and downstream consumer proof with the Day 8 static-first
package decision.

### Actions

- Re-read the Day 8 package decision artifact and Day 9 plan.
- Reviewed consumer-facing and proof surfaces:
  - `README.md`
  - `INSTALL.md`
  - `examples/README.md`
  - `examples/cmake_example/`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `CMakeLists.txt`
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`
  - Linux, macOS, and Windows workflows
- Added a no-shared-artifact assertion to `tests/test_install.sh`.
- Added a no-shared-artifact assertion to `tests/test_cmake_install.sh`.
- Preserved existing Make install, CMake install/export, package metadata, and
  workflow behavior.
- Recorded the proof alignment in
  `artifacts/day9-consumer-proof-follow-through.md`.

### Findings

- README and INSTALL already point consumers to the supported static package
  surface after Day 8.
- The examples remain aligned with supported build-tree and installed CMake
  consumer paths; no example doc edit was needed.
- The install scripts already proved positive consumer behavior but did not
  explicitly assert the negative shared-library non-claim.
- Adding no-shared-artifact assertions makes the local proof match the
  static-first contract more directly without adding a new package promise.

### Validation

- `bash tests/test_install.sh` passed: 14 passed, 0 failed.
- `bash tests/test_cmake_install.sh` passed: 16 passed, 0 failed, 0 skipped.
- `python3 scripts/check_library_sources.py` passed.
- `git diff --check` passed.
- Trailing-whitespace scan across touched shell scripts and Sprint 97 planning
  files passed with no matches.
- No `.c` or `.h` files were modified.
- Full `make format && make lint && make test` is not required for this
  shell-script/docs consumer-proof update.

### Day 9 Exit State

Sprint 97 consumer proof now matches the static-first package contract more
closely: install/export scripts verify the static archive and downstream
consumer paths, and they also fail if a shared-library artifact appears in the
installed package prefix.

## Day 10 - Workflow Coherence Follow-Through

### Goal

Align local workflow command descriptions, CI labels, expected counts, and
platform proof language with the Day 5 source-list guard and the Day 8-9
static-first package proof updates.

### Actions

- Re-read the Day 9 consumer proof artifact and Day 10 plan.
- Compared local reviewed targets, CMake parity targets, and workflow names:
  - `make quality-review-compile`
  - `make source-list-check`
  - `make quality-review-cmake-compile`
  - `make quality-review-cmake`
  - Linux CI reviewed Make/CMake/dead-code jobs
  - macOS reviewed Apple Clang path and supplemental install/`pkg-config` job
  - Windows reviewed CMake-first consumer subset
- Checked Windows CTest expected-count ownership against current CMake test
  registration and workflow env.
- Updated `README.md` command-map wording so `quality-review-compile` lists
  `source-list-check`.
- Confirmed no workflow YAML edit was needed.
- Recorded the workflow coherence pass in
  `artifacts/day10-workflow-coherence-follow-through.md`.

### Findings

- `README.md` was the remaining stale user-facing command surface: it still
  described `quality-review-compile` as `format-check + lint`.
- `INSTALL.md` and `Makefile` already described
  `quality-review-compile` as `format-check + source-list-check + lint`.
- Linux and macOS workflows already call `make quality-review-compile`, so they
  inherit the source-list guard without YAML changes.
- The macOS install/`pkg-config` job remains supplemental static-first package
  confidence and inherits the Day 9 no-shared-artifact assertion through
  `tests/test_install.sh`.
- Windows still expects 51 CTest registrations and remains reviewed CMake-first
  consumer proof only; Days 5-9 did not change Windows test registration or
  package ownership.

### Validation

- `python3 scripts/check_library_sources.py` passed.
- `make source-list-check` passed.
- `git diff --check` passed.
- README, INSTALL, and Makefile command descriptions now agree on
  `format-check + source-list-check + lint`.
- Trailing-whitespace scan across `README.md` and Sprint 97 planning files
  passed with no matches.
- No `.c` or `.h` files were modified.
- Full `make format && make lint && make test` is not required for this
  README/docs workflow-coherence update.

### Day 10 Exit State

Sprint 97 workflow wording now matches the current reviewed target behavior.
CI workflow labels and expected counts remain truthful, and the remaining
platform differences are explicit proof-boundary choices rather than
contradictions.

## Day 11 - Cross-Platform Truth Calibration

### Goal

Recalibrate macOS, Windows, and generic POSIX proof claims to match current
evidence without overstating reviewed platform parity.

### Actions

- Re-read the Day 10 workflow coherence artifact and Day 11 plan.
- Reviewed platform-facing surfaces:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- Mapped Linux, macOS, Windows, local install/export, CMake consumer, and
  documentation-only proof owners.
- Updated `docs/maintainer_guide.md` to replace older Windows
  `install-consumer lane` wording with the current reviewed CMake-first
  consumer subset language.
- Removed the stale Sprint 68 timestamp from the active fuzz/property platform
  confidence note while preserving the boundary.
- Recorded the calibration in
  `artifacts/day11-cross-platform-calibration.md`.

### Findings

- README, INSTALL, macOS workflow comments, and Windows workflow comments
  already match the current platform story.
- Linux remains the strongest reviewed source of truth.
- macOS remains narrower than Linux and carries supplemental static-first
  Make install/`pkg-config` confidence, not full reviewed install/export
  parity.
- Windows remains reviewed CMake-first consumer proof only with expected CTest
  count `51` and staged exclusions for `test_threads`,
  `test_sprint4_integration`, and `test_fuzz`.
- The maintainer guide had the only stale phrasing found during Day 11.

### Validation

- `python3 scripts/check_library_sources.py` passed.
- `git diff --check` passed.
- Trailing-whitespace scan across `docs/maintainer_guide.md` and Sprint 97
  planning files passed with no matches.
- Targeted platform wording check confirmed the maintained guide now uses the
  reviewed CMake-first consumer subset language.
- No `.c` or `.h` files were modified.
- Full `make format && make lint && make test` is not required for this
  documentation-only platform calibration.

### Day 11 Exit State

Sprint 97 now has current, evidence-backed platform calibration notes. The
remaining macOS and Windows limitations are explicit staged exclusions and
non-parity claims rather than hidden contradictions.

## Day 12 - Cross-Platform Product Follow-Through

### Goal

Apply the calibrated platform story to active build, package, workflow, and
front-door surfaces, then record the remaining platform residual queue.

### Actions

- Re-read the Day 11 calibration artifact and Day 12 plan.
- Checked active platform-facing surfaces:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- Confirmed README, INSTALL, maintainer guide, Linux workflow, and macOS
  workflow already match the Day 11 calibration.
- Updated `.github/workflows/windows-ci.yml` to remove the stale `Sprint 68`
  label from the active fuzz/property exclusion comment and log message.
- Preserved the Windows expected CTest count, staged exclusions, CMake-first
  proof scope, no-Makefile-parity statement, and no-install-validation-lane
  statement.
- Recorded the follow-through in
  `artifacts/day12-cross-platform-product-follow-through.md`.

### Findings

- The Windows workflow had the only remaining active product-surface stale
  platform wording found by Day 12.
- The stale text was historical labeling, not a proof-boundary contradiction.
- Removing the sprint-history label keeps the platform message current while
  preserving the important exclusion: Windows still does not claim reviewed
  `test_fuzz` or bounded lifecycle property evidence.
- No README, INSTALL, maintainer guide, Linux workflow, or macOS workflow edit
  was needed.

### Validation

- `python3 scripts/check_library_sources.py` passed.
- `git diff --check` passed.
- Trailing-whitespace scan across the touched workflow and Sprint 97 planning
  files passed with no matches.
- Targeted stale-wording scan for `Sprint 68` and `install-consumer lane` in
  active platform surfaces passed with no matches.
- No `.c` or `.h` files were modified.
- Full `make format && make lint && make test` is not required for this
  workflow/docs platform follow-through.

### Day 12 Exit State

Sprint 97 platform product surfaces now match the calibrated story. Remaining
macOS and Windows limitations are explicit residuals for Sprint 98 or later,
not implied support gaps hidden in current workflow language.

## Day 13 - Full Validation & Residual Queue

### Goal

Validate the Sprint 97 build/package/platform convergence work and freeze the
residual queue for Day 14 closeout.

### Actions

- Re-read the Day 12 platform follow-through artifact and Day 13 plan.
- Re-read the Day 2 duplication map and compared it with the current tree.
- Identified all active touched surfaces:
  - `Makefile`
  - `build-metadata/library_sources.txt`
  - `scripts/check_library_sources.py`
  - `CMakeLists.txt`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `.github/workflows/windows-ci.yml`
  - Sprint 97 planning artifacts
- Ran the strongest required validation for the touched surfaces.
- Ranked Sprint 98 candidates, package/product non-claims, platform proof gaps,
  and intentionally preserved independent proof surfaces.
- Recorded the validation and residual queue in
  `artifacts/day13-validation-and-residual-queue.md`.

### Findings

- The library source list remains duplicated in Make and CMake for readability,
  but it no longer carries silent drift risk because the manifest/checker is
  enforced through `quality-review-compile`.
- Test registration remains the next highest-value residual because it combines
  list duplication with platform exclusions and expected-count proof.
- Benchmark and example registration remain lower-priority residuals.
- Static-first package proof is stronger after the install scripts gained
  no-shared-artifact assertions.
- Windows and macOS platform limits remain explicit and current.

### Validation

- `python3 scripts/check_library_sources.py` passed.
- `make quality-review-compile` passed.
- `make quality-review-cmake-compile` passed:
  - CMake tests: 54
  - Makefile tests: 54
  - test-count parity passed
- `bash tests/test_install.sh` passed: 14 passed, 0 failed.
- `bash tests/test_cmake_install.sh` passed: 16 passed, 0 failed, 0 skipped.
- `git diff --check` passed.
- Trailing-whitespace scan across touched docs, workflow, scripts, build
  metadata, Makefile, and CMake passed with no matches.
- No `.c` or `.h` files were modified.
- Full `make format && make lint && make test` is not required by the sprint
  validation rule because no code/header files changed; the reviewed
  compile-quality wrapper was run because Make/script/build metadata changed.

### Day 13 Exit State

Sprint 97 has a validated build/package/platform convergence package and a
ranked residual queue. Day 14 can close the sprint from passed validation
rather than discovering hidden build, package, or platform contradictions.

## Day 14 - Sprint Closeout

### Goal

Close Sprint 97 with a validated build/package/product convergence record,
artifact index, and Sprint 98 handoff queue.

### Actions

- Re-read the Day 13 validation and residual queue artifact.
- Reviewed the complete Sprint 97 artifact set.
- Confirmed the active repository diff is scoped to:
  - source-list metadata/checking
  - Makefile reviewed compile-quality integration
  - static-first package wording
  - install/export proof strengthening
  - workflow/platform claim calibration
  - Sprint 97 planning artifacts
- Wrote the Sprint 97 closeout artifact:
  `artifacts/day14-sprint-closeout-and-handoff.md`.
- Included the final artifact index, Sprint 98 handoff queue, package/product
  non-claims, platform residuals, and validation evidence.

### Findings

- Sprint 97 reduced the highest-ranked Day 2 duplication risk by adding a
  source-list manifest/checker and wiring it into the reviewed Make
  compile-quality path.
- The static-first package contract is explicit and better proven by the
  install/export scripts.
- Workflow wording, README/INSTALL command descriptions, and platform
  maintainer guidance now match the changed proof ownership.
- The remaining work is intentionally residual: test registration, benchmark
  registration, example registration, optional checker self-tests, and possible
  future Windows CMake install validation.

### Validation

- Day 14 changed Sprint 97 planning documentation only.
- Day 13 validation remains the closeout evidence:
  - `python3 scripts/check_library_sources.py` passed.
  - `make quality-review-compile` passed.
  - `make quality-review-cmake-compile` passed.
  - `bash tests/test_install.sh` passed: 14 passed, 0 failed.
  - `bash tests/test_cmake_install.sh` passed: 16 passed, 0 failed, 0 skipped.
  - `git diff --check` passed.
  - trailing-whitespace scan passed with no matches.
- Required Day 14 hygiene:
  - `git diff --check`
  - trailing-whitespace scan across Sprint 97 planning files
- No `.c` or `.h` files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only closeout step.

### Day 14 Exit State

Sprint 97 is PR-ready pending retrospective creation. The sprint has a complete
artifact set, validated implementation notes, and a ranked Sprint 98 handoff
queue.
