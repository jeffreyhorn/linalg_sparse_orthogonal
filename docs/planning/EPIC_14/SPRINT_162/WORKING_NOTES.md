# Sprint 162 Working Notes

## Goal

Decide and close the remaining Windows package parity gap for `pkg-config` and
Makefile support without confusing it with CMake install validation.

Sprint 162 implements the Epic 14 Sprint 162 section in
`docs/planning/EPIC_14/PROJECT_PLAN.md`. The user prompt referenced the older
Epic 12 project-plan path, but the current branch and plan place Sprint 162
under Epic 14.

## Branch Baseline

- Branch: `sprint-162`
- Starting commit: `6cd56db4 Merge pull request #179 from jeffreyhorn/sprint-161`
- Starting state: Sprint 161 has landed selected partial-SVD comparison
  evidence and explicitly handed off Windows package parity as a separate
  product decision.

## Starting Evidence

| Surface | Current State | Sprint 162 Implication |
| --- | --- | --- |
| Windows CMake build/test | `.github/workflows/windows-ci.yml` runs the reviewed CMake configure/build/CTest path on `windows-2022` with `EXPECTED_WINDOWS_CTEST_COUNT=59`. | Treat this as Windows CMake consumer proof, not Makefile or `pkg-config` parity. |
| Windows CMake install/downstream | The Windows workflow installs a static `.lib`, headers, CMake package files, and `sparse.pc`; verifies static imported target metadata, exact-version behavior, mismatch rejection, downstream CMake consumers, and absence of DLL/shared metadata. | Starting proof is CMake install/downstream scoped and static-first. |
| Unix Make install and `pkg-config` | `tests/test_install.sh` validates `make install`, installed headers/library, `sparse.pc`, exact version, `pkg-config --cflags/--libs`, downstream compile/link/run, maintained example, and uninstall. Linux and macOS CI run this proof. | This is the Unix-side Make/`pkg-config` baseline Windows is compared against. |
| CMake install/export | `tests/test_cmake_install.sh` validates CMake install, static imported target metadata, no source/build leaks, exact-version behavior, downstream CMake example, installed `sparse.pc` metadata, and no shared-loader metadata. | CMake proof is cross-platform in concept, with Windows lane implemented separately in PowerShell. |
| Static-first guard | `scripts/static_package_deferral_check.sh` rejects `BUILD_SHARED_LIBS=ON`, checks static target/install metadata, and guards against shared export/import, ABI, loader, and package selector drift. | Retained non-claims should reuse or strengthen this guard. |
| Package metadata | `CMakeLists.txt` installs static archive metadata and `sparse.pc`; `sparse.pc.in` describes static archive package metadata. | Any Windows decision must keep static archive metadata bounded. |
| Public docs | `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` state Windows remains CMake-first and does not claim Makefile parity or `pkg-config` execution parity. | Docs already contain the retained non-claim baseline to audit. |
| Downstream example | `examples/cmake_example` is the maintained installed CMake consumer example. | Windows downstream proof should remain tied to selected package metadata and exact-version behavior. |

## Explicit Non-Goals

Sprint 162 does not claim or attempt to prove:

- package-manager availability through Homebrew, apt, dnf, pacman, vcpkg,
  Conan, or similar systems;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- static/shared selectors in package metadata;
- Windows Makefile parity unless explicitly selected by the Day 4 product
  decision;
- Windows `pkg-config` execution parity unless explicitly selected by the Day
  4 product decision;
- broad Windows platform parity;
- performance, release, or state-of-the-art evidence.

## Assumptions

- Windows CMake install/downstream validation is already maintained and should
  not be weakened.
- Unix Make install and `pkg-config` proof remains the strongest baseline for
  Makefile-style installed consumers.
- A retained non-claim is an acceptable product decision if backed by stronger
  wording and unsupported-surface checks.
- `sparse.pc` may be installed on Windows as metadata without proving Windows
  `pkg-config` execution parity.
- The selected Sprint 162 decision must be implemented with package/install
  evidence, not borrowed from solver, corpus, comparison, or benchmark
  evidence.

## Stop Conditions

Stop and reassess if a proposed change:

- implies Windows `pkg-config` execution support without a working provider
  and downstream validation path;
- implies Windows Makefile parity without a reviewed Windows Make execution
  path;
- weakens the static-first CMake install/downstream proof;
- introduces shared-library, dynamic ABI, runtime-loader, or package-manager
  wording as supported;
- uses package metadata existence as proof of execution parity;
- changes `.c` or `.h` files without running `make format`, `make lint`, and
  `make test`.

## Daily Log

### Day 1

- Re-read the Sprint 162 Epic 14 project-plan section and confirmed the prompt
  path mismatch with the older Epic 12 reference.
- Reviewed the Sprint 161 closeout handoff, which separates Windows package
  proof from solver comparison evidence.
- Inventoried current package surfaces across `Makefile`, `CMakeLists.txt`,
  `tests/test_install.sh`, `tests/test_cmake_install.sh`,
  `scripts/static_package_deferral_check.sh`, `.github/workflows/ci.yml`,
  `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml`,
  `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, and
  `examples/cmake_example`.
- Recorded explicit Sprint 162 non-goals for package-manager availability,
  shared libraries, dynamic ABI, runtime-loader behavior, broad Windows
  parity, performance, release, and state-of-the-art evidence.
- Created `artifacts/day1-sprint-intake.md`.

### Day 2

- Audited `.github/workflows/windows-ci.yml::install-and-downstream` as the
  reviewed Windows CMake install/downstream proof owner.
- Mapped Windows proof to static `.lib`, 19 headers, generated version header,
  CMake package files, installed `sparse.pc` metadata, static imported target
  metadata, exact-version behavior, mismatch rejection, generated and
  maintained CMake downstream consumers, source/build leak checks, and
  unsupported shared-loader/static-shared selector rejection.
- Audited `tests/test_install.sh` as the Unix Make install and `pkg-config`
  execution proof owner for Linux/macOS package lanes.
- Compared Windows CMake proof against Unix Make/`pkg-config` proof and
  recorded deltas for Make install, Make uninstall, `pkg-config --exists`,
  exact version, variables, cflags/libs, modversion, and downstream
  `pkg-config` compile/link/run behavior.
- Classified Windows-specific blockers: missing reviewed `pkg-config`
  provider, MSVC flag-shape mismatch, POSIX Makefile assumptions, CMake-first
  product wording, `sparse.pc` metadata misread risk, and lack of Make
  uninstall parity.
- Created `artifacts/day2-windows-package-audit.md`.

### Day 3

- Reviewed CMake package generation, installed target metadata, exact-version
  metadata, `sparse.pc` generation, Make install/uninstall validation, CMake
  install validation, Windows CMake install/downstream validation, and the
  static package deferral guard.
- Confirmed the static-first boundary is explicit: `BUILD_SHARED_LIBS=ON` is
  rejected, `sparse_lu_ortho` is a static target, install rules use archive
  destinations, CMake package metadata has no static/shared selectors, and
  `sparse.pc` describes static archive package metadata.
- Identified the main ambiguity risk: Windows installs and checks `sparse.pc`
  as metadata, but the reviewed Windows lane does not run `pkg-config`, map
  Unix-style flags to MSVC, or prove a downstream `pkg-config` consumer.
- Identified the Makefile parity blocker: the maintained install/uninstall
  proof is POSIX shell and utility based, so Windows Makefile parity remains a
  separate execution path unless explicitly promoted.
- Defined retained non-claim guard candidates for static-package deferral
  checks, public docs wording, Windows workflow comments/assertions, package
  metadata unsupported-wording scans, CMake selector scans, and report-index
  evidence separation.
- Created `artifacts/day3-metadata-boundary.md`.

### Day 4

- Built the product decision matrix for four options: promote Windows
  `pkg-config`, promote Windows Makefile parity, promote both, or retain both
  as explicit non-claims with stronger guards.
- Scored each option by maintainer cost, CI availability, user value,
  portability risk, and documentation complexity.
- Selected the retained non-claim path: Windows remains CMake-first for
  package install/downstream proof, while Windows Makefile parity and Windows
  `pkg-config` execution parity remain unsupported unless separately
  promoted.
- Documented the rationale that Windows already proves installed static
  package shape through CMake, but does not prove the Makefile or `pkg-config`
  execution front ends.
- Defined retained non-claim proof requirements for public docs, Windows
  workflow wording/assertions, static package guards, package metadata checks,
  evidence separation, and preservation of Linux/macOS Make/`pkg-config`
  validation.
- Defined rollback criteria for any later implementation that weakens CMake
  proof, blurs `sparse.pc` metadata with execution parity, or introduces
  unsupported ABI/package-manager/shared-library claims.
- Created `artifacts/day4-product-decision.md`.

### Day 5

- Converted the retained non-claim product decision into an implementation
  design for scripts, workflow wording, public docs, maintainer docs, and
  sprint evidence.
- Identified `scripts/static_package_deferral_check.sh`,
  `.github/workflows/windows-ci.yml`, `README.md`, `INSTALL.md`, and
  `docs/maintainer_guide.md` as the expected implementation surfaces.
- Defined files that should remain unchanged for the selected decision:
  `sparse.pc.in`, CMake target/install shape, `cmake/SparseConfig.cmake.in`,
  Unix install tests, CMake install tests, and library `.c`/`.h` files.
- Defined expected installed artifacts and metadata assertions for Linux/macOS
  Make/pkg-config proof, Linux/macOS CMake install proof, and Windows
  CMake-first install/downstream proof.
- Mapped downstream consumer expectations so Windows CMake consumers remain
  reviewed while Windows Makefile and `pkg-config` execution consumers remain
  retained non-claims.
- Defined failure diagnostics, support-tier wording, validation commands, and
  implementation acceptance criteria for Days 6-7.
- Created `artifacts/day5-proof-or-guard-design.md`.

### Day 6

- Began the retained non-claim implementation path by extending
  `scripts/static_package_deferral_check.sh`.
- Added `check_windows_package_nonclaim_wording` to make the Windows
  CMake-first package tier and retained Windows Makefile/`pkg-config`
  non-claims executable through the existing static package guard.
- Guarded `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, and
  `.github/workflows/windows-ci.yml` for wording that separates Windows CMake
  install/downstream validation from Windows Makefile and `pkg-config`
  execution parity.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed with the new
  Windows package non-claim wording check.
- Preserved package templates, CMake install metadata, Unix install tests,
  CMake install tests, Windows workflow runtime behavior, and library source
  files.
- Created `artifacts/day6-implementation-foundation.md`.

### Day 7

- Completed the retained non-claim guard implementation by adding
  `check_windows_workflow_no_unselected_package_execution` to
  `scripts/static_package_deferral_check.sh`.
- Added workflow-level diagnostics that fail if the Windows workflow starts
  running `pkg-config`, `make install`, or `make uninstall` without a new
  product decision and reviewed proof path.
- Preserved the Day 6 wording checks that keep README, INSTALL, maintainer
  guide, and Windows workflow language aligned with the CMake-first Windows
  package tier.
- Kept package templates, CMake install metadata, Windows workflow runtime
  behavior, Unix install tests, CMake install tests, library source, and public
  headers unchanged.
- Created `artifacts/day7-implementation-completion.md`.

### Day 8

- Reviewed `.github/workflows/windows-ci.yml` against the retained non-claim
  product decision and the Day 6-7 static package guard.
- Preserved the Windows runner, `EXPECTED_WINDOWS_CTEST_COUNT=59`, CMake
  configure/build/CTest lane, and CMake install/downstream validation behavior.
- Updated workflow wording to describe `sparse.pc` as metadata-only
  inspection, not Windows `pkg-config` command execution proof.
- Updated the CTest inspection log line to keep the hosted output aligned with
  the retained non-claim boundary.
- Updated the static package guard pattern to require the stricter
  metadata-only Windows workflow wording.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed after the CI
  wording and guard alignment.
- Confirmed no Windows `pkg-config`, `make install`, or `make uninstall`
  execution was added.
- Created `artifacts/day8-ci-alignment.md`.

### Day 9

- Reviewed the Windows install/downstream job's generated CMake consumer,
  maintained CMake example, exact-version consumer, mismatch rejection, and
  static metadata checks.
- Confirmed the downstream evidence matches the selected Windows package
  decision: installed CMake package consumers are reviewed, while Windows
  Makefile and `pkg-config` execution remain non-claims.
- Added hosted-output diagnostics to `.github/workflows/windows-ci.yml` so CI
  logs identify `sparse.pc` validation as metadata-only and label each
  downstream consumer as installed CMake package evidence.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed with the
  downstream wording and retained non-claim checks.
- Preserved workflow commands and did not add Windows `pkg-config`,
  `pkg-config --exists`, `pkg-config --cflags`, `pkg-config --libs`,
  `pkg-config --modversion`, `make install`, or `make uninstall` execution.
- Created `artifacts/day9-downstream-evidence.md`.

### Day 10

- Ran `bash scripts/static_package_deferral_check.sh`; it passed with the
  static-first guard, Windows non-claim wording checks, and no unselected
  Windows package execution checks.
- Ran `bash tests/test_install.sh`; it passed 23 install validation checks for
  Make install/uninstall, `pkg-config`, downstream consumers, static archive
  metadata, and unsupported package/ABI wording.
- Ran `bash tests/test_cmake_install.sh`; it passed 27 CMake install/export
  checks for static package metadata, downstream CMake consumers,
  exact-version behavior, mismatch rejection, and installed `sparse.pc`
  metadata.
- Recorded the changed-file quality-gate decision: no `.c` or `.h` files were
  modified, so `make format && make lint && make test` is not required for Day
  10.
- Created `artifacts/day10-focused-validation.md`.

### Day 11

- Updated `README.md` to state that Windows installs and inspects `sparse.pc`
  as static package metadata through the reviewed CMake install/downstream
  lane, without running `pkg-config` or claiming Windows Makefile or
  `pkg-config` execution parity.
- Updated `INSTALL.md` support-tier wording so Windows `sparse.pc` handling is
  described as metadata-only inspection and remains narrower than Unix
  Makefile/`pkg-config` proof.
- Updated `docs/maintainer_guide.md` to use metadata-only `sparse.pc`
  inspection wording and added a Sprint 162 history note for the retained
  package non-claim boundary.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed with the
  updated documentation wording.
- Preserved package templates, install scripts, CMake metadata, workflow
  command behavior, and library source/header files.
- Created `artifacts/day11-docs-alignment.md`.

### Day 12

- Re-ran `bash scripts/static_package_deferral_check.sh`; it passed with the
  static-first package guard, Windows package non-claim wording checks, and no
  unselected Windows package execution checks.
- Re-ran `bash tests/test_install.sh`; it passed 23 Make install,
  `pkg-config`, downstream consumer, uninstall, and package metadata checks.
- Re-ran `bash tests/test_cmake_install.sh`; it passed 27 CMake
  install/export, static metadata, downstream CMake consumer, exact-version,
  mismatch rejection, and installed `sparse.pc` metadata checks.
- Confirmed `actionlint` and `pwsh` are not available locally, so Windows
  workflow execution remains hosted-only.
- Recorded the hosted-only Windows verification checklist for CTest count,
  CMake install/downstream validation, installed static `.lib`, metadata-only
  `sparse.pc` inspection, downstream CMake consumers, exact-version behavior,
  mismatch rejection, and absence of unselected Windows package execution.
- Created `artifacts/day12-cross-platform-validation.md`.

### Day 13

- Traced positive Windows package claims to `.github/workflows/windows-ci.yml`,
  `CMakeLists.txt`, `sparse.pc.in`, `tests/test_cmake_install.sh`,
  `tests/test_install.sh`, and `scripts/static_package_deferral_check.sh`.
- Traced retained non-claims for Windows Makefile parity, Windows
  `pkg-config` execution parity, shared-library support, dynamic ABI,
  runtime-loader behavior, package-manager support, and broad Windows parity
  to docs, workflow wording, and static package guard checks.
- Reviewed README, INSTALL, maintainer guide, Windows workflow, working notes,
  and Sprint 162 artifacts for CMake, Makefile, `pkg-config`, static archive,
  exact-version, package-manager, shared-library ABI, and platform wording.
- Confirmed changed implementation surfaces remain narrow: workflow wording and
  output diagnostics, static guard checks, support-tier docs, and Sprint 162
  evidence artifacts.
- Finalized the Sprint 163 handoff: performance publication must keep
  performance evidence separate from Sprint 162 package proof and must not
  soften Windows package, package-manager, shared-library, dynamic ABI,
  runtime-loader, or broad platform non-claims.
- Created `artifacts/day13-evidence-claim-review.md`.

### Day 14

- Re-ran final targeted validation for the changed package/docs/workflow
  surface.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed with the
  static-first guard, Windows package non-claim wording, and no unselected
  Windows package execution checks.
- Ran `bash tests/test_install.sh`; it passed 23 Make install,
  `pkg-config`, downstream consumer, uninstall, and package metadata checks.
- Ran `bash tests/test_cmake_install.sh`; it passed 27 CMake
  install/export, static metadata, downstream CMake consumer, exact-version,
  mismatch rejection, and installed `sparse.pc` metadata checks.
- Reviewed changed files and confirmed implementation remains limited to
  workflow wording/output diagnostics, static guard checks, support-tier docs,
  and Sprint 162 evidence artifacts.
- Prepared the retrospective input set from `PLAN.md`, `WORKING_NOTES.md`, and
  Day 1-14 artifacts.
- Confirmed the Sprint 163 handoff is ready: methodology-bound performance
  publication must keep performance evidence separate from Sprint 162 package
  proof and preserve retained package non-claims.
- Created `artifacts/day14-closeout.md`.
