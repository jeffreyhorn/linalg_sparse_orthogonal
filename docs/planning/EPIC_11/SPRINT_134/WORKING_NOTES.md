# Sprint 134 Working Notes

## Sprint Goal

Advance platform support where feasible and make Linux/macOS/Windows install
and staged validation tiers even more explicit.

Sprint 134 follows the Sprint 133 static-first package/ABI decision. It must
not widen local package proof, supplemental CI confidence, or staged Windows
tests into reviewed cross-platform install parity unless the selected
platform-specific implementation and validation prove that claim.

## Starting Constraints

- Treat Sprint 133 as the package contract baseline: static-first install and
  export support is maintained; shared-library packaging, dynamic ABI
  compatibility, package-manager support, runtime-loader behavior, and broad
  platform install parity remain explicit non-claims.
- Keep `BUILD_SHARED_LIBS=ON` rejection unless a new product decision funds
  the full shared-library proof stack.
- Treat `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh` as local package-contract gates.
- Treat Linux as the strongest reviewed source of truth today: reviewed
  Makefile compile-quality, reviewed CMake parity, and reviewed dead-code
  lanes live in `.github/workflows/ci.yml`.
- Treat macOS as reviewed Apple Clang compile/CMake/sanitize/wall-check plus
  supplemental Homebrew GCC, supplemental static-first Make
  install/`pkg-config`, and supplemental CMake install/export confidence.
- Treat Windows as reviewed MSVC CMake-first consumer subset plus
  supplemental CMake install/downstream confidence, with no separate reviewed
  install-validation lane and no Windows Makefile parity.
- Treat Windows `test_threads`, `test_sprint4_integration`, and `test_fuzz`
  as staged exclusions until Sprint 134 explicitly changes that status with
  updated CTest counts and validation ownership.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check` and a focused
  markdown whitespace scan over `docs/planning/EPIC_11/SPRINT_134`.

## Input Artifact Inventory

| Input | Role in Sprint 134 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 134 | Defines the seven Sprint 134 items for platform gap audit, Linux install CI, macOS install/export parity, Windows install validation, Windows staged tests, validation, and closeout. |
| `docs/planning/EPIC_11/SPRINT_134/PLAN.md` | Provides day-level execution order and 168-hour budget. |
| `docs/planning/EPIC_11/SPRINT_133/artifacts/day14-closeout-package-abi-handoff.md` | Provides static-first package truth, residual queues, and Sprint 134 handoff notes. |
| `docs/planning/EPIC_11/SPRINT_133/RETROSPECTIVE.md` | Records Sprint 133 validation metrics and residual platform/package debt. |
| `.github/workflows/ci.yml` | Linux reviewed and supplemental CI topology. |
| `.github/workflows/macos-ci.yml` | macOS reviewed Apple Clang lane, supplemental Homebrew GCC lane, supplemental Make install/`pkg-config` confidence job, and supplemental CMake install/export confidence job. |
| `.github/workflows/windows-ci.yml` | Windows reviewed MSVC CMake subset, supplemental CMake install/downstream confidence job, expected CTest count, and staged exclusion output. |
| `README.md` | Front-door cross-platform CI and install support summary. |
| `INSTALL.md` | Operational install, staged install, platform setup, and install-validation support truth. |
| `docs/maintainer_guide.md` | Maintainer package/platform support truth and staged/non-claim boundaries. |
| `CMakeLists.txt` | CMake test registration, Windows staged exclusions, install/export rules, and package generation. |
| `tests/test_install.sh` | Local Unix Make install/uninstall and `pkg-config` consumer proof; also macOS supplemental CI proof. |
| `tests/test_cmake_install.sh` | Local CMake install/export and installed CMake consumer proof. |
| `scripts/static_package_deferral_check.sh` | Static-first package claim guard. |

## Candidate Platform Surfaces

| Surface | Current role | Day owner |
| --- | --- | --- |
| `.github/workflows/ci.yml` | Linux reviewed Makefile compile-quality, reviewed CMake parity, reviewed dead-code, plus supplemental runtime, bench-fast, TSan, and coverage. | Days 2-4, 12-14 |
| `.github/workflows/macos-ci.yml` | macOS reviewed Apple Clang path, supplemental Homebrew GCC path, supplemental Make install/`pkg-config` path, and supplemental CMake install/export path. | Days 2, 5-7, 12-14 |
| `.github/workflows/windows-ci.yml` | Windows reviewed MSVC CMake-first subset, supplemental CMake install/downstream confidence path, expected 54 CTests, and staged exclusion output. | Days 2, 8-11, 13-14 |
| `CMakeLists.txt` | Test registration, Windows exclusions, CMake install/export, exact package version, and static target ownership. | Days 2, 8-11, 13 |
| `tests/test_install.sh` | Make install/`pkg-config` proof and macOS supplemental job command. | Days 2-4, 5-7, 13 |
| `tests/test_cmake_install.sh` | Local CMake install/export proof and candidate Linux/macOS/Windows install proof source. | Days 2-9, 13 |
| `scripts/static_package_deferral_check.sh` | Static-first support wording and build-system deferral guard. | Days 4, 7, 12-14 |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Public and maintainer support-tier wording. | Days 2, 4, 7, 9, 11-14 |
| Windows staged tests | `test_threads`, `test_sprint4_integration`, and `test_fuzz` are outside the reviewed Windows subset. | Days 8-11, 13-14 |
| Package-manager surfaces | No recipes or manager-specific consumer proof currently exist. | Days 12-14 if wording changes |

## Day-Level Ownership

| Day | Owner focus | Project-plan items |
| --- | --- | --- |
| 1 | Sprint intake, artifact baseline, platform support-tier inventory, and claim fences | Items 1-7 |
| 2 | Linux/macOS/Windows platform gap audit | Item 1 |
| 3 | Linux install CI promotion decision | Item 2 |
| 4 | Linux install CI implementation or explicit deferral | Item 2 |
| 5 | macOS install/export parity audit | Item 3 |
| 6 | macOS install/export decision | Item 3 |
| 7 | macOS install/export implementation or explicit deferral | Item 3 |
| 8 | Windows install validation audit/design and CTest impact plan | Item 4 |
| 9 | Windows install validation implementation or explicit deferral | Item 4 |
| 10 | Windows staged test and CTest membership re-audit | Item 5 |
| 11 | Windows staged lane follow-through | Item 5 |
| 12 | Support-tier docs and CI comment alignment | Items 2-6 |
| 13 | Integrated platform validation and required quality gates | Item 6 |
| 14 | Platform tier closeout, staged-exclusion register, and Sprint 135 handoff | Item 7 |

## Validation Expectations

| Change type | Required validation |
| --- | --- |
| Documentation-only Sprint 134 artifacts | `git diff --check` and focused markdown whitespace scan over `docs/planning/EPIC_11/SPRINT_134`. |
| Workflow comment-only edits | `git diff --check`, focused workflow text scan, and support wording drift scan. |
| Linux workflow install proof edits | Workflow-equivalent local command, relevant package proof scripts, and docs hygiene. |
| macOS workflow install/export edits | Local equivalent where possible, package proof scripts, and explicit note for macOS-only runner limits. |
| Windows workflow or CTest membership edits | CMake configure/build/`ctest -N` registration proof where possible, expected-count update, and explicit Windows-only validation notes. |
| CMake test registration edits | `make quality-review-cmake` or focused CMake configure/build/CTest registration proof, plus affected package proof. |
| Package proof script edits | `bash -n` for changed scripts, focused script execution, relevant install proof, and docs hygiene. |
| Public header or C source edits | Focused platform/package proof plus `make format && make lint && make test`. |
| Support-tier docs edits | Claim drift scan across README, INSTALL, maintainer guide, workflow comments, and Sprint artifacts. |

## Scope Boundaries

- Sprint 134 may promote, strengthen, or explicitly defer platform install
  proof, but each decision must name the support tier it creates.
- Sprint 134 must not infer package-manager support, shared-library support,
  dynamic ABI compatibility, or runtime-loader behavior from the Linux
  reviewed static-first package-contract lane.
- Sprint 134 must not infer macOS full reviewed install/export parity from the
  supplemental Make install/`pkg-config` or CMake install/export confidence
  jobs.
- Sprint 134 must not infer reviewed Windows install-validation parity,
  Windows Makefile parity, or reviewed Windows property/fuzz/thread coverage
  from the current MSVC CMake-first subset or supplemental CMake
  install/downstream job.
- Sprint 134 must keep CTest counts and staged exclusions synchronized with
  workflow comments and maintainer docs.
- Sprint 134 must preserve Sprint 133 static-first package/ABI non-claims
  unless a new product decision explicitly changes them.

## Day 1 Notes

- Created the Sprint 134 working-notes baseline and artifact directory.
- Re-read the Sprint 134 project-plan section and mapped Items 1-7 to
  day-level owners.
- Reviewed Sprint 133 closeout and retrospective as the inherited static-first
  package/ABI support baseline.
- Inventoried current platform workflow owners:
  `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, and
  `.github/workflows/windows-ci.yml`.
- Recorded Linux current state as strongest reviewed source of truth:
  reviewed Makefile compile-quality, reviewed CMake parity, reviewed dead-code,
  and supplemental runtime, sanitizer, benchmark, TSan, and coverage signals.
- Recorded macOS current state as reviewed Apple Clang compile/CMake/sanitize
  path plus supplemental Homebrew GCC and supplemental static-first Make
  install/`pkg-config` proof.
- Recorded Windows current state as reviewed MSVC CMake-first subset with
  `EXPECTED_WINDOWS_CTEST_COUNT=54` and staged exclusions for `test_threads`,
  `test_sprint4_integration`, and `test_fuzz`.
- Recorded validation expectations for docs, workflows, package proof scripts,
  CMake registration, Windows CTest membership, and any future C/header edits.
- Preserved the Sprint 133 package fences: no shared-library packaging,
  dynamic ABI compatibility, package-manager support, runtime-loader claim, or
  broad platform install parity without explicit implementation and proof.

## Day 2 Notes

- Wrote the platform gap audit artifact.
- Re-audited Linux workflow topology in `.github/workflows/ci.yml`:
  reviewed Makefile compile-quality, reviewed CMake parity, reviewed
  dead-code, and supplemental direct runtime, sanitizer, benchmark, TSan, and
  coverage lanes.
- Recorded the Linux install CI gap: local package proof scripts remain
  developer-side proof surfaces and are not yet a separate reviewed Linux
  install CI lane.
- Re-audited macOS workflow topology in `.github/workflows/macos-ci.yml`:
  reviewed Apple Clang path, supplemental Homebrew GCC path, and supplemental
  static-first Make install/`pkg-config` job.
- Recorded the macOS install/export parity gap: the current supplemental
  package job is not reviewed macOS CMake install/export parity.
- Re-audited Windows workflow topology in `.github/workflows/windows-ci.yml`:
  reviewed MSVC CMake configure/build, `ctest -N`, expected count check, and
  full `ctest`.
- Recorded Windows expected CTest count as 54 and staged exclusions as
  `test_threads`, `test_sprint4_integration`, and `test_fuzz`.
- Separated Windows install validation from Windows Makefile parity; both are
  currently non-claims unless Sprint 134 explicitly changes support tiers.
- Compared README, INSTALL, maintainer guide, and workflow comments for
  wording drift.
- Found one support wording drift item: `docs/maintainer_guide.md` still
  records the older Sprint 112 Windows count of 51 registered CTests, while
  the current Windows workflow expects 54.
- No code, workflow, or support docs were changed on Day 2 beyond Sprint 134
  planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 3 Notes

- Wrote the Linux install CI decision artifact.
- Reviewed `.github/workflows/ci.yml` and confirmed Linux currently carries
  reviewed Makefile compile-quality, reviewed CMake parity, reviewed
  dead-code, plus supplemental direct runtime, sanitizer, benchmark, TSan, and
  coverage lanes.
- Compared the Sprint 133 package proof stack against CI promotion criteria:
  `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh`.
- Decided to promote a bounded Linux reviewed static-first package-contract
  lane on Day 4.
- Scoped the selected lane to the static-first contract only: Make
  install/`pkg-config`, CMake install/export and installed CMake consumer, and
  static deferral proof.
- Preserved non-claims for shared-library packaging, dynamic ABI
  compatibility, package-manager support, runtime-loader behavior, macOS
  install/export parity, Windows install validation, and Windows Makefile
  parity.
- Defined Day 4 workflow comment and support-doc update queue so the Linux
  package lane is not confused with broader platform/package support.
- No code, workflow, or support docs were changed on Day 3 beyond Sprint 134
  planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 4 Notes

- Implemented the Day 3 Linux install CI decision.
- Added a separate `package-contract` job to `.github/workflows/ci.yml` named
  `Linux reviewed static-first package contract`.
- The new Linux job installs package proof tooling and runs
  `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, and
  `bash scripts/static_package_deferral_check.sh`.
- Updated the Linux workflow top comment so the reviewed static-first
  package-contract lane is explicit and bounded.
- Updated `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` so support
  wording distinguishes Linux reviewed package-contract proof from macOS
  supplemental Make install/`pkg-config` confidence and Windows CMake-first
  reviewed support.
- Corrected the stale maintainer-guide Windows CTest count from 51 to the
  current workflow value of 54 while touching support-tier docs.
- Preserved Sprint 133 non-claims for shared-library packaging, dynamic ABI
  compatibility, package-manager support, runtime-loader behavior, macOS full
  install/export parity, Windows install validation, and Windows Makefile
  parity.
- Ran `bash -n scripts/static_package_deferral_check.sh`; it passed.
- Ran `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/ci.yml")'`;
  it passed.
- Ran `bash tests/test_install.sh`; it passed 22 checks with 0 failures.
- Ran `bash tests/test_cmake_install.sh`; it passed 21 checks with 0 failures
  and 0 skips.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over touched workflow, support docs, and
  Sprint 134 planning paths; it passed.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 5 Notes

- Wrote the macOS install/export parity audit artifact.
- Re-audited `.github/workflows/macos-ci.yml` and separated the reviewed Apple
  Clang path, supplemental Homebrew GCC path, and supplemental Make
  install/`pkg-config` job.
- Confirmed the reviewed Apple Clang lane runs `make quality-review-compile`,
  `make quality-review-cmake`, `make wall-check`, and `make sanitize`.
- Confirmed the Homebrew GCC matrix leg remains supplemental direct
  build/test/wall-check coverage.
- Confirmed the `install-and-pkgconfig` job runs `bash tests/test_install.sh`
  as supplemental static-first Make install/`pkg-config` confidence only.
- Recorded the macOS CMake install/export gap: no macOS CI lane currently runs
  `bash tests/test_cmake_install.sh`, installed `find_package(Sparse)`,
  static imported-target metadata checks, or source/build path leak scans.
- Recorded toolchain and runtime risks around adding `tests/test_cmake_install.sh`
  to macOS CI, including runner CMake availability, Apple Clang versus
  Homebrew GCC tiering, path-format behavior, runtime cost, and failure
  classification.
- Confirmed README, INSTALL, maintainer guide, and macOS workflow comments
  already keep current macOS support wording evidence-bounded.
- No code, workflow, or support docs were changed on Day 5 beyond Sprint 134
  planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 6 Notes

- Wrote the macOS install/export decision artifact.
- Decided to add a separate supplemental macOS CMake install/export confidence
  job on Day 7.
- Selected the supplemental tier rather than reviewed parity because the lane
  should first collect hosted-runner runtime and flake evidence before any
  reviewed macOS install/export parity claim.
- Scoped the selected job to default Apple Clang on `macos-latest`; no
  Homebrew GCC package matrix is selected for Sprint 134.
- Selected Day 7 commands: `bash tests/test_cmake_install.sh` and
  `bash scripts/static_package_deferral_check.sh`.
- Preserved the existing macOS reviewed Apple Clang compile-quality/CMake
  parity/sanitize/wall-check path and the existing supplemental Make
  install/`pkg-config` job.
- Preserved non-claims for shared-library packaging, dynamic ABI
  compatibility, package-manager support, runtime-loader behavior, Windows
  install validation, and Windows Makefile parity.
- Defined Day 7 workflow comment, support-doc update, validation, rollback,
  and triage plan.
- No code, workflow, or support docs were changed on Day 6 beyond Sprint 134
  planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 7 Notes

- Implemented the Day 6 macOS decision by adding a separate
  `cmake-install-export` job to `.github/workflows/macos-ci.yml`.
- The new job is named `macOS supplemental CMake install/export confidence
  path` and runs `bash tests/test_cmake_install.sh` followed by
  `bash scripts/static_package_deferral_check.sh`.
- Updated the macOS workflow top comment so the Make install/`pkg-config` and
  CMake install/export package jobs are both explicitly supplemental.
- Updated `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` so macOS
  support wording includes supplemental CMake install/export confidence while
  preserving the narrower-than-Linux support tier.
- Preserved the existing reviewed Apple Clang compile-quality/CMake
  parity/sanitize/wall-check path, supplemental Homebrew GCC path, and
  supplemental Make install/`pkg-config` job.
- Preserved non-claims for full reviewed macOS install/export parity,
  shared-library packaging, dynamic ABI compatibility, package-manager
  support, runtime-loader behavior, Windows install validation, and Windows
  Makefile parity.
- Wrote the macOS install/export implementation artifact with residual queue
  and rollback notes.
- Ran `bash -n scripts/static_package_deferral_check.sh`; it passed.
- Ran `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/macos-ci.yml")'`;
  it passed.
- Ran `bash tests/test_cmake_install.sh`; it passed 21 checks with 0 failures
  and 0 skips.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over touched workflow, support docs, and
  Sprint 134 planning paths; it passed.
- No `.c` or `.h` files changed, so the full C quality gate was not required.
## Day 8 Notes

- Wrote the Windows install validation audit/design artifact.
- Re-audited `.github/workflows/windows-ci.yml` as the reviewed MSVC
  CMake-first consumer subset: configure, build, `ctest -N`, expected count
  check, and full `ctest`.
- Confirmed the workflow still expects `EXPECTED_WINDOWS_CTEST_COUNT=54` and
  names staged exclusions for `test_threads`, `test_sprint4_integration`, and
  `test_fuzz`.
- Re-audited `CMakeLists.txt` test registration and confirmed the selected
  Windows install/downstream path should not change CTest membership.
- Ran a local CMake configure plus `ctest -N` registration audit; this
  non-Windows host registered 57 tests, which reconciles to the Windows
  expected 54 after the three staged Windows exclusions.
- Decided to add a separate supplemental Windows CMake install/downstream
  confidence job on Day 9 rather than a reviewed install-validation lane.
- Selected Day 9 proof shape: configure/build/install with the Visual Studio
  2022 generator, inspect installed static artifacts/config files, configure
  `examples/cmake_example` with `CMAKE_PREFIX_PATH`, run the installed
  downstream example, and verify exact/mismatched package-version behavior.
- Kept Windows Makefile parity, Windows `pkg-config` support, package-manager
  support, shared-library packaging, dynamic ABI compatibility, and Windows
  staged CTest promotions as separate non-claims or later-day owners.
- Removed the temporary local `build-sprint134-day8` CMake audit directory.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over Sprint 134 planning paths; it passed.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 9 Notes

- Implemented the Day 8 Windows install validation decision by adding a
  separate `install-and-downstream` job to `.github/workflows/windows-ci.yml`.
- The new job is named `Windows supplemental CMake install/downstream
  confidence path` and runs on `windows-2022` with the Visual Studio 2022 x64
  generator.
- The supplemental proof configures, builds, installs, checks the installed
  static `.lib`, rejects unexpected `.dll` artifacts, checks 19 installed
  headers and CMake package files, configures/builds/runs
  `examples/cmake_example` through `CMAKE_PREFIX_PATH`, and checks exact and
  mismatched package-version behavior.
- Preserved the existing reviewed Windows CMake configure/build/`ctest -N`/full
  `ctest` job unchanged.
- Left `EXPECTED_WINDOWS_CTEST_COUNT=54` unchanged because the selected
  implementation adds a separate workflow job rather than CTest membership.
- Updated `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` so Windows
  support wording includes supplemental CMake install/downstream confidence
  without claiming a separate reviewed install-validation lane.
- Preserved non-claims for Windows Makefile parity, Windows `pkg-config`
  support, package-manager support, shared-library packaging, dynamic ABI
  compatibility, and Windows staged CTest promotions.
- Wrote the Windows install validation implementation artifact with rollback
  notes and residual queue.
- Ran `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/windows-ci.yml")'`;
  it passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `bash tests/test_cmake_install.sh`; it passed 21 checks with 0 failures
  and 0 skips.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over touched workflow, support docs, and
  Sprint 134 planning paths; it passed.
- Confirmed no `.c`, `.h`, or `CMakeLists.txt` files changed, so the full C
  quality gate was not required and CTest membership remained unchanged.

## Day 10 Notes

- Wrote the Windows staged test re-audit artifact.
- Re-audited Windows CTest membership and confirmed
  `.github/workflows/windows-ci.yml` still expects
  `EXPECTED_WINDOWS_CTEST_COUNT=54`.
- Counted 57 `add_sparse_test(...)` registrations on non-Windows platforms.
- Ran a local CMake configure plus `ctest -N` registration audit; this
  non-Windows host registered 57 tests.
- Reconciled the Windows expected count as 57 total non-Windows tests minus
  the three staged Windows exclusions: `test_threads`,
  `test_sprint4_integration`, and `test_fuzz`.
- Re-audited `test_threads` and `test_sprint4_integration`; both include
  `<pthread.h>` directly and remain gated by `if(Threads_FOUND AND NOT WIN32)`.
- Re-audited `test_fuzz`; it includes `<unistd.h>` and uses POSIX temp-file
  APIs including `mkstemps`, `close`, and `unlink`, so the fuzz/property lane
  remains gated by `if(NOT WIN32 AND NOT MSVC)`.
- Rejected as-is promotion for all three staged tests because each still has
  source-level portability blockers under MSVC.
- Recommended Day 11 reinforce the staged exclusions rather than promote tests
  without Windows-native source changes and hosted MSVC proof.
- Removed the temporary local `build-sprint134-day10` CMake audit directory.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over Sprint 134 planning paths; it passed.
- No `.c`, `.h`, workflow, or `CMakeLists.txt` files changed on Day 10, so the
  full C quality gate was not required and CTest membership remained
  unchanged.

## Day 11 Notes

- Wrote the Windows staged lane follow-through artifact.
- Implemented the Day 10 recommendation: no Windows staged CTest member was
  promoted.
- Updated `.github/workflows/windows-ci.yml` so the top comment and `ctest -N`
  output name the current source-level blockers: pthread APIs for
  `test_threads` and `test_sprint4_integration`, and POSIX temp-file APIs for
  `test_fuzz`.
- Updated `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` so support
  wording says pthread/POSIX-backed staged tests remain outside the reviewed
  Windows subset.
- Added maintainer-guide promotion gates: source portability or Windows-native
  equivalents, intentional CTest count update, and hosted MSVC
  configure/build/execute evidence.
- Preserved `CMakeLists.txt` test registration unchanged.
- Preserved `.github/workflows/windows-ci.yml`
  `EXPECTED_WINDOWS_CTEST_COUNT=54` unchanged.
- Ran `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/windows-ci.yml")'`;
  it passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran a local CMake configure plus `ctest -N` registration audit; this
  non-Windows host registered 57 tests.
- Confirmed the Windows CTest count reconciliation remains 57 total
  non-Windows tests minus the three staged Windows exclusions equals 54.
- Confirmed the CMake staged gates remain `if(Threads_FOUND AND NOT WIN32)`
  and `if(NOT WIN32 AND NOT MSVC)`.
- Removed the temporary local `build-sprint134-day11` CMake audit directory.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over touched workflow, support docs, and
  Sprint 134 planning paths; it passed.
- Confirmed no `.c`, `.h`, or `CMakeLists.txt` files changed, so the full C
  quality gate was not required and CTest membership remained unchanged.

## Day 12 Notes

- Wrote the support-tier documentation alignment artifact.
- Re-reviewed the implemented Sprint 134 platform decisions from Days 4, 7, 9,
  and 11.
- Updated the live Sprint 134 baseline in this file so Windows includes the
  supplemental CMake install/downstream confidence job alongside the reviewed
  MSVC CMake-first subset.
- Updated the input and candidate-surface tables in this file so
  `.github/workflows/windows-ci.yml` includes the supplemental
  install/downstream job, expected 54 CTests, and staged-exclusion output.
- Updated the Sprint 134 scope-boundary bullets in this file from
  pre-decision install-lane language to final post-decision claim fences.
- Confirmed `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`,
  `.github/workflows/windows-ci.yml`, `README.md`, `INSTALL.md`, and
  `docs/maintainer_guide.md` align on reviewed, supplemental, staged,
  deferred, and unsupported support tiers.
- Left older Day 1 and Day 2 artifacts unchanged as historical pre-decision
  snapshots; the Day 12 artifact records the current Sprint 134 platform
  truth.
- Ran YAML parsing for `.github/workflows/ci.yml`,
  `.github/workflows/macos-ci.yml`, and `.github/workflows/windows-ci.yml`;
  all passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over workflow, support-doc, and Sprint 134
  planning paths; it passed.
- No `.c`, `.h`, workflow behavior, package script, or `CMakeLists.txt` files
  changed on Day 12, so the full C quality gate was not required and CTest
  membership remained unchanged.

## Day 13 Notes

- Wrote the integrated platform validation artifact.
- Confirmed the diff surface is workflows, support docs, and Sprint 134
  planning artifacts only.
- Confirmed no `.c`, `.h`, or `CMakeLists.txt` changes.
- Ran YAML parsing for `.github/workflows/ci.yml`,
  `.github/workflows/macos-ci.yml`, and `.github/workflows/windows-ci.yml`;
  all passed.
- Ran shell syntax checks for `tests/test_install.sh`,
  `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh`; all passed.
- Ran `bash tests/test_install.sh`; it passed 22 checks with 0 failures.
- Ran `bash tests/test_cmake_install.sh`; it passed 21 checks with 0 failures
  and 0 skips.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran a local CMake configure plus `ctest -N` registration audit; this
  non-Windows host registered 57 tests.
- Confirmed the Windows CTest count reconciliation remains 57 total
  non-Windows tests minus the three staged Windows exclusions equals 54.
- Confirmed this host has neither `actionlint` nor `pwsh`; Windows PowerShell
  job execution remains hosted-runner proof.
- Removed the temporary local `build-sprint134-day13` CMake audit directory.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over workflow, support-doc, and Sprint 134
  planning paths; it passed.
- Ran a support-claim drift scan; hits were explicit non-claims or historical
  count notes, not positive support overclaims.
- The full C quality gate was not required because no `.c` or `.h` files
  changed.

## Day 14 Notes

- Wrote the platform tier closeout and handoff artifact.
- Published the final Linux, macOS, and Windows support-tier truth table.
- Published the final Windows staged-exclusion register with owners, blockers,
  and promotion gates for `test_threads`, `test_sprint4_integration`, and
  `test_fuzz`.
- Recorded residual install/platform queues for hosted macOS supplemental
  package evidence, hosted Windows supplemental package evidence, reviewed
  macOS install/export parity, reviewed Windows install-validation parity,
  Windows staged test promotion, package-manager support, shared-library
  packaging, dynamic ABI compatibility, and runtime-loader behavior.
- Added PR review summary material and Sprint 135 handoff notes.
- Reordered the Day sections in this file chronologically while preserving
  their content.
- Ran YAML parsing for `.github/workflows/ci.yml`,
  `.github/workflows/macos-ci.yml`, and `.github/workflows/windows-ci.yml`;
  all passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over workflow, support-doc, and Sprint 134
  planning paths; it passed.
- Ran the final support-claim scan; hits were explicit non-claims or
  historical count notes, not positive support overclaims.
- Confirmed Sprint 134 did not modify `.c`, `.h`, or `CMakeLists.txt`, so the
  full C quality gate was not required.

## Post-PR CI Follow-Up

- Fixed `tests/test_install.sh` after the Linux reviewed static-first package
  contract lane reported `pkg-config --cflags` as
  `-I/tmp/sparse.../usr/include ` with a trailing space.
- Changed the `pkg-config --cflags` check to parse shell tokens, require
  exactly one `-I...` token, and compare that path to the installed include
  directory with `-ef`.
- This keeps the semantic check strict while accepting harmless
  `pkg-config` output whitespace on hosted Linux.
- Ran `bash -n tests/test_install.sh`; it passed.
- Ran `bash tests/test_install.sh`; it passed 22 checks with 0 failures.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
