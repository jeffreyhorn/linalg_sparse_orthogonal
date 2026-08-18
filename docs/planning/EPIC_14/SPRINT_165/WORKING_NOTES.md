# Sprint 165 Working Notes

## Sprint Goal

Harden the static-first package boundary so shared-library, dynamic ABI,
runtime-loader, and package-manager non-claims cannot drift.

## Source Artifact Note

The Sprint 165 request referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but
the current merged planning source for Sprint 165 is
`docs/planning/EPIC_14/PROJECT_PLAN.md` in the section "Sprint 165:
Static-First Package Boundary Hardening".

## Branch Baseline

- Branch: `sprint-165`
- Starting point: current `master` after Sprint 164 merge.
- Sprint 162 handoff: Windows package parity remains CMake-first; Windows
  Makefile install/uninstall parity and Windows `pkg-config` execution parity
  remain unsupported unless a future product decision adds provider-specific
  proof.
- Sprint 163 handoff: methodology-bound performance publication must not be
  reused as package, ABI, runtime-loader, shared-library, package-manager,
  broad platform, portable performance, or state-of-the-art proof.
- Sprint 164 handoff: public-header/API cleanup status is known, and package
  wording must remain aligned with the static-first non-claim boundary.

## Current Package And Install Surfaces

| Surface | Current Role | Sprint 165 Implication |
| --- | --- | --- |
| `CMakeLists.txt` | Defines the static `sparse_lu_ortho` target, rejects `BUILD_SHARED_LIBS=ON`, installs headers, generated version header, CMake package files, and `sparse.pc`. | Primary static deferral guard and CMake install/export owner. |
| `cmake/SparseConfig.cmake.in` | Minimal installed CMake package config that imports `SparseTargets.cmake`. | Metadata audit surface for accidental shared/dynamic claims. |
| `sparse.pc.in` | Source template for installed `sparse.pc` metadata. | Primary pkg-config metadata wording and link flag owner. |
| `Makefile` install targets | Installs `libsparse_lu_ortho.a`, checked-in public headers, generated `sparse_version.h`, and `sparse.pc`; removes them on uninstall. | Unix-side static archive install/uninstall owner. |
| `tests/test_install.sh` | Local Unix proof for Make install/uninstall plus `pkg-config` downstream compile/link/run. | Validation owner for Make install and `pkg-config` package behavior. |
| `tests/test_cmake_install.sh` | Local Unix proof for CMake install/export plus installed `find_package(Sparse)` consumers. | Validation owner for CMake package metadata and downstream CMake behavior. |
| `scripts/static_package_deferral_check.sh` | Static package-contract guard for shared-library deferral and metadata non-claims. | Static-first drift guard owner. |
| `examples/cmake_example/` | Maintained installed CMake consumer example. | Downstream proof and user-facing example owner. |
| `.github/workflows/ci.yml` | Linux reviewed source of truth including static-first package-contract lane. | Hosted Linux package validation owner. |
| `.github/workflows/macos-ci.yml` | macOS reviewed static-first Make install/pkg-config and CMake install/export proof. | Hosted macOS package validation owner. |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake install/downstream validation and metadata-only `sparse.pc` inspection. | Hosted Windows CMake-first package owner; does not imply Make/pkg-config parity. |
| `README.md` | Short first-use install summary and package non-claims. | Public package front-door wording must stay concise and bounded. |
| `INSTALL.md` | Operational install, downstream consumer, validation, and support split owner. | Primary public package-boundary documentation owner. |
| `docs/maintainer_guide.md` | Maintainer package/ABI contract and validation interpretation owner. | Primary maintainer policy and proof-owner interpretation owner. |

## Explicit Non-Goals

Sprint 165 does not claim or attempt to deliver:

- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- Linux SONAME policy;
- macOS install-name or RPATH policy;
- Windows DLL/import-library behavior;
- package-manager distribution through Homebrew, apt, dnf, pacman, vcpkg,
  Conan, or similar systems;
- static/shared selector UX beyond fail-closed shared deferral;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- broad platform package parity;
- performance or backend superiority;
- state-of-the-art sparse linear algebra claims from package evidence.

## Working Assumptions

- Day 1 is planning and inventory only; no package metadata, source, header,
  script, workflow, or user-facing docs are edited today except Sprint 165
  planning artifacts.
- `INSTALL.md` owns operational package instructions; `README.md` should stay
  short and route users to `INSTALL.md` for detail.
- `docs/maintainer_guide.md` owns proof interpretation and package/ABI policy.
- `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh` are the focused local validation
  owners for package-boundary changes.
- Windows package proof remains CMake-first unless a future reviewed change
  adds Windows Makefile and `pkg-config` execution parity.
- If `.c` or `.h` files are changed during later days, run
  `make format && make lint && make test`.

## Stop Conditions

Stop and revise before proceeding if a change:

- silently accepts `BUILD_SHARED_LIBS=ON`;
- installs, exports, documents, or validates shared-library artifacts as
  supported;
- introduces `Libs.private`, static/shared selectors, loader metadata, SONAME,
  install-name/RPATH, DLL/import-library, or package-manager wording without a
  reviewed product decision;
- describes exact package version metadata as a dynamic ABI guarantee;
- treats Windows `sparse.pc` metadata inspection as Windows `pkg-config`
  execution parity;
- treats Windows CMake install proof as Windows Makefile parity;
- cites benchmark, sentinel, hosted-report, or API-header evidence as package
  or ABI proof;
- changes package metadata or install behavior without a focused install/export
  validation record;
- modifies `.c` or `.h` files without running the required full quality gate;
- leaves generated build, install, docs, coverage, or cache output in the
  worktree for commit.

## Daily Log

### Day 1: Sprint Intake And Package Surface Inventory

- Re-read the Sprint 165 section of
  `docs/planning/EPIC_14/PROJECT_PLAN.md` and recorded the prompt path
  mismatch as a source artifact note.
- Reviewed package and ABI handoff context from Sprints 162, 163, and 164.
- Created Sprint 165 working notes and artifact directory structure.
- Inventoried package/install owners across CMake, Makefile, pkg-config,
  install validation scripts, downstream examples, CI workflows, README,
  INSTALL, and maintainer guide.
- Recorded explicit non-goals, working assumptions, and stop conditions for
  shared-library, ABI, runtime-loader, package-manager, and platform parity
  drift.
- Created `artifacts/day1-sprint-intake.md`.

### Day 2: Package Metadata Audit

- Inspected current package metadata and validation owners:
  - `CMakeLists.txt`
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`
  - `Makefile` install/uninstall section
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `scripts/static_package_deferral_check.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
- Classified current CMake and pkg-config metadata behavior as source-backed:
  explicit static CMake target, archive-only CMake install destination,
  exact-version CMake package compatibility, static archive `sparse.pc`
  description, and installed downstream consumer validation.
- Confirmed validation coverage for:
  - no installed shared-library artifacts;
  - static CMake imported target metadata;
  - absent shared imported metadata and loader/static-shared selector metadata;
  - installed prefix include/archive paths and source/build path leak checks;
  - `sparse.pc` field checks, absent unsupported package/ABI wording, exact
    version resolution, and downstream compile/link/run consumers;
  - Windows CMake-first metadata-only `sparse.pc` inspection with explicit
    non-claims for Makefile and `pkg-config` execution parity.
- Recorded Day 3 guard-design focus areas:
  - keep `BUILD_SHARED_LIBS=ON` rejection wording and validation centralized;
  - verify CMake install destinations remain archive-only;
  - preserve exact-version metadata as package metadata, not dynamic ABI proof;
  - keep Windows package lane CMake-first and metadata-only for `sparse.pc`.
- Created `artifacts/day2-package-metadata-audit.md`.

### Day 3: Static Deferral Guard Design

- Reviewed current `BUILD_SHARED_LIBS=ON` behavior in `CMakeLists.txt` and
  `scripts/static_package_deferral_check.sh`.
- Defined the static deferral guard contract:
  - `BUILD_SHARED_LIBS=ON` must fail at configure time;
  - the failure text must name the rejected input, static archive package
    surface, shared-library packaging deferral, dynamic ABI deferral,
    export/import blocker, symbol visibility blocker, dynamic ABI policy
    blocker, Linux SONAME blocker, macOS install-name/RPATH blocker, Windows
    DLL/import-library blocker, installed shared consumer proof blocker, and
    runtime-loader validation blocker;
  - `sparse_lu_ortho` must remain an explicit `STATIC` CMake target;
  - CMake install metadata must remain archive-only until shared support is
    selected;
  - installed CMake package metadata must not contain shared imported targets,
    loader metadata, static/shared selectors, or source/build path leaks;
  - `sparse.pc` must remain static archive scoped and must not carry
    `Libs.private`, shared/static selectors, loader wording, package-manager
    wording, or ABI claims.
- Identified Day 4 implementation candidates:
  - keep `scripts/static_package_deferral_check.sh` as the central local guard;
  - add or refine checks only if they close an explicit absence-rule gap;
  - avoid adding policy prose to `cmake/SparseConfig.cmake.in`;
  - preserve Windows CMake-first and metadata-only `sparse.pc` boundaries.
- Defined local and hosted validation commands for static guard behavior.
- Created `artifacts/day3-static-deferral-guard-design.md`.

### Day 4: Static Deferral Guard Implementation

- Compared `scripts/static_package_deferral_check.sh` against the Day 3 public
  header/export macro absence rules.
- Found one source-level guard gap: the script blocked `SPARSE_API`,
  `SPARSE_EXPORT`, and `SPARSE_IMPORT`, but did not yet block
  `SPARSE_SHARED`, `SPARSE_STATIC`, or `SPARSE_ABI` in public headers.
- Updated `scripts/static_package_deferral_check.sh` so
  `check_no_export_or_abi_metadata` rejects all six public export/import and
  static/shared ABI macro names before a shared-library product decision.
- Left `CMakeLists.txt`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in`,
  install scripts, CI workflows, README, INSTALL, and maintainer guide
  behavior unchanged because Day 3 did not identify a source-behavior or
  metadata wording violation there.
- Ran focused package validation:
  - `bash scripts/static_package_deferral_check.sh` passed;
  - `bash tests/test_install.sh` passed with 23 passes, 0 failures;
  - `bash tests/test_cmake_install.sh` passed with 27 passes, 0 failures,
    0 skips.
- Created `artifacts/day4-static-deferral-guard-implementation.md`.

### Day 5: ABI Non-Claim Audit

- Scanned public headers, version/package docs, README, INSTALL, maintainer
  guide, API reference, tutorial, cookbook, solver-selection docs,
  `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, and `sparse.pc.in` for ABI,
  binary compatibility, layout, SONAME, loader, shared-library, and
  package-manager wording.
- Separated the supported source/package facts from unsupported binary ABI
  claims:
  - source API declarations and zero-initialized option structs are public API
    contracts;
  - `VERSION`, generated `sparse_version.h`, `SparseConfigVersion.cmake`, and
    `sparse.pc` version fields are package metadata;
  - dynamic ABI compatibility, stable binary struct layout, SONAME/install
    name/RPATH/DLL policies, and downstream binary compatibility remain
    non-claims.
- Identified one public-header wording candidate for Day 6 cleanup:
  `include/sparse_cholesky.h` uses "**ABI break in v2.0.0**" while newer
  layout warnings in `include/sparse_ldlt.h` and `include/sparse_eigs.h` use
  the clearer "**Source rebuild required ... layout**" pattern.
- Confirmed package docs already state that exact package version metadata is
  not a dynamic ABI guarantee.
- Drafted replacement wording for unsupported or ambiguous ABI-adjacent
  phrases and recorded deferred ABI product decisions.
- Created `artifacts/day5-abi-non-claim-audit.md`.

### Day 6: ABI And Package Wording Cleanup

- Applied the Day 5 public-header cleanup in `include/sparse_cholesky.h`:
  replaced the older "**ABI break in v2.0.0**" warning with the
  "**Source rebuild required for v2.0.0 options layout**" wording pattern
  already used by newer public option/result structs.
- Preserved the documented source-initializer behavior while removing wording
  that could imply a maintained dynamic ABI policy.
- Rechecked README, INSTALL, maintainer guide, `CMakeLists.txt`, and package
  metadata wording. Existing exact-version package metadata language already
  states that it is not a dynamic ABI guarantee, so no docs were changed.
- Confirmed the remaining ABI/package wording hits are explicit non-claims in
  README, CMake comments, and maintainer guidance.
- Ran validation:
  - `make format && make lint && make test` passed;
  - `bash scripts/static_package_deferral_check.sh` passed;
  - `git diff --check` passed;
  - targeted ABI wording scan passed with only expected non-claim hits.
- Created `artifacts/day6-abi-package-wording-cleanup.md`.

### Day 7: Downstream Proof Scope Refresh

- Reviewed the downstream install proof owners:
  - `tests/test_install.sh`;
  - `tests/test_cmake_install.sh`;
  - `examples/cmake_example/CMakeLists.txt`;
  - `examples/cmake_example/main.c`;
  - `.github/workflows/ci.yml`;
  - `.github/workflows/macos-ci.yml`;
  - `.github/workflows/windows-ci.yml`;
  - `INSTALL.md`;
  - `README.md`;
  - `docs/maintainer_guide.md`.
- Defined the downstream proof requirements for static archive presence,
  absent shared artifacts, installed header/version metadata, exact-version
  package handling, installed include/archive paths, downstream compile/link/run
  consumers, metadata non-claims, and uninstall cleanup.
- Separated supported proof lanes:
  - Unix Make install plus `pkg-config` command execution;
  - Unix CMake install/export plus `find_package(Sparse)`;
  - macOS reviewed execution of the same Unix proof scripts;
  - Windows CMake-first install/downstream validation;
  - Windows metadata-only `sparse.pc` inspection.
- Recorded deferred boundaries for Windows Makefile install/uninstall parity,
  Windows `pkg-config` execution parity, shared-library support, runtime-loader
  behavior, package-manager distribution, and dynamic ABI support.
- Identified stale expectation risks:
  - raw string comparisons for temp paths can fail on double-slash path
    spelling even when filesystem paths are equivalent;
  - one-line output assumptions can fail when examples print valid multi-line
    output;
  - hard-coded Windows CTest counts must be updated whenever reviewed CMake
    tests are added or removed;
  - installed header-count expectations must include generated
    `sparse_version.h`;
  - Windows `sparse.pc` inspection must not be described as `pkg-config`
    command execution.
- Ran validation:
  - `bash tests/test_install.sh` passed with 23 passes, 0 failures;
  - `bash tests/test_cmake_install.sh` passed with 27 passes, 0 failures,
    0 skips.
- Created `artifacts/day7-downstream-proof-scope.md`.

### Day 8: Downstream Proof Implementation

- Implemented the concrete Day 7 raw-path stale-expectation fix in
  `tests/test_install.sh`.
- Added a `same_dir` helper that requires both paths to exist as directories
  and compares them with filesystem identity.
- Updated `pkg-config` `prefix`, `libdir`, `includedir`, `--cflags`, and
  `--libs` path checks to use `same_dir` instead of raw string equality or
  duplicated `-ef` checks.
- Left downstream examples unchanged because their output checks already match
  semantic tokens rather than a single-line raw output string.
- Left Windows workflow expectations unchanged because the current reviewed
  CMake CTest count is already recorded as `EXPECTED_WINDOWS_CTEST_COUNT: "59"`
  and the Windows install lane already describes `sparse.pc` validation as
  metadata-only inspection.
- Ran validation:
  - `bash -n tests/test_install.sh tests/test_cmake_install.sh
    scripts/static_package_deferral_check.sh` passed;
  - `bash tests/test_install.sh` passed with 23 passes, 0 failures;
  - `bash tests/test_cmake_install.sh` passed with 27 passes, 0 failures,
    0 skips;
  - `bash scripts/static_package_deferral_check.sh` passed.
- Created `artifacts/day8-downstream-proof-implementation.md`.

### Day 9: Package Documentation Alignment

- Aligned public and maintainer documentation with the Day 7-8 downstream
  proof boundaries.
- Updated `README.md` to state that Unix install proof validates installed
  include and library paths by filesystem identity, so staged-prefix spelling
  differences do not masquerade as package failures.
- Updated `INSTALL.md` to:
  - document filesystem-identity validation for `pkg-config` prefix, libdir,
    includedir, cflags, and libs path checks;
  - state that both focused install scripts reject unsupported shared-library,
    loader, package-manager, or dynamic ABI wording in their installed package
    files;
  - clarify again that Windows `sparse.pc` checks are metadata-only inspection,
    not `pkg-config` command execution.
- Updated `docs/maintainer_guide.md` to reflect the Day 8 `same_dir` path
  validation and semantic output checks in the `tests/test_install.sh` proof
  owner row.
- Updated the `CMakeLists.txt` `sparse.pc` comment to tie generated flags to
  semantic install tests.
- Verified package-support cross-link wording in tutorial, cookbook,
  solver-selection, API reference, README, INSTALL, and maintainer guide.
  Tutorial, cookbook, and solver-selection docs did not introduce package
  support claims; API reference and install docs keep explicit non-claims.
- Ran validation:
  - `bash scripts/static_package_deferral_check.sh` passed;
  - cross-link package/support wording scan found only expected non-claims;
  - `git diff --check` passed.
- Created `artifacts/day9-package-documentation-alignment.md`.

### Day 10: Package Metadata Validation

- Generated installed package metadata through a temporary CMake install outside
  the repository and inspected the installed files directly.
- Confirmed installed metadata shape:
  - static archive present at the installed library directory;
  - no `.so`, `.so.*`, `.dylib`, or `.dll` artifacts installed;
  - 19 installed headers present;
  - `Sparse::sparse_lu_ortho` exported as `STATIC IMPORTED`;
  - CMake include and archive metadata use `_IMPORT_PREFIX`;
  - installed CMake package files contain no source-tree or build-tree path
    leaks;
  - installed CMake package files contain no shared imported target, loader,
    SONAME, install-name/RPATH, DLL/import-library, runtime, or static/shared
    selector metadata;
  - installed `sparse.pc` contains `Name: sparse`,
    `Description: Static archive package metadata for sparse linear algebra`,
    `Version: 2.2.0`, `Cflags: -I${includedir}`, and
    `Libs: -L${libdir} -lsparse_lu_ortho -lm`;
  - installed `sparse.pc` contains no `Libs.private`, shared-library, SONAME,
    dynamic-library, ABI, or package-manager wording.
- Recorded local environment constraints:
  - local validation ran on macOS with Unix shell/CMake/pkg-config tools;
  - Windows `.lib` install metadata and Windows `sparse.pc` metadata-only
    inspection remain hosted CI proof, not locally executed in this run;
  - Windows Makefile install/uninstall parity and Windows `pkg-config` command
    execution parity remain deferred non-claims.
- Ran validation:
  - temporary CMake install metadata inspection passed;
  - `bash scripts/static_package_deferral_check.sh` passed;
  - `bash tests/test_install.sh` passed with 23 passes, 0 failures;
  - `bash tests/test_cmake_install.sh` passed with 27 passes, 0 failures,
    0 skips.
- Created `artifacts/day10-package-metadata-validation.md`.

### Day 11: Install And Downstream Validation

- Ran the maintained local install/downstream validation scripts on the
  available macOS Unix toolchain.
- Confirmed Make install/`pkg-config` downstream proof:
  - static archive installed;
  - no shared-library artifacts installed;
  - all 19 headers installed;
  - `sparse.pc` installed and resolved by `pkg-config`;
  - exact `pkg-config` version constraint passed for `2.2.0`;
  - prefix, libdir, includedir, cflags, and libs point at installed paths;
  - `--static` libs match the current self-contained link flags;
  - no `Libs.private` stanza or unsupported package/ABI wording exists;
  - generated basic consumer compiled, linked, and ran;
  - maintained example source compiled, linked, and ran through `pkg-config`;
  - uninstall removed the library, headers, and `sparse.pc`.
- Confirmed CMake install/export downstream proof:
  - CMake configure/build/install succeeded;
  - static archive, headers, CMake package files, and `sparse.pc` installed;
  - CMake target metadata remained static and install-prefix based;
  - no shared imported metadata, loader/static-shared selector metadata, or
    source/build path leaks appeared;
  - maintained CMake example configured, built, and ran through
    `find_package(Sparse)`;
  - exact-version CMake consumer configured, built, and ran;
  - lower same-major mismatched CMake package version was rejected.
- Recorded hosted-only boundaries:
  - Windows CMake-first install/downstream validation remains hosted CI proof;
  - Windows `sparse.pc` validation remains metadata-only inspection;
  - Windows Makefile install/uninstall parity and Windows `pkg-config` command
    execution parity remain deferred.
- Ran validation:
  - `bash tests/test_install.sh` passed with 23 passes, 0 failures;
  - `bash tests/test_cmake_install.sh` passed with 27 passes, 0 failures,
    0 skips;
  - `bash scripts/static_package_deferral_check.sh` passed.
- Created `artifacts/day11-install-downstream-validation.md`.

### Day 12: Quality Gate

- Ran the required full quality gate because the accumulated Sprint 165 change
  set includes a public header:
  - `make format && make lint && make test` passed;
  - `make test` ended with `All tests passed.`;
  - the only clang-tidy warning output was the existing suppressed
    non-user-code/NOLINT analyzer chatter.
- Re-ran focused package checks after the full gate:
  - `bash scripts/static_package_deferral_check.sh` passed;
  - `bash tests/test_install.sh` passed with 23 passes, 0 failures;
  - `bash tests/test_cmake_install.sh` passed with 27 passes, 0 failures,
    0 skips;
  - `python3 scripts/normalize_report_index.py --family package --check`
    passed with 6 rows;
  - `python3 scripts/normalize_report_index.py --family package
    --check-freshness` passed with 6 source-controlled advisory rows.
- Scanned changed files and Sprint 165 artifacts for unsupported
  shared-library, dynamic ABI, runtime-loader, package-manager, Windows
  Makefile, Windows `pkg-config`, and static/shared selector wording.
  Matches were expected guard text, metadata rejection checks, explicit
  non-claims, or sprint evidence; no unsupported support claims were found.
- Ran `git diff --check`; it passed.
- Created `artifacts/day12-quality-gate-drift-scan.md`.

### Day 13: Evidence Review And Residual Register

- Reviewed Day 1-12 artifacts against the Sprint 165 project-plan items and
  deliverables.
- Confirmed evidence exists for:
  - package metadata audit and static package surface inventory;
  - strengthened static deferral guard behavior;
  - ABI/package non-claim audit and wording cleanup;
  - Make install/`pkg-config` and CMake install/export downstream proof;
  - README, INSTALL, maintainer-guide, CMake, and `sparse.pc` documentation
    alignment;
  - full quality gate, focused package checks, package report-index checks,
    and claim-drift scan.
- Built the Sprint 165 residual register for true shared-library support,
  dynamic ABI policy, package-manager distribution, runtime-loader behavior,
  and Windows Make/`pkg-config` parity.
- Identified Sprint 166 handoff items for final evidence inventory, hosted CI
  reconciliation, claim audit, project-plan reconciliation, Epic 14
  retrospective, and residual queue publication.
- Created `artifacts/day13-evidence-review-residual-register.md`.

### Day 14: Closeout And Handoff

- Closed Sprint 165 with a final changed-file, validation, residual, PR, and
  review-risk summary.
- Final tracked changed files for Sprint 165:
  - `CMakeLists.txt`
  - `INSTALL.md`
  - `README.md`
  - `docs/maintainer_guide.md`
  - `include/sparse_cholesky.h`
  - `scripts/static_package_deferral_check.sh`
  - `tests/test_install.sh`
- Final Sprint 165 planning artifacts live under
  `docs/planning/EPIC_14/SPRINT_165/`.
- Verified stale Epic 12 references inside the Sprint 165 folder:
  - `PLAN.md` and `WORKING_NOTES.md` carry explicit source-note explanations;
  - Day 1 records the same source-plan correction;
  - no unexplained stale Epic 12 artifact path was found.
- Confirmed plan, artifacts, and working notes are internally consistent:
  - Day 1-14 entries exist in `WORKING_NOTES.md`;
  - Day 1-14 artifacts exist in `artifacts/`;
  - Day 13 maps Sprint 165 deliverables to evidence;
  - Day 14 records closeout and Sprint 166 handoff.
- Validation evidence carried into closeout:
  - Day 12 `make format && make lint && make test` passed;
  - Day 12 `bash scripts/static_package_deferral_check.sh` passed;
  - Day 12 `bash tests/test_install.sh` passed with 23 passes, 0 failures;
  - Day 12 `bash tests/test_cmake_install.sh` passed with 27 passes,
    0 failures, 0 skips;
  - Day 12 package report-index check and freshness check passed;
  - Day 14 `git diff --check` passed.
- Residuals remain explicit product decisions:
  - true shared-library support;
  - dynamic ABI policy;
  - runtime-loader behavior;
  - package-manager distribution;
  - static/shared selector UX;
  - Windows Makefile install/uninstall parity;
  - Windows `pkg-config` command execution parity;
  - broader platform package parity.
- Created `artifacts/day14-closeout-and-handoff.md`.
