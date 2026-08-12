# Sprint 153 Working Notes

## Goal

Sprint 153 makes a product-level shared-library ABI decision and either
implements the first supported shared surface or publishes a stronger
static-first deferral with exact blockers.

## Starting Evidence

- Sprint 149 package parity decision is available through the maintained
  static-first package and install proof surfaces.
- Sprint 152 created the ABI/package handoff at
  `docs/planning/EPIC_13/SPRINT_152/artifacts/sprint153-abi-package-handoff.md`.
- Current `CMakeLists.txt` rejects `BUILD_SHARED_LIBS=ON` at configure time so
  shared-library requests are not silently treated as supported.
- Current Make and CMake install paths install static archives, public headers,
  CMake package metadata, and `sparse.pc`.
- Current install tests assert no shared-library artifacts are installed.
- Current documentation states shared-library packaging, dynamic ABI
  compatibility, runtime-loader behavior, static/shared selectors, package
  manager distribution, Windows Makefile parity, and Windows `pkg-config`
  parity are out of scope.

## Item-To-Day Owner Map

| Sprint 153 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: ABI Surface Audit | Days 1-2 | Day 1 records baseline scope, Day 2 inventories public ABI-relevant surface. |
| Item 2: Platform Loader Audit | Day 3 | Day 3 records Linux, macOS, Windows loader and export/import requirements. |
| Item 3: Product Decision | Days 4-5 | Day 4 defines decision criteria, Day 5 records the selected product decision. |
| Item 4: Selected Implementation | Days 6-7 | Day 6 designs build/install changes, Day 7 implements the selected path. |
| Item 5: Downstream Proof | Days 8-10 | Day 8 designs consumer proof, Day 9 implements it, Day 10 decides CI policy. |
| Item 6: Documentation Alignment | Day 11 | Day 11 aligns docs, metadata, comments, and stale wording. |
| Item 7: Validation And Closeout | Days 12-14 | Day 12 runs integrated validation, Day 13 runs quality/residual review, Day 14 closes with Sprint 154 handoff. |

## Stop Conditions

- A local static archive install is described as shared-library or dynamic ABI
  support.
- `BUILD_SHARED_LIBS=ON` is allowed without a product decision, exported target
  semantics, install rules, downstream consumer proof, and loader validation.
- A `.so`, `.dylib`, `.dll`, import library, SONAME, install name, or loader
  behavior is claimed without platform-specific proof.
- CMake package metadata, `sparse.pc`, or install docs imply package-manager
  availability, shared/static selectors, or dynamic-loader support before the
  selected decision is implemented and validated.
- Windows Makefile parity or Windows `pkg-config` execution parity is inferred
  from the reviewed Windows CMake install/downstream lane.
- Sprint 152 local oracle freshness is cited as package, ABI, loader, hosted
  CI, platform, or release evidence.
- Source-controlled package rows are treated as generated freshness proof.
- Shared-library support is implemented without symbol visibility, public
  header, allocator/lifetime, callback, and version metadata review.
- Static-first deferral is left vague after the product decision; blockers must
  be exact and test-backed.
- Required install/package, report-index, or C quality gates fail after
  implementation changes.

## Daily Log

### Day 1: Sprint Intake And ABI Baseline

- Re-read the Sprint 153 section of
  `docs/planning/EPIC_13/PROJECT_PLAN.md`.
- Reviewed the Sprint 152 ABI/package handoff and non-claims.
- Created the Sprint 153 artifact directory and Day 1 ABI baseline artifact.
- Inventoried current package/install surfaces:
  - Make install installs `$(LIB)`, public headers, generated
    `sparse_version.h`, and `sparse.pc`;
  - CMake install exports `Sparse::sparse_lu_ortho`, public headers,
    generated `sparse_version.h`, CMake package metadata, version metadata,
    and `sparse.pc`;
  - `sparse.pc.in` describes static archive package metadata and link flags;
  - `cmake/SparseConfig.cmake.in` imports `SparseTargets.cmake`;
  - `tests/test_install.sh` and `tests/test_cmake_install.sh` assert no
    shared-library artifacts are installed;
  - Linux, macOS, and Windows CI lanes preserve the static-first package
    interpretation.
- Inventoried installed public headers under `include/`: `18` source headers
  plus generated `sparse_version.h` from `include/sparse_version.h.in`, for
  `19` installed headers.
- Captured current shared-library deferral behavior: `CMakeLists.txt` rejects
  `BUILD_SHARED_LIBS=ON`, the library target is `STATIC`, install rules use
  `ARCHIVE DESTINATION`, and docs state shared-library packaging and dynamic
  ABI support are deferred.
- Captured stop conditions for accidental shared-library, ABI, loader,
  package-manager, platform, release, or generated-oracle evidence claims.
- Day 2 handoff: audit installed public headers, public symbols, public data
  layouts, allocator/lifetime behavior, callback contracts, version metadata,
  internal symbols, and accidental export risks.

### Day 2: Public ABI Surface Audit

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day2-public-abi-surface-audit.md`.
- Inventoried the installed public header surface:
  - `18` source headers under `include/`;
  - generated `sparse_version.h` from `include/sparse_version.h.in`;
  - `19` installed headers total.
- Classified every installed public header for ABI relevance and owner
  candidates, covering the core matrix API, direct solvers, QR, SVD,
  iterative solvers, eigensolvers, preconditioners, analysis/refactor,
  reordering, dense helpers, vector helpers, and version metadata.
- Identified ABI-sensitive concrete public layouts, including CSR/CSC,
  CSR-LU, dense matrices, solver options/results/factors, analysis/factor
  statistics, iterative/eigensolver handles, and callback typedefs.
- Recorded ownership and lifetime contracts:
  - `SparseMatrix` values use `sparse_free`;
  - factors, handles, CSR/CSC values, dense helpers, and analysis artifacts use
    type-specific free APIs;
  - callback signatures, caller-owned array returns, and allocator boundaries
    need explicit ABI treatment before shared-library support.
- Compared installed headers against internal headers under `src/`; installed
  header boundaries are clear, but compiled-object export boundaries are not
  shared-library-ready.
- Identified static/process state that would need loader governance under a
  shared build: thread-local error state, graph/reorder override state, dense
  backend handles/probes, LDLT provider state, and default option objects.
- Identified accidental export risks from non-static internal helpers,
  including allocator, factor-state, workspace, QR householder, graph/reorder,
  dense backend, symbolic analysis, and etree implementation symbols.
- Day 3 handoff: audit Linux, macOS, and Windows loader requirements,
  especially SONAME/install-name/import-library behavior, export filtering,
  downstream link proof, and MSVC import/export decoration.

### Day 3: Platform Loader Audit

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day3-platform-loader-audit.md`.
- Audited current loader claim status:
  - `CMakeLists.txt` still declares `sparse_lu_ortho` as a static target;
  - `BUILD_SHARED_LIBS=ON` remains a configure-time rejection;
  - Make and CMake install paths install static archive package artifacts;
  - install proofs reject `.so`, `.so.*`, `.dylib`, and `.dll` artifacts;
  - CMake package metadata is checked as `STATIC IMPORTED`;
  - `sparse.pc` is checked as static archive package metadata.
- Captured Linux shared-library requirements: ELF `.so` artifact,
  SONAME/version policy, PIC, exported-symbol filtering, dependency metadata,
  `pkg-config` shared/static semantics, CMake imported target metadata,
  downstream installed consumer proof, and `readelf`/`nm`/`objdump`
  inspection.
- Captured macOS shared-library requirements: Mach-O `.dylib` artifact,
  install-name and RPATH policy, exported-symbol filtering, Homebrew/OpenMP
  runtime dependency behavior, installed consumer proof, and `otool`/`nm`
  inspection.
- Captured Windows shared-library requirements: DLL plus import library,
  `SPARSE_API` and `__declspec(dllexport/dllimport)` policy, allocator/C
  runtime boundary review, runtime DLL lookup policy, CMake
  `IMPORTED_IMPLIB`/`IMPORTED_LOCATION` metadata, and MSVC installed consumer
  runtime proof.
- Built a platform proof matrix for shared artifact build/install,
  exported-symbol curation, installed CMake consumers, installed
  `pkg-config` consumers, loader metadata, and ABI compatibility governance.
- Recorded that current CI lanes prove static-first package contracts only:
  Linux and macOS have reviewed static package proof; Windows has reviewed
  CMake-first static install/downstream proof and still does not claim
  Makefile parity, Windows `pkg-config` execution parity, shared-library
  support, dynamic ABI support, or runtime-loader behavior.
- Day 4 handoff: convert Days 2-3 findings into shared-support and
  static-deferral decision criteria with explicit rollback triggers for
  missing export, loader, metadata, ABI, or downstream proof.

### Day 4: Product Decision Criteria

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day4-product-decision-criteria.md`.
- Converted Day 2 and Day 3 findings into two product paths:
  - implement a first supported shared-library surface in Sprint 153;
  - keep static-first packaging and strengthen the deferral with exact,
    test-backed blockers.
- Defined minimum viable shared-library support gates for scope, build,
  install, exported symbols, public headers, ABI policy, CMake package
  metadata, `pkg-config` semantics, runtime loader behavior, CI ownership, and
  documentation alignment.
- Defined minimum viable static-first deferral gates for
  `BUILD_SHARED_LIBS=ON` rejection, no shared artifacts, no unsupported
  package metadata claims, exact blocker wording, platform evidence
  boundaries, and Sprint 154 handoff clarity.
- Built a scorecard across feasibility, ABI risk, platform coverage clarity,
  test cost, documentation cost, user value, release confidence, and
  maintenance burden.
- Recorded the Day 4 decision bias: static-first deferral is the safer Day 5
  default unless shared support is deliberately narrowed to a platform scope
  small enough to finish with complete proof.
- Defined rollback rules for incomplete export allowlists, static/build-tree
  consumer leakage, vague platform splits, unproven Windows DLL behavior,
  untested package selectors, undocumented ABI compatibility, vague static
  deferral wording, failing install/package tests, and C/header changes
  without full quality gates.
- Day 5 handoff: record the selected product path, supported and unsupported
  claims, proof owners, exact blockers, and Day 6 implementation checklist.

### Day 5: Product Decision Record

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day5-shared-library-abi-product-decision.md`.
- Selected the Sprint 153 product path: keep static-first packaging and
  strengthen the shared-library deferral with exact, test-backed blockers.
- Rejected implementing shared-library support in Sprint 153 because the repo
  does not yet have a public export/import macro, symbol visibility/export map,
  dynamic ABI compatibility policy, SONAME/install-name/DLL policy, installed
  shared consumer proof, runtime loader proof, or Windows allocator/runtime
  boundary policy.
- Defined the selected implementation scope:
  - preserve the static target and static install/export behavior;
  - keep `BUILD_SHARED_LIBS=ON` as a hard configure-time rejection;
  - strengthen rejection diagnostics around exact shared-library blockers;
  - keep shared artifacts, shared CMake metadata, and package selectors
    blocked;
  - align README, INSTALL, maintainer guide, package metadata, and sprint
    artifacts with the decision;
  - leave Sprint 154 a clean external-comparison handoff.
- Recorded explicit non-claims for shared-library packaging, dynamic ABI
  compatibility, Linux `.so`, macOS `.dylib`, Windows `.dll`/import-library,
  SONAME/install-name/RPATH/runtime loader behavior, static/shared selectors,
  package-manager distribution, Windows Makefile parity, and Windows
  `pkg-config` execution parity.
- Mapped proof ownership to `scripts/static_package_deferral_check.sh`,
  `tests/test_install.sh`, `tests/test_cmake_install.sh`, `CMakeLists.txt`,
  `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, README, INSTALL, maintainer
  guide, and platform CI wording.
- Day 6 handoff: design the focused static-first implementation changes,
  especially stronger `BUILD_SHARED_LIBS=ON` diagnostics and any matching
  static deferral guard or install/package proof updates.

### Day 6: Build And Install Design

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day6-build-install-design.md`.
- Inspected `CMakeLists.txt`, `Makefile`, `sparse.pc.in`,
  `cmake/SparseConfig.cmake.in`, `tests/test_install.sh`,
  `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh`.
- Designed the selected build behavior as stronger
  `BUILD_SHARED_LIBS=ON` rejection diagnostics, not shared target support.
- Confirmed Day 7 should keep:
  - `add_library(sparse_lu_ortho STATIC ...)`;
  - CMake `ARCHIVE DESTINATION` install behavior;
  - Make install of `libsparse_lu_ortho.a`, headers, generated
    `sparse_version.h`, and `sparse.pc`;
  - `Sparse::sparse_lu_ortho` as a static imported target;
  - `sparse.pc` static archive metadata and no `Libs.private` stanza;
  - no CMake or `pkg-config` static/shared selectors.
- Designed a focused Day 7 script/test update: make the CMake fatal error name
  exact blockers and make `scripts/static_package_deferral_check.sh` assert
  those blocker tokens.
- Confirmed the existing install tests already cover the selected package
  claim: static archive install, exact header count, static `sparse.pc`
  metadata, no shared artifacts, no shared CMake imported metadata, and
  installed downstream static consumers.
- Day 7 handoff: edit `CMakeLists.txt` and
  `scripts/static_package_deferral_check.sh`, then run
  `bash scripts/static_package_deferral_check.sh`,
  `bash tests/test_install.sh`, and `bash tests/test_cmake_install.sh`.

### Day 7: Build And Install Implementation

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day7-build-install-implementation.md`.
- Updated `CMakeLists.txt` so the `BUILD_SHARED_LIBS=ON` fatal error names
  the exact shared-library blockers selected on Day 5:
  - export/import policy;
  - symbol visibility policy;
  - dynamic ABI policy;
  - Linux SONAME metadata;
  - macOS install-name/RPATH metadata;
  - Windows DLL/import-library behavior;
  - installed shared consumer proof;
  - runtime-loader validation.
- Updated `scripts/static_package_deferral_check.sh` so the static deferral
  guard fails if any of those blocker tokens disappear from the CMake
  configure-failure output.
- Preserved static-first behavior: no shared target, no `LIBRARY`/`RUNTIME`
  install destination, no Make install change, no `sparse.pc` selector change,
  no CMake package selector change, and no public export/import macro.
- Focused validation planned and run for the selected claim:
  `bash scripts/static_package_deferral_check.sh`,
  `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, and
  `git diff --check`.
- Validation results:
  - `bash scripts/static_package_deferral_check.sh` passed;
  - `bash tests/test_install.sh` passed with `23` checks and `0` failures;
  - `bash tests/test_cmake_install.sh` passed with `26` checks, `0`
    failures, and `0` skips.

### Day 8: Downstream Proof Design

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day8-downstream-proof-design.md`.
- Inventoried current downstream proof:
  - Unix Make install and `pkg-config` proof in `tests/test_install.sh`;
  - Unix CMake install/export and installed `find_package(Sparse)` proof in
    `tests/test_cmake_install.sh`;
  - Windows CMake-first static install/downstream proof in
    `.github/workflows/windows-ci.yml`;
  - static-first shared-request rejection proof in
    `scripts/static_package_deferral_check.sh`.
- Designed Day 9 around unsupported-loader proof, not runtime loader proof,
  because Day 5 selected stronger static-first deferral.
- Preserved the current downstream proof responsibilities:
  - generated and maintained `pkg-config` consumers compile/link/run on Unix;
  - maintained CMake example and exact-version CMake consumer
    configure/build/run from the installed prefix;
  - mismatched CMake package version is rejected;
  - shared artifacts and shared imported CMake metadata remain absent.
- Identified the main Day 9 enhancement candidate: add or strengthen installed
  CMake package metadata scans for unsupported loader/static-shared selector
  tokens such as `SOVERSION`, `IMPORTED_SONAME`, `INSTALL_NAME`,
  `MACOSX_RPATH`, `IMPORTED_IMPLIB`, and `RUNTIME`.
- Defined platform boundaries:
  - Linux and macOS prove static package install/export only;
  - Windows proves CMake-first static install/downstream behavior only;
  - no platform claims shared-library packaging, dynamic ABI compatibility,
    runtime-loader behavior, or package-manager support.
- Day 9 handoff: implement only missing unsupported-loader metadata checks,
  keep static package behavior unchanged, and validate with
  `bash scripts/static_package_deferral_check.sh`,
  `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, and
  `git diff --check`.

### Day 9: Downstream Proof Implementation

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day9-downstream-proof-implementation.md`.
- Updated `tests/test_cmake_install.sh` with a dedicated installed CMake
  package metadata scan for unsupported loader and shared-selector tokens.
- The new check fails if installed package metadata contains `SOVERSION`,
  `IMPORTED_SONAME`, `INSTALL_NAME`, `MACOSX_RPATH`, `IMPORTED_IMPLIB`,
  standalone `RUNTIME`, static/shared component selectors, or
  `Sparse::*shared` target naming before shared-loader support is selected.
- Preserved the existing downstream proof surfaces:
  - Make install/`pkg-config` generated and maintained installed consumers;
  - CMake installed maintained example and exact-version consumers;
  - mismatched CMake package version rejection;
  - static archive install and no shared artifacts;
  - static imported target metadata and no shared imported metadata.
- Validation planned and run:
  `bash scripts/static_package_deferral_check.sh`,
  `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, and
  `git diff --check`.
- Validation results:
  - `bash scripts/static_package_deferral_check.sh` passed;
  - `bash tests/test_install.sh` passed with `23` checks and `0` failures;
  - `bash tests/test_cmake_install.sh` passed with `27` checks, `0`
    failures, and `0` skips.

### Day 10: Platform Lane Review And CI Policy

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day10-platform-ci-policy.md`.
- Audited Linux, macOS, and Windows workflow package/ABI lanes after the Day 9
  unsupported-loader metadata check.
- Selected no Linux workflow change: the reviewed package-contract lane already
  runs `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh`.
- Selected no macOS workflow command change: the reviewed install/pkg-config
  and CMake install/export lanes already run the updated Unix scripts.
- Identified the only Day 11 CI follow-through candidate: mirror the Day 9
  unsupported-loader/static-shared selector metadata scan in the Windows inline
  PowerShell CMake install/downstream proof.
- Kept `EXPECTED_WINDOWS_CTEST_COUNT: "59"` unchanged because Sprint 153 does
  not change CMake test registration.
- Selected no artifact upload or retention change because Sprint 153 adds
  metadata assertions, not generated runtime-loader logs.
- Preserved non-claims for shared-library packaging, dynamic ABI
  compatibility, Linux `.so`, macOS `.dylib`, Windows `.dll`/import-library,
  SONAME/install-name/RPATH/runtime-loader behavior, static/shared selectors,
  package-manager distribution, Windows Makefile parity, and Windows
  `pkg-config` execution parity.
- Day 11 handoff: update `.github/workflows/windows-ci.yml` inline install
  proof if feasible, search active docs and metadata for stale shared/ABI
  wording, and rerun the focused static package validation scripts.

### Day 11: CI And Documentation Implementation

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day11-ci-docs-implementation.md`.
- Updated `.github/workflows/windows-ci.yml` so the inline Windows CMake
  install/downstream proof now rejects unsupported loader/static-shared
  selector metadata in installed CMake package files.
- Kept Windows CI CMake-first and static-first:
  - no CTest count change;
  - no Windows Makefile parity claim;
  - no Windows `pkg-config` execution parity claim;
  - no DLL, dynamic ABI, or runtime-loader support claim.
- Updated `INSTALL.md` so shared-library deferral names the exact blockers:
  export/import policy, symbol visibility policy, dynamic ABI policy, Linux
  SONAME metadata, macOS install-name/RPATH metadata, Windows
  DLL/import-library behavior, installed shared consumer proof, and
  runtime-loader validation.
- Updated `README.md` to point top-level package readers at the exact
  `BUILD_SHARED_LIBS=ON` blocker categories.
- Updated `docs/maintainer_guide.md` so proof-owner descriptions include
  unsupported-loader metadata checks, exact shared deferral blocker wording,
  and the Windows CI mirror of the installed CMake metadata scan.
- Searched active package/ABI docs, workflows, package templates, install
  tests, deferral guard, and CMake metadata for stale shared-library, dynamic
  ABI, loader, selector, or package-manager wording.
- Validation planned and run:
  `bash scripts/static_package_deferral_check.sh`,
  `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, and
  `git diff --check`.
- Validation results:
  - `bash scripts/static_package_deferral_check.sh` passed;
  - `bash tests/test_install.sh` passed with `23` checks and `0` failures;
  - `bash tests/test_cmake_install.sh` passed with `27` checks, `0`
    failures, and `0` skips.

### Day 12: Integrated Package And ABI Validation

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day12-integrated-package-abi-validation.md`.
- Ran integrated static-first package validation:
  - `bash scripts/static_package_deferral_check.sh` passed;
  - `bash tests/test_install.sh` passed with `23` checks and `0` failures;
  - `bash tests/test_cmake_install.sh` passed with `27` checks, `0`
    failures, and `0` skips.
- Ran report-index evidence-meaning checks:
  - `python3 scripts/normalize_report_index.py --family package --check`
    reported `6` rows ok;
  - `python3 scripts/normalize_report_index.py --family package --check-freshness`
    reported freshness ok for `6` source-controlled package rows;
  - `python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness`
    reported freshness ok for `1` source-controlled runtime-backend row.
- `git diff --check` passed.
- Recorded that `scripts/normalize_report_index.py` does not expose a separate
  maintained `ci` family; CI evidence meaning is represented through package
  proof ownership, workflow comments, and source-controlled policy rows.
- Confirmed unsupported shared-library paths remain absence/rejection proof:
  no shared install artifacts, no shared CMake imported metadata, no
  unsupported loader/static-shared selector metadata, `BUILD_SHARED_LIBS=ON`
  exact blocker rejection, and no unsupported `sparse.pc` claims.
- Day 13 handoff: determine whether `.c` or public `.h` files changed, run the
  required quality gate level, rerun focused package/report/doc checks, and
  record residual shared-library/ABI/package/platform/loader debt.

### Day 13: Quality Gate And Residual Review

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day13-quality-gate-residual-review.md`.
- Reviewed changed files and confirmed Sprint 153 has no `.c` file or public
  `.h` header changes. The full `make format && make lint && make test` C
  gate is not required; focused package/report/doc validation is the correct
  gate for workflow, CMake, docs, shell, and install-test changes.
- Ran focused validation:
  - `bash scripts/static_package_deferral_check.sh` passed;
  - `bash tests/test_install.sh` passed with `23` checks and `0` failures;
  - `bash tests/test_cmake_install.sh` passed with `27` checks, `0`
    failures, and `0` skips;
  - `python3 scripts/normalize_report_index.py --family package --check`
    reported `6` rows ok;
  - `python3 scripts/normalize_report_index.py --family package --check-freshness`
    reported freshness ok for `6` source-controlled package rows;
  - `python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness`
    reported freshness ok for `1` source-controlled runtime-backend row.
- Ran a focused stale wording scan over active package/ABI docs, workflows,
  package templates, install tests, deferral guard, and CMake metadata. The
  remaining hits are expected non-claims, deferral diagnostics, or guard/test
  patterns; no active searched wording claims shared-library packaging,
  dynamic ABI compatibility, runtime-loader behavior, package-manager
  distribution, static/shared selectors, Windows Makefile parity, or Windows
  `pkg-config` execution parity.
- Recorded residual debt for export/import macros, symbol visibility/export
  allowlists, dynamic ABI policy, Linux `.so`, macOS `.dylib`, Windows
  DLL/import-library behavior, static/shared package selectors,
  package-manager distribution, Windows Makefile parity, and Windows
  `pkg-config` execution parity.
- Day 14 handoff: finalize Sprint 153 artifacts, write the Sprint 154
  external-comparison handoff, rerun final whitespace/status checks, and keep
  the selected package/ABI evidence boundary explicit.

### Day 14: Closeout And Sprint 154 Handoff

- Created
  `docs/planning/EPIC_13/SPRINT_153/artifacts/day14-closeout-sprint154-handoff.md`.
- Finalized the Sprint 153 artifact index and closeout evidence boundary.
- Recorded the Sprint 153 product decision in closeout form:
  static-first package support remains maintained, and shared-library support
  remains deferred by design.
- Summarized the selected implementation:
  - exact `BUILD_SHARED_LIBS=ON` blocker diagnostics in `CMakeLists.txt`;
  - static deferral guard checks in
    `scripts/static_package_deferral_check.sh`;
  - unsupported loader/static-shared selector metadata checks in
    `tests/test_cmake_install.sh`;
  - mirrored Windows CMake install metadata checks in
    `.github/workflows/windows-ci.yml`;
  - aligned README, INSTALL, and maintainer-guide wording.
- Wrote the Sprint 154 external-comparison handoff:
  - external comparisons may cite static archive package install/export proof,
    installed Unix `pkg-config` consumer proof, installed Unix and Windows
    CMake consumer proof, and exact unsupported shared-library diagnostics;
  - external comparisons must not infer shared-library, dynamic ABI,
    runtime-loader, package-manager, Windows Makefile, or Windows
    `pkg-config` execution support from the static package evidence.
- Final validation planned:
  `bash scripts/static_package_deferral_check.sh`,
  `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`,
  `python3 scripts/normalize_report_index.py --family package --check`,
  `python3 scripts/normalize_report_index.py --family package --check-freshness`,
  `python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness`,
  focused stale wording scan, `git diff --check`, and `git status --short`.
- Final validation results:
  - `bash scripts/static_package_deferral_check.sh` passed;
  - `bash tests/test_install.sh` passed with `23` checks and `0` failures;
  - `bash tests/test_cmake_install.sh` passed with `27` checks, `0`
    failures, and `0` skips;
  - `python3 scripts/normalize_report_index.py --family package --check`
    reported `6` rows ok;
  - `python3 scripts/normalize_report_index.py --family package --check-freshness`
    reported freshness ok for `6` source-controlled package rows;
  - `python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness`
    reported freshness ok for `1` source-controlled row;
  - focused stale wording scan returned only expected non-claims, static
    deferral diagnostics, and guard/test patterns;
  - `git diff --check` passed;
  - `git status --short` showed only intended Sprint 153 branch changes.
- Sprint 153 closeout status: ready for retrospective preparation, with the
  static-first package boundary and Sprint 154 external-comparison handoff
  explicitly recorded.
