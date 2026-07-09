# Sprint 115 Working Notes

## Sprint Goal

Sprint 115 resolves residual package/platform parity and ABI productization
decisions in dependency order. The sprint decides which local install proofs
should become reviewed lanes, which Windows and macOS exclusions can move
toward parity, and which broader ABI/package-manager product claims remain
future contracts.

## Starting Constraints

- Consume only package/platform-facing residual debt from Sprint 114.
- Do not pull eigensolver source movement, broad direct/iterative oracle, or
  broad SVD abstraction work into Sprint 115.
- Do not repeat Sprint 112 completed package surface audit, static-first
  support decision, Make install proof, CMake install/export proof,
  downstream consumer proof, platform-tier contract, packaging docs alignment,
  or validation closeout.
- Do not infer shared-library, dynamic ABI, package-manager, Windows installed
  package, full macOS install/export, Windows thread/fuzz/property, or
  expanded platform parity claims without reviewed evidence.
- If CI, build metadata, scripts, public docs, `.c`, `.h`, or workflows change,
  run checks appropriate to the touched surface before proceeding.

## Completed Work Excluded From Sprint 115 Scope

| Completed work | Source evidence | Sprint 115 handling |
|---|---|---|
| Package surface audit | Sprint 112 Day 2 artifact and retrospective | Use as baseline; do not repeat as unresolved debt. |
| Static-first versus shared-library/ABI support decision | Sprint 112 Days 3-4 artifacts | Revisit only as a product-contract decision, not as hidden support. |
| Install/consumer proof design | Sprint 112 Day 5 artifact | Use to evaluate promotion to reviewed lanes. |
| Make install and pkg-config proof | Sprint 112 Day 6 artifact | Treat as local proof unless Sprint 115 promotes it. |
| CMake install/export proof | Sprint 112 Day 7 artifact | Treat as local proof unless reviewed platform lanes are added. |
| Downstream consumer proof | Sprint 112 Day 8 artifact | Use as install/export evidence baseline. |
| Platform-tier contract | Sprint 112 Day 9 artifact | Preserve unless Sprint 115 adds reviewed platform evidence. |
| Windows reviewed-scope follow-through | Sprint 112 Day 10 artifact | Use staged-exclusion decisions as starting point. |
| macOS package/platform follow-through | Sprint 112 Day 11 artifact | Use staged-exclusion decisions as starting point. |
| Packaging documentation alignment | Sprint 112 Day 12 artifact | Update only if Sprint 115 decisions change claim wording. |
| Integrated package/platform validation | Sprint 112 Day 13 artifact | Use as validation pattern. |
| Sprint 112 closeout and handoff | Sprint 112 Day 14 artifact and retrospective | Use residual queue as Sprint 115 input. |
| Sprint 114 eigensolver/source-boundary residuals | Sprint 114 retrospective and Epic 10 deferral decision | Defer to Sprint 117 residual queue/post-Epic handoff. |

## Residual Package/Platform Owners

| Residual owner | Primary surfaces | Sprint 115 day(s) |
|---|---|---:|
| Linux install proof CI promotion | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `.github/workflows/ci.yml`, README/INSTALL wording | 2-3 |
| macOS CMake install/export parity | `.github/workflows/macos-ci.yml`, CMake install/export proof, README/INSTALL wording | 4-5 |
| Windows install-validation lane | `.github/workflows/windows-ci.yml`, CMake install/export proof, downstream consumer proof | 6-7 |
| Windows thread/fuzz portability | `tests/test_threads.c`, `tests/test_sprint4_integration.c`, `tests/test_fuzz.c`, CMake/CTest membership, Windows CI comments | 8-9 |
| macOS backend and toolchain follow-through | Makefile sanitizer/coverage notes, macOS CI, Homebrew GCC/libomp wording | 10 |
| Shared-library and dynamic ABI contract | `CMakeLists.txt`, Makefile install/pkg-config, README/INSTALL, package docs | 11 |
| Package-manager support decision | README/INSTALL/package docs, potential Homebrew/vcpkg/distro/Windows package references | 12 |
| Sprint 114 package/platform residual intake | Sprint 114 retrospective, Epic 10 deferral decision, Sprint 115 package/platform decisions | 13 |
| Validation and package/platform handoff | All touched docs, workflows, scripts, build metadata, and code surfaces | 14 |

## Touched-Surface Inventory

| Surface | Files / paths | Notes |
|---|---|---|
| Local install proof | `tests/test_install.sh`, `tests/test_cmake_install.sh` | Existing local Unix-side install/export evidence. |
| CI workflows | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | Reviewed lane promotion or deferral decisions may touch these. |
| CMake package surface | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, `examples/cmake_example/CMakeLists.txt` | Static-first package/export and downstream consumer surfaces. |
| Make/package install surface | `Makefile` | `install`, `uninstall`, pkg-config, sanitizer, coverage, and macOS toolchain notes. |
| Adoption docs | `README.md`, `INSTALL.md`, `docs/` package/platform references | Claim wording must match reviewed evidence. |
| Windows portability tests | `tests/test_threads.c`, `tests/test_sprint4_integration.c`, `tests/test_fuzz.c` | Staged-exclusion or bounded reviewed proof candidates. |
| Platform support comments | `tests/test_framework.h`, test skip comments, workflow comments | Must not imply broader Windows/macOS support than reviewed. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Residual package/platform intake, duplicate fence, working notes baseline. | Item 1 |
| 2 | Linux install proof CI promotion design. | Item 2 |
| 3 | Linux install proof CI promotion or no-promotion decision. | Item 2 |
| 4 | macOS CMake install/export parity design. | Item 3 |
| 5 | macOS CMake install/export proof or deferral. | Item 3 |
| 6 | Windows install-validation lane design. | Item 4 |
| 7 | Windows install-validation proof or deferral. | Item 4 |
| 8 | Windows thread/fuzz portability audit. | Item 5 |
| 9 | Windows thread/fuzz proof or staged-exclusion follow-through. | Item 5 |
| 10 | macOS backend, Homebrew GCC, coverage, and TSan follow-through. | Item 6 |
| 11 | Shared-library and dynamic ABI product-contract decision. | Item 7 |
| 12 | Package-manager support decision. | Item 8 |
| 13 | Sprint 114 package/platform residual intake and deferral boundary. | Item 9 |
| 14 | Validation, metrics, and package/platform handoff. | Item 10 |

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation only | `git diff --check`; trailing-whitespace scan over touched docs; local relative Markdown link check when links change. |
| Shell scripts | syntax check where practical; focused script dry run or documented reason not run; `git diff --check`. |
| GitHub workflows | YAML/structure review; focused command review; no unsupported reviewed-lane claim without CI evidence. |
| Make/CMake/package metadata | focused Make/CMake configure/build/install checks as applicable; `make source-list-check` if source membership changes. |
| C test files or headers | focused affected test binary; `make format && make lint && make test`. |
| CTest registration | `ctest -N` through relevant CMake build path and explicit reviewed-count documentation. |

## Day 1 Notes

- Created Sprint 115 working notes and artifact directory.
- Re-read Sprint 115 project-plan scope and Day 1 plan.
- Re-read Sprint 112 package/platform residual deferred debt:
  - Linux local install scripts versus reviewed CI lanes;
  - macOS CMake install/export parity;
  - Windows install-validation lane;
  - Windows thread/fuzz/property portability;
  - macOS coverage, Homebrew GCC, and TSan assumptions;
  - shared-library/dynamic ABI product contract;
  - package-manager support.
- Re-read Sprint 114 residual deferred debt and Epic 10 deferral decision.
- Explicitly excluded Sprint 114 non-package residuals from Sprint 115:
  - eigensolver private-owner movement;
  - `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert, and
    `lanczos_iterate_op` movement;
  - broad direct/iterative oracle abstraction;
  - broad SVD proof-helper abstraction.
- Inventoried affected surfaces:
  - local install scripts;
  - Linux, macOS, and Windows CI workflows;
  - Make/CMake install/export/package metadata;
  - README/INSTALL/adoption documentation;
  - Windows thread/fuzz portability tests and CTest membership;
  - macOS sanitizer, coverage, Homebrew GCC, and libomp toolchain notes.
- Added Day 1 artifact:
  - `artifacts/day1-residual-package-platform-intake.md`.

## Day 2 Notes

- Inspected current local Linux/Unix install proof scripts:
  - `tests/test_install.sh`;
  - `tests/test_cmake_install.sh`.
- Inspected the Linux CI workflow:
  - `.github/workflows/ci.yml`.
- Confirmed current Linux CI contract:
  - reviewed source-of-truth lanes remain Makefile compile-quality, reviewed
    CMake parity, and dead-code;
  - direct runtime, sanitizer, TSan, benchmark, and coverage lanes remain
    supplemental;
  - focused install/package regression scripts are explicitly documented as
    developer-side proof rather than separate reviewed Linux CI lanes.
- Inventoried local Make install proof:
  - `make clean`;
  - `make install PREFIX=<tmp>`;
  - static archive installation;
  - no shared-library artifacts;
  - public header count plus generated `sparse_version.h`;
  - `sparse.pc`;
  - `pkg-config --cflags`, `--libs`, and `--modversion`;
  - installed downstream compile/link/run smoke;
  - maintained example source compile/link/run through `pkg-config`;
  - `make uninstall` cleanup.
- Inventoried local CMake install/export proof:
  - CMake configure, build, and install into a temp prefix;
  - static archive and headers;
  - `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and
    `SparseTargets.cmake`;
  - `sparse.pc`;
  - installed `examples/cmake_example` configured via `find_package(Sparse)`;
  - exact-version `find_package` acceptance;
  - mismatched-version rejection when a lower same-major version exists;
  - `pkg-config --modversion`.
- Defined Day 3 promotion criteria:
  - reviewed lane must run both scripts from a clean checkout on Ubuntu;
  - `pkg-config`, CMake, compiler, and install-prefix assumptions must be
    explicit;
  - lane ownership must remain static-first and must not imply shared-library,
    ABI, package-manager, Windows, or macOS parity;
  - runtime/cache risk must be acceptable relative to existing reviewed CI
    cost.
- Defined no-promotion criteria:
  - if the lane duplicates local proof without adding reviewed claim value;
  - if it materially increases PR runtime or flakiness;
  - if existing CI comments/docs would remain more truthful with the scripts
    as developer-side proof.
- Added Day 2 artifact:
  - `artifacts/day2-linux-install-ci-promotion-design.md`.

## Day 3 Notes

- Applied Day 2 promotion criteria to the current Linux CI contract.
- Decision: do not promote `tests/test_install.sh` or
  `tests/test_cmake_install.sh` to a separate reviewed Linux CI lane in
  Sprint 115.
- Rationale:
  - the scripts remain strong local Unix-side static install/export proof;
  - existing Linux reviewed lanes already own compile quality, CMake parity,
    and dead-code source-of-truth checks;
  - a new PR-time reviewed install lane would mostly duplicate local package
    proof without changing the current public support claim;
  - current `.github/workflows/ci.yml`, `docs/maintainer_guide.md`, README,
    and INSTALL wording already describe the local-only proof boundary;
  - keeping install proof local avoids new reviewed-lane runtime and
    dependency ownership while preserving the static-first package contract.
- No CI workflow, script, README, INSTALL, or maintainer-guide wording changed
  because the current wording already matches the Day 3 decision.
- Preserved non-claims:
  - no shared-library or dynamic ABI support claim;
  - no package-manager support claim;
  - no Windows or macOS install parity claim;
  - no reviewed Linux install lane claim.
- Added Day 3 artifact:
  - `artifacts/day3-linux-install-ci-no-promotion-decision.md`.

## Day 4 Notes

- Inspected existing macOS CI:
  - `.github/workflows/macos-ci.yml`.
- Confirmed current reviewed macOS surface:
  - Apple Clang reviewed path runs `make quality-review-compile`,
    `make quality-review-cmake`, `make wall-check`, and `make sanitize`;
  - Homebrew GCC remains supplemental direct build/test/wall-check coverage;
  - Make install/`pkg-config` runs as supplemental static-first package
    confidence only.
- Re-read Sprint 112 macOS follow-through:
  - local `tests/test_install.sh` passed on macOS;
  - local `tests/test_cmake_install.sh` passed on macOS;
  - local CMake install/export proof was recorded but not promoted to reviewed
    macOS install/export parity.
- Defined reviewed macOS CMake install/export lane requirements:
  - run `bash tests/test_cmake_install.sh` or an equivalent installed CMake
    consumer proof on `macos-latest`;
  - verify `cmake --install`, installed `SparseConfig*.cmake`,
    `SparseTargets.cmake`, `find_package(Sparse)`, exact-version behavior,
    installed consumer build/run, static archive shape, and no shared-library
    artifacts;
  - keep the claim static-first and separate from package-manager, dynamic ABI,
    dylib, runtime-loader, and Linux/Windows parity claims.
- Defined Day 5 deferral criteria:
  - if the supplemental Make install/`pkg-config` lane already provides the
    right package confidence for macOS;
  - if adding CMake install/export would duplicate local proof without
    changing a reviewed claim;
  - if CI runtime/toolchain ownership would exceed Sprint 115's package truth
    needs.
- Added Day 4 artifact:
  - `artifacts/day4-macos-cmake-install-export-design.md`.

## Day 5 Notes

- Applied Day 4 reviewed-lane requirements and deferral criteria to the
  current macOS workflow.
- Decision: defer reviewed macOS CMake install/export parity in Sprint 115.
- Rationale:
  - macOS already has a reviewed Apple Clang path for compile quality, CMake
    parity, wall-check, and sanitizer coverage;
  - macOS already has a supplemental static-first Make install/`pkg-config`
    confidence job;
  - local `tests/test_cmake_install.sh` remains the Unix-side proof for CMake
    install/export and `find_package(Sparse)`;
  - adding a reviewed macOS `tests/test_cmake_install.sh` job would broaden
    reviewed package ownership without changing current public support
    wording;
  - current `.github/workflows/macos-ci.yml`, README, INSTALL, and maintainer
    guide wording already says macOS does not claim full reviewed
    install/export parity.
- No workflow, README, INSTALL, or maintainer-guide wording changed because
  existing wording already matches the deferral decision.
- Preserved non-claims:
  - no full reviewed macOS CMake install/export parity;
  - no macOS shared-library, dylib, dynamic ABI, package-manager, or
    runtime-loader support claim;
  - no claim that local CMake install/export proof replaces reviewed CI.
- Added Day 5 artifact:
  - `artifacts/day5-macos-cmake-install-export-deferral.md`.

## Day 6 Notes

- Inspected existing Windows CI:
  - `.github/workflows/windows-ci.yml`.
- Confirmed current reviewed Windows surface:
  - pinned `windows-2022`;
  - `Visual Studio 17 2022` x64 CMake configure;
  - Release build;
  - `ctest -N` registration guard with `EXPECTED_WINDOWS_CTEST_COUNT=51`;
  - full `ctest` execution for the registered reviewed subset.
- Confirmed current staged exclusions:
  - no reviewed Windows Makefile parity;
  - no reviewed Windows install-validation lane;
  - `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain
    outside the reviewed Windows CMake subset.
- Defined required Windows install-validation proof:
  - configure and build with MSVC;
  - `cmake --install` into a temp prefix;
  - verify installed static library, public headers, and CMake package files;
  - configure a downstream project with `CMAKE_PREFIX_PATH`;
  - use `find_package(Sparse REQUIRED)` and link `Sparse::sparse_lu_ortho`;
  - build and run the downstream executable;
  - document whether CTest registration count is unchanged or intentionally
    changed.
- Defined Day 7 deferral criteria:
  - if adding install validation would broaden Windows support beyond the
    current CMake-first reviewed subset;
  - if a downstream installed consumer proof needs more PowerShell/CMake
    scaffolding than Sprint 115 should introduce;
  - if reviewed CTest count and staged exclusions would become less clear.
- Added Day 6 artifact:
  - `artifacts/day6-windows-install-validation-design.md`.

## Day 7 Notes

- Applied Day 6 install-validation requirements and deferral criteria to the
  current Windows workflow.
- Decision: defer a separate reviewed Windows install-validation lane in
  Sprint 115.
- Rationale:
  - Windows currently owns a reviewed MSVC CMake-first subset with configure,
    build, guarded `ctest -N`, and full `ctest` execution;
  - existing workflow output and docs already say there is no separate reviewed
    Windows install-validation lane;
  - adding `cmake --install` plus downstream installed `find_package(Sparse)`
    proof would broaden reviewed Windows package ownership beyond current
    support wording;
  - a robust lane needs PowerShell install-prefix handling, Release
    config-specific package-file checks, downstream consumer configure/build/run
    proof, and explicit non-claims;
  - keeping the lane deferred preserves the current CTest count guard and
    staged-exclusion clarity.
- No workflow, README, INSTALL, maintainer-guide, CMake, or script wording
  changed because the existing wording already matches the Day 7 decision.
- Preserved non-claims:
  - no Windows installed-package support claim;
  - no Windows Makefile parity;
  - no Windows `pkg-config` or package-manager support;
  - no Windows shared-library/DLL, dynamic ABI, or runtime-loader claim;
  - no change to Windows reviewed CTest membership.
- Added Day 7 artifact:
  - `artifacts/day7-windows-install-validation-deferral.md`.

## Day 8 Notes

- Audited the current Windows staged exclusions:
  - `test_threads`;
  - `test_sprint4_integration`;
  - `test_fuzz`.
- Confirmed current registration contract:
  - `CMakeLists.txt` registers `test_threads` and
    `test_sprint4_integration` only when `Threads_FOUND AND NOT WIN32`;
  - `CMakeLists.txt` registers `test_fuzz` only when
    `NOT WIN32 AND NOT MSVC`;
  - `.github/workflows/windows-ci.yml` pins the reviewed Windows count at
    `EXPECTED_WINDOWS_CTEST_COUNT=51`;
  - the Windows workflow prints the staged exclusions explicitly.
- Identified portability blockers:
  - `test_threads` directly includes `<pthread.h>` and owns many
    `pthread_create` / `pthread_join` proof cases, including optional
    `SPARSE_MUTEX` concurrent insert coverage;
  - `test_sprint4_integration` directly includes `<pthread.h>` and has a
    concurrent Cholesky SuiteSparse proof over `nos4.mtx`;
  - `test_fuzz` directly includes `<unistd.h>`, uses `mkstemps`, `close`, and
    `unlink`, and mixes Matrix Market parser fuzz cases with seeded solver
    property lanes;
  - `test_fuzz` also owns the bounded large-n lifecycle property lanes that
    are intentionally not claimed as reviewed Windows evidence.
- Decision: do not select a bounded Windows-native proof owner on Day 8.
- Rationale:
  - porting either thread test would require a Windows threading abstraction,
    a split test owner, or native Win32 thread calls rather than a narrow
    support-wording clarification;
  - porting `test_fuzz` would require a Windows-safe temporary-file helper and
    a careful split between parser fuzz coverage and broader property lanes;
  - adding any of the three binaries to Windows CTest would change the reviewed
    count and imply broader Windows parity than the current evidence supports.
- Day 9 should therefore publish the staged-exclusion follow-through unless a
  narrow documentation or comment update is still needed.
- No CMake, workflow, README, INSTALL, maintainer-guide, or test source changes
  were made because the current wording already matches the Day 8 audit.
- Added Day 8 artifact:
  - `artifacts/day8-windows-thread-fuzz-portability-audit.md`.

## Day 9 Notes

- Applied the Day 8 audit decision.
- Decision: preserve the current Windows staged exclusions rather than adding
  a bounded Windows-native thread, fuzz, or property proof in Sprint 115.
- Confirmed no source or registration changes are needed:
  - `test_threads` remains gated by `Threads_FOUND AND NOT WIN32`;
  - `test_sprint4_integration` remains gated by
    `Threads_FOUND AND NOT WIN32`;
  - `test_fuzz` remains gated by `NOT WIN32 AND NOT MSVC`;
  - Windows CI keeps `EXPECTED_WINDOWS_CTEST_COUNT=51`.
- Confirmed no workflow or documentation wording changes are needed:
  - `.github/workflows/windows-ci.yml` already prints the excluded test names
    and says the lane is reviewed CMake-first consumer proof only;
  - `docs/maintainer_guide.md` already records the three excluded tests and
    warns that `test_fuzz` property evidence is not reviewed Windows evidence;
  - README and INSTALL continue to describe Windows as the reviewed CMake
    subset rather than full platform parity.
- Preserved Day 9 non-claims:
  - no Windows thread-safety parity;
  - no Windows fuzz/property parity;
  - no reviewed Windows lifecycle property lane;
  - no Windows CTest count change;
  - no Windows Makefile, install-validation, package-manager, shared-library,
    DLL, dynamic ABI, or runtime-loader claim.
- Added Day 9 artifact:
  - `artifacts/day9-windows-thread-fuzz-staged-exclusion-follow-through.md`.

## Day 10 Notes

- Inspected current macOS CI:
  - `.github/workflows/macos-ci.yml`.
- Confirmed the reviewed macOS lane remains Apple Clang only:
  - `make quality-review-compile`;
  - `make quality-review-cmake`;
  - `make wall-check`;
  - `make sanitize`.
- Confirmed supplemental macOS lanes:
  - Homebrew GCC (`gcc-15`) direct build/test/wall-check coverage;
  - Make install/`pkg-config` confidence path through
    `tests/test_install.sh`.
- Reviewed Makefile backend/toolchain notes:
  - Apple Clang coverage routes to `coverage-gcovr` because Homebrew lcov
    cannot parse Apple gcov's `.gcno` format;
  - Homebrew GCC coverage can use `coverage-lcov`, but macOS 15+
    CommandLineTools SDK mismatch remains a documented caveat;
  - OpenMP on macOS requires Homebrew `libomp` for Apple Clang;
  - Apple Clang TSan remains blocked by the dyld initialization hang, while
    `sanitize-thread` requires Homebrew LLVM and remains local/specialized.
- Decision: do not promote new macOS backend, Homebrew GCC, coverage, OpenMP,
  or TSan lanes in Sprint 115.
- Rationale:
  - current workflow comments and docs already distinguish reviewed Apple
    Clang evidence from supplemental Homebrew GCC/install confidence;
  - coverage remains tree-mutating and supplemental rather than a reviewed
    macOS CI claim;
  - Homebrew GCC is useful second-compiler evidence but should not own
    reviewed macOS package/toolchain truth;
  - TSan remains Linux-reviewed/supplemental for CI, with macOS TSan blocked
    or local-only depending on compiler/runtime.
- No workflow, Makefile, README, INSTALL, or maintainer-guide wording changed
  because the current wording already matches the Day 10 decision.
- Preserved non-claims:
  - no reviewed macOS CMake install/export parity;
  - no macOS coverage reviewed-lane claim;
  - no Homebrew GCC reviewed-lane promotion;
  - no macOS TSan reviewed-lane claim;
  - no macOS package-manager, dylib, shared-library, dynamic ABI, or
    runtime-loader claim.
- Added Day 10 artifact:
  - `artifacts/day10-macos-backend-toolchain-follow-through.md`.

## Day 11 Notes

- Inspected current static-first package/build surfaces:
  - `CMakeLists.txt`;
  - `Makefile`;
  - `README.md`;
  - `INSTALL.md`;
  - `docs/maintainer_guide.md`;
  - `sparse.pc.in`;
  - `cmake/SparseConfig.cmake.in`.
- Confirmed current contract:
  - CMake warns when `BUILD_SHARED_LIBS=ON` is requested and still builds
    `sparse_lu_ortho` as `STATIC`;
  - Makefile install copies only the static archive plus headers and
    `sparse.pc`;
  - `cmake --install` exports the static `Sparse::sparse_lu_ortho` target;
  - `SparseConfigVersion.cmake` uses exact-version compatibility;
  - README, INSTALL, and maintainer guide explicitly defer shared-library
    packaging and dynamic ABI compatibility.
- Decision: do not add shared-library or dynamic ABI support in Sprint 115.
  Publish it as a future product contract with explicit acceptance criteria.
- Future proof must include:
  - shared-library build rules across Make and CMake;
  - exported/imported target metadata for shared artifacts;
  - platform runtime-loader proof for Linux, macOS, and Windows;
  - symbol visibility/export policy;
  - SONAME/SOVERSION or equivalent versioning policy;
  - ABI compatibility/rejection tests;
  - installed downstream consumer proof for shared linkage;
  - documentation that distinguishes static, shared, and source-level API
    compatibility.
- No support wording changed because existing public and maintainer docs
  already preserve the static-first contract and dynamic-ABI non-claim.
- Preserved non-claims:
  - no shared-library package support;
  - no dynamic ABI compatibility guarantee;
  - no SONAME/SOVERSION policy;
  - no DLL/import-library support;
  - no dylib install-name/rpath support;
  - no runtime-loader validation;
  - no package-manager support claim.
- Added Day 11 artifact:
  - `artifacts/day11-shared-library-dynamic-abi-contract.md`.

## Day 12 Notes

- Inventoried package-manager and install-consumer references across:
  - `README.md`;
  - `INSTALL.md`;
  - `docs/maintainer_guide.md`;
  - `.github/workflows/ci.yml`;
  - `.github/workflows/macos-ci.yml`;
  - `Makefile`;
  - `CMakeLists.txt`;
  - package metadata templates and install validation scripts.
- Confirmed current package-manager state:
  - Homebrew is referenced only for dependencies such as GCC, LLVM, libomp,
    lcov, gcovr, and cppcheck;
  - apt/dnf are referenced only for developer dependencies;
  - no Homebrew formula, vcpkg port, distro spec/debian packaging, Chocolatey,
    winget, Conan, Spack, or MSYS2 package recipe exists for this library;
  - `pkg-config` and `find_package(Sparse)` are installed-consumer metadata,
    not package-manager distribution recipes.
- Decision: package-manager support remains future work. Sprint 115 does not
  add a bounded package-manager proof plan or recipe.
- Future package-manager support must include:
  - recipe metadata for each claimed manager;
  - source/archive integrity and version mapping;
  - static-first build/install proof;
  - downstream `pkg-config` or `find_package(Sparse)` consumer proof;
  - CI or reproducible local validation for every claimed platform/manager;
  - explicit handling of shared-library, ABI, and runtime-loader non-claims.
- No README, INSTALL, maintainer-guide, workflow, Makefile, or CMake wording
  changed because current docs do not claim package-manager support.
- Preserved non-claims:
  - no Homebrew formula support;
  - no vcpkg port support;
  - no distro package support;
  - no Windows package-manager support;
  - no Conan or Spack support;
  - no package-manager shared-library or dynamic ABI support claim.
- Added Day 12 artifact:
  - `artifacts/day12-package-manager-support-decision.md`.

## Day 13 Notes

- Re-read Sprint 114 retrospective residual deferred debt and the Epic 10
  project-plan deferral decision.
- Confirmed Sprint 115 consumed only package/platform-facing residuals:
  - package, ABI, Windows, CMake parity, install-header, and adoption claim
    fences were checked against Sprint 115 decisions;
  - Linux install proof remains local-only rather than a reviewed CI lane;
  - macOS CMake install/export parity remains deferred;
  - Windows install-validation remains deferred;
  - Windows thread/fuzz/property tests remain staged exclusions;
  - macOS backend/toolchain lanes remain reviewed/supplemental as documented;
  - shared-library/dynamic ABI support remains a future product contract;
  - package-manager support remains future work.
- Confirmed Sprint 114 non-package residuals remain outside Sprint 115:
  - eigensolver private-owner movement;
  - `s20_select_indices` movement;
  - `s20_lift_ritz_vectors` movement;
  - shift-invert setup/conversion movement;
  - `lanczos_iterate_op` movement;
  - broad direct/iterative generated-RHS oracle abstraction;
  - broad SVD proof-helper abstraction.
- Handoff routing:
  - Sprint 116 adoption QA should verify user-facing docs do not advertise
    unreviewed install, platform, ABI, or package-manager support;
  - Sprint 117 residual queue should carry the Sprint 114 source-boundary and
    proof-owner residuals unless it explicitly promotes one item with build,
    CMake, source-list, CTest, and rollback evidence;
  - Epic closeout should preserve package/platform non-claims as final truth
    unless later reviewed evidence changes them.
- No README, INSTALL, maintainer-guide, workflow, CMake, Makefile, source, or
  header wording changed because the Day 13 work is residual routing only.
- Added Day 13 artifact:
  - `artifacts/day13-sprint114-package-platform-residual-intake.md`.

## Day 14 Notes

- Reviewed all Sprint 115 artifacts and working notes.
- Confirmed Sprint 115 touched only planning documentation:
  - `docs/planning/EPIC_10/SPRINT_115/`;
  - `docs/planning/EPIC_10/PROJECT_PLAN.md` for the Sprint 117 residual-queue
    follow-up.
- Confirmed no CI workflow, build metadata, Makefile, CMake, script, source,
  header, package metadata, public API, install-header, helper-target, or
  reviewed CTest registration changes were made.
- Final package/platform decision matrix:
  - Linux install proof remains local Unix-side evidence; no reviewed Linux
    install CI lane was added.
  - macOS CMake install/export parity remains deferred.
  - Windows install-validation remains deferred.
  - Windows thread/fuzz/property proof remains staged.
  - macOS Homebrew GCC, coverage, OpenMP, and TSan paths remain supplemental
    or local as documented.
  - Shared-library and dynamic ABI support remain future product contracts.
  - Package-manager support remains future work.
- Final handoff:
  - Sprint 116 adoption QA should use these decisions to guard user-facing
    install, package, platform, ABI, and package-manager claims.
  - Sprint 117 closeout should carry Sprint 114 non-package residuals unless
    explicitly promoted with full proof and rollback evidence.
  - Epic closeout should preserve Sprint 115 package/platform non-claims unless
    later reviewed evidence changes them.
- Validation:
  - `git diff --check` passed;
  - trailing-whitespace scan over `docs/planning/EPIC_10/SPRINT_115` passed.
- No `.c` or `.h` files changed, so the full `make format && make lint &&
  make test` gate was not required.
- Added Day 14 artifact:
  - `artifacts/day14-validation-package-platform-handoff.md`.
