# Sprint 112 Working Notes

## Sprint Goal

Sprint 112 strengthens package, ABI, and cross-platform support truth. The
sprint decides whether Epic 10 proves a shared-library/ABI support tier or
keeps the maintained package story static-first, then aligns install/export,
downstream consumer, platform-tier, and documentation evidence with that
decision.

## Starting Constraints

- Do not infer shared-library or dynamic-ABI support from static install proof.
- Do not infer expanded Windows coverage from Sprint 110's no-public-header
  drift result.
- Keep the maintained package surface static-first unless a later decision
  artifact explicitly proves a broader support tier.
- Treat Linux as the strongest reviewed source of truth until platform-tier
  evidence proves otherwise.
- Treat macOS install/pkg-config CI as supplemental static-first confidence,
  not full reviewed install/export parity.
- Treat Windows as a reviewed CMake-first consumer subset, with no separate
  reviewed install-validation lane unless Sprint 112 adds and proves one.
- Keep user-facing package docs concise; put detailed proof ownership and
  reviewed-lane interpretation in maintainer-facing docs and Sprint artifacts.
- If `.c`, `.h`, build-system, CI, or install scripts change, run the strongest
  checks required for the touched surface.

## Package and Platform Surface Inventory

| Surface | Files / Targets | Current Role | Sprint 112 Disposition |
|---|---|---|---|
| Make install | `Makefile` `install` / `uninstall` | Installs static archive, public headers, generated version header, and `sparse.pc` under `PREFIX`/`DESTDIR`. | Audit and validate as Unix-side static install surface. |
| CMake install/export | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in` | Installs static target export as `Sparse::sparse_lu_ortho`, config/version files, headers, library, and pkg-config metadata. | Audit and validate exact-version static package behavior. |
| pkg-config | `sparse.pc.in`, generated `sparse.pc` | Exposes include path, `-lsparse_lu_ortho`, `-lm`, and optional thread/OpenMP flags. | Validate compile/link flags against staged installs. |
| Make install validation | `tests/test_install.sh` | Stages Make install, checks static library/headers/pkg-config, compiles and runs pkg-config consumers, then uninstalls. | Use as baseline for Day 6 and proof design. |
| CMake install validation | `tests/test_cmake_install.sh` | Stages CMake install/export, builds downstream `find_package(Sparse)` consumer, verifies exact-version behavior and pkg-config version. | Use as baseline for Day 7 and proof design. |
| Installed CMake consumer | `examples/cmake_example/` | Minimal downstream `find_package(Sparse)` example. | Keep as primary installed-consumer example. |
| Public examples | `examples/*.c`, `examples/README.md` | Local build-tree and adoption examples. | Audit for package assumptions; avoid private headers. |
| Install docs | `INSTALL.md` | Operational setup, staged installs, installed-consumer workflows, platform setup, verification scripts. | Primary user-facing package/install authority. |
| README package summary | `README.md` | Compact install and support-tier summary. | Keep concise; avoid maintainer proof overload. |
| Maintainer package docs | `docs/maintainer_guide.md` | Reviewed-lane, package/ABI, CI, and proof-owner interpretation. | Primary place for detailed evidence boundaries. |
| Linux CI | `.github/workflows/ci.yml` | Strongest reviewed Makefile/CMake/dead-code source of truth plus supplemental runtime, TSan, coverage, and benchmark signals. | Treat as strongest reviewed platform tier. |
| macOS CI | `.github/workflows/macos-ci.yml` | Apple Clang reviewed path, supplemental Homebrew GCC, and supplemental Make install/pkg-config proof. | Keep narrower than Linux; do not overclaim install/export parity. |
| Windows CI | `.github/workflows/windows-ci.yml` | Reviewed MSVC CMake subset and CMake-first consumer proof. | Preserve staged exclusions and no separate reviewed install-validation claim. |

## Support-Claim Baseline

| Claim Area | Baseline Status | Evidence / Boundary |
|---|---|---|
| Static archive package | Maintained | `Makefile`, `CMakeLists.txt`, `INSTALL.md`, install scripts, and docs describe static-first package behavior. |
| `pkg-config` consumer path | Maintained on Unix-side install surface | `sparse.pc.in` plus `tests/test_install.sh` compile/run consumers with installed flags. |
| CMake `find_package(Sparse)` consumer path | Maintained | `cmake/SparseConfig.cmake.in`, CMake export install, and `tests/test_cmake_install.sh`. |
| Shared library package | Not claimed | CMake forces static archive behavior and install tests reject shared-library artifacts. |
| Dynamic ABI compatibility | Not claimed | CMake package version compatibility is exact; public header drift is evidence for source/package stability, not ABI stability. |
| Linux support tier | Strongest reviewed source of truth | `.github/workflows/ci.yml` reviewed Makefile/CMake/dead-code lanes plus supplemental checks. |
| macOS support tier | Reviewed Apple Clang path plus supplemental package confidence | `.github/workflows/macos-ci.yml`; install/pkg-config is supplemental. |
| Windows support tier | Reviewed CMake-first subset | `.github/workflows/windows-ci.yml`; excludes Makefile parity, separate install-validation lane, and staged tests. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Package/platform evidence baseline, surface inventory, validation expectations. | Items 1, 2, 4 |
| 2 | Package surface audit across Make, CMake, pkg-config, examples, versioning, and consumers. | Item 1 |
| 3 | ABI support options and evidence requirements. | Item 2 |
| 4 | Static-first versus shared-library/ABI support decision. | Item 2 |
| 5 | Install/export and downstream consumer proof design. | Item 3 |
| 6 | Make install and pkg-config proof. | Item 3 |
| 7 | CMake install/export and pkg-config proof. | Item 3 |
| 8 | Downstream consumer proof batch. | Item 3 |
| 9 | Linux, macOS, and Windows platform-tier contract. | Item 4 |
| 10 | Windows reviewed-scope follow-through. | Item 5 |
| 11 | macOS package/platform follow-through. | Item 5 |
| 12 | Install, README, CMake/pkg-config, and maintainer documentation alignment. | Item 6 |
| 13 | Integrated package, platform, consumer, and documentation validation. | Item 7 |
| 14 | Sprint closeout, residual queue, and downstream handoff. | Item 7 |

## Dependency Order

1. Package and platform inventory must precede package-surface audit.
2. Package-surface audit must precede ABI support options.
3. ABI options must precede the support-tier decision.
4. The support-tier decision must precede install/export proof design.
5. Proof design must precede Make, CMake, pkg-config, and consumer validation.
6. Platform-tier wording must follow package support truth.
7. Windows and macOS follow-through must follow the platform-tier contract.
8. Documentation alignment must follow package/platform validation evidence.
9. Integrated validation must precede closeout and residual handoff.

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation only | `git diff --check`; trailing-whitespace scan over touched docs; local relative Markdown link check when links change. |
| Makefile install logic or `sparse.pc.in` | `bash tests/test_install.sh`; `git diff --check`; relevant format/lint checks if C/H files change. |
| CMake install/export logic or CMake package files | `bash tests/test_cmake_install.sh`; focused CMake configure/build/install checks; `git diff --check`. |
| Example `.c` files | `make examples` or focused example build/run; full C quality chain if examples/public headers are formatted by the project gate. |
| Public headers or installed-header behavior | Public/install-header drift review; `make format && make lint && make test`; install validation when package semantics change. |
| CI workflow files | Syntax/claim review, plus the closest local command equivalent when available; no platform claim without reviewed or documented evidence. |
| Mixed package/build/docs changes | Apply the strongest check required by any touched surface. |

## Day 1 Notes

- Created Sprint 112 working notes and artifact directory.
- Re-read Sprint 112 project-plan scope and the Day 1 plan.
- Reviewed Sprint 100 evidence contracts:
  - package proof template;
  - platform tier template;
  - ABI decision template;
  - consumer validation checklist;
  - static-first package baseline and non-claims.
- Reviewed Sprint 110 evidence:
  - no public/install-header drift;
  - Matrix builder and Matrix Market movement stayed private;
  - Sprint 110 explicitly warned not to infer shared-library/ABI support or
    expanded Windows coverage from unchanged public headers.
- Reviewed Sprint 111 adoption docs and residuals:
  - install and downstream-consumer detail is owned by `INSTALL.md`;
  - user-facing docs should remain concise and evidence-bounded;
  - no public Matrix I/O module, public builder API, shared-library/ABI,
    universal benchmark, or broadened platform claim was made.
- Inventoried package/platform surfaces across:
  - `Makefile`;
  - `CMakeLists.txt`;
  - `cmake/SparseConfig.cmake.in`;
  - `sparse.pc.in`;
  - `tests/test_install.sh`;
  - `tests/test_cmake_install.sh`;
  - `examples/cmake_example/`;
  - `README.md`;
  - `INSTALL.md`;
  - `docs/maintainer_guide.md`;
  - `.github/workflows/ci.yml`;
  - `.github/workflows/macos-ci.yml`;
  - `.github/workflows/windows-ci.yml`.
- Established the support-claim baseline:
  - static archive package is maintained;
  - `pkg-config` and `find_package(Sparse)` describe the installed static
    archive surface;
  - shared-library packaging and dynamic ABI stability remain unclaimed;
  - Linux is strongest reviewed truth;
  - macOS is narrower with supplemental static-first package confidence;
  - Windows is a reviewed CMake-first subset.
- Added Day 1 artifact:
  - `artifacts/day1-package-platform-evidence-baseline.md`.

## Day 2 Notes

- Audited Make install behavior:
  - `make install` installs `libsparse_lu_ortho.a`, public headers,
    generated `sparse_version.h`, and `sparse.pc`;
  - `PREFIX` defaults to `/usr/local`;
  - `DESTDIR` is supported for staged packaging;
  - `make uninstall` removes the static archive, installed sparse include
    directory, and `sparse.pc`;
  - `SPARSE_MUTEX` and `SPARSE_OPENMP` append package link flags where
    configured.
- Audited CMake install/export behavior:
  - `sparse_lu_ortho` is a static target;
  - installed target namespace is `Sparse::`;
  - install interface includes `${CMAKE_INSTALL_INCLUDEDIR}`;
  - public headers install under `${includedir}/sparse`;
  - generated `sparse_version.h` installs with the public headers;
  - `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and
    `SparseTargets.cmake` install under `${libdir}/cmake/Sparse`;
  - CMake package version compatibility is `ExactVersion`.
- Audited `pkg-config` behavior:
  - `sparse.pc.in` emits `prefix`, `libdir`, `includedir`, `Version`,
    `Cflags`, and `Libs`;
  - link flags include `-lsparse_lu_ortho` and `-lm`;
  - optional thread/OpenMP flags are appended by Make/CMake generation paths.
- Audited downstream consumer entry points:
  - `tests/test_install.sh` compiles/runs a generated pkg-config consumer and
    `examples/cmake_example/main.c` against the staged Make install;
  - `tests/test_cmake_install.sh` builds `examples/cmake_example/` against a
    staged CMake install through `find_package(Sparse)`;
  - `examples/cmake_example/` uses installed `<sparse/...>` headers and
    `Sparse::sparse_lu_ortho`.
- Audited versioning and exact-package behavior:
  - `VERSION` is currently `2.2.0`;
  - Make generates `sparse_version.h` from `include/sparse_version.h.in`;
  - CMake configures `SparseConfigVersion.cmake` from project version with
    exact-version compatibility;
  - install scripts compare `pkg-config --modversion sparse` against
    `VERSION`;
  - CMake install validation checks exact-version success and lower-version
    mismatch rejection.
- Identified support-claim status:
  - static-first package wording is consistent across README, `INSTALL.md`,
    CMake comments, tests, and maintainer guide;
  - shared-library packaging remains explicitly deferred;
  - dynamic ABI compatibility remains unclaimed;
  - Windows install validation remains unclaimed beyond the CMake-first
    consumer subset;
  - macOS install/pkg-config proof remains supplemental, not reviewed
    install/export parity.
- Ordered proof gaps for later days:
  - Day 3 should define what additional evidence would be needed for any ABI
    or shared-library claim;
  - Day 4 should decide whether to preserve static-first or attempt broader
    ABI proof;
  - Day 5 should turn the existing install scripts into the proof design for
    the selected tier;
  - Days 6-8 should run or extend Make, CMake, pkg-config, and downstream
    consumer proof.
- Added Day 2 artifact:
  - `artifacts/day2-package-surface-audit.md`.

## Day 3 Notes

- Reviewed Sprint 100 ABI and package proof templates:
  - `templates/abi-decision-template.md`;
  - `templates/package-proof-template.md`;
  - `day11-platform-packaging-evidence-template.md`.
- Inventoried current ABI-adjacent surfaces:
  - public and installed headers under `include/*.h`;
  - generated `sparse_version.h`;
  - static archive `libsparse_lu_ortho.a`;
  - installed CMake target `Sparse::sparse_lu_ortho`;
  - pkg-config library flag `-lsparse_lu_ortho`;
  - exact-version CMake package metadata.
- Confirmed current build products and package metadata:
  - `sparse_lu_ortho` is always built as `STATIC`;
  - `BUILD_SHARED_LIBS=ON` emits a warning and still produces the maintained
    static archive;
  - no shared object, dylib, DLL, import library, SONAME, SOVERSION, symbol
    export map, visibility policy, or runtime-loader proof exists;
  - install tests reject unexpected shared-library artifacts.
- Defined static-first support evidence:
  - Make staged install/uninstall proof;
  - CMake install/export proof;
  - pkg-config compile/link/run proof;
  - installed CMake `find_package(Sparse)` consumer proof;
  - exact-version package metadata;
  - explicit shared-library and ABI non-claims in public and maintainer docs.
- Defined required proof for shared-library or dynamic-ABI support:
  - actual shared artifact generation on supported platforms;
  - install/export metadata for shared artifacts;
  - runtime loader behavior and path handling;
  - symbol export/visibility policy;
  - ABI versioning and compatibility policy;
  - downstream compile/link/run proof against shared artifacts;
  - Linux/macOS/Windows platform ownership;
  - uninstall/cleanup behavior for shared artifacts.
- Set Sprint 110 evidence-use limits:
  - no-public-header drift is useful source/package stability evidence;
  - it is not a binary ABI guarantee;
  - it does not prove shared-library support;
  - it does not expand Windows platform support beyond the reviewed CMake
    subset.
- Compared Day 4 options:
  - preserve static-first support as already earned and validate it again;
  - attempt shared-library/ABI support, which would require substantial build,
    metadata, loader, symbol, CI, and documentation work outside current
    evidence.
- Added Day 3 artifact:
  - `artifacts/day3-abi-support-options.md`.

## Day 4 Notes

- Reviewed Day 3 ABI options against Sprint 112 package and platform goals.
- Made the Sprint 112 package support decision:
  - preserve static-first as the explicit Epic 10 package support tier;
  - do not attempt shared-library packaging in Sprint 112;
  - do not claim dynamic ABI compatibility.
- Recorded decision rationale:
  - current Make and CMake build products intentionally produce a static
    archive;
  - install scripts already validate the absence of shared artifacts;
  - CMake package version compatibility is exact-version only;
  - no symbol export policy, shared runtime-loader proof, SONAME/SOVERSION, or
    cross-platform shared artifact validation exists.
- Froze non-claims for Sprint 112:
  - no shared-library package support;
  - no dynamic ABI stability;
  - no ABI compatibility inferred from public-header stability;
  - no Windows Makefile parity or separate install-validation lane;
  - no macOS reviewed install/export parity beyond supplemental package
    confidence.
- Scoped validation for Days 5-8 to the selected static-first tier:
  - Make staged install/uninstall;
  - pkg-config metadata, compile/link, and run checks;
  - CMake install/export;
  - CMake `find_package(Sparse)` consumer;
  - exact-version package behavior;
  - downstream public-header consumer checks.
- Identified documentation surfaces to revisit only after validation evidence:
  - `INSTALL.md`;
  - `README.md`;
  - `docs/maintainer_guide.md`;
  - CMake and pkg-config comments/templates;
  - platform CI comments if Days 9-11 change reviewed scope.
- Added Day 4 artifact:
  - `artifacts/day4-abi-support-decision.md`.

## Day 5 Notes

- Converted the Day 4 static-first support decision into a concrete proof
  design for Days 6-8.
- Defined the Make install validation command:
  - `bash tests/test_install.sh`.
- Defined Make install proof expectations:
  - staged install under a temporary `PREFIX`;
  - static archive present;
  - no shared-library artifacts present;
  - all public headers plus generated `sparse_version.h` present;
  - `sparse.pc` present;
  - `pkg-config --cflags`, `--libs`, and `--modversion` valid;
  - generated pkg-config consumer compiles, links, and runs;
  - `examples/cmake_example/main.c` compiles, links, and runs through
    pkg-config flags;
  - `make uninstall` removes installed library, headers, and `sparse.pc`.
- Defined the CMake install/export validation command:
  - `bash tests/test_cmake_install.sh`.
- Defined CMake proof expectations:
  - staged CMake configure, build, and install pass;
  - static archive present;
  - no shared-library artifacts present;
  - installed headers present;
  - `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and
    `SparseTargets.cmake` present;
  - `sparse.pc` present;
  - `examples/cmake_example/` configures, builds, and runs through
    `find_package(Sparse)`;
  - exact installed version is accepted;
  - lower mismatched version is rejected when applicable;
  - pkg-config version matches `VERSION`.
- Selected downstream consumers:
  - generated minimal pkg-config consumer from `tests/test_install.sh`;
  - `examples/cmake_example/main.c` built through pkg-config;
  - `examples/cmake_example/` built through installed CMake package metadata.
- Recorded repeatability rules:
  - both scripts use `mktemp -d`;
  - both scripts clean temporary state with `trap`;
  - Make proof uses `make clean` before install;
  - CMake proof uses isolated build and install directories;
  - package lookups are scoped by `PKG_CONFIG_PATH` or `CMAKE_PREFIX_PATH`;
  - consumers use installed `<sparse/...>` headers and avoid private headers.
- Added Day 5 artifact:
  - `artifacts/day5-install-consumer-proof-design.md`.

## Day 6 Notes

- Ran the Make install proof command:
  - `bash tests/test_install.sh`.
- Validation passed:
  - `14` checks passed;
  - `0` checks failed.
- Confirmed staged Make install behavior:
  - static library installed;
  - no shared-library artifacts installed;
  - all `19` expected headers installed, including generated
    `sparse_version.h`;
  - `sparse.pc` installed.
- Confirmed pkg-config behavior:
  - `pkg-config --cflags sparse` returned an include path;
  - `pkg-config --libs sparse` returned the library flag;
  - `pkg-config --modversion sparse` returned `2.2.0`.
- Confirmed downstream consumer behavior:
  - generated pkg-config consumer compiled and linked;
  - generated pkg-config consumer ran successfully;
  - `examples/cmake_example/main.c` compiled and linked through pkg-config;
  - maintained example source ran successfully through the staged install.
- Confirmed uninstall behavior:
  - installed library removed;
  - installed headers removed;
  - installed `sparse.pc` removed.
- No Makefile, pkg-config, install-script, or documentation fix was required by
  Day 6 proof.
- Kept Day 6 claim bounded:
  - this proves the local Unix-side Make static install and pkg-config
    consumer path;
  - it does not prove CMake install/export behavior;
  - it does not prove shared-library packaging;
  - it does not prove dynamic ABI stability;
  - it does not prove Windows install-validation parity or platform-wide
    reviewed install parity.
- Added Day 6 artifact:
  - `artifacts/day6-make-install-proof.md`.

## Day 7 Notes

- Ran the CMake install/export proof command:
  - `bash tests/test_cmake_install.sh`.
- Validation passed:
  - `16` checks passed;
  - `0` checks failed;
  - `0` checks skipped.
- Confirmed CMake package flow:
  - CMake configure passed;
  - CMake build passed;
  - CMake install passed.
- Confirmed staged CMake installed artifacts:
  - static library installed;
  - no shared-library artifacts installed;
  - `19` headers installed;
  - `SparseConfig.cmake` installed;
  - `SparseConfigVersion.cmake` installed;
  - `SparseTargets.cmake` installed;
  - `sparse.pc` installed.
- Confirmed downstream CMake consumer behavior:
  - `examples/cmake_example/` configured with `find_package(Sparse)`;
  - `examples/cmake_example/` built successfully;
  - installed CMake consumer ran successfully.
- Confirmed package version behavior:
  - exact installed version was accepted;
  - lower mismatched version was rejected;
  - `pkg-config --modversion sparse` returned `2.2.0`.
- No CMake, pkg-config, package-template, or documentation fix was required by
  Day 7 proof.
- Kept Day 7 claim bounded:
  - this proves the local CMake install/export and installed
    `find_package(Sparse)` consumer path for the selected static-first tier;
  - it does not prove dynamic ABI compatibility;
  - it does not prove shared-library runtime-loader behavior;
  - it does not prove Makefile parity on every platform;
  - it does not create a Windows separate install-validation lane.
- Added Day 7 artifact:
  - `artifacts/day7-cmake-install-export-proof.md`.

## Day 8 Notes

- Compared Day 6 Make/pkg-config consumer proof and Day 7 CMake
  `find_package(Sparse)` consumer proof.
- Confirmed selected static-first downstream consumer coverage:
  - generated pkg-config consumer;
  - `examples/cmake_example/main.c` compiled and linked through pkg-config;
  - `examples/cmake_example/` configured, built, and ran through installed
    CMake package metadata;
  - exact-version CMake consumer probe;
  - mismatched-version CMake probe.
- Confirmed public-header coverage across installed consumers:
  - `<sparse/sparse_types.h>`;
  - `<sparse/sparse_matrix.h>`;
  - `<sparse/sparse_lu.h>`;
  - `<sparse/sparse_lu_csr.h>`;
  - generated `sparse_version.h` through version macros.
- Confirmed common workflow coverage:
  - version macro access;
  - matrix creation;
  - matrix insertion;
  - matrix cleanup;
  - one direct solver path through LU/CSR solve;
  - package metadata through pkg-config and CMake.
- Reviewed installed-consumer documentation:
  - `INSTALL.md` verification section matches the observed proof commands and
    bounded claim language;
  - `examples/README.md` points installed-consumer users to
    `examples/cmake_example/` and `INSTALL.md`;
  - `docs/solver_selection.md` includes installed CMake consumption in the
    example handoff table;
  - README compact package summary points to `INSTALL.md`.
- No additional Day 8 consumer source or docs update was required:
  - the existing consumers prove the selected static-first package tier;
  - compressed-input and Matrix Market workflows remain local/build-tree
    adoption examples and are not required for the installed package proof
    claim;
  - private headers are not required by any installed consumer proof.
- Kept Day 8 claim bounded:
  - this proves installed public-header consumers for the selected static-first
    package tier;
  - it does not prove shared-library support, dynamic ABI compatibility,
    Windows install-validation parity, or macOS reviewed install/export
    parity.
- Added Day 8 artifact:
  - `artifacts/day8-downstream-consumer-proof.md`.

## Day 9 Notes

- Reviewed current platform evidence across:
  - `.github/workflows/ci.yml`;
  - `.github/workflows/macos-ci.yml`;
  - `.github/workflows/windows-ci.yml`;
  - `README.md`;
  - `INSTALL.md`;
  - `docs/maintainer_guide.md`;
  - Day 6 Make install proof;
  - Day 7 CMake install/export proof;
  - Day 8 downstream consumer proof.
- Defined the Sprint 112 platform support tiers:
  - Linux remains the strongest reviewed source of truth, with reviewed
    Makefile compile-quality, reviewed CMake parity, and dead-code
    completeness paths;
  - Linux direct runtime, sanitizer, TSan/OpenMP, benchmark, and coverage
    lanes are supplemental signals;
  - macOS has a reviewed Apple Clang lane covering compile-quality, CMake
    parity, `wall-check`, and sanitizer checks;
  - macOS Homebrew GCC remains supplemental second-compiler confidence;
  - macOS Make install/`pkg-config` proof remains supplemental static-first
    package confidence, not reviewed install/export parity;
  - Windows remains the reviewed MSVC CMake-first consumer subset, with
    configure, build, `ctest -N`, and full `ctest`.
- Confirmed current Windows reviewed-scope guard:
  - expected registered CTest count is `51`;
  - staged exclusions remain `test_threads`, `test_sprint4_integration`, and
    `test_fuzz`, including the bounded lifecycle property/fuzz lane.
- Preserved explicit platform non-claims:
  - no Windows Makefile parity;
  - no Windows separate reviewed install-validation lane;
  - no macOS full reviewed install/export parity;
  - no shared-library package support;
  - no dynamic ABI compatibility;
  - no platform-wide runtime-loader, SONAME/SOVERSION, DLL/import-library, or
    package-manager support claim.
- Confirmed docs and workflow comments already broadly match the Day 9
  contract:
  - README carries the compact cross-platform CI summary;
  - `INSTALL.md` distinguishes local install scripts from reviewed platform
    confidence;
  - `docs/maintainer_guide.md` owns detailed package/platform interpretation;
  - workflow comments already describe Linux, macOS, and Windows lane
    boundaries.
- Deferred any wording changes until Days 10-12 unless Windows or macOS
  follow-through changes the evidence.
- Added Day 9 artifact:
  - `artifacts/day9-platform-tier-contract.md`.

## Day 10 Notes

- Reviewed current Windows CI:
  - `windows-2022` runner;
  - Visual Studio 17 2022 x64 CMake configure;
  - Release CMake build;
  - `ctest -N` registration inspection;
  - full `ctest` execution;
  - `EXPECTED_WINDOWS_CTEST_COUNT=51`.
- Reviewed Windows CMake registration gates:
  - `test_threads` and `test_sprint4_integration` are gated by
    `Threads_FOUND AND NOT WIN32` because those test sources use pthread APIs;
  - `test_fuzz` is gated by `NOT WIN32 AND NOT MSVC`;
  - POSIX-only benchmark binaries are gated by `NOT WIN32`;
  - regular portable tests remain part of the reviewed Windows CMake subset.
- Decided not to promote any Windows staged exclusion in Day 10:
  - no Windows-native pthread replacement or thread-test portability layer was
    added;
  - no Windows fuzz/property proof was added;
  - no Windows Makefile parity was added;
  - no Windows dead-code parity was added;
  - no separate Windows install-validation lane was added.
- Preserved Windows non-claims:
  - reviewed scope remains MSVC CMake-first consumer subset only;
  - no Makefile parity;
  - no Unix install-script parity;
  - no separate reviewed install-validation lane;
  - no bounded lifecycle property/fuzz evidence;
  - no shared-library, dynamic ABI, DLL/import-library, runtime-loader, or
    package-manager support claim.
- Reviewed Windows-related wording in:
  - `.github/workflows/windows-ci.yml`;
  - `INSTALL.md`;
  - `README.md`;
  - `docs/maintainer_guide.md`.
- No Day 10 documentation or workflow edit was justified because the existing
  wording already matches the Day 9 platform-tier contract and the reviewed
  Windows evidence.
- Added Day 10 artifact:
  - `artifacts/day10-windows-follow-through.md`.

## Day 11 Notes

- Reviewed current macOS CI:
  - Apple Clang reviewed path runs `make quality-review-compile`,
    `make quality-review-cmake`, `make wall-check`, and `make sanitize`;
  - Homebrew GCC leg runs supplemental direct build/test and `wall-check`;
  - Make install/`pkg-config` job runs `bash tests/test_install.sh` as
    supplemental static-first package confidence.
- Captured local macOS environment:
  - macOS `15.7.3` (`24G419`);
  - Darwin kernel `24.6.0`;
  - Apple Clang `11.0.0`;
  - CMake `4.3.2`;
  - pkg-config `2.5.1`.
- Ran practical local macOS package validation:
  - `bash tests/test_install.sh` passed with 14 passed, 0 failed;
  - `bash tests/test_cmake_install.sh` passed with 16 passed, 0 failed,
    0 skipped.
- Confirmed local package behavior:
  - static library installed;
  - no shared-library artifacts installed;
  - all 19 headers installed;
  - `sparse.pc` installed;
  - pkg-config compile/link/run consumers passed;
  - CMake installed `SparseConfig.cmake`, `SparseConfigVersion.cmake`,
    `SparseTargets.cmake`, and `sparse.pc`;
  - installed CMake consumer configured, built, and ran;
  - exact-version CMake package lookup succeeded;
  - mismatched-version CMake package lookup was rejected.
- Decided no Day 11 docs or workflow edits were required:
  - existing macOS workflow comments already distinguish reviewed Apple Clang,
    supplemental Homebrew GCC, and supplemental Make install/`pkg-config`;
  - `INSTALL.md`, README, and `docs/maintainer_guide.md` already keep macOS
    package/platform wording evidence-bounded.
- Preserved macOS non-claims:
  - no full reviewed install/export parity;
  - no reviewed dead-code or coverage parity;
  - no dynamic ABI compatibility;
  - no shared-library, dylib, install-name, runtime-loader, or symbol
    visibility support;
  - no package-manager support;
  - no macOS TSan claim beyond the documented Linux-side maintained TSan lane.
- Added Day 11 artifact:
  - `artifacts/day11-macos-follow-through.md`.

## Day 12 Notes

- Reviewed documentation alignment inputs:
  - Day 4 ABI support decision;
  - Day 6 Make install proof;
  - Day 7 CMake install/export proof;
  - Day 8 downstream consumer proof;
  - Day 9 platform-tier contract;
  - Day 10 Windows follow-through;
  - Day 11 macOS follow-through.
- Rechecked package/platform wording in:
  - `README.md`;
  - `INSTALL.md`;
  - `docs/maintainer_guide.md`;
  - `CMakeLists.txt`;
  - `sparse.pc.in`;
  - `cmake/SparseConfig.cmake.in`;
  - `.github/workflows/ci.yml`;
  - `.github/workflows/macos-ci.yml`;
  - `.github/workflows/windows-ci.yml`.
- Left README unchanged:
  - it already provides the compact package summary;
  - it points install and downstream-consumer users to `INSTALL.md`;
  - it preserves the static-first package boundary.
- Left `INSTALL.md` unchanged:
  - it already owns operational install and validation steps;
  - it distinguishes local install scripts from reviewed platform confidence;
  - it preserves macOS and Windows non-claims.
- Left CMake, pkg-config, and workflow comments unchanged:
  - CMake comments already preserve static-first and exact-version package
    behavior;
  - `sparse.pc.in` does not imply ABI compatibility;
  - workflow comments already match the reviewed/supplemental/staged platform
    split.
- Updated `docs/maintainer_guide.md` with a Sprint 112 package/platform proof
  snapshot covering:
  - selected static-first tier;
  - local Make install proof;
  - local CMake install/export proof;
  - Linux, macOS, and Windows platform-tier interpretation;
  - Windows CTest count and staged exclusions;
  - explicit package/platform non-claims.
- Added Day 12 artifact:
  - `artifacts/day12-packaging-documentation-alignment.md`.

## Day 13 Notes

- Ran integrated package validation:
  - `bash tests/test_install.sh` passed with 14 passed, 0 failed;
  - `bash tests/test_cmake_install.sh` passed with 16 passed, 0 failed,
    0 skipped.
- Confirmed Make install proof still covers:
  - static archive install;
  - no shared-library artifacts;
  - 19 installed headers;
  - `sparse.pc`;
  - pkg-config compile/link/run consumers;
  - uninstall cleanup.
- Confirmed CMake install/export proof still covers:
  - CMake configure, build, and install;
  - static archive install;
  - no shared-library artifacts;
  - 19 installed headers;
  - `SparseConfig.cmake`, `SparseConfigVersion.cmake`,
    `SparseTargets.cmake`, and `sparse.pc`;
  - installed `examples/cmake_example/` configure/build/run through
    `find_package(Sparse)`;
  - exact-version success;
  - mismatched-version rejection;
  - pkg-config version `2.2.0`.
- Checked drift:
  - no `.c`, `.h`, `include/*`, `src/*`, or `tests/*` files changed;
  - no `Makefile`, `CMakeLists.txt`, `sparse.pc.in`,
    `cmake/SparseConfig.cmake.in`, or workflow files changed;
  - no public API, install-header, package metadata, helper-target, or
    reviewed CTest scope drift was introduced.
- Verified package/platform wording remains consistent across:
  - README;
  - `INSTALL.md`;
  - `docs/maintainer_guide.md`;
  - CMake/package metadata comments;
  - Linux, macOS, and Windows workflow comments;
  - Sprint 112 artifacts.
- Added Day 13 artifact:
  - `artifacts/day13-integrated-validation.md`.

## Day 14 Notes

- Reviewed all Sprint 112 artifacts and working notes from Days 1-13.
- Confirmed all seven Sprint 112 project-plan items are complete:
  - Package Surface Audit;
  - ABI Support Decision;
  - Install/Consumer Proof Batch;
  - Platform Tier Contract;
  - Windows/macOS Follow-Through;
  - Packaging Docs;
  - Validation and Closeout.
- Final package support tier:
  - static-first package surface;
  - exact-version CMake package metadata;
  - pkg-config and CMake consumers for the installed static archive;
  - no shared-library package claim;
  - no dynamic ABI compatibility claim.
- Final platform support tiers:
  - Linux is strongest reviewed source of truth;
  - macOS has a reviewed Apple Clang lane plus supplemental Homebrew GCC and
    static-first Make install/`pkg-config` confidence;
  - Windows remains the reviewed MSVC CMake-first consumer subset with
    expected registered CTest count `51`.
- Recorded residual deferred debt for:
  - optional Linux CI promotion of local install scripts;
  - optional reviewed macOS CMake install/export parity;
  - optional Windows install-validation lane;
  - Windows-native thread-test ownership;
  - Windows fuzz/property portability;
  - macOS coverage parity;
  - Homebrew GCC version drift;
  - macOS TSan revisit;
  - future shared-library/dynamic ABI product contract;
  - future package-manager support.
- Added Day 14 artifact:
  - `artifacts/day14-closeout-handoff.md`.
