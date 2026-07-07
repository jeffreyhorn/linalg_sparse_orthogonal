# Day 14 Closeout and Handoff

## Purpose

Day 14 closes Sprint 112 by consolidating the package, ABI, platform,
install/export, consumer, documentation, and validation evidence from the prior
days. It records which Sprint 112 project-plan items are complete, which work
remains deferred, and what Sprint 113 or later should inherit.

## Completed Item Checklist

| Item # | Item | Status | Evidence |
|---:|---|---|---|
| 1 | Package Surface Audit | Complete | Day 2 audited Make install, CMake install/export, pkg-config, examples, downstream consumers, versioning, exact-package behavior, and stale claims. |
| 2 | ABI Support Decision | Complete | Day 4 selected the static-first package tier and kept shared-library packaging and dynamic ABI compatibility as explicit non-claims. |
| 3 | Install/Consumer Proof Batch | Complete | Days 5-8 designed and ran Make install, CMake install/export, pkg-config, exact-version, and downstream consumer proof. |
| 4 | Platform Tier Contract | Complete | Day 9 defined Linux, macOS, and Windows reviewed, supplemental, staged, local-only, and unsupported lanes. |
| 5 | Windows/macOS Follow-Through | Complete | Day 10 kept Windows staged exclusions explicit; Day 11 validated local macOS package commands and preserved macOS reviewed/supplemental boundaries. |
| 6 | Packaging Docs | Complete | Day 12 updated maintainer proof detail and verified README, INSTALL, CMake, pkg-config, and workflow comments already matched support truth. |
| 7 | Validation and Closeout | Complete | Day 13 reran package validation and drift checks; Day 14 records final handoff and final documentation hygiene. |

## Final Package Support Tier

Sprint 112 closes with the maintained package tier explicitly static-first:

- maintained installed artifact is the static archive package surface;
- `pkg-config` and `find_package(Sparse)` describe the installed static
  archive surface;
- CMake package version compatibility remains exact-version only;
- generated `sparse_version.h`, `SparseConfigVersion.cmake`, and `sparse.pc`
  all derive version truth from `VERSION`;
- shared-library package support is not claimed;
- dynamic ABI compatibility is not claimed;
- SONAME/SOVERSION, symbol export stability, DLL/import-library, dylib
  install-name, runtime-loader, and package-manager support are not claimed.

## Final Platform Support Tiers

| Platform | Final Sprint 112 tier |
|---|---|
| Linux | Strongest reviewed source of truth: reviewed Makefile compile-quality, reviewed CMake parity, and dead-code completeness, with supplemental runtime, sanitizer, TSan/OpenMP, benchmark, and coverage signals. |
| macOS | Reviewed Apple Clang lane plus supplemental Homebrew GCC and supplemental static-first Make install/`pkg-config` confidence. Local CMake install/export proof was recorded but not promoted to reviewed macOS install/export parity. |
| Windows | Reviewed MSVC CMake-first consumer subset only, with CMake configure/build, `ctest -N`, full `ctest`, and expected registered CTest count `51`. |

## Validation Summary

| Validation | Latest Sprint 112 result |
|---|---|
| Make install / pkg-config proof | Day 13: `bash tests/test_install.sh` passed with 14 passed, 0 failed. |
| CMake install/export proof | Day 13: `bash tests/test_cmake_install.sh` passed with 16 passed, 0 failed, 0 skipped. |
| C/header drift | Day 13: no `.c`, `.h`, `include/*`, `src/*`, or `tests/*` files changed. |
| Build/package/workflow drift | Day 13: no `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, or workflow files changed. |
| Documentation alignment | Day 12/13: README, INSTALL, maintainer guide, package metadata comments, and workflow comments agree on static-first package and platform-tier boundaries. |
| Final docs hygiene | Day 14: final `git diff --check`, trailing-whitespace scan, and local Markdown link check are required after this artifact lands. |

## Residual Deferred Debt

These residuals are dependency-ordered and intentionally avoid duplicating the
work already completed in Sprint 112.

| Order | Residual | Owner Sprint | Dependency / reason |
|---:|---|---|---|
| 1 | Decide whether local Unix install scripts should become reviewed Linux CI lanes. | Sprint 113+ | Requires an explicit CI-product decision; Sprint 112 kept them local proof surfaces. |
| 2 | Add a reviewed macOS CMake install/export lane if the project wants full macOS install/export parity. | Sprint 113+ | Depends on preserving the Day 11 boundary that local proof is not reviewed macOS parity. |
| 3 | Add a separate reviewed Windows install-validation lane if Windows installed-package support should be claimed. | Sprint 113+ | Requires `cmake --install`, installed target lookup, and downstream compile/link/run proof under MSVC. |
| 4 | Add a Windows-native thread-test owner before promoting `test_threads` or `test_sprint4_integration`. | Sprint 113+ | Current tests use pthread APIs and remain intentionally gated off Windows. |
| 5 | Make `test_fuzz` portable and reviewed under MSVC before claiming Windows fuzz/property coverage. | Sprint 113+ | Current Windows reviewed subset excludes `test_fuzz`; bounded lifecycle property/fuzz evidence is not Windows evidence. |
| 6 | Promote macOS coverage parity only after gcov/lcov/gcovr backend behavior is stable enough to own as reviewed evidence. | Sprint 113+ | Current macOS coverage guidance is operational but not reviewed parity. |
| 7 | Revisit Homebrew GCC version assumptions when Homebrew's default GCC changes. | Sprint 113+ | macOS supplemental second-compiler lane currently documents `gcc-15` as the expected Homebrew toolchain. |
| 8 | Revisit macOS TSan only if the upstream dyld/runtime limitation is resolved and a reviewed lane is added. | Sprint 113+ | Maintained TSan evidence remains Linux-side. |
| 9 | Add shared-library/dynamic ABI support only as a separate product contract. | Future Epic | Requires build rules, package metadata, runtime-loader proof, symbol policy, versioning policy, and platform ownership across supported systems. |
| 10 | Add package-manager support only after package recipes and install/consumer proof exist. | Future Epic | No vcpkg, Homebrew formula, Chocolatey, winget, distro package, or equivalent package-manager proof exists today. |

## Handoff Summary

- Sprint 112 gives Sprint 113 a clean package/platform baseline: static-first,
  exact-version package metadata, validated local install/export consumers, and
  explicit platform non-claims.
- Sprint 113 should not infer any broader ABI, shared-library, Windows,
  macOS, or package-manager support from Sprint 112 evidence.
- Future package/platform work should either preserve the static-first support
  tier or explicitly create new reviewed lanes and update public claims only
  after those lanes pass.

## Completion Criteria

- All seven Sprint 112 project-plan items are closed.
- Deferred package/platform work is dependency-ordered and non-duplicative.
- Final support-tier and platform-tier claims are evidence-bounded.
- Sprint 113 and later Epic 10 closeout work have a clear package/platform
  handoff.
