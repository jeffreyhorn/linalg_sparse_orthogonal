# Day 6 Package, ABI, And Platform Baseline

## Scope

Day 6 freezes the current install/export, package metadata, ABI, and platform
support baseline for Epic 14. It records where the static-first package contract
is already maintained, where Windows support is intentionally narrower, and
which shared-library and dynamic-ABI blockers must remain non-claims unless a
future sprint explicitly closes them.

This artifact does not run install validations or change package behavior.

## Inventory Inputs

| Surface | Files reviewed | Day 6 purpose |
| --- | --- | --- |
| Public install contract | `INSTALL.md`, README installation sections | Capture user-facing static-first package claims and non-claims. |
| Maintainer package policy | `docs/maintainer_guide.md` | Capture authoritative interpretation, proof-owner rows, and platform support tiers. |
| Build/install metadata | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, `sparse.pc.in`, `VERSION` | Capture installed archive, exported CMake package, `pkg-config`, and version metadata ownership. |
| Unix install proofs | `tests/test_install.sh`, `tests/test_cmake_install.sh` | Capture Make install/`pkg-config` and CMake install/export proof scope. |
| Static deferral proof | `scripts/static_package_deferral_check.sh` | Capture shared-library, dynamic ABI, loader, and package selector blocker checks. |
| Hosted package lanes | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | Capture reviewed Linux/macOS/Windows package proof boundaries. |

## Static-First Package Baseline

| Contract element | Current source of truth | Validation owner | Claim boundary |
| --- | --- | --- | --- |
| Static archive install | `Makefile`, `CMakeLists.txt` install rules | `tests/test_install.sh`, `tests/test_cmake_install.sh`, Linux/macOS/Windows package lanes | Installs `libsparse_lu_ortho.a` on Unix-like systems and `sparse_lu_ortho.lib` on Windows/MSVC. |
| Installed public headers | `include/*.h`, generated `sparse_version.h` from `include/sparse_version.h.in` | Install scripts derive the expected count from checked-in public headers plus generated `sparse_version.h` (currently 19 installed headers) | Installed headers are public package surface; header presence is not a dynamic ABI guarantee. |
| `pkg-config` metadata | `sparse.pc.in` | Unix `tests/test_install.sh`, Unix CMake install script metadata checks, Windows metadata inspection | Describes static archive package metadata and link flags; no `Libs.private`, static/shared selector, or package-manager claim. |
| CMake package export | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in` | `tests/test_cmake_install.sh`, Windows CMake install/downstream lane | Exports `Sparse::sparse_lu_ortho` as `STATIC IMPORTED` and supports `find_package(Sparse)`. |
| Version metadata | `VERSION`, generated install files | Install scripts and Windows lane exact-version checks | Exact-version package matching only; not broad ABI compatibility. |
| Shared-library request handling | `CMakeLists.txt`, `scripts/static_package_deferral_check.sh` | Static deferral guard in Linux/macOS package lanes | `BUILD_SHARED_LIBS=ON` is rejected at configure time. |

## Reviewed Package Proofs

| Platform | Reviewed package proof | Validated surface | Explicit non-claims |
| --- | --- | --- | --- |
| Linux | `.github/workflows/ci.yml::package-contract` | Runs `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, and `bash scripts/static_package_deferral_check.sh`. | No shared-library packaging, dynamic ABI compatibility, runtime-loader behavior, or package-manager support. |
| macOS | `.github/workflows/macos-ci.yml::install-and-pkgconfig` and `::cmake-install-export` | Runs Make install/`pkg-config` proof, CMake install/export proof, and static deferral guard on hosted macOS. | No shared-library packaging, dynamic ABI compatibility, runtime-loader compatibility, package-manager support, static/shared selectors, or broad macOS parity. |
| Windows | `.github/workflows/windows-ci.yml::install-and-downstream` | Runs CMake configure/build/install, checks installed static `.lib`, derives the expected installed-header count from checked-in public headers plus generated `sparse_version.h` (currently 19), checks CMake package files, `sparse.pc` metadata, installed downstream CMake consumers, exact-version behavior, mismatched-version rejection, no DLL/shared imported metadata, and no unsupported loader/static-shared selector metadata. | No Windows Makefile parity, no Windows `pkg-config` execution parity, no package-manager support, no shared-library support, no dynamic ABI support, no runtime-loader behavior, and no broad Windows parity. |

## Unix Install Proof Surface

| Script | Owned checks | Day 6 interpretation |
| --- | --- | --- |
| `tests/test_install.sh` | `make clean`, `make install PREFIX=...`, static archive installed, no shared artifacts, installed header count equals checked-in public headers plus generated version header, `sparse.pc` installed, `pkg-config` existence/exact-version/prefix/libdir/includedir/cflags/libs/static-libs checks, no `Libs.private`, static archive description, no unsupported package/ABI words, basic consumer compile/link/run, maintained example compile/link/run, uninstall cleanup. | Unix-side Make install and `pkg-config` proof only. It should not be read as Windows `pkg-config` execution parity or package-manager support. |
| `tests/test_cmake_install.sh` | CMake configure/build/install, static archive installed, no shared artifacts, installed header count, CMake package files, static imported target metadata, no shared imported metadata, no loader/static-shared selector metadata, no source/build path leaks, static `.pc` metadata, installed CMake example, exact-version consumer, mismatched-version rejection. | Installed CMake export proof for Unix-like local validation and reviewed hosted lanes. It supports `find_package(Sparse)` static archive consumption only. |

## Windows Package Parity Delta

| Surface | Current Windows status | Delta from Unix-side package surface |
| --- | --- | --- |
| CMake configure/build/CTest | Reviewed on `windows-2022` MSVC with expected 59 CTest registrations and full `ctest`. | Equivalent reviewed CMake consumer subset, but CMake-first only. |
| CMake install/export | Reviewed hosted Windows lane. | Validates installed static `.lib`, CMake package metadata, and CMake downstream consumers. |
| `sparse.pc` metadata | Installed and inspected on Windows. | Metadata syntax/content is checked; `pkg-config` is not executed as a Windows consumer proof. |
| Makefile install/uninstall | Not claimed on Windows. | Unix-side only until a future sprint explicitly ports and reviews it. |
| `pkg-config` downstream compile/link/run | Not claimed on Windows. | Unix-side only; Windows remains `find_package(Sparse)`/CMake downstream scoped. |
| Package-manager distribution | Not claimed on any platform. | No Homebrew, apt, dnf, pacman, vcpkg, conan, or other package-manager support. |

## Shared-Library And Dynamic-ABI Blocker List

Shared-library support remains intentionally deferred. A future product decision
would need to add and review all of the following before any shared-library,
dynamic ABI, or runtime-loader claim is made:

| Blocker | Current guard or owner |
| --- | --- |
| Public export/import macro policy such as `SPARSE_API`, `SPARSE_EXPORT`, or `SPARSE_IMPORT` | `scripts/static_package_deferral_check.sh` rejects those macros under `include/`. |
| Symbol visibility policy | `BUILD_SHARED_LIBS=ON` rejection wording and static deferral guard require visibility blocker wording. |
| Dynamic ABI policy | `INSTALL.md`, README, maintainer guide, and static deferral guard keep ABI compatibility as a non-claim. |
| Linux SONAME metadata | Static deferral guard checks rejection wording; CMake metadata must not gain shared ABI fields without decision. |
| macOS install-name/RPATH metadata | Static deferral guard checks rejection wording and package metadata absence. |
| Windows DLL/import-library behavior | Windows lane rejects DLL artifacts and shared imported metadata; deferral guard requires blocker wording. |
| Installed shared consumer proof | Explicitly absent and named as a blocker. |
| Runtime-loader validation | Explicitly absent and named as a blocker. |
| Static/shared package selectors | `sparse.pc.in` and `cmake/SparseConfig.cmake.in` must not expose selectors before a support decision. |

## Package Metadata Ownership Map

| Metadata or wording | Primary owner | Must stay synchronized with |
| --- | --- | --- |
| Package version | `VERSION` | `include/sparse_version.h.in`, generated `sparse_version.h`, `SparseConfigVersion.cmake`, `sparse.pc`. |
| Installed header count | `include/*.h` plus generated version header | `tests/test_install.sh`, `tests/test_cmake_install.sh`, Windows install lane expected count. |
| Library source/install target | `CMakeLists.txt`, `Makefile`, `build-metadata/library_sources.txt` | `scripts/check_library_sources.py`, install scripts, CMake export metadata. |
| CMake package config | `cmake/SparseConfig.cmake.in`, `CMakeLists.txt` export/version rules | `tests/test_cmake_install.sh`, Windows install/downstream lane, `INSTALL.md`. |
| `pkg-config` template | `sparse.pc.in` | `tests/test_install.sh`, `tests/test_cmake_install.sh`, Windows metadata inspection, `INSTALL.md`, README. |
| Static deferral wording | `CMakeLists.txt`, README, `INSTALL.md`, `docs/maintainer_guide.md` | `scripts/static_package_deferral_check.sh`, CI package lanes. |
| Platform support wording | README, `INSTALL.md`, `docs/maintainer_guide.md`, workflow comments | Linux/macOS/Windows CI lane names and scripts. |
| Report-index package rows | `tests/corpus/manifests/report_families.tsv`, `scripts/normalize_report_index.py` | Package proof-owner files and maintainer guide report-index interpretation. |

## Sprint 162 Prerequisites

Sprint 162 package decision work should begin from these prerequisites:

1. Preserve static-first install/export as the selected supported package tier
   unless a new product decision explicitly changes it.
2. Treat Linux and macOS Unix-side `pkg-config` execution as reviewed package
   proof, but keep Windows `pkg-config` execution out of scope.
3. Treat Windows CMake install/downstream validation as reviewed and real, but
   narrower than Unix Makefile/`pkg-config` parity.
4. Keep `BUILD_SHARED_LIBS=ON` rejected until export/import, symbol visibility,
   dynamic ABI, platform loader metadata, installed shared consumer proof, and
   runtime-loader validation exist.
5. Update README, `INSTALL.md`, `docs/maintainer_guide.md`, workflow comments,
   install scripts, `sparse.pc.in`, CMake package templates, and static
   deferral guard together when package claims change.
6. Run the relevant install proofs before citing install-run results; package
   report-index proof-owner rows are source-controlled ownership rows, not a
   substitute for a fresh install validation run.

## Completion Check

- Static-first support is separated from shared-library and ABI support.
- Linux/macOS reviewed static-first package proofs are recorded.
- Windows CMake install/downstream validation is recorded without implying
  Windows Makefile or `pkg-config` execution parity.
- Shared-library, dynamic ABI, runtime-loader, and package-manager blockers are
  listed.
- Package metadata owners and synchronization requirements are captured.
- Sprint 162 package decision prerequisites are recorded.
