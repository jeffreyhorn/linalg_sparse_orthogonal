# Sprint 167 Day 7: Package And Install Evidence Inventory

## Purpose

Day 7 inventories the static-first package and install evidence surfaces. The
goal is to separate supported package behavior from unsupported package, ABI,
runtime-loader, shared-library, package-manager, and Windows parity claims.

## Source Files Reviewed

| Surface | Files | Day 7 role |
| --- | --- | --- |
| Make install/uninstall | `Makefile` | Unix-style static archive install, header install, generated version header install, pkg-config generation, and uninstall cleanup. |
| CMake install/export | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in` | Static target install, exported `Sparse::sparse_lu_ortho`, exact-version CMake package metadata, and install-prefix include/archive paths. |
| pkg-config metadata | `sparse.pc.in` | Static archive metadata for `pkg-config --cflags --libs sparse`. |
| Unix install proof | `tests/test_install.sh` | Make install/uninstall, pkg-config command execution, downstream compile/link/run, exact version, and unsupported metadata checks. |
| CMake install proof | `tests/test_cmake_install.sh` | CMake install/export, installed CMake package metadata, downstream `find_package(Sparse)`, exact/mismatch version behavior, and static metadata checks. |
| Static deferral guard | `scripts/static_package_deferral_check.sh` | Fail-closed shared-library deferral, no export/import ABI macros, no static/shared selectors, and Windows non-claim wording checks. |
| User install docs | `README.md`, `INSTALL.md` | Public static-first package contract, supported platform split, downstream consumer guidance, and non-claims. |
| Maintainer package docs | `docs/maintainer_guide.md` | Maintainer interpretation of package evidence, Windows package boundaries, and ABI/package non-claims. |
| Examples | `examples/cmake_example/`, selected example sources | Downstream consumer proof and installed-example validation inputs. |

## Supported Static-First Package Contract

| Package element | Owner | Evidence |
| --- | --- | --- |
| Static library archive | Makefile and CMake install rules | Unix installs `libsparse_lu_ortho.a`; Windows CMake installs `sparse_lu_ortho.lib`. |
| Public headers | Makefile and CMake install rules | Installs public headers under `include/sparse/`, excluding `.h.in` templates. |
| Generated version header | Makefile and CMake generation/install rules | Installs generated `sparse_version.h` from `VERSION` and `include/sparse_version.h.in`. |
| pkg-config metadata | `sparse.pc.in`, Makefile, CMake configure/install | Describes static archive package metadata with `Cflags: -I${includedir}` and `Libs: -L${libdir} -lsparse_lu_ortho -lm ...`. |
| CMake package metadata | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in` | Exports `Sparse::sparse_lu_ortho` as a static imported target after install. |
| Exact CMake package version | `write_basic_package_version_file(... COMPATIBILITY ExactVersion)` | Avoids broad dynamic ABI compatibility implications. |
| Unix downstream pkg-config consumer | `tests/test_install.sh` | Compiles, links, and runs a basic installed consumer and maintained example using pkg-config. |
| CMake downstream consumer | `tests/test_cmake_install.sh` and Windows workflow inline proof | Configures, builds, and runs generated and maintained CMake consumers using `find_package(Sparse)`. |
| Uninstall cleanup | `Makefile uninstall`, `tests/test_install.sh` | Removes static library, headers, and `sparse.pc` from the install prefix. |

## Install Validation Inventory

| Validation path | Hosted coverage | What it proves | What it does not prove |
| --- | --- | --- | --- |
| `bash tests/test_install.sh` | Linux reviewed package job; macOS reviewed install/pkg-config job | Unix Make install/uninstall, `pkg-config` command resolution, exact version, static metadata, no shared artifacts, downstream compile/link/run. | Windows `pkg-config` execution, package-manager distribution, shared-library support, dynamic ABI. |
| `bash tests/test_cmake_install.sh` | Linux reviewed package job; macOS reviewed CMake install/export job | CMake install/export, installed static target metadata, exact/mismatch package version handling, no source/build path leaks, downstream `find_package` consumers. | Shared-library ABI, runtime-loader behavior, package-manager distribution. |
| Windows inline CMake install/downstream workflow | Windows reviewed CMake install/downstream job | Static `.lib`, 19 headers plus version header, CMake package files, `sparse.pc` metadata inspection, generated and maintained downstream CMake consumers, exact/mismatch versions. | Windows Makefile parity, Windows `pkg-config` command execution, DLL/shared support, dynamic ABI. |
| `bash scripts/static_package_deferral_check.sh` | Linux package job; macOS CMake install/export job; local package checks | `BUILD_SHARED_LIBS=ON` rejection, static target/install metadata, no export/import or ABI macros, no package selectors, Windows package non-claims. | Shared-library support; it proves deferral, not implementation. |

## Package Metadata Boundaries

| Metadata | Supported interpretation | Guarded non-claim |
| --- | --- | --- |
| `Description: Static archive package metadata for sparse linear algebra` in `sparse.pc.in` | The pkg-config file describes the maintained static archive package surface. | Does not imply shared library, dynamic ABI, package-manager provider, or runtime-loader support. |
| `Libs: -L${libdir} -lsparse_lu_ortho -lm ...` | Downstream consumers link the installed static archive and math library, with optional build flags appended. | No `Libs.private` or static/shared selector contract. |
| CMake `ARCHIVE DESTINATION` install | CMake installs the static archive package surface. | No `RUNTIME`, `LIBRARY`, DLL, dylib, `.so`, SONAME, install-name, or loader metadata claim. |
| `Sparse::sparse_lu_ortho` exported target | Downstream CMake projects can link the installed static target. | No `Sparse::...shared` target or component selector support. |
| `SparseConfigVersion.cmake` exact version | Consumers can require the exact installed package version. | Exact package metadata is not dynamic ABI compatibility. |
| Installed `sparse.pc` on Windows | Windows workflow inspects static metadata text. | Windows CI does not execute `pkg-config` and does not claim Windows pkg-config parity. |

## ABI And Package Non-Claim Register

| Non-claim | Current evidence boundary |
| --- | --- |
| Shared-library support | `BUILD_SHARED_LIBS=ON` is intentionally rejected by CMake and checked by `static_package_deferral_check.sh`. |
| Dynamic ABI compatibility | No ABI policy, exported-symbol/version checks, binary compatibility matrix, or compatibility promise exists. |
| Runtime-loader behavior | No `.so`, `.dylib`, `.dll`, SONAME, install-name/RPATH, import-library, or installed shared consumer proof exists. |
| Static/shared package selector UX | CMake and pkg-config metadata intentionally provide no static/shared component selector. |
| Package-manager distribution | No Homebrew, vcpkg, Conan, apt, dnf, pacman, or package-manager provider proof is supported. |
| Provider upgrade behavior | Source install/uninstall proof does not prove package-manager upgrade, migration, or provenance behavior. |
| Windows Makefile parity | Windows reviewed package proof remains CMake-first. |
| Windows `pkg-config` execution parity | Windows installs and inspects `sparse.pc` metadata but does not run `pkg-config`. |
| Broad platform package parity | Linux, macOS, and Windows package proof surfaces differ and must remain named by platform. |

## Package-Manager Readiness Candidate List

| Candidate | Closure mode | Benefits | Risks |
| --- | --- | --- | --- |
| Formal package-manager deferral | Publish a decision and strengthen non-claim checks. | Fully close ambiguity with low implementation risk. | Does not reduce adoption friction. |
| Source-package archive proof | Add a source archive/install validation path without provider-specific distribution. | Improves release-readiness while avoiding provider maintenance. | Still not package-manager support. |
| vcpkg manifest proof | Add a local vcpkg-style manifest/port proof for static CMake install. | Strong Windows/CMake adoption fit. | Requires provider-specific semantics and may imply more support than intended. |
| Homebrew formula proof | Add a local formula proof for Unix-like static install. | Good macOS/Linux source package path. | Homebrew-specific behavior and external formula maintenance risk. |
| CPack/archive installer proof | Add CMake package/archive generation proof. | Leverages existing CMake install/export metadata. | Not equivalent to package-manager ecosystem distribution. |

Day 7 recommends that Epic 15 choose only one package-manager readiness
closure mode after the shared-library ABI decision is recorded. Package-manager
work should remain static-first unless Sprint 170 explicitly selects a shared
ABI track.

## Source-Backed Validation Owners

| Claim surface | Validation owner |
| --- | --- |
| Unix Make static install and pkg-config execution | `tests/test_install.sh` |
| Unix uninstall cleanup | `tests/test_install.sh` |
| CMake static install/export | `tests/test_cmake_install.sh` |
| Installed CMake downstream consumer | `tests/test_cmake_install.sh`, `examples/cmake_example/`, Windows workflow inline generated consumer |
| Exact package version handling | `tests/test_install.sh`, `tests/test_cmake_install.sh`, Windows workflow inline exact-version consumer |
| Shared-library deferral | `scripts/static_package_deferral_check.sh`, CMake configure guard |
| Windows CMake-first package proof | `.github/workflows/windows-ci.yml` |
| macOS static install/export package proof | `.github/workflows/macos-ci.yml` |
| Linux static package contract | `.github/workflows/ci.yml` |

## Day 8 Handoff

Day 8 should inventory documentation and claim surfaces with attention to:

- README and INSTALL package wording;
- generated API HTML local-only language;
- state-of-the-art, performance, external parity, platform, package, ABI, and
  report publication wording;
- authoritative versus historical planning docs;
- report-index schema docs and maintained interpretation rules;
- any stale or ambiguous links that could affect the Epic 15 evidence ledger.

## Validation Notes

Day 7 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Supported package behavior is separated from unsupported package claims. | Complete | Supported static-first contract, metadata boundaries, and non-claim register distinguish package proof from unsupported shared/ABI/provider claims. |
| Install validation owners are source-backed. | Complete | Validation owner table maps package claims to install scripts, workflow lanes, CMake/Make metadata, and examples. |
| Package-manager decision candidates are concrete. | Complete | Candidate list identifies formal deferral, source-package archive proof, vcpkg, Homebrew, and CPack/archive paths with benefits and risks. |
