# Sprint 143 Day 4 Platform And Loader Risk Audit

## Purpose

Audit platform, dynamic-loader, and CI support-tier risks before Sprint 143
chooses between shared-library ABI support and a stricter static-first-only
package contract.

This artifact is decision input only. It does not promote macOS or Windows
package lanes to reviewed parity, and it does not claim shared-library ABI
support.

## CI Package Lane Inventory

| Lane | Workflow | Support tier | Commands and proof | Explicit non-claims |
| --- | --- | --- | --- | --- |
| Linux reviewed static-first package contract | `.github/workflows/ci.yml` `package-contract` | Reviewed package baseline | Installs `cmake` and `pkg-config`; runs `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, and `bash scripts/static_package_deferral_check.sh`. | No shared-library packaging, dynamic ABI compatibility, package-manager support, macOS parity, or Windows parity. |
| Linux reviewed CMake parity | `.github/workflows/ci.yml` `cmake-parity` | Reviewed CMake source/build baseline | Runs `make quality-review-cmake`. | CMake parity is not separate package-manager or shared-library proof. |
| macOS Make install and `pkg-config` confidence | `.github/workflows/macos-ci.yml` `install-and-pkgconfig` | Supplemental | Runs `bash tests/test_install.sh` on Apple Clang/macOS. | Not reviewed macOS install/export parity and not shared-library proof. |
| macOS CMake install/export confidence | `.github/workflows/macos-ci.yml` `cmake-install-export` | Supplemental | Runs `bash tests/test_cmake_install.sh` and `bash scripts/static_package_deferral_check.sh`. | Not reviewed macOS package parity and not package-manager support. |
| Windows CMake consumer subset | `.github/workflows/windows-ci.yml` `build-and-test` | Reviewed Windows CMake subset | Configures/builds with Visual Studio 2022 x64, verifies `EXPECTED_WINDOWS_CTEST_COUNT=56`, then runs CTest. | No reviewed Makefile parity, no reviewed install-validation parity, no `pkg-config` lane, and staged POSIX blocker tests remain excluded. |
| Windows CMake install/downstream confidence | `.github/workflows/windows-ci.yml` `install-and-downstream` | Supplemental | CMake install proof checks installed `.lib`, no `.dll`, 19 headers, CMake package files, installed example execution, exact version acceptance, and mismatched version rejection. | Confidence path only; does not promote Windows package parity or shared-library support. |

## Shared-Library Blockers By Platform

### Linux

- No `SOVERSION` or soname policy exists for a shared object.
- No hidden-by-default visibility policy exists for implementation symbols.
- No reviewed exported-symbol allowlist exists; Day 2 found many non-`sparse_`
  or internal-looking global symbols in the static archive.
- No dynamic downstream install/loader smoke exists.
- No RPATH/RUNPATH policy exists for installed examples or downstream
  consumers.
- `sparse.pc.in` has a static archive-oriented `Libs` line and no
  static/shared selection semantics or `Libs.private` policy.

### macOS

- No `.dylib` install name, id, or `@rpath` policy exists.
- No loader proof exists for installed dynamic consumers.
- No reviewed macOS install/export parity exists; package install/export lanes
  are explicitly supplemental confidence paths.
- No exported-symbol list or default-hidden build policy exists for Darwin.
- CMake package files currently describe the static imported target surface,
  not a shared target with loader metadata.

### Windows/MSVC

- No public export/import macro exists for `__declspec(dllexport)` and
  `__declspec(dllimport)`.
- No import-library policy exists for a DLL product.
- The supplemental install proof explicitly expects `sparse_lu_ortho.lib` and
  rejects `*.dll` artifacts.
- No `RUNTIME DESTINATION` proof, DLL search-path proof, or installed dynamic
  downstream consumer exists.
- Windows reviewed scope remains CMake-first source/build/test subset only;
  Makefile parity, `pkg-config`, and install-validation parity are not reviewed.

### Cross-Platform ABI Risks

- Public struct layout, enum values, callback signatures, `idx_t` width, and
  version macros are ABI-sensitive and currently lack a compatibility policy.
- The current exact package version is an install metadata check, not an ABI
  compatibility promise.
- Static and shared artifacts do not have coexistence semantics, package
  selectors, or separate CMake target names.
- Package report rows are source-controlled proof-owner rows, not proof that
  fresh dynamic loader validation ran.

## Static-First Strengthening Opportunities

- Preserve and tighten the `BUILD_SHARED_LIBS=ON` rejection as an intentional
  product decision rather than an incidental limitation.
- Keep negative installed-artifact checks for `.so`, `.dylib`, and `.dll`
  artifacts on all feasible package lanes.
- Clarify that Linux owns the reviewed package contract while macOS and
  Windows remain supplemental until Sprint 144 promotion work.
- Keep `pkg-config` and CMake package metadata static-only; avoid adding
  static/shared selectors unless the shared-library path is selected.
- Strengthen maintainer docs and CI messages so static archive support is
  described as maintained, while dynamic ABI and package-manager support are
  explicitly deferred.
- Keep the static package deferral guard as the single negative-proof script
  for unsupported shared ABI metadata.

## Sprint 144 Platform-Separation Notes

Sprint 143 should make the package/ABI product decision without implying
platform promotion. If stricter static-first support is selected, Sprint 144
can separately decide whether to promote macOS or Windows package lanes from
supplemental confidence to reviewed parity. If shared-library support is
selected, Sprint 144 must own platform-specific loader and distribution
promotion before any broader cross-platform package claim is made.

Windows staged blockers remain source-level and are separate from Sprint 143:
pthread APIs affect `test_threads` and `test_sprint4_integration`, while POSIX
temporary-file APIs affect `test_fuzz`. Those exclusions should not be hidden
inside package/ABI claims.

## Validation Scenarios For A Shared-Library Decision

1. Configure, build, install, and export a shared library on each claimed
   platform.
2. Verify exported symbols against a reviewed allowlist using platform-native
   tools.
3. Verify hidden implementation symbols stay hidden where the platform supports
   symbol visibility.
4. Compile, link, and run downstream dynamic consumers through CMake package
   metadata.
5. Compile, link, and run downstream dynamic consumers through `pkg-config`
   where that lane is supported.
6. Check loader paths, RPATH/RUNPATH, install name, DLL search path, and import
   library behavior for each claimed platform.
7. Define and test static/shared coexistence semantics.
8. Define and test version, soname, and ABI compatibility boundaries.

## Validation Scenarios For A Static-First Decision

1. Run Make install and `pkg-config` downstream proof with
   `bash tests/test_install.sh`.
2. Run CMake install/export and downstream proof with
   `bash tests/test_cmake_install.sh`.
3. Run unsupported shared-library negative proof with
   `bash scripts/static_package_deferral_check.sh`.
4. Check installed artifacts exclude `.so`, `.dylib`, and `.dll` files where
   feasible.
5. Check public docs and maintainer docs state maintained static archive
   support and deferred shared-library ABI support.
6. Check CI workflow comments preserve the Linux reviewed baseline and the
   macOS/Windows supplemental confidence-tier split.
7. Check package report rows remain proof-owner metadata and do not imply fresh
   dynamic loader validation.

## Day 5 Decision Input

Shared-library support is feasible only as a multi-surface product effort:
visibility, export macros, symbol allowlists, versioning, CMake/pkg-config
selection semantics, and platform loader proof would all need to land together.
The current lower-risk path is to make static-first support explicit and
stricter, while routing platform-promotion decisions to Sprint 144.
