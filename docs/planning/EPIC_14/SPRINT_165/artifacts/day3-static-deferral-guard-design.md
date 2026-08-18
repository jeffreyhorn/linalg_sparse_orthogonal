# Sprint 165 Day 3 Static Deferral Guard Design

## Purpose

Day 3 defines the static-first guard contract before implementation. The goal
is to make the `BUILD_SHARED_LIBS=ON` failure behavior, package metadata
absence rules, affected files, and validation commands explicit without
claiming shared-library support.

## Current Guard Owners Reviewed

| Owner | Current Role | Day 3 Design Decision |
| --- | --- | --- |
| `CMakeLists.txt` | Rejects `BUILD_SHARED_LIBS=ON`, declares `sparse_lu_ortho` as `STATIC`, installs archive-only target metadata, generates exact-version package metadata, and generates `sparse.pc`. | Keep as the source of configure-time behavior and CMake package shape. |
| `scripts/static_package_deferral_check.sh` | Local static-first drift guard for shared deferral wording, target type, install metadata, export/ABI metadata, package selectors, docs wording, and Windows non-claims. | Keep as the central local guard; Day 4 should strengthen only specific uncovered absence rules. |
| `tests/test_cmake_install.sh` | Validates generated installed CMake package output and installed CMake consumers. | Keep generated-output checks here rather than committing generated package snapshots. |
| `tests/test_install.sh` | Validates Make install, installed `sparse.pc`, pkg-config output, downstream consumers, and uninstall. | Keep Unix-side pkg-config proof here. |
| `.github/workflows/windows-ci.yml` | Validates Windows CMake install/downstream behavior and metadata-only `sparse.pc` content. | Keep Windows package proof CMake-first and do not add `pkg-config` execution without a product decision. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Public and maintainer package/support-boundary wording. | Keep documentation guard checks tied to these authoritative surfaces. |

## Expected `BUILD_SHARED_LIBS=ON` Behavior

The CMake configure step must fail when a caller requests
`-DBUILD_SHARED_LIBS=ON`.

Required configure-time behavior:

- nonzero configure exit status;
- no fallback to static output after accepting the shared request;
- no generated install package that suggests shared support;
- failure message includes the rejected input token `BUILD_SHARED_LIBS=ON`;
- failure message states the maintained package surface is the static archive
  package surface;
- failure message states shared-library packaging is deferred;
- failure message states dynamic ABI support is deferred.

Required blocker terms in the failure message:

- export/import policy;
- symbol visibility policy;
- dynamic ABI policy;
- Linux SONAME metadata;
- macOS install-name/RPATH metadata;
- Windows DLL/import-library behavior;
- installed shared consumer proof;
- runtime-loader validation.

## Metadata Absence Rules

### CMake Source Metadata

`CMakeLists.txt` must keep:

- `add_library(sparse_lu_ortho STATIC ...)`;
- no `add_library(sparse_lu_ortho SHARED ...)`;
- no `add_library(sparse_lu_ortho MODULE ...)`;
- `install(TARGETS sparse_lu_ortho ... ARCHIVE DESTINATION ...)`;
- no `RUNTIME DESTINATION` for `sparse_lu_ortho`;
- no `LIBRARY DESTINATION` for `sparse_lu_ortho`;
- no shared-library ABI metadata such as `SOVERSION`,
  `WINDOWS_EXPORT_ALL_SYMBOLS`, `C_VISIBILITY_PRESET`,
  `VISIBILITY_INLINES_HIDDEN`, `INSTALL_NAME_DIR`, or `MACOSX_RPATH`.

### Installed CMake Package Metadata

Generated installed CMake package files must keep:

- `Sparse::sparse_lu_ortho` as `STATIC IMPORTED`;
- import locations pointing to installed static archive paths;
- include directories pointing to installed include paths;
- no source-tree or build-tree path leaks;
- no `SHARED IMPORTED` or `MODULE IMPORTED` target metadata;
- no imported `.so`, `.dylib`, or `.dll` locations;
- no `SOVERSION`, `IMPORTED_SONAME`, `INSTALL_NAME`, `MACOSX_RPATH`,
  `IMPORTED_IMPLIB`, runtime destination metadata, or static/shared component
  selector metadata.

### pkg-config Metadata

`sparse.pc.in` and installed `sparse.pc` must keep:

- `Name: sparse`;
- `Description: Static archive package metadata for sparse linear algebra`;
- version metadata sourced from `VERSION`;
- `Cflags` based on `${includedir}`;
- `Libs` based on `${libdir}`, `-lsparse_lu_ortho`, `-lm`, and supported build
  option extras;
- no `Libs.private` stanza under the current self-contained static link
  contract;
- no shared-library, dynamic ABI, loader, package-manager, or static/shared
  selector wording.

### Public Header And Export Macro Absence

Public headers must not gain export/import scaffolding before a shared-library
product decision. Guarded terms include:

- `SPARSE_API`;
- `SPARSE_EXPORT`;
- `SPARSE_IMPORT`;
- `SPARSE_SHARED`;
- `SPARSE_STATIC`;
- `SPARSE_ABI`.

## Documentation Wording Rules

Public and maintainer docs should say:

- the maintained install/export surface is static-first;
- `pkg-config` and `find_package(Sparse)` describe the installed static archive
  surface;
- exact package version metadata is exact package metadata, not a dynamic ABI
  guarantee;
- shared-library packaging is intentionally deferred;
- dynamic ABI compatibility is a non-claim;
- runtime-loader behavior is a non-claim;
- package-manager distribution is a non-claim;
- Windows package validation is CMake-first;
- Windows `sparse.pc` validation is metadata-only;
- Windows Makefile parity and Windows `pkg-config` execution parity are
  non-claims.

Docs should not add package-manager guidance for Homebrew, apt, dnf, pacman,
vcpkg, Conan, or similar systems except as explicit non-claims.

## Affected File List For Day 4

Day 4 implementation should inspect or edit only as needed:

| File | Expected Day 4 Role |
| --- | --- |
| `scripts/static_package_deferral_check.sh` | Most likely guard-hardening owner if any absence rule is missing. |
| `tests/test_cmake_install.sh` | Generated CMake package output checks if installed metadata guard gaps exist. |
| `tests/test_install.sh` | Installed `sparse.pc` checks if pkg-config guard gaps exist. |
| `.github/workflows/windows-ci.yml` | Windows metadata-only package checks if Windows CMake-first wording or validation gaps exist. |
| `CMakeLists.txt` | Edit only if the source behavior itself fails the guard design. |
| `sparse.pc.in` | Edit only if the template violates the static archive metadata contract. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Edit only if support-boundary wording has drifted. |

No change is currently required to `cmake/SparseConfig.cmake.in`; it should
remain a minimal package config template unless generated output validation
finds a real gap.

## Validation Command List

Focused local guard commands:

```sh
bash scripts/static_package_deferral_check.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
```

Source/header quality commands if any `.c` or `.h` files change:

```sh
make format
make lint
make test
```

Useful focused CMake rejection probe:

```sh
cmake -S . -B /tmp/sparse-build-shared-request -DBUILD_SHARED_LIBS=ON
```

Expected result: nonzero configure exit with the static-first shared-deferral
message.

Hosted validation owners:

- Linux: `.github/workflows/ci.yml` package-contract job runs
  `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh`.
- macOS: `.github/workflows/macos-ci.yml` install/pkg-config and
  CMake-install-export jobs run the local package proof scripts and static
  deferral guard.
- Windows: `.github/workflows/windows-ci.yml` install/downstream job validates
  installed static `.lib`, generated CMake metadata, generated and maintained
  CMake consumers, exact-version handling, absence of DLL/shared metadata, and
  metadata-only `sparse.pc` content.

## Day 4 Implementation Guidance

Day 4 should not add shared-library support. It should only close guard gaps
that can be expressed as fail-closed checks. The safest implementation order is:

1. Run or dry-read the existing static deferral guard against the Day 3 absence
   rules.
2. Strengthen `scripts/static_package_deferral_check.sh` if a source-level
   absence rule is missing.
3. Strengthen `tests/test_cmake_install.sh` or the Windows workflow only if
   generated installed metadata has an uncovered absence rule.
4. Strengthen `tests/test_install.sh` only if installed `sparse.pc` metadata has
   an uncovered absence rule.
5. Update docs only if support-boundary wording has drifted.
6. Run focused package validation commands after any package guard edit.

## Validation Notes

Day 3 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 3.

## Completion Check

- Guard behavior is explicit before edits.
- Checks block drift without claiming shared-library support.
- Implementation scope is narrow and testable for Day 4.
