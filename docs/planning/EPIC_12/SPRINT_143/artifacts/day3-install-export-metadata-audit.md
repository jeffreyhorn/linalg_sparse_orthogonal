# Day 3 Install Export Metadata Audit

## Purpose

Day 3 audits the installed package metadata surface before the Sprint 143
product decision. The audit covers Make install/uninstall, CMake
install/export, `pkg-config`, versioning, static/shared guards, downstream
proof scripts, CI package lanes, and normalized package report rows.

## Make Install And pkg-config Surface

| Surface | Current behavior | Proof owner |
| --- | --- | --- |
| `PREFIX` | Defaults to `/usr/local`; `DESTDIR` staging is supported by path concatenation. | `Makefile`, `tests/test_install.sh` |
| Static archive install | Installs `build/libsparse_lu_ortho.a` into `$(DESTDIR)$(PREFIX)/lib`. | `tests/test_install.sh` |
| Header install | Installs all source `include/*.h` headers plus generated `build/include/sparse_version.h` into `include/sparse`. | `tests/test_install.sh` |
| `sparse.pc` generation | Substitutes `@PREFIX@`, `@VERSION@`, and `@SPARSE_PC_LIBS_EXTRA@` into `sparse.pc.in`. | `tests/test_install.sh` |
| Build-option link flags | Adds `-pthread` for `SPARSE_MUTEX`; adds OpenMP flags for `SPARSE_OPENMP`. | Future selected-path validation if build-option metadata changes. |
| Uninstall | Removes static archive, `include/sparse`, and `lib/pkgconfig/sparse.pc`; attempts to remove empty pkgconfig dir. | `tests/test_install.sh` |

Current `sparse.pc.in`:

- declares `Name`, `Description`, `Version`, `Cflags`, and `Libs`;
- links with `-lsparse_lu_ortho -lm @SPARSE_PC_LIBS_EXTRA@`;
- has no `Libs.private`;
- has no static/shared selector;
- has no shared-library, ABI, package-manager, or dynamic-loader wording.

## CMake Install And Export Surface

| Surface | Current behavior | Proof owner |
| --- | --- | --- |
| Shared request guard | `BUILD_SHARED_LIBS=ON` fails at configure time with explicit static-first deferral wording. | `scripts/static_package_deferral_check.sh` |
| Library target | `add_library(sparse_lu_ortho STATIC ...)` makes the target explicitly static. | `tests/test_cmake_install.sh`, static deferral guard |
| Installed target | `install(TARGETS sparse_lu_ortho EXPORT SparseTargets ARCHIVE DESTINATION ... LIBRARY DESTINATION ...)`. | `tests/test_cmake_install.sh` |
| Installed headers | Installs source headers under `include/sparse`, excluding `.h.in` and source `sparse_version.h`, then installs generated `sparse_version.h`. | `tests/test_cmake_install.sh` |
| CMake package config | Generates `SparseConfig.cmake` from `cmake/SparseConfig.cmake.in`; imports `SparseTargets.cmake` and checks requested components. | `tests/test_cmake_install.sh` |
| CMake version config | Generates `SparseConfigVersion.cmake` with `COMPATIBILITY ExactVersion`. | `tests/test_cmake_install.sh` |
| Export namespace | Exports `Sparse::sparse_lu_ortho`. | `tests/test_cmake_install.sh`, installed example |
| `pkg-config` via CMake | CMake also configures and installs `sparse.pc`. | `tests/test_cmake_install.sh` |

The CMake package currently describes one static imported target. Exact-version
package compatibility is intentionally conservative and does not imply dynamic
ABI compatibility.

## Proof Script Inventory

| Script | Current proof | Static/shared semantics |
| --- | --- | --- |
| `tests/test_install.sh` | Make clean/install, static archive presence, no shared artifacts, installed header count, `pkg-config` existence, exact version, prefix/libdir/includedir variables, `--cflags`, `--libs`, `--static`, no `Libs.private`, no unsupported ABI/package wording, downstream compile/link/run, example compile/run, uninstall cleanup. | Proves Unix-side Make install and `pkg-config` static archive consumer path. |
| `tests/test_cmake_install.sh` | CMake configure/build/install, static archive presence, no shared artifacts, installed header count, package files, static imported target, install-prefix include/archive paths, no source/build path leaks, installed example configure/build/run, exact version success, mismatched version rejection, `pkg-config --modversion`. | Proves Unix-side CMake install/export and `find_package(Sparse)` static imported target path. |
| `scripts/static_package_deferral_check.sh` | `BUILD_SHARED_LIBS=ON` rejection, explicit `STATIC` target, no public export/import macro, no shared ABI metadata, no CMake/package selector, no `pkg-config` selector, preserved README/INSTALL/maintainer deferral wording. | Proves shared-library packaging and dynamic ABI support remain explicit deferrals. |

Day 3 syntax validation:

```sh
bash -n tests/test_install.sh tests/test_cmake_install.sh scripts/static_package_deferral_check.sh
```

Result: passed.

## Versioning Surface

| Version surface | Current behavior | Decision relevance |
| --- | --- | --- |
| `VERSION` | Single source of truth for repo package version. | Shared ABI path would need to distinguish package version from ABI compatibility. |
| Generated `sparse_version.h` | Make and CMake generate installed version macros from `VERSION`. | Public source/header version contract exists. |
| `pkg-config --modversion` | Uses `Version: @VERSION@` from `sparse.pc.in`. | Current package metadata exact version is source/package metadata, not ABI compatibility. |
| `SparseConfigVersion.cmake` | Uses `ExactVersion`. | Conservative static-first choice; shared ABI support would need explicit compatibility policy. |
| Mismatched CMake version test | `tests/test_cmake_install.sh` expects lower same-major version lookup to fail. | Preserves exact package version boundary. |

## Static/Shared Guard Inventory

| Guard | Current owner | What it prevents |
| --- | --- | --- |
| Configure-time `BUILD_SHARED_LIBS=ON` failure | `CMakeLists.txt` | Prevents silent shared-library builds under unsupported mode. |
| Explicit `STATIC` target check | `scripts/static_package_deferral_check.sh` | Prevents CMake target from drifting into default shared/static behavior. |
| Export/import macro absence check | `scripts/static_package_deferral_check.sh` | Prevents public headers from adding `SPARSE_API`/export markers without a decision. |
| Shared ABI metadata absence check | `scripts/static_package_deferral_check.sh` | Prevents `SOVERSION`, visibility, install-name, or similar metadata from appearing without a support decision. |
| CMake package selector absence check | `scripts/static_package_deferral_check.sh` | Prevents unsupported static/shared component selectors. |
| `pkg-config` selector absence check | `scripts/static_package_deferral_check.sh` | Prevents unsupported `Libs.private` or static/shared selector semantics. |
| Support wording checks | `scripts/static_package_deferral_check.sh` | Keeps README, INSTALL, and maintainer guide aligned with static-first deferral wording. |
| No shared installed artifacts | `tests/test_install.sh`, `tests/test_cmake_install.sh`, Windows supplemental workflow | Prevents accidental `.so`, `.dylib`, `.dll` installs. |

## Package Report Rows

Current package rows are source-controlled proof-owner rows, not fresh local
install-run evidence.

Validation commands:

```sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
```

Results:

- package check passed with 6 rows;
- package freshness passed with 6 source-controlled advisory rows:
  - `report_contract_package_static_install_package_install_proof_owner_v1`;
  - `package_make_install_pkg_config_v1`;
  - `package_cmake_install_export_v1`;
  - `package_pkg_config_template_v1`;
  - `package_cmake_package_config_v1`;
  - `package_static_package_deferral_v1`.

These rows identify maintained proof owners and static-first scope. They do
not claim package-manager availability, shared-library ABI support, dynamic
linking behavior, or broad platform support.

## Selected-Path Metadata Requirements

| If Sprint 143 selects... | Required metadata changes |
| --- | --- |
| Shared-library ABI support | Replace configure-time rejection with deliberate shared build option; add export/import macro policy; add hidden-by-default visibility or symbol allowlist; add shared artifact install rules; add CMake target metadata for shared artifacts; decide static/shared coexistence and selectors; update `pkg-config` semantics; add loader/version/SOVERSION or explicit compatibility policy; add downstream dynamic consumer proof and platform limits. |
| Stricter static-first-only support | Preserve configure-time rejection; strengthen negative checks for no shared artifacts/selectors/ABI metadata; tighten docs and package comments; improve install script diagnostics; refresh package report rows if row meaning changes; keep CMake exact-version and static imported target proof explicit. |

## Decision Inputs For Day 5

| Finding | Implication |
| --- | --- |
| Make and CMake static install paths are already well covered. | Static-first strengthening is likely lower risk and can close ambiguity completely. |
| CMake currently installs `LIBRARY DESTINATION` even though the target is static. | Harmless today, but shared path would need deliberate runtime/archive/library destination semantics. |
| `sparse.pc.in` has no `Libs.private` or static/shared selector. | Good for current self-contained static contract; insufficient for dual static/shared package variants. |
| Package report rows are proof-owner rows, not generated install-run rows. | Sprint 143 should not cite report freshness as evidence that install commands were just run. |
| Existing guard script is already static-first oriented. | Shared path would need substantial guard redesign; static path can strengthen it. |
| Windows supplemental proof is CMake-only. | Package decision must not imply Windows Makefile parity or reviewed install-validation parity. |

## Day 4 Inputs

Day 4 should focus on platform and loader risk:

- Linux shared loader and `SOVERSION`/RPATH expectations;
- macOS `.dylib` install-name behavior;
- Windows DLL/import-library/export macro mechanics;
- CMake generator/platform differences;
- CI lane support-tier wording;
- separation between Sprint 143 package decision and Sprint 144 platform
  promotion.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Make, CMake, `pkg-config`, and package-report surfaces are accounted for. | Complete | Surface tables cover Make install, CMake install/export, `sparse.pc.in`, proof scripts, and package rows. |
| Unsupported shared-library behavior is documented before the decision. | Complete | Static/shared guard inventory records configure rejection, no shared artifacts, no selectors, and no ABI metadata. |
| Day 5 can decide from concrete metadata and proof requirements. | Complete | Selected-path requirements and decision inputs list the metadata needed for each path. |
