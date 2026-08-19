# Sprint 170 Day 10: Guard Update Design

## Purpose

Day 10 turns the Day 9 shared-library ABI product decision into a concrete
guard design. The selected posture remains static-first-only: the project may
claim maintained static archive package behavior, but must not claim
shared-library packaging, dynamic ABI compatibility, runtime-loader behavior,
package-manager distribution, or broad platform parity.

This artifact scopes the guard updates before implementation so Day 11 can
change scripts or checks without changing the selected product decision.

## Decision Inputs

| Input | Role In Guard Design |
| --- | --- |
| `day9-shared-library-abi-product-decision.md` | Canonical Sprint 170 product decision and claim boundary. |
| `scripts/static_package_deferral_check.sh` | Local static-first contract guard for build metadata, package metadata, public wording, and unsupported Windows package execution. |
| `tests/test_install.sh` | Unix Make install/uninstall plus `pkg-config` execution proof. |
| `tests/test_cmake_install.sh` | Unix CMake install/export plus installed CMake consumer proof. |
| `.github/workflows/{ci,macos-ci,windows-ci}.yml` | Reviewed platform package lanes and Windows metadata-only boundary. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Public and maintainer claim surfaces. |

## Existing Guard Coverage

| Guard Surface | Current Coverage | Day 10 Disposition |
| --- | --- | --- |
| CMake shared request | `BUILD_SHARED_LIBS=ON` is expected to fail and name shared-library, dynamic ABI, symbol visibility, loader, and consumer-proof blockers. | Keep; this is the primary accidental shared-build gate. |
| CMake target type | `sparse_lu_ortho` must be declared as `STATIC`; `SHARED` and `MODULE` targets are rejected. | Keep; add no behavior change. |
| CMake install metadata | Static archive install via `ARCHIVE DESTINATION` is required; runtime and shared-library destinations are rejected. | Keep; this enforces archive-only installation. |
| Public export macros | Public headers are scanned for export/import/static/shared ABI macros. | Keep; shared ABI work must introduce an explicit policy first. |
| Package selectors | CMake package and `sparse.pc.in` are scanned for shared/static selectors and `Libs.private`. | Keep; no selector is selected by the product decision. |
| Make install package proof | `tests/test_install.sh` checks static archive install, no shared artifacts, exact header count, `pkg-config` fields, downstream compile/link/run, and uninstall. | Keep; it is the maintained Unix Make package proof. |
| CMake install package proof | `tests/test_cmake_install.sh` checks static imported target metadata, archive path, no source/build leaks, version behavior, and installed consumers. | Keep; it is the maintained Unix CMake package proof. |
| Windows package boundary | Windows CI checks CMake install/downstream behavior and inspects `sparse.pc` metadata without executing `pkg-config` or Make install. | Keep; this preserves the Windows CMake-first claim. |
| Public wording | Static package and unsupported shared ABI wording is required in README, INSTALL, and maintainer guide. | Keep; extend with a decision-record reference check after documentation is aligned. |

## Required Day 11 Guard Updates

1. Add a decision-record existence and wording check to
   `scripts/static_package_deferral_check.sh`.
   - Required file:
     `docs/planning/EPIC_15/SPRINT_170/artifacts/day9-shared-library-abi-product-decision.md`
   - Required tokens:
     - `static-first-only package`
     - `Shared-library packaging and dynamic ABI compatibility remain explicitly unsupported and deferred`
     - `BUILD_SHARED_LIBS=ON`
     - `Sparse::sparse_lu_ortho`
     - `Windows pkg-config command execution parity`
   - Reason: the guard should fail if the canonical Sprint 170 decision record
     disappears or is rewritten away from the selected claim boundary.

2. Add public-documentation decision-link checks after Day 12 documentation
   alignment.
   - Candidate files: `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`.
   - Required behavior: each file should either cite the Sprint 170 decision
     record or carry equivalent exact wording for static-first-only support and
     shared ABI deferral.
   - Reason: claim wording should not drift after the decision record exists.

3. Add Makefile static archive expectation checks if they are not already
   covered by script-level static deferral guards.
   - Required tokens:
     - `libsparse_lu_ortho.a`
     - install target copies the archive to `$(PREFIX)/lib`
     - no generated install path for `.so`, `.dylib`, or `.dll`
   - Reason: `tests/test_install.sh` proves behavior, but a direct source
     guard catches unsupported Makefile shared-package additions sooner.

4. Keep generated package metadata exactness narrow.
   - `sparse.pc.in` must retain:
     - `Description: Static archive package metadata for sparse linear algebra`
     - `Cflags: -I${includedir}`
     - `Libs: -L${libdir} -lsparse_lu_ortho -lm`
   - CMake package exports must retain:
     - static imported target metadata
     - install-prefix-relative includes and archive location
     - exact package version compatibility
   - Reason: exact metadata checks are useful only where the product decision
     has selected exact metadata.

5. Keep unsupported-wording scans targeted.
   - Broad scans across all documentation will create false positives because
     planning artifacts legitimately discuss future shared-library and ABI
     work.
   - Negative scans should remain scoped to package metadata templates, CMake
     package exports, public headers, selected public docs, and workflow script
     blocks.

## Negative Check List

The guard stack should continue to fail on these unsupported additions unless
a future product decision explicitly selects shared-library support:

| Area | Negative Check |
| --- | --- |
| CMake target model | No `SHARED` or `MODULE` `sparse_lu_ortho` target. |
| CMake install model | No `RUNTIME DESTINATION` or `LIBRARY DESTINATION` for the maintained install target. |
| CMake ABI metadata | No `SOVERSION`, `WINDOWS_EXPORT_ALL_SYMBOLS`, visibility presets, install-name metadata, imported SONAME, or import-library metadata. |
| Public headers | No `SPARSE_API`, `SPARSE_EXPORT`, `SPARSE_IMPORT`, `SPARSE_SHARED`, `SPARSE_STATIC`, or `SPARSE_ABI` macros without an ABI policy. |
| `pkg-config` metadata | No `Libs.private`, static/shared selector, shared-library claim, ABI claim, loader wording, or package-manager distribution wording. |
| CMake package metadata | No static/shared components, shared imported targets, shared target aliases, loader metadata, source-tree paths, or build-tree paths. |
| Installed artifacts | No `.so`, `.so.*`, `.dylib`, or `.dll` artifacts under install `lib` or `bin`. |
| Windows workflow | No Windows Make install/uninstall proof and no Windows `pkg-config` command execution until those lanes are separately selected and proven. |
| Public claims | No broad platform parity, package-manager support, state-of-the-art status, shared-library support, dynamic ABI compatibility, or runtime-loader support claim from current package evidence. |

## Metadata Expectation List

| Producer | Expected Static-First Metadata |
| --- | --- |
| Make install | Installs `libsparse_lu_ortho.a`, installed public headers plus generated `sparse_version.h`, and `lib/pkgconfig/sparse.pc`; uninstall removes those installed files. |
| CMake build/install | Configuring with `BUILD_SHARED_LIBS=ON` fails; `sparse_lu_ortho` is an explicit `STATIC` target; install exports archive-only metadata under `lib/cmake/Sparse`. |
| CMake package consumer | `find_package(Sparse <version> EXACT REQUIRED)` configures, builds, links, and runs against `Sparse::sparse_lu_ortho`; lower same-major mismatched package versions are rejected. |
| `pkg-config` | `prefix`, `libdir`, and `includedir` resolve to the install prefix; `--cflags` returns one include flag; `--libs` returns the static archive link flags; `--static --libs` matches the self-contained current link surface. |
| Windows CMake lane | Installed `.lib`, headers, CMake package files, and metadata-only `sparse.pc` are inspected; no Windows Makefile or `pkg-config` execution claim is implied. |

## Validation Plan

Day 11 implementation should run the smallest quality set that matches the
changed files:

```sh
bash scripts/static_package_deferral_check.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
git diff --check
```

If Day 11 changes any `.c` or `.h` files, the required full quality gate is:

```sh
make format && make lint && make test
```

No `.c` or `.h` changes are expected for the guard implementation path.

## Day 10 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Guard-update design | Complete | Existing and required guard surfaces are listed. |
| Negative-check list | Complete | Unsupported shared-library, ABI, loader, package-manager, and platform claims are scoped. |
| Package metadata expectation list | Complete | Make, CMake, `pkg-config`, and Windows metadata expectations are defined. |
| Validation plan | Complete | Focused guard commands and the conditional full C quality gate are listed. |
| Day 10 guard-design artifact | Complete | This file. |

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Guard changes are scoped before implementation. | Complete | Day 11 has a bounded implementation list. |
| Current package evidence remains static-first unless the decision says otherwise. | Complete | No shared-library support or ABI claim is introduced. |
| Guard validation expectations are clear. | Complete | Focused package/guard commands and conditional C quality gates are listed. |
