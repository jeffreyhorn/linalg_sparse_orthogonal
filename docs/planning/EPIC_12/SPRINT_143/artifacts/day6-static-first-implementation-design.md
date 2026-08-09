# Sprint 143 Day 6 Static-First Implementation Design

## Purpose

Convert the Day 5 package/ABI product decision into a bounded implementation
design before editing build, install, package, script, CI, or public
documentation surfaces.

The selected path is stricter static-first-only support. This design protects
existing static consumers and makes unsupported shared-library behavior
observable without adding shared-library build, loader, ABI, or package-manager
claims.

## Selected Path Summary

Sprint 143 will maintain one package contract:

- static archive install;
- installed public headers;
- CMake `find_package(Sparse)` static imported target;
- `pkg-config` static archive consumer flags;
- exact package version checks;
- no installed `.so`, `.dylib`, or `.dll` artifacts;
- explicit rejection of `BUILD_SHARED_LIBS=ON`;
- explicit non-claims for shared libraries, dynamic ABI compatibility,
  runtime-loader behavior, package-manager support, and broad platform parity.

## Edit Order

| Order | Batch | Goal | Expected day |
| ---: | --- | --- | --- |
| 1 | Static-first guard design | Tighten unsupported shared-library negative proof before metadata edits. | Day 7 |
| 2 | Install/export metadata | Clarify CMake and `pkg-config` static-only package metadata without changing consumer commands. | Day 7-8 |
| 3 | Install proof diagnostics | Improve Make/CMake install test messages and static/no-shared checks. | Day 8-9 |
| 4 | CI support-tier alignment | Keep Linux reviewed package baseline and macOS/Windows supplemental confidence wording aligned. | Day 10 |
| 5 | Public and maintainer docs | Align README, INSTALL, maintainer guide, and package comments with the selected contract. | Day 11 |
| 6 | Report boundary check | Preserve package report rows as proof-owner metadata unless row meaning changes deliberately. | Day 11-12 |
| 7 | Validation and closeout | Run selected package checks, quality checks for touched surfaces, and publish Sprint 144 handoff. | Day 12-14 |

## File Ownership Map

| File or path | Owner role | Planned change | Compatibility rule |
| --- | --- | --- | --- |
| `CMakeLists.txt` | CMake package owner | Preserve explicit `STATIC` target and `BUILD_SHARED_LIBS=ON` rejection; clarify comments or metadata only if needed. | Do not rename `sparse_lu_ortho`; do not add shared targets, selectors, `SOVERSION`, visibility, install-name, or DLL runtime install behavior. |
| `cmake/SparseConfig.cmake.in` | CMake package owner | Keep package config focused on the existing static imported target. | Preserve `find_package(Sparse REQUIRED)` and `Sparse::sparse_lu_ortho`. |
| `sparse.pc.in` | `pkg-config` package owner | Clarify the static archive consumer contract if comments or generated metadata comments are added later. | Preserve `pkg-config --cflags --libs sparse` output for existing consumers. |
| `Makefile` | Make install owner | Preserve static archive install/uninstall behavior and generated `sparse.pc` substitution. | Preserve `make install PREFIX=...` and `DESTDIR` staging semantics. |
| `tests/test_install.sh` | Make install proof owner | Strengthen diagnostics for static archive, no-shared artifacts, `.pc` variables, link flags, and downstream consumers. | Preserve existing successful static `pkg-config` compile/link/run path. |
| `tests/test_cmake_install.sh` | CMake install proof owner | Strengthen diagnostics for static imported-target metadata, no source/build path leaks, no shared artifacts, and downstream consumers. | Preserve existing installed `find_package(Sparse)` example behavior. |
| `scripts/static_package_deferral_check.sh` | Unsupported shared guard owner | Strengthen negative checks and wording around static-first decision, no selectors, no export/import macros, and no shared ABI metadata. | Keep `BUILD_SHARED_LIBS=ON` rejected and avoid false positives against legitimate static package wording. |
| `.github/workflows/ci.yml` | Linux CI owner | Preserve Linux reviewed static-first package-contract lane wording and commands. | Do not imply shared ABI, package-manager support, or platform parity. |
| `.github/workflows/macos-ci.yml` | macOS CI owner | Preserve supplemental package confidence wording. | Do not promote macOS package parity in Sprint 143. |
| `.github/workflows/windows-ci.yml` | Windows CI owner | Preserve reviewed CMake subset and supplemental install/downstream confidence wording. | Do not add reviewed Windows install-validation parity, Makefile parity, `pkg-config`, or DLL support in Sprint 143. |
| `README.md` | Public docs owner | Clarify selected static-first contract and shared-library deferral if current wording is ambiguous after implementation. | Do not widen public claims beyond validated package proof. |
| `INSTALL.md` | Install docs owner | Keep user-facing install instructions aligned with static archive, `pkg-config`, CMake, exact version, and support-tier proof. | Preserve existing install commands. |
| `docs/maintainer_guide.md` | Maintainer docs owner | Document proof ownership, stop conditions, and Sprint 144 platform separation. | Keep runtime/backend and package/ABI boundaries separate. |
| `tests/corpus/manifests/report_families.tsv` and related package rows | Report owner | No planned row-semantic change; run checks to confirm source-controlled proof-owner rows stay valid. | Do not imply fresh local install-run evidence unless row semantics change. |

## Compatibility Behavior

Existing static consumers must continue to work:

1. Unix Make install consumers keep using
   `pkg-config --cflags --libs sparse`.
2. CMake consumers keep using `find_package(Sparse REQUIRED)` and
   `Sparse::sparse_lu_ortho`.
3. Exact-version behavior remains conservative through
   `SparseConfigVersion.cmake`.
4. Installed header count remains 19 unless a later explicit public-header
   change is made and validated.
5. Static archive artifact names stay stable:
   `libsparse_lu_ortho.a` on Unix-like installs and `sparse_lu_ortho.lib` on
   MSVC CMake installs.
6. `DESTDIR` and `PREFIX` staging behavior remains unchanged.
7. `SPARSE_MUTEX` and `SPARSE_OPENMP` link-flag substitutions remain
   build-option metadata, not ABI claims.

## Negative-Proof Behavior

The stricter static-first path should prove unsupported shared behavior by
absence and rejection:

- `BUILD_SHARED_LIBS=ON` configure requests fail with explicit static-first
  and shared ABI deferral wording.
- CMake continues to declare `sparse_lu_ortho` as an explicit `STATIC` target.
- Installed package directories contain no `.so`, `.so.*`, `.dylib`, or
  `.dll` artifacts.
- Public headers contain no `SPARSE_API`, `SPARSE_EXPORT`, or `SPARSE_IMPORT`
  macro policy.
- CMake metadata contains no `SOVERSION`, `WINDOWS_EXPORT_ALL_SYMBOLS`,
  `C_VISIBILITY_PRESET`, `VISIBILITY_INLINES_HIDDEN`, install-name, or soname
  policy.
- CMake package metadata contains no shared/static component selector.
- `sparse.pc` contains no `Libs.private` or shared/static selector under the
  current self-contained static link contract.
- Public docs and maintainer docs keep shared-library packaging, dynamic ABI,
  loader behavior, package-manager support, and platform parity as explicit
  non-claims.

## Focused Validation Plan

| Touched surface | Focused commands |
| --- | --- |
| Planning artifacts only | `git diff --check`; `rg -n "[ \t]+$" docs/planning/EPIC_12/SPRINT_143` |
| Shell scripts | `bash -n tests/test_install.sh tests/test_cmake_install.sh scripts/static_package_deferral_check.sh` |
| Make install or `sparse.pc.in` | `bash tests/test_install.sh` |
| CMake install/export metadata | `bash tests/test_cmake_install.sh` |
| Unsupported shared-library guard | `bash scripts/static_package_deferral_check.sh` |
| Package report rows | `python3 scripts/normalize_report_index.py --family package --check`; `python3 scripts/normalize_report_index.py --family package --check-freshness` |
| CI workflow comments or commands | Focused `rg` review for support-tier wording, package commands, expected Windows CTest count, and non-claims. |
| README, INSTALL, maintainer guide | Focused claim scan for `shared`, `ABI`, `package-manager`, `platform parity`, `pkg-config`, and `find_package`. |
| C or public headers | Focused behavior tests, then `make format && make lint && make test`. |

## Implementation Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Guard script becomes too broad and rejects legitimate static wording. | CI or local package proof fails despite correct static behavior. | Use targeted regex checks and run the guard after every script/doc wording change. |
| CMake comments or metadata imply shared support. | Public package contract widens without proof. | Keep target explicitly static and keep all shared ABI metadata absent. |
| `pkg-config` wording breaks generated output checks. | Existing downstream `pkg-config` consumers fail. | Preserve `Cflags` and `Libs` output; validate with `tests/test_install.sh`. |
| CI wording promotes macOS or Windows package parity. | Sprint 143 overclaims platform support. | Keep macOS/Windows package jobs supplemental and route promotion to Sprint 144. |
| Docs cite package report freshness as fresh install proof. | Report rows overclaim validation status. | Keep rows described as source-controlled proof-owner metadata. |
| Static consumer command surface changes accidentally. | Existing adopters need migration for no product reason. | Preserve Make, CMake, and `pkg-config` commands and target names. |

## Day 7 Input

Day 7 should implement the first package batch in this order:

1. Update `scripts/static_package_deferral_check.sh` only if the current
   negative-proof checks need sharper wording or coverage.
2. Update CMake/package comments only where they clarify the selected
   static-first contract without changing generated consumer behavior.
3. Run shell syntax checks and the static package deferral guard.
4. Stop before any change that adds shared-library metadata or alters existing
   static consumer commands.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Implementation edits are scoped before changes begin. | Complete | Edit order and file ownership map define the batches and affected files. |
| Existing static consumers remain protected unless the decision explicitly changes them. | Complete | Compatibility behavior preserves Make, CMake, `pkg-config`, target names, archive names, headers, and version behavior. |
| Validation commands are mapped to touched surfaces. | Complete | Focused validation plan maps planning, scripts, Make, CMake, package rows, CI, docs, and C/header changes to commands. |
