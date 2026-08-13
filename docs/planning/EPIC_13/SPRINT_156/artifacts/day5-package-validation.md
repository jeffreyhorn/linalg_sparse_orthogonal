# Sprint 156 Day 5 Package Validation

## Purpose

Day 5 validates the static-first package, install, CMake export, `pkg-config`,
and downstream consumer evidence inherited from Epic 13 package work. This
artifact records local package proof and keeps platform/package non-claims
separate from what the commands actually prove.

## Commands Run

```sh
bash scripts/static_package_deferral_check.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness
```

## Static Package Deferral Guard

| Check | Result |
| --- | --- |
| `BUILD_SHARED_LIBS=ON` rejection | Pass |
| explicit static target declaration | Pass |
| static install metadata | Pass |
| no shared export/ABI metadata | Pass |
| no package static/shared selector | Pass |
| support wording remains deferred | Pass |
| overall `static_package_deferral_check.sh` | Pass |

The guard confirms the package surface still rejects shared-library requests
and keeps shared-library packaging, dynamic ABI compatibility, runtime-loader
behavior, static/shared selectors, and package-manager support out of the
maintained product claim.

## Make Install And `pkg-config` Proof

Command:

```sh
bash tests/test_install.sh
```

Result: passed with `23` checks and `0` failures.

| Proof Area | Result |
| --- | --- |
| static library installed | Pass |
| no shared-library artifacts installed | Pass |
| installed headers | Pass: `19` headers |
| `sparse.pc` installed | Pass |
| `pkg-config` can resolve `sparse` | Pass |
| exact version constraint | Pass |
| `pkg-config` prefix/libdir/includedir | Pass |
| `pkg-config --cflags` installed include path | Pass |
| `pkg-config --libs` static archive link flags | Pass |
| `pkg-config --static --libs` matches current self-contained link flags | Pass |
| no `Libs.private` stanza | Pass |
| static archive package description | Pass |
| no unsupported package or ABI claims in `sparse.pc` | Pass |
| `pkg-config --modversion` | Pass: `2.2.0` |
| basic downstream consumer compile/link/run | Pass |
| maintained example compile/run through installed package | Pass |
| uninstall removes library, headers, and `sparse.pc` | Pass |

## CMake Install And Downstream Proof

Command:

```sh
bash tests/test_cmake_install.sh
```

Result: passed with `27` checks, `0` failures, and `0` skips.

| Proof Area | Result |
| --- | --- |
| CMake configure/build/install | Pass |
| static library installed | Pass |
| no shared-library artifacts installed | Pass |
| installed headers | Pass: `19` headers |
| `SparseConfig.cmake` installed | Pass |
| `SparseConfigVersion.cmake` installed | Pass |
| `SparseTargets.cmake` installed | Pass |
| `sparse.pc` installed | Pass |
| imported CMake target is static | Pass |
| no shared-library imported metadata | Pass |
| no unsupported loader/shared-selector metadata | Pass |
| imported target include/archive paths use install prefix | Pass |
| no source-tree or build-tree paths in installed package | Pass |
| `sparse.pc` describes static archive package | Pass |
| no unsupported package or ABI claims in `sparse.pc` | Pass |
| `examples/cmake_example` configures/builds/runs via `find_package(Sparse)` | Pass |
| exact installed version configure/build/run | Pass |
| mismatched package version rejected | Pass |
| `pkg-config` version | Pass: `2.2.0` |

## Report Index Package Checks

| Command | Result | Boundary |
| --- | --- | --- |
| `python3 scripts/normalize_report_index.py --family package --check` | Pass: `6` rows ok | Structure check for package report rows. |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Pass: freshness ok for `6` source-controlled rows | Source-controlled rows are governed by schema and Git review. |
| `python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness` | Pass: freshness ok for `1` source-controlled row | Runtime-backend row remains source-controlled evidence, not generated runtime proof. |

## Installed Surface Comparison

The maintained local package proofs agree on the installed static-first
surface:

- one static archive library;
- no `.so`, `.dylib`, `.dll`, or shared-library artifacts;
- `19` installed public headers;
- installed CMake package metadata for `Sparse::sparse_lu_ortho`;
- installed `sparse.pc` metadata for the static archive package;
- downstream C and CMake consumers compile, link, and run.

## Package Evidence Boundaries

This Day 5 package validation supports only local static-first package
confidence for the checked commands. It does not prove:

- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- package-manager distribution;
- static/shared package selectors;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad Windows package parity;
- hosted Linux/macOS/Windows package status;
- portable performance;
- external-library parity;
- state-of-the-art sparse linear algebra status.

## Platform Package Residuals

| Residual | Status | Promotion Gate |
| --- | --- | --- |
| Hosted Linux package proof | Deferred to Day 6 platform reconciliation | Review final Linux CI package/install lanes and tie claims only to passing hosted evidence. |
| Hosted macOS package proof | Deferred to Day 6 platform reconciliation | Review final macOS CI package/install lanes and preserve support-tier wording. |
| Hosted Windows CMake install/downstream proof | Deferred to Day 6 platform reconciliation | Review final Windows CMake install/downstream job outcomes. |
| Windows Makefile parity | Deferred non-claim | Implement, test, and review Windows Makefile install/build behavior before claiming. |
| Windows `pkg-config` execution parity | Deferred non-claim | Install and exercise Windows `pkg-config` in a reviewed lane before claiming. |
| Shared-library ABI support | Deferred product decision | Design export/import macros, symbol visibility, ABI policy, shared build/install/export, loader proof, and hosted platform validation. |
| Package-manager support | Deferred productization work | Create actual package-manager artifacts and CI proof before claiming. |

## Day 5 Completion Check

- Make install and `pkg-config` proof passed.
- CMake install/export and downstream proof passed.
- Static package deferral guard passed.
- Package and runtime-backend report-index checks passed.
- Installed header/library/package metadata surface is recorded.
- Shared-library, dynamic ABI, package-manager, Windows Makefile, and Windows
  `pkg-config` non-claims remain intact.
