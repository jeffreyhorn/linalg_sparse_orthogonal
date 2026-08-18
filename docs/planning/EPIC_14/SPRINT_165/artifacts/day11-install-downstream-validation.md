# Sprint 165 Day 11 Install And Downstream Validation

## Purpose

Day 11 validates install/uninstall behavior and downstream static consumers on
the available local platform. This complements Day 10 metadata inspection by
recording consumer compile/link/run behavior and uninstall cleanup.

## Local Environment

Validation ran on the local macOS Unix toolchain with shell, CMake, compiler,
and `pkg-config` available.

Hosted-only checks were not executed locally:

- Windows CMake-first install/downstream validation;
- Windows metadata-only `sparse.pc` inspection;
- Windows reviewed CTest count inspection.

Deferred/non-claim boundaries remain:

- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- package-manager distribution;
- shared-library packaging;
- runtime-loader behavior;
- dynamic ABI compatibility.

## Make Install And pkg-config Proof

Command:

```text
bash tests/test_install.sh
```

Summary:

```text
Passed: 23
Failed: 0
ALL INSTALL TESTS PASSED
```

Validated behavior:

| Area | Result |
| --- | --- |
| Static archive install | passed |
| No shared-library artifacts | passed |
| Installed headers | all 19 headers installed |
| Installed `sparse.pc` | passed |
| `pkg-config --exists sparse` | passed |
| Exact `pkg-config` version constraint | passed for `2.2.0` |
| Installed path variables | prefix, libdir, and includedir point at installed paths |
| Installed compile/link flags | cflags and libs point at installed include/lib paths |
| Static link flags | `--static` libs match current self-contained flags |
| Unsupported metadata absence | no `Libs.private` stanza and no unsupported package/ABI wording |
| Generated downstream consumer | compiled, linked, and ran |
| Maintained example via `pkg-config` | compiled, linked, and ran |
| Uninstall cleanup | library, headers, and `sparse.pc` removed |

## CMake Install And find_package Proof

Command:

```text
bash tests/test_cmake_install.sh
```

Summary:

```text
Passed: 27
Failed: 0
Skipped: 0
ALL CMAKE INSTALL TESTS PASSED
```

Validated behavior:

| Area | Result |
| --- | --- |
| CMake configure/build/install | passed |
| Static archive install | passed |
| No shared-library artifacts | passed |
| Installed headers | all 19 headers installed |
| CMake package files | `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and `SparseTargets.cmake` installed |
| Installed `sparse.pc` | passed |
| Static imported target metadata | passed |
| Unsupported CMake metadata absence | no shared imported metadata and no loader/static-shared selector metadata |
| Source/build path leaks | absent |
| `sparse.pc` static archive metadata | passed |
| Maintained CMake example | configured, built, and ran through `find_package(Sparse)` |
| Exact-version CMake consumer | configured, built, and ran |
| Mismatched-version CMake consumer | rejected |
| `pkg-config --modversion` | returned `2.2.0` |

## Static Deferral Guard

Command:

```text
bash scripts/static_package_deferral_check.sh
```

Result: passed.

The guard confirmed:

- `BUILD_SHARED_LIBS=ON` remains a configure-time rejection;
- the CMake target remains explicitly static;
- static install metadata remains bounded;
- public export/import and static/shared ABI macros remain absent;
- package metadata has no static/shared selector;
- support wording remains deferred;
- Windows workflow wording does not claim unselected package execution.

## Version Handling Notes

- `pkg-config --exists "sparse = 2.2.0"` succeeded.
- `pkg-config --modversion sparse` returned `2.2.0`.
- `find_package(Sparse 2.2.0 EXACT REQUIRED)` configured, built, and ran.
- A lower same-major mismatched CMake package version was rejected.

These checks prove exact package metadata behavior for installed package
resolution. They are not dynamic ABI compatibility evidence.

## Completion Check

- Static archive downstream consumers build and run through both maintained
  Unix package routes.
- Uninstall behavior remains covered for Make install artifacts.
- Exact-version success and mismatch failure behavior are recorded.
- Hosted-only Windows validation requirements and deferred parity boundaries
  remain explicit.
