# Sprint 165 Day 8 Downstream Proof Implementation

## Purpose

Day 8 implements the concrete stale-expectation fixes identified by the Day 7
downstream proof scope. The implementation stays focused on proof-script
robustness and does not expand the supported package surface.

## Implementation Summary

| File | Change | Reason |
| --- | --- | --- |
| `tests/test_install.sh` | Added `same_dir` and used it for `pkg-config` `prefix`, `libdir`, `includedir`, `--cflags`, and `--libs` installed-path checks. | Prevent valid staged installs from failing because of raw path spelling differences such as double slashes, while still requiring the resolved path to identify the installed directory. |

## Day 7 Risks Closed

| Day 7 Risk | Day 8 Result |
| --- | --- |
| Raw path spelling comparisons | Closed for Unix Make/`pkg-config` path assertions in `tests/test_install.sh`. |
| Output shape assumptions | No edit required. `tests/test_install.sh`, `tests/test_cmake_install.sh`, and `.github/workflows/windows-ci.yml` already check required semantic output tokens rather than requiring a single-line exact output string. |
| Hard-coded Windows CTest count | No edit required today. `.github/workflows/windows-ci.yml` currently records `EXPECTED_WINDOWS_CTEST_COUNT: "59"` for the reviewed CMake test surface. Future CMake test additions/removals must update this explicitly. |
| Header count drift | No edit required. Unix install scripts derive checked-in public headers plus generated `sparse_version.h`; Windows install validation checks installed headers and the generated version header. |
| Windows `sparse.pc` wording | No edit required. Windows workflow text says `sparse.pc` validation is metadata-only and does not claim Windows `pkg-config` execution parity. |
| Exact-version metadata interpretation | No edit required. Existing README, INSTALL, CMake, and maintainer wording continue to state that exact package metadata is not a dynamic ABI guarantee. |

## Proof Scope After Implementation

The maintained downstream proof remains:

- Unix Make install/uninstall plus `pkg-config` command execution;
- Unix CMake install/export plus `find_package(Sparse)` consumers;
- macOS hosted execution of the same Unix proof scripts;
- Windows CMake-first install/downstream validation;
- Windows metadata-only `sparse.pc` inspection.

The implementation does not add or imply:

- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- package-manager distribution;
- shared-library support;
- runtime-loader behavior;
- dynamic ABI support.

## Validation

Shell syntax:

```text
bash -n tests/test_install.sh tests/test_cmake_install.sh scripts/static_package_deferral_check.sh
```

Result: passed.

Focused install proof:

```text
bash tests/test_install.sh
```

Result:

```text
Passed: 23
Failed: 0
ALL INSTALL TESTS PASSED
```

CMake install/export proof:

```text
bash tests/test_cmake_install.sh
```

Result:

```text
Passed: 27
Failed: 0
Skipped: 0
ALL CMAKE INSTALL TESTS PASSED
```

Static package deferral proof:

```text
bash scripts/static_package_deferral_check.sh
```

Result: passed.

## Completion Check

- Downstream proof now uses filesystem identity for Unix `pkg-config`
  installed-path checks.
- Exact-version checks remain active in both `pkg-config` and CMake proof
  paths.
- Installed `sparse.pc` and CMake package checks continue to enforce the static
  archive boundary.
- Unsupported Windows Makefile and `pkg-config` execution parity remain
  non-claims.
