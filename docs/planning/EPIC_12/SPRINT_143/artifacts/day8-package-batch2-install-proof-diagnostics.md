# Sprint 143 Day 8 Package Batch 2 Install Proof Diagnostics

## Purpose

Complete the selected static-first package implementation batch by
strengthening install proof diagnostics and metadata checks. This batch keeps
the package contract static-only and does not change public headers, C source,
consumer commands, target names, or archive names.

## Changes Implemented

| Surface | Change | Reason |
| --- | --- | --- |
| `tests/test_install.sh` | Improved the missing `sparse.pc` failure message to include the expected file path. | Makes Make install failures easier to triage. |
| `tests/test_install.sh` | Added an installed `.pc` description check for `Static archive package metadata for sparse linear algebra`. | Ensures the Day 7 static-first metadata decision is covered by executable Make install proof. |
| `tests/test_cmake_install.sh` | Improved the missing `sparse.pc` failure message to include the expected file path. | Makes CMake install failures easier to triage. |
| `tests/test_cmake_install.sh` | Added a negative CMake package scan for shared/module imported targets and shared imported locations. | Ensures installed CMake package metadata does not imply shared-library support. |
| `tests/test_cmake_install.sh` | Added installed `.pc` static-description and unsupported package/ABI wording checks. | Aligns CMake install/export proof with the static-first package metadata contract. |

## Compatibility Notes

- `pkg-config --cflags sparse` and `pkg-config --libs sparse` are unchanged.
- `find_package(Sparse REQUIRED)` and `Sparse::sparse_lu_ortho` are unchanged.
- `libsparse_lu_ortho.a`, `sparse_lu_ortho.lib`, installed header count,
  exact-version checks, and downstream example behavior are unchanged.
- The new checks add proof around generated metadata; they do not change
  generated link flags or CMake target names.

## Repair And Deferral Notes

No implementation repair was required after the Day 8 edits.

Deferred by decision:

- shared-library build/install/export;
- dynamic ABI compatibility;
- Linux/macOS/Windows loader behavior;
- CMake or `pkg-config` static/shared selectors;
- package-manager distribution;
- macOS and Windows reviewed package-lane promotion.

## Focused Validation

Focused checks run for this batch:

```sh
bash -n tests/test_install.sh tests/test_cmake_install.sh scripts/static_package_deferral_check.sh
bash scripts/static_package_deferral_check.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_12/SPRINT_143 tests/test_install.sh tests/test_cmake_install.sh
```

Results:

| Check | Result |
| --- | --- |
| Shell syntax checks | Passed |
| `scripts/static_package_deferral_check.sh` | Passed |
| `tests/test_install.sh` | Passed: 23 passed, 0 failed |
| `tests/test_cmake_install.sh` | Passed: 24 passed, 0 failed, 0 skipped |
| Package report index check | Passed: 6 rows |
| Package report freshness check | Passed: 6 source-controlled advisory rows |

## Day 9 Input

Day 9 should review downstream consumer proof end to end:

1. Confirm the Make `pkg-config` consumer and maintained example still compile,
   link, and run against the installed static archive.
2. Confirm the CMake installed example and exact/mismatched version checks
   remain deterministic.
3. Confirm unsupported artifacts are checked explicitly without converting
   missing shared support into pass evidence.
4. Add loader/runtime checks only if the selected path changes, which it has
   not.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The selected product path is mechanically complete. | Complete | Static-only CMake metadata, `.pc` metadata, guard script, and install proof scripts now align with the Day 5 decision. |
| Package metadata and guards match the Day 5 decision. | Complete | Tests check static `.pc` description, no shared imported metadata, no `Libs.private`, no shared artifacts, and no unsupported ABI/package wording. |
| Remaining implementation risk has an owner and stop condition. | Complete | Shared ABI, loader, package selectors, package-manager support, and platform promotion remain deferred with Day 5 stop conditions. |
