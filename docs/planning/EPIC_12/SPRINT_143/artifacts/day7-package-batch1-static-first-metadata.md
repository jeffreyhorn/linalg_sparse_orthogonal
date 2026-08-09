# Sprint 143 Day 7 Package Batch 1 Static-First Metadata

## Purpose

Implement the first selected-path package batch from the Day 6 design. This
batch strengthens the maintained static-first package mechanics without
changing existing static downstream consumer commands.

## Changes Implemented

| Surface | Change | Reason |
| --- | --- | --- |
| `CMakeLists.txt` install metadata | Removed the unused `LIBRARY DESTINATION` entry from the `sparse_lu_ortho` install rule. | The target is explicitly `STATIC`; the install metadata should not carry a shared-library destination that can be mistaken for supported shared packaging. |
| `CMakeLists.txt` `pkg-config` comment | Clarified that generated `pkg-config` metadata belongs to the maintained static archive package surface. | Keeps generated metadata ownership aligned with the Day 5 product decision. |
| `sparse.pc.in` description | Changed the description to `Static archive package metadata for sparse linear algebra`. | Makes the installed `.pc` file identify the selected package contract without adding ABI, loader, package-manager, or shared-library claims. |
| `scripts/static_package_deferral_check.sh` | Added checks for non-static `sparse_lu_ortho` target declarations, static archive install metadata, absence of runtime/shared install destinations, and the static archive `.pc` description. | Converts the Day 5 decision into executable negative proof. |

## Compatibility Notes

- `pkg-config --cflags sparse` remains `-I${includedir}`.
- `pkg-config --libs sparse` remains
  `-L${libdir} -lsparse_lu_ortho -lm @SPARSE_PC_LIBS_EXTRA@`.
- `find_package(Sparse REQUIRED)` and `Sparse::sparse_lu_ortho` remain
  unchanged.
- The static archive target name, installed archive name, installed headers,
  exact-version package behavior, `PREFIX`, and `DESTDIR` staging behavior are
  unchanged.
- No C source or public header files were changed.

## Unsupported Artifact Behavior

This batch keeps unsupported shared-library behavior as rejection and absence:

- `BUILD_SHARED_LIBS=ON` remains a configure-time error.
- `sparse_lu_ortho` remains an explicit `STATIC` target.
- CMake install metadata has an archive destination only.
- Runtime and shared-library install destinations are guarded against.
- `sparse.pc.in` remains free of `Libs.private` and static/shared selectors.

## Focused Validation

Focused checks run for this batch:

```sh
bash -n scripts/static_package_deferral_check.sh
bash -n tests/test_install.sh tests/test_cmake_install.sh
bash scripts/static_package_deferral_check.sh
bash tests/test_cmake_install.sh
bash tests/test_install.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_12/SPRINT_143
```

Results:

| Check | Result |
| --- | --- |
| Shell syntax checks | Passed |
| `scripts/static_package_deferral_check.sh` | Passed |
| `tests/test_install.sh` | Passed: 22 passed, 0 failed |
| `tests/test_cmake_install.sh` | Passed: 21 passed, 0 failed, 0 skipped |
| Package report index check | Passed: 6 rows |
| Package report freshness check | Passed: 6 source-controlled advisory rows |
| `git diff --check` | Passed |
| Trailing-whitespace scan | Passed |

## Day 8 Input

Day 8 should continue with the second package batch:

1. Review install proof diagnostics in `tests/test_install.sh` and
   `tests/test_cmake_install.sh`.
2. Strengthen no-shared-artifact and metadata failure messages where they are
   ambiguous.
3. Preserve generated consumer flags and CMake target behavior.
4. Rerun the install and static deferral proof scripts after any script edits.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected-path mechanics are partially implemented and testable. | Complete | CMake install metadata, `pkg-config` description, and deferral guard were updated. |
| Unsupported package artifacts are rejected or explicitly proved absent. | Complete | Guard now rejects runtime/shared install destinations and non-static target declarations. |
| Static consumer behavior remains coherent. | Complete | Consumer commands, target names, archive names, and link flags are unchanged. |
