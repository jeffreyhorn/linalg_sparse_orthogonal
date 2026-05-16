# Epic 3 Warning Workflow Summary: day7-workflow

**Date:** 2026-05-16  
**Repository root:** `/Users/jeff/experiments/linalg_sparse_orthogonal`

## Build Paths

- CMake build directory: `build/day7-workflow-cmake`
- Makefile build directory: `build/day7-workflow-make`
- Artifact directory: `docs/planning/EPIC_3/SPRINT_30/artifacts`

## Commands Run

1. `cmake -S . -B build/day7-workflow-cmake`
2. `cmake --build build/day7-workflow-cmake -j4`
3. `ctest --test-dir build/day7-workflow-cmake --output-on-failure`
4. `make BUILDDIR=build/day7-workflow-make all`

## Warning Counts

- CMake configure warnings: `0`
- CMake build warnings: `112`
- Makefile build warnings: `0`

## Derived Summaries

- `day7-workflow-cmake-warning-counts-by-area.txt`
- `day7-workflow-cmake-warning-counts-by-class.txt`
- `day7-workflow-cmake-warning-counts-by-file.txt`

## Raw Logs

- `day7-workflow-cmake-configure.stdout.txt`
- `day7-workflow-cmake-configure.stderr.txt`
- `day7-workflow-cmake-build.stdout.txt`
- `day7-workflow-cmake-build.stderr.txt`
- `day7-workflow-ctest.stdout.txt`
- `day7-workflow-ctest.stderr.txt`
- `day7-workflow-make-build.stdout.txt`
- `day7-workflow-make-build.stderr.txt`

## Notes

- The CMake path is the authoritative full-tree warning inventory for Sprint 30.
- The Makefile `all` path remains a library-only cross-check unless a wider Makefile target is used.
- Re-run this workflow with a new label before and after edits to compare summaries without overwriting prior artifacts.
