# Epic 3 Warning Workflow Summary: day13-final

**Date:** 2026-05-16  
**Repository root:** `/Users/jeff/experiments/linalg_sparse_orthogonal`

## Build Paths

- CMake build directory: `build/day13-final-cmake`
- Makefile build directory: `build/day13-final-make`
- Artifact directory: `docs/planning/EPIC_3/SPRINT_30/artifacts`
- CMake build jobs: `1`

## Commands Run

1. `cmake -S . -B build/day13-final-cmake`
2. `cmake --build build/day13-final-cmake --parallel 1`
3. `ctest --test-dir build/day13-final-cmake --output-on-failure`
4. `make BUILDDIR=build/day13-final-make all`

## Warning Counts

- CMake configure warnings: `0`
- CMake build warnings: `112`
- Makefile build warnings: `0`

## Derived Summaries

- `day13-final-cmake-warning-counts-by-area.txt`
- `day13-final-cmake-warning-counts-by-class.txt`
- `day13-final-cmake-warning-counts-by-file.txt`

## Raw Logs

- `day13-final-cmake-configure.stdout.txt`
- `day13-final-cmake-configure.stderr.txt`
- `day13-final-cmake-build.stdout.txt`
- `day13-final-cmake-build.stderr.txt`
- `day13-final-ctest.stdout.txt`
- `day13-final-ctest.stderr.txt`
- `day13-final-make-build.stdout.txt`
- `day13-final-make-build.stderr.txt`

## Notes

- The CMake path is the authoritative full-tree warning inventory for Sprint 30.
- The Makefile `all` path remains a library-only cross-check unless a wider Makefile target is used.
- The workflow defaults to a serialized CMake warning capture (`WARNING_WORKFLOW_JOBS=1`) so area/file attribution remains stable and non-interleaved.
- Re-run this workflow with a new label before and after edits to compare summaries without overwriting prior artifacts.
