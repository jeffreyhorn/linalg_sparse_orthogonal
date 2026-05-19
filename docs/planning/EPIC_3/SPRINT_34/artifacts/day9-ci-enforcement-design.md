# Sprint 34 Day 9: CI enforcement design

## Goal

Define a non-flaky first-phase CI integration plan for the Sprint 34 reviewed quality paths without overloading unrelated sanitizer, benchmark-runtime, or portability jobs.

## Current workflow surface

### Linux CI

- `build-and-test`
  - `make test`
  - `make sanitize`
  - `make asan`
  - `make bench-build`
  - `make bench-fast`
- `cmake-build-and-test`
  - configure
  - build
  - full `ctest`
  - CMake/Makefile test-count parity check
- `tsan`
- `lint`
  - `make format-check`
  - `make lint`
- `coverage`

### macOS CI

- compiler matrix build/test
- `make wall-check`
- Apple Clang `make sanitize`

### Windows CI

- MSVC CMake configure/build/ctest

## Phase-1 enforced jobs

Sprint 34 should enforce the new reviewed paths in exactly three Linux jobs:

1. reviewed Makefile compile-quality job
2. reviewed CMake parity job
3. serialized dead-code job

## Chosen job wiring

### Linux lint job

Replace the current split:

- `make format-check`
- `make lint`

with:

- `make quality-review-compile`

Reason:

- same quality surface
- better phase attribution through the Sprint 34 wrapper banners
- direct CI use of the maintained reviewed Makefile path

### Linux CMake job

Replace the current hand-written configure/build/test sequence with:

- `make quality-review-cmake`

Reason:

- direct CI use of the maintained reviewed CMake path
- keeps `ctest -N` visible in CI
- preserves clean rebuild + full CTest parity in one named flow

### New Linux dead-code job

Add a dedicated single-runner job that:

1. installs `cppcheck` and the toolchain prerequisites needed to build `xunused`
2. builds and installs `xunused` in the job environment
3. runs:
   - `make deadcode-report`
   - `make deadcode-check`
4. uploads dead-code artifacts on `if: always()`

Artifacts to upload:

- `build/deadcode/report.md`
- `build/deadcode/report.tsv`
- `build/deadcode/coverage-notes.txt`
- `build/deadcode/cppcheck.txt`
- `build/deadcode/xunused.txt`

## Why Linux-first

- `cppcheck` already fits naturally in the Linux CI environment
- the dead-code scripts are currently POSIX-oriented
- it avoids forcing the initial xunused-install burden across macOS and Windows before the Linux path is stable
- Windows/MSVC warning semantics remain a separate portability/quality-parity concern for later sprints

## Explicit phase-1 non-goals

Do not repurpose these jobs yet:

- Linux `build-and-test`
- Linux `tsan`
- Linux `coverage`
- macOS matrix jobs
- Windows build/test job

Those jobs should keep their current purpose:

- runtime execution
- sanitizer coverage
- benchmark runtime signal
- portability
- MSVC build/test parity

## Dead-code execution rule in CI

The inherited Sprint 33 shared-path constraint remains active:

- no dead-code matrix
- no parallel sibling deadcode targets
- one job
- one runner
- one artifact/build-tree pair:
  - `build/deadcode-cmake`
  - `build/deadcode/`

The CI order must stay serial:

1. install tools
2. build/install `xunused`
3. `make deadcode-report`
4. `make deadcode-check`
5. upload artifacts with `if: always()`

## Failure-output contract

### reviewed Makefile compile-quality job

Must preserve wrapper banners so failures clearly map to:

- `format-check`
- `lint`

### reviewed CMake parity job

Must preserve wrapper banners so failures clearly map to:

- configure
- build
- `ctest -N`
- full `ctest`

### dead-code job

Must use explicit step names and always upload artifacts so failures remain actionable:

- install dead-code tools
- build/install `xunused`
- generate dead-code report
- validate dead-code report
- upload dead-code artifacts

The uploaded artifacts must let a contributor distinguish between:

- tool-install failure
- xunused-build failure
- workflow generation failure
- report completeness failure
- preserved compile-db coverage-gap limitation

## Expected Day 10 implementation

Day 10 should update `.github/workflows/ci.yml` only:

- convert Linux `lint` to `make quality-review-compile`
- convert Linux `cmake-build-and-test` to `make quality-review-cmake`
- add one serialized Linux `deadcode` job with artifact upload

macOS and Windows should remain unchanged in phase 1.
