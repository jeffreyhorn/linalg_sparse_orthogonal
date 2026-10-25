# Day 3: CMake Width Selection

**Date:** 2026-10-24
**Scope:** Sprint 219 item 219.2, first 10 of 30 planned hours
**Host:** macOS x86_64, Apple Clang 17.0.0

## Implementation

`CMakeLists.txt` now defines the cache selector `SPARSE_IDX_BITS`, defaulting
to `32` and accepting only exact `32` or `64`. Invalid values fail during
configure with `SPARSE_IDX_BITS must be 32 or 64`. The configure path also
rejects an ambient `SPARSE_IDX_BITS` definition in `CMAKE_C_FLAGS*`, including
quoted and split `-D` forms; such a flag could otherwise disagree with the
selected width. The static library
publishes the chosen definition through its build interface, which reaches
all linked tests, examples, and benchmarks. The install interface is
intentionally unchanged until the package-width work on Days 6-8.

The existing `test_idx_width_contract` in `tests/test_sparse_matrix.c`
compares the consumer's `sizeof(idx_t) * CHAR_BIT`, `SPARSE_IDX_BITS`, and
the archive's `sparse_idx_bits()`. The CMake install script now covers five
invalid values and a conflicting ambient compiler flag at configure time.

## Clean-Tree Evidence

| Width | Configure/build/test | Result |
| --- | --- | --- |
| 32 | `cmake -S . -B build/s219-day3-cmake-32 -DSPARSE_IDX_BITS=32 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`; `cmake --build build/s219-day3-cmake-32 --target test_sparse_matrix --parallel 4`; `ctest --test-dir build/s219-day3-cmake-32 --output-on-failure -R '^test_sparse_matrix$'` | Configure/build exited 0; focused CTest passed 1/1. Its width assertion binds 32-bit `idx_t` to a 32-bit archive query. |
| 64 | Same commands under `build/s219-day3-cmake-64` with selector `64` | Configure/build exited 0; focused CTest passed 1/1. Its width assertion binds 64-bit `idx_t` to a 64-bit archive query. |

Both trees were newly configured for this day. A `jq` check over each
`compile_commands.json` returned `true` for the assertion that every one of
139 project compile commands contains its tree's `-DSPARSE_IDX_BITS=N`.
Sample commands include `src/sparse_types.c`, `tests/test_sparse_matrix.c`,
`benchmarks/bench_matmul.c`, and `examples/example_basic_solve.c`. The latter
two 64-bit targets were also built successfully without running a benchmark
or example solver. The CMake build emitted existing Apple Clang
`-Wdouble-promotion` warnings about `NAN`/`INFINITY`; they were not build
errors and are not evidence of a width-contract failure.

## Configure Rejection And Package Regression

`CMAKE_BUILD_PARALLEL_LEVEL=4 bash tests/test_cmake_install.sh` passed
38 checks, failed 0, skipped 0. Its new configure-negative cases returned
the expected diagnostic:

| Selector attempt | Expected failure |
| --- | --- |
| `-DSPARSE_IDX_BITS=0`, `16`, `128`, `text`, or empty | Configure rejects with `SPARSE_IDX_BITS must be 32 or 64`. |
| `-DSPARSE_IDX_BITS=32` with a plain, quoted, or split `-DSPARSE_IDX_BITS=64` in `CMAKE_C_FLAGS` | Configure rejects with `do not define it in CMAKE_C_FLAGS`. |

The suite also built and installed the default package, validated its
static-only files, and compiled/ran the existing `find_package(Sparse)`
consumer. No new installed-width guarantee is inferred from that default
consumer: `sparse_version.h`, `sparse.pc`, and the imported target still lack
the Day 2 package-width identity. Duplicate `-D` options for the same CMake
cache key are reduced by CMake to an effective final value before project
logic can inspect them; this day validates that effective value and rejects
ambient compiler definitions, not the history of CLI arguments.

## Handoff

Day 4 adds the equivalent Make selector and stale-tree protection. Day 5
compares clean Make/CMake trees and generated identities. Days 6-8 must add
the installed header, target, pkg-config, validator, and pre-use archive
check described in the [Day 2 design](day2-width-contract-failure-design.md).
Day 3 demonstrates 32/64 CMake build-tree selection, not 64-bit installed
consumer support or broad 64-bit algorithm correctness.
