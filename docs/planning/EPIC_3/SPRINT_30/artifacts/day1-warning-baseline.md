# Sprint 30 Day 1 Warning Baseline

**Date:** 2026-05-16  
**Branch:** `sprint-30`

## Scope

Day 1 reran the Sprint 30 baseline capture in bounded steps with dedicated build directories and artifact files:

- CMake configure: `build/sprint30-day1-cmake`
- CMake full build: `build/sprint30-day1-cmake`
- Makefile default build path: `BUILDDIR=build/sprint30-day1-make make all`

## Raw Artifacts

- `day1-cmake-configure.stdout.txt`
- `day1-cmake-configure.stderr.txt`
- `day1-cmake-build.stdout.txt`
- `day1-cmake-build.stderr.txt`
- `day1-make-build.stdout.txt`
- `day1-make-build.stderr.txt`

Derived summaries:

- `day1-cmake-warning-counts-by-area.txt`
- `day1-cmake-warning-counts-by-class.txt`
- `day1-cmake-warning-counts-by-file.txt`

## Results

### CMake Configure

- Result: success
- stderr size: `0 B`
- warnings observed: `0`

### CMake Full Build

- Result: success
- warning lines observed: `123`
- warning-bearing areas:
  - `tests`: `98`
  - `benchmarks`: `13`
  - `src`: `11`
  - `examples`: `1`

Warning classes:

- `-Wmissing-field-initializers`: `72`
- `-Wdouble-promotion`: `45`
- `-Wunused-function`: `3`
- `-Wimplicit-function-declaration`: `2`
- `-Wswitch`: `1`

Highest-warning files:

- `tests/test_ldlt.c`: `18`
- `tests/test_sprint20_integration.c`: `9`
- `tests/test_colamd.c`: `8`
- `tests/test_chol_csc.c`: `8`
- `tests/test_reorder_nd.c`: `7`
- `tests/test_svd.c`: `6`
- `tests/test_sprint6_integration.c`: `6`
- `tests/test_sprint18_integration.c`: `6`
- `src/sparse_svd.c`: `5`
- `benchmarks/bench_main.c`: `5`
- `src/sparse_qr.c`: `4`

### Makefile Default Build Path

- Result: success
- target invoked: `all`
- stderr size: `0 B`
- warnings observed: `0`

## Important Caveat

The CMake baseline is the useful full-tree warning baseline for Sprint 30 Day 1.

The Makefile baseline is **not** directly comparable in scope because `make all` builds only the library target, while the CMake full build also compiles tests, benchmarks, and examples. The zero-warning Makefile result should therefore be treated as:

- a clean baseline for the Makefile default library-only path, not
- evidence that the repository as a whole is warning-clean under Make.

This scope mismatch should be reconciled explicitly in later Sprint 30 work and in Sprint 31 / Sprint 34 quality-gate planning.

## Day 1 Baseline Conclusion

The clean Day 1 baseline for Sprint 30 is:

- CMake configure: clean
- CMake full build: `123` warnings
- Makefile default `all` path: clean, but library-only in scope

The highest-signal Sprint 30 core-library warning cluster remains the `-Wdouble-promotion` sites in:

- `src/sparse_lu.c`
- `src/sparse_ldlt.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
