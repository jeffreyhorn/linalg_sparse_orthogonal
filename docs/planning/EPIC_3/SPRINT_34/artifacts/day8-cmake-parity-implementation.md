# Sprint 34 Day 8: CMake parity implementation

## Goal

Implement the reviewed CMake parity path designed on Day 7 so Sprint 34 has a named, maintained build-tree workflow for clean rebuild + `ctest -N` + full `ctest`.

## Shipped changes

### Makefile

Added the reviewed CMake wrapper layer:

- `QUALITY_REVIEW_CMAKE_DIR ?= build/quality-review-cmake`
- `quality-review-cmake-compile`
- `quality-review-cmake`

Behavior:

- `quality-review-cmake-compile`
  - configure a dedicated reviewed build tree
  - perform a clean serialized rebuild
  - run `ctest -N` on that tree
- `quality-review-cmake`
  - run `quality-review-cmake-compile`
  - then run full `ctest`

The new targets were also added to `.NOTPARALLEL` so the reviewed CMake path stays serial and bannered under `make -j`, matching the Sprint 34 reviewed-wrapper style.

### README

Documented the new operator-facing commands in two places:

- main "With Make" command list
- "Reviewed Local Quality Path" section

The README now states explicitly that the CMake wrappers provide reviewed parity for:

- clean rebuild
- `ctest -N`
- full `ctest`

It also states explicitly that they do not replace:

- `make format-check`
- `make lint` static-analysis phases
- `make deadcode-check`

## Validation

### Dry-run shape

- `make -n quality-review-cmake-compile`
  - configure
  - clean build
  - `ctest -N`
- `make -n quality-review-cmake`
  - recursive compile wrapper
  - full `ctest`

### Live reviewed parity path

- `make quality-review-cmake-compile`: passed
  - configured `build/quality-review-cmake`
  - completed a clean serialized rebuild
  - `ctest -N` reported `53` registered tests
- `make quality-review-cmake`: passed
  - reran the reviewed compile path
  - full `ctest` passed:
    - `53 / 53` tests passed
    - total reported real time: `236.99 sec`

## Resulting parity contract

### Named reviewed Makefile path

- `make quality-review-compile`
- `make quality-review`

### Named reviewed CMake path

- `make quality-review-cmake-compile`
- `make quality-review-cmake`

### Preserved truthful limitation

The dead-code compile-db gap is unchanged and still must not be treated as closed:

- missing benchmark:
  - `bench_svd`
- missing examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

## Conclusion

Sprint 34 now has maintained reviewed-target paths on both sides of the local build split:

- Makefile-reviewed quality flow
- CMake-reviewed rebuild + suite-parity flow

That is enough for phase-1 parity without overstating dead-code coverage or pretending the CMake path replaces formatter/static-analysis/dead-code enforcement.
