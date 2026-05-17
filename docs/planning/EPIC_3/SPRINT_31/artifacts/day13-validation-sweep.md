# Sprint 31 Day 13: Validation Sweep & Residual Mechanical Cleanup

## Goal

Re-run the Sprint 31 validation paths, close the last benchmark/example
mechanical warning if it still exists, and record the final warning
delta against the Sprint 30 baseline.

## Residual Fix

The fresh serialized CMake build confirmed that one benchmark warning
remained after Days 1-12:

- `benchmarks/bench_convergence.c:43`
- `implicit conversion increases floating-point precision`
- `[-Wdouble-promotion]`

Root cause:

- `compute_rel_residual()` returns `double`
- the allocation-failure path returned `NAN`
- on this compiler that macro expands as a `float` constant and trips
  `-Wdouble-promotion`

Day 13 fix:

- changed `return NAN;`
- to `return nan("");`

No behavior change was intended beyond removing the warning.

## Validation

### Compile-only tooling gate

- `make tooling-build`
- result: passed

### Authoritative serialized CMake path

Pre-fix reproduction:

- total warnings: `99`
- benchmark/example warnings: `1`

Post-fix reproduction:

- total warnings: `98`
- benchmark/example warnings: `0`

The post-fix serialized CMake capture contains no warning lines outside
`tests/`.

### Standard local validation flow

- `make format`
- `make lint`
- `make test`

All three passed.

Notes:

- `make lint` stderr is the normal `clang-tidy` progress/suppression
  summary, not a failure
- `make test` completed successfully across the full test-binary set

## Final Sprint 31 Warning Delta

Relative to the Sprint 30 closing baseline:

- full-tree warnings: `112 -> 98`
- benchmark/example warnings: `14 -> 0`

That means Sprint 31 removed all warning debt from the benchmark/example
queue it explicitly targeted, while leaving the remaining warning debt
in tests only.

## Deferred Queue After Day 13

Remaining warnings in the serialized CMake capture are test-only:

- `62` `-Wmissing-field-initializers`
- `33` `-Wdouble-promotion`
- `3` `-Wunused-function`

Largest remaining test files:

- `test_ldlt.c` `18`
- `test_sprint20_integration.c` `9`
- `test_colamd.c` `8`
- `test_chol_csc.c` `8`
- `test_reorder_nd.c` `7`

## End State

Sprint 31 now closes with:

- benchmark CLI/help behavior synced to the current reorder contract
- benchmark/example portability cleanup landed
- benchmark/example designated-initializer cleanup landed
- compile-only tooling gate wired into `make lint`
- benchmark/example docs updated
- benchmark/example warning queue reduced to zero
