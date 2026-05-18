# Sprint 31 Handoff

**Source sprint:** 31  
**Prepared on:** Day 14  
**Purpose:** Convert Sprint 31's benchmark/tooling cleanup into concrete follow-up inputs for Sprint 32 and later Epic 3 work.

## Starting State For Sprint 32

Authoritative Apple Clang serialized CMake full-tree warning state at
Sprint 31 close:

- full-tree warnings: `98`
- `src`: `0`
- `tests`: `98`
- `benchmarks`: `0`
- `examples`: `0`

By warning class:

- `-Wmissing-field-initializers`: `62`
- `-Wdouble-promotion`: `33`
- `-Wunused-function`: `3`

Important constraints:

- `src/` warnings are not acceptable
- benchmark/example warnings are now closed and should stay at `0`
- Apple Clang serialized CMake remains the authoritative full-tree
  warning inventory path
- `make tooling-build` is now the compile-only benchmark/example gate
- `make lint` now includes that tooling gate automatically

Important benchmark/tooling behavior contracts now in force:

- `bench_main --reorder none|rcm|amd|nd`
- `bench_main` intentionally does **not** accept `colamd`
- `bench_reorder` remains the cross-ordering harness for:
  - `none`
  - `rcm`
  - `amd`
  - `colamd`
  - `nd`
- `bench_colamd` and `example_colamd` are the QR/COLAMD comparison tools

Validation state at Sprint 31 close:

- `make tooling-build`: passed
- `make format`: passed
- `make lint`: passed
- `make test`: passed

## Sprint 32 First-Fix Queue

### Priority A: test-suite truthfulness and dormant scaffold cleanup

File:

- `tests/test_reorder_nd.c`

Target problems:

- dormant or non-executed helper scaffold still present
- `-Wunused-function` debt
- companion initializer drift in the same file

Expected outcome:

- the file honestly reflects what is executed today
- unused-function debt is removed by deletion, wiring, or explicit
  quarantine rather than silence

### Priority B: high-volume test initializer cleanup

Highest-volume files:

- `tests/test_ldlt.c` `18`
- `tests/test_colamd.c` `8`
- `tests/test_chol_csc.c` `8`
- `tests/test_cholesky.c` `5`
- `tests/test_sprint12_integration.c` `5`
- `tests/test_reorder.c` `4`

Target problem:

- positional options-struct initialization against structs that have
  grown backend/progress trailing fields

Expected outcome:

- test warning volume drops without changing behavior
- the designated-initializer style used in Sprint 31 tooling is reused
  consistently in tests

### Priority C: residual mechanical double-promotion cleanup

Highest-volume files:

- `tests/test_sprint20_integration.c` `9`
- `tests/test_svd.c` `6`
- `tests/test_sprint6_integration.c` `6`
- `tests/test_sprint18_integration.c` `6`
- `tests/test_bidiag.c` `3`
- `tests/test_sprint19_integration.c` `3`

Smaller companion files:

- `tests/test_qr.c` `2`
- `tests/test_sprint10_integration.c` `2`
- `tests/test_block_solvers.c` `1`
- `tests/test_ilu.c` `1`
- `tests/test_lu_csr.c` `1`

Target problem:

- mechanical `float` to `double` promotion warnings in test constants and
  helper-return paths

Expected outcome:

- the residual warning queue is reduced without broad test refactoring

## Later Auxiliary Cleanup Queue

Smaller remaining initializer files:

- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_sprint20_integration.c`
- `tests/test_etree.c`

Single-warning mechanical files:

- `tests/test_block_solvers.c`
- `tests/test_ilu.c`
- `tests/test_lu_csr.c`
- `tests/test_sprint5_integration.c`

## Reproduction Commands

Use these commands before and after Sprint 32 cleanup work:

1. `rm -rf build/sprint32-cmake`
2. `cmake -S . -B build/sprint32-cmake`
3. `cmake --build build/sprint32-cmake --parallel 1 --clean-first`
4. `make tooling-build`
5. `make format`
6. `make lint`
7. `make test`

Expected stable comparison target at Sprint 31 close:

- CMake warnings: `98`
- benchmark/example warnings: `0`
- standard local validation flow: passing

## Key References

- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day10-compile-only-gate-design.md](./artifacts/day10-compile-only-gate-design.md)
- [day11-tooling-gate-implementation.md](./artifacts/day11-tooling-gate-implementation.md)
- [day12-documentation-refresh.md](./artifacts/day12-documentation-refresh.md)
- [day13-validation-sweep.md](./artifacts/day13-validation-sweep.md)
