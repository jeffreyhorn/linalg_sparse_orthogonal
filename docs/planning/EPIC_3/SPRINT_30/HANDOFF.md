# Sprint 30 Handoff

**Source sprint:** 30  
**Prepared on:** Day 14  
**Purpose:** Convert Sprint 30’s baseline and triage work into concrete follow-up inputs for Sprint 31 and later Epic 3 cleanup.

## Starting State For Sprint 31

Authoritative Apple Clang CMake full-tree warning state at Sprint 30 close:

- full-tree warnings: `112`
- `src`: `0`
- `tests`: `98`
- `benchmarks`: `13`
- `examples`: `1`

By warning class:

- `-Wmissing-field-initializers`: `72`
- `-Wdouble-promotion`: `34`
- `-Wunused-function`: `3`
- `-Wimplicit-function-declaration`: `2`
- `-Wswitch`: `1`

Important constraints:

- `src/` warnings are not acceptable
- new warnings in any area are not acceptable
- Apple Clang CMake remains the authoritative full-tree inventory path
- Makefile `all` remains a narrower library-only cross-check

## Sprint 31 First-Fix Queue

### Priority A: benchmark tool correctness and portability

Files:

- `benchmarks/bench_main.c`
- `benchmarks/bench_convergence.c`

Target problems:

- stale reorder CLI and usage text in `bench_main.c`
- missing `COLAMD` and `ND` enum coverage in `reorder_name()`
- CLI parsing still rejecting `colamd` and `nd`
- `_POSIX_C_SOURCE` / `snprintf` portability debt in benchmark entry points

Expected outcome:

- benchmark tooling matches the current public reorder API
- benchmark portability warnings around implicit declaration are gone

### Priority B: designated-initializer cleanup in public-facing tooling

Files:

- `benchmarks/bench_colamd.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`
- `examples/example_colamd.c`

Target problems:

- positional options-struct initialization against structs that have grown trailing fields
- public examples still teaching brittle initialization style

Expected outcome:

- benchmark/example initializer warnings reduced
- public-facing code demonstrates designated-initializer usage consistently

### Priority C: low-signal benchmark follow-through

File:

- `benchmarks/bench_convergence.c`

Target problem:

- residual `-Wdouble-promotion` mechanical cleanup

## Sprint 32 First-Fix Queue

File:

- `tests/test_reorder_nd.c`

Target problems:

- dormant or non-executed test scaffolding
- `-Wunused-function` warning debt
- designated-initializer drift in the same file

Expected outcome:

- test-honesty improvement, not just warning-count reduction

## Later Auxiliary Cleanup Queue

### High-volume test initializer files

- `tests/test_ldlt.c`
- `tests/test_chol_csc.c`
- `tests/test_colamd.c`
- `tests/test_cholesky.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_sprint20_integration.c`
- `tests/test_reorder.c`

### Residual mechanical double-promotion files

- `tests/test_sprint6_integration.c`
- `tests/test_svd.c`
- `tests/test_bidiag.c`
- `benchmarks/bench_convergence.c`
- smaller one- and two-warning files already enumerated in Day 10 and Day 11 artifacts

## Reproduction Commands

Use these commands before and after cleanup work:

1. `make warning-workflow WARNING_WORKFLOW_LABEL=<label>`
2. `make format`
3. `make lint`
4. `make test`

Expected stable comparison target at Sprint 30 close:

- CMake warnings: `112`
- Makefile warnings: `0`

## Key References

- [COMPILE_HYGIENE_PLAYBOOK.md](./COMPILE_HYGIENE_PLAYBOOK.md)
- [REBUILD_WORKFLOW.md](./REBUILD_WORKFLOW.md)
- [day10-test-warning-triage.md](./artifacts/day10-test-warning-triage.md)
- [day11-tooling-warning-triage.md](./artifacts/day11-tooling-warning-triage.md)
- [day12-baseline-reconciliation.md](./artifacts/day12-baseline-reconciliation.md)
- [day13-validation-pass.md](./artifacts/day13-validation-pass.md)
