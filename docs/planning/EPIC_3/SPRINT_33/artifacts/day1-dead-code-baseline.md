# Sprint 33 Day 1 Dead-Code Baseline

**Date:** 2026-05-18  
**Branch:** `sprint-33`

## Objective

Turn the Sprint 32 closeout into a concrete Sprint 33 starting inventory by confirming the inherited clean-state guarantees, auditing the current build and analysis entry points, generating a dedicated compilation-database path for later `xunused` work, and naming the highest-value low-risk surfaces for the first dead-code pass.

## Baseline Summary

Sprint 33 starts from the Sprint 32 closeout exactly as intended:

- full-tree warnings: `0`
- dormant-scaffold debt: `0`
- active `ctest` registry: `53`
- active opt-in test policy already in force:
  - `RUN_TEST_SLOW(...)`
  - `SPARSE_TEST_SLOW=1`
  - `RUN_TEST_EXPERIMENTAL(...)`
  - `SPARSE_TEST_EXPERIMENTAL=1`

Current branch head during the Day 1 baseline capture:

- `ff3cfe6`

This means Sprint 33 does not inherit residual warning cleanup or truthfulness cleanup. Its work starts directly on dead-code policy, tooling, and the first conservative removal pass.

## Current Tooling State

What exists today:

- `make lint` depends on `tooling-build`
- `make lint` runs:
  - strict `src/*.c` compile under `-Werror`
  - `clang-tidy` on `src/*.c`
  - `cppcheck` on `src/` and `tests/`
- `ctest -N --test-dir build/sprint32-day1-cmake` still reports `53` registered tests

What does **not** exist yet:

- `make deadcode`
- `make deadcode-report`
- `make deadcode-check`
- any Makefile path that generates `compile_commands.json`

Local tool availability at Day 1:

- `cppcheck`: present
- `clang-tidy`: present
- `xunused`: missing from `PATH`

Interpretation:

- Sprint 33 already has useful static-analysis precedent in `make lint`
- but the dead-code workflow requested by `PROJECT_PLAN.md` is still entirely unimplemented
- local `xunused` availability is an explicit prerequisite/documentation gap, not a hidden assumption

## Compilation Database Baseline

Day 1 generated a dedicated CMake configure tree:

- `build/sprint33-day1-cmake`
- `build/sprint33-day1-cmake/compile_commands.json`

Compilation-database size:

- total translation units: `97`

Coverage split by area:

- `src`: `25`
- `tests`: `53`
- `benchmarks`: `13`
- `examples`: `6`

This is enough to prove that CMake can supply the `compile_commands.json` input Sprint 33 wants. It also shows that the compilation database is narrower than the repo’s full Makefile tooling surface.

## Scope Counts

Repository `.c` file counts:

- `src`: `25`
- `tests`: `54`
- `benchmarks`: `14`
- `examples`: `12`

Important comparison:

- full repo bench sources: `14`
- bench entries in `compile_commands.json`: `13`
- full repo example sources: `12`
- example entries in `compile_commands.json`: `6`

## Coverage Gaps To Carry Forward

Benchmark source absent from the Day 1 compilation database:

- `bench_svd`

Example sources absent from the Day 1 compilation database:

- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

Day 1 conclusion from that mismatch:

- a raw `xunused build/compile_commands.json` run would not see every benchmark/example translation unit the Makefile currently builds
- Sprint 33 should either document that limitation explicitly or improve the compilation-database coverage before treating the report as a complete bench/example dead-code signal

## Likely First-Pass Candidate Areas

These are Day 1 audit candidates only, not confirmed dead-code findings:

### 1. `tests/`

Why it is attractive:

- largest internal non-library surface by file count
- already governed by the Sprint 32 truthfulness model
- generally low public API risk

Constraint:

- `tests/test_framework_optin.c` remains live coverage and must not be treated as dormant scaffold

### 2. `benchmarks/` and `examples/`

Why they matter:

- natural place for stale fixture builders, legacy one-off helpers, and experiment residue
- lower public-contract risk than installed headers

Constraint:

- current compilation-database coverage is incomplete here, so raw `xunused` results will under-describe this surface unless Sprint 33 addresses the gap

### 3. private helper layer in `src/`

Representative current internal-header surfaces:

- `src/sparse_analysis_internal.h`
- `src/sparse_bicgstab_internal.h`
- `src/sparse_chol_csc_internal.h`
- `src/sparse_colamd_internal.h`
- `src/sparse_eigs_internal.h`
- `src/sparse_errno_internal.h`
- `src/sparse_graph_internal.h`
- `src/sparse_ldlt_csc_internal.h`
- `src/sparse_matrix_internal.h`
- `src/sparse_reorder_amd_qg_internal.h`
- `src/sparse_reorder_nd_internal.h`
- `src/sparse_svd_internal.h`

Constraint:

- Sprint 33 should not delete private helper code until the report policy and false-positive handling rules are defined

### 4. public headers and documented surface

Day 1 rule:

- treat this as a separate review queue, not as part of the first “definitely-unused internal code” removal pass

## Day 1 Conclusion

Sprint 33 begins from a clean technical baseline but an incomplete dead-code-tooling baseline:

- the inherited quality state is strong and explicit
- the repo has no dead-code targets yet
- CMake can generate the needed `compile_commands.json`
- `xunused` is a real local prerequisite gap
- and the initial compilation-database coverage is narrower than the Makefile bench/example surface

That makes the correct next step clear: define the policy and limitations before deletion work begins, then implement dead-code targets in a way that makes the reporting scope and coverage boundaries explicit.
