# Sprint 37 Day 1 Auxiliary Maintainability Baseline

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Turn the Sprint 36 closeout state into a concrete Sprint 37 starting inventory
by confirming the inherited validated-quality contract, auditing the repo's
largest non-core maintenance surfaces, and naming the first helper/target/file
cleanup targets before implementation begins.

## Baseline Summary

Sprint 37 starts from the Sprint 36 close exactly as intended:

- no inherited reviewed-quality regression queue
- no inherited cross-platform contract ambiguity
- maintained direct gates already validated at close:
  - `make format`
  - `make lint`
  - `make test`
- maintained reviewed wrapper paths already validated at close:
  - `make quality-review-compile`
  - `make quality-review`
  - `make quality-review-cmake-compile`
  - `make quality-review-cmake`
- maintained support paths already validated at close:
  - `make wall-check`
  - `make deadcode-report`
  - `make deadcode-check`
  - `make sanitize`
- active reviewed CMake suite baseline remains:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Current branch head during the Day 1 baseline capture:

- `58f296f`

This means Sprint 37 is not a warning-debt sprint or a parity-definition
sprint. It is a maintainability sprint layered on top of a validated quality
contract.

## Current Auxiliary Maintainability Surface

### Large-file hotspots

The highest-maintenance non-core files are concentrated in tests, workflow
plumbing, and benchmark harnesses:

#### Tests

- `tests/test_chol_csc.c` = `4,643` lines
- `tests/test_svd.c` = `3,712`
- `tests/test_ldlt_csc.c` = `3,637`
- `tests/test_qr.c` = `3,259`
- `tests/test_etree.c` = `2,890`
- `tests/test_iterative.c` = `2,819`

#### Benchmarks

- `benchmarks/bench_eigs.c` = `958`
- `benchmarks/bench_main.c` = `774`
- `benchmarks/bench_ldlt_csc.c` = `579`
- `benchmarks/bench_chol_csc.c` = `446`

#### Quality/workflow support

- `Makefile` = `812`
- `scripts/deadcode_report.py` = `472`
- `.github/workflows/ci.yml` = `231`
- `scripts/epic3_warning_workflow.sh` = `215`
- `scripts/deadcode_workflow.sh` = `189`
- `scripts/wall_check.sh` = `162`

Interpretation:

- the large-file pass should stay in auxiliary/support code, not numerical
  kernels
- the most likely one-or-two-file cleanup targets are large tests plus quality
  target/script plumbing

### Size distribution by major non-core surface

- tests:
  - `54` `.c` files
  - `62,005` total lines
- benchmarks:
  - `14` `.c` files
  - `5,170` total lines
- workflow/helper support:
  - `Makefile` + `scripts/*.sh` + `scripts/*.py` + workflow YAML =
    `2,464` lines

Interpretation:

- test maintainability dominates the auxiliary line count
- benchmark/helper cleanup is smaller in absolute size but likely cheaper to
  consolidate usefully
- quality-target normalization is structurally important even though it is a
  smaller line-count surface than the test tree

## Current Quality-Target Surface

Maintained named targets already in play:

- direct/build surfaces:
  - `all`
  - `examples-build`
  - `examples`
  - `smoke`
  - `test`
  - `bench`
  - `bench-build`
  - `tooling-build`
  - `bench-fast`
- sanitizer/openmp/platform-support surfaces:
  - `sanitize`
  - `asan`
  - `sanitize-all`
  - `omp`
  - `tsan`
  - `sanitize-thread`
- formatting/lint/reviewed quality surfaces:
  - `format`
  - `format-check`
  - `lint`
  - `check`
  - `quality-review-compile`
  - `quality-review`
  - `quality-review-cmake-compile`
  - `quality-review-cmake`
- reporting/dead-code/coverage surfaces:
  - `warning-workflow`
  - `deadcode-compile-db`
  - `deadcode`
  - `deadcode-report`
  - `deadcode-check`
  - `wall-check`
  - `coverage`
  - `coverage-lcov`
  - `coverage-gcovr`

Interpretation:

- the repo already has the needed quality entry points
- Sprint 37's target work is about ownership and normalization, not feature
  addition
- the current target surface is large enough that clearer layering will reduce
  future maintenance cost

## Inherited Constraints From Sprint 36

### 1. Sanitizer/build-tree interaction is still a live maintainer caveat

Sprint 36 handed off a real operational constraint:

- a prior `make sanitize` run can leave an instrumented `build/` tree behind
- later direct or reviewed sweeps may then fail unless the tree is cleaned
  first

Implication:

- Sprint 37 should treat this as a target-layout and maintainer-doc problem,
  not just a retrospective footnote

### 2. Cross-platform quality wording is already correct and must not be blurred

The Sprint 36 platform contract now in force is:

- Linux:
  - enforced reviewed Makefile path
  - enforced reviewed CMake path
  - enforced dead-code path
- macOS:
  - enforced Apple Clang reviewed path
  - staged dead-code path
  - supplemental Homebrew GCC leg
- Windows:
  - enforced reviewed CMake subset
  - staged local Makefile reviewed-wrapper parity
  - excluded dead-code path

Implication:

- Sprint 37 docs and target cleanup must preserve this contract explicitly
- target normalization should avoid fake symmetry or collapsed naming that
  hides staged vs enforced status

### 3. Dead-code still has real structural limits

Still open from earlier work:

- compile-db exclusion list:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`
- shared execution paths:
  - `build/deadcode-cmake`
  - `build/deadcode/`

Implication:

- Sprint 37 can improve maintainability around dead-code tooling
- but it should not overclaim universality or fully solved gate maturity

## First Audit Targets

### Test-helper consolidation

Day 1 already shows likely repeated helper families across tests:

- SPD/tridiagonal builders
- KKT builders
- identity/diagonal builders
- Jacobi/precondition callback helpers
- residual and norm helpers

Most likely Day 2 concentration surface:

- very large test files plus repeated integration helpers

### Benchmark-helper consolidation

Day 1 already shows likely repeated benchmark-side helper families:

- synthetic matrix builders
- timing/reorder/result formatting helpers
- benchmark CLI/utility helpers

Most likely Day 3 concentration surface:

- `bench_eigs.c`
- `bench_main.c`
- `bench_chol_csc.c`
- `bench_ldlt_csc.c`

### Quality-target normalization

Primary Day 4+ surface:

- `Makefile`
- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`
- supporting maintainer docs

Main operational goal:

- make direct/reviewed/sanitizer/dead-code relationships easier to understand
  and safer to rerun

## Day 1 Conclusion

Sprint 37 starts from a strong validated quality baseline and a clear
maintainability problem statement:

- tests dominate the auxiliary maintenance footprint
- benchmarks have a smaller but likely cheaper shared-helper opportunity set
- `Makefile` and dead-code tooling now carry enough quality behavior that
  ownership and naming clarity matter materially
- the Sprint 36 sanitizer caveat should be treated as an implementation input,
  not just documentation residue

The Day 1 inventory supports a bounded Sprint 37 implementation sequence:

- Day 2:
  - test-helper audit
- Day 3:
  - benchmark-helper audit
- Day 4:
  - quality-target normalization design
