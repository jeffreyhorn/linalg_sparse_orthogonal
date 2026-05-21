# Sprint 37 Handoff

**Source sprint:** 37  
**Prepared on:** Day 14  
**Purpose:** Convert Sprint 37's auxiliary-code cleanup and maintainability
refactor work into explicit starting constraints for Sprint 38, Sprint 39, and
later Epic 3 quality-gate expansion work.

## Starting State For Sprint 38

Sprint 37 does **not** hand off a broken validation path, a new auxiliary-code
warning queue, or a partially-landed helper framework that still needs rescue
work.

Authoritative validated close state at Sprint 37 close:

- `make format`: passed
- `make lint`: passed
- `make test`: passed
- `make quality-review-compile`: passed
- `make quality-review`: passed
- `make quality-review-cmake-compile`: passed
- `make quality-review-cmake`: passed
- `ctest -N --test-dir build/quality-review-cmake`: `53` registered tests
- full reviewed CMake `ctest`: `53 / 53` passed

Validated timings captured on Day 13:

- `make lint`: `235.65 s`
- `make test`: `111.30 s`
- `make quality-review-compile`: `256.69 s`
- `make quality-review`: `313.09 s`
- `make quality-review-cmake-compile`: `47.31 s`
- `make quality-review-cmake`: `210.24 s`
- full reviewed CMake `ctest` real time: `156.66 s`

## Maintainability Contract Now In Force

Sprint 37 did not create a new general helper framework. It established a
lower-duplication, narrower-ownership auxiliary-code model.

### Test-helper ownership

The maintained shared test-helper surface remains intentionally small:

- global shared policy layer:
  - `tests/test_framework.h`
- narrow residual-helper cluster layer:
  - `tests/test_solver_helpers.h`

Still **not** the intended direction:

- broad shared test `.c` helper libraries
- opaque helper frameworks that hide test intent
- indiscriminate extraction out of large feature-owner test files

### Benchmark-helper ownership

The maintained shared benchmark-helper surface also remains intentionally
narrow:

- backend-comparison pair layer:
  - `benchmarks/bench_backend_compare_helpers.h`

Still **not** the intended direction:

- broad `bench_common.c`-style runtime layers
- flattening benchmark-owner behavior into one generic benchmark framework

### Quality-target ownership

Sprint 37 clarified the Makefile surface into three categories:

- maintained operator entry points
- helper/prerequisite plumbing
- tree-mutating instrumentation or alternate-build modes

That ownership model should be preserved when later sprints expand gates or
reporting.

## Highest-Value Shipped Sprint 37 Results

Sprint 37 closed the main maintainability pain points from its Day 1 audit:

- repeated iterative/preconditioner/integration residual helpers were reduced
  through `tests/test_solver_helpers.h`
- repeated Cholesky/LDLT backend-comparison support logic was reduced through
  `bench_backend_compare_helpers.h`
- `Makefile` target ownership and reset guidance are easier to read without
  changing the actual operator contract
- `deadcode_report.py` now renders through smaller named section helpers
  instead of one long mixed report block
- workflow/dead-code/README maintainer docs now describe the current contract
  with less sprint-history-first duplication

## Sprint 34-Sprint 36 Baselines Still Preserved

Later Epic 3 work should preserve all of these:

- Sprint 34 reviewed Makefile wrappers still define the maintained local
  quality contract
- Sprint 34 reviewed CMake parity wrappers still define the maintained CMake
  parity contract
- Sprint 35 public-doc ownership split remains in force:
  - headers = authoritative API contract
  - `README.md` = concise entrypoint
  - `docs/tutorial.md` = fuller teaching surface
- Sprint 36 cross-platform enforced/staged/supplemental contract remains in
  force
- active CTest registry remains `53` until intentionally changed
- `tests/test_framework_optin.c` remains live opt-in/skip policy coverage

## Residual Deferred Queue

Sprint 37 closes without a new auxiliary cleanup backlog.

Not carried forward as new Sprint 37 debt:

- broken helper consolidation: none
- broken reviewed local quality path: none
- broken reviewed CMake parity path: none
- new mandatory large-file refactor queue: none
- stale maintainer-workflow wording after Day 11 cleanup: none

Still carried forward from earlier/later-sprint work:

- dead-code remains authoritative only in serial mode because of shared-path
  execution
- tree-mutating instrumentation modes still require `make clean` before
  returning to the normal direct/reviewed path
- Windows local Makefile reviewed-wrapper parity remains staged
- macOS dead-code remains staged
- Windows dead-code remains excluded
- dead-code compile-db exclusion list remains open:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

## Suggested First-Fix Queue For Sprint 38+

Sprint 38 should start from quality-gate expansion and regression-proofing, not
from reopening Sprint 37 implementation work.

Immediate later-sprint emphasis belongs here instead:

- Sprint 38:
  - use the lower-duplication Sprint 37 target/script surface to make gate
    expansion safer
  - address dead-code compile-db gap closure and shared-path isolation
  - preserve the `make clean` return-from-instrumentation rule until a safer
    build-tree model exists
- Sprint 39:
  - keep the Sprint 37 helper/ownership model in the final standards/docs
    closeout
  - avoid reintroducing broad shared helper frameworks during final cleanup

## Reproduction Commands

Use these commands before and after later Epic 3 gate/readiness work:

1. `make format`
2. `make lint`
3. `make test`
4. `make quality-review-compile`
5. `make quality-review`
6. `make quality-review-cmake-compile`
7. `make quality-review-cmake`

If a sanitizer or other tree-mutating mode ran immediately before the direct
or reviewed sweep:

8. `make clean`

Expected stable comparison targets at Sprint 37 close:

- `53` registered CTest tests
- full reviewed CMake `ctest`: `53 / 53` passing
- reviewed local wrapper path: green
- no new helper-refactor fallout

## Key References

- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day5-test-helper-consolidation-batch1.md](./artifacts/day5-test-helper-consolidation-batch1.md)
- [day6-benchmark-helper-consolidation-batch1.md](./artifacts/day6-benchmark-helper-consolidation-batch1.md)
- [day7-quality-target-normalization-batch1.md](./artifacts/day7-quality-target-normalization-batch1.md)
- [day9-large-file-maintainability-batch1.md](./artifacts/day9-large-file-maintainability-batch1.md)
- [day11-maintainer-workflow-docs-batch.md](./artifacts/day11-maintainer-workflow-docs-batch.md)
- [day12-focused-validation-and-reconciliation.md](./artifacts/day12-focused-validation-and-reconciliation.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
