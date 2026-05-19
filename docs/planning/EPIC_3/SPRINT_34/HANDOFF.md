# Sprint 34 Handoff

**Source sprint:** 34  
**Prepared on:** Day 14  
**Purpose:** Convert Sprint 34's phase-1 build-quality enforcement work into
concrete starting constraints for Sprint 35 and later Epic 3 hardening work.

## Starting State For Sprint 35

Sprint 34 does **not** hand off warning debt, dead-code cleanup debt, or a
broken reviewed-quality flow.

Authoritative validated quality state at Sprint 34 close:

- `make format`: passed
- `make lint`: passed
- `make test`: passed
- `make quality-review-compile`: passed
- `make quality-review`: passed
- `make quality-review-cmake-compile`: passed
- `make quality-review-cmake`: passed
- `ctest -N --test-dir build/quality-review-cmake`: `53` registered tests
- `ctest --test-dir build/quality-review-cmake --output-on-failure`: `53 / 53`
  passed
- `make deadcode-report`: passed
- `make deadcode-check`: passed

Validated timings captured on Day 13:

- `make lint`: `244.14 s`
- `make test`: `62.34 s`
- `make quality-review-compile`: `314.65 s`
- `make quality-review`: `438.34 s`
- `make quality-review-cmake-compile`: `63.14 s`
- `make quality-review-cmake`: `239.97 s`
- full `ctest` real time: `181.55 s`
- `make deadcode-report`: `0.39 s`
- `make deadcode-check`: `0.56 s`

## Build-Quality Contract Now In Force

Sprint 34 delivered these maintainer-facing reviewed-quality targets:

- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`

Current reviewed-path interpretation:

- `quality-review-compile` is the reviewed local compile-quality path
  - `format-check`
  - `lint`
- `quality-review` is the reviewed local end-to-end quality path
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`
- `quality-review-cmake-compile` is the reviewed CMake parity path without full
  suite execution
  - configure `build/quality-review-cmake`
  - serialized clean rebuild
  - `ctest -N`
- `quality-review-cmake` extends that path with full `ctest`

Execution-model contract now in force:

- reviewed wrapper targets are intentionally serial
- `.NOTPARALLEL` protects:
  - `quality-review-compile`
  - `quality-review`
  - `quality-review-cmake-compile`
  - `quality-review-cmake`
  - `deadcode`
  - `deadcode-report`
  - `deadcode-check`
- `check` remains unchanged; Sprint 34 added reviewed wrappers instead of
  silently repurposing older shortcuts

## CI Contract Now In Force

Sprint 34 delivered the first CI enforcement pass in Linux CI:

- Linux `lint` job runs `make quality-review-compile`
- Linux `cmake-build-and-test` job runs `make quality-review-cmake`
- Linux `deadcode` job runs:
  - `make deadcode-report`
  - `make deadcode-check`

Current phase-1 CI boundaries:

- macOS and Windows workflows were intentionally left unchanged
- dead-code execution remains one serialized job, not a matrix
- dead-code artifacts are uploaded for operator review

## Regression-Prevention Result From Sprint 34

Sprint 34 also closed one regression-prevention queue that surfaced during
enforcement work:

- reviewed non-zero positional `sparse_analysis_opts_t` initializers were
  converted to designated initializers across the reviewed target set

Interpretation:

- this was not inherited Sprint 32 backlog reopening
- it was phase-1 warning-gate hygiene to keep reviewed targets aligned with the
  designated-initializer standard before stronger future enforcement

## Sprint 32 / Sprint 33 Invariants Still Preserved

Later Epic 3 work should preserve all of these:

- no warning debt reintroduced into the reviewed local/CMake quality paths
- `53` registered CTest tests remain the auditable active-suite count until
  intentionally changed
- `tests/test_framework_optin.c` remains live coverage for:
  - `SKIP_TEST`
  - `RUN_TEST_SLOW`
  - `RUN_TEST_EXPERIMENTAL`
- `tests/test_reorder_nd.c` remains free of commented-out dormant registrations
- `deadcode-check` still enforces report completeness, not zero findings

## Residual Deferred Queue

Sprint 34 hands off a **phase-2 enforcement** queue, not a cleanup/breakage
queue.

### Priority A: cross-platform CI parity

Sprint 34 enforced reviewed-quality paths in Linux CI only.

Deferred follow-through:

- macOS reviewed-quality parity
- Windows/MSVC reviewed-quality parity
- platform-specific expectations for:
  - reviewed Makefile path
  - reviewed CMake path
  - dead-code/tooling availability

### Priority B: dead-code compile-db coverage gap

The current dead-code compilation database still excludes:

- benchmark:
  - `bench_svd`
- examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Implication:

- stronger future dead-code enforcement should either bring those paths under
  compile-db coverage or preserve the exclusion list explicitly in reporting and
  CI expectations

### Priority C: dead-code execution isolation

The dead-code workflow still shares:

- `build/deadcode-cmake`
- `build/deadcode/`

Implication:

- the current reviewed path is safe when serialized
- later CI/local expansion should isolate those paths automatically before
  encouraging concurrent sibling invocation

## Suggested First-Fix Queue For Sprint 35+

Sprint 35 should start from documentation and public-surface consistency, not
from reworking the Sprint 34 enforcement contract.

Immediate later-sprint follow-through belongs here instead:

- Sprint 36:
  - macOS/Windows reviewed-quality parity
  - cross-platform CI expectation alignment
  - script/target portability for reviewed and dead-code flows
- Sprint 38:
  - broaden compile-only/dead-code coverage to the current excluded benchmark
    and examples
  - mature the dead-code workflow into a safer regression signal by addressing
    shared-path isolation and residual advisory buckets

## Reproduction Commands

Use these commands before and after later Epic 3 quality work:

1. `make format`
2. `make lint`
3. `make test`
4. `make quality-review-compile`
5. `make quality-review`
6. `make quality-review-cmake-compile`
7. `make quality-review-cmake`
8. `make deadcode-report`
9. `make deadcode-check`

Expected stable comparison targets at Sprint 34 close:

- `53` registered CTest tests
- full `ctest`: `53 / 53` passing
- Linux reviewed CI paths map directly to:
  - `make quality-review-compile`
  - `make quality-review-cmake`
  - `make deadcode-report`
  - `make deadcode-check`
- dead-code compile-db exclusion list: unchanged unless intentionally broadened

## Key References

- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day5-makefile-enforcement-batch1.md](./artifacts/day5-makefile-enforcement-batch1.md)
- [day6-makefile-enforcement-batch2.md](./artifacts/day6-makefile-enforcement-batch2.md)
- [day8-cmake-parity-implementation.md](./artifacts/day8-cmake-parity-implementation.md)
- [day10-ci-enforcement-implementation.md](./artifacts/day10-ci-enforcement-implementation.md)
- [day11-initializer-regression-audit.md](./artifacts/day11-initializer-regression-audit.md)
- [day12-failure-ux-and-operator-docs.md](./artifacts/day12-failure-ux-and-operator-docs.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
