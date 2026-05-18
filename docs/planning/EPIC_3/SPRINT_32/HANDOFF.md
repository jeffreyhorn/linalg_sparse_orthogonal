# Sprint 32 Handoff

**Source sprint:** 32  
**Prepared on:** Day 14  
**Purpose:** Convert Sprint 32's test-truthfulness and warning-cleanup work into concrete starting constraints for Sprint 33 and later Epic 3 quality work.

## Starting State For Sprint 33

Authoritative Apple Clang serialized CMake full-tree warning state at
Sprint 32 close:

- full-tree warnings: `0`
- `src`: `0`
- `tests`: `0`
- `benchmarks`: `0`
- `examples`: `0`

Sprint 32 closure against the Day 1 baseline:

- `-Wmissing-field-initializers`: `62 -> 0`
- `-Wdouble-promotion`: `33 -> 0`
- `-Wunused-function`: `3 -> 0`

Important constraints now in force:

- the authoritative full-tree warning inventory path remains:
  - `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first`
- benchmark/example warnings were already closed in Sprint 31 and remain at `0`
- test warnings are now also closed and should stay at `0`
- `make lint` still includes the Sprint 31 compile-only tooling gate
- `make test` and `ctest` are both required validation paths for later Epic 3 work

Important test-truthfulness contracts now in force:

- `tests/test_reorder_nd.c`
  - active `RUN_TEST(...)` calls: `23`
  - commented-out `RUN_TEST(...)` calls: `0`
- active default coverage belongs in `RUN_TEST(...)`
- slow opt-in coverage belongs behind:
  - `RUN_TEST_SLOW(...)`
  - `SPARSE_TEST_SLOW=1`
- experimental opt-in coverage belongs behind:
  - `RUN_TEST_EXPERIMENTAL(...)`
  - `SPARSE_TEST_EXPERIMENTAL=1`
- historical or retired target evidence belongs in `docs/planning/`, not as compiled commented-out test scaffold

Validation state at Sprint 32 close:

- `make format`: passed
- `make lint`: passed
- `make test`: passed
- `ctest -N --test-dir build/sprint32-day1-cmake`: `53` registered tests
- `ctest --test-dir build/sprint32-day1-cmake --output-on-failure`: `53 / 53` passed

## Sprint 33 First-Fix Queue

Sprint 32 does **not** hand off residual warning debt or dormant test scaffold.

That means Sprint 33 should start directly on its planned dead-code work:

### Priority A: dead-code policy and tooling

Work items:

- define the repository dead-code policy
- implement `make deadcode`
- implement the reporting companion target

Constraints from Sprint 32:

- do not treat opt-in wrappers (`RUN_TEST_SLOW`, `RUN_TEST_EXPERIMENTAL`) as dormant scaffold
- do not reclassify documented historical evidence as active test code
- keep `tests/test_framework_optin.c` live so the opt-in framework remains tested

### Priority B: first cleanup pass

Scope guidance:

- prioritize definitely-unused internal code
- start with:
  - `tests/`
  - `benchmarks/`
  - `examples/`
  - private helpers in `src/`
- keep candidate-unused public API separate from definitely-dead internal code

### Priority C: preserve the cleaned quality baseline

Later Epic 3 work should preserve these Sprint 32 closeout invariants:

- full-tree warnings stay at `0`
- test truthfulness stays explicit
- active suite size remains auditable from `ctest -N`
- warning/dead-code tasks should be treated as regression prevention, not inherited Sprint 32 cleanup

## No Carried Warning Or Truthfulness Debt

Remaining Sprint 32 carried-forward queue:

- warning debt: none
- dormant scaffold debt: none
- truthfulness-policy debt: none

If later sprints discover new warning or dead-code findings, they should be treated as new regressions or new cleanup scope rather than as unresolved Sprint 32 backlog.

## Reproduction Commands

Use these commands before and after later Epic 3 quality work:

1. `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first`
2. `make format`
3. `make lint`
4. `make test`
5. `ctest -N --test-dir build/sprint32-day1-cmake`
6. `ctest --test-dir build/sprint32-day1-cmake --output-on-failure`

Expected stable comparison target at Sprint 32 close:

- CMake warnings: `0`
- `make lint`: passing
- `make test`: passing
- CTest registered tests: `53`
- full CTest run: `53 / 53` passing

## Key References

- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day6-coverage-honesty-docs.md](./artifacts/day6-coverage-honesty-docs.md)
- [day10-double-promotion-batch-design.md](./artifacts/day10-double-promotion-batch-design.md)
- [day11-double-promotion-batch1.md](./artifacts/day11-double-promotion-batch1.md)
- [day12-double-promotion-batch2-and-reconciliation.md](./artifacts/day12-double-promotion-batch2-and-reconciliation.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
