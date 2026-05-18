# Sprint 32 Retrospective

**Sprint:** 32 — Test-Suite Truthfulness & Dormant Scaffold Resolution  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Test-tree warning baseline captured on the authoritative Apple Clang CMake path
- [x] `tests/test_reorder_nd.c` dormant scaffold audited before edits
- [x] Test-truthfulness policy defined
- [x] Test framework extended with skip and opt-in support
- [x] Historical dormant `test_reorder_nd.c` scaffold removed
- [x] Coverage-honesty documentation written
- [x] Test-side designated-initializer cleanup completed in Sprint 32 scope
- [x] Residual test-side `-Wdouble-promotion` queue closed
- [x] Final validation passed from both Makefile and CTest paths
- [x] Sprint 33+ handoff inputs written

## What Went Well

1. **The sprint stayed attached to the named Sprint 30 and Sprint 31 handoff queue.** Work started from explicit files and warning classes rather than a vague “clean up tests” goal, so every day had measurable closure criteria.

2. **The policy work happened before the deletions.** Defining the slow/experimental/historical split before touching `test_reorder_nd.c` kept the cleanup honest. The sprint deleted historical-only scaffold because that was the right contract, not just because the code was inconvenient.

3. **The framework support stayed minimal and useful.** `SKIP_TEST`, `RUN_TEST_SLOW`, and `RUN_TEST_EXPERIMENTAL` were enough to formalize the distinction without turning the test framework into a second planning system.

4. **The warning cleanup stayed mechanical.** The initializer and double-promotion queues were closed without algorithm churn, tolerance rewrites, or broader behavioral refactors.

5. **The final validation covered both real execution paths.** The sprint did not stop at `make test`; it also revalidated the `ctest` registry and full `ctest` execution path, which is important now that Sprint 32 made claims about the executed protection surface.

## What Didn't Go Well

1. **The full quality flow remains slow.** Sprint 32 closed the actual warning debt, but `make lint`, the serialized Apple Clang full-tree rebuild, and the longest CTest paths are still expensive enough that validation takes time and attention.

2. **The earlier handoff queue mixed truthful cleanup and warning cleanup in the same files.** That was manageable, but it meant Days 1-5 had to separate “what is honest to execute” from “what is noisy to compile” before implementation could proceed cleanly.

3. **A few older narrative comments elsewhere in the test tree still look historical.** Day 6 noted them, but they did not represent active dormant registration and were correctly left out of Sprint 32 scope. The sprint closed the actionable queue, not every stylistic trace of project history.

## Final Metrics

### Authoritative Apple Clang CMake full-tree path

| Metric | Day 1 | Day 13 final | Delta |
|---|---:|---:|---:|
| Full-tree warnings | 98 | 0 | -98 |
| `src` warnings | 0 | 0 | 0 |
| `tests` warnings | 98 | 0 | -98 |
| `benchmarks` warnings | 0 | 0 | 0 |
| `examples` warnings | 0 | 0 | 0 |

### Warning classes

| Warning class | Day 1 | Day 13 final | Delta |
|---|---:|---:|---:|
| `-Wmissing-field-initializers` | 62 | 0 | -62 |
| `-Wdouble-promotion` | 33 | 0 | -33 |
| `-Wunused-function` | 3 | 0 | -3 |

### Truthfulness / coverage metrics

| Metric | Day 1 | Day 13 final | Delta |
|---|---:|---:|---:|
| `tests/test_reorder_nd.c` active `RUN_TEST(...)` | 23 | 23 | 0 |
| `tests/test_reorder_nd.c` commented-out `RUN_TEST(...)` | 3 | 0 | -3 |
| `ctest -N` registered tests | not captured on Day 1 | 53 | n/a |

### Final validation

- `make format`: passed
- `make lint`: passed
- `make test`: passed
- `ctest -N --test-dir build/sprint32-day1-cmake`: `53` registered tests
- `ctest --test-dir build/sprint32-day1-cmake --output-on-failure`: `53 / 53` passed

## Residual Deferred Debt

Deferred warning debt at Sprint 32 close:

- none

Deferred dormant-scaffold debt at Sprint 32 close:

- none

Sprint 32 conclusion on carry-forward scope:

- later sprints should treat warning regressions, dead-code findings, or test-truthfulness regressions as new work
- there is no remaining named Sprint 32 cleanup queue to preserve as backlog

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [HANDOFF.md](./HANDOFF.md)
- [day3-test-truthfulness-policy.md](./artifacts/day3-test-truthfulness-policy.md)
- [day4-test-framework-support.md](./artifacts/day4-test-framework-support.md)
- [day5-test-reorder-nd-cleanup.md](./artifacts/day5-test-reorder-nd-cleanup.md)
- [day10-double-promotion-batch-design.md](./artifacts/day10-double-promotion-batch-design.md)
- [day11-double-promotion-batch1.md](./artifacts/day11-double-promotion-batch1.md)
- [day12-double-promotion-batch2-and-reconciliation.md](./artifacts/day12-double-promotion-batch2-and-reconciliation.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)

## Bottom Line

Sprint 32 achieved its engineering goal:

- the test suite now more honestly reflects what is active, opt-in, and historical
- the dormant `test_reorder_nd.c` scaffold is gone
- the test-tree warning backlog closed from `98` to `0`
- the opt-in framework behavior is live and tested
- and the cleaned state holds under both the Makefile and CTest validation paths

Sprint 33 can now start directly on dead-code detection infrastructure instead of inheriting unresolved warning or truthfulness debt.
