# Sprint 37 Retrospective

**Sprint:** 37 — Auxiliary-Code Cleanup & Maintainability Refactor  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] auxiliary maintainability baseline captured before refactor work
- [x] test-helper duplication audited before extraction
- [x] benchmark-helper duplication audited before extraction
- [x] quality-target ownership model defined before Makefile normalization
- [x] first narrow test-helper consolidation batch landed
- [x] first narrow benchmark-helper consolidation batch landed
- [x] quality-target normalization batch landed without changing operator behavior
- [x] large-file maintainability audit completed before structural split
- [x] large-file maintainability batch landed on chosen Tier 1 surfaces
- [x] comment/maintainer-doc audit completed before wording cleanup
- [x] maintainer workflow docs batch landed
- [x] focused validation/reconciliation passed
- [x] final direct/wrapper/CMake validation passed
- [x] Sprint 38+ handoff inputs written

## What Went Well

1. **The sprint stayed disciplined about narrow shared layers.** The Day 2 and
   Day 3 audits were correct: the right maintainability move was a few small
   shared headers, not a new helper framework. That kept ownership clear and
   avoided obscuring test/benchmark intent.

2. **The maintainability work stayed behavior-preserving.** The Day 12 focused
   validation and Day 13 full sweep confirmed that the helper extractions,
   report-render cleanup, and workflow-doc compression did not change the
   validated quality contract.

3. **The sprint attacked the right large files.** Day 8 correctly rejected the
   instinct to refactor the biggest feature-owner tests just because they were
   large. Targeting `Makefile` and `deadcode_report.py` on Day 9 produced a
   better maintenance payoff with lower behavioral risk.

4. **The quality-target normalization clarified operator expectations.** The
   Day 7 ownership model made the Makefile surface easier to scan and gave the
   Sprint 36 sanitizer/build-tree caveat a durable place in the operator
   contract.

5. **The docs cleanup improved operator usability without moving ownership.**
   Day 11 reduced repeated sprint-history framing while keeping the README,
   workflow files, and dead-code tools aligned to the same maintained contract.

## What Didn't Go Well

1. **The dead-code path is still operationally awkward.** Sprint 37 preserved
   the known serial-only constraint correctly, but it remains a real workflow
   limitation until later shared-path work lands.

2. **Full wrapper validation is still expensive.** Day 13 reconfirmed that the
   reviewed wrapper paths and reviewed CMake path are valuable, but they are
   not lightweight. Future gate expansion work still needs careful attention to
   failure attribution and operator ergonomics.

3. **The sprint could only normalize the sanitizer caveat, not eliminate it.**
   Day 12/13 proved that `make clean` is the right return-from-instrumentation
   rule, but the underlying tree-mutating build model remains a real
   maintainer concern.

4. **Several large feature-owner files remain intentionally untouched.** That
   was the right choice for Sprint 37, but it means later work still needs the
   same discipline about not turning "large file" into "mandatory refactor" by
   default.

## Final Metrics

### Direct maintained gates

| Metric | Day 13 final |
|---|---:|
| `make format` wall time | `2.74 s` |
| `make lint` wall time | `235.65 s` |
| `make test` wall time | `111.30 s` |

### Reviewed wrapper paths

| Metric | Day 13 final |
|---|---:|
| `make quality-review-compile` wall time | `256.69 s` |
| `make quality-review` wall time | `313.09 s` |
| `make quality-review-cmake-compile` wall time | `47.31 s` |
| `make quality-review-cmake` wall time | `210.24 s` |
| full reviewed CMake `ctest` real time | `156.66 s` |

### Focused reconciliation checks

| Metric | Day 12 final |
|---|---:|
| Day 5 helper-cluster direct reruns | passed |
| Day 6 benchmark-pair direct reruns | passed |
| `make sanitize` | passed |
| `make clean` reset after sanitizer | confirmed required |
| serial `make deadcode-report && make deadcode-check` | passed |
| `ctest -N` registered tests | `53` |
| Makefile/CMake test-count parity | `53` vs `53` |

## Residual Deferred Debt

Sprint 37 closes without a new cleanup backlog, but it preserves several known
later-sprint operational constraints.

Carried forward:

- dead-code shared-path serial-only execution remains open
- tree-mutating instrumentation mode reset remains an operator caveat
- Windows local Makefile reviewed-wrapper parity remains staged
- macOS dead-code remains staged
- Windows dead-code remains excluded
- dead-code compile-db exclusion list remains open

Not carried forward as residual Sprint 37 debt:

- broken test-helper extraction: none
- broken benchmark-helper extraction: none
- broken reviewed local wrapper path: none
- broken reviewed CMake parity path: none
- mandatory new large-file refactor queue: none

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [HANDOFF.md](./HANDOFF.md)
- [day5-test-helper-consolidation-batch1.md](./artifacts/day5-test-helper-consolidation-batch1.md)
- [day6-benchmark-helper-consolidation-batch1.md](./artifacts/day6-benchmark-helper-consolidation-batch1.md)
- [day7-quality-target-normalization-batch1.md](./artifacts/day7-quality-target-normalization-batch1.md)
- [day9-large-file-maintainability-batch1.md](./artifacts/day9-large-file-maintainability-batch1.md)
- [day11-maintainer-workflow-docs-batch.md](./artifacts/day11-maintainer-workflow-docs-batch.md)
- [day12-focused-validation-and-reconciliation.md](./artifacts/day12-focused-validation-and-reconciliation.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)

## Bottom Line

Sprint 37 achieved its goal:

- auxiliary code is less duplicated and easier to maintain
- helper ownership is narrower and clearer
- the Makefile/workflow/operator surfaces are easier to understand
- the validated quality contract from Sprints 34-36 remained intact

The sprint did **not** create a new backlog class. Sprint 38 and Sprint 39
should treat Sprint 37 as a stable maintainability baseline and continue with
gate expansion, dead-code maturation, and final Epic 3 closeout from there.
