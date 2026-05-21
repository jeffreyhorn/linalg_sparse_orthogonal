# Sprint 36 Retrospective

**Sprint:** 36 — Cross-Platform Quality Parity  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 34 reviewed-quality baseline treated as the non-regression floor
- [x] macOS reviewed-path gap audited and aligned
- [x] Windows/MSVC reviewed subset audited and made explicit
- [x] enforced/staged/supplemental parity vocabulary fixed before broad CI edits
- [x] macOS and Windows workflow expectation wording aligned to the reviewed contract
- [x] reviewed Makefile portability batch landed without regressing the baseline
- [x] compact parity report written
- [x] final consistency pass reconciled workflows and README wording
- [x] platform-focused validation passed
- [x] final direct/wrapper/CMake validation passed
- [x] Sprint 37+ handoff inputs written

## What Went Well

1. **The sprint stayed focused on contract truthfulness instead of fake
   symmetry.** The most important Day 4 decision was not to force identical
   commands everywhere. Treating Linux, macOS, and Windows as a mix of
   enforced, staged, supplemental, and excluded surfaces kept the final
   contract honest.

2. **The reviewed CMake path proved to be the right anchor.** Day 7 through
   Day 13 confirmed that reviewed CMake parity is the strongest honest
   cross-platform baseline. That kept Windows work grounded instead of pushing
   premature Makefile parity claims.

3. **The CI wording work materially improved maintainability.** Sprint 36 did
   not just change comments; it made workflow step names, README language, and
   the parity report describe the same platform model. That lowers future
   operator confusion substantially.

4. **The portability batch stayed usefully narrow.** Removing avoidable `find`
   and hardcoded `/bin/*` assumptions from the maintained reviewed path
   improved portability without reopening Unix-only maintainer helpers that
   were intentionally out of scope.

5. **The sprint preserved inherited baselines cleanly.** The Sprint 34
   reviewed-wrapper contract, Sprint 35 public-doc ownership split, the `53`
   test CTest baseline, and the Sprint 32 truthfulness/opt-in test coverage all
   survived unchanged.

## What Didn't Go Well

1. **The repo still has several real staged limits that Sprint 36 could only
   document, not eliminate.** Windows local Makefile reviewed-wrapper parity,
   macOS dead-code parity, and Windows dead-code parity are still later-phase
   work.

2. **Dead-code remains operationally awkward across platforms.** Sprint 36 was
   correct not to overclaim it, but the compile-db exclusion list and the
   shared-path serialized execution model are still real friction points.

3. **Full wrapper validation is expensive.** Day 13 reconfirmed that the
   reviewed wrappers are useful, but their cost is high enough that operator
   guidance and good failure attribution remain important.

4. **The sanitizer/build-tree interaction was only discovered during the final
   direct sweep.** The stale UBSan archive from Day 12 was easy to fix with
   `make clean`, but it is still a workflow caveat that needs to be remembered
   unless later maintainability work makes the build-tree interaction safer.

## Final Metrics

### Direct maintained gates

| Metric | Day 13 final |
|---|---:|
| `make format` wall time | `3.31 s` |
| `make lint` wall time | `303.32 s` |
| `make test` wall time | `264.17 s` |

### Reviewed wrapper paths

| Metric | Day 13 final |
|---|---:|
| `make quality-review-compile` wall time | `696.45 s` |
| `make quality-review` wall time | `487.59 s` |
| `make quality-review-cmake-compile` wall time | `93.11 s` |
| `make quality-review-cmake` wall time | `817.17 s` |
| full reviewed CMake `ctest` real time | `703.03 s` |

### Supporting platform-focused checks

| Metric | Day 12 final |
|---|---:|
| `make wall-check` | passed |
| `make deadcode-report` | passed |
| `make deadcode-check` | passed |
| `make sanitize` | passed |
| workflow YAML parse check | `3 / 3` files passed |

### Suite state

| Metric | Final |
|---|---:|
| `ctest -N` registered tests | `53` |
| full reviewed CMake `ctest` result | `53 / 53` passed |
| Makefile/CMake test-count parity | `53` vs `53` |

## Residual Deferred Debt

Sprint 36 closes without a regression queue, but it does leave a bounded
staged-parity queue.

Carried forward:

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
- dead-code shared-path isolation remains open:
  - `build/deadcode-cmake`
  - `build/deadcode/`
- sanitizer/direct-sweep build-tree cleanup remains an operational maintainer
  caveat

Not carried forward as residual debt:

- broken reviewed local quality path: none
- broken reviewed CMake parity path: none
- stale cross-platform CI vocabulary: none
- broken Sprint 34/Sprint 35 baseline contract: none

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [HANDOFF.md](./HANDOFF.md)
- [day4-cross-platform-parity-design.md](./artifacts/day4-cross-platform-parity-design.md)
- [day5-macos-workflow-alignment.md](./artifacts/day5-macos-workflow-alignment.md)
- [day6-windows-workflow-alignment.md](./artifacts/day6-windows-workflow-alignment.md)
- [day8-portability-batch1.md](./artifacts/day8-portability-batch1.md)
- [day10-cross-platform-parity-report.md](./artifacts/day10-cross-platform-parity-report.md)
- [day11-final-parity-consistency-pass.md](./artifacts/day11-final-parity-consistency-pass.md)
- [day12-platform-focused-validation.md](./artifacts/day12-platform-focused-validation.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)

## Bottom Line

Sprint 36 achieved its goal:

- the repo now has an operationally truthful cross-platform quality contract
  instead of a Linux-implied one
- macOS and Windows workflow expectations are explicit rather than inferred
- reviewed CMake parity is clearly established as the strongest shared reviewed
  baseline
- the Sprint 34 reviewed-wrapper and Sprint 35 public-doc baselines stayed
  intact

Sprint 37 and Sprint 38 should treat Sprint 36 as a stable parity baseline and
carry forward only the named staged limits, not reopen a solved contract-audit
problem.
