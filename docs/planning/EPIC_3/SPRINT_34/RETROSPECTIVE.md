# Sprint 34 Retrospective

**Sprint:** 34 — Build-Quality Enforcement Phase 1  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 33 handoff constraints converted into an explicit Sprint 34
  enforcement baseline
- [x] reviewed Makefile quality wrappers implemented
- [x] reviewed CMake parity wrappers implemented
- [x] Linux CI phase-1 enforcement wired to the reviewed local commands
- [x] dead-code reporting/check flow preserved inside the reviewed quality path
- [x] operator-facing failure output and command map improved
- [x] regression-prevention initializer drift closed in the reviewed target set
- [x] final validation passed from direct, reviewed-wrapper, and CMake parity
  paths
- [x] Sprint 35+ handoff inputs written

## What Went Well

1. **The sprint preserved command meaning while still adding real enforcement.**
   `lint`, `test`, `check`, and `deadcode-check` kept their original roles, and
   the new reviewed wrappers layered on top instead of breaking existing habits.

2. **The reviewed paths are attributable and operator-readable.** Recursive
   wrapper targets plus explicit phase banners made the new enforcement flow
   understandable locally and in CI without hiding behavior behind opaque
   prerequisite fan-out.

3. **The CMake parity story became concrete.** Sprint 34 stopped treating CMake
   validation as an informal side channel and turned it into named reviewed
   commands that prove both clean rebuild parity and active-suite parity.

4. **Linux CI now maps directly to maintained local commands.** That sharply
   reduced ambiguity between “what contributors run locally” and “what CI
   enforces first.”

5. **Phase-1 regression prevention stayed disciplined.** The initializer cleanup
   batch was a real reviewed-target regression-prevention pass, not a reopening
   of Sprint 32 backlog for its own sake.

## What Didn't Go Well

1. **The CI enforcement story is still Linux-first.** Sprint 34 made the
   reviewed paths real in CI, but macOS and Windows still lag behind that
   contract.

2. **Dead-code coverage is still not full tooling coverage.** The compile-db
   gap from Sprint 33 remained intact through Sprint 34, so stronger future
   dead-code enforcement still has to account for `bench_svd` plus six examples
   explicitly.

3. **Dead-code concurrency remains a controlled limitation.** `.NOTPARALLEL`
   protects the reviewed wrapper path, but the underlying shared
   `build/deadcode-cmake` and `build/deadcode/` paths are still not a clean
   foundation for broader concurrent invocation.

4. **The reviewed quality paths are not cheap.** They are now reliable and
   explicit, but `quality-review` and `quality-review-cmake` are expensive
   enough that future expansion work should stay deliberate about cost and job
   placement.

## Final Metrics

### Direct quality gates

| Metric | Day 13 final |
|---|---:|
| `make lint` wall time | `244.14 s` |
| `make test` wall time | `62.34 s` |

### Reviewed wrapper paths

| Metric | Day 13 final |
|---|---:|
| `make quality-review-compile` wall time | `314.65 s` |
| `make quality-review` wall time | `438.34 s` |
| `make quality-review-cmake-compile` wall time | `63.14 s` |
| `make quality-review-cmake` wall time | `239.97 s` |
| full `ctest` real time inside reviewed CMake path | `181.55 s` |

### Suite and dead-code state

| Metric | Day 13 final |
|---|---:|
| `ctest -N` registered tests | `53` |
| full `ctest` result | `53 / 53` passed |
| `make deadcode-report` wall time | `0.39 s` |
| `make deadcode-check` wall time | `0.56 s` |

## Residual Deferred Debt

Deferred Sprint 34 queue at close:

- Linux-first CI enforcement only
- dead-code compile-db coverage gap:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`
- dead-code shared-path isolation for safer future concurrent invocation

Not carried forward as residual debt:

- reviewed local quality path breakage: none
- reviewed CMake parity breakage: none
- Linux reviewed CI-command ambiguity: none
- warning debt in the reviewed target set: none
- dormant test-scaffold debt: none
- definitely-unused internal dead-code queue: none

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [HANDOFF.md](./HANDOFF.md)
- [day3-warning-gate-design.md](./artifacts/day3-warning-gate-design.md)
- [day5-makefile-enforcement-batch1.md](./artifacts/day5-makefile-enforcement-batch1.md)
- [day6-makefile-enforcement-batch2.md](./artifacts/day6-makefile-enforcement-batch2.md)
- [day8-cmake-parity-implementation.md](./artifacts/day8-cmake-parity-implementation.md)
- [day10-ci-enforcement-implementation.md](./artifacts/day10-ci-enforcement-implementation.md)
- [day11-initializer-regression-audit.md](./artifacts/day11-initializer-regression-audit.md)
- [day12-failure-ux-and-operator-docs.md](./artifacts/day12-failure-ux-and-operator-docs.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)

## Bottom Line

Sprint 34 achieved its engineering goal:

- the repository now has named reviewed-quality wrapper paths instead of an
  implied mix of unrelated local commands
- CMake parity is explicit and auditable
- Linux CI enforces the first reviewed compile/test/dead-code pass using those
  maintained commands
- operator-facing failure output is much clearer than the pre-sprint state

Sprint 35 should treat this enforcement layer as a stable baseline, while
Sprint 36 and Sprint 38 take the remaining cross-platform and dead-code
expansion work forward deliberately rather than pretending phase 1 already
closed those gaps.
