# Sprint 46 Retrospective

**Sprint:** 46 — Eigensolver Workspace Reuse & Advanced Repeated-Run Efficiency  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 46 baseline and eigensolver-workspace scope captured before implementation
- [x] eigensolver repeated-allocation seam inventory refreshed against live code
- [x] bounded reusable eigensolver workspace/state design completed
- [x] shared buffer-layer design and validation plan completed
- [x] shared eigensolver workspace layer landed in live code
- [x] grow-m Lanczos workspace migration landed
- [x] thick-restart Lanczos workspace migration landed
- [x] post-primary workspace landing audit completed
- [x] LOBPCG workspace migration landed
- [x] compatibility-wrapper cleanup for the one-shot public eigensolver entry landed
- [x] repeated-run benchmark design completed
- [x] repeated-run benchmark batch landed
- [x] internal workspace contract and residual audit completed
- [x] full validation sweep completed
- [x] Sprint 46 closeout and handoff completed from the measured baseline

## What Went Well

1. **Sprint 46 delivered a real internal reusable-workspace package for the eigensolver family.**
   The sprint landed a shared internal workspace/state owner plus typed reusable
   views, then used that seam to migrate the three main eigensolver families it
   set out to cover:
   - grow-m Lanczos
   - thick-restart Lanczos
   - LOBPCG
   That is a meaningful structural handoff for later advanced repeated-run
   work.

2. **The migration order was correct.** Sprint 46 did not start by reopening
   public surfaces or late benchmark churn. It took the strongest repeated-run
   seams first:
   - shared workspace/storage layer
   - grow-m Lanczos
   - thick-restart Lanczos
   - LOBPCG
   - compatibility-wrapper cleanup only after the direct workspace owners were
     in place
   That kept the sprint focused on genuine repeated-allocation reduction rather
   than diffusing into every eigensolver-adjacent seam at once.

3. **The internal-first compatibility boundary held.** Sprint 46 did not widen
   into:
   - public explicit eigensolver workspace APIs
   - public result-buffer contract churn
   - broad benchmark CLI redesign
   - broad README/tutorial refresh
   The public one-shot API remained a compatibility wrapper over the new
   internal reusable-workspace seam, which was the right internal-first Epic 4
   move.

4. **The benchmark evidence was narrow and honest.** Day 11 added a dedicated
   repeated-run benchmark instead of overloading existing eigensolver benchmark
   flows:
   - `benchmarks/bench_eigs_reuse.c`
   It measured repeated one-shot vs reusable-workspace-backed solves for:
   - grow-m Lanczos
   - thick-restart Lanczos
   That gave Sprint 46 direct evidence without forcing exaggerated performance
   claims.

5. **The example and direct test surfaces remained aligned with the new seam.**
   By Day 13, the touched eigensolver surfaces all reran cleanly:
   - `test_eigs`
   - `test_eigs_thick_restart`
   - `test_eigs_lobpcg`
   - `example_eigs`
   - `bench_eigs_reuse`
   That matters because Sprint 46 touched not just storage/layout internals,
   but also the repeated-run composition path and benchmark evidence surface.

6. **The residual eigensolver queue is now much clearer.** By Day 12 and Day
   14, the remaining work was classified instead of left implicit:
   - family-local helper/state scratch:
     - refinement scratch
     - dense Jacobi scratch
     - arrowhead/tridiagonal helper scratch
     - `lanczos_restart_state_t` internal restart state
   - outward-facing later work:
     - public explicit workspace APIs
     - broader benchmark CLI redesign
     - broader public repeated-run docs/tutorial refresh
     - corpus-wide repeated-run benchmark expansion

7. **The sprint closed from a measured maintained baseline.** Day 13 validated
   both the normal code-change floor and the strongest local reviewed path:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   It also revalidated the touched eigensolver, example, and benchmark
   surfaces directly.

## What Didn't Go Well

1. **The measured runtime effect was modest and timing-sensitive.** The Day 11
   and Day 13 benchmark runs showed exact behavior parity but only small timing
   differences, and the direction could move across reruns. That is not a
   failure, but it means Sprint 46’s performance story is necessarily
   restrained.

2. **Some family-local helper/state scratch remains outside the shared owner.**
   Sprint 46 correctly stopped before broadening again, but some solver-local
   scratch remains intentionally separate:
   - refinement scratch
   - dense Jacobi scratch
   - arrowhead/tridiagonal helper scratch
   - `lanczos_restart_state_t`
   That is acceptable at this phase, but it leaves a later specialization
   queue.

3. **The benchmark evidence is not broad enough for universal claims.**
   Sprint 46 proved that the repeated-run seam exists and behaves correctly,
   but only across a bounded benchmark pair. That means later work still owns
   any broader corpus-wide or CLI-level repeated-run evidence expansion.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 46 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |

### Sprint 46 artifact package

| Metric | Sprint 46 close state |
|---|---:|
| total artifact files under `SPRINT_46/artifacts/` | `15` |
| implementation-focused artifacts (Days 5, 6, 8, 9, 11) | `5` |
| validation / closeout artifacts (Days 13-14) | `2` |

### Eigensolver workspace and repeated-run outputs

| Metric | Sprint 46 close state |
|---|---:|
| new internal eigensolver implementation modules/headers added | `2` |
| maintained build surfaces updated for Sprint 46 code landings | `2` |
| direct migrated reusable-workspace eigensolver families | `3` |
| targeted direct reruns in Day 13 | `5` |
| repeated-run benchmarked backend cases | `2` |

Notes:

- new internal eigensolver implementation modules/headers:
  - `src/sparse_eigs_workspace_internal.c`
  - `src/sparse_eigs_workspace_internal.h`
- maintained build surfaces:
  - `Makefile`
  - `CMakeLists.txt`
- direct migrated reusable-workspace eigensolver families:
  - grow-m Lanczos
  - thick-restart Lanczos
  - LOBPCG
- repeated-run benchmarked backend cases:
  - grow-m Lanczos
  - thick-restart Lanczos

## Residual Deferred Debt

Sprint 46 was explicitly about internal eigensolver workspace reuse and bounded
repeated-run evidence. The main open work it intentionally hands forward is:

- family-local helper/state cleanup outside the main shared owner:
  - refinement scratch
  - dense Jacobi scratch
  - arrowhead/tridiagonal helper scratch
  - `lanczos_restart_state_t`
- any future public explicit eigensolver workspace API only when a later sprint
  chooses that outward-facing scope directly
- broader benchmark CLI modernization
- README/tutorial/public repeated-run guidance refresh
- corpus-wide repeated-run benchmark expansion

Not carried forward as unresolved Sprint 46 debt:

- missing shared internal eigensolver workspace owner
- missing grow-m Lanczos workspace migration
- missing thick-restart Lanczos workspace migration
- missing LOBPCG workspace migration
- missing one-shot compatibility-wrapper cleanup
- missing repeated-run benchmark evidence
- missing measured validation closeout

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-reusable-eigensolver-workspace-state-design.md](./artifacts/day3-reusable-eigensolver-workspace-state-design.md)
- [day4-shared-buffer-layer-design-and-validation-plan.md](./artifacts/day4-shared-buffer-layer-design-and-validation-plan.md)
- [day5-shared-eigensolver-buffer-layer-batch1.md](./artifacts/day5-shared-eigensolver-buffer-layer-batch1.md)
- [day6-lanczos-migration-batch1.md](./artifacts/day6-lanczos-migration-batch1.md)
- [day8-lobpcg-workspace-migration-batch.md](./artifacts/day8-lobpcg-workspace-migration-batch.md)
- [day9-compatibility-wrapper-batch.md](./artifacts/day9-compatibility-wrapper-batch.md)
- [day11-repeated-run-benchmark-batch.md](./artifacts/day11-repeated-run-benchmark-batch.md)
- [day12-workspace-contract-memory-behavior-and-residual-audit.md](./artifacts/day12-workspace-contract-memory-behavior-and-residual-audit.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 46 achieved its goal:

- Epic 4 now has a real internal reusable-workspace layer for eigensolver
  repeated runs
- the main grow-m, thick-restart, and LOBPCG repeated-allocation targets are
  migrated
- the public one-shot entry remains compatibility-preserving
- the sprint now has direct repeated-run benchmark evidence
- the remaining eigensolver queue is narrower and more explicit
- the sprint closed from a measured maintained validation baseline

Later advanced repeated-run and any future outward-facing eigensolver workspace
work can now start from an explicit reusable-workspace model and direct
benchmark evidence instead of reopening whether that internal seam exists at
all.
