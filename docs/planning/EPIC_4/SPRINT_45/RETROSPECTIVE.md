# Sprint 45 Retrospective

**Sprint:** 45 — Iterative Solver Workspace Reuse & repeated-Solve Efficiency  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 45 baseline and iterative-workspace scope captured before implementation
- [x] iterative repeated-allocation seam inventory refreshed against live code
- [x] bounded reusable-workspace API design completed
- [x] shared buffer-layer design and validation plan completed
- [x] shared iterative workspace layer landed in live code
- [x] primary scalar CG / GMRES workspace migrations landed
- [x] post-primary workspace landing audit completed
- [x] block-CG workspace migration landed
- [x] wrapper compatibility cleanup for block GMRES / MINRES / BiCGSTAB landed
- [x] repeated-solve benchmark design completed
- [x] repeated-solve benchmark batch landed
- [x] internal workspace contract and residual audit completed
- [x] full validation sweep completed
- [x] Sprint 45 closeout and handoff completed from the measured baseline

## What Went Well

1. **Sprint 45 delivered a real internal reusable-workspace package, not just local allocation cleanup.**
   The sprint landed a shared internal workspace owner plus typed reusable
   views, then used that seam to migrate the main direct iterative targets:
   - scalar CG
   - matrix-free CG
   - scalar GMRES
   - matrix-free GMRES
   - block CG
   That is a meaningful structural handoff for later repeated-run work.

2. **The migration order was correct.** Sprint 45 did not start by reopening
   later or more specialized seams. It took the strongest, most reusable
   targets first:
   - shared workspace/storage layer
   - primary scalar CG/GMRES paths
   - block CG
   - wrapper/composition cleanup only after the direct workspace owners were in
     place
   That kept the sprint focused on genuine repeated-allocation reduction rather
   than diffusing into every iterative family at once.

3. **The compatibility boundary held.** Sprint 45 did not widen into:
   - public explicit iterative workspace APIs
   - public solver-usage contract churn
   - broad benchmark CLI redesign
   - broad README/tutorial refresh
   The public one-shot APIs remained compatibility wrappers over the new
   internal reusable-workspace seam, which was the right internal-first Epic 4
   move.

4. **The benchmark evidence was narrow and honest.** Day 11 added a dedicated
   repeated-solve benchmark instead of overloading existing convergence or CLI
   surfaces:
   - `benchmarks/bench_iterative_reuse.c`
   It measured repeated one-shot vs reusable-workspace-backed solves for:
   - scalar CG
   - scalar GMRES
   That gave Sprint 45 direct evidence without forcing exaggerated performance
   claims.

5. **The residual iterative queue is now much clearer.** By Day 12, the
   remaining work was classified instead of left implicit:
   - wrapper/composition surfaces:
     - block GMRES
     - block MINRES
     - block BiCGSTAB
   - specialized later solver-local seams:
     - scalar MINRES
     - the separate BiCGSTAB workspace precedent
   - explicit non-goals:
     - eigensolver workspace reuse
     - public explicit workspace APIs
     - broad benchmark CLI and tutorial refresh

6. **The sprint closed from a measured maintained baseline.** Day 13 validated
   both the normal code-change floor and the strongest local reviewed path:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   It also revalidated the touched iterative, benchmark, and example surfaces.

## What Didn't Go Well

1. **The measured runtime effect was modest and timing-sensitive.** The Day 11
   and Day 13 benchmark runs showed stable behavior but only small timing
   differences, and the sign could move across reruns. That is not a failure,
   but it means Sprint 45’s performance story is necessarily restrained.

2. **Scalar MINRES still remains outside the shared workspace owner.**
   Sprint 45 correctly stopped before broadening again, but `sparse_solve_minres(...)`
   still owns its own packed local allocation path. That remains one of the
   clearest later iterative reuse seams.

3. **BiCGSTAB remains separate rather than unified.** This is intentional, but
   Sprint 45 leaves a split internal model:
   - shared iterative workspace owner for CG / GMRES / block CG
   - separate `bicgstab_workspace_t` precedent for BiCGSTAB
   That is acceptable at this phase, but later repeated-run work may want to
   revisit the long-term relationship between those models.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 45 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |

### Sprint 45 artifact package

| Metric | Sprint 45 close state |
|---|---:|
| total artifact files under `SPRINT_45/artifacts/` | `16` |
| implementation-focused artifacts (Days 5, 6, 8, 9, 11) | `5` |
| validation / closeout artifacts (Days 13-14) | `2` |

### Iterative workspace and repeated-solve outputs

| Metric | Sprint 45 close state |
|---|---:|
| new internal iterative implementation modules/headers added | `3` |
| maintained build surfaces updated for Sprint 45 code landings | `2` |
| direct migrated reusable-workspace solver paths | `5` |
| normalized block wrapper/composition surfaces | `3` |
| direct iterative / benchmark / example reruns in Day 13 | `7` |

Notes:

- new internal iterative implementation modules/headers:
  - `src/sparse_iterative_workspace_internal.c`
  - `src/sparse_iterative_workspace_internal.h`
  - `src/sparse_iterative_internal.h`
- maintained build surfaces:
  - `Makefile`
  - `CMakeLists.txt`
- direct migrated reusable-workspace solver paths:
  - scalar CG
  - matrix-free CG
  - scalar GMRES
  - matrix-free GMRES
  - block CG
- normalized block wrapper/composition surfaces:
  - block GMRES
  - block MINRES
  - block BiCGSTAB

## Residual Deferred Debt

Sprint 45 was explicitly about internal iterative workspace reuse and bounded
repeated-solve evidence. The main open work it intentionally hands forward is:

- scalar MINRES workspace migration / unification with the shared owner
- later unification or evolution of the separate BiCGSTAB workspace precedent
- eigensolver repeated-run workspace reuse
- any future public explicit iterative workspace API only when a later sprint
  chooses that outward-facing scope directly
- broader benchmark CLI modernization
- README/tutorial/public repeated-solve guidance refresh

Not carried forward as unresolved Sprint 45 debt:

- missing shared internal workspace owner
- missing primary scalar CG/GMRES workspace migration
- missing block-CG workspace migration
- missing wrapper/composition cleanup for the remaining block convenience
  surfaces
- missing repeated-solve benchmark evidence
- missing measured validation closeout

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-reusable-workspace-api-design.md](./artifacts/day3-reusable-workspace-api-design.md)
- [day4-shared-buffer-layer-design-and-validation-plan.md](./artifacts/day4-shared-buffer-layer-design-and-validation-plan.md)
- [day5-shared-iterative-buffer-layer-batch1.md](./artifacts/day5-shared-iterative-buffer-layer-batch1.md)
- [day6-cg-gmres-migration-batch1.md](./artifacts/day6-cg-gmres-migration-batch1.md)
- [day8-block-iterative-migration-batch.md](./artifacts/day8-block-iterative-migration-batch.md)
- [day9-wrapper-compatibility-batch.md](./artifacts/day9-wrapper-compatibility-batch.md)
- [day11-repeated-solve-benchmark-batch.md](./artifacts/day11-repeated-solve-benchmark-batch.md)
- [day12-workspace-contract-and-residual-audit.md](./artifacts/day12-workspace-contract-and-residual-audit.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 45 achieved its goal:

- Epic 4 now has a real internal reusable-workspace layer for iterative
  repeated solves
- the main scalar CG/GMRES repeated-allocation targets are migrated
- block CG now uses the shared workspace seam too
- the remaining block convenience surfaces are cleaner and more obviously
  compatibility-oriented
- the sprint now has direct repeated-solve benchmark evidence
- the remaining iterative queue is narrower and more explicit
- the sprint closed from a measured maintained validation baseline

Later iterative and eigensolver repeated-run work can now start from an
explicit reusable-workspace model and direct benchmark evidence instead of
reopening whether that internal seam exists at all.
