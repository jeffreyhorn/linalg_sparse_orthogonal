# Sprint 120 Retrospective

**Sprint:** 120 - Direct/Iterative Oracle Architecture & Giant-Test Split
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 120 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read Epic 11 Sprint 120 scope and inherited Sprint 118/Sprint 119
      evidence patterns.
- [x] Audited direct oracle ownership for QR, LDLT, LDLT CSC, LU, and
      Cholesky proof owners.
- [x] Audited iterative oracle ownership for CG, GMRES, BiCGSTAB, MINRES,
      block solver, callback, and preconditioner proof owners.
- [x] Designed shared fixture rules while keeping tolerances, convergence
      criteria, failure modes, and skip policies solver-local.
- [x] Ranked direct and iterative split candidates by proof value, coupling
      risk, and rollback cost.
- [x] Split QR solve scenario tests from `tests/test_qr.c` into
      `tests/test_qr_solve.c`.
- [x] Registered the QR solve owner in Makefile and CMake.
- [x] Split block BiCGSTAB tests from `tests/test_bicgstab.c` into
      `tests/test_bicgstab_block.c`.
- [x] Registered the block BiCGSTAB owner in Makefile and CMake.
- [x] Designed and implemented a bounded cross-solver oracle pilot in
      `tests/test_cross_solver_oracle.c`.
- [x] Registered the cross-solver oracle pilot in Makefile and CMake.
- [x] Validated focused direct, iterative, and cross-solver pilot owners.
- [x] Validated source-list parity and reviewed CMake/CTest membership.
- [x] Ran required full quality gates for the branch's `.c` and build metadata
      changes: `make format`, `make lint`, and `make test`.
- [x] Published residual direct, iterative, and oracle cleanup queues.
- [x] Published explicit non-claims for direct parity, direct/iterative parity,
      external-oracle completeness, package/platform/ABI support, performance,
      and state-of-the-art positioning.
- [x] Published Sprint 121 handoff guidance for proof-owner movement,
      shared-helper discipline, CMake/CTest count evidence, and oracle
      expansion.
- [x] Finalized this retrospective and ran focused documentation hygiene.

## What Went Well

1. **The sprint selected split candidates with low ownership ambiguity.**
   QR solve scenarios and block BiCGSTAB behavior had clear proof-owner
   boundaries. That let the sprint reduce giant-test pressure without dragging
   LDLT CSC lifecycle, GMRES restart policy, or external-reference behavior
   into the first movement batch.

2. **Solver-specific interpretation stayed visible after the splits.**
   QR solve tolerances, reported-vs-true residual checks, generated-RHS
   semantics, block BiCGSTAB aggregation behavior, preconditioner failure
   propagation, and convergence expectations stayed in focused test owners
   instead of disappearing behind generic fixtures.

3. **Build registration stayed paired with new test ownership.**
   `tests/test_qr_solve.c`, `tests/test_bicgstab_block.c`, and
   `tests/test_cross_solver_oracle.c` were all registered in both Makefile and
   CMake. Day 13 then proved the reviewed CTest surface had 57 tests and that
   the new owners appeared in the expected adjacent positions.

4. **The cross-solver pilot stayed appropriately bounded.**
   The pilot compares LU, Cholesky, QR, and CG on one generated-RHS SPD
   fixture. That is useful as an oracle pattern without implying broad
   direct/iterative parity or external-oracle completeness.

5. **Validation was packaged instead of scattered.**
   Day 13 collected focused tests, source-list checks, CMake build and CTest
   membership, lint, and full tests into one validation package. That made the
   Day 14 closeout and this retrospective evidence-based rather than
   aspirational.

6. **Residuals were explicit.**
   LDLT, LDLT CSC, GMRES, MINRES, matrix-free BiCGSTAB, handle lifecycle,
   shared fixture, external oracle, and package/platform work were deferred
   with reasons and follow-up requirements.

## What Did Not Go Well

1. **The highest-risk direct owners still need future work.**
   LDLT and LDLT CSC remain dense with backend telemetry, inertia, external
   references, permutation lifecycle, in-place solve behavior, and singular
   detection. Sprint 120 correctly avoided overreaching there, but those files
   remain significant maintainability debt.

2. **The iterative cleanup is still partial.**
   Block BiCGSTAB moved cleanly, but GMRES restart/right-preconditioner
   behavior, MINRES LDLT/GMRES comparisons, matrix-free BiCGSTAB callbacks,
   and iterative handle lifecycle proof owners still need focused designs.

3. **Shared fixture extraction remains intentionally conservative.**
   The sprint produced a fixture architecture, but most helpers stayed local.
   That was the right call for claim hygiene, but it means some duplication is
   still present until repeated split pressure justifies broader helper
   promotion.

4. **Local validation cannot prove platform-specific count behavior.**
   The branch validated local Make and reviewed CMake membership. Windows/MSVC
   CTest count and platform exclusion behavior still rely on CI rather than a
   local lane.

5. **No performance or package evidence was added.**
   Sprint 120 was a maintainability and proof-owner sprint. It did not execute
   benchmarks, refresh package/install validation, or expand ABI evidence.

## Final Metrics

### Validation

| Metric | Sprint 120 close state |
|---|---:|
| library source-list count | 49 |
| CMake registered tests | 57 |
| focused QR tests | 63 passed, 0 failed |
| focused QR solve tests | 10 passed, 0 failed |
| focused BiCGSTAB tests | 49 passed, 0 failed |
| focused block BiCGSTAB tests | 12 passed, 0 failed |
| focused cross-solver oracle tests | 1 passed, 0 failed |
| cross-solver pilot assertions | 20 |
| required formatting gate | `make format` passed |
| required lint gate | `make lint` passed |
| required full Make test gate | `make test` passed |
| full Make test final result | `All tests passed.` |
| clean CMake membership proof | configured, built, and `ctest -N` reported 57 tests |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 120 docs |

### Sprint Artifact Package

| Metric | Sprint 120 close state |
|---|---:|
| artifact files under `SPRINT_120/artifacts/` | 14 |
| sprint plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| new focused test files | 3 |
| modified existing test files | 2 |
| modified build registration files | 2 |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| QR solve ownership | Completed focused owner split into `tests/test_qr_solve.c`. |
| QR factorization owner | Preserved in `tests/test_qr.c`. |
| Block BiCGSTAB ownership | Completed focused owner split into `tests/test_bicgstab_block.c`. |
| Scalar BiCGSTAB owner | Preserved in `tests/test_bicgstab.c`. |
| Cross-solver oracle pilot | Completed bounded generated-RHS SPD pilot in `tests/test_cross_solver_oracle.c`. |
| Make/CMake registration | Preserved; all new test owners are registered in both build surfaces. |
| Source-list parity | Preserved; no new library source was added. |
| Public API | Unchanged. |
| Public documentation claims | Unchanged; closeout records non-claims. |
| Benchmarks/performance | Not claimed and not refreshed. |
| External oracle completeness | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Design LDLT Matrix Market and KKT fixture helper extraction only with
  LDLT-local inertia and generated-RHS contracts.
- Design an LDLT cross-backend scenario owner that keeps route selection,
  dense/native backend behavior, linked-list/CSC agreement, and telemetry
  visible.
- Design an LDLT CSC solve owner before moving AMD behavior, in-place solve,
  relative infinity residual, inertia, linked-list agreement, and singular
  detection proof blocks.
- Pair any LDLT CSC external dense-reference split with platform skip/error
  policy, external oracle trust boundaries, permutation lifecycle ownership,
  and full quality evidence.
- Revisit QR reconstruction or sparse-mode split only in a QR maintainability
  sprint.
- Promote QR exact-RHS or residual helpers only after repeated split pressure
  proves duplication is worse than helper coupling.
- Design a block MINRES owner that preserves LDLT/GMRES comparison semantics
  and convergence expectations.
- Split GMRES SuiteSparse, restart, and right-preconditioner behavior only
  after fixture taxonomy, restart tolerance policy, and focused old/new
  validation are explicit.
- Extract matrix-free BiCGSTAB callback behavior only with callback-specific
  error propagation and scalar owner stability proof.
- Revisit public iterative handle helper movement when handle-lifecycle proof
  owners are being consolidated.
- Expand broad CG/GMRES shared fixtures only after at least two focused owners
  need the same fixture without hiding solver-local tolerances.
- Treat external dense-reference oracle expansion as future work with trust
  boundaries, skip/error policy, and platform behavior defined first.
- Decide in a future packaging or CI sprint whether the focused oracle owners
  need install-consumer or platform count lanes.

Still consciously constrained rather than silently solved:

- no new QR capability claim;
- no new BiCGSTAB capability claim;
- no broad direct solver parity claim;
- no broad direct/iterative parity claim;
- no external-oracle completeness claim;
- no package/install/platform/ABI claim;
- no performance or scalability claim;
- no state-of-the-art claim;
- no local Windows/MSVC validation claim.

Not carried forward as unresolved Sprint 120 debt:

- direct oracle owner inventory;
- iterative oracle owner inventory;
- shared fixture architecture policy;
- split candidate ranking;
- QR solve split design and implementation;
- block BiCGSTAB split design and implementation;
- bounded cross-solver oracle pilot design and implementation;
- Make/CMake registration for the new test owners;
- final validation and closeout package.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-direct-oracle-ownership-audit.md](./artifacts/day2-direct-oracle-ownership-audit.md)
- [day3-iterative-oracle-ownership-audit.md](./artifacts/day3-iterative-oracle-ownership-audit.md)
- [day4-shared-fixture-architecture.md](./artifacts/day4-shared-fixture-architecture.md)
- [day5-split-ranking-proof-plan.md](./artifacts/day5-split-ranking-proof-plan.md)
- [day6-direct-split-implementation-checklist.md](./artifacts/day6-direct-split-implementation-checklist.md)
- [day7-direct-split-implementation.md](./artifacts/day7-direct-split-implementation.md)
- [day8-direct-validation-consolidation.md](./artifacts/day8-direct-validation-consolidation.md)
- [day9-iterative-split-implementation-checklist.md](./artifacts/day9-iterative-split-implementation-checklist.md)
- [day10-iterative-split-implementation.md](./artifacts/day10-iterative-split-implementation.md)
- [day11-cross-solver-oracle-pilot-design.md](./artifacts/day11-cross-solver-oracle-pilot-design.md)
- [day12-cross-solver-oracle-pilot-implementation.md](./artifacts/day12-cross-solver-oracle-pilot-implementation.md)
- [day13-validation-package.md](./artifacts/day13-validation-package.md)
- [day14-oracle-closeout.md](./artifacts/day14-oracle-closeout.md)
