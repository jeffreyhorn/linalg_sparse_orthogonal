# Sprint 146 Retrospective

**Sprint:** 146 - Epic 12 Final Validation, Claim Recalibration & Closeout
**Duration:** 14 days (Days 1-14 landed on branch `sprint-146`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 146 day-by-day plan, working notes, artifact directory,
      and final closeout artifacts.
- [x] Consumed Sprint 137-145 closeout evidence as prerequisite input:
  - Sprint 137 baseline, gap selection, evidence contract, and non-goals;
  - Sprint 138 maintained corpus/report architecture;
  - Sprint 139 bounded QR residual closure;
  - Sprint 140 bounded partial-SVD residual closure;
  - Sprint 141 normalized report index and freshness semantics;
  - Sprint 142 runtime/backend governance and sentinel boundaries;
  - Sprint 143 static-first package/ABI decision;
  - Sprint 144 platform support-tier promotion;
  - Sprint 145 adoption-surface simplification and claim map.
- [x] Published the final Epic 12 evidence inventory across corpus, QR,
      partial-SVD, report, runtime/backend, package, platform, adoption, and
      validation surfaces.
- [x] Designed and ran the strongest feasible local closeout validation
      package for the final claimed surfaces.
- [x] Reconciled hosted Linux, macOS, and Windows support-tier evidence against
      the latest inspected green `master` baseline while preserving the
      missing branch-specific Sprint 146 hosted CI boundary.
- [x] Audited public and support/maintainer surfaces for unsupported
      state-of-the-art, external parity, package, platform, performance, ABI,
      Windows parity, and generated-report freshness wording.
- [x] Published a final residual queue with stable IDs, owner surfaces,
      blockers, prerequisites, promotion gates, and non-claim mappings.
- [x] Drafted and finalized the Epic 12 retrospective.
- [x] Reconciled the Epic 12 project plan against completed, bounded,
      deferred, rejected, and residual outcomes.
- [x] Published the Sprint 146 final closeout package and next-epic handoff.
- [x] Ran Sprint 146 documentation validation:
  - `git diff --check`;
  - trailing-whitespace scans over the Sprint 146 and Epic 12 closeout docs;
  - artifact/reference existence checks.

## What Went Well

1. **The sprint kept final claims tied to evidence.** The closeout separated
   fixture-local numerical proof, source-controlled report metadata, generated
   local rows, hosted master CI, local package checks, and support-tier wording
   instead of collapsing them into broad claims.

2. **The evidence inventory made the final audit straightforward.** Days 2 and
   3 turned Sprints 137-145 into a source-owned map, so Day 8 and Day 9 could
   audit public and maintainer wording against specific artifacts and commands.

3. **Validation matched the changed surfaces.** Sprint 146 was documentation
   closeout work, but it still reused the Day 5 local validation package for
   report, package, CMake, examples, QR, partial-SVD, and generated-local
   oracle/report proof.

4. **The residual queue is actionable.** Day 11 published stable residual IDs
   `R1` through `R14` with owners and promotion gates, making future planning
   clearer than a generic deferred-work list.

5. **The state-of-the-art decision stayed conservative.** The final Epic 12
   retrospective explicitly rejects unqualified state-of-the-art status because
   no direct comparative external-library evidence was produced.

## What Didn't Go Well

1. **Branch-specific hosted CI was unavailable during local closeout.** The
   sprint reconciled the latest inspected green `master` baseline, but there
   was no hosted `sprint-146` run before PR creation. This remains residual
   `R1`.

2. **Windows remains the largest platform gap.** Windows CMake-first support is
   documented, but staged pthread/POSIX tests, Makefile parity, `pkg-config`
   parity, and reviewed install-validation parity remain unpromoted.

3. **Broad numerical claims remain outside Epic 12.** The sprint confirmed
   bounded QR and partial-SVD closures, but broad QR, SVD, partial-SVD,
   SuiteSparse, LAPACK, NumPy, SciPy, PETSc, Trilinos, and ARPACK parity still
   need direct evidence.

4. **Generated report freshness is still claim-specific.** The final closeout
   preserves report-index semantics and local generated proof, but broad
   generated benchmark, sentinel, coverage, dead-code, and guardrail refreshes
   remain future work.

5. **Adoption cleanup is not fully exhausted.** Sprint 145 improved the
   first-use path and Sprint 146 validated its claim boundaries, but tutorial
   alignment and broader public-header cleanup remain residuals.

## Final Metrics

### Validation

| Metric | Sprint 146 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked `.h` changes | no |
| full C quality gate required | no |
| corpus schema validation | passed on Day 5 |
| report-index unit tests | passed on Day 5 |
| source-controlled report normalization | passed on Day 5: 47 rows ok |
| generated-aware report normalization | passed on Day 5: 47 rows ok |
| report freshness | passed on Day 5: freshness ok for 47 rows |
| selected support-family normalization | passed on Day 5: 9 rows ok |
| selected support-family freshness | passed on Day 5: freshness ok for 9 rows |
| static package deferral | passed on Day 5 |
| Make install/`pkg-config` validation | passed on Day 5: 23 passed, 0 failed |
| CMake install/export validation | passed on Day 5: 26 passed, 0 failed, 0 skipped |
| maintained example build | passed on Day 5: 14 example binaries built |
| focused QR corpus proof | passed on Day 5: 4 tests, 0 failures, 83 assertions |
| focused partial-SVD corpus proof | passed on Day 5: 6 tests, 0 failures, 140 assertions |
| local oracle/report refresh | passed on Day 5 with ignored `build/` outputs |
| latest inspected hosted Linux baseline | passed on `master` commit `daac9a85d516f72100c34b90b92ec78941a72200` |
| latest inspected hosted macOS baseline | passed on `master` commit `daac9a85d516f72100c34b90b92ec78941a72200` |
| latest inspected hosted Windows baseline | passed on `master` commit `daac9a85d516f72100c34b90b92ec78941a72200` with `56` expected CTest registrations |
| branch-specific hosted Sprint 146 CI | residual R1 until PR/branch CI exists |
| final Markdown hygiene | passed |

### Artifact Package

| Metric | Sprint 146 close state |
| --- | ---: |
| daily artifacts under `SPRINT_146/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| epic retrospective files | 1 |
| source-controlled generated report files committed | 0 |
| source files changed | 0 |
| public headers changed | 0 |

## Closed Claim

Sprint 146 closes this claim:

Epic 12 now has a final closeout package that inventories its evidence,
validates the final local surfaces, reconciles hosted support tiers against
the latest inspected green `master` baseline, audits public and support claims,
publishes residuals with promotion gates, rejects unsupported
state-of-the-art and broad parity claims, and records the Epic 12 retrospective
and next-epic handoff.

This claim is supported by:

- `docs/planning/EPIC_12/EPIC_12_RETROSPECTIVE.md`;
- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-closeout-intake-evidence-map.md](./artifacts/day1-closeout-intake-evidence-map.md);
- [day2-corpus-solver-evidence-inventory.md](./artifacts/day2-corpus-solver-evidence-inventory.md);
- [day3-support-evidence-inventory.md](./artifacts/day3-support-evidence-inventory.md);
- [day4-final-validation-baseline-design.md](./artifacts/day4-final-validation-baseline-design.md);
- [day5-final-local-validation-command-log.md](./artifacts/day5-final-local-validation-command-log.md);
- [day6-ci-evidence-intake.md](./artifacts/day6-ci-evidence-intake.md);
- [day7-cross-platform-reconciliation.md](./artifacts/day7-cross-platform-reconciliation.md);
- [day8-public-claim-audit.md](./artifacts/day8-public-claim-audit.md);
- [day9-support-maintainer-claim-audit.md](./artifacts/day9-support-maintainer-claim-audit.md);
- [day10-residual-queue-design.md](./artifacts/day10-residual-queue-design.md);
- [day11-published-residual-queue.md](./artifacts/day11-published-residual-queue.md);
- [day12-epic-12-retrospective-draft.md](./artifacts/day12-epic-12-retrospective-draft.md);
- [day13-final-project-plan-reconciliation.md](./artifacts/day13-final-project-plan-reconciliation.md);
- [day14-final-closeout-package.md](./artifacts/day14-final-closeout-package.md).

## Next-Epic Readiness

The next epic should select one complete gap closure instead of advancing every
residual shallowly:

| Candidate | Residuals | Readiness |
| --- | --- | --- |
| Windows platform closure | R1, R2, R3 | Ready for planning once hosted PR evidence exists and Windows staged/parity scope is selected. |
| Numerical corpus expansion | R5, R6, R12 | Ready to extend the maintained corpus/report pattern from Sprints 138-140. |
| Shared-library and ABI productization | R4, R14 | Ready only as a product-level package/ABI decision, not as a small CMake toggle. |
| Report evidence refresh | R1, R7 | Ready when concrete generated families are needed for a claim or review. |
| Adoption/documentation completion | R8, R9 | Ready for tutorial alignment and broader header cleanup without support-claim expansion. |
| Runtime/backend follow-through | R10, R11 | Ready for typed-control promotion review and additional sentinel rows. |
| Competitive positioning | R12, R13 | Not ready for a claim until direct comparative evidence is designed and collected. |

## Residual Deferred Debt

The final residual queue is published in
[day11-published-residual-queue.md](./artifacts/day11-published-residual-queue.md).
The most important carry-forward items are:

- branch-specific hosted Sprint 146 CI reconciliation;
- Windows staged pthread/POSIX test portability;
- reviewed Windows install-validation parity;
- shared-library ABI productization;
- package-manager distribution;
- broad QR and partial-SVD corpus expansion;
- external-library parity study;
- generated benchmark, sentinel, coverage, dead-code, and guardrail refreshes;
- tutorial alignment with the first-use ladder;
- broader public-header cleanup;
- runtime/backend typed-control and sentinel follow-through;
- state-of-the-art competitive decision.

Still consciously constrained rather than silently solved:

- no unqualified state-of-the-art sparse linear algebra claim;
- no broad external-library parity;
- no broad QR, SVD, or partial-SVD correctness beyond reviewed fixtures;
- no portable performance or backend-superiority claim;
- no generated report freshness proof from source-controlled rows alone;
- no coverage-completeness or zero-dead-code claim;
- no shared-library support or dynamic ABI compatibility;
- no runtime-loader compatibility;
- no package-manager support;
- no Windows Makefile or `pkg-config` parity;
- no Windows reviewed install-validation parity;
- no Windows staged test closure;
- no branch-specific hosted Sprint 146 CI pass until PR/branch workflows run.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [Epic 12 retrospective](../EPIC_12_RETROSPECTIVE.md)
- [day1-closeout-intake-evidence-map.md](./artifacts/day1-closeout-intake-evidence-map.md)
- [day2-corpus-solver-evidence-inventory.md](./artifacts/day2-corpus-solver-evidence-inventory.md)
- [day3-support-evidence-inventory.md](./artifacts/day3-support-evidence-inventory.md)
- [day4-final-validation-baseline-design.md](./artifacts/day4-final-validation-baseline-design.md)
- [day5-final-local-validation-command-log.md](./artifacts/day5-final-local-validation-command-log.md)
- [day6-ci-evidence-intake.md](./artifacts/day6-ci-evidence-intake.md)
- [day7-cross-platform-reconciliation.md](./artifacts/day7-cross-platform-reconciliation.md)
- [day8-public-claim-audit.md](./artifacts/day8-public-claim-audit.md)
- [day9-support-maintainer-claim-audit.md](./artifacts/day9-support-maintainer-claim-audit.md)
- [day10-residual-queue-design.md](./artifacts/day10-residual-queue-design.md)
- [day11-published-residual-queue.md](./artifacts/day11-published-residual-queue.md)
- [day12-epic-12-retrospective-draft.md](./artifacts/day12-epic-12-retrospective-draft.md)
- [day13-final-project-plan-reconciliation.md](./artifacts/day13-final-project-plan-reconciliation.md)
- [day14-final-closeout-package.md](./artifacts/day14-final-closeout-package.md)
