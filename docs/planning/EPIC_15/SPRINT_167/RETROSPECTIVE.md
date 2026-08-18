# Sprint 167 Retrospective

**Sprint:** 167 - Epic 15 Baseline, Evidence Ledger & Claim Gate
**Duration:** 14 days (Days 1-14 landed on branch `sprint-167`)
**Status:** Complete

## Source Artifact Note

Sprint 167 was executed from the active Epic 15 project-plan section for
Sprint 167 and lives under `docs/planning/EPIC_15/SPRINT_167/` with its plan,
working notes, daily artifacts, closeout artifact, and retrospective in one
package. The original sprint prompt referenced an older Epic 12 project-plan
path; `WORKING_NOTES.md` records that mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 167 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited Epic 13 and Epic 14 residuals and classified their Epic 15
      relevance by claim risk, value, feasibility, dependencies, and
      recommended handling.
- [x] Inventoried source, public headers, tests, corpus manifests, report
      families, CI workflows, package/install proof, and public documentation
      claim surfaces.
- [x] Drafted and reviewed the Epic 15 evidence ledger.
- [x] Added explicit non-claim rows for unqualified state-of-the-art status,
      broad external-library parity, portable performance superiority,
      shared-library support, dynamic ABI stability, package-manager
      distribution, broad platform parity, Windows Makefile/`pkg-config`
      parity, generated API HTML publication, broad report freshness, broad
      allocation-failure guarantees, and solver correctness beyond maintained
      fixtures.
- [x] Selected the finite Epic 15 closure targets for Sprints 168-176.
- [x] Defined acceptance criteria, validation command expectations, hosted
      evidence requirements, stop conditions, and a future implementation
      handoff template.
- [x] Prepared the Sprint 168 handoff with `bench_refactor_csc` through
      `make bench-canonical-report` as the recommended hosted performance
      publication candidate.
- [x] Ran final docs-only hygiene validation with `git diff --check`.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required for Sprint 167 edits.

## What Went Well

1. **The sprint established a clear evidence baseline.** The artifact sequence
   moved from prior-epic residuals through source, test, CI, package, docs,
   ledger, selection, acceptance gates, and closeout without widening current
   claims.

2. **The ledger separated support from non-claims.** Day 10 made unsupported
   surfaces explicit instead of leaving them as implicit caveats. That gives
   later sprints a cleaner guard against claim drift.

3. **The Epic 15 scope became finite.** Day 11 selected bounded closures for
   performance publication, ABI, package-manager readiness, headers,
   generated API docs, external comparison, report freshness, allocation
   failure, and final claim recalibration.

4. **Acceptance gates are concrete.** Day 12 gave each selected gap objective
   completion criteria, validation expectations, hosted-evidence rules, and
   stop conditions.

5. **Sprint 168 has a practical handoff.** Day 13 identified the existing
   `bench_refactor_csc` canonical report path as the preferred starting point
   for hosted methodology-bound performance publication, while preserving the
   no-portable-superiority boundary.

6. **Docs-only validation stayed proportionate.** Sprint 167 did not edit
   source or public headers, so it avoided unnecessary full C gates while
   still running `git diff --check` after each day.

## What Didn't Go Well

1. **The initial prompt path was stale.** The request referenced Epic 12 while
   the active Sprint 167 plan belongs to Epic 15. The sprint handled this by
   recording the mismatch and proceeding from `docs/planning/EPIC_15/`.

2. **The evidence surface is large.** Public docs, package metadata, CI
   workflows, generated report conventions, benchmark scripts, and planning
   artifacts all own pieces of the claim story. The sprint needed many
   inventories before selection could be made safely.

3. **Hosted proof remains future work.** Sprint 167 can define hosted evidence
   requirements, but the actual hosted performance, package, comparison,
   platform, and failure-path proof depends on Sprints 168-176 and their PR CI.

4. **Several valuable gaps are still non-closeable broadly.** Broad
   state-of-the-art positioning, external-library ecosystem parity, portable
   performance superiority, broad platform parity, broad report freshness,
   broad solver correctness, and broad allocation-failure guarantees remain
   out of scope.

5. **The Sprint 168 performance recommendation still needs runtime validation.**
   `bench_refactor_csc` is the best starting candidate, but Sprint 168 must
   confirm runtime budget, methodology metadata, and hosted suitability before
   publishing it as hosted evidence.

## Final Metrics

### Validation

| Metric | Sprint 167 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required for Sprint 167 edits | no |
| final `git diff --check` | passed |
| benchmark/report generation | skipped; Sprint 167 selected future evidence work only |
| hosted CI proof | not applicable for planning-only artifacts |
| generated build/docs/report/cache artifacts committed | 0 |

### Artifact Package

| Metric | Sprint 167 close state |
| --- | ---: |
| daily artifacts under `SPRINT_167/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| source files changed | 0 |
| public headers changed | 0 |
| selected Epic 15 gap IDs | 9 |
| explicit non-claim rows | 12 |
| stop conditions | 12 |

### Claim Governance

| Metric | Sprint 167 close state |
| --- | ---: |
| reviewed evidence ledger rows | 18 |
| selected future sprint owners | 9 |
| broad state-of-the-art claims added | 0 |
| broad external-library parity claims added | 0 |
| portable performance superiority claims added | 0 |
| shared-library or dynamic ABI claims added | 0 |
| package-manager distribution claims added | 0 |
| broad platform parity claims added | 0 |

## Closed Claim

Sprint 167 closes this Epic 15 baseline and claim-gate claim:

Epic 15 now has a source-controlled planning baseline that inventories the
current evidence surface, reviews claim and non-claim boundaries, selects a
finite set of closeable gaps for Sprints 168-176, defines acceptance criteria
and stop conditions for those gaps, and prepares Sprint 168 to begin hosted
methodology-bound performance publication work without widening unsupported
state-of-the-art, parity, package, ABI, platform, report, or solver-correctness
claims.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-prior-epic-residual-audit.md](./artifacts/day2-prior-epic-residual-audit.md);
- [day3-residual-risk-value-classification.md](./artifacts/day3-residual-risk-value-classification.md);
- [day4-source-header-surface-inventory.md](./artifacts/day4-source-header-surface-inventory.md);
- [day5-test-corpus-surface-inventory.md](./artifacts/day5-test-corpus-surface-inventory.md);
- [day6-ci-workflow-inventory.md](./artifacts/day6-ci-workflow-inventory.md);
- [day7-package-install-evidence-inventory.md](./artifacts/day7-package-install-evidence-inventory.md);
- [day8-documentation-claim-surface-inventory.md](./artifacts/day8-documentation-claim-surface-inventory.md);
- [day9-evidence-ledger-draft.md](./artifacts/day9-evidence-ledger-draft.md);
- [day10-evidence-ledger-review.md](./artifacts/day10-evidence-ledger-review.md);
- [day11-gap-selection-gate.md](./artifacts/day11-gap-selection-gate.md);
- [day12-claim-gates.md](./artifacts/day12-claim-gates.md);
- [day13-sprint-reconciliation.md](./artifacts/day13-sprint-reconciliation.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No broad state-of-the-art sparse linear algebra status, broad external-library
parity, portable performance superiority, backend superiority,
package-manager distribution, shared-library support, dynamic ABI stability,
runtime-loader behavior, broad platform parity, broad report freshness, broad
solver correctness, or broad allocation-failure guarantee was added.

## Epic 15 Readiness

| Future sprint | Ready input from Sprint 167 |
| --- | --- |
| Sprint 168 | Start with the `bench_refactor_csc` canonical report candidate and prove one hosted methodology-bound performance lane. |
| Sprint 169 | Harden the selected performance lane with methodology policy, stable schema, sentinels, and caveats. |
| Sprint 170 | Use the package/header inventory and non-claim rows to decide shared-library ABI posture. |
| Sprint 171 | Build from the Sprint 170 decision to prove one package-manager path or formally defer provider distribution. |
| Sprint 172 | Select one high-impact public-header family from the Day 4 inventory. |
| Sprint 173 | Decide generated API HTML publication after header cleanup. |
| Sprint 174 | Add one bounded external comparison family using the Day 5 corpus/comparison candidate set. |
| Sprint 175 | Promote one report freshness path beyond Linux or formally close the deferral. |
| Sprint 176 | Add one deterministic allocation-failure proof and recalibrate final Epic 15 claims. |

## Follow-Up Risks

| Risk | Handling |
| --- | --- |
| Hosted performance publication may exceed runtime budget. | Sprint 168 must narrow fixture scope or retain local-only status. |
| ABI/package wording may drift before decisions land. | Sprint 170 should enforce static/shared and ABI non-claims mechanically. |
| Package-manager readiness may be confused with source install support. | Sprint 171 must keep provider support separate from Make/CMake install proof. |
| Generated API docs may be treated as published before a policy exists. | Sprint 173 must either publish them or enforce local-only status. |
| One added comparison family may be overread as ecosystem parity. | Sprint 174 must keep fixture, comparator, metric, and tolerance scope explicit. |
| Final closeout may be tempted to broaden claims. | Sprint 176 should use Day 12 stop conditions and the evidence ledger as the control surface. |
