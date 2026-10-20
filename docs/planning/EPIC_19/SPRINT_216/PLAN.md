# Sprint 216 Plan: Epic 19 Final Validation, Claim Calibration & Closeout

**Sprint Duration:** 14 days
**Goal:** Reconcile Sprint 207-215 outcomes, calibrate public claims to earned
evidence, publish the Epic 19 retrospective and prioritized residual queue,
and decide explicitly whether any stronger support or state-of-the-art claim
is justified.

**Time budget:** Twelve 12-hour days and two 11-hour days total `166` hours.
No day exceeds 12 hours.

**Primary scope:** Reconcile merged sprint artifacts and PR evidence; update
README, INSTALL, maintainer, benchmark, API, corpus, and planning claims;
status every Epic 19 item; validate changed contracts; and publish the epic
retrospective, residual queue, and state-of-the-art assessment.

**Evidence rule:** A completed sprint deliverable is not automatically a
completed product capability. Keep local validation separate from hosted
evidence, a blueprint separate from an executed external comparison, and a
blocked pilot separate from a passing result. No stronger claim is approved
without a named evidence bundle, scope, and publication decision.

**Non-goals:** This closeout does not assume shared-library/ABI release
readiness, generated API publication, Windows support parity, a timing
threshold, external-baseline correctness or speed, or a broad state-of-the-art
ranking. Residual work may be promoted only with its own acceptance evidence.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 216.1 Evidence Reconciliation | 26 h | Days 1-3: 12 + 12 + 2 |
| 216.2 Claim Recalibration | 34 h | Days 3-5: 10 + 12 + 12 |
| 216.3 Project Plan Status | 24 h | Days 6-7: 12 + 12 |
| 216.4 Integrated Validation | 34 h | Days 9-11, 14: 2 + 12 + 10 + 10 |
| 216.5 Epic Retrospective | 26 h | Days 11-14: 2 + 12 + 11 + 1 |
| 216.6 Residual Queue | 22 h | Days 8-9: 12 + 10 |
| **Total** | **166 h** | **Days 1-12: 12 h each; Days 13-14: 11 h each** |

---

## Day 1: Epic Evidence Intake

**Title:** Epic Evidence Intake
**Theme:** Establish the sources of truth for final closeout.
**Time estimate:** 12 hours

### Tasks

1. Inventory Sprint 207-215 plans, retrospectives, working notes, validation
   records, decision artifacts, residual queues, and merged PR evidence.
2. Read the Epic 19 project plan, review, and todo documents; map items
   207.1-216.6 to their intended closure evidence.
3. Record which outcomes are implemented, documented-only, selected/local,
   blocked, deferred, or superseded without inferring success from a plan.
4. Create `WORKING_NOTES.md` with an evidence ledger, source links, claim
   register, validation matrix, decisions, and unresolved questions.
5. Write `artifacts/day1-epic-evidence-intake.md` with missing records and
   dependencies that could block final claims.

### Deliverables

- Evidence-source index and item-to-record map.
- Working-notes scaffold and Day 1 intake artifact.

### Completion Criteria

- Every Sprint 207-215 has an identified outcome source or a named gap.
- Local, hosted, selected, and blueprint-only evidence are distinguishable.
- No item is marked complete solely because its sprint was scheduled.

---

## Day 2: Outcome And Validation Reconciliation

**Title:** Outcome And Validation Reconciliation
**Theme:** Compare sprint conclusions with their actual validation records.
**Time estimate:** 12 hours

### Tasks

1. Reconcile package/distribution, Windows solver, allocation-failure, and
   review-surface decisions from Sprints 207-211 with test and CI evidence.
2. Reconcile benchmark, generated API, ABI/release, and external-baseline
   decisions from Sprints 212-215 with their recorded limits.
3. Identify stale links, contradictory statuses, unmerged evidence, or
   statements that treat blocked or optional work as delivered capability.
4. Build a cross-sprint table of decision, exact evidence, platform/scope,
   unresolved dependency, and proposed final disposition.
5. Write `artifacts/day2-outcome-validation-reconciliation.md` and update
   the working-notes ledger.

### Deliverables

- Cross-sprint decision and validation table.
- Discrepancy list with owner or next decision point.
- Day 2 reconciliation artifact.

### Completion Criteria

- Each proposed outcome links to a concrete artifact, check, or PR result.
- Failed, skipped, unavailable, and not-run checks retain their real status.
- No local result is presented as cross-platform or release evidence.

---

## Day 3: Evidence Freeze And Claim Inventory

**Title:** Evidence Freeze And Claim Inventory
**Theme:** Finish reconciliation and identify every public claim surface.
**Time estimate:** 12 hours (216.1: 2 h; 216.2: 10 h)

### Tasks

1. Freeze the item 216.1 evidence map and record any unresolved discrepancy
   as a blocker or residual, not an assumed pass.
2. Search README, `INSTALL.md`, `docs/maintainer_guide.md`, benchmark and API
   documentation, corpus documentation, and Epic 19 planning text for support,
   platform, release, publication, performance, and state-of-the-art claims.
3. Classify each statement as supported, too broad, ambiguous, or missing;
   attach the narrowest evidence-supported replacement wording.
4. Define a claim matrix with claim, surface, evidence link, scope, negative
   boundary, and reviewer decision.
5. Write `artifacts/day3-claim-inventory.md` and update the claim register.

### Deliverables

- Final item 216.1 evidence map.
- Public-claim inventory and Day 3 claim matrix.

### Completion Criteria

- Every proposed stronger claim has specific evidence or is rejected.
- Blueprint, provisional pilot, and accepted external evidence have separate
  statuses.
- The inventory covers all documentation surfaces named in item 216.2.

---

## Day 4: User And Maintainer Claim Calibration

**Title:** User And Maintainer Claim Calibration
**Theme:** Align installation, support, and operational language.
**Time estimate:** 12 hours

### Tasks

1. Update README and `INSTALL.md` for exact supported install paths,
   platform/toolchain evidence, static package scope, and known limits.
2. Update `docs/maintainer_guide.md` so validation commands, CI lane scope,
   release/ABI decisions, and blocked promotion gates match merged evidence.
3. Check user-facing solver guidance for statements that imply Windows,
   shared-library, or external-baseline support beyond validated cells.
4. Add focused documentation regressions where a repeatable overclaim can be
   checked without treating a textual guard as proof of product behavior.
5. Record edits and rationale in `artifacts/day4-user-maintainer-claims.md`.

### Deliverables

- Recalibrated user and maintainer support language.
- Day 4 edit and evidence record.

### Completion Criteria

- Documented install and release claims cite the exact supported contract.
- Unimplemented or blocked paths remain explicit and actionable.
- New wording does not imply ABI or platform readiness from static-only proof.

---

## Day 5: Benchmark, API, Corpus, And Planning Claims

**Title:** Benchmark, API, Corpus, And Planning Claims
**Theme:** Close the remaining documentation claim surfaces.
**Time estimate:** 12 hours

### Tasks

1. Reconcile benchmark docs with the selected canonical freshness policy,
   local sentinels, timing limitations, and external-baseline non-claims.
2. Reconcile `docs/api_reference.md` and related API text with the generated
   HTML local-only decision and actual publication policy.
3. Reconcile `tests/corpus/README.md`, schema docs, and relevant planning
   pages with provenance, optional-data, and report acceptance rules.
4. Extend focused claim guards for wording likely to recur, preserving
   legitimate negated or qualified statements.
5. Write `artifacts/day5-benchmark-api-corpus-claims.md` and close the
   item 216.2 claim matrix.

### Deliverables

- Recalibrated benchmark, API, corpus, and planning documentation.
- Completed claim matrix and Day 5 evidence record.

### Completion Criteria

- Freshness is not described as a timing threshold or external speed proof.
- Local generated API output is not described as a published artifact.
- Corpus and report claims distinguish source-controlled evidence from
  optional or provisional records.

---

## Day 6: Project Plan Status, Sprints 207-211

**Title:** Project Plan Status, Sprints 207-211
**Theme:** Turn early Epic 19 outcomes into auditable item dispositions.
**Time estimate:** 12 hours

### Tasks

1. Assign each Sprint 207-211 project-plan item one status: `complete`,
   `promoted`, `deferred`, `residualized`, or `superseded`.
2. Add evidence links and concise rationale beside each status; identify the
   next closure target for any promoted or residualized work.
3. Check that selected Windows, allocation-failure, package-distribution,
   and review-surface outcomes are not widened beyond their tested scope.
4. Preserve the original item descriptions and estimates while annotating
   actual outcomes in `PROJECT_PLAN.md` or an explicitly linked status table.
5. Write `artifacts/day6-project-plan-status-207-211.md` with open questions.

### Deliverables

- Status and evidence links for Sprint 207-211 items.
- Day 6 disposition artifact.

### Completion Criteria

- Each early-sprint item has exactly one current disposition.
- Status rationale points to merged evidence rather than unmerged notes.
- Deferred and residualized work has a next decision or acceptance target.

---

## Day 7: Project Plan Status, Sprints 212-215

**Title:** Project Plan Status, Sprints 212-215
**Theme:** Complete the project-plan ledger and verify cross-item consistency.
**Time estimate:** 12 hours

### Tasks

1. Assign the same five-way status to each Sprint 212-215 item and link its
   decision, validation record, retrospective, and remaining gate.
2. Distinguish the Sprint 215 blueprint and blocked pilot from a completed
   external correctness or performance comparison.
3. Verify benchmark threshold, API publication, ABI/shared-library, and
   release-readiness statuses against their authoritative decisions.
4. Add Sprint 216 item status placeholders that will be filled only after
   final validation and closeout review.
5. Write `artifacts/day7-project-plan-status-212-215.md`; audit all status
   links and contradictions across Sprints 207-215.

### Deliverables

- Complete Sprint 207-215 status ledger in the project plan.
- Day 7 status artifact and contradiction log.

### Completion Criteria

- All prior-sprint items have a status, evidence link, and scoped rationale.
- Blocked/optional pilot work is not labeled an accepted external result.
- Sprint 216 status remains provisional until the final gate passes.

---

## Day 8: Residual Queue Intake And Prioritization

**Title:** Residual Queue Intake And Prioritization
**Theme:** Consolidate all unclosed work into one actionable queue.
**Time estimate:** 12 hours

### Tasks

1. Merge Sprint 207-215 residual queues and the Epic 19 review/todo tracks;
   de-duplicate repeated issues without losing evidence or dependencies.
2. Separate near-term closure candidates from long-horizon research,
   external-package, hosted-evidence, and release-policy decisions.
3. Rank entries by user impact, correctness or publication risk,
   prerequisite order, and feasibility; state why each priority is earned.
4. For each entry, specify owner role, exact acceptance evidence, stop
   condition, dependencies, and claim unlocked only if closed.
5. Draft `RESIDUAL_QUEUE.md` and write
   `artifacts/day8-residual-queue-intake.md`.

### Deliverables

- Deduplicated next-epic residual inventory and first priority order.
- Day 8 queue-intake artifact.

### Completion Criteria

- No unresolved item disappears because a sprint marked its design complete.
- Queue entries are testable closure targets, not generic follow-up labels.
- Long-horizon deferrals are explicit and carry a revisit decision point.

---

## Day 9: Residual Queue Decision And Validation Matrix

**Title:** Residual Queue Decision And Validation Matrix
**Theme:** Finalize handoff priorities and select the required checks.
**Time estimate:** 12 hours (216.6: 10 h; 216.4: 2 h)

### Tasks

1. Finalize `RESIDUAL_QUEUE.md` with priority, owner role, closure target,
   prerequisites, and claim gate for each retained item.
2. Reconcile queue entries with project-plan dispositions and document why
   any older residual was closed, superseded, or intentionally deferred.
3. Make an explicit state-of-the-art evidence decision: what can be said now,
   what cannot, and the minimum validated external matrix/platform coverage
   and publication approval needed to revisit stronger wording.
4. Inventory changed files and choose focused documentation, schema,
   comparison, install, and pilot tests; flag any `.c`/`.h` edits for the
   mandatory `make format && make lint && make test` gate.
5. Write `artifacts/day9-residual-decision-validation-matrix.md` with the
   exact commands, environment, and expected outcomes.

### Deliverables

- Prioritized next-epic residual queue and state-of-the-art decision.
- Day 9 changed-surface validation matrix.

### Completion Criteria

- Each unclosed project-plan item appears in the queue or has a documented
  supersession/deferral rationale.
- No claim is unlocked by the existence of a plan or provisional record.
- Every changed contract has an executable check or a named validation gap.

---

## Day 10: Focused Contract Validation

**Title:** Focused Contract Validation
**Theme:** Test claim wording and changed documentation/tooling contracts.
**Time estimate:** 12 hours

### Tasks

1. Run the focused claim, API publication/routing, corpus/schema, selected
   report/comparison, and external-pilot synthetic checks identified on Day 9.
2. Run `make docs-check` and affected benchmark/report freshness targets
   where the changed surfaces require them; record skips and prerequisites.
3. Review generated report or pilot artifacts as data, verifying statuses,
   hashes, provenance, and non-claim handling rather than relying on exit
   codes alone.
4. Fix any failing check within scope and rerun it; otherwise mark closeout
   blocked and keep the result in the validation ledger.
5. Write `artifacts/day10-focused-validation.md` with commands, exit codes,
   platform, and artifact locations.

### Deliverables

- Focused validation logs and Day 10 record.
- Corrected claim/tooling defects, if found.

### Completion Criteria

- Focused checks pass or an explicit blocker prevents stronger claims.
- Expected blocked/not-run outcomes are not converted into passing evidence.
- The record says which hosted/platform checks were not executed locally.

---

## Day 11: Full Gate And Retrospective Outline

**Title:** Full Gate And Retrospective Outline
**Theme:** Apply required source gates and frame the epic assessment.
**Time estimate:** 12 hours (216.4: 10 h; 216.5: 2 h)

### Tasks

1. If any `.c` or `.h` file changed, run `make format && make lint && make
   test` in order and retain raw output; do not close while any phase fails.
2. Run other full quality gates required by changed surfaces and project
   policy, keeping tree-mutating alternate-build modes isolated from the
   normal reviewed path.
3. Reconcile Day 10 focused outcomes with Day 9's validation matrix and
   identify tests that must be repeated after closeout edits.
4. Outline `EPIC_19_RETROSPECTIVE.md` around outcomes, evidence, decisions,
   unsupported claims, limitations, and residual handoff.
5. Write `artifacts/day11-quality-gate-and-retro-outline.md` with phase
   results and outstanding validation risks.

### Deliverables

- Required quality-gate evidence or an explicit blocking failure.
- Epic retrospective outline and Day 11 validation record.

### Completion Criteria

- Every required gate has a command, result, and platform attribution.
- Failing gates block closeout; no failure is relabeled as a harmless skip.
- Retrospective outline points to evidence, not only sprint summaries.

---

## Day 12: Epic Outcomes And Non-Claims

**Title:** Epic Outcomes And Non-Claims
**Theme:** Draft the retrospective from the reconciled record.
**Time estimate:** 12 hours

### Tasks

1. Draft `EPIC_19_RETROSPECTIVE.md` with Sprint 207-215 outcomes and the
   item-status decisions, grouped by capability and evidence level.
2. Explain what changed for package distribution, Windows, allocation
   testing, review surface, benchmarks, generated API, ABI/release, and
   external-baseline design.
3. Record negative evidence, validation failures, missing hosted checks,
   blocked pilot status, and claims the epic did not earn.
4. Link the final residual queue and make the boundary between completed
   closeout paperwork and incomplete capability work explicit.
5. Write `artifacts/day12-epic-outcomes-draft.md` and cross-check cited
   artifacts against the Day 1-2 ledger.

### Deliverables

- Substantial epic retrospective draft.
- Day 12 outcomes and evidence cross-check.

### Completion Criteria

- Each outcome has a source link and a scoped acceptance statement.
- Retrospective does not count design, local preflight, or documentation as
  external performance or release proof.
- Residuals are traceable to the next-epic queue.

---

## Day 13: State-Of-The-Art Assessment And Review

**Title:** State-Of-The-Art Assessment And Review
**Theme:** Finish the retrospective and audit its claims.
**Time estimate:** 11 hours

### Tasks

1. Finish the state-of-the-art assessment with the selected baseline status,
   validated matrix/platform coverage, missing independent comparisons,
   publication permissions, and exact future promotion gates.
2. Compare every retrospective statement with the final claim matrix,
   project-plan ledger, residual queue, and Day 10-11 validation records.
3. Review lessons learned and process changes without treating them as
   completed product capabilities.
4. Run link and documentation consistency checks; correct broken or stale
   evidence pointers and rerun affected focused tests.
5. Write `artifacts/day13-retrospective-review.md` with reviewer questions
   and the final non-claim list.

### Deliverables

- Review-ready `EPIC_19_RETROSPECTIVE.md`.
- Day 13 state-of-the-art and claim audit.

### Completion Criteria

- Stronger support or state-of-the-art wording is either narrowly justified
  by named accepted evidence or explicitly declined.
- Retrospective, status ledger, and queue do not contradict one another.
- Missing hosted, external, and release evidence remains visible.

---

## Day 14: Final Validation And Epic Closeout

**Title:** Final Validation And Epic Closeout
**Theme:** Verify the final tree and publish an honest handoff.
**Time estimate:** 11 hours (216.5: 1 h; 216.4: 10 h)

### Tasks

1. Incorporate final evidence and validation results into
   `EPIC_19_RETROSPECTIVE.md` without strengthening unsupported claims.
2. Rerun every focused check affected by Days 12-14 edits, `make docs-check`,
   and `git diff --check`; rerun the full C gate if any `.c`/`.h` edits were
   made after Day 11.
3. Verify the complete item 216.1-216.6 ledger, project-plan statuses,
   residual queue, claim matrix, and retrospective links.
4. Write `artifacts/day14-integrated-validation-closeout.md` with exact
   commands, exit statuses, platform, changed-surface coverage, failures,
   and unrun hosted/optional lanes.
5. Mark Sprint 216 items complete only when their deliverables and required
   checks pass; otherwise record a blocked or residual outcome with a precise
   closure target.

### Deliverables

- Final claim-recalibrated documentation and project-plan status ledger.
- Integrated validation record, Epic 19 retrospective, and residual queue.
- Explicit state-of-the-art assessment and closeout decision.

### Completion Criteria

- All required changed-surface and source-quality checks pass; otherwise
  closeout remains blocked with the failure preserved.
- Every Epic 19 item has a traceable final disposition and evidence link.
- Public claims remain no stronger than the validated solver, matrix,
  platform, package, and publication evidence actually available.
