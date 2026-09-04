# Sprint 196 Plan: Epic 17 Final Validation, Claim Calibration & Closeout

**Sprint Duration:** 14 days
**Goal:** Reconcile all Epic 17 work, run final validation, calibrate public
claims, and publish the Epic 17 retrospective and residual queue.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 196 estimate in the Epic 17
project plan.

**Primary scope:** Reconcile Sprints 187 through 195 outcomes, evidence,
validation records, decisions, and residuals; recalibrate README, INSTALL,
maintainer, benchmark, API, and planning documentation claims; mark Epic 17
project-plan status with evidence links; run integrated focused and full
quality gates appropriate to changed surfaces; publish
`EPIC_17_RETROSPECTIVE.md`; and produce a prioritized residual queue with
exact closure targets and long-horizon deferrals.

**Non-goals:** New solver algorithms, public API or ABI expansion,
package-manager implementation, new platform support, broad Windows parity,
portable performance claims, release claims, state-of-the-art claims not backed
by Epic 17 evidence, or partial implementation of new residuals beyond final
calibration and closeout.

---

## Day 1: Sprint Intake and Evidence Map

**Title:** Closeout Intake
**Theme:** Establish Sprint 196 scope and map Epic 17 evidence sources before
claim or status edits.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 196 section of the Epic 17 project plan and map items
   196.1 through 196.6 to likely documentation, validation, retrospective, and
   residual-queue surfaces.
2. Inventory Sprint 187 through Sprint 195 plans, working notes,
   retrospectives, closeout artifacts, review comments, validation logs, and
   residual lists.
3. Create `WORKING_NOTES.md` with a sprint-item checklist, evidence ledger
   scaffold, claim-surface inventory, risk register, and Day 2 reconciliation
   questions.
4. Identify public and maintainer documentation files that may need claim
   recalibration after Epic 17.
5. Record validation gates that must be considered for final integrated
   evidence.

### Deliverables

- Sprint 196 working-notes scaffold.
- Epic 17 evidence-source inventory.
- Claim-surface inventory.
- Initial validation-gate map and closeout risk register.

### Completion Criteria

- Sprint scope is traceable to items 196.1 through 196.6.
- Sprint 187 through Sprint 195 evidence sources are identified.
- No claim or status edits begin before the evidence map exists.

---

## Day 2: Sprint Outcome Reconciliation

**Title:** Outcome Ledger
**Theme:** Reconcile Sprint 187 through Sprint 195 outcomes, decisions,
validation records, and residuals into one ledger.
**Time estimate:** 12 hours

### Tasks

1. Read each Epic 17 sprint retrospective and closeout artifact for completed,
   narrowed, deferred, residualized, and superseded outcomes.
2. Build an item-level ledger for package, Windows, comparison, performance,
   review-surface, adoption, API coherence, and reliability work.
3. Record the exact evidence files or commands supporting each outcome.
4. Identify conflicts between earlier plans and final sprint outcomes.
5. Create a Day 2 reconciliation artifact with item 196.1 acceptance evidence.

### Deliverables

- Sprint 187-195 outcome ledger.
- Evidence-link table by sprint and topic.
- Conflict and supersession list.
- Item 196.1 reconciliation notes.

### Completion Criteria

- Every Sprint 187-195 outcome has a status and evidence link.
- Deferred and residual outcomes are separated from completed outcomes.
- Conflicting or superseded wording is identified before documentation edits.

---

## Day 3: Residual and Deferred Work Triage

**Title:** Residual Triage
**Theme:** Consolidate Epic 17 residuals and rank them by closure value,
evidence requirement, risk, and next-epic fit.
**Time estimate:** 12 hours

### Tasks

1. Extract residuals from Sprint 187 through Sprint 195 retrospectives,
   working notes, closeout artifacts, and review notes.
2. Deduplicate residuals that point at the same underlying gap.
3. Classify residuals as next-epic candidates, long-horizon deferrals,
   documentation-only follow-ups, validation/tooling follow-ups, or
   out-of-scope historical notes.
4. Define exact closure evidence needed for each retained residual.
5. Produce the Day 3 residual-triage artifact with initial item 196.6 inputs.

### Deliverables

- Consolidated residual table.
- Deduplicated deferred-work queue.
- Closure-evidence requirements.
- Residual priority rationale.

### Completion Criteria

- Residuals have owners or owner conditions.
- Closure targets are concrete enough to drive future planning.
- Long-horizon deferrals are not mixed into near-term next-epic work.

---

## Day 4: Claim Surface Audit

**Title:** Claim Audit
**Theme:** Audit README, INSTALL, maintainer, benchmark, API, and planning docs
for claims that must match earned Epic 17 evidence.
**Time estimate:** 12 hours

### Tasks

1. Search maintained documentation for package, Windows, comparison,
   performance, support, reliability, adoption, API, release, and
   state-of-the-art claims.
2. Compare each claim against the Day 2 evidence ledger and Day 3 residual
   queue.
3. Mark wording that is accurate, too broad, stale, duplicated, or missing
   evidence links.
4. Identify docs where active user-facing truth should point to consolidated
   support/readiness or maintainer interpretation.
5. Create the Day 4 claim-surface audit artifact for item 196.2.

### Deliverables

- Claim-surface audit artifact.
- Overclaim, underclaim, stale-claim, and duplication table.
- Documentation edit plan.
- Evidence-link requirements by document.

### Completion Criteria

- Claim recalibration targets are evidence-backed.
- Public and maintainer documentation surfaces are separated by audience.
- No state-of-the-art, performance, platform, or package claim is widened
  without evidence.

---

## Day 5: Public Documentation Recalibration

**Title:** Public Claims
**Theme:** Update README, INSTALL, benchmark, and user-facing docs so public
claims match Epic 17 evidence.
**Time estimate:** 12 hours

### Tasks

1. Update README wording for package support, Windows evidence, selected
   external comparisons, performance methodology, reliability proofs,
   support/readiness truth, and retained non-claims.
2. Update INSTALL and support/readiness wording so users can distinguish
   supported, validated, local-only, hosted-evidence, deferred, and unclaimed
   surfaces.
3. Update benchmark or performance docs so methodology-bound evidence remains
   selected and non-portable unless broader evidence exists.
4. Preserve links to sprint evidence without making planning artifacts the
   primary user workflow.
5. Record changed files, claim edits, and retained non-claims in working notes.

### Deliverables

- Claim-recalibrated public documentation.
- Updated support/readiness interpretation.
- Selected performance and comparison non-claim wording.
- Day 5 change log.

### Completion Criteria

- Public claims are no broader than earned Epic 17 evidence.
- Users can find active support/readiness truth without reading sprint notes.
- Package, platform, performance, release, and state-of-the-art non-claims
  remain explicit.

---

## Day 6: Maintainer and API Documentation Recalibration

**Title:** Maintainer Claims
**Theme:** Update maintainer, API, and planning-adjacent docs with exact Epic
17 evidence boundaries and ownership rules.
**Time estimate:** 12 hours

### Tasks

1. Update `docs/maintainer_guide.md` to summarize Epic 17 evidence ownership,
   focused gates, decision records, and residual interpretation.
2. Audit API documentation and generated-doc source inputs for claim drift
   caused by Epic 17 adoption or support wording.
3. Update planning-adjacent docs that now need final Epic 17 status,
   non-claim, or residual references.
4. Ensure maintainer guidance identifies which gates own package, Windows,
   comparison, benchmark, review-surface, adoption, and reliability evidence.
5. Record the Day 6 maintainer/API recalibration artifact.

### Deliverables

- Updated maintainer evidence interpretation.
- API documentation claim-boundary notes or edits.
- Planning-adjacent claim updates.
- Gate-owner map for maintainers.

### Completion Criteria

- Maintainer docs align with public docs but carry deeper evidence ownership.
- API docs do not imply unsupported package, platform, ABI, performance, or
  release claims.
- Focused gates and residual owners are discoverable.

---

## Day 7: Project Plan Status Pass

**Title:** Project Status
**Theme:** Mark Epic 17 sprint items complete, narrowed, deferred,
residualized, or superseded with evidence links.
**Time estimate:** 12 hours

### Tasks

1. Review `docs/planning/EPIC_17/PROJECT_PLAN.md` against the Day 2 outcome
   ledger and Day 3 residual queue.
2. Add or update item-status annotations for Sprints 187 through 196 using
   consistent wording and evidence links.
3. Mark narrowed outcomes distinctly from fully completed original scope.
4. Mark deferred, residualized, and superseded work without hiding the reason
   or closure evidence required.
5. Create the Day 7 project-plan-status artifact with item 196.3 evidence.

### Deliverables

- Updated Epic 17 project-plan status.
- Evidence links for completed and narrowed items.
- Deferred/residual/superseded status notes.
- Day 7 status artifact.

### Completion Criteria

- Item 196.3 has project-plan status coverage.
- Every sprint item has a reviewable final status.
- Status wording does not convert residuals into completed claims.

---

## Day 8: Epic Retrospective Outline and Metrics

**Title:** Retrospective Outline
**Theme:** Draft the Epic 17 retrospective structure, outcome metrics, closed
claims, and residual sections before final text.
**Time estimate:** 12 hours

### Tasks

1. Review prior epic retrospectives for format, metrics, closed-claim wording,
   residual queues, and next-epic handoff style.
2. Draft the `EPIC_17_RETROSPECTIVE.md` outline with sections for outcome
   summary, sprint-by-sprint results, validation, changed surface, closed
   claims, non-claims, residuals, and state-of-the-art assessment.
3. Populate initial metrics from Sprint 187 through Sprint 195 retrospectives
   and closeout artifacts.
4. Identify missing evidence or metrics that must be filled before finalizing
   the retrospective.
5. Record Day 8 retrospective outline notes.

### Deliverables

- Epic 17 retrospective outline.
- Initial outcome and validation metrics.
- Missing-evidence checklist.
- Day 8 retrospective planning artifact.

### Completion Criteria

- Item 196.5 has a concrete retrospective structure.
- Metrics are sourced from sprint evidence, not memory.
- Missing evidence is listed before final writing begins.

---

## Day 9: Epic Retrospective Draft

**Title:** Retrospective Draft
**Theme:** Write the first complete Epic 17 retrospective from reconciled
evidence and calibrated claims.
**Time estimate:** 12 hours

### Tasks

1. Draft `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` from the Day 8
   outline and evidence ledger.
2. Summarize Sprint 187 through Sprint 196 outcomes with links to sprint
   plans, retrospectives, closeout artifacts, and validation records.
3. Include what went well, what did not, final metrics, closed claims,
   non-claims, and residuals.
4. Add a state-of-the-art assessment that distinguishes achieved evidence from
   remaining gaps.
5. Record unresolved retrospective questions in working notes.

### Deliverables

- Draft Epic 17 retrospective.
- Sprint-by-sprint outcome summary.
- Initial state-of-the-art assessment.
- Open question list.

### Completion Criteria

- The retrospective exists and covers every Epic 17 sprint.
- Claims in the retrospective match the evidence ledger.
- Remaining open questions are explicit and bounded.

---

## Day 10: Prioritized Residual Queue

**Title:** Residual Queue
**Theme:** Publish a prioritized next-epic residual queue with closure targets
and long-horizon deferrals.
**Time estimate:** 12 hours

### Tasks

1. Convert the Day 3 residual triage into a publishable residual queue.
2. Rank next-epic candidates by user impact, risk reduction, evidence gap,
   implementation feasibility, and review cost.
3. For each near-term residual, define exact closure target, owner condition,
   acceptance evidence, validation gate, and non-claim boundary.
4. Separate long-horizon deferrals from actionable next-epic work.
5. Link the residual queue from the Epic 17 retrospective and relevant
   planning docs.

### Deliverables

- Prioritized residual queue.
- Closure-target table.
- Long-horizon deferral list.
- Links from retrospective and planning docs.

### Completion Criteria

- Item 196.6 has a publishable residual queue.
- Residuals can seed future planning without re-auditing Epic 17 from scratch.
- Deferred work is not presented as completed or implicitly claimed.

---

## Day 11: Integrated Focused Validation

**Title:** Focused Validation
**Theme:** Run focused gates across Epic 17 evidence owners and resolve
sprint-caused drift.
**Time estimate:** 12 hours

### Tasks

1. Run focused gates for package/install evidence, Windows PowerShell
   structural ownership, selected report targets, selected comparison
   freshness, selected benchmark freshness, review-surface guards, and
   reliability guards as applicable to the current repository.
2. Run documentation and planning checks that cover source lists, report
   normalizers, selected manifests, generated docs, and claim-boundary guards.
3. Fix any Sprint 196-caused failures or stop with exact failure context if a
   gate is unclear or environment-limited.
4. Record command results, fixes, skipped gates, and environment residuals.
5. Create the Day 11 focused-validation artifact for item 196.4.

### Deliverables

- Integrated focused validation log.
- Fix log for focused gate drift.
- Environment residual list.
- Day 11 validation artifact.

### Completion Criteria

- Focused Epic 17 evidence owners pass or have exact residual context.
- Claim-boundary and source ownership checks are synchronized.
- No unclear validation failure is hidden or converted into a pass.

---

## Day 12: Full Quality Gate

**Title:** Full Validation
**Theme:** Run full quality gates required by changed surfaces and resolve
Sprint 196 regressions.
**Time estimate:** 12 hours

### Tasks

1. Run `make format` and inspect any formatting changes before keeping them.
2. Run `make lint` and fix warnings or style defects caused by Sprint 196
   edits.
3. Run `make test` if any `.c` or `.h` files changed, or document why full C
   tests are not required if the sprint remained documentation-only.
4. Run additional full or reviewed quality wrappers needed by touched
   surfaces, such as compile-quality, CMake-quality, docs, install, generated
   artifact, or report freshness gates.
5. Record final full-gate commands, results, fixes, and residuals.

### Deliverables

- Passing formatting result.
- Passing lint result when required.
- Passing test result when required.
- Final full-quality validation log.

### Completion Criteria

- Item 196.4 full validation is complete for changed surfaces.
- No known Sprint 196 regression remains unresolved.
- Environment limitations are documented with exact commands and context.

---

## Day 13: Final Claim and Retrospective Review

**Title:** Final Review
**Theme:** Review all calibrated claims, retrospective wording, residuals, and
validation evidence before closeout.
**Time estimate:** 11 hours

### Tasks

1. Re-read README, INSTALL, maintainer guide, benchmark docs, API docs, Epic
   17 project plan, Epic 17 retrospective, and residual queue for consistency.
2. Verify that public claims, maintainer interpretation, project-plan status,
   and retrospective claims agree with the evidence ledger.
3. Confirm no broad package-manager, Windows parity, external-library parity,
   portable performance, reliability, release, ABI, or state-of-the-art claim
   was introduced without evidence.
4. Run targeted grep or documentation checks for high-risk claim words and
   evidence links.
5. Create the Day 13 final-review artifact and update working notes with any
   fixes.

### Deliverables

- Final claim-review artifact.
- Retrospective and residual consistency notes.
- High-risk claim grep results.
- Final documentation fix log.

### Completion Criteria

- Claims across public, maintainer, planning, and retrospective docs are
  consistent.
- Residuals remain explicit and prioritized.
- Day 14 can focus on closeout packaging rather than substantive rewrites.

---

## Day 14: Epic Closeout and Review Package

**Title:** Epic Closeout
**Theme:** Package Sprint 196 and Epic 17 evidence for review with final
traceability, validation, residuals, and handoff.
**Time estimate:** 11 hours

### Tasks

1. Review all Sprint 196 changed files for scope control, stable wording,
   evidence links, and no unrelated cleanup.
2. Ensure items 196.1 through 196.6 each have evidence in artifacts,
   documentation, validation logs, project-plan status, retrospective, or
   residual queue.
3. Update `WORKING_NOTES.md` with final closeout summary, residuals,
   validation results, and reviewer notes.
4. Prepare a concise review checklist covering evidence reconciliation, claim
   recalibration, project-plan status, integrated validation, Epic
   retrospective, residual queue, and non-claims.
5. Confirm there are no unstaged generated files, missing artifacts, missing
   evidence links, or unrecorded validation residuals.

### Deliverables

- Sprint 196 closeout notes.
- Item-to-evidence traceability checklist.
- Final Epic 17 residual and non-claim list.
- Review-ready Epic 17 closeout package.

### Completion Criteria

- Sprint 196 deliverables are complete and traceable.
- Reviewers can reproduce integrated validation from documented commands.
- Epic 17 closes with calibrated claims, explicit residuals, and no unsupported
  state-of-the-art assertion.
