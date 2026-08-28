# Sprint 186 Plan: Epic 16 Final Validation, Claim Calibration & Closeout

**Sprint Duration:** 14 days
**Goal:** Reconcile all Epic 16 deliverables, recalibrate public claims, run
final validation, and publish the Epic 16 retrospective and residual queue.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 186 estimate in the Epic 16
project plan.

**Primary scope:** Reconcile the evidence and status matrix across Sprints
177-185, calibrate public documentation claims against earned evidence, update
project-plan status, run integrated validation, publish the Epic 16
retrospective, and create the next-epic residual handoff.

**Non-goals:** Adding new product capabilities, reopening completed sprint
scope without evidence of drift, broad refactors, changing numerical behavior,
changing public API contracts, or claiming support that is not backed by
validation and artifacts.

---

## Day 1: Sprint Intake and Closeout Map

**Title:** Final Closeout Intake
**Theme:** Establish the Sprint 186 closeout scope and inherited evidence.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 186 section of the Epic 16 project plan and capture the
   acceptance boundaries for items 186.1 through 186.6.
2. Inventory Sprint 177-185 plans, working notes, retrospectives, and artifact
   directories.
3. Identify deliverables that are complete, narrowed, deferred, residualized,
   or still ambiguous.
4. Start `WORKING_NOTES.md` with closeout scope, source artifacts, validation
   expectations, risks, and open questions.
5. Draft the evidence reconciliation checklist for Days 2 and 3.

### Deliverables

- Sprint 186 working-notes scaffold.
- Epic 16 closeout source-artifact inventory.
- Evidence reconciliation checklist tied to items 186.1 through 186.6.

### Completion Criteria

- Sprint scope is traceable to the project-plan items.
- Sprint 177-185 artifact sources are known and accessible.
- Open questions are explicit before evidence reconciliation begins.

---

## Day 2: Sprint Evidence Matrix

**Title:** Evidence Matrix Baseline
**Theme:** Build the cross-sprint evidence/status matrix.
**Time estimate:** 12 hours

### Tasks

1. Extract completed deliverables, decisions, validation records, and residuals
   from Sprint 177-185 artifacts.
2. Map each Epic 16 item to its supporting plan, artifact, working-note, or
   retrospective evidence.
3. Identify missing or weak evidence links that need follow-up before closeout.
4. Separate implementation evidence from documentation-only assertions.
5. Record the initial matrix in a reviewable artifact.

### Deliverables

- Initial Epic 16 evidence/status matrix.
- Missing-evidence and weak-evidence list.
- Artifact links for completed sprint outcomes.

### Completion Criteria

- Every Sprint 177-185 item has an initial status row.
- Evidence gaps are visible and bounded.
- The matrix can drive claim calibration work.

---

## Day 3: Evidence Reconciliation

**Title:** Status Reconciliation
**Theme:** Resolve evidence gaps and classify final item outcomes.
**Time estimate:** 12 hours

### Tasks

1. Review weak evidence rows from Day 2 and inspect the underlying code, docs,
   scripts, generated records, or validation outputs.
2. Classify each Epic 16 item as complete, narrowed, deferred, residualized, or
   superseded.
3. Document rationale for any item that cannot be marked complete.
4. Record final evidence links and residual references for each classification.
5. Update `WORKING_NOTES.md` with reconciliation decisions.

### Deliverables

- Reconciled Epic 16 evidence/status matrix.
- Classification rationale for incomplete or narrowed items.
- Residual candidate list for Day 13 prioritization.

### Completion Criteria

- Each Epic 16 item has a final closeout classification.
- Non-complete classifications include rationale and evidence.
- Claim calibration can proceed from reconciled facts.

---

## Day 4: Public Claim Inventory

**Title:** Claim Surface Inventory
**Theme:** Find all public and maintainer-facing claims affected by Epic 16.
**Time estimate:** 12 hours

### Tasks

1. Inventory README, INSTALL, maintainer guide, report docs, package docs,
   generated API docs, and planning documents for Epic 16-related claims.
2. Identify claims about supported platforms, package managers, generated API
   coverage, report freshness, provider status, validation, and public header
   coherence.
3. Compare each claim to the reconciled evidence matrix.
4. Flag unsupported, overbroad, stale, duplicated, or missing claims.
5. Draft a claim-calibration checklist grouped by document family.

### Deliverables

- Public claim inventory.
- Unsupported and stale claim list.
- Document-family checklist for calibration edits.

### Completion Criteria

- All relevant claim surfaces are identified.
- Unsupported or overbroad claims are separated from earned claims.
- Day 5 can begin documentation edits without further broad discovery.

---

## Day 5: README, INSTALL, and Package Claims

**Title:** User-Facing Claim Calibration
**Theme:** Align primary user-facing documentation with earned Epic 16 support.
**Time estimate:** 12 hours

### Tasks

1. Update README claims to match completed Epic 16 evidence and retained
   non-claims.
2. Update INSTALL and package-manager documentation for selected provider
   status, package availability, and platform caveats.
3. Preserve useful guidance while removing unsupported promises or ambiguous
   support wording.
4. Cross-link evidence or status records where users need a precise support
   boundary.
5. Run focused docs checks or formatting checks relevant to changed files.

### Deliverables

- Calibrated README, INSTALL, and package documentation.
- Notes for removed, narrowed, or retained non-claims.
- Focused documentation validation results.

### Completion Criteria

- Primary user-facing docs match the evidence matrix.
- Package and provider claims are precise and current.
- No unsupported support or availability claims remain in edited surfaces.

---

## Day 6: Maintainer and Report Documentation Claims

**Title:** Maintainer Claim Calibration
**Theme:** Align internal guidance and report metadata docs with final status.
**Time estimate:** 12 hours

### Tasks

1. Update maintainer guidance for Epic 16 validation responsibilities, report
   metadata, generated API status, and residual handling.
2. Update report documentation for selected target manifest status, freshness
   policy, Windows evidence boundaries, and known caveats.
3. Ensure maintainer-facing docs distinguish required gates from optional or
   environment-dependent checks.
4. Remove stale references to superseded sprint decisions.
5. Run focused docs and link checks available in the repository.

### Deliverables

- Calibrated maintainer and report documentation.
- Updated validation responsibility notes.
- Focused docs/link validation evidence.

### Completion Criteria

- Maintainer guidance reflects actual Epic 16 closeout responsibilities.
- Report docs distinguish earned claims from caveats and residuals.
- Stale sprint-decision wording is removed or replaced.

---

## Day 7: Generated API and Header Coherence Claims

**Title:** API Claim Calibration
**Theme:** Align generated API and public header claims with final evidence.
**Time estimate:** 12 hours

### Tasks

1. Review generated API documentation inputs and outputs for Epic 16 claim
   drift.
2. Update generated API status notes, public header coherence docs, and any
   related maintainer references.
3. Confirm declaration-preserving header cleanup claims are tied to validation
   evidence rather than broad API-change promises.
4. Record any generated-doc regeneration commands or deferred regeneration
   risks.
5. Run focused generated-doc, header, or declaration drift checks where
   available.

### Deliverables

- Calibrated generated API and header coherence documentation.
- Generated-doc regeneration or deferral notes.
- Focused validation results for API/header claim surfaces.

### Completion Criteria

- Generated API claims match completed evidence and known limitations.
- Public header coherence claims are precise and evidence-backed.
- Any generated output risk is documented before full validation.

---

## Day 8: Project Plan Status Update

**Title:** Project Plan Closeout
**Theme:** Mark Epic 16 project-plan items with final evidence-linked status.
**Time estimate:** 12 hours

### Tasks

1. Update the Epic 16 project plan to mark Sprint 177-186 items complete,
   narrowed, deferred, residualized, or superseded.
2. Add evidence links to sprint artifacts, validation records, decisions, and
   retrospectives.
3. Ensure status wording is consistent across all Epic 16 sprint sections.
4. Avoid changing estimates or scope history unless the update is explicitly a
   closeout status note.
5. Record unresolved project-plan residuals for Day 13.

### Deliverables

- Evidence-linked Epic 16 project-plan status updates.
- Consistent status vocabulary across the epic.
- Residual list carried forward from project-plan review.

### Completion Criteria

- Project-plan status reflects the reconciled evidence matrix.
- Each non-complete item points to a residual or deferral rationale.
- Scope history remains understandable for future reviewers.

---

## Day 9: Integrated Validation Plan

**Title:** Validation Matrix Design
**Theme:** Define the final Epic 16 validation suite before running it.
**Time estimate:** 12 hours

### Tasks

1. Inventory required quality gates, package checks, report checks, docs checks,
   workflow guards, generated API checks, and provider/status guards.
2. Map each validation command to the Epic 16 evidence or claim it protects.
3. Identify environment-dependent checks and document prerequisites or skip
   rules.
4. Define failure triage rules for validation issues found on Days 10 and 11.
5. Prepare the integrated validation command list.

### Deliverables

- Integrated Epic 16 validation matrix.
- Command-to-claim traceability notes.
- Failure triage and rerun plan.

### Completion Criteria

- Required final checks are known before execution.
- Environment-dependent checks have explicit handling.
- Validation coverage is traceable to closeout claims.

---

## Day 10: Focused Integrated Checks

**Title:** Focused Final Validation
**Theme:** Run bounded checks for docs, reports, package/provider status, and
generated API surfaces.
**Time estimate:** 12 hours

### Tasks

1. Run focused package-manager, provider, report freshness, selected manifest,
   generated API, docs, and workflow guard checks.
2. Fix failures that are within Sprint 186 closeout scope.
3. Record out-of-scope failures as residuals with exact reproduction commands.
4. Rerun failed focused checks after fixes.
5. Update the validation matrix with results and caveats.

### Deliverables

- Focused integrated validation results.
- Fixes for in-scope documentation or guard failures.
- Residual records for out-of-scope failures.

### Completion Criteria

- Focused validation either passes or has explicit residual records.
- In-scope claim drift found by focused checks is corrected.
- The full quality gate is ready to run.

---

## Day 11: Full Repository Quality Gate

**Title:** Full Final Validation
**Theme:** Run the repository-level closeout quality gate.
**Time estimate:** 12 hours

### Tasks

1. Run `make format`.
2. Run `make lint`.
3. Run `make test`.
4. Run source-list checks, workflow guards, docs/report checks, generated API
   checks, package/provider checks, and `git diff --check` as defined on Day 9.
5. Fix in-scope failures and rerun the failing command until required checks
   pass.

### Deliverables

- Passing repository-level validation record.
- Final command output summary for the retrospective.
- Remaining residuals for failures outside Sprint 186 scope, if any.

### Completion Criteria

- `make format`, `make lint`, and `make test` pass when code files are touched
  or when required by closeout scope.
- All required closeout guards pass or have documented environment-based skip
  rationale.
- No unresolved in-scope validation failures remain.

---

## Day 12: Epic Retrospective Draft

**Title:** Retrospective Assembly
**Theme:** Draft the Epic 16 retrospective from evidence and validation.
**Time estimate:** 12 hours

### Tasks

1. Create `docs/planning/EPIC_16/EPIC_16_RETROSPECTIVE.md`.
2. Summarize Epic 16 goals, delivered outcomes, narrowed outcomes,
   non-claims, and validation evidence.
3. Include state-of-the-art assessment notes grounded in completed artifacts.
4. Capture major decisions, tradeoffs, regressions avoided, and validation
   limitations.
5. Link to the evidence matrix, sprint retrospectives, and final validation
   record.

### Deliverables

- Draft Epic 16 retrospective.
- Outcome and non-claim summary.
- Linked validation and evidence references.

### Completion Criteria

- The retrospective is grounded in evidence rather than aspiration.
- Completed and residualized work are clearly separated.
- Validation limitations are visible to future maintainers.

---

## Day 13: Residual Queue and Next-Epic Handoff

**Title:** Residual Handoff Queue
**Theme:** Prioritize remaining work with exact closure targets.
**Time estimate:** 12 hours

### Tasks

1. Convert all residual candidates from Days 3, 8, 10, and 11 into a
   prioritized handoff queue.
2. Define closure targets, owning surface, expected evidence, validation
   command, and deferral horizon for each residual.
3. Separate near-term cleanup from long-horizon research or infrastructure
   work.
4. Link residuals back to the evidence matrix and Epic 16 retrospective.
5. Review the queue for duplicated or already-closed items.

### Deliverables

- Prioritized Epic 16 residual queue.
- Next-epic handoff notes with closure targets.
- Long-horizon deferral list.

### Completion Criteria

- Every residual has a concrete closure target or explicit deferral rationale.
- The handoff queue is prioritized and deduplicated.
- Future sprint planning can consume the queue without rediscovery.

---

## Day 14: Closeout Review and PR Handoff

**Title:** Epic Closeout Handoff
**Theme:** Package Sprint 186 and Epic 16 closeout for review.
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 186 changes against project-plan items 186.1 through
   186.6.
2. Confirm claim calibration, project-plan status, validation records,
   retrospective, and residual queue are internally consistent.
3. Check for stale TODOs, unresolved open questions, broken links, generated
   artifacts, and accidental scope expansion.
4. Produce final review-ready notes summarizing completed work, validation,
   residuals, and non-claims.
5. Update `WORKING_NOTES.md` with closeout results and retrospective-ready
   summary.

### Deliverables

- Review-ready Sprint 186 working notes.
- Final Epic 16 closeout summary.
- PR-ready validation, residual, and non-claim notes.

### Completion Criteria

- All Sprint 186 project-plan items have a documented outcome.
- Epic 16 closeout artifacts are consistent and reviewable.
- The branch is ready for retrospective creation and PR preparation.
