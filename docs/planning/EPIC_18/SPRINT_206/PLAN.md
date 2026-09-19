# Sprint 206 Plan: Epic 18 Final Validation, Claim Calibration & Closeout

**Sprint Duration:** 14 days
**Goal:** Reconcile Epic 18 outcomes, run final validation, calibrate claims,
publish the retrospective and residual queue, and decide whether any stronger
support claims are earned.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 206 estimate in the Epic 18
project plan.

**Primary scope:** Reconcile Sprint 197 through Sprint 205 outcomes, update
claim-bearing public and maintainer documentation, mark project-plan status
with evidence links, run integrated validation, publish the Epic 18
retrospective, and create a prioritized residual queue for future work.

**Non-goals:** New solver behavior, new public API or ABI guarantees, broad
package-manager support, Homebrew/core readiness, Windows parity, Linuxbrew,
bottle support, portable performance claims, release claims, hosted generated
API publication, or state-of-the-art claims not earned by existing Epic 18
evidence.

---

## Day 1: Closeout Intake

**Title:** Closeout Intake
**Theme:** Establish Sprint 206 scope, evidence surfaces, and closeout
decision rules before changing claim-bearing documentation.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 206 Epic 18 project-plan section and map items 206.1
   through 206.6 to expected artifacts, docs, validation commands, and final
   closeout records.
2. Inventory Sprint 197 through Sprint 205 plans, working notes,
   retrospectives, review comment follow-ups, residual notes, and merged PR
   outcomes.
3. Identify public, maintainer, corpus, benchmark, API, install, support, and
   planning docs that currently make or constrain Epic 18 claims.
4. Create `WORKING_NOTES.md` with an item checklist, evidence map, risk
   register, validation matrix, and open questions.
5. Record closeout non-goals and claim-boundary rules before editing any
   source-controlled claim surface.

### Deliverables

- Sprint 206 working-notes scaffold.
- Item-to-evidence traceability map.
- Closeout surface inventory.
- Initial risk register and validation matrix.

### Completion Criteria

- Every Sprint 206 item has an initial evidence path or artifact category.
- All claim-bearing surfaces are identified before claim recalibration starts.
- Unsupported package, ABI, platform, performance, release, publication, and
  state-of-the-art claims remain explicitly out of scope.

---

## Day 2: Sprint Outcome Reconciliation

**Title:** Outcome Reconciliation
**Theme:** Reconcile Sprint 197 through Sprint 205 outcomes into one
evidence-backed status ledger.
**Time estimate:** 12 hours

### Tasks

1. Review each Epic 18 sprint retrospective and artifact directory for closed,
   narrowed, deferred, superseded, and residualized work.
2. Compare sprint outcomes against the Epic 18 project-plan item list and
   identify any status vocabulary drift.
3. Record exact evidence paths for package metadata, Windows freshness,
   benchmark freshness, allocation-failure proof, review-surface reduction,
   generated API policy, and adoption-support consolidation.
4. Identify any missing artifacts, stale status rows, or contradictory
   planning summaries that must be corrected later in the sprint.
5. Write the Day 2 outcome reconciliation artifact.

### Deliverables

- Sprint 197-205 status ledger.
- Evidence-link inventory.
- Contradiction and stale-status list.
- Day 2 reconciliation artifact.

### Completion Criteria

- Item 206.1 has evidence-backed status for every Sprint 197-205 item.
- Closed, narrowed, deferred, residualized, and superseded statuses are used
  consistently.
- Any unresolved evidence gap is explicitly recorded before claim updates.

---

## Day 3: Claim Surface Audit

**Title:** Claim Surface Audit
**Theme:** Audit public and maintainer documentation for claims that must be
updated, narrowed, or retained as non-claims.
**Time estimate:** 12 hours

### Tasks

1. Audit README, INSTALL, API reference, tutorial, cookbook, solver
   selection, benchmark docs, corpus docs, maintainer guide, and support
   matrix text.
2. Classify each claim as earned, narrowed, unsupported, residual, or
   historical planning evidence.
3. Identify duplicated caveats that should be replaced with links to the
   current authoritative support/readiness surfaces.
4. Record claim text that needs stronger non-claim wording for package
   distribution, generated API publication, Windows support, performance,
   ABI, release, and state-of-the-art status.
5. Write the Day 3 claim-surface audit artifact.

### Deliverables

- Claim-bearing documentation inventory.
- Earned-claim and non-claim classification table.
- Duplicate-caveat and stale-wording list.
- Day 3 audit artifact.

### Completion Criteria

- Item 206.2 has a complete claim audit before edits begin.
- Every proposed stronger claim has a specific Epic 18 evidence source.
- Unsupported claims are routed to non-claims or residuals.

---

## Day 4: Project Plan Status Design

**Title:** Status Design
**Theme:** Design the final Epic 18 project-plan status update and evidence
index before changing the project plan.
**Time estimate:** 12 hours

### Tasks

1. Define final status wording for Sprints 197 through 206 and for each
   Sprint 206 item.
2. Decide which evidence links belong in the project-plan current-status
   table versus retrospective or residual-queue records.
3. Map stale project-plan snapshots, residual-queue preambles, and Epic
   retrospective summary rows that must be reconciled.
4. Define a status vocabulary for complete, narrowed, deferred, residualized,
   and superseded outcomes.
5. Write the Day 4 project-plan status design artifact.

### Deliverables

- Final project-plan status update design.
- Evidence-link placement map.
- Status vocabulary and consistency rules.
- Day 4 design artifact.

### Completion Criteria

- Item 206.3 has an implementation-ready project-plan update plan.
- Status wording cannot overstate support beyond available evidence.
- Planning docs have one source of truth for current Epic 18 closeout status.

---

## Day 5: Claim Recalibration Batch One

**Title:** Public Claim Update
**Theme:** Update user-facing documentation so public claims match earned Epic
18 evidence.
**Time estimate:** 12 hours

### Tasks

1. Update README support, adoption, benchmark, generated API, package, and
   platform wording according to the Day 3 claim audit.
2. Update INSTALL support/readiness and package-manager guidance so local
   proof and non-claim boundaries remain clear.
3. Update tutorial, cookbook, solver-selection, or API reference routing only
   where it reduces user confusion without adding unsupported claims.
4. Keep quick-reference and support-matrix wording consistent with Sprint 205
   decisions.
5. Record changed public surfaces and claim rationale in `WORKING_NOTES.md`.

### Deliverables

- Claim-recalibrated public documentation.
- Public-doc evidence and rationale notes.
- Updated user-facing non-claim wording where needed.
- Day 5 implementation record.

### Completion Criteria

- Public documentation no longer contradicts Sprint 197-205 evidence.
- Package, platform, benchmark, performance, API, ABI, and release boundaries
  are clear to users.
- Item 206.2 has user-facing implementation progress with recorded evidence.

---

## Day 6: Claim Recalibration Batch Two

**Title:** Maintainer Claim Update
**Theme:** Update maintainer, benchmark, corpus, API, and planning-adjacent
documentation to align with the final claim boundary.
**Time estimate:** 12 hours

### Tasks

1. Update maintainer guide claim-boundary, validation, and support/readiness
   wording.
2. Update benchmark, corpus, generated API, selected-report, and workflow
   docs affected by Sprint 197-205 outcomes.
3. Remove or qualify stale references to pending work that has been closed or
   residualized.
4. Preserve historical artifacts as evidence while ensuring current-status
   surfaces are not stale.
5. Record changed maintainer surfaces and claim rationale in `WORKING_NOTES.md`.

### Deliverables

- Claim-recalibrated maintainer documentation.
- Updated benchmark, corpus, report, or API policy wording.
- Current-status stale-reference cleanup.
- Day 6 implementation record.

### Completion Criteria

- Maintainer documentation agrees with public support boundaries.
- Evidence docs distinguish current status from historical sprint records.
- Item 206.2 has maintainer-facing implementation progress with recorded
  evidence.

---

## Day 7: Project Plan Status Implementation

**Title:** Project Plan Update
**Theme:** Update Epic 18 project-plan and status surfaces with final
evidence-backed outcomes.
**Time estimate:** 12 hours

### Tasks

1. Update `docs/planning/EPIC_18/PROJECT_PLAN.md` status rows, snapshots, and
   evidence links for Sprints 197 through 206.
2. Update current Epic 18 status summaries that reference pending or closed
   sprint work.
3. Ensure Sprint 206 plan, working notes, and artifact paths are included in
   the status evidence index.
4. Cross-check project-plan totals, status vocabulary, and item dispositions.
5. Write the Day 7 project-plan status implementation artifact.

### Deliverables

- Updated Epic 18 project-plan status records.
- Updated current-status summaries.
- Evidence index including Sprint 206 surfaces.
- Day 7 implementation artifact.

### Completion Criteria

- Item 206.3 is implemented for project-plan and current-status docs.
- The project plan does not describe closed work as future pending work.
- Every status update has a supporting evidence link or explicit residual.

---

## Day 8: Validation Scope Design

**Title:** Validation Design
**Theme:** Define the final focused and broad validation matrix required by
the changed surfaces.
**Time estimate:** 12 hours

### Tasks

1. Inventory changed file types and map them to required validation commands.
2. Select focused documentation, guard, manifest, corpus, API, benchmark,
   routing, and support-matrix checks needed for the closeout changes.
3. Determine whether any `.c` or `.h` changes occurred and whether `make
   format`, `make lint`, and `make test` are required.
4. Define fallback evidence capture for environment-dependent checks that
   cannot run locally.
5. Write the Day 8 validation scope artifact.

### Deliverables

- Integrated validation matrix.
- Changed-surface to command mapping.
- Environment blocker and fallback evidence rules.
- Day 8 validation design artifact.

### Completion Criteria

- Item 206.4 has a complete validation plan before final gates run.
- Validation scope is proportional to changed surfaces.
- Any skipped command requires an explicit blocker, not convenience.

---

## Day 9: Focused Validation And Fixes

**Title:** Focused Validation
**Theme:** Run focused closeout checks and fix claim, routing, manifest, or
documentation regressions before broad gates.
**Time estimate:** 12 hours

### Tasks

1. Run focused checks for changed documentation, support matrix, API routing,
   local-only policy, selected report manifests, corpus schema, and planning
   status consistency.
2. Fix any failures that are local to Sprint 206 claim or planning changes.
3. Re-run failed focused checks until they pass or an environment blocker is
   recorded.
4. Capture exact command results in `WORKING_NOTES.md`.
5. Write the Day 9 focused validation artifact.

### Deliverables

- Focused validation command log.
- Fixes for any closeout-specific guard failures.
- Updated validation evidence.
- Day 9 validation artifact.

### Completion Criteria

- Focused checks pass or have documented blockers.
- No known claim-boundary, routing, manifest, or planning-status regression
  remains open.
- Item 206.4 has focused validation evidence.

---

## Day 10: Broad Quality Gates

**Title:** Broad Gates
**Theme:** Run required broad quality gates and capture final validation
evidence for changed surfaces.
**Time estimate:** 12 hours

### Tasks

1. Run `make format` and any documentation formatting or generated-index
   checks required by changed files.
2. Run `make lint` and `make test` if code or header files changed, or record
   why they are not required for documentation-only closeout.
3. Run project-level documentation and planning validation targets relevant to
   Epic 18 closeout.
4. Fix failures that are within Sprint 206 scope, then re-run affected gates.
5. Record the Day 10 integrated validation results and any residual blockers.

### Deliverables

- Broad quality gate command log.
- Final validation pass/fail evidence.
- Any required fixes from broad gates.
- Day 10 validation artifact.

### Completion Criteria

- Item 206.4 has final broad validation evidence.
- Required quality gates pass before closeout, unless a blocker is explicit
  and actionable.
- Documentation-only changes are not over-tested or under-tested.

---

## Day 11: Epic Retrospective Draft

**Title:** Retrospective Draft
**Theme:** Draft the Epic 18 retrospective with outcomes, evidence,
non-claims, and state-of-the-art assessment.
**Time estimate:** 12 hours

### Tasks

1. Draft `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` from Sprint 197
   through Sprint 206 evidence.
2. Summarize completed outcomes, narrowed outcomes, deferred items,
   residuals, validation evidence, and review-comment hardening.
3. Include a state-of-the-art assessment that distinguishes implemented
   capability from unsupported external claims.
4. Add evidence links to sprint retrospectives, artifacts, validation logs,
   and claim-boundary docs.
5. Record retrospective open questions in `WORKING_NOTES.md`.

### Deliverables

- Draft Epic 18 retrospective.
- Evidence-linked outcome summary.
- Non-claim and state-of-the-art assessment draft.
- Day 11 retrospective artifact.

### Completion Criteria

- Item 206.5 has a complete draft retrospective.
- The retrospective does not overstate package, platform, performance, API,
  ABI, release, or state-of-the-art support.
- Every major claim has an evidence link or explicit residual.

---

## Day 12: Residual Queue Draft

**Title:** Residual Queue
**Theme:** Publish a prioritized residual queue with closure targets and
long-horizon deferrals.
**Time estimate:** 12 hours

### Tasks

1. Draft or update `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` from
   Sprint 197 through Sprint 206 residuals.
2. Prioritize residuals by user value, claim risk, implementation cost,
   evidence availability, and maintenance burden.
3. Define exact closure targets for package distribution, Windows proof,
   generated API publication, benchmark claims, ABI/release support, and
   state-of-the-art positioning.
4. Separate next-epic candidates from long-horizon deferrals.
5. Record residual queue rationale in `WORKING_NOTES.md`.

### Deliverables

- Prioritized Epic 18 residual queue.
- Closure-target checklist for each residual.
- Long-horizon deferral list.
- Day 12 residual artifact.

### Completion Criteria

- Item 206.6 has a complete prioritized residual queue.
- Residuals are actionable and not vague placeholders.
- Deferred claims remain visibly unearned until future evidence closes them.

---

## Day 13: Consistency Hardening

**Title:** Consistency Hardening
**Theme:** Re-read final public, maintainer, planning, retrospective, and
residual surfaces for contradictions before closeout.
**Time estimate:** 12 hours

### Tasks

1. Review all changed claim-bearing docs for contradictions, stale pending
   language, broken evidence paths, and inconsistent support vocabulary.
2. Cross-check the project plan, Epic retrospective, residual queue, Sprint
   206 working notes, and daily artifacts against each other.
3. Re-run focused status, routing, manifest, and documentation checks that
   protect the changed surfaces.
4. Fix any wording or evidence mismatches found during hardening.
5. Write the Day 13 consistency hardening artifact.

### Deliverables

- Final consistency review notes.
- Corrected stale or contradictory status wording.
- Re-run focused validation evidence.
- Day 13 hardening artifact.

### Completion Criteria

- No known public, maintainer, or planning document contradicts the final
  Epic 18 status.
- Evidence links are current and point to existing files.
- Claim-boundary wording is consistent across closeout surfaces.

---

## Day 14: Closeout Review

**Title:** Closeout Review
**Theme:** Finalize Sprint 206 and Epic 18 closeout evidence, validation
summary, and handoff notes.
**Time estimate:** 10 hours

### Tasks

1. Review Sprint 206 `WORKING_NOTES.md`, artifacts, final validation logs,
   project-plan updates, retrospective, and residual queue for completeness.
2. Create the Day 14 closeout artifact with final changed surfaces,
   validation commands, outcomes, non-claims, and residual handoff.
3. Confirm Sprint 206 items 206.1 through 206.6 have final dispositions.
4. Prepare the eventual Sprint 206 retrospective inputs and PR summary notes.
5. Run final lightweight checks for links, status wording, and untracked
   generated artifacts.

### Deliverables

- Day 14 closeout review artifact.
- Final Sprint 206 item disposition table.
- Validation and residual handoff summary.
- PR-ready closeout notes.

### Completion Criteria

- Sprint 206 has complete closeout evidence for all six project-plan items.
- Epic 18 final validation, claim calibration, retrospective, and residual
  queue are ready for review.
- No generated, temporary, or unsupported artifact is accidentally staged.
