# Sprint 197 Plan: Epic 18 Final Validation, Claim Calibration & Closeout

**Sprint Duration:** 14 days
**Goal:** Reconcile Epic 18 outcomes, run final validation, calibrate claims,
publish the retrospective and residual queue, and decide whether any stronger
support claims are earned.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the cited Epic 18 final-validation estimate.

**Primary scope:** Reconcile outcomes and evidence from Sprints 197 through
205; recalibrate README, INSTALL, maintainer guide, benchmark docs, API docs,
corpus docs, and planning docs; mark project-plan item status; run integrated
validation appropriate to changed surfaces; publish `EPIC_18_RETROSPECTIVE.md`;
and produce a prioritized next-epic residual queue.

**Non-goals:** New solver algorithms, broad API or ABI expansion, new
package-manager promises, broad platform parity, unqualified Windows support,
portable performance claims, release claims, broad external-library parity, or
state-of-the-art claims not backed by exact Epic 18 evidence.

---

## Day 1: Closeout Intake and Evidence Source Map

**Title:** Closeout Intake
**Theme:** Establish the final-validation scope and identify every evidence
source needed for closeout.
**Time estimate:** 12 hours

### Tasks

1. Re-read the cited Epic 18 final-validation project-plan section and map
   items 206.1 through 206.6 to expected closeout artifacts.
2. Inventory Sprint 197 through Sprint 205 plans, working notes,
   retrospectives, validation logs, generated reports, PR review comments, and
   residual queues.
3. Create `WORKING_NOTES.md` with a sprint checklist, evidence ledger
   scaffold, claim-surface inventory, validation matrix, and risk register.
4. Identify package, Windows, benchmark, comparison, API, review-surface,
   adoption, and reliability evidence that must be reconciled.
5. Record open questions or missing artifacts that could block closeout.

### Deliverables

- Sprint 197 working-notes scaffold.
- Epic 18 evidence-source inventory.
- Initial closeout risk register.
- Validation and claim-surface map.

### Completion Criteria

- Every closeout item has an owner artifact or validation source.
- Evidence sources for Sprints 197 through 205 are listed.
- Missing evidence is recorded before any claim edits begin.

---

## Day 2: Outcome Reconciliation Ledger

**Title:** Outcome Ledger
**Theme:** Reconcile sprint outcomes, decisions, validation records, and
residuals into one closeout ledger.
**Time estimate:** 12 hours

### Tasks

1. Read Sprint 197 through Sprint 205 retrospectives, day artifacts, working
   notes, and PR review resolutions.
2. Classify every planned item as complete, narrowed, deferred,
   residualized, superseded, or not attempted.
3. Record evidence links, command outputs, CI jobs, docs, scripts, and tests
   supporting each status.
4. Separate completed closures from partial work that must not support broader
   claims.
5. Produce the Day 2 evidence reconciliation artifact for item 206.1.

### Deliverables

- Sprint 197-205 outcome ledger.
- Evidence-link table by sprint and topic.
- Completion, deferral, and supersession list.
- Item 206.1 reconciliation notes.

### Completion Criteria

- Every prior sprint outcome has a status and evidence reference.
- Deferred and residual items are visibly distinct from completed work.
- The ledger can drive claim recalibration without additional interpretation.

---

## Day 3: Evidence Conflict and Gap Review

**Title:** Evidence Conflicts
**Theme:** Find conflicts between plans, docs, CI evidence, review feedback,
and final outcomes.
**Time estimate:** 12 hours

### Tasks

1. Compare the Day 2 ledger against the Epic 18 project plan, prior sprint
   plans, retrospectives, and public documentation.
2. Identify any outcome that is contradicted by CI failures, review comments,
   unsupported commands, missing artifacts, or environment-only evidence.
3. Mark evidence as hosted, local, generated, checked-in, optional,
   stale-risk, or human-review-required.
4. Define exact wording constraints for package, Windows, benchmark,
   comparison, API publication, reliability, and support claims.
5. Update working notes with conflict resolutions and open evidence gaps.

### Deliverables

- Evidence conflict matrix.
- Gap and stale-risk list.
- Claim-boundary notes by evidence type.
- Updated item 206.1 acceptance evidence.

### Completion Criteria

- No contradicted or stale evidence remains unclassified.
- Hosted and local evidence are not conflated.
- Claim edits can proceed from explicit evidence boundaries.

---

## Day 4: Public Claim Surface Audit

**Title:** Public Claims Audit
**Theme:** Audit user-facing documentation for overclaims, stale claims,
missing evidence links, and duplicate caveats.
**Time estimate:** 12 hours

### Tasks

1. Search README, INSTALL, support/readiness docs, benchmark docs, cookbook,
   tutorial, examples, and solver-selection docs for claim-sensitive wording.
2. Compare each package, Windows, benchmark, comparison, release,
   support-tier, performance, ABI, and state-of-the-art statement against the
   evidence ledger.
3. Flag claims that are too broad, stale, duplicated, under-linked, or missing
   explicit non-claim boundaries.
4. Identify public docs that should link to central support/readiness truth
   instead of repeating caveats.
5. Draft the public-documentation edit plan for item 206.2.

### Deliverables

- Public claim-surface audit artifact.
- Overclaim and stale-claim table.
- Documentation edit plan.
- Evidence-link requirements for public docs.

### Completion Criteria

- User-facing claim changes are evidence-backed.
- Non-claims remain clear for package, platform, performance, release, ABI,
  and state-of-the-art surfaces.
- Duplicate caveats have a consolidation plan.

---

## Day 5: Maintainer, API, and Planning Claim Audit

**Title:** Maintainer Claims Audit
**Theme:** Audit maintainer-facing and generated-documentation surfaces for
claim drift and ownership ambiguity.
**Time estimate:** 12 hours

### Tasks

1. Review `docs/maintainer_guide.md`, API-reference source inputs, corpus
   docs, generated-report docs, planning docs, and validation-owner docs.
2. Compare maintainer guidance against the Day 2 and Day 3 evidence ledger.
3. Identify ownership statements that need stronger boundaries for package,
   Windows, comparison, benchmark, API publication, and reliability evidence.
4. Flag generated API publication or source-reference claims that remain
   selected, deferred, or conditional.
5. Record maintainer/API edits needed for item 206.2 and status edits needed
   for item 206.3.

### Deliverables

- Maintainer/API claim audit.
- Planning status edit inventory.
- Generated API publication boundary notes.
- Validation-owner update plan.

### Completion Criteria

- Maintainer guidance reflects actual evidence ownership.
- API and corpus documentation do not imply unsupported publication or parity.
- Project-plan status edits are ready to apply.

---

## Day 6: Public Documentation Recalibration

**Title:** Public Recalibration
**Theme:** Update user-facing docs so claims match earned Epic 18 evidence.
**Time estimate:** 12 hours

### Tasks

1. Update README wording for selected package-manager, Windows, benchmark,
   comparison, reliability, support/readiness, and state-of-the-art boundaries.
2. Update INSTALL and support/readiness docs so users can distinguish
   supported, selected, hosted-evidence, local-only, deferred, and unclaimed
   surfaces.
3. Update benchmark/performance wording so any evidence remains
   methodology-bound and platform-specific unless broader data exists.
4. Replace repeated caveats with links to central support/readiness truth
   where that improves maintainability.
5. Record all public claim edits and retained non-claims in working notes.

### Deliverables

- Claim-recalibrated public documentation.
- Updated support/readiness routing.
- Public non-claim and evidence-link log.
- Item 206.2 public-doc evidence.

### Completion Criteria

- Public documentation makes no stronger claim than the evidence supports.
- Users can find active support/readiness truth without reading sprint notes.
- Package, Windows, performance, release, ABI, and state-of-the-art caveats are
  precise and current.

---

## Day 7: Maintainer and API Documentation Recalibration

**Title:** Maintainer Recalibration
**Theme:** Update maintainer-facing docs, API guidance, and planning-adjacent
surfaces with final evidence boundaries.
**Time estimate:** 12 hours

### Tasks

1. Update maintainer guidance with final Epic 18 evidence ownership,
   validation expectations, and residual interpretation.
2. Update API, corpus, generated-report, and benchmark-adjacent docs where
   final evidence changes require wording calibration.
3. Ensure package-manager, Windows, comparison, benchmark, API publication,
   reliability, and review-surface ownership is explicit.
4. Preserve selected-evidence wording where broad claims remain unearned.
5. Record maintainer/API edits and item 206.2 acceptance notes.

### Deliverables

- Updated maintainer and API documentation.
- Evidence-owner routing notes.
- Generated API and corpus claim-boundary updates.
- Day 7 change log.

### Completion Criteria

- Maintainer docs identify the exact gates and artifacts that own final
  evidence.
- API and generated-documentation claims remain calibrated.
- Public and maintainer surfaces agree on support boundaries.

---

## Day 8: Project Plan Status Update

**Title:** Plan Status
**Theme:** Mark Epic 18 project-plan outcomes with evidence links and exact
dispositions.
**Time estimate:** 12 hours

### Tasks

1. Update Epic 18 project-plan status for Sprints 197 through 205 using the
   Day 2 ledger.
2. Mark each item complete, narrowed, deferred, residualized, superseded, or
   explicitly not claimed.
3. Link status notes to validation records, retrospective entries, generated
   reports, docs, scripts, and PR review resolutions.
4. Ensure project-plan wording does not convert partial work into a completed
   claim.
5. Record item 206.3 completion evidence in working notes.

### Deliverables

- Updated Epic 18 project-plan status notes.
- Evidence-linked item disposition table.
- Supersession and residualization record.
- Item 206.3 acceptance notes.

### Completion Criteria

- Every Sprint 197-205 planned item has a final disposition.
- Status notes point at evidence or explicit deferral records.
- Partial outcomes cannot be mistaken for completed closures.

---

## Day 9: Focused Validation Planning

**Title:** Gate Plan
**Theme:** Define the final integrated validation matrix before running costly
or broad checks.
**Time estimate:** 12 hours

### Tasks

1. Convert changed files and evidence owners into a validation command matrix.
2. Identify focused gates for docs, claim boundaries, package/install checks,
   Windows ownership, generated reports, benchmarks, API docs, and changed C
   surfaces.
3. Decide when full `make format`, `make lint`, and `make test` are required
   based on changed `.c` and `.h` files.
4. Record expected outputs, environment limitations, and hosted-only evidence
   that cannot be reproduced locally.
5. Prepare a validation log template for item 206.4.

### Deliverables

- Integrated validation matrix.
- Focused-gate command list.
- Full-gate trigger decision record.
- Validation log template.

### Completion Criteria

- Every changed surface has an appropriate validation owner.
- Full C gate requirements are explicit and traceable.
- Environment residuals are recorded before validation execution.

---

## Day 10: Focused Validation Execution

**Title:** Focused Gates
**Theme:** Run focused validation gates and fix documentation or tooling
issues uncovered by those checks.
**Time estimate:** 12 hours

### Tasks

1. Run docs checks, claim guards, install-doc checks, API coverage checks,
   report-index checks, freshness guards, and selected ownership checks
   identified on Day 9.
2. Capture command outputs, pass/fail status, and environment notes in the
   validation log.
3. Fix any documentation, manifest, report, or guard issues that have clear
   resolutions.
4. Re-run failed focused checks until they pass or are recorded as blocked by
   unavailable environment evidence.
5. Update working notes with item 206.4 focused-validation evidence.

### Deliverables

- Focused validation log.
- Fixed focused-gate issues.
- Environment residual notes.
- Updated item 206.4 evidence.

### Completion Criteria

- Focused checks pass or have explicit environment residuals.
- Any fix made during validation is rechecked.
- Validation evidence is specific enough for the retrospective.

---

## Day 11: Full Quality Gate Execution

**Title:** Full Gates
**Theme:** Run the full quality gates required by changed surfaces and record
results for closeout.
**Time estimate:** 12 hours

### Tasks

1. Run `make format`, `make lint`, and `make test` if any `.c` or `.h` files
   changed during the sprint.
2. Run the full documentation and planning checks even if the sprint remains
   documentation-only.
3. Run `git diff --check` and verify generated or ignored artifacts have not
   created tracked noise.
4. Fix any clear failures and re-run the affected gates.
5. Record final integrated-validation evidence for item 206.4.

### Deliverables

- Full quality-gate log.
- Final validation command list with results.
- Any required fixes from gate failures.
- Clean tracked-worktree verification notes.

### Completion Criteria

- Required full gates pass before closeout proceeds.
- Validation commands match the final changed surfaces.
- No tracked generated artifacts remain accidentally modified.

---

## Day 12: Epic Retrospective Draft

**Title:** Retrospective Draft
**Theme:** Draft the Epic 18 retrospective from evidence, outcomes,
non-claims, and residuals.
**Time estimate:** 12 hours

### Tasks

1. Create `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` with outcome,
   evidence, validation, non-claim, and residual sections.
2. Summarize completed closures for package, Windows, benchmark, comparison,
   API, reliability, review-surface, adoption, and closeout work.
3. Record unresolved or deferred gaps without implying they were completed.
4. Include a state-of-the-art assessment calibrated to actual Epic 18
   evidence.
5. Cross-link the retrospective to project-plan status notes and residual
   queue drafts.

### Deliverables

- Draft Epic 18 retrospective.
- State-of-the-art assessment.
- Outcome and non-claim summary.
- Retrospective evidence-link table.

### Completion Criteria

- Retrospective claims are backed by the evidence ledger.
- Residuals and non-claims are explicit.
- The retrospective can be reviewed against validation logs.

---

## Day 13: Residual Queue and Claim Decision

**Title:** Residual Queue
**Theme:** Publish the prioritized next-epic queue and decide whether any
stronger claims are earned.
**Time estimate:** 12 hours

### Tasks

1. Create the prioritized next-epic residual queue with exact closure targets,
   evidence requirements, owner surfaces, and long-horizon deferrals.
2. Rank residuals by user value, claim risk, validation feasibility, and
   ability to fully close within a sprint.
3. Decide whether any stronger support, package, Windows, benchmark,
   comparison, API publication, reliability, release, or state-of-the-art
   claims are earned.
4. Update retrospective, project-plan status, and working notes with claim
   decisions.
5. Prepare the final review checklist for Day 14.

### Deliverables

- Prioritized residual queue.
- Long-horizon deferral list.
- Final claim decision table.
- Day 14 review checklist.

### Completion Criteria

- Residuals have concrete closure criteria.
- Claim decisions are explicit and evidence-linked.
- Long-horizon work is not presented as near-term completion.

---

## Day 14: Final Closeout Review

**Title:** Closeout Review
**Theme:** Perform final coherence review, validation confirmation, and
handoff preparation.
**Time estimate:** 10 hours

### Tasks

1. Review all Sprint 197 closeout artifacts for internal consistency,
   evidence links, and claim calibration.
2. Verify the project plan, retrospective, residual queue, public docs,
   maintainer docs, and validation logs agree.
3. Re-run final lightweight checks needed after Day 13 edits.
4. Update `WORKING_NOTES.md` with final status, validation summary, known
   residuals, and handoff notes.
5. Prepare commit and PR summary text with changed files, validation evidence,
   non-claims, and residuals.

### Deliverables

- Final closeout artifact review.
- Updated working notes with final validation and handoff.
- Commit-ready Sprint 197 changes.
- PR summary inputs.

### Completion Criteria

- All sprint deliverables are present and internally consistent.
- Required validation is complete and recorded.
- Epic 18 final claims are calibrated to exact evidence.
- Remaining work is captured in a prioritized residual queue.

