# Sprint 117 Plan: Final Integration, Competitive Calibration & Epic 10 Closeout

**Sprint Duration:** 14 days
**Goal:** Integrate Epic 10 outcomes, validate all reviewed surfaces, compare
final evidence against the state-of-the-art target, and close the epic with
truthful earned claims and residuals.

**Starting Point:** Sprint 117 begins from:
- completed Sprint 100-116 implementation, validation, documentation, package,
  platform, maintainability, comparison, and adoption QA artifacts
- Sprint 100 state-of-the-art target and evidence contract
- Sprint 114 proof-owner, source-boundary, direct/iterative oracle, and SVD
  helper residual decisions
- Sprint 115 package/platform, install, Windows, macOS, ABI, and
  package-manager residual decisions
- Sprint 116 adoption-surface claim guardrails and non-claims checklist

The sprint must:
- compare final implementation evidence against the Epic 10 target
- run the strongest reviewed validation appropriate to touched surfaces
- regenerate final solver, reorder, benchmark, coverage, and package evidence
- remove, downgrade, or fence any unsupported public/support wording
- publish the post-Epic residual queue and explicit non-claims
- write the Sprint 117 retrospective
- write the Epic 10 retrospective and post-epic handoff queue

**End State:** Sprint 117 leaves behind:
- final Epic 10 validation package
- competitive calibration against the state-of-the-art target
- unsupported-claim cleanup evidence
- Sprint 117 retrospective
- Epic 10 retrospective
- post-Epic residual queue and non-claims

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `164` hours, matching the Sprint 117 project-plan estimate.

---

## Day 1: Final Integration Intake and Evidence Map

**Title:** Closeout Intake
**Theme:** Establish Epic 10 closeout scope, artifact map, and validation lanes
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 117 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 100 state-of-the-art target and evidence-contract artifacts.
3. Inventory Sprint 100-116 retrospectives, working notes, and final handoff
   artifacts that feed Epic 10 closeout.
4. Build an evidence map for implementation, benchmarks, package/platform
   support, maintainability, docs/adoption, and residual proof-owner decisions.
5. Identify validation commands required by likely touched surfaces.
6. Create Sprint 117 working notes and artifact directory.
7. Write the final integration intake artifact.

### Deliverables
- Sprint 117 working-notes baseline
- artifact directory
- Epic 10 evidence map
- validation-lane inventory
- day-level owner map

### Completion Criteria
- all Sprint 117 project-plan items have day-level owners
- Sprint 100 target and Sprint 114-116 residual decisions are available as
  closeout guardrails
- no unsupported implementation, package/platform, or claim work is silently
  pulled into Sprint 117

---

## Day 2: End-State Claim Audit Inventory

**Title:** Claim Inventory
**Theme:** Compare final Epic 10 evidence against the Sprint 100 target map
**Time estimate:** 12 hours

### Tasks
1. Extract Sprint 100 state-of-the-art target claims and evidence criteria.
2. Map each target claim to final Epic 10 artifacts, tests, benchmarks,
   package/platform evidence, and public documentation.
3. Classify claims as earned, partially earned, unsupported, deferred, or
   non-claim.
4. Identify claims requiring final validation before they can remain public.
5. Write the end-state claim inventory artifact.

### Deliverables
- state-of-the-art claim inventory
- evidence mapping table
- unsupported or partially-supported claim candidates
- Day 3 audit checklist

### Completion Criteria
- every Sprint 100 target claim has an owner and evidence disposition
- unsupported or partially-supported claims are visible before cleanup
- no public claim is accepted without an evidence source or explicit non-claim

---

## Day 3: End-State Claim Audit Decision

**Title:** Claim Decision
**Theme:** Finalize earned claims, downgrades, and non-claims before edits
**Time estimate:** 12 hours

### Tasks
1. Review the Day 2 claim inventory.
2. Decide which claims are earned by final evidence.
3. Decide which claims must be downgraded, fenced, or removed.
4. Align decisions with Sprint 114, Sprint 115, and Sprint 116 non-claims.
5. Cross-check claim decisions against public README, install, benchmark,
   solver-selection, Matrix Market, algorithm, and example docs.
6. Write the end-state claim decision artifact.

### Deliverables
- earned-claim decision table
- unsupported-claim cleanup candidates
- explicit non-claim list
- Day 8 cleanup checklist

### Completion Criteria
- Item 1 is complete
- Day 8 can apply cleanup without rediscovering claim evidence
- final public claims are either earned, downgraded, or explicitly non-claims

---

## Day 4: Full Validation Design

**Title:** Validation Design
**Theme:** Define strongest reviewed and supplemental validation lanes
**Time estimate:** 12 hours

### Tasks
1. Review Makefile, CMake, workflow, install, benchmark, and documentation
   validation surfaces.
2. Decide the reviewed quality baseline required for Sprint 117 closeout.
3. Decide supplemental evidence commands for package, platform, benchmark,
   source-list, and coverage surfaces.
4. Record expected outputs, known exclusions, and timeout/risk handling.
5. Write the full validation design artifact.

### Deliverables
- validation command matrix
- reviewed versus supplemental lane map
- expected-output and exclusion notes
- Day 5 execution checklist

### Completion Criteria
- all required validation commands are known before execution
- command choices match touched surfaces and Epic 10 closeout needs
- staged exclusions and supplemental lanes are not mistaken for reviewed proof

---

## Day 5: Full Validation Execution

**Title:** Validation Run
**Theme:** Execute strongest reviewed validation and capture evidence
**Time estimate:** 12 hours

### Tasks
1. Run the reviewed quality baseline from Day 4.
2. Run CMake parity, source-list, install/export, benchmark, coverage, or
   supplemental commands selected by the validation design.
3. Capture command outputs, failures, skips, and environment notes.
4. Stop and investigate any failed required lane.
5. Write the validation execution artifact.

### Deliverables
- validation execution log
- command-output summary
- pass/fail/skip table
- issue list if any required lane fails

### Completion Criteria
- required reviewed lanes pass or a blocker is explicitly identified
- supplemental lanes are recorded without widening support claims
- Day 6 can package final evidence without rerunning discovery

---

## Day 6: Final Validation Package

**Title:** Validation Package
**Theme:** Assemble final Epic 10 validation evidence and changed-surface proof
**Time estimate:** 12 hours

### Tasks
1. Summarize Day 5 validation results into a final evidence package.
2. Record changed source/header, build metadata, workflow, package, docs, and
   benchmark surfaces.
3. Confirm required validation is complete for touched files.
4. Record any residual validation risk or skipped supplemental lane.
5. Link validation evidence back to the Day 4 command matrix and Day 5
   execution log.
6. Prepare retrospective-ready validation metrics.
7. Write the final validation package artifact.

### Deliverables
- final validation package
- changed-surface matrix
- validation residual list
- Item 2 closeout notes

### Completion Criteria
- Item 2 is complete
- validation evidence can be cited by Sprint and Epic retrospectives
- no required quality check remains unrun or unexplained

---

## Day 7: Final Comparison Package Inventory

**Title:** Comparison Inventory
**Theme:** Regenerate and classify final solver, reorder, benchmark, coverage, and package evidence
**Time estimate:** 12 hours

### Tasks
1. Identify final solver, reorder, benchmark, coverage, and package artifacts
   required for Epic 10 closeout.
2. Regenerate selected artifacts or record current final artifacts where
   regeneration is not needed.
3. Classify each artifact as public claim evidence, local measurement context,
   supplemental proof, or residual background.
4. Capture artifact paths and command provenance.
5. Write the final comparison inventory artifact.

### Deliverables
- final comparison artifact inventory
- command provenance table
- public-claim versus local-evidence classification
- Day 8 package checklist

### Completion Criteria
- all final comparison surfaces have artifact owners
- regenerated evidence is classified before public claim cleanup
- no local benchmark artifact is treated as portable performance proof

---

## Day 8: Final Comparison Package And Claim Cleanup

**Title:** Comparison Cleanup
**Theme:** Package final competitive calibration and apply unsupported-claim cleanup
**Time estimate:** 12 hours

### Tasks
1. Assemble final comparison package from Day 7 artifacts.
2. Apply unsupported-claim cleanup identified on Days 2-3 and Day 7.
3. Remove, downgrade, or fence unsupported public/support wording.
4. Preserve useful local evidence without broadening public claims.
5. Write the final comparison and cleanup artifact.

### Deliverables
- final comparison package
- unsupported-claim cleanup artifact
- edited public/support docs if required
- evidence-bounded claim table

### Completion Criteria
- Items 3 and 4 are complete
- public/support wording matches final evidence
- final comparison package is ready for retrospectives

---

## Day 9: Residual Queue Intake

**Title:** Residual Intake
**Theme:** Build the post-Epic residual queue and explicit non-claims
**Time estimate:** 12 hours

### Tasks
1. Re-read Sprint 114 proof-owner and source-boundary residual decisions.
2. Re-read Sprint 115 package/platform residual decisions.
3. Re-read Sprint 116 adoption non-claims and closeout handoff.
4. Classify residuals as promoted, deferred, post-Epic, or explicitly closed.
5. Write the residual queue intake artifact.

### Deliverables
- residual intake table
- Sprint 114 residual disposition
- Sprint 115 residual disposition
- Sprint 116 adoption residual disposition
- Day 10 residual publication checklist

### Completion Criteria
- all named residuals from Sprint 114-116 have a disposition
- no deferred item is silently dropped
- no residual is promoted without validation and public claim cleanup

---

## Day 10: Residual Queue And Non-Claims Publication

**Title:** Residual Publication
**Theme:** Publish final post-Epic residual queue and explicit non-claims
**Time estimate:** 12 hours

### Tasks
1. Publish the post-Epic residual queue.
2. Publish explicit non-claims for deferred package/platform, proof-owner,
   source-boundary, oracle, SVD helper, ABI, package-manager, and platform
   parity work.
3. Link residuals to final validation and claim-cleanup evidence.
4. Cross-check residuals against Sprint 114, Sprint 115, and Sprint 116
   retrospective deferred-debt sections.
5. Mark residuals as post-Epic, future-epic candidate, optional scanability
   work, or consciously closed.
6. Write the residual queue and non-claims artifact.

### Deliverables
- post-Epic residual queue
- explicit non-claims list
- residual owner and dependency notes
- Item 5 closeout notes

### Completion Criteria
- Item 5 is complete
- future work can start from an explicit residual queue
- Epic 10 closes without implying deferred support or implementation claims

---

## Day 11: Sprint 117 Retrospective Draft

**Title:** Sprint Retro Draft
**Theme:** Draft Sprint 117 retrospective from artifacts and working notes
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 117 artifacts and working notes.
2. Draft definition-of-done checklist, what went well, what did not go well,
   final metrics, residual deferred debt, and key deliverables.
3. Include validation, comparison, claim cleanup, and residual queue outcomes.
4. Identify gaps to close before finalizing the retrospective.
5. Write the Sprint 117 retrospective draft artifact or draft file.

### Deliverables
- Sprint 117 retrospective draft
- metric and deliverable inventory
- gap list for Day 12
- Item 6 draft closeout notes

### Completion Criteria
- retrospective content exists before final validation
- metrics and residuals are grounded in Sprint 117 artifacts
- Day 12 can finalize without rediscovering sprint evidence

---

## Day 12: Sprint 117 Retrospective Finalization

**Title:** Sprint Retro Final
**Theme:** Finalize Sprint 117 retrospective and sprint closeout evidence
**Time estimate:** 8 hours

### Tasks
1. Resolve Day 11 retrospective gaps.
2. Finalize Sprint 117 retrospective.
3. Re-run focused documentation hygiene for touched retrospective and artifacts.
4. Confirm Item 6 is complete.
5. Prepare Epic 10 retrospective source inventory for Day 13.

### Deliverables
- finalized Sprint 117 retrospective
- sprint closeout metrics
- validation notes for retrospective
- Epic retrospective source inventory

### Completion Criteria
- Item 6 is complete
- Sprint 117 retrospective is ready for PR review
- Epic 10 retrospective inputs are ready

---

## Day 13: Epic 10 Retrospective Draft

**Title:** Epic Retro Draft
**Theme:** Draft Epic 10 earned claims, lessons, metrics, and carry-forward work
**Time estimate:** 12 hours

### Tasks
1. Review Epic 10 project plan, sprint plans, sprint retrospectives, and key
   final artifacts.
2. Draft earned claims, unearned/non-claims, final metrics, validation
   evidence, lessons learned, and post-epic residual queue.
3. Compare final evidence against the Sprint 100 state-of-the-art target.
4. Write the Epic 10 retrospective draft artifact or draft file.

### Deliverables
- Epic 10 retrospective draft
- earned-claim table
- unearned/non-claim table
- post-epic carry-forward queue
- Day 14 finalization checklist

### Completion Criteria
- Epic retrospective content exists before final day validation
- earned and unearned claims are evidence-bounded
- post-epic residual queue matches Day 10 publication

---

## Day 14: Epic 10 Retrospective Finalization And Handoff

**Title:** Epic Closeout
**Theme:** Finalize Epic 10 retrospective, validation, and post-epic handoff
**Time estimate:** 12 hours

### Tasks
1. Finalize the Epic 10 retrospective.
2. Reconcile Sprint 117 retrospective, Epic 10 retrospective, final validation
   package, comparison package, and residual queue.
3. Run final required documentation and code-quality checks for touched
   surfaces.
4. Capture final changed-surface summary and validation evidence.
5. Write the final closeout handoff artifact.

### Deliverables
- finalized Epic 10 retrospective
- final validation evidence
- post-epic handoff queue
- Sprint 117 closeout artifact
- Item 7 closeout notes

### Completion Criteria
- Item 7 is complete
- required checks pass before closeout
- Epic 10 closes with truthful earned claims and explicit residual non-claims
