# Sprint 99 Plan: Final Integration, Competitive Calibration & Epic 9 Closeout

**Sprint Duration:** 14 days
**Goal:** Re-audit the live repository against the Epic 9 target, run the final
competitive and validation evidence package, land only a justified bounded fix
batch, and close Epic 9 from a validated baseline. This sprint implements the
Sprint 99 section of `docs/planning/EPIC_9/PROJECT_PLAN.md`.

**Starting Point:** Sprint 99 begins from:
- the Sprint 90 target-state and claim fence
- Sprints 91-98 landed and validated
- a repository with stronger capability, workflow, build, maintainability,
  packaging, and assurance surfaces than the Epic 9 opening baseline
- a remaining need to prove that the live end state matches the Epic 9 closeout
  story without overstating competitive, package, platform, or performance
  claims

The strongest Sprint 99 pressure is not to start a new implementation wave. It
is to close Epic 9 from evidence by:
- re-reading the original Epic 9 contradiction classes against the live tree
- executing the final correctness, runtime, package, usability, and workflow
  comparison sweep
- deciding whether any final fix batch is justified by live evidence
- landing only bounded fixes that resolve real closeout contradictions
- separating carry-forward work from deliberate non-claims
- rebuilding the strongest reviewed validation and reporting baseline
- producing the Sprint 99, Epic 9, and handoff closeout package

**End State:** Sprint 99 leaves behind:
- a final Epic 9 end-state contradiction map
- a final competitive calibration evidence package
- one bounded final fix batch if the evidence requires it
- a validated final baseline
- an explicit post-Epic-9 residual queue
- a Sprint 99 retrospective, Epic 9 retrospective, and handoff package

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `164` hours, matching the Sprint 99 project-plan estimate.

---

## Day 1: Sprint 99 Scope & Closeout Baseline

**Title:** Closeout Baseline
**Theme:** Turn the Sprint 99 project-plan section and prior Epic 9 evidence
into one bounded final-closeout execution package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 99 section of
   `docs/planning/EPIC_9/PROJECT_PLAN.md`.
2. Re-read the Sprint 90 claim fence and Sprints 91-98 retrospectives,
   closeout artifacts, and residual queues.
3. Inventory final closeout workstreams:
   - end-state re-audit
   - competitive comparison sweep
   - final fix decision
   - bounded fix batch
   - residual queue finalization
   - validation and reporting
   - Sprint/Epic closeout package
4. Open Sprint 99 working notes and record validation expectations for docs,
   scripts, workflows, benchmark/reporting, and code-touch days.
5. Write the Day 1 baseline artifact with the closeout landing order.

### Deliverables
- Sprint 99 scope inventory
- final closeout workstream map
- Sprint 99 working-notes baseline
- validation expectations for each workstream

### Completion Criteria
- Sprint 99 starts from the merged Sprint 98 end state
- closeout work is bounded before audit work begins
- validation requirements are visible before any final edits

---

## Day 2: Epic 9 Contradiction Class Re-audit

**Title:** Contradiction Re-audit
**Theme:** Re-read the original Epic 9 contradiction classes against the live
repository and separate resolved, residual, and non-claim surfaces
**Time estimate:** 12 hours

### Tasks
1. Reconstruct the original Epic 9 contradiction classes from planning and
   early sprint artifacts.
2. Compare each contradiction class against the live tree.
3. Mark each class as resolved, partially resolved, still active, or deliberate
   non-claim.
4. Identify any closeout contradictions that would justify a final bounded fix
   batch.
5. Write the Day 2 end-state re-audit artifact.

### Deliverables
- final Epic 9 contradiction class map
- resolved vs residual vs non-claim classification
- candidate final-fix queue

### Completion Criteria
- every original contradiction class has a live-tree status
- unresolved items are evidence-backed, not inferred from stale plans
- non-claims are explicit enough to protect closeout language

---

## Day 3: Competitive Comparison Scope Freeze

**Title:** Comparison Scope
**Theme:** Freeze the final correctness, runtime, package, and usability
comparison surfaces before running the evidence sweep
**Time estimate:** 12 hours

### Tasks
1. Review maintained comparison and assurance surfaces from Sprints 90, 94,
   97, and 98.
2. Select the final comparison lanes for:
   - correctness evidence
   - runtime and fill evidence
   - package and install/export evidence
   - usability and documentation evidence
   - workflow and CI evidence
3. Define exact commands, artifacts, and pass/fail expectations for each lane.
4. Identify comparison language that is allowed, disallowed, or deferred.
5. Write the Day 3 comparison-scope artifact.

### Deliverables
- final comparison lane list
- command and artifact checklist
- allowed/disallowed competitive-language table

### Completion Criteria
- no comparison sweep begins before scope is frozen
- each lane has an explicit proof owner
- competitive claims remain tied to maintained evidence

---

## Day 4: Final Correctness & Runtime Evidence Sweep

**Title:** Evidence Sweep 1
**Theme:** Execute the final correctness, runtime, and fill comparison package
against maintained proof surfaces
**Time estimate:** 12 hours

### Tasks
1. Run the selected correctness proof commands from Day 3.
2. Run the selected runtime and fill comparison commands from Day 3.
3. Capture outputs, notable deltas, skipped lanes, and environment assumptions.
4. Classify each result as closeout-ready, residual, or fix-candidate.
5. Write the Day 4 correctness/runtime evidence artifact.

### Deliverables
- final correctness evidence notes
- final runtime/fill evidence notes
- fix-candidate list from comparison results

### Completion Criteria
- comparison outputs are captured with enough context for retrospective use
- skipped or unavailable lanes are explicitly explained
- no performance or fill claim exceeds the evidence collected

---

## Day 5: Package, Usability & Workflow Evidence Sweep

**Title:** Evidence Sweep 2
**Theme:** Execute the final package, install/export, documentation, usability,
and workflow coherence checks
**Time estimate:** 12 hours

### Tasks
1. Run selected package and install/export proof commands from Day 3.
2. Recheck public narrative, examples, maintainer guide, and workflow docs
   against the current product contract.
3. Inspect CI and local reviewed validation surfaces for stale counts, labels,
   or claim text.
4. Capture any package, usability, or workflow contradictions.
5. Write the Day 5 package/usability/workflow evidence artifact.

### Deliverables
- final package and install/export evidence notes
- usability and documentation coherence notes
- workflow proof-surface contradiction list

### Completion Criteria
- package and platform claims match the validated proof surface
- user-facing docs do not imply unsupported parity
- workflow assertions are current or listed for final-fix consideration

---

## Day 6: Final Fix Decision

**Title:** Fix Decision
**Theme:** Decide whether a last bounded implementation/support batch is
necessary and define its exact boundary
**Time estimate:** 12 hours

### Tasks
1. Review the Day 2, Day 4, and Day 5 fix-candidate lists.
2. Reject candidates that are broad improvements, speculative polish, or
   post-Epic-9 carry-forward work.
3. Select only contradictions that block truthful Epic 9 closeout.
4. Define the final fix batch boundary, touched surfaces, validation commands,
   and rollback notes.
5. Write the Day 6 final-fix decision artifact.

### Deliverables
- final fix/no-fix decision
- bounded fix batch plan if needed
- deferred residual queue draft
- validation and rollback checklist

### Completion Criteria
- every accepted fix is tied to a live closeout contradiction
- every rejected candidate has a residual or non-claim classification
- implementation does not begin without a written boundary

---

## Day 7: Final Bounded Fix Batch 1

**Title:** Fix Batch 1
**Theme:** Land the first half of the final bounded implementation or support
fixes if Day 6 selected a batch
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-priority fixes from the Day 6 boundary.
2. Keep edits scoped to the selected code, docs, script, workflow, or artifact
   surfaces.
3. Run focused checks for each touched surface during development.
4. Update working notes with implementation decisions and any new risks.
5. Stop and reclassify if a selected fix expands beyond the written boundary.

### Deliverables
- first final-fix implementation batch or no-op evidence if no fixes were
  selected
- focused validation notes
- updated risk and residual list

### Completion Criteria
- batch work stays inside the Day 6 boundary
- focused checks pass for touched surfaces
- no new unsupported Epic 9 claim is introduced

---

## Day 8: Final Bounded Fix Batch 2

**Title:** Fix Batch 2
**Theme:** Complete the final bounded fix batch and reconcile adjacent proof
owners
**Time estimate:** 12 hours

### Tasks
1. Finish remaining Day 6 fixes or document why the batch closed early.
2. Reconcile adjacent docs, scripts, workflows, examples, and tests that own
   the same closeout claim.
3. Run the targeted validation commands from the Day 6 checklist.
4. Confirm fixed contradictions are no longer present in the live tree.
5. Write the Day 8 final-fix closeout artifact.

### Deliverables
- completed final-fix batch
- updated proof-owner surfaces
- targeted validation results
- final-fix closeout notes

### Completion Criteria
- accepted fix candidates are resolved or explicitly reclassified
- adjacent proof owners agree with the final product contract
- targeted validation passes before broader validation begins

---

## Day 9: Residual Queue Classification

**Title:** Residual Queue
**Theme:** Separate real carry-forward work from deliberate non-claims and
closeout-safe deferrals
**Time estimate:** 12 hours

### Tasks
1. Review residual items from Sprints 90-98 and the Sprint 99 evidence sweep.
2. Classify each item as:
   - post-Epic-9 carry-forward
   - deliberate non-claim
   - unsupported claim to remove
   - already resolved
3. Remove duplicates and stale residual entries.
4. Define owner, rationale, and validation expectation for each carry-forward
   item.
5. Write the Day 9 residual queue artifact.

### Deliverables
- final post-Epic-9 residual queue
- deliberate non-claim list
- stale/resolved residual cleanup notes

### Completion Criteria
- residual work is explicit and non-duplicative
- non-claims are strong enough to prevent accidental promise creep
- no unresolved closeout blocker is hidden in the residual queue

---

## Day 10: Full Reviewed Validation Sweep

**Title:** Reviewed Validation
**Theme:** Rebuild the strongest reviewed local baseline after all final-fix
and residual decisions
**Time estimate:** 12 hours

### Tasks
1. Run the strongest reviewed validation baseline selected on Day 1.
2. Run implementation-day checks if code, header, workflow, or script files
   were modified.
3. Capture failures, skips, timing notes, and environment assumptions.
4. If validation fails, stop and triage before proceeding to closeout writing.
5. Write the Day 10 reviewed-validation artifact.

### Deliverables
- reviewed validation command log
- failure/skip/environment notes
- go/no-go decision for closeout package writing

### Completion Criteria
- strongest reviewed baseline passes or the sprint stops for triage
- implementation-day quality checks pass when required
- closeout writing starts only from a validated tree

---

## Day 11: Install, Export, Example & Reporting Validation

**Title:** Surface Validation
**Theme:** Validate install/export, example, consumer, and reporting surfaces
that support the final Epic 9 product story
**Time estimate:** 11 hours

### Tasks
1. Run install/export and CMake consumer proof commands selected on Day 3.
2. Run representative example and smoke-test surfaces selected on Day 3.
3. Run selected benchmark/reporting generation commands without overstating
   runtime conclusions.
4. Capture output locations and any skipped platform-specific lanes.
5. Write the Day 11 surface-validation artifact.

### Deliverables
- install/export validation notes
- example and consumer proof notes
- benchmark/reporting validation notes

### Completion Criteria
- package and consumer surfaces match final documentation claims
- reporting outputs are reproducible enough for closeout evidence
- any unvalidated platform lane is documented as a non-claim or residual

---

## Day 12: Final Closeout Evidence Package

**Title:** Evidence Package
**Theme:** Consolidate audit, comparison, fix, residual, and validation
evidence into one final Epic 9 closeout package
**Time estimate:** 11 hours

### Tasks
1. Collect Day 1-11 artifacts into a single closeout evidence index.
2. Summarize resolved Epic 9 contradiction classes.
3. Summarize final competitive calibration results and claim limits.
4. Summarize final validation results and known residuals.
5. Write the Day 12 closeout evidence package.

### Deliverables
- final closeout evidence index
- resolved contradiction summary
- competitive calibration summary
- validation and residual summary

### Completion Criteria
- closeout package cites evidence instead of aspirational status
- claim limits are visible next to supporting evidence
- residuals are carried forward without weakening the validated baseline

---

## Day 13: Sprint 99 & Epic 9 Retrospective Drafts

**Title:** Retrospective Drafts
**Theme:** Draft the Sprint 99 retrospective and Epic 9 retrospective from the
validated evidence package
**Time estimate:** 11 hours

### Tasks
1. Draft the Sprint 99 retrospective from working notes and Day 1-12 artifacts.
2. Draft the Epic 9 retrospective from the final contradiction map, evidence
   package, and residual queue.
3. Identify lessons that affect future epics without reopening Sprint 99 scope.
4. Cross-check retrospective claims against validation logs and artifacts.
5. Record final closeout edits needed for Day 14.

### Deliverables
- Sprint 99 retrospective draft
- Epic 9 retrospective draft
- lessons-learned and handoff notes
- Day 14 closeout edit list

### Completion Criteria
- retrospective drafts are grounded in artifacts and validation results
- lessons distinguish process findings from product claims
- remaining closeout edits are editorial or explicitly bounded

---

## Day 14: Epic 9 Closeout & Handoff

**Title:** Closeout Handoff
**Theme:** Finalize the Sprint 99 retrospective, Epic 9 retrospective, and
post-Epic-9 handoff package from the validated baseline
**Time estimate:** 11 hours

### Tasks
1. Finalize Sprint 99 retrospective and Epic 9 retrospective documents.
2. Finalize the post-Epic-9 residual queue and handoff notes.
3. Re-run lightweight hygiene checks for touched documentation and artifact
   surfaces.
4. Confirm working notes, artifacts, retrospectives, and plan references are
   internally consistent.
5. Prepare the sprint closeout commit and pull request description.

### Deliverables
- finalized Sprint 99 retrospective
- finalized Epic 9 retrospective
- final post-Epic-9 residual queue
- pull request closeout summary

### Completion Criteria
- Epic 9 closes from a validated and documented baseline
- Sprint 99 deliverables are complete and internally consistent
- handoff items are explicit enough for the next planning cycle
