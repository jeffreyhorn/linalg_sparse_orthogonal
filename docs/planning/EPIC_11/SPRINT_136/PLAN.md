# Sprint 136 Plan: Final Integration, Competitive Recalibration & Epic 11 Closeout

**Sprint Duration:** 14 days
**Goal:** Validate Epic 11 outcomes, compare them against the
state-of-the-art target, publish earned claims and non-claims, and close the
epic with a retrospective and post-epic handoff queue.

**Starting Point:** Sprint 136 begins from:
- Sprint 118-135 artifacts, validation packages, residual queues, and
  closeout handoffs
- Sprint 131 generated report-index and freshness decisions
- Sprint 133 static-first package, ABI, shared-library, and package-manager
  product decisions
- Sprint 134 Linux, macOS, Windows, staged-test, and supplemental
  platform-tier decisions
- Sprint 135 adoption, cookbook, algorithm-reference, historical-appendix, and
  report-index documentation productization
- current source, test, benchmark, package, install, CI, documentation, and
  maintainer-guide surfaces
- the end-of-epic deferred QR residual queue in
  `docs/planning/EPIC_11/PROJECT_PLAN.md`

The sprint must:
- inventory final Epic 11 evidence, ownership changes, validation artifacts,
  package/platform proof, adoption docs, and residuals
- design and execute reviewed and supplemental validation without silently
  widening support or performance claims
- compare the final evidence against Epic 11 goals and state-of-the-art
  non-claims
- remove, downgrade, or fence unsupported public/support wording
- publish the post-Epic-11 residual queue, including the Sprint 128 deferred
  QR residual queue with promotion criteria
- write Sprint 136 and Epic 11 retrospectives
- publish the final Epic 11 closeout handoff

**End State:** Sprint 136 leaves behind:
- final evidence inventory
- full validation design and execution package
- competitive claim recalibration package
- unsupported-claim cleanup evidence
- post-Epic-11 residual queue
- Sprint 136 retrospective
- Epic 11 retrospective
- final closeout and post-epic handoff artifact

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `164` hours, matching the Sprint 136 project-plan estimate.

---

## Day 1: Epic Closeout Intake

**Title:** Closeout Intake
**Theme:** Establish Sprint 136 scope, inherited evidence, artifact structure,
and final claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 136 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Review Sprint 118-135 closeout artifacts, retrospectives, and residual
   queues.
3. Create Sprint 136 working notes and artifact directory.
4. Map Sprint 136 Items 1-7 to day-level owners.
5. Record inherited claim fences for source/test ownership, oracle evidence,
   report indexes, package/ABI, platform support, adoption docs, benchmarks,
   competitive positioning, and QR residuals.
6. Write the closeout intake artifact.

### Deliverables
- Sprint 136 working-notes baseline
- artifact directory structure
- inherited-input inventory
- item-to-day owner map
- final claim-boundary register

### Completion Criteria
- every Sprint 136 project-plan item has a day-level owner
- Sprint 118-135 evidence and residual queues are visible before validation
  and comparison begin
- final package, platform, performance, and competitive non-claims are stated
  before public wording is touched

---

## Day 2: Final Evidence Inventory

**Title:** Evidence Inventory
**Theme:** Inventory final Epic 11 source, test, oracle, performance, package,
platform, documentation, and residual evidence
**Time estimate:** 12 hours

### Tasks
1. Inventory source/test ownership changes across Epic 11.
2. Inventory oracle and external-reference evidence from Sprints 118-135.
3. Inventory performance sentinel, canonical benchmark, and large-matrix
   guardrail evidence.
4. Inventory package, install, ABI, and platform support evidence.
5. Inventory adoption, cookbook, algorithm-reference, history, and maintainer
   documentation changes.
6. Write the final evidence inventory artifact.

### Deliverables
- final evidence inventory
- source/test ownership summary
- oracle and report evidence summary
- package/platform/adoption evidence summary
- initial residual grouping

### Completion Criteria
- every major Epic 11 evidence family has an owner surface
- generated report and validation artifacts are separated from public claims
- residuals are visible before validation design begins

---

## Day 3: Validation Architecture

**Title:** Validation Design
**Theme:** Define reviewed and supplemental validation required for final Epic
11 closeout
**Time estimate:** 11 hours

### Tasks
1. Review current Make, CMake, install, package, benchmark, report, docs, and
   CI validation surfaces.
2. Classify validation lanes as reviewed, supplemental, local, staged,
   deferred, or unsupported.
3. Define validation requirements for source/test changes, documentation-only
   surfaces, report indexes, package proofs, platform tiers, benchmarks, and
   adoption docs.
4. Identify validation commands that must run locally versus commands that
   require hosted CI or remain supplemental.
5. Record full-validation risks and stop conditions.
6. Write the validation architecture artifact.

### Deliverables
- validation lane matrix
- command ownership map
- reviewed/supplemental/local/deferred classification
- validation risk register
- stop-condition notes

### Completion Criteria
- validation requirements match touched surfaces and inherited support tiers
- no supplemental lane is promoted to reviewed evidence by wording alone
- expensive or hosted-runner validation gaps are explicit

---

## Day 4: Validation Command Plan

**Title:** Command Plan
**Theme:** Convert validation architecture into an executable final validation
sequence
**Time estimate:** 11 hours

### Tasks
1. Build the exact command plan for source, docs, package, install, CMake,
   benchmark, report-index, and adoption validation.
2. Decide which commands must run before claim recalibration can proceed.
3. Define capture paths for stdout/stderr summaries and generated report
   metadata when relevant.
4. Define pass/fail interpretation for each validation command.
5. Record commands intentionally skipped or deferred with reasons.
6. Write the validation command-plan artifact.

### Deliverables
- executable validation command matrix
- artifact capture plan
- pass/fail interpretation table
- explicit skip/defer list
- Day 5-7 execution plan

### Completion Criteria
- validation can run without guessing command ownership or pass/fail meaning
- generated reports are read as evidence with freshness context
- validation scope is bounded before execution starts

---

## Day 5: Reviewed Validation Batch 1

**Title:** Validation Batch 1
**Theme:** Run core quality, source-list, docs, and package/static proof
validation
**Time estimate:** 12 hours

### Tasks
1. Run documentation hygiene and local markdown link/path checks.
2. Run source-list or repository-structure checks required by the command
   plan.
3. Run static package deferral or package-contract checks when applicable.
4. Run install/package proof scripts selected for local validation.
5. Capture failures, fixes, or explicit stop conditions.
6. Write the Batch 1 validation artifact.

### Deliverables
- documentation validation evidence
- source-list/package validation evidence
- static-first package proof status
- failure/fix notes or clean pass summary
- updated validation status table

### Completion Criteria
- core documentation and package-support checks have clear pass/fail status
- any failing required check stops the sprint for user input or a focused fix
- no package, ABI, or platform claim is widened by validation wording

---

## Day 6: Reviewed Validation Batch 2

**Title:** Validation Batch 2
**Theme:** Run CMake, test, and reviewed local quality validation required by
the final command plan
**Time estimate:** 12 hours

### Tasks
1. Run the reviewed local C quality gate if any `.c` or `.h` files changed.
2. Run selected CMake configure/build/registration or CTest commands from the
   command plan.
3. Run package/install CMake proof where applicable.
4. Reconcile local reviewed test counts or registration counts against known
   platform-tier expectations.
5. Capture failures, fixes, or explicit deferrals.
6. Write the Batch 2 validation artifact.

### Deliverables
- local quality-gate evidence or not-required statement
- CMake validation evidence
- test/registration reconciliation notes
- package/install CMake proof status
- updated validation status table

### Completion Criteria
- reviewed local validation commands have clear status
- any `.c`/`.h` changes are covered by `make format && make lint && make test`
- CMake/test wording remains bounded by local platform evidence

---

## Day 7: Supplemental And Report Validation

**Title:** Report Validation
**Theme:** Run supplemental benchmark/report/package validation and reconcile
generated evidence
**Time estimate:** 12 hours

### Tasks
1. Run selected canonical benchmark, sentinel, or guardrail report commands
   when required by the command plan.
2. Inspect generated `index.tsv`, `sentinels.tsv`, `manifest.txt`, or related
   report artifacts for freshness and row interpretation.
3. Run supplemental package/platform checks selected for local confidence.
4. Record skipped supplemental lanes with support-tier reasons.
5. Reconcile validation outputs against Sprint 131 report-index boundaries.
6. Write the supplemental/report validation artifact.

### Deliverables
- supplemental validation evidence
- generated report freshness summary
- report-index interpretation notes
- skipped supplemental lane register
- final validation execution summary

### Completion Criteria
- generated reports are interpreted with freshness and support-tier context
- supplemental evidence is not promoted into reviewed support claims
- validation execution is complete enough for competitive recalibration

---

## Day 8: Competitive Evidence Baseline

**Title:** Competitive Baseline
**Theme:** Compare final Epic 11 evidence against goals, state-of-the-art
targets, and explicit non-claims
**Time estimate:** 12 hours

### Tasks
1. Re-read Epic 11 goals, state-of-the-art target language, and prior
   competitive comparison artifacts.
2. Compare final source/test/oracle evidence against claimed solver families.
3. Compare final benchmark/report evidence against performance and efficiency
   wording.
4. Compare package/platform/install evidence against adoption and support
   wording.
5. Identify claims that are earned, local-only, supplemental, deferred, or
   unsupported.
6. Write the competitive evidence baseline artifact.

### Deliverables
- goal-to-evidence comparison table
- earned/local/supplemental/deferred/unsupported claim classification
- competitive evidence gap list
- state-of-the-art non-claim register
- recalibration input for Day 9

### Completion Criteria
- every major Epic 11 claim class has evidence classification
- state-of-the-art language is compared against actual evidence, not intent
- unsupported or overbroad claims are queued for cleanup

---

## Day 9: Competitive Claim Recalibration

**Title:** Claim Recalibration
**Theme:** Decide final public, maintainer, benchmark, package, and residual
claim wording for Epic 11 closeout
**Time estimate:** 12 hours

### Tasks
1. Convert Day 8 classification into final claim decisions.
2. Decide which public claims are earned and where they may appear.
3. Decide which claims must remain local, supplemental, deferred, or explicit
   non-claims.
4. Identify public/support wording requiring cleanup on Day 10-11.
5. Record competitive positioning without state-of-the-art overclaiming.
6. Write the claim recalibration artifact.

### Deliverables
- final claim decision table
- earned claim register
- non-claim register
- unsupported wording cleanup queue
- competitive recalibration summary

### Completion Criteria
- public/support wording decisions have evidence and owner surfaces
- unsupported or ambiguous claims have a cleanup path
- competitive comparison language is bounded and defensible

---

## Day 10: Unsupported-Claim Audit

**Title:** Claim Audit
**Theme:** Scan public and maintainer surfaces for unsupported package,
platform, performance, competitive, coverage, and report wording
**Time estimate:** 11 hours

### Tasks
1. Scan README, install docs, cookbook, solver selection, algorithm docs,
   benchmark docs, maintainer guide, and planning artifacts for claim drift.
2. Scan package/platform wording against Sprint 133-134 decisions.
3. Scan benchmark/performance wording against validation and report evidence.
4. Scan competitive wording against Day 8-9 recalibration decisions.
5. Build the final unsupported-claim cleanup queue.
6. Write the unsupported-claim audit artifact.

### Deliverables
- unsupported-claim scan evidence
- package/platform wording queue
- performance/report wording queue
- competitive wording queue
- cleanup priority list

### Completion Criteria
- unsupported or ambiguous wording is located before edits begin
- cleanup scope is bounded to wording, links, and claim fences unless code
  changes are explicitly required
- non-claims remain explicit and findable

---

## Day 11: Unsupported-Claim Cleanup

**Title:** Claim Cleanup
**Theme:** Remove, downgrade, or fence unsupported public/support wording and
validate the cleanup
**Time estimate:** 11 hours

### Tasks
1. Edit public/support docs to remove, downgrade, or fence unsupported claims.
2. Add links to evidence owners where claim context is valid but misplaced.
3. Preserve install/package/platform support-tier truth.
4. Preserve benchmark/report local-measurement boundaries.
5. Run focused docs hygiene, link/path, and claim-boundary checks.
6. Write the unsupported-claim cleanup artifact.

### Deliverables
- claim cleanup edits
- evidence-owner link updates
- focused validation evidence
- remaining non-claim register
- cleanup closeout notes

### Completion Criteria
- unsupported wording identified on Day 10 is fixed or explicitly deferred
- public docs say only what final evidence supports
- claim-boundary validation passes before residual publication

---

## Day 12: Residual Queue Publication

**Title:** Residual Queue
**Theme:** Publish post-Epic-11 residuals, future-epic candidates, optional
work, non-claims, and deferred QR residual promotion criteria
**Time estimate:** 12 hours

### Tasks
1. Consolidate residuals from Sprint 118-135 closeouts and retrospectives.
2. Publish the end-of-epic deferred QR residual queue with promotion criteria
   instead of immediate sprint implementation wording.
3. Classify residuals as future-epic candidates, optional-local work,
   metadata-blocked work, evidence-blocked work, or explicit non-claims.
4. Assign owner surfaces and promotion criteria.
5. Link residuals to validation and claim recalibration outcomes.
6. Write the residual queue publication artifact.

### Deliverables
- post-Epic-11 residual queue
- deferred QR residual queue with promotion criteria
- future-epic candidate list
- explicit non-claim register
- owner and promotion-criteria table

### Completion Criteria
- residuals are visible, classified, and actionable
- QR residual work is preserved without becoming immediate sprint scope
- future work is separated from earned Epic 11 claims

---

## Day 13: Retrospective Drafts And Handoff Synthesis

**Title:** Retro Drafts
**Theme:** Draft Sprint 136 and Epic 11 retrospectives and synthesize the final
post-epic handoff
**Time estimate:** 12 hours

### Tasks
1. Draft Sprint 136 retrospective inputs from validation, claim cleanup,
   residual publication, and closeout work.
2. Draft Epic 11 retrospective structure and key outcomes.
3. Synthesize final handoff sections for evidence, validation, claims,
   residuals, and non-claims.
4. Reconcile retrospective statements against validation and claim-boundary
   artifacts.
5. Identify any final closeout gaps for Day 14.
6. Write the retrospective draft and handoff synthesis artifact.

### Deliverables
- Sprint 136 retrospective draft inputs
- Epic 11 retrospective draft structure
- final handoff synthesis notes
- closeout gap list
- Day 14 finalization plan

### Completion Criteria
- retrospectives can be finalized without re-reading every daily artifact
- handoff language is evidence-bounded and aligned with residual queue
- Day 14 has a short, concrete closeout checklist

---

## Day 14: Final Epic Closeout

**Title:** Epic Closeout
**Theme:** Finalize Sprint 136 retrospective, Epic 11 retrospective, final
validation summary, and post-epic handoff
**Time estimate:** 12 hours

### Tasks
1. Write the final Sprint 136 retrospective.
2. Write the final Epic 11 retrospective.
3. Write or finalize the Epic 11 closeout handoff artifact.
4. Confirm package/platform/performance/support-tier claim boundaries one last
   time.
5. Reconcile Sprint 136 deliverables against Items 1-7.
6. Run final docs hygiene, link/path, and claim-boundary checks.

### Deliverables
- Sprint 136 retrospective
- Epic 11 retrospective
- final Epic 11 closeout handoff
- final validation summary
- final claim-boundary and residual summary

### Completion Criteria
- all Sprint 136 deliverables are represented by artifacts or explicit
  residual decisions
- Epic 11 closeout evidence, claims, non-claims, and residuals are coherent
- final validation and claim-boundary checks pass
