# Sprint 146 Plan: Epic 12 Final Validation, Claim Recalibration & Closeout

**Sprint Duration:** 14 days
**Goal:** Validate the final Epic 12 product state, recalibrate claims, publish
closed gaps and residuals, and decide whether any state-of-the-art claim has
actually been earned. This sprint implements the Sprint 146 section of
`docs/planning/EPIC_12/PROJECT_PLAN.md`.

**Starting Point:** Sprint 146 begins from:
- Sprint 137 baseline, gap selection, and evidence-contract work
- Sprint 138 maintained numerical corpus architecture
- Sprint 139 QR priority residual closure
- Sprint 140 partial-SVD edge-case and convergence residual closure
- Sprint 141 report-index normalization and freshness gates
- Sprint 142 runtime/backend governance and sentinel expansion
- Sprint 143 shared-library ABI decision and static-first package contract
- Sprint 144 platform promotion lane closure
- Sprint 145 adoption surface simplification, claim map, and closeout handoff
- final public documentation, package/platform/report support tiers, and
  validation commands identified by prior sprint artifacts

The sprint must:
- inventory all Epic 12 evidence across corpus, QR, partial-SVD, report,
  runtime/backend, package, platform, adoption, and validation surfaces
- run the strongest feasible local quality baseline plus package, report,
  corpus, and documentation checks required by touched surfaces
- reconcile hosted Linux, macOS, and Windows evidence with reviewed,
  supplemental, staged, local-only, and deferred support tiers
- rescan public and support documentation for unsupported state-of-the-art,
  external parity, package, platform, performance, ABI, and report wording
- publish a residual queue with owners, blockers, prerequisites, and promotion
  gates
- write the Epic 12 retrospective and final project-plan reconciliation
- prepare a next-epic handoff grounded in closed gaps and explicit non-claims

**End State:** Sprint 146 leaves behind:
- final Epic 12 evidence inventory
- final validation package
- public claim and non-claim audit
- residual queue with promotion gates
- Epic 12 retrospective
- final project-plan reconciliation
- next-epic handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 146 project-plan estimate.

---

## Day 1: Closeout Intake And Evidence Map

**Title:** Closeout Intake
**Theme:** Establish Sprint 146 scope, prerequisite artifacts, evidence
families, and final closeout criteria
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 146 section of
   `docs/planning/EPIC_12/PROJECT_PLAN.md`.
2. Review Sprint 137-145 retrospectives, closeout artifacts, and handoffs.
3. Create Sprint 146 working notes and artifact directory structure.
4. Define final Epic 12 evidence families: corpus, QR, partial-SVD, report,
   runtime/backend, package, platform, adoption, and validation.
5. Map Sprint 146 Items 1-7 to day-level owners.
6. Record final closeout criteria, non-claim guardrails, validation
   requirements, and stop conditions.

### Deliverables
- Sprint 146 working-notes baseline
- artifact directory structure
- evidence-family inventory
- item-to-day owner map
- final closeout criteria and non-claim register

### Completion Criteria
- every Sprint 146 project-plan item has a day-level owner
- Sprint 137-145 outputs are treated as prerequisite evidence, not re-opened
  without cause
- final closeout criteria distinguish earned claims from residual work

---

## Day 2: Evidence Inventory Part 1

**Title:** Corpus Evidence
**Theme:** Inventory corpus, QR, partial-SVD, and solver-correctness evidence
from Epic 12
**Time estimate:** 12 hours

### Tasks
1. Inventory Sprint 137-140 corpus, QR, partial-SVD, and solver evidence
   artifacts.
2. Identify the exact fixture-local claims each artifact supports.
3. Record validation commands, test files, generated or source-controlled
   artifacts, and freshness status for each evidence row.
4. Separate generated-local evidence from source-controlled evidence.
5. Identify any unresolved numerical gaps, fixtures, or claim boundaries that
   remain after Sprints 139-140.
6. Write the corpus and solver evidence inventory artifact.

### Deliverables
- corpus and solver evidence table
- QR evidence boundary summary
- partial-SVD evidence boundary summary
- generated-vs-source-controlled evidence classification
- unresolved numerical gap notes

### Completion Criteria
- every corpus, QR, and partial-SVD claim has an explicit evidence owner
- generated evidence is not treated as source-controlled pass proof
- residual numerical gaps are explicit and source-owned

---

## Day 3: Evidence Inventory Part 2

**Title:** Support Evidence
**Theme:** Inventory report, runtime/backend, package, platform, adoption, and
validation evidence
**Time estimate:** 12 hours

### Tasks
1. Inventory Sprint 141-145 report, runtime/backend, package, platform,
   adoption, and validation artifacts.
2. Map report-family row meanings, freshness policy, support tiers, owners,
   and non-claims to current source-controlled files.
3. Map package and platform evidence to current Make, CMake, `pkg-config`,
   workflow, and documentation surfaces.
4. Map adoption evidence to README, INSTALL, examples, cookbook,
   solver-selection, and selected public headers.
5. Identify support-tier gaps that need CI reconciliation or residual-queue
   publication.
6. Write the support evidence inventory artifact.

### Deliverables
- report evidence inventory
- runtime/backend governance evidence inventory
- package and platform evidence inventory
- adoption evidence inventory
- support-tier gap list

### Completion Criteria
- every non-numerical Epic 12 claim has an evidence owner
- support-tier wording is tied to concrete files or CI lanes
- unresolved support gaps are ready for validation and CI reconciliation

---

## Day 4: Validation Baseline Design

**Title:** Baseline Design
**Theme:** Define the strongest feasible final local validation package before
running it
**Time estimate:** 12 hours

### Tasks
1. Derive validation requirements from the Day 2-3 evidence inventories and
   current changed surfaces.
2. Select the strongest feasible local baseline: C/header quality, report
   schema, report normalization/freshness, package install, CMake install,
   examples, corpus, and documentation checks.
3. Identify validation commands that are hosted-CI-only or platform-specific.
4. Define pass/fail capture format, skipped-check rationale, and environment
   constraint register.
5. Define stop conditions for failed local validation or unclear hosted
   evidence.
6. Write the final validation baseline design artifact.

### Deliverables
- final validation command plan
- skipped-check and environment-constraint plan
- hosted-CI-only evidence list
- validation stop conditions
- pass/fail capture template

### Completion Criteria
- validation scope covers every touched or claimed surface
- hosted-only evidence is not confused with local proof
- failed checks have a clear stop-or-fix rule

---

## Day 5: Final Local Quality Baseline

**Title:** Local Baseline
**Theme:** Run the strongest feasible local quality, package, report, corpus,
and documentation checks
**Time estimate:** 12 hours

### Tasks
1. Run report schema, normalization, and freshness checks required by current
   report-family evidence.
2. Run maintained example build or smoke checks required by adoption evidence.
3. Run Make install and CMake install/downstream checks required by package
   evidence.
4. Run corpus/oracle checks that are feasible locally and required by final
   claim audit.
5. If `.c` or `.h` files are changed, run `make format && make lint &&
   make test`.
6. Record exact commands, results, failures, skips, and environment
   constraints.

### Deliverables
- final local validation command log
- pass/fail summary
- skipped-check rationale
- environment constraint register
- fix list for any validation failures

### Completion Criteria
- all required local checks pass or the sprint stops for failure triage
- skipped checks are explicitly justified
- validation evidence is ready for claim recalibration

---

## Day 6: Cross-Platform And CI Evidence Intake

**Title:** CI Intake
**Theme:** Collect hosted Linux, macOS, Windows, reviewed, supplemental,
staged, and deferred evidence
**Time estimate:** 12 hours

### Tasks
1. Inspect current workflow definitions for Linux, macOS, Windows, and any
   supplemental lanes.
2. Collect latest available CI run statuses and logs for the Sprint 146 branch
   or current master baseline where applicable.
3. Classify lanes as reviewed, supplemental, staged, local-only, hosted-only,
   or deferred.
4. Record lane commands, platform assumptions, expected test counts, and known
   blockers.
5. Identify mismatches between source-controlled support-tier wording and
   hosted CI evidence.
6. Write the CI evidence intake artifact.

### Deliverables
- CI lane inventory
- hosted evidence status table
- reviewed/supplemental/staged classification
- platform blocker register
- support-tier mismatch list

### Completion Criteria
- Linux, macOS, and Windows evidence is classified consistently
- missing or unavailable hosted evidence is explicit
- support-tier mismatches are ready for reconciliation

---

## Day 7: Cross-Platform Reconciliation

**Title:** CI Reconciliation
**Theme:** Reconcile platform support tiers against hosted CI and local
validation evidence
**Time estimate:** 12 hours

### Tasks
1. Compare Linux source-of-truth claims against current workflow evidence and
   local validation.
2. Compare macOS static-first install/export claims against current workflow
   evidence and local validation.
3. Compare Windows CMake-first reviewed subset claims against workflow
   evidence, test-count expectations, and staged blockers.
4. Confirm supplemental lanes do not get promoted without reviewed proof.
5. Fix or document any support-tier wording mismatch found during the pass.
6. Write the cross-platform reconciliation artifact.

### Deliverables
- reconciled platform support-tier table
- Linux evidence summary
- macOS evidence summary
- Windows evidence and staged-blocker summary
- support-tier fix or defer list

### Completion Criteria
- platform claims match available evidence
- supplemental and staged lanes remain clearly labeled
- any mismatch is either fixed or explicitly deferred with owner and blocker

---

## Day 8: Public Claim Audit

**Title:** Claim Audit
**Theme:** Rescan public docs and headers for unsupported state-of-the-art,
external parity, package, platform, performance, ABI, and report claims
**Time estimate:** 12 hours

### Tasks
1. Audit README, INSTALL, examples, cookbook, tutorial, solver-selection,
   benchmark/report docs, maintainer guide, and selected public headers.
2. Scan for unsupported state-of-the-art, external-library parity, package
   manager, shared-library, dynamic ABI, Windows parity, portable performance,
   and generated-report freshness wording.
3. Map each public claim to Day 2-7 evidence.
4. Fix small wording issues that clearly overstate evidence.
5. Record non-claims that remain intentionally visible.
6. Write the public claim audit artifact.

### Deliverables
- public claim inventory
- unsupported-claim scan results
- claim-to-evidence map
- wording fix list
- non-claim preservation summary

### Completion Criteria
- every public claim has evidence or is removed/reworded
- explicit non-claims are preserved
- no unsupported state-of-the-art or parity claim remains

---

## Day 9: Support And Maintainer Claim Audit

**Title:** Support Audit
**Theme:** Audit maintainer, report, benchmark, package, CI, and planning
surfaces for claim coherence
**Time estimate:** 12 hours

### Tasks
1. Audit maintainer guidance, benchmark/report guidance, report schemas,
   report-family rows, CI comments, install validation docs, and recent sprint
   artifacts.
2. Confirm source-controlled report rows do not imply generated pass evidence.
3. Confirm benchmark and sentinel wording remains local measurement context
   rather than portable performance proof.
4. Confirm package/ABI wording remains static-first unless explicit
   shared-library proof exists.
5. Fix small coherence issues or document residual work.
6. Write the support and maintainer claim audit artifact.

### Deliverables
- maintainer/support claim inventory
- report and benchmark non-claim summary
- package/ABI support-boundary summary
- support-surface fix or defer list
- final support claim audit notes

### Completion Criteria
- support docs agree with public docs and evidence inventories
- report and benchmark rows preserve freshness and local-only boundaries
- package/ABI claims remain evidence-backed

---

## Day 10: Residual Queue Design

**Title:** Residual Design
**Theme:** Design the final residual queue with owners, blockers,
prerequisites, and promotion gates
**Time estimate:** 12 hours

### Tasks
1. Consolidate residual debt from Sprint 137-145 retrospectives, Day 2-9
   inventories, validation results, and CI reconciliation.
2. Group residuals by numerical coverage, report/validation, runtime/backend,
   package/ABI, platform, adoption/docs, and competitive positioning.
3. Assign owner roles, blockers, prerequisites, and promotion gates.
4. Distinguish future product work from documentation cleanup and evidence
   refresh work.
5. Prioritize residuals by user value, risk, and dependency order.
6. Write the residual queue design artifact.

### Deliverables
- consolidated residual inventory
- owner and blocker map
- promotion-gate definitions
- priority ordering
- future-work classification

### Completion Criteria
- residual work is explicit and source-owned
- every residual has a promotion gate
- the queue favors complete gap closure over partial progress across many gaps

---

## Day 11: Residual Queue Publication

**Title:** Residual Queue
**Theme:** Publish the final residual queue and next-epic handoff candidates
**Time estimate:** 12 hours

### Tasks
1. Convert the Day 10 residual design into a source-controlled publication
   artifact.
2. Add next-epic handoff candidates with prerequisites and acceptance gates.
3. Identify residuals that should stay explicit non-claims until proof exists.
4. Cross-link residuals to evidence inventory, claim audit, and CI
   reconciliation artifacts.
5. Run documentation consistency and whitespace checks for the residual queue.
6. Record final residual queue publication notes.

### Deliverables
- published residual queue
- next-epic handoff candidate list
- residual-to-non-claim map
- residual validation summary
- cross-reference checklist

### Completion Criteria
- residual queue is readable as future planning input
- non-claims are not accidentally converted into roadmap promises
- residual publication checks pass

---

## Day 12: Epic 12 Retrospective Draft

**Title:** Retrospective Draft
**Theme:** Draft the Epic 12 retrospective with earned claims, non-claims,
closed gaps, validation evidence, and state-of-the-art assessment
**Time estimate:** 12 hours

### Tasks
1. Summarize Epic 12 goals, closed gaps, changed surfaces, and validation
   evidence.
2. Identify earned claims from the final evidence inventory and claim audit.
3. Record non-claims that remain after Sprint 146 validation and
   reconciliation.
4. Assess whether any state-of-the-art claim has been earned, narrowed, or
   rejected.
5. Draft lessons learned and next-epic recommendations.
6. Write the Epic 12 retrospective draft artifact.

### Deliverables
- Epic 12 retrospective draft
- earned-claim summary
- non-claim summary
- state-of-the-art assessment
- lessons learned and next-epic recommendations

### Completion Criteria
- retrospective claims are backed by evidence from Days 2-11
- state-of-the-art assessment is explicit and conservative
- remaining gaps are not hidden behind adoption or validation wording

---

## Day 13: Final Project Plan Reconciliation

**Title:** Plan Reconciliation
**Theme:** Reconcile the Epic 12 project plan against completed sprint
artifacts and prepare the final next-epic handoff
**Time estimate:** 11 hours

### Tasks
1. Compare Epic 12 project-plan items for Sprints 137-146 against completed
   sprint artifacts and retrospectives.
2. Mark items as complete, explicitly deferred, rejected, or residual.
3. Identify any mismatch between project-plan expectations and actual sprint
   outcomes.
4. Draft the final next-epic handoff using the residual queue and
   retrospective draft.
5. Run lightweight documentation checks for reconciliation artifacts.
6. Write the final project-plan reconciliation artifact.

### Deliverables
- Epic 12 project-plan reconciliation
- completed/deferred/rejected/residual item table
- mismatch and correction notes
- final next-epic handoff draft
- reconciliation validation summary

### Completion Criteria
- every Epic 12 sprint item has a closeout status
- mismatches are explicit and source-owned
- next-epic handoff is grounded in residual queue promotion gates

---

## Day 14: Final Closeout Package

**Title:** Epic Closeout
**Theme:** Finalize Epic 12 closeout, validation package, retrospective, and
handoff
**Time estimate:** 11 hours

### Tasks
1. Review all Sprint 146 artifacts and working notes for consistency.
2. Finalize the Epic 12 retrospective and final validation package.
3. Finalize public claim/non-claim audit, residual queue, and next-epic
   handoff.
4. Confirm Sprint 146 deliverables from the project-plan section are complete
   or explicitly deferred with proof.
5. Run final lightweight repository checks such as status, diff review,
   whitespace checks, and any required touched-surface checks.
6. Prepare Sprint 146 retrospective input notes.

### Deliverables
- finalized Epic 12 retrospective
- final validation package
- final claim/non-claim audit
- finalized residual queue with promotion gates
- next-epic handoff
- Sprint 146 closeout notes

### Completion Criteria
- Epic 12 closeout claims match available evidence
- residual work is explicit, owner-mapped, and promotion-gated
- no unsupported state-of-the-art, platform, package, ABI, performance, report,
  or external-parity claim remains
- Sprint 146 can close with a clear retrospective and next-epic handoff
