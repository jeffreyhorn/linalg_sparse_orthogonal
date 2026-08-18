# Sprint 166 Plan: Epic 14 Final Validation, Claim Recalibration & Closeout

**Sprint Duration:** 14 days
**Goal:** Validate the final Epic 14 state, recalibrate all public claims, and
publish a closeout that makes completed work, retained non-claims, and future
residuals unambiguous. This sprint implements the Sprint 166 section of
`docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md` and the title "Sprint 166: Final
Validation, Claim Calibration & Closeout", but the current merged planning
source for Sprint 166 lives in `docs/planning/EPIC_14/PROJECT_PLAN.md` as
"Sprint 166: Epic 14 Final Validation, Claim Recalibration & Closeout". This
plan follows the active Epic 14 source.

**Starting Point:** Sprint 166 begins from:
- Sprints 157-165 completed or explicitly residualized;
- updated generated API, oracle/comparison, package, performance, and public
  header evidence from Epic 14;
- Sprint 165 static-first package-boundary closeout and residual register;
- current local validation, hosted CI, reviewed, supplemental, local-only, and
  advisory evidence boundaries;
- retained non-claims for state-of-the-art status, external-library parity,
  portable performance, shared-library support, dynamic ABI compatibility,
  runtime-loader behavior, package-manager distribution, and broad platform
  parity.

The sprint must:
- inventory final Epic 14 evidence before changing claims;
- run the strongest feasible local baseline and touched-surface supplemental
  checks;
- reconcile hosted CI evidence separately from local/advisory evidence;
- audit public docs for unsupported package, ABI, platform, performance,
  generated-report, external-parity, and state-of-the-art wording;
- reconcile every Epic 14 project-plan item as complete, narrowed, or
  residualized with evidence links;
- write the Epic 14 retrospective and final residual queue;
- prepare the next-epic handoff without hiding deferred product decisions.

**End State:** Sprint 166 leaves behind:
- final Epic 14 evidence inventory;
- final validation record;
- public claim/non-claim audit;
- reconciled Epic 14 project-plan status;
- Epic 14 retrospective;
- final residual queue and next-epic candidates;
- Sprint 166 closeout handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 166 project-plan estimate.

---

## Day 1: Sprint Intake And Evidence Map

**Title:** Sprint Intake
**Theme:** Establish Sprint 166 scope, artifact layout, and final evidence
categories
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 166 section of
   `docs/planning/EPIC_14/PROJECT_PLAN.md`.
2. Review Sprint 157-165 retrospectives, closeout artifacts, and handoff
   notes.
3. Create Sprint 166 working notes and artifact directory structure.
4. Define final evidence categories: generated API, hosted oracle/comparison,
   QR, partial-SVD, Windows package, performance, public header, package
   boundary, and validation.
5. Record explicit non-goals for unsupported package, ABI, platform,
   performance, external-parity, generated-report, and state-of-the-art claims.
6. Write the Day 1 sprint-intake artifact.

### Deliverables
- Sprint 166 working-notes baseline
- artifact directory structure
- final evidence category map
- non-goal and stop-condition register
- Day 1 sprint-intake artifact

### Completion Criteria
- Sprint 166 scope is tied to the Epic 14 project plan
- all Epic 14 evidence categories are identified
- unsupported claims are separated from final validation work

---

## Day 2: Final Evidence Inventory Part 1

**Title:** Evidence Inventory I
**Theme:** Inventory generated docs, API, and report-publication evidence
**Time estimate:** 12 hours

### Tasks
1. Inventory Sprint 157 quality-surface and evidence-gate artifacts.
2. Inventory Sprint 158 generated API HTML publication closure artifacts.
3. Inventory Sprint 159 and Sprint 160 hosted oracle/comparison freshness
   artifacts.
4. Map generated documentation and report-index files to validation commands
   and claim boundaries.
5. Identify stale or missing generated-report evidence links for later
   reconciliation.
6. Write the Day 2 evidence-inventory artifact.

### Deliverables
- generated API evidence inventory
- hosted oracle/comparison evidence inventory
- report-index validation command map
- stale/missing evidence register
- Day 2 evidence-inventory artifact

### Completion Criteria
- generated API and report-publication evidence has source-backed owners
- hosted versus local-only generated evidence is distinguished
- missing evidence is recorded before claim audit begins

---

## Day 3: Final Evidence Inventory Part 2

**Title:** Evidence Inventory II
**Theme:** Inventory solver, package, performance, and public-header evidence
**Time estimate:** 12 hours

### Tasks
1. Inventory Sprint 161 partial-SVD comparison publication evidence.
2. Inventory Sprint 162 Windows package parity decision evidence.
3. Inventory Sprint 163 methodology-bound performance publication evidence.
4. Inventory Sprint 164 public-header/API coherence evidence.
5. Inventory Sprint 165 static-first package-boundary evidence.
6. Write the Day 3 evidence-inventory artifact.

### Deliverables
- partial-SVD comparison evidence map
- Windows package decision evidence map
- performance publication evidence map
- public-header/API evidence map
- static-first package evidence map
- Day 3 evidence-inventory artifact

### Completion Criteria
- solver, package, performance, and API evidence surfaces are mapped
- each surface has validation and non-claim boundaries attached
- Sprint 165 package handoff is ready for final validation planning

---

## Day 4: Validation Baseline Design

**Title:** Validation Design
**Theme:** Select the strongest feasible final validation baseline
**Time estimate:** 12 hours

### Tasks
1. Review Makefile, CMake, shell, Python, docs, package, report, and benchmark
   validation targets.
2. Classify required checks by touched surface and execution cost.
3. Define the final local baseline command set and supplemental checks.
4. Decide which hosted CI evidence must be cited instead of locally reproduced.
5. Define pass/fail, advisory, source-controlled, local-only, and hosted-only
   evidence semantics for the validation record.
6. Write the Day 4 validation-design artifact.

### Deliverables
- final validation command matrix
- local versus hosted evidence split
- advisory/source-controlled evidence classification
- validation risk register
- Day 4 validation-design artifact

### Completion Criteria
- final validation scope is explicit before command execution
- required checks are tied to touched surfaces
- hosted-only evidence is not confused with local proof

---

## Day 5: Full Local Validation Baseline

**Title:** Local Baseline
**Theme:** Run the strongest feasible local validation baseline
**Time estimate:** 12 hours

### Tasks
1. Run `make format` if source/header formatting surfaces require it.
2. Run `make lint` if source/header surfaces require it.
3. Run `make test` if `.c` or `.h` files changed or if final validation
   requires a fresh C baseline.
4. Run core docs/report/package validation commands selected on Day 4.
5. Capture pass/fail output summaries and any local environment constraints.
6. Write the Day 5 local-validation artifact.

### Deliverables
- final local validation baseline record
- command output summary
- local environment constraint notes
- blocker register if any command fails
- Day 5 local-validation artifact

### Completion Criteria
- strongest feasible local baseline is complete or blockers are documented
- failures stop follow-on claim work until resolved or explicitly narrowed
- validation evidence is source-backed and reproducible

---

## Day 6: Supplemental Validation Sweep

**Title:** Supplemental Checks
**Theme:** Run targeted generated docs, report, package, and claim-boundary
checks
**Time estimate:** 12 hours

### Tasks
1. Run generated API documentation checks selected by Day 4.
2. Run oracle/comparison report-index and freshness checks selected by Day 4.
3. Run package install/export and static deferral checks selected by Day 4.
4. Run benchmark/sentinel methodology checks selected by Day 4.
5. Run targeted stale-reference and unsupported-claim scans across changed
   docs and evidence files.
6. Write the Day 6 supplemental-validation artifact.

### Deliverables
- supplemental validation record
- generated docs/report/package/performance check summaries
- unsupported-claim scan results
- stale-reference scan results
- Day 6 supplemental-validation artifact

### Completion Criteria
- selected supplemental checks pass or blockers are recorded
- generated/report/package evidence remains bounded
- no unsupported claim wording is introduced by validation artifacts

---

## Day 7: Hosted CI Evidence Reconciliation

**Title:** CI Reconciliation
**Theme:** Reconcile hosted, reviewed, supplemental, local-only, and advisory
evidence
**Time estimate:** 12 hours

### Tasks
1. Review current Linux, macOS, and Windows workflow definitions.
2. Map reviewed versus supplemental hosted lanes for Epic 14 evidence.
3. Separate hosted CI proof from local-only generated and advisory rows.
4. Record unresolved hosted-only dependencies or unavailable external CI
   evidence.
5. Update the evidence inventory with CI lane ownership and support-tier
   wording.
6. Write the Day 7 CI-reconciliation artifact.

### Deliverables
- hosted CI lane map
- reviewed/supplemental/advisory evidence classification
- unresolved hosted evidence register
- support-tier wording notes
- Day 7 CI-reconciliation artifact

### Completion Criteria
- hosted CI evidence is reconciled by platform and lane
- local-only rows are not promoted to hosted proof
- unresolved hosted dependencies are explicit

---

## Day 8: Public Claim Audit Part 1

**Title:** Claim Audit I
**Theme:** Audit state-of-the-art, external parity, performance, and hosted
report claims
**Time estimate:** 12 hours

### Tasks
1. Scan README, tutorial, cookbook, solver-selection docs, benchmark docs,
   maintainer guide, and report-index docs.
2. Identify wording that could imply state-of-the-art status, broad external
   library parity, portable performance, hosted publication, or release proof.
3. Classify hits as supported claim, explicit non-claim, stale wording, or
   required cleanup.
4. Draft replacement wording for stale or overbroad claim language.
5. Apply narrow documentation cleanup if needed.
6. Write the Day 8 claim-audit artifact.

### Deliverables
- performance/external-parity claim audit
- hosted-report wording audit
- replacement wording list
- documentation cleanup if required
- Day 8 claim-audit artifact

### Completion Criteria
- public performance and external-parity claims are evidence-bounded
- generated-report wording does not exceed hosted proof
- stale claim wording is fixed or explicitly deferred with owner

---

## Day 9: Public Claim Audit Part 2

**Title:** Claim Audit II
**Theme:** Audit package, Windows, shared-library, ABI, and runtime-loader
claims
**Time estimate:** 12 hours

### Tasks
1. Scan README, INSTALL, maintainer guide, API reference, CMake comments,
   `sparse.pc`, and package validation scripts.
2. Identify wording that could imply package-manager distribution,
   shared-library support, dynamic ABI compatibility, runtime-loader behavior,
   broad Windows parity, or static/shared selector support.
3. Classify hits as supported static-first claim, explicit non-claim, stale
   wording, or required cleanup.
4. Apply narrow documentation cleanup if needed.
5. Confirm Sprint 165 package residuals remain product decisions.
6. Write the Day 9 claim-audit artifact.

### Deliverables
- package/ABI/Windows claim audit
- static-first support statement
- package residual confirmation
- documentation cleanup if required
- Day 9 claim-audit artifact

### Completion Criteria
- package and ABI wording remains static-first and bounded
- Windows support wording matches reviewed hosted evidence
- residual package decisions are not hidden as implementation gaps

---

## Day 10: Project Plan Reconciliation Part 1

**Title:** Plan Reconciliation I
**Theme:** Reconcile Sprint 157-161 Epic 14 project-plan items
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 157 plan, working notes, artifacts, retrospective, and PR
   outcome.
2. Review Sprint 158 plan, working notes, artifacts, retrospective, and PR
   outcome.
3. Review Sprint 159 and Sprint 160 hosted oracle/comparison freshness
   artifacts and outcomes.
4. Review Sprint 161 partial-SVD comparison publication artifacts and outcomes.
5. Mark each related Epic 14 item as complete, narrowed, deferred, or
   residualized with evidence links.
6. Write the Day 10 project-plan-reconciliation artifact.

### Deliverables
- Sprint 157-161 reconciliation table
- evidence links for completed items
- narrowed/deferred/residualized item register
- Day 10 project-plan-reconciliation artifact

### Completion Criteria
- Sprint 157-161 items have explicit close states
- evidence links support completed claims
- narrowed or deferred items are not presented as complete

---

## Day 11: Project Plan Reconciliation Part 2

**Title:** Plan Reconciliation II
**Theme:** Reconcile Sprint 162-166 Epic 14 project-plan items
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 162 Windows package parity decision artifacts and outcome.
2. Review Sprint 163 performance publication artifacts and outcome.
3. Review Sprint 164 public-header/API coherence artifacts and outcome.
4. Review Sprint 165 static-first package-boundary artifacts and outcome.
5. Reconcile Sprint 166 in-progress evidence and remaining closeout work.
6. Write the Day 11 project-plan-reconciliation artifact.

### Deliverables
- Sprint 162-166 reconciliation table
- completed/narrowed/deferred/residualized status map
- final Epic 14 success-criteria status draft
- Day 11 project-plan-reconciliation artifact

### Completion Criteria
- Sprint 162-166 items have explicit close states
- Sprint 166 remaining work is reduced to retrospective and residual queue
- Epic 14 success criteria have evidence-backed status

---

## Day 12: Epic 14 Retrospective Draft

**Title:** Retrospective Draft
**Theme:** Draft the Epic 14 retrospective with earned claims and non-claims
**Time estimate:** 12 hours

### Tasks
1. Review Epic 14 project plan, Sprint 157-166 artifacts, and prior epic
   retrospective style.
2. Draft completed work, earned claims, validation evidence, and support-tier
   boundaries.
3. Draft retained non-claims and state-of-the-art assessment.
4. Draft lessons learned and next-epic candidate themes.
5. Link supporting sprint artifacts and final validation evidence.
6. Write the Day 12 retrospective-draft artifact.

### Deliverables
- Epic 14 retrospective draft outline
- earned-claim summary
- retained non-claim summary
- state-of-the-art assessment draft
- Day 12 retrospective-draft artifact

### Completion Criteria
- retrospective draft is evidence-backed
- earned claims and non-claims are clearly separated
- state-of-the-art assessment does not exceed evidence

---

## Day 13: Final Residual Queue And Closeout Prep

**Title:** Residual Queue
**Theme:** Publish final residual queue and prepare closeout materials
**Time estimate:** 12 hours

### Tasks
1. Consolidate residuals from Sprint 157-165 retrospectives and Sprint 166
   reconciliation artifacts.
2. For each residual, record owner, blocker, prerequisite, promotion gate, and
   recommended next-epic priority.
3. Confirm residuals are not described as completed Epic 14 claims.
4. Prepare PR description bullets, review-risk notes, and validation summary.
5. Prepare Sprint 166 closeout checklist for Day 14.
6. Write the Day 13 residual-queue artifact.

### Deliverables
- final residual queue
- owner/blocker/prerequisite/promotion-gate table
- PR description outline
- review-risk notes
- Day 13 residual-queue artifact

### Completion Criteria
- residuals are actionable and evidence-bounded
- future work has promotion gates rather than vague aspirations
- closeout materials are ready for Day 14

---

## Day 14: Closeout And Handoff

**Title:** Closeout
**Theme:** Finalize Sprint 166 and Epic 14 closeout handoff
**Time estimate:** 12 hours

### Tasks
1. Update working notes with final changed files, validation commands, and
   known residuals.
2. Finalize `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md` if Sprint 166
   has enough evidence to publish it in this branch.
3. Verify Sprint 166 plan, working notes, daily artifacts, retrospective draft,
   project-plan reconciliation, and residual queue are internally consistent.
4. Run final documentation and touched-surface validation checks selected by
   Days 4-6.
5. Record PR description bullets, review-risk notes, and next-epic handoff.
6. Write the Day 14 closeout artifact.

### Deliverables
- final Sprint 166 working-notes update
- Day 14 closeout artifact
- final validation summary
- Epic 14 retrospective or explicit handoff to create it
- next-epic handoff

### Completion Criteria
- Epic 14 final validation and claim recalibration are complete or explicitly
  residualized
- public claims and non-claims are evidence-bounded
- branch is ready for review with validation evidence and closeout handoff
