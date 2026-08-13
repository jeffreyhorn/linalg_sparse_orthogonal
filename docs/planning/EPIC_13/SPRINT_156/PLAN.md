# Sprint 156 Plan: Epic 13 Final Validation, Claim Recalibration & Closeout

**Sprint Duration:** 14 days
**Goal:** Validate the final Epic 13 product state, reconcile evidence across
platform, corpus, report, ABI/package, comparison, and adoption work, recalibrate
public claims, publish residuals, and prepare the next-epic handoff. This sprint
implements the Sprint 156 section of
`docs/planning/EPIC_13/PROJECT_PLAN.md`.

**Starting Point:** Sprint 156 begins from:
- Sprint 147 baseline, claim-target, and evidence-gate artifacts available;
- Sprint 148 Windows staged portability results available;
- Sprint 149 Windows install-validation parity decision available;
- Sprint 150 QR maintained corpus expansion available;
- Sprint 151 partial-SVD maintained corpus expansion available;
- Sprint 152 report freshness publication available;
- Sprint 153 shared-library ABI and static-first package decision available;
- Sprint 154 external comparison harness and first narrow study available;
- Sprint 155 tutorial, header cleanup, and API reference coherence available;
- all final public claims must remain tied to reviewed evidence and explicit
  non-claims.

The sprint must:
- inventory Epic 13 artifacts and evidence rows across every completed sprint;
- run the strongest feasible local baseline and focused package, report,
  corpus, comparison, and documentation checks required by touched surfaces;
- reconcile Linux, macOS, Windows, supplemental, staged, reviewed, and deferred
  evidence;
- audit public and support documentation for unsupported state-of-the-art,
  parity, ABI, package-manager, platform, and performance wording;
- publish a residual queue with owners, blockers, prerequisites, and promotion
  gates;
- write the Epic 13 retrospective and final competitive assessment;
- reconcile the Epic 13 project plan against completed work;
- leave the next epic with a clean, actionable handoff.

**End State:** Sprint 156 leaves behind:
- final Epic 13 evidence inventory;
- final validation package;
- public claim and non-claim audit;
- residual queue with promotion gates;
- Epic 13 retrospective;
- reconciled Epic 13 project-plan status;
- next-epic handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 156 project-plan estimate.

---

## Day 1: Sprint Intake And Closeout Baseline

**Title:** Closeout Baseline
**Theme:** Establish Sprint 156 scope, artifact structure, evidence sources,
and closeout constraints
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 156 section of
   `docs/planning/EPIC_13/PROJECT_PLAN.md`.
2. Create Sprint 156 working notes and artifact directory structure.
3. Inventory Sprint 147 through Sprint 155 plans, working notes, artifacts,
   retrospectives, and handoff files.
4. Record final closeout scope, non-scope, and stop conditions for broad
   state-of-the-art, parity, ABI, package-manager, and platform claims.
5. Identify validation surfaces touched during Epic 13.
6. Write the Day 1 closeout-baseline artifact.

### Deliverables
- Sprint 156 working-notes baseline
- artifact directory structure
- Sprint 147-155 source inventory
- closeout stop-condition register
- Day 1 closeout-baseline artifact

### Completion Criteria
- Sprint 156 scope is tied to the Epic 13 project plan
- every prior sprint has a known evidence source or explicit gap
- unsupported claim categories are blocked before validation starts

---

## Day 2: Final Evidence Inventory

**Title:** Evidence Inventory
**Theme:** Build the final cross-sprint inventory for platform, corpus, report,
package, comparison, adoption, and validation evidence
**Time estimate:** 12 hours

### Tasks
1. Inventory Sprint 147 baseline and claim-target evidence.
2. Inventory Sprint 148 and Sprint 149 Windows portability and install
   evidence.
3. Inventory Sprint 150 and Sprint 151 QR and partial-SVD corpus evidence.
4. Inventory Sprint 152 report freshness and publication evidence.
5. Inventory Sprint 153 ABI/package evidence and static-first boundaries.
6. Inventory Sprint 154 comparison-study evidence and Sprint 155 adoption/API
   evidence.

### Deliverables
- final Epic 13 evidence inventory
- source-to-sprint evidence matrix
- missing or stale evidence list
- validation follow-up queue
- Day 2 evidence-inventory artifact

### Completion Criteria
- each Epic 13 deliverable has an evidence row or explicit residual
- cross-sprint dependencies are visible before quality checks
- validation work can target concrete files, commands, and artifacts

---

## Day 3: Validation Matrix Design

**Title:** Validation Matrix
**Theme:** Define the final quality baseline and focused checks required for
all touched Epic 13 surfaces
**Time estimate:** 12 hours

### Tasks
1. Identify mandatory local checks for documentation-only changes.
2. Identify full quality gates required if `.c` or public `.h` files change.
3. Identify package, install, CMake, pkg-config, and downstream-consumer checks
   relevant to Sprint 153 and Sprint 149 evidence.
4. Identify corpus and generated-report checks relevant to Sprints 150 through
   152.
5. Identify comparison-harness checks relevant to Sprint 154.
6. Write the final validation matrix with command ownership and skip rules.

### Deliverables
- final validation matrix
- command list and ownership notes
- skip/defer policy for local-only or unavailable checks
- quality-gate escalation rules
- Day 3 validation-matrix artifact

### Completion Criteria
- every planned validation command has a reason and owner
- unavailable checks have explicit deferral semantics
- final validation will not silently widen support claims

---

## Day 4: Full Local Quality Baseline

**Title:** Local Baseline
**Theme:** Run the strongest feasible local validation baseline and record
results without overstating platform coverage
**Time estimate:** 12 hours

### Tasks
1. Run repository formatting checks required by current touched surfaces.
2. Run lint checks required by current touched surfaces.
3. Run the full local test suite when code or header changes require it.
4. Capture local environment details relevant to reproducibility.
5. Record failures, skips, or unavailable tools with exact blockers.
6. Write the local quality baseline artifact.

### Deliverables
- local quality command log
- environment and tool-version notes
- failure, skip, and unavailable-tool register
- remediation or deferral list
- Day 4 local-baseline artifact

### Completion Criteria
- local baseline result is reproducible from recorded commands
- failures are fixed or explicitly blocked before claim audit
- local-only evidence is labeled as local-only

---

## Day 5: Package And Install Validation

**Title:** Package Validation
**Theme:** Validate the static-first package, install, CMake export, and
pkg-config evidence inherited from Epic 13 package work
**Time estimate:** 12 hours

### Tasks
1. Re-run or review maintained Make install and pkg-config proof where
   available.
2. Re-run or review maintained CMake install/downstream proof where available.
3. Confirm static-first package metadata does not imply unsupported shared
   library, package-manager, or ABI guarantees.
4. Compare install outputs against documented header and library surfaces.
5. Record platform-specific package gaps and staged-lane boundaries.
6. Write the package validation artifact.

### Deliverables
- package/install validation notes
- static-first metadata check results
- CMake/pkg-config downstream-consumer evidence
- platform package residual list
- Day 5 package-validation artifact

### Completion Criteria
- package claims are tied to install/downstream proof
- shared-library and package-manager non-claims remain intact
- unresolved package gaps have promotion gates

---

## Day 6: Platform And CI Reconciliation

**Title:** Platform Reconciliation
**Theme:** Reconcile Linux, macOS, Windows, reviewed, supplemental, staged,
local-only, and deferred evidence
**Time estimate:** 12 hours

### Tasks
1. Review CI workflow definitions and final reviewed/supplemental lanes.
2. Reconcile Windows promoted tests against staged exclusions and blockers.
3. Reconcile Linux and macOS package, report, runtime, and corpus evidence.
4. Identify failures caused by external service outages versus repository
   defects.
5. Update the platform evidence table with support tier and confidence labels.
6. Write the platform/CI reconciliation artifact.

### Deliverables
- final platform evidence table
- CI lane reconciliation notes
- staged and deferred blocker list
- support-tier labels
- Day 6 platform-reconciliation artifact

### Completion Criteria
- each platform claim maps to a reviewed or supplemental lane
- staged exclusions are explicit and not treated as pass coverage
- external outage evidence is separated from code-quality evidence

---

## Day 7: Corpus And Report Validation

**Title:** Corpus Reports
**Theme:** Validate maintained QR, partial-SVD, generated report freshness, and
report-index publication evidence
**Time estimate:** 12 hours

### Tasks
1. Reconcile QR corpus fixtures, expected rows, tolerances, and report entries.
2. Reconcile partial-SVD corpus fixtures, expected rows, tolerances, and report
   entries.
3. Validate generated report freshness metadata and normalization semantics.
4. Check report-index documentation for stale dates, stale counts, or orphaned
   rows.
5. Record corpus families deferred from Epic 13.
6. Write the corpus/report validation artifact.

### Deliverables
- QR corpus validation summary
- partial-SVD corpus validation summary
- report freshness and index validation notes
- deferred corpus-family queue
- Day 7 corpus-report artifact

### Completion Criteria
- maintained corpus claims are backed by current rows
- report freshness wording matches generated evidence
- deferred corpus work has owners and promotion criteria

---

## Day 8: Comparison Study Reconciliation

**Title:** Comparison Reconciliation
**Theme:** Reconcile the external comparison harness and first narrow study
against claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 154 target selection, dependency, skip/defer, and provenance
   artifacts.
2. Validate the first narrow comparison study against its fixture, metric, and
   tolerance definitions.
3. Confirm optional external dependency behavior is documented and reproducible.
4. Audit comparison wording for broad ecosystem or performance parity claims.
5. Publish comparison residuals for missing targets or incomplete report rows.
6. Write the comparison reconciliation artifact.

### Deliverables
- comparison harness validation notes
- first narrow study reconciliation
- dependency and provenance status
- comparison non-claim audit notes
- Day 8 comparison-reconciliation artifact

### Completion Criteria
- comparison evidence supports only the selected narrow study
- skipped external dependencies do not appear as passed evidence
- future comparison work is staged behind explicit gates

---

## Day 9: Adoption And API Surface Reconciliation

**Title:** Adoption Surface
**Theme:** Reconcile tutorial, API reference, cookbook, examples, headers, and
maintainer guidance against final Epic 13 evidence
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 155 tutorial and API reference updates against current claims.
2. Reconcile public header cleanup against declaration-preservation evidence.
3. Check README, cookbook, examples, solver-selection, package, and support-tier
   docs for inconsistent adoption guidance.
4. Identify duplicated or stale onboarding paths.
5. Record API reference and generated-reference residuals.
6. Write the adoption/API reconciliation artifact.

### Deliverables
- adoption surface reconciliation notes
- public header preservation summary
- tutorial/API/reference residuals
- documentation consistency checklist
- Day 9 adoption-surface artifact

### Completion Criteria
- adoption docs describe one coherent first-use path
- header cleanup remains documentation-only unless explicitly validated
- API reference residuals are ready for next-epic planning

---

## Day 10: Public Claim And Non-Claim Audit

**Title:** Claim Audit
**Theme:** Audit public and support docs for unsupported state-of-the-art,
external parity, package, platform, performance, ABI, and report wording
**Time estimate:** 12 hours

### Tasks
1. Search public docs for state-of-the-art, parity, performance, ABI, package,
   and platform wording.
2. Compare each claim against the final evidence inventory.
3. Reword unsupported claims into evidence-bound statements or explicit
   non-claims.
4. Check support-tier and package docs for consistency with staged evidence.
5. Record claim changes and rationale.
6. Write the claim/non-claim audit artifact.

### Deliverables
- public claim inventory
- unsupported claim correction list
- final non-claim register
- documentation patch list
- Day 10 claim-audit artifact

### Completion Criteria
- every public claim maps to evidence or is removed
- broad state-of-the-art and ecosystem parity claims remain blocked
- support-tier and package wording is internally consistent

---

## Day 11: Residual Queue Publication

**Title:** Residual Queue
**Theme:** Publish remaining future work with owners, blockers, prerequisites,
and promotion gates
**Time estimate:** 12 hours

### Tasks
1. Consolidate residuals from Sprints 147 through 155 and Days 1 through 10.
2. Deduplicate residuals into platform, corpus, report, ABI/package,
   comparison, adoption, and validation categories.
3. Assign each residual an owner role, blocker, prerequisite, and promotion
   gate.
4. Separate next-epic candidates from long-horizon research or ecosystem work.
5. Publish the final Epic 13 residual queue.
6. Write the residual-queue publication artifact.

### Deliverables
- final Epic 13 residual queue
- owner, blocker, prerequisite, and gate fields
- next-epic candidate list
- long-horizon deferral list
- Day 11 residual-queue artifact

### Completion Criteria
- residuals are actionable rather than vague backlog notes
- next-epic work is prioritized by complete-gap closure potential
- deferred work cannot be mistaken for completed support

---

## Day 12: Epic 13 Retrospective Draft

**Title:** Retrospective Draft
**Theme:** Draft the Epic 13 retrospective with earned claims, non-claims,
validation evidence, competitive assessment, and residuals
**Time estimate:** 12 hours

### Tasks
1. Review previous epic retrospectives for structure and expected content.
2. Draft Epic 13 accomplishments by sprint and deliverable area.
3. Summarize validation evidence and support-tier boundaries.
4. Summarize earned claims, rejected claims, and competitive-position changes.
5. Incorporate residual queue and next-epic handoff.
6. Prepare the retrospective for Day 13 reconciliation.

### Deliverables
- Epic 13 retrospective draft
- sprint-by-sprint accomplishment summary
- earned-claim and non-claim sections
- validation and residual sections
- Day 12 retrospective-draft artifact

### Completion Criteria
- retrospective claims are evidence-bound
- competitive assessment is narrow and defensible
- residuals and next-epic handoff are represented

---

## Day 13: Project Plan Reconciliation

**Title:** Plan Reconciliation
**Theme:** Reconcile the Epic 13 project plan against completed sprint
artifacts, validation results, and residuals
**Time estimate:** 11 hours

### Tasks
1. Compare Sprint 147 through Sprint 156 planned items against delivered
   artifacts.
2. Mark completed, partially completed, deferred, and superseded work.
3. Reconcile estimate drift and validation surprises.
4. Update final handoff notes for the next epic.
5. Reconcile the Epic 13 retrospective draft against the final project-plan
   status.
6. Write the project-plan reconciliation artifact.

### Deliverables
- final project-plan reconciliation table
- completed/deferred/superseded work summary
- estimate and validation variance notes
- next-epic handoff updates
- Day 13 plan-reconciliation artifact

### Completion Criteria
- project-plan status matches real artifacts
- deferred work appears in the residual queue
- next-epic handoff is grounded in completed-gap closure

---

## Day 14: Final Closeout And Handoff

**Title:** Final Closeout
**Theme:** Publish final Sprint 156 artifacts, finalize the Epic 13
retrospective, and leave a clean next-epic handoff
**Time estimate:** 11 hours

### Tasks
1. Finalize Sprint 156 working notes and artifact index.
2. Finalize the Epic 13 retrospective.
3. Re-check claim/non-claim wording after retrospective and residual updates.
4. Run final docs-only checks, or full quality gates if final edits touch
   `.c` or public `.h` files.
5. Prepare Sprint 156 retrospective inputs and next-epic handoff.
6. Write the Day 14 closeout artifact.

### Deliverables
- finalized Sprint 156 artifacts
- finalized Epic 13 retrospective
- final claim/non-claim check notes
- validation closeout notes
- next-epic handoff package

### Completion Criteria
- Sprint 156 deliverables are complete or explicitly deferred
- Epic 13 retrospective is ready for review
- next-epic planning starts from evidence, residuals, and clear support
  boundaries
