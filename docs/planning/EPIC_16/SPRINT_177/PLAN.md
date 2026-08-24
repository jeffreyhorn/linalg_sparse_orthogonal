# Sprint 177 Plan: Epic 16 Baseline, Evidence Matrix & Closure Gates

**Sprint Duration:** 14 days
**Goal:** Freeze the post-Epic-15 baseline, select Epic 16 closure targets,
and publish acceptance gates for product evidence, failure-path proof,
distribution, generated API status, and report metadata. This sprint
implements the Sprint 177 section of
`docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Source Artifact Note:** This plan lives under
`docs/planning/EPIC_16/SPRINT_177/PLAN.md` and implements the Sprint 177
section of `docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Starting Point:** Sprint 177 begins from:

- Epic 15 retrospective and PR #195 merged on `master`;
- Epic 16 planning review, gap-closure todo, and project plan available;
- current README, INSTALL, maintainer guide, generated API, package,
  generated-report, benchmark, workflow, and selected hosted evidence surfaces;
- prior Epic 13, Epic 14, and Epic 15 residual queues.

The sprint must:

- extract and classify residuals from recent epics;
- create a current evidence/status matrix;
- select the exact Epic 16 closure targets and non-goals;
- define acceptance gates for each selected gap;
- map required validation commands by change surface;
- set up Sprint 177 working notes, artifacts, and handoffs.

**End State:** Sprint 177 leaves behind:

- an Epic 16 selected-gap register;
- an evidence/status matrix;
- acceptance gate templates for selected Epic 16 work;
- a quality surface map;
- Sprint 177 working notes, daily artifacts, and handoffs to Sprints 178 and
  179.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 177 project-plan estimate.

---

## Day 1: Sprint Intake And Source Baseline

**Title:** Sprint Intake
**Theme:** Establish Sprint 177 scope, source-plan authority, sprint path, and
initial working structure
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 177 section of
   `docs/planning/EPIC_16/PROJECT_PLAN.md`.
2. Review the Epic 16 review and gap-closure todo.
3. Create Sprint 177 working notes and artifact directory structure.
4. Record the sprint output path and source-plan authority.
5. Capture the starting branch, relevant prior PRs, and baseline
   documentation surfaces.
6. Write the Day 1 sprint-intake artifact.

### Deliverables

- Sprint 177 working-notes baseline
- artifact directory structure
- source artifact note
- starting evidence and documentation surface list
- Day 1 sprint-intake artifact

### Completion Criteria

- Sprint 177 scope is tied to the Epic 16 project plan
- sprint output path is explicitly recorded
- no closure target is selected before the residual audit begins

---

## Day 2: Epic 13-15 Residual Queue Audit

**Title:** Residual Audit
**Theme:** Extract unresolved residuals from prior epic retrospectives and
closeout artifacts
**Time estimate:** 12 hours

### Tasks

1. Read Epic 13, Epic 14, and Epic 15 retrospectives.
2. Extract residuals related to allocation failure, generated API HTML,
   package-manager support, report freshness, external comparison breadth,
   public headers, maintainability, Windows parity, and performance evidence.
3. Deduplicate repeated residuals across epics.
4. Preserve long-horizon residuals that are still too broad for Epic 16.
5. Record evidence owners and blocker notes for each residual.
6. Write the Day 2 residual-audit artifact.

### Deliverables

- deduplicated residual queue
- blocker and evidence-owner notes
- long-horizon residual list
- Day 2 residual-audit artifact

### Completion Criteria

- prior-epic residuals are visible in one place
- duplicate residuals are collapsed without losing history
- broad state-of-the-art, ABI, platform, and ecosystem parity gaps remain
  explicit non-goals unless selected later

---

## Day 3: Residual Classification Matrix

**Title:** Residual Classification
**Theme:** Rank residuals by user value, claim risk, feasibility, and closure
quality
**Time estimate:** 12 hours

### Tasks

1. Define classification columns for user value, claim risk, implementation
   risk, testability, hosted/local evidence need, and estimated sprint cost.
2. Score each residual from the Day 2 queue.
3. Separate complete-closure candidates from partial-progress traps.
4. Identify dependencies between allocation-failure, package, generated API,
   report-target, Windows, comparison, header, and maintainability work.
5. Draft the initial closure-candidate shortlist.
6. Write the Day 3 residual-classification artifact.

### Deliverables

- residual classification matrix
- complete-closure candidate shortlist
- dependency notes
- partial-progress trap list
- Day 3 residual-classification artifact

### Completion Criteria

- every selected-candidate discussion has explicit closure reasoning
- high-risk but non-closeable gaps are marked as deferred
- candidate selection is based on evidence rather than novelty

---

## Day 4: Repository Surface Inventory

**Title:** Surface Inventory
**Theme:** Inventory current source, header, test, script, docs, benchmark,
package, report, and workflow surfaces
**Time estimate:** 12 hours

### Tasks

1. Inventory public headers, implementation files, tests, scripts, examples,
   benchmarks, package templates, and workflows.
2. Record large source/test review surfaces and duplicated registration lists.
3. Identify existing generated-report and generated API owners.
4. Identify current package, ABI, shared-library, and provider-support guards.
5. Identify current platform support-tier wording locations.
6. Write the Day 4 repository-surface inventory artifact.

### Deliverables

- repository surface inventory
- large-file and duplicated-list notes
- generated-report/API owner map
- package/platform guard inventory
- Day 4 surface-inventory artifact

### Completion Criteria

- planned closure targets can be mapped to concrete files and commands
- duplicated target-list and large-review-surface risks are visible
- no code edits are made during inventory

---

## Day 5: Evidence Status Matrix Schema

**Title:** Matrix Schema
**Theme:** Design the evidence/status matrix that will anchor Epic 16 claim
governance
**Time estimate:** 12 hours

### Tasks

1. Define evidence/status matrix columns for surface, support tier, owner file,
   hosted/local status, validation command, artifact path, claim boundary, and
   non-claim.
2. Select initial rows for package, API docs, reports, comparisons,
   performance, platform, ABI, allocation-failure, and maintainability
   surfaces.
3. Define pass, defer, local-only, hosted, advisory, and unsupported row
   semantics.
4. Align row semantics with existing report-index and maintainer-guide
   language.
5. Draft the matrix artifact.
6. Write the Day 5 matrix-schema artifact.

### Deliverables

- evidence/status matrix schema
- initial row list
- row-status semantics
- Day 5 matrix-schema artifact

### Completion Criteria

- the matrix can distinguish evidence from non-claims
- row semantics match existing report and support-tier vocabulary
- future sprint deliverables can update the matrix without ambiguity

---

## Day 6: Evidence Status Matrix Population

**Title:** Matrix Population
**Theme:** Populate the current evidence/status matrix from repository
surfaces and prior sprint outcomes
**Time estimate:** 12 hours

### Tasks

1. Populate package and install evidence rows.
2. Populate generated API, API reference, and public-header evidence rows.
3. Populate oracle, comparison, performance, and report freshness rows.
4. Populate Linux, macOS, and Windows platform support rows.
5. Populate allocation-failure, shared-library, dynamic ABI, and
   package-manager rows.
6. Write the Day 6 populated evidence-matrix artifact.

### Deliverables

- populated evidence/status matrix
- support-tier and validation-command mapping
- hosted/local/advisory/deferred row breakdown
- Day 6 populated-matrix artifact

### Completion Criteria

- every major public claim surface has a matrix row
- unsupported surfaces are recorded as non-claims
- selected hosted evidence is distinguishable from local-only evidence

---

## Day 7: Closure Target Selection

**Title:** Target Selection
**Theme:** Select exact Epic 16 closure targets and record explicit non-goals
**Time estimate:** 12 hours

### Tasks

1. Compare residual classification with the populated evidence/status matrix.
2. Select the Epic 16 closure targets for Sprints 178-186.
3. Record why each selected target can be fully closed in one sprint.
4. Record explicit non-goals for broad state-of-the-art, external parity,
   portable performance, shared-library, dynamic ABI, runtime-loader,
   package-manager, Windows, and generated-report claims.
5. Draft the selected-gap register.
6. Write the Day 7 closure-target selection artifact.

### Deliverables

- Epic 16 selected-gap register
- explicit non-goal register
- per-target closure rationale
- Day 7 target-selection artifact

### Completion Criteria

- selected targets match the Epic 16 project plan
- each selected target has a full-closure path
- broad claims remain rejected unless their evidence is selected and funded

---

## Day 8: Acceptance Gate Template Design

**Title:** Gate Templates
**Theme:** Design reusable acceptance gate templates for Epic 16 closure work
**Time estimate:** 12 hours

### Tasks

1. Design a gate template for allocation-failure proof.
2. Design a gate template for generated API publication or local-only status.
3. Design a gate template for package-manager provider proof or deferral.
4. Design a gate template for selected report target metadata.
5. Design a gate template for Windows report freshness promotion or deferral.
6. Write the Day 8 acceptance-gate-template artifact.

### Deliverables

- allocation-failure gate template
- generated API status gate template
- package-provider gate template
- selected-report metadata gate template
- Windows report freshness gate template
- Day 8 gate-template artifact

### Completion Criteria

- each selected target has an explicit pass/fail definition
- each gate names owner files, validation commands, and claim boundaries
- gates prevent broad claims from adjacent evidence

---

## Day 9: Acceptance Gate Completion

**Title:** Gate Completion
**Theme:** Complete acceptance gates for comparison, header cleanup,
maintainability, and closeout work
**Time estimate:** 12 hours

### Tasks

1. Design the bounded external comparison family gate.
2. Design the public-header coherence cleanup gate.
3. Design the large test/source review-surface reduction gate.
4. Design the final claim-recalibration and closeout gate.
5. Cross-check all templates against prior Epic 13-15 review comments.
6. Write the Day 9 acceptance-gate completion artifact.

### Deliverables

- comparison-family gate template
- header-coherence gate template
- review-surface reduction gate template
- final closeout gate template
- Day 9 gate-completion artifact

### Completion Criteria

- every Sprint 178-186 target has an acceptance gate
- gates define validation and documentation expectations
- previous review-comment failure modes are reflected where relevant

---

## Day 10: Quality Surface Map

**Title:** Quality Map
**Theme:** Map validation commands by change surface so later sprints choose
the right quality gate
**Time estimate:** 12 hours

### Tasks

1. Map required commands for documentation-only changes.
2. Map required commands for Python/script/report changes.
3. Map required commands for workflow changes.
4. Map required commands for package/install changes.
5. Map required commands for public-header and C source changes.
6. Write the Day 10 quality-surface map artifact.

### Deliverables

- quality surface map
- validation command matrix
- code/header quality-gate requirements
- workflow/package/report validation notes
- Day 10 quality-map artifact

### Completion Criteria

- later sprint days can select validation from the map
- `make format && make lint && make test` remains required for C/header
  changes
- docs-only and script-only changes still have focused validation commands

---

## Day 11: Claim Boundary Freeze

**Title:** Claim Freeze
**Theme:** Freeze public claim boundaries before implementation sprints begin
**Time estimate:** 12 hours

### Tasks

1. Audit README, INSTALL, maintainer guide, benchmark docs, solver-selection
   docs, generated API docs, and workflow comments for Epic 16 claim surfaces.
2. Compare public wording against the evidence/status matrix.
3. Identify wording that later sprints may update after earning evidence.
4. Identify wording that must remain an explicit non-claim.
5. Add claim-freeze notes to the selected-gap register.
6. Write the Day 11 claim-boundary freeze artifact.

### Deliverables

- public claim-boundary freeze
- candidate wording-update list
- protected non-claim wording list
- Day 11 claim-freeze artifact

### Completion Criteria

- current public wording is consistent with current evidence
- future claim updates are tied to specific sprint gates
- unsupported surfaces remain explicit non-claims

---

## Day 12: Sprint Handoff Package

**Title:** Handoff Package
**Theme:** Prepare detailed handoffs for allocation-failure and generated API
work
**Time estimate:** 12 hours

### Tasks

1. Create the Sprint 178 allocation-failure handoff.
2. Create the Sprint 179 generated API publication/status handoff.
3. Identify required owner files, commands, and documentation surfaces for
   both handoffs.
4. Record known risks and review-comment traps for both sprints.
5. Update working notes with handoff links.
6. Write the Day 12 handoff package artifact.

### Deliverables

- Sprint 178 allocation-failure handoff
- Sprint 179 generated API handoff
- owner-file and validation-command lists
- risk and review-trap notes
- Day 12 handoff artifact

### Completion Criteria

- Sprint 178 can begin without redoing baseline work
- Sprint 179 can begin with a clear publication decision frame
- handoffs preserve scoped claim boundaries

---

## Day 13: Sprint Reconciliation

**Title:** Reconciliation
**Theme:** Reconcile Sprint 177 artifacts with the Epic 16 project-plan items
and prepare closeout
**Time estimate:** 11 hours

### Tasks

1. Reconcile artifacts against items 177.1 through 177.6.
2. Confirm the evidence/status matrix, selected-gap register, gate templates,
   quality map, and handoffs are complete.
3. Identify any unresolved ambiguity or path mismatch for the retrospective.
4. Run documentation whitespace checks.
5. Update working notes with final status.
6. Write the Day 13 sprint-reconciliation artifact.

### Deliverables

- item-by-item reconciliation
- final artifact inventory
- unresolved ambiguity list
- Day 13 reconciliation artifact

### Completion Criteria

- all Sprint 177 project-plan items are covered
- no selected target lacks an acceptance gate
- closeout can proceed without creating new scope

---

## Day 14: Sprint Closeout

**Title:** Closeout
**Theme:** Finalize Sprint 177 records and leave a clean handoff for Sprint
178
**Time estimate:** 11 hours

### Tasks

1. Finalize Sprint 177 working notes.
2. Finalize the selected-gap register, evidence/status matrix, acceptance
   gates, quality map, and handoff artifacts.
3. Confirm generated files and artifacts are placed under the requested sprint
   directory.
4. Run `git diff --check`.
5. Prepare Sprint 177 retrospective inputs.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables

- finalized Sprint 177 working notes
- finalized artifact inventory
- Sprint 178 handoff confirmation
- validation note
- Day 14 closeout artifact

### Completion Criteria

- Sprint 177 is ready for retrospective creation
- Sprint 178 has actionable allocation-failure prerequisites
- working tree contains only intended Sprint 177 planning artifacts
