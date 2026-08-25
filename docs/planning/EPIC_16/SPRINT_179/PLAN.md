# Sprint 179 Plan: Generated API HTML Publication Decision

**Sprint Duration:** 14 days
**Goal:** Close generated API HTML status with either a hosted publication
path or a stronger enforced local-only product decision. This sprint implements
the Sprint 179 section of `docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Source Artifact Note:** This plan lives under
`docs/planning/EPIC_16/SPRINT_179/PLAN.md` and implements the Sprint 179
section of `docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Starting Point:** Sprint 179 begins from:

- Sprint 177 evidence matrix status for generated API HTML;
- current Doxygen configuration, generated-output ignore rules, and local-only
  guard behavior;
- current README, API reference, maintainer guide, and support-tier navigation;
- existing docs-generation and docs-check targets;
- Epic 16 emphasis on evidence-backed product decisions instead of aspirational
  documentation claims.

The sprint must:

- audit Doxygen inputs, ignored outputs, warnings, page coverage, source-header
  authority, and current docs navigation;
- decide whether generated API HTML is hosted, retained as a CI artifact,
  committed, or explicitly kept local-only;
- implement the selected publication or local-only enforcement path;
- add or tighten checks for stale output, missing pages, staged generated
  files, and publication metadata;
- update README, API reference, maintainer guide, and support-tier docs to point
  to the supported API path;
- run docs generation, docs checks, generated API checks, and
  `git diff --check`.

**End State:** Sprint 179 leaves behind:

- a generated API HTML product decision artifact;
- implemented publication or enforced local-only behavior;
- freshness and staging guard evidence;
- updated docs navigation and support-tier wording;
- validation records for docs generation, docs checks, generated API checks, and
  whitespace review;
- Sprint 179 working notes, daily artifacts, and retrospective inputs.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 179 project-plan estimate.

---

## Day 1: Sprint Intake And Evidence Baseline

**Title:** Intake And Evidence Baseline
**Theme:** Establish Sprint 179 scope, artifact layout, and current generated
API status
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 179 section of
   `docs/planning/EPIC_16/PROJECT_PLAN.md`.
2. Review Sprint 177 evidence matrix entries for generated API HTML status.
3. Create Sprint 179 working notes and artifact directory structure.
4. Record the current generated API claim surface in README, API reference,
   maintainer guide, and support-tier docs.
5. Record the current Doxygen configuration, output locations, ignored output
   paths, and local-only guard behavior.
6. Write the Day 1 evidence-baseline artifact.

### Deliverables

- Sprint 179 working-notes baseline
- artifact directory structure
- generated API claim baseline
- Doxygen and guard status baseline
- Day 1 evidence-baseline artifact

### Completion Criteria

- Sprint 179 scope is tied to the Epic 16 project plan
- current generated API status is recorded before changes begin
- publication and local-only options remain open pending the audit

---

## Day 2: Doxygen Input And Output Audit

**Title:** Doxygen Surface Audit
**Theme:** Inventory generated API inputs, outputs, ignored paths, and source
authority
**Time estimate:** 12 hours

### Tasks

1. Inspect the Doxygen configuration and all configured input paths.
2. Map public headers, generated pages, examples, and supplemental docs that
   contribute to API output.
3. Confirm which generated output paths are ignored, retained, staged, or
   checked.
4. Identify source-header authority rules for public API descriptions.
5. Record any generated output that can drift without review.
6. Write the Day 2 Doxygen surface audit artifact.

### Deliverables

- Doxygen input inventory
- generated output inventory
- ignored-path and staging-policy notes
- source-header authority notes
- Day 2 Doxygen surface audit artifact

### Completion Criteria

- every configured Doxygen input path is accounted for
- generated output paths are tied to current repository policy
- source authority for API text is explicit

---

## Day 3: Warning And Page Coverage Audit

**Title:** Warning And Coverage Audit
**Theme:** Determine whether generated API HTML is complete enough to publish
or must remain local-only
**Time estimate:** 12 hours

### Tasks

1. Run or inspect the current docs-generation command.
2. Capture Doxygen warnings and categorize actionable versus accepted warnings.
3. Audit page coverage for public headers, major API families, examples, and
   install/adoption entry points.
4. Identify missing pages, stale pages, orphaned pages, and unsupported claim
   risks.
5. Compare generated API coverage against README and support-tier claims.
6. Write the Day 3 warning and page-coverage artifact.

### Deliverables

- warning inventory
- page coverage matrix
- missing and stale page findings
- generated API claim-risk notes
- Day 3 coverage artifact

### Completion Criteria

- generated API readiness is supported by concrete warning and coverage data
- publish blockers are separated from polish issues
- no publication decision is made without documented coverage evidence

---

## Day 4: Current Guard And CI Audit

**Title:** Guard And CI Audit
**Theme:** Audit existing docs checks, generated-output guards, and CI evidence
paths
**Time estimate:** 12 hours

### Tasks

1. Inspect Make targets, scripts, and tests that generate or validate docs.
2. Inspect workflow steps that publish, upload, retain, or validate docs
   artifacts.
3. Identify whether generated API HTML is protected against stale output.
4. Identify whether staged generated files are rejected or intentionally
   allowed.
5. Record CI artifact retention and metadata gaps.
6. Write the Day 4 guard and CI audit artifact.

### Deliverables

- docs target inventory
- CI artifact and workflow inventory
- stale-output guard findings
- staged generated-file guard findings
- Day 4 guard audit artifact

### Completion Criteria

- current generated API guard coverage is explicit
- publication metadata gaps are visible before the decision
- workflow behavior is tied to checked-in scripts or tests

---

## Day 5: Publication Option Decision Matrix

**Title:** Publication Decision Matrix
**Theme:** Compare hosted, retained artifact, committed output, and local-only
options
**Time estimate:** 12 hours

### Tasks

1. Define decision criteria for user value, maintenance cost, reviewability,
   freshness, reproducibility, and CI complexity.
2. Evaluate hosted generated API HTML publication.
3. Evaluate retained CI artifact publication.
4. Evaluate committed generated HTML output.
5. Evaluate continued local-only status with stronger enforcement.
6. Write the Day 5 publication decision matrix artifact.

### Deliverables

- publication option criteria
- option-by-option tradeoff matrix
- recommended decision candidate
- rejected-option rationale
- Day 5 decision matrix artifact

### Completion Criteria

- every project-plan option is evaluated
- tradeoffs are concrete enough for implementation planning
- rejected options have evidence-backed rationale

---

## Day 6: Product Decision Record

**Title:** Product Decision Record
**Theme:** Choose the supported generated API HTML path and define acceptance
requirements
**Time estimate:** 12 hours

### Tasks

1. Select the supported generated API HTML product path.
2. Document the decision, alternatives, rationale, and non-goals.
3. Define acceptance requirements for implementation, freshness, staging, and
   navigation.
4. Define which claims are allowed after the decision.
5. Define which claims remain unsupported.
6. Write the Day 6 product-decision artifact.

### Deliverables

- generated API HTML product decision
- implementation acceptance requirements
- supported and unsupported claim list
- rejected alternatives record
- Day 6 product-decision artifact

### Completion Criteria

- one supported path is selected
- implementation work has a clear pass/fail contract
- documentation updates cannot overstate the selected path

---

## Day 7: Implementation Design

**Title:** Implementation Design
**Theme:** Design the selected publication or local-only enforcement path
before editing behavior
**Time estimate:** 12 hours

### Tasks

1. Identify files, scripts, Make targets, tests, and workflows to change.
2. Define command names, artifact names, metadata paths, and failure messages.
3. Define how generated output freshness will be verified.
4. Define how staged generated files will be accepted or rejected.
5. Define how navigation docs will point to the selected supported API path.
6. Write the Day 7 implementation-design artifact.

### Deliverables

- implementation file list
- command and artifact naming plan
- freshness and staging design
- navigation update design
- Day 7 implementation-design artifact

### Completion Criteria

- planned edits map directly to the Day 6 decision
- freshness and staging behavior has a testable design
- no implementation area is left without an owner

---

## Day 8: Implementation Batch 1

**Title:** Core Implementation Batch
**Theme:** Implement the core publication or local-only enforcement behavior
**Time estimate:** 12 hours

### Tasks

1. Implement the primary Make target, script, workflow step, or guard required
   by the selected path.
2. Add deterministic failure messages for missing generated API prerequisites.
3. Ensure generated output paths are created, ignored, retained, or rejected
   according to the decision.
4. Add a focused test or check for the core behavior.
5. Capture early validation output.
6. Write the Day 8 implementation artifact.

### Deliverables

- core implementation changes
- initial test or guard coverage
- generated output policy behavior
- early validation notes
- Day 8 implementation artifact

### Completion Criteria

- the selected path is implemented at a usable first pass
- failure behavior is explicit
- generated output handling matches the product decision

---

## Day 9: Implementation Batch 2

**Title:** Enforcement Completion
**Theme:** Complete implementation details and close behavior gaps from the
first batch
**Time estimate:** 12 hours

### Tasks

1. Complete remaining script, target, test, workflow, or metadata edits.
2. Tighten path handling for local and CI execution.
3. Confirm generated API commands do not rely on unstated working-directory
   assumptions.
4. Confirm generated output is not accidentally committed unless the decision
   explicitly requires it.
5. Record remaining risks and deferrals.
6. Write the Day 9 enforcement-completion artifact.

### Deliverables

- completed implementation path
- path-handling notes
- generated output commit-policy evidence
- remaining-risk list
- Day 9 enforcement artifact

### Completion Criteria

- implementation satisfies the Day 6 acceptance requirements
- local and CI command assumptions are documented
- accidental generated-output publication paths are guarded

---

## Day 10: Freshness And Staging Guard

**Title:** Freshness And Staging Guard
**Theme:** Add or tighten fail-closed checks for stale output, missing pages,
staged generated files, and publication metadata
**Time estimate:** 12 hours

### Tasks

1. Add or tighten stale-output checks for generated API HTML.
2. Add or tighten missing-page checks for required generated API entry points.
3. Add or tighten staged generated-file checks.
4. Add or tighten publication metadata checks for the selected path.
5. Add tests that exercise both passing and failing guard behavior where
   practical.
6. Write the Day 10 freshness and staging artifact.

### Deliverables

- stale-output guard
- missing-page guard
- staged-file guard
- publication metadata guard
- Day 10 guard artifact

### Completion Criteria

- required generated API artifacts cannot silently drift
- staged generated files follow the selected product policy
- guard failures are actionable

---

## Day 11: Navigation And Claim Update

**Title:** Navigation And Claim Update
**Theme:** Update public and maintainer docs to point to the supported API path
without overstating coverage
**Time estimate:** 12 hours

### Tasks

1. Update README API and CI wording for the selected generated API path.
2. Update API reference docs to point users to the supported generated API
   behavior.
3. Update maintainer guide instructions for generation, publication, retention,
   or local-only enforcement.
4. Update support-tier docs with supported and unsupported claims.
5. Ensure terminology is consistent across generated API, docs generation,
   publication, and local-only wording.
6. Write the Day 11 navigation and claim artifact.

### Deliverables

- README navigation update
- API reference update
- maintainer guide update
- support-tier wording update
- Day 11 navigation artifact

### Completion Criteria

- users can find the supported API documentation path
- maintainers can reproduce or enforce the selected behavior
- docs claims stay within validated evidence

---

## Day 12: Focused Verification

**Title:** Focused Verification
**Theme:** Run the generated API and docs checks required by the selected path
**Time estimate:** 12 hours

### Tasks

1. Run docs generation for generated API HTML.
2. Run docs checks and generated API checks.
3. Run freshness and staging guards.
4. Run targeted tests for any new scripts or guard logic.
5. Run `git diff --check`.
6. Write the Day 12 focused-verification artifact.

### Deliverables

- docs-generation validation record
- docs-check validation record
- generated API check record
- freshness and staging guard record
- Day 12 verification artifact

### Completion Criteria

- selected generated API path passes its focused checks
- whitespace and staged-file checks are clean
- validation commands are reproducible from the artifact

---

## Day 13: Integrated Validation And Reconciliation

**Title:** Integrated Validation
**Theme:** Reconcile docs, CI, guards, and claims before closeout
**Time estimate:** 11 hours

### Tasks

1. Re-run the full selected validation chain after documentation updates.
2. Inspect diffs for generated output policy violations.
3. Confirm all README, API reference, maintainer, and support-tier claims align
   with the product decision.
4. Confirm Sprint 179 artifacts cover all project-plan items.
5. Record residual risks and explicit deferrals.
6. Write the Day 13 integrated-validation artifact.

### Deliverables

- integrated validation record
- final claim reconciliation notes
- project-plan item coverage checklist
- residual-risk and deferral list
- Day 13 validation artifact

### Completion Criteria

- generated API implementation, guards, and docs tell the same story
- every Sprint 179 item has evidence or a documented deferral
- no unsupported generated API publication claim remains

---

## Day 14: Sprint Closeout And Handoff

**Title:** Closeout And Handoff
**Theme:** Prepare Sprint 179 closeout artifacts and downstream handoff
**Time estimate:** 11 hours

### Tasks

1. Finalize Sprint 179 working notes.
2. Summarize completed deliverables and validation evidence.
3. Capture the generated API product decision in closeout-ready form.
4. Identify any follow-up work for later Epic 16 sprints.
5. Prepare retrospective inputs with completed work, risks, and lessons.
6. Write the Day 14 closeout and handoff artifact.

### Deliverables

- finalized Sprint 179 working notes
- generated API decision summary
- validation evidence summary
- follow-up and deferral list
- Day 14 closeout artifact

### Completion Criteria

- Sprint 179 has a complete closeout trail
- future sprints inherit clear generated API status and guard expectations
- retrospective inputs are ready without re-auditing the sprint
