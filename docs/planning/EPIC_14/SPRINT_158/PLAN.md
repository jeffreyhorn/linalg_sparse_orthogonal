# Sprint 158 Plan: Generated API HTML Publication Closure

**Sprint Duration:** 14 days
**Goal:** Close the generated API reference residual with either committed
generated HTML and coverage evidence or an explicit no-commit product decision
with a recurring freshness guard. This sprint implements the Sprint 158 section
of `docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Starting Point:** Sprint 158 begins from:
- Sprint 157 baseline inventory, evidence contracts, quality surface map, risk
  register, and generated API reference handoff;
- current Doxygen configuration in `Doxyfile`;
- public headers under `include/`, including generated install behavior for
  `sparse_version.h` from `include/sparse_version.h.in`;
- existing source-header-first API documentation policy in
  `docs/api_reference.md` and maintainer documentation;
- `docs/api/`, build outputs, and generated documentation outputs currently
  treated as ignored/generated artifacts unless this sprint explicitly changes
  that policy.

The sprint must:
- run and record the current generated API documentation baseline;
- capture Doxygen warnings, generated page inventory, missing page coverage,
  and generated version-header behavior;
- make a product decision for generated HTML publication: source-controlled,
  CI-published, or local-only with recurring freshness guard;
- add or document a public-header page-coverage check for the intended input
  set;
- triage Doxygen warnings and fix only the warnings selected for this sprint;
- align README, API reference, maintainer guide, tutorial, and header-doc
  policy wording with the selected publication decision;
- preserve unsupported claim boundaries around generated docs, hosted
  evidence, package, ABI, platform, performance, and state-of-the-art wording;
- leave Sprint 159 with a clear hosted-report handoff.

**End State:** Sprint 158 leaves behind:
- generated API HTML publication decision;
- Doxygen warning inventory and triage results;
- public-header page-coverage evidence or explicit coverage guard;
- updated documentation policy and user-facing references;
- validation evidence for docs generation and documentation hygiene;
- Sprint 159 hosted-report handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 158 project-plan estimate.

---

## Day 1: Sprint Intake And API Docs Baseline Setup

**Title:** API Docs Intake
**Theme:** Establish Sprint 158 scope, artifact layout, inputs, and stop
conditions
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 158 section of
   `docs/planning/EPIC_14/PROJECT_PLAN.md`.
2. Review Sprint 157 Day 9 evidence-contract templates, Day 10 quality surface
   map, Day 12 risk register, and Day 14 Sprint 158 handoff.
3. Create Sprint 158 working notes and artifact directory structure.
4. Record the exact branch, commit, Doxygen tool availability, and current
   generated-doc tracking state.
5. Identify API documentation inputs: `Doxyfile`, public headers,
   `include/sparse_version.h.in`, `docs/api_reference.md`, README, tutorial,
   and maintainer guide.
6. Write the Day 1 API-docs intake artifact.

### Deliverables
- Sprint 158 working-notes baseline
- artifact directory structure
- API documentation input inventory
- initial stop-condition and risk list
- Day 1 API-docs intake artifact

### Completion Criteria
- Sprint 158 scope is tied to the Epic 14 project plan and Sprint 157 handoff
- all generated API documentation inputs are identified
- unsupported generated-evidence and broad documentation claims are blocked
  before generation work begins

---

## Day 2: Doxygen Baseline Run

**Title:** Doxygen Baseline
**Theme:** Run the current documentation generator and capture outputs without
changing policy
**Time estimate:** 12 hours

### Tasks
1. Run the current documentation generation command, expected to be
   `make docs`, or record the exact blocker if unavailable.
2. Capture Doxygen version, command line, exit status, warnings, and generated
   output location.
3. Record whether `docs/api/html/` or other generated paths are tracked,
   ignored, absent, or stale.
4. Capture generated page categories, index files, and obvious missing public
   surfaces.
5. Record whether generated outputs include local machine paths or other
   non-portable metadata.
6. Write the Day 2 Doxygen baseline artifact.

### Deliverables
- Doxygen command and environment record
- warning log summary
- generated output inventory
- generated-output tracking-state note
- Day 2 Doxygen baseline artifact

### Completion Criteria
- generated API documentation can be reproduced or has an explicit blocker
- warning and output evidence is recorded before policy decisions
- no generated HTML is promoted as public evidence by accident

---

## Day 3: Public Header And Generated Version Coverage Map

**Title:** Header Coverage Map
**Theme:** Define the intended public header input set and generated
version-header treatment
**Time estimate:** 12 hours

### Tasks
1. Inventory checked-in public headers under `include/*.h`.
2. Record how generated `sparse_version.h` is produced from
   `include/sparse_version.h.in`.
3. Map Doxygen inputs to public headers and generated installed-header
   behavior.
4. Identify missing generated pages, undocumented headers, and headers whose
   comments need explicit exclusion.
5. Decide whether coverage should count checked-in headers only or checked-in
   headers plus generated installed header behavior.
6. Write the Day 3 public-header coverage-map artifact.

### Deliverables
- public-header input list
- generated version-header treatment note
- header-to-page coverage matrix
- missing-page or exclusion list
- Day 3 header coverage artifact

### Completion Criteria
- page coverage has an explicit source set
- generated version-header behavior is not confused with checked-in public
  headers
- later coverage checks can compare expected headers against generated pages

---

## Day 4: Warning Triage Policy

**Title:** Warning Triage
**Theme:** Classify Doxygen warnings into fix, defer, exclude, or blocker
categories
**Time estimate:** 12 hours

### Tasks
1. Normalize the Day 2 Doxygen warning log into stable warning categories.
2. Classify each warning as selected for this sprint, deferred with owner,
   explicit exclusion, or release blocker.
3. Identify warnings caused by generated files, internal-only symbols,
   unsupported API claims, or Doxygen configuration issues.
4. Record which warning fixes would require public-header comment changes and
   which would require code declaration changes.
5. Define escalation rules for any `.c` or public `.h` edits.
6. Write the Day 4 warning triage artifact.

### Deliverables
- warning category inventory
- fix/defer/exclude/blocker table
- owner and blocker notes
- quality-gate escalation notes
- Day 4 warning triage artifact

### Completion Criteria
- every warning has an explicit disposition
- warnings selected for closure are bounded to Sprint 158
- any code or public-header edits are pre-classified for full quality gates

---

## Day 5: Publication Decision Options

**Title:** Publication Options
**Theme:** Compare committed HTML, CI-published HTML, and local-only freshness
guard options
**Time estimate:** 12 hours

### Tasks
1. Define the source-controlled `docs/api/html/` publication option, including
   repository size, review noise, regeneration, and freshness implications.
2. Define the CI-published artifact or hosted-pages option, including artifact
   retention, branch policy, and public claim boundaries.
3. Define the local-only option, including recurring freshness guard and
   documentation wording requirements.
4. Evaluate each option against Sprint 157 evidence contracts and stop
   conditions.
5. Recommend the Sprint 158 product decision with tradeoffs and required
   validation.
6. Write the Day 5 publication-options artifact.

### Deliverables
- publication option comparison
- selected recommendation
- required follow-up checks
- claim-boundary notes
- Day 5 publication-options artifact

### Completion Criteria
- each viable publication path has explicit cost and evidence implications
- rejected paths have clear reasons
- the selected path does not overclaim generated documentation freshness

---

## Day 6: Publication Decision Implementation Plan

**Title:** Publication Decision
**Theme:** Convert the selected generated-doc policy into concrete repository
or CI changes
**Time estimate:** 12 hours

### Tasks
1. Finalize whether generated HTML will be committed, CI-published, or kept
   local-only with a recurring guard.
2. Identify exact files to update for the selected policy: `.gitignore`,
   `Doxyfile`, Make targets, CI workflows, docs, scripts, or artifacts.
3. Define rollback and stale-output prevention rules for the selected policy.
4. Record publication support-tier wording for README, API reference, and
   maintainer guide.
5. Create an implementation checklist for Days 7 through 11.
6. Write the Day 6 publication-decision artifact.

### Deliverables
- final publication decision
- file-change checklist
- stale-output prevention rules
- support-tier wording draft
- Day 6 publication-decision artifact

### Completion Criteria
- the selected publication path has a concrete implementation checklist
- repository tracking policy for generated HTML is unambiguous
- unsupported hosted or freshness claims remain blocked

---

## Day 7: Page Coverage Check Design

**Title:** Coverage Check Design
**Theme:** Design the generated API page-coverage check for the selected
public-header source set
**Time estimate:** 12 hours

### Tasks
1. Choose whether the coverage check is implemented as a script, Make target,
   documented manual check, CI guard, or combination.
2. Define expected inputs, generated output paths, pass/fail behavior, and
   missing-page reporting.
3. Include generated version-header handling from Day 3.
4. Define how the check avoids depending on ignored local artifacts unless it
   runs generation first.
5. Write test or validation expectations for the coverage check.
6. Write the Day 7 page-coverage design artifact.

### Deliverables
- page-coverage check design
- expected-input and output-path definitions
- generated version-header handling rule
- validation expectations
- Day 7 page-coverage design artifact

### Completion Criteria
- coverage expectations are deterministic
- missing public-header pages can be reported with concrete paths
- local-only and published generated outputs are not conflated

---

## Day 8: Page Coverage Check Implementation

**Title:** Coverage Check Implementation
**Theme:** Implement or document the page-coverage guard selected on Day 7
**Time estimate:** 12 hours

### Tasks
1. Add the selected script, Make target, CI step, or documentation procedure.
2. Ensure the check uses the public-header source set from Day 3.
3. Ensure generated version-header behavior is explicitly handled.
4. Add focused validation or self-check coverage for the new guard when
   applicable.
5. Run the guard locally or record any unavailable tool blocker.
6. Write the Day 8 coverage implementation artifact.

### Deliverables
- implemented or documented page-coverage guard
- local run output or blocker record
- focused validation notes
- Day 8 coverage implementation artifact

### Completion Criteria
- the guard can detect missing expected generated pages or has an explicit
  documented manual procedure
- generated version-header behavior matches the Day 3 policy
- no code or public-header declaration changes occur without required gates

---

## Day 9: Selected Warning Fixes

**Title:** Warning Fix Batch
**Theme:** Fix the selected Doxygen warnings without widening API or support
claims
**Time estimate:** 12 hours

### Tasks
1. Apply only the warning fixes selected on Day 4.
2. Keep public-header changes comment-only unless a deliberate declaration
   change is separately justified.
3. Avoid introducing package, ABI, platform, performance, or broad
   state-of-the-art claims in generated docs.
4. Re-run Doxygen generation or the focused warning check after the fix batch.
5. Record remaining warnings and their dispositions.
6. Write the Day 9 warning-fix artifact.

### Deliverables
- selected warning fixes
- regenerated warning summary
- remaining-warning disposition table
- Day 9 warning-fix artifact

### Completion Criteria
- selected warnings are fixed or explicitly reclassified
- remaining warnings have owners and blockers
- any `.c` or public `.h` changes trigger the full required quality gate

---

## Day 10: Documentation Policy Alignment

**Title:** Policy Alignment
**Theme:** Align public and maintainer documentation with the generated API
publication decision
**Time estimate:** 12 hours

### Tasks
1. Update `docs/api_reference.md` to reflect the selected generated-doc policy.
2. Update `docs/maintainer_guide.md` with generation, freshness, coverage, and
   publication maintenance rules.
3. Update README and tutorial references if the user-facing API reference path
   changes.
4. Preserve source-header-first wording unless Day 6 explicitly changed the
   policy.
5. Add explicit non-claim wording for local-only, hosted, or committed output
   boundaries as needed.
6. Write the Day 10 policy-alignment artifact.

### Deliverables
- updated API reference policy
- updated maintainer guidance
- README/tutorial alignment if needed
- non-claim wording updates
- Day 10 policy-alignment artifact

### Completion Criteria
- user-facing docs and maintainer docs describe the same generated-doc policy
- generated API docs are not represented as fresher than the selected evidence
  supports
- source-header-first ownership remains clear

---

## Day 11: Publication Path Finalization

**Title:** Publication Finalization
**Theme:** Finalize generated output, CI publication, or recurring guard
behavior
**Time estimate:** 12 hours

### Tasks
1. If committing generated HTML, regenerate it, check page coverage, inspect
   size and path churn, and stage only intended files.
2. If CI-publishing generated HTML, update workflow and artifact/publication
   documentation with exact support-tier wording.
3. If keeping generated HTML local-only, add the recurring freshness guard and
   document why generated files remain untracked.
4. Verify `.gitignore`, generated paths, and documentation agree.
5. Capture final publication-path evidence.
6. Write the Day 11 publication-finalization artifact.

### Deliverables
- finalized generated API publication path
- committed generated files, CI publication changes, or local-only freshness
  guard
- path tracking and ignore-policy evidence
- Day 11 publication-finalization artifact

### Completion Criteria
- generated output policy is implemented consistently
- no stale or unintended generated files are committed
- support claims match the chosen publication path

---

## Day 12: Validation And Freshness Evidence

**Title:** Validation Evidence
**Theme:** Run required docs generation, warning, coverage, and hygiene checks
for the touched surface
**Time estimate:** 12 hours

### Tasks
1. Run the selected docs generation command.
2. Run the warning and page-coverage checks.
3. Run documentation whitespace and diff hygiene checks.
4. Run any focused script or Make target added during the sprint.
5. If `.c` or public `.h` files changed, run
   `make format && make lint && make test`.
6. Write the Day 12 validation evidence artifact.

### Deliverables
- validation command log
- warning and coverage results
- docs hygiene results
- full C quality-gate results if required
- Day 12 validation artifact

### Completion Criteria
- all required quality checks pass before closeout
- skipped checks have explicit reasons and claim implications
- validation evidence supports the selected publication decision

---

## Day 13: Claim And Artifact Reconciliation

**Title:** Claim Reconciliation
**Theme:** Reconcile generated API claims, artifacts, and Sprint 159 handoff
before closeout
**Time estimate:** 12 hours

### Tasks
1. Audit README, API reference, maintainer guide, tutorial, and generated-doc
   wording against the selected publication decision.
2. Confirm no unsupported hosted freshness, package, ABI, platform,
   performance, parity, or state-of-the-art claims were introduced.
3. Reconcile Day 1 through Day 12 artifacts against Sprint 158 project-plan
   items.
4. Draft Sprint 159 hosted-report handoff, including required publication and
   freshness boundaries.
5. Record residuals not closed by Sprint 158.
6. Write the Day 13 reconciliation artifact.

### Deliverables
- claim audit
- artifact-to-item reconciliation
- Sprint 159 handoff draft
- residual list
- Day 13 reconciliation artifact

### Completion Criteria
- generated API documentation claims are evidence-bound
- Sprint 158 deliverables are either closed or explicitly residualized
- Sprint 159 starts with concrete hosted-report prerequisites

---

## Day 14: Sprint Closeout And Handoff

**Title:** Closeout Handoff
**Theme:** Finalize Sprint 158 artifacts, documentation state, and next-sprint
handoff
**Time estimate:** 12 hours

### Tasks
1. Finalize working notes with daily outcomes, commands, blockers, and
   validation status.
2. Write the Sprint 158 closeout artifact.
3. Update the Sprint 158 plan status or retrospective inputs if the local
   pattern requires it.
4. Confirm final git status includes only intended files.
5. Confirm final validation evidence remains current after final doc edits.
6. Publish the Sprint 159 hosted-report handoff.

### Deliverables
- completed Sprint 158 working notes
- Day 14 closeout artifact
- final validation and status notes
- Sprint 159 handoff
- retrospective-ready summary

### Completion Criteria
- all Sprint 158 artifacts are complete and internally consistent
- generated API publication decision is closed with evidence or explicit
  no-commit guard
- next sprint can begin without re-discovering generated-doc policy decisions
