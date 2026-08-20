# Sprint 173 Plan: Generated API HTML Publication Closure

**Sprint Duration:** 14 days
**Goal:** Decide and implement the supported generated API HTML publication
path. This sprint implements the Sprint 173 section of
`docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md`, but the active merged Sprint 173
project-plan section lives in `docs/planning/EPIC_15/PROJECT_PLAN.md` and has
the title "Sprint 173: Generated API HTML Publication Closure".

**Starting Point:** Sprint 173 begins from:

- Sprint 172 public-header coherence work, especially the cleaned
  `include/sparse_lu.h` generated-doc input;
- existing local generated API HTML and freshness-check infrastructure;
- Epic 15 claim-gate conventions for separating published evidence from
  unsupported package, ABI, runtime-loader, platform, performance,
  external-parity, and state-of-the-art claims;
- Sprint 170 static-first package/shared-library ABI deferral guard;
- Sprint 171 package-manager deferral guard;
- current README, docs indexes, API reference docs, Doxygen configuration, and
  generated-output staging conventions.

The sprint must:

- decide whether generated API HTML is hosted, committed, artifact-only, or
  local-only;
- audit generation scripts, inputs, output paths, ignored files, and freshness
  behavior;
- implement the selected publication or local-only enforcement path;
- add or update freshness gates so the selected generated API status remains
  accurate;
- update README and documentation navigation so users can find the supported
  API reference;
- run generator, freshness, docs, guard, and relevant build checks.

**End State:** Sprint 173 leaves behind:

- a source-controlled generated API HTML publication decision;
- the implemented publication path or explicit local-only enforcement;
- updated docs navigation for the supported API reference surface;
- freshness and staging checks that match the selected decision;
- Sprint 173 working notes, daily artifacts, and closeout records.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 173 project-plan estimate.

---

## Day 1: Sprint Intake And Generated API Boundary

**Title:** API Docs Intake
**Theme:** Establish Sprint 173 scope, inherited claim boundaries, and
generated-output constraints
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 173 section of
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.
2. Review Sprint 172 closeout, retrospective, and LU header/docs guard
   handoff.
3. Review existing generated API HTML, generation scripts, freshness checks,
   documentation indexes, and ignore/staging rules.
4. Create Sprint 173 working notes and artifact directory structure.
5. Record the prompt path/source-artifact mismatch.
6. Write the Day 1 API-docs intake artifact.

### Deliverables

- Sprint 173 working-notes baseline
- artifact directory structure
- source artifact note
- inherited generated-doc and claim-boundary summary
- Day 1 API-docs intake artifact

### Completion Criteria

- Sprint 173 scope is tied to the active Epic 15 project plan
- generated API HTML publication choices are explicit before implementation
- unsupported package, ABI, platform, performance, external-parity, and
  state-of-the-art claims remain protected

---

## Day 2: Generator And Output Inventory

**Title:** Generator Inventory
**Theme:** Inventory generated API HTML inputs, commands, outputs, ignore
rules, and freshness assumptions
**Time estimate:** 12 hours

### Tasks

1. Locate Doxygen/configuration files, generation scripts, Make targets, CI
   references, and documentation links for generated API output.
2. Inventory generated output paths and determine which paths are tracked,
   ignored, artifact-only, or local-only.
3. Identify source inputs that should affect generated API freshness.
4. Identify stale, untracked, or ambiguous generated-output artifacts.
5. Map existing freshness checks and gaps.
6. Write the Day 2 generator inventory artifact.

### Deliverables

- generator command inventory
- generated-output path inventory
- source-input freshness map
- ignored/tracked/staged-output classification
- Day 2 generator inventory artifact

### Completion Criteria

- generation inputs and outputs are visible
- freshness behavior is understood before a publication decision
- generated output is not staged unintentionally

---

## Day 3: Publication Decision Matrix

**Title:** Publication Options
**Theme:** Compare hosted, committed, artifact-only, and local-only API HTML
paths against maintenance cost and claim risk
**Time estimate:** 12 hours

### Tasks

1. Define candidate publication modes: hosted site, committed HTML,
   CI artifact, and local-only generation.
2. Compare each mode against repository size, reviewability, CI reliability,
   freshness enforcement, user discoverability, and release process.
3. Identify claim risks for stale generated docs, unsupported hosting, package
   assumptions, ABI assumptions, and platform assumptions.
4. Identify required gates for each candidate mode.
5. Recommend a selected mode for Day 4 decision.
6. Write the Day 3 publication decision matrix artifact.

### Deliverables

- publication-mode comparison matrix
- maintenance and CI risk assessment
- freshness-gate requirements by mode
- recommended publication path
- Day 3 publication options artifact

### Completion Criteria

- all viable publication modes have explicit tradeoffs
- no publication support claim is made before a decision record exists
- selected-mode prerequisites are clear

---

## Day 4: Publication Decision Record

**Title:** Publication Decision
**Theme:** Choose the supported generated API HTML publication path and define
its supported and unsupported claims
**Time estimate:** 12 hours

### Tasks

1. Convert the Day 3 recommendation into a formal publication decision.
2. Define the supported generated API reference surface for users.
3. Define unsupported claims: stale generated HTML freshness, unselected
   hosting modes, package-manager distribution, shared-library ABI,
   runtime-loader behavior, broad platform parity, performance, and
   state-of-the-art API completeness.
4. Define required checks for the selected mode.
5. Define implementation scope for Days 5 through 9.
6. Write the Day 4 generated API publication decision artifact.

### Deliverables

- selected generated API HTML publication mode
- supported claim list
- unsupported claim list
- required freshness/staging checks
- Day 4 decision artifact

### Completion Criteria

- exactly one publication path is selected or local-only enforcement is
  explicitly selected
- the selected path has clear validation requirements
- unselected publication modes remain non-claims

---

## Day 5: Generator Command Normalization Design

**Title:** Generator Design
**Theme:** Design the command, target, and output-path changes needed for the
selected generated API path
**Time estimate:** 12 hours

### Tasks

1. Review the selected Day 4 path against existing generator scripts and Make
   or CI targets.
2. Design command naming, output directory handling, cleanup behavior, and
   reproducibility expectations.
3. Define how generator failures should surface locally and in CI.
4. Define generated-output staging rules for the selected path.
5. Identify docs and guard updates needed for Day 6 or later.
6. Write the Day 5 generator normalization design artifact.

### Deliverables

- generator command design
- output-path and cleanup design
- local/CI failure behavior
- staging and ignore-rule expectations
- Day 5 generator design artifact

### Completion Criteria

- implementation is scoped before editing generator commands
- output paths and staging rules are unambiguous
- generated output cannot drift silently by design

---

## Day 6: Generator Command Implementation

**Title:** Generator Implementation
**Theme:** Implement the selected generator command, output-path, cleanup, or
local-only enforcement changes
**Time estimate:** 12 hours

### Tasks

1. Implement the Day 5 generator command or enforcement changes.
2. Update Make targets, scripts, or configuration files as required by the
   selected path.
3. Preserve existing docs/build behavior outside generated API HTML scope.
4. Run the generator or local-only enforcement proof.
5. Record generated-output staging results.
6. Write the Day 6 generator implementation artifact.

### Deliverables

- implemented generator command or local-only enforcement path
- updated scripts/targets/configuration as needed
- generated-output staging evidence
- Day 6 implementation artifact

### Completion Criteria

- selected generation path is executable or explicitly enforced
- generated output appears only where the decision allows it
- failures are visible to maintainers

---

## Day 7: Freshness Gate Design

**Title:** Freshness Design
**Theme:** Design freshness checks that prove generated API status is accurate
for the selected publication path
**Time estimate:** 12 hours

### Tasks

1. Identify source inputs that should invalidate generated API HTML:
   public headers, Doxygen config, generator scripts, documentation indexes,
   and selected claim records.
2. Define freshness metadata or comparison behavior for the selected path.
3. Define acceptable local-only, artifact-only, committed, or hosted
   freshness semantics.
4. Define generated-output exclusion rules for unselected paths.
5. Plan CI or local check integration points.
6. Write the Day 7 freshness gate design artifact.

### Deliverables

- source-input freshness list
- freshness check design
- unselected-output exclusion rules
- CI/local integration plan
- Day 7 freshness design artifact

### Completion Criteria

- freshness checks match the Day 4 publication decision
- stale generated API docs cannot be represented as current
- unselected generated-output paths remain rejected or ignored

---

## Day 8: Freshness Gate Implementation

**Title:** Freshness Gate
**Theme:** Add or update the generated API freshness and staging checks
**Time estimate:** 12 hours

### Tasks

1. Implement the Day 7 freshness check or selected-path verification script.
2. Add or update CI/local target integration where appropriate.
3. Add failure messages that explain how maintainers should regenerate,
   publish, or remove generated output.
4. Run the freshness check in passing and, where practical, fail-mode proof.
5. Record generated-output staging results.
6. Write the Day 8 freshness implementation artifact.

### Deliverables

- implemented freshness check
- updated CI/local target integration if selected
- passing freshness evidence
- Day 8 freshness gate artifact

### Completion Criteria

- selected generated API status is mechanically checkable
- check failures are actionable
- no unselected generated output is staged unintentionally

---

## Day 9: Documentation Navigation Design

**Title:** Navigation Design
**Theme:** Design README and docs-index navigation for the supported API
reference surface
**Time estimate:** 12 hours

### Tasks

1. Review README, docs index pages, API reference docs, tutorial links, and
   maintainer guide references.
2. Identify user-facing entry points for the supported API reference surface.
3. Draft navigation wording for the selected Day 4 publication mode.
4. Define non-claim wording for unselected hosting, package, ABI, platform,
   performance, and generated freshness surfaces.
5. Define validation commands for docs navigation changes.
6. Write the Day 9 navigation design artifact.

### Deliverables

- README/docs navigation map
- selected API reference link wording
- unsupported-claim wording constraints
- Day 9 navigation design artifact

### Completion Criteria

- users can find the supported API reference path by design
- docs wording matches the selected publication decision
- unselected publication modes remain non-claims

---

## Day 10: Documentation Navigation Update

**Title:** Navigation Update
**Theme:** Update README and docs indexes so users can find the supported API
reference
**Time estimate:** 12 hours

### Tasks

1. Update README and docs indexes according to the Day 9 design.
2. Update maintainer guidance for generating, checking, publishing, or
   intentionally keeping generated API HTML local-only.
3. Keep navigation wording aligned with selected generated API support.
4. Avoid package-manager, shared-library, dynamic ABI, runtime-loader,
   platform parity, performance, external-parity, and state-of-the-art claims.
5. Run targeted docs and claim scans.
6. Write the Day 10 navigation update artifact.

### Deliverables

- updated README/docs navigation
- updated maintainer guidance
- docs and claim-scan results
- Day 10 navigation update artifact

### Completion Criteria

- supported API reference location is discoverable
- documentation matches the Day 4 publication decision
- unsupported claim boundaries remain intact

---

## Day 11: Integrated Generator Validation

**Title:** Generator Validation
**Theme:** Run the generator, freshness gate, staging checks, and focused docs
checks together
**Time estimate:** 12 hours

### Tasks

1. Run the selected generator or local-only enforcement command.
2. Run the generated API freshness gate.
3. Run generated-output staging and unselected-output checks.
4. Run docs navigation and claim scans.
5. Run package-manager/static-package deferral guards if docs touch adoption,
   package, ABI, runtime-loader, or platform wording.
6. Write the Day 11 integrated generator validation artifact.

### Deliverables

- generator/local-only enforcement result
- freshness result
- generated-output staging result
- docs/claim-scan result
- Day 11 validation artifact

### Completion Criteria

- selected generated API status is proven locally
- no unintended generated output is staged
- failures stop the sprint for user input

---

## Day 12: CI And Maintenance Surface Review

**Title:** Maintenance Review
**Theme:** Reconcile CI, maintainer commands, and report/freshness ownership for
the selected generated API path
**Time estimate:** 12 hours

### Tasks

1. Review CI workflows, Make targets, maintainer docs, and report-index rows
   affected by generated API publication.
2. Confirm the selected freshness gate has a clear owner and invocation path.
3. Confirm generated API output is either published, artifact-only, committed,
   or local-only exactly as selected.
4. Confirm no unsupported package, ABI, runtime-loader, platform, performance,
   external-parity, or state-of-the-art claims are introduced.
5. Record any CI or report-index residuals.
6. Write the Day 12 maintenance surface artifact.

### Deliverables

- CI/local maintenance surface review
- freshness-owner review
- claim-boundary review
- residual list
- Day 12 maintenance artifact

### Completion Criteria

- maintainers know which command/check owns generated API status
- CI/local behavior matches the publication decision
- residuals are explicit and bounded

---

## Day 13: Integrated Claim Review

**Title:** Claim Review
**Theme:** Reconcile generated API publication, navigation, freshness gates,
and inherited non-claims
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 173 artifacts and working notes.
2. Reconcile the selected generated API path against docs navigation,
   generator behavior, freshness checks, and staging rules.
3. Run targeted hosted/committed/artifact/local-only claim scans.
4. Run package-manager and static-package deferral guards if claim wording
   touches their surfaces.
5. Identify Sprint 174 handoff needs.
6. Write the Day 13 integrated claim review artifact.

### Deliverables

- generated API claim review
- selected/unselected publication-mode summary
- claim-scan results
- Sprint 174 handoff list
- Day 13 claim-review artifact

### Completion Criteria

- generated API publication claim boundary is internally coherent
- docs navigation and freshness gates match the selected path
- unselected publication modes remain non-claims

---

## Day 14: Sprint Closeout And Sprint 174 Handoff

**Title:** Sprint Closeout
**Theme:** Reconcile Sprint 173 deliverables, final validation, and handoff to
Sprint 174 external comparison work
**Time estimate:** 10 hours

### Tasks

1. Reconcile Sprint 173 outcomes against project-plan items 173.1 through
   173.6.
2. Confirm publication decision, implementation/enforcement, freshness gate,
   docs navigation, and validation artifacts are complete.
3. Re-run final lightweight hygiene checks and required focused checks.
4. Confirm no generated output is staged unintentionally outside the selected
   path.
5. Prepare Sprint 174 handoff notes for external comparison publication work.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables

- final Sprint 173 validation record
- project-plan item reconciliation
- generated-output staging check
- Sprint 174 handoff
- Day 14 sprint-closeout artifact

### Completion Criteria

- generated API HTML publication status is source-controlled and validated
- documentation and freshness gates match the selected generated API path
- Sprint 174 can begin from clear generated-doc publication boundaries
