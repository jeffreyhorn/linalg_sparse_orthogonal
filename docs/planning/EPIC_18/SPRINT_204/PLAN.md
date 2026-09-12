# Sprint 204 Plan: Generated API Publication Decision

**Sprint Duration:** 14 days
**Goal:** Decide and implement either hosted generated API publication or a
stronger local-only generated API policy.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 204 estimate in the Epic 18
project plan.

**Primary scope:** Reassess generated API HTML as a product and maintainer
surface, choose one policy path, implement the selected publication or
local-only guard changes, add freshness/link/coverage checks needed by that
policy, calibrate public and maintainer docs, and validate the resulting
claim boundary.

**Non-goals:** Broad API completeness claims, ABI compatibility claims,
package-manager readiness, state-of-the-art claims, solver behavior changes,
unrelated generated report publication, performance publication, or committed
generated HTML unless this sprint explicitly selects and guards that policy.

---

## Day 1: Generated API Intake

**Title:** API Intake
**Theme:** Establish Sprint 204 scope, inherited generated API policy, and
current validation surfaces.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 204 Epic 18 project-plan section and map items 204.1
   through 204.6 to artifacts, code surfaces, and validation commands.
2. Review the current generated API policy in `docs/api_reference.md`,
   README, INSTALL, maintainer guide, `.gitignore`, `Doxyfile`, and Make
   targets.
3. Review prior generated API closure artifacts from Sprint 158, Sprint 179,
   Sprint 186, and Epic 18 residual notes.
4. Create `WORKING_NOTES.md` with an item checklist, decision log, evidence
   map, risk register, validation matrix, and open questions.
5. Record explicit non-goals so the sprint cannot be interpreted as broad API,
   ABI, package, hosted docs, or release readiness support.

### Deliverables

- Sprint 204 working-notes scaffold.
- Item-to-artifact traceability map.
- Current generated API policy inventory.
- Initial risk register and validation matrix.

### Completion Criteria

- Every Sprint 204 item has an initial evidence path or artifact category.
- Current local-only generated API semantics are understood before any policy
  change.
- Unsupported API, ABI, package, and publication claims remain explicitly out
  of scope.

---

## Day 2: Current Doxygen Baseline

**Title:** Doxygen Baseline
**Theme:** Reproduce the current local generated API output and freshness
checks without changing policy.
**Time estimate:** 12 hours

### Tasks

1. Run or attempt `make docs-check` and `make api-docs-freshness`, recording
   exact command output or blockers.
2. Capture Doxygen version, generated output paths, warning count, checked
   public-header coverage, and local-only guard behavior.
3. Verify whether `docs/api/`, `docs/api/html/`, and generated staging paths
   are ignored, untracked, tracked, or accidentally staged.
4. Inventory public headers, generated installed headers, generated pages,
   source pages, indexes, and cross-links relevant to publication.
5. Write the Day 2 baseline artifact with the current pass/fail state.

### Deliverables

- Doxygen and API freshness command record.
- Generated output tracking-state inventory.
- Public-header and generated-page baseline.
- Day 2 baseline artifact.

### Completion Criteria

- The existing local generated API policy is backed by current command
  evidence or an explicit environment blocker.
- Any generated output visible to git is classified before decision work.
- No publication path is inferred from local generated output alone.

---

## Day 3: Publication Option Inventory

**Title:** Option Inventory
**Theme:** Compare hosted, retained-artifact, committed-output, and stronger
local-only generated API policy options.
**Time estimate:** 12 hours

### Tasks

1. Define the hosted generated API publication option, including workflow,
   hosting, retention, freshness, link, and access implications.
2. Define the retained CI artifact option, including artifact naming,
   retention period, discoverability, and stale-output risks.
3. Define the committed generated HTML option, including repository size,
   review noise, regeneration discipline, and merge-conflict risks.
4. Define the stronger local-only option, including guards, docs wording,
   staging prevention, and user guidance.
5. Score each option by user value, implementation cost, maintenance cost,
   claim risk, review burden, and testability.

### Deliverables

- Generated API publication option matrix.
- Cost, risk, and evidence scoring.
- Rejected-option preliminary notes.
- Day 3 option inventory artifact.

### Completion Criteria

- Each viable policy path has concrete implementation and validation needs.
- Publication choices are separated from API completeness and ABI claims.
- Review burden and stale-output risks are documented for every option.

---

## Day 4: Decision Criteria And Acceptance Gate

**Title:** Decision Gate
**Theme:** Convert product and maintainer tradeoffs into an explicit acceptance
gate for the selected policy.
**Time estimate:** 12 hours

### Tasks

1. Define the acceptance criteria for selecting hosted publication,
   retained artifacts, committed generated output, or stronger local-only
   policy.
2. Identify the minimum freshness, coverage, link, staging, and workflow
   checks required for each acceptable policy.
3. Specify the claim boundaries that must appear in README, INSTALL,
   `docs/api_reference.md`, maintainer guide, and planning closeout.
4. Define rollback rules for stale generated output, failed Doxygen runs,
   broken links, and accidental hosted/committed publication.
5. Write the Day 4 acceptance-gate artifact and update the working-notes
   decision log.

### Deliverables

- Generated API decision acceptance gate.
- Required validation checklist by policy path.
- Claim-boundary wording requirements.
- Rollback and stale-output prevention rules.

### Completion Criteria

- Item 204.1 has a concrete decision framework before the decision is made.
- The selected policy cannot pass without matching validation coverage.
- Generated API docs cannot imply unsupported ABI, package, or completeness
  claims.

---

## Day 5: Product Decision

**Title:** Product Decision
**Theme:** Choose the Sprint 204 generated API policy and freeze the
implementation boundary.
**Time estimate:** 12 hours

### Tasks

1. Apply the Day 4 acceptance gate to the option matrix.
2. Select exactly one policy path: hosted publication, retained artifact
   publication, committed generated output, or stronger local-only policy.
3. Document rejected paths with specific reasons and residuals.
4. Define the exact files, workflows, scripts, docs, and tests allowed to
   change for the selected path.
5. Record the implementation plan for Days 6 through 11.

### Deliverables

- Explicit generated API publication decision.
- Rejected-option residual ledger.
- Selected implementation boundary.
- Day 5 product-decision artifact.

### Completion Criteria

- Item 204.1 is complete with one selected policy and documented rationale.
- Unselected publication paths remain blocked unless a future sprint reopens
  them.
- Implementation can proceed without ambiguity about supported claims.

---

## Day 6: Workflow And Tracking Design

**Title:** Tracking Design
**Theme:** Design workflow, artifact, Pages, ignore, or staging changes needed
by the selected policy.
**Time estimate:** 12 hours

### Tasks

1. Map the selected policy to `.gitignore`, `Doxyfile`, Makefile, workflow,
   artifact upload, Pages, or guard changes.
2. Define path ownership for generated API inputs and outputs, including
   `docs/api/`, `docs/api/html/`, and any staging directory.
3. Specify how generated output must be cleaned, regenerated, retained,
   uploaded, ignored, or rejected by guards.
4. Map each planned implementation change to a regression test or validation
   command.
5. Write the Day 6 design artifact before editing implementation surfaces.

### Deliverables

- Selected policy implementation design.
- Generated output path-ownership map.
- Test and validation mapping.
- Day 6 workflow/tracking design artifact.

### Completion Criteria

- Item 204.2 has a narrow implementation design.
- Generated output tracking semantics are explicit before file changes.
- Workflow or ignore changes have matching guard coverage planned.

---

## Day 7: Policy Implementation Batch

**Title:** Policy Batch
**Theme:** Implement the selected publication or strengthened local-only policy
with minimal surface area.
**Time estimate:** 12 hours

### Tasks

1. Apply the selected `.gitignore`, Makefile, script, workflow, artifact, or
   staging changes.
2. Preserve existing `make docs-check` and `make api-docs-freshness`
   semantics unless the selected policy requires a documented change.
3. Keep generated API input limited to the configured public-header source set
   unless the decision explicitly changes it.
4. Add focused guard coverage for accidental stale, staged, tracked, uploaded,
   or missing generated output behavior.
5. Run focused validation for the modified policy surfaces.

### Deliverables

- Implemented generated API policy changes.
- Focused policy guard updates.
- Initial validation record.
- Updated working-notes implementation log.

### Completion Criteria

- Item 204.2 has landed in code, workflow, docs tooling, or guards as selected.
- Unselected publication paths still fail or remain absent.
- Focused checks exercise the changed policy behavior.

---

## Day 8: Freshness And Coverage Checks

**Title:** Freshness Checks
**Theme:** Strengthen generated API freshness and generated-page coverage for
the selected policy.
**Time estimate:** 12 hours

### Tasks

1. Review `scripts/check_api_docs_coverage.py`,
   `scripts/check_api_docs_local_only.sh`, Make targets, and Doxygen inputs
   against the selected policy.
2. Add or refine checks for missing pages, stale pages, wrong output paths,
   unexpected generated files, and incomplete public-header coverage.
3. Ensure diagnostics identify the failing header, generated page, output
   directory, or policy boundary clearly.
4. Add regression fixtures or current-tree tests for the new diagnostics where
   practical.
5. Run focused freshness and coverage validation.

### Deliverables

- Freshness and generated-page coverage guard changes.
- Diagnostic regression coverage.
- Day 8 freshness-check artifact.
- Updated validation matrix.

### Completion Criteria

- Item 204.3 has concrete freshness or coverage enforcement.
- Missing or stale generated API output fails clearly.
- Checks remain aligned with the chosen policy rather than implying broader
  API completeness.

---

## Day 9: Link And Routing Validation

**Title:** Link Validation
**Theme:** Validate API reference routing, generated-page links, and user entry
points for the selected policy.
**Time estimate:** 12 hours

### Tasks

1. Inventory links from README, INSTALL, `docs/api_reference.md`, maintainer
   guide, and any generated or hosted entry point selected by the policy.
2. Add or document link checks needed to prevent broken API entry points,
   stale generated paths, or unsupported hosted URLs.
3. Verify routing from user-facing docs to source headers, local generated
   HTML, hosted output, retained artifacts, or local-only instructions.
4. Ensure link wording distinguishes source-of-truth headers from generated
   rendered output.
5. Write the Day 9 routing and link-validation artifact.

### Deliverables

- API routing inventory.
- Link-check or manual validation record.
- User-facing entry-point map.
- Day 9 link-validation artifact.

### Completion Criteria

- Item 204.3 has link and routing coverage appropriate to the selected policy.
- User docs do not point to unavailable generated API locations.
- Generated output is described as rendered documentation, not the API source
  of truth.

---

## Day 10: User-Facing API Docs Update

**Title:** User Docs
**Theme:** Update public API routing docs to match the selected generated API
policy.
**Time estimate:** 12 hours

### Tasks

1. Update `docs/api_reference.md` with source-of-truth headers, generated
   output semantics, freshness command, and selected publication policy.
2. Update README API/docs sections with the selected support tier and command
   guidance.
3. Update INSTALL support/readiness wording for generated API HTML and package
   or ABI non-claims.
4. Check that all user-facing references agree on whether generated API HTML
   is hosted, artifact-retained, committed, or local-only.
5. Record changed user-facing surfaces in working notes.

### Deliverables

- Updated `docs/api_reference.md`.
- Updated README and INSTALL generated API wording.
- User-facing claim-boundary record.
- Day 10 user-docs artifact.

### Completion Criteria

- Item 204.4 is implemented for user-facing docs.
- Public docs use one consistent generated API policy vocabulary.
- No user-facing doc implies unsupported ABI, package, hosted, or completeness
  guarantees.

---

## Day 11: Maintainer And Claim Boundary Docs

**Title:** Maintainer Docs
**Theme:** Align maintainer guidance and claim-boundary guards with the
selected generated API policy.
**Time estimate:** 12 hours

### Tasks

1. Update `docs/maintainer_guide.md` with selected generated API ownership,
   regeneration, staging, publication, and validation instructions.
2. Update any claim-boundary guards or docs tests that enforce generated API
   publication wording.
3. Add or revise non-claim markers for ABI, package-manager, hosted
   publication, completeness, release, platform, and state-of-the-art
   boundaries.
4. Verify maintainer wording matches README, INSTALL, and
   `docs/api_reference.md`.
5. Write the Day 11 maintainer and claim-boundary artifact.

### Deliverables

- Updated maintainer generated API guidance.
- Claim-boundary guard updates.
- Non-claim marker inventory.
- Day 11 maintainer-docs artifact.

### Completion Criteria

- Item 204.5 has guard-backed documentation boundaries.
- Maintainer instructions describe exactly how to validate or reject generated
  API output.
- Documentation cannot drift into unsupported publication or completeness
  claims without guard failures.

---

## Day 12: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run the selected generated API validation set and escalate any code
or header changes to the full quality gate.
**Time estimate:** 12 hours

### Tasks

1. Run `make docs-check` and `make api-docs-freshness`.
2. Run any new link, workflow, staging, local-only, hosted publication, or
   artifact-retention checks added by the sprint.
3. Run focused Python or shell guard tests that cover changed validation
   scripts.
4. If any `.c` or `.h` file changed, run `make format`, `make lint`, and
   `make test`.
5. Record command outputs, failures, fixes, and residuals in the validation
   matrix.

### Deliverables

- Integrated validation command log.
- Pass/fail matrix for Sprint 204 gates.
- Residual or blocker ledger.
- Day 12 validation artifact.

### Completion Criteria

- Item 204.6 has current validation evidence.
- Required docs/API checks pass or have explicit blockers.
- Full C quality gates are run if code or public headers changed.

---

## Day 13: Review Hardening

**Title:** Review Hardening
**Theme:** Audit the implementation and documentation for consistency,
minimality, and review readiness.
**Time estimate:** 11 hours

### Tasks

1. Review the git diff for unrelated changes, generated-output accidents,
   broad workflow changes, and inconsistent policy wording.
2. Reconcile working notes, daily artifacts, README, INSTALL,
   `docs/api_reference.md`, maintainer guide, and guards against the Day 5
   decision.
3. Add missing regression coverage for any review-surface or claim-boundary
   gap found during the audit.
4. Re-run focused checks affected by Day 13 hardening.
5. Write the Day 13 review-hardening artifact.

### Deliverables

- Review-hardening notes.
- Final consistency fixes or explicit no-change record.
- Updated validation evidence.
- Day 13 hardening artifact.

### Completion Criteria

- The selected policy is coherent across implementation, docs, guards, and
  planning artifacts.
- Generated API output is not accidentally tracked, staged, uploaded, or
  omitted contrary to the selected policy.
- Remaining residuals are narrow and named.

---

## Day 14: Closeout Package

**Title:** Closeout Package
**Theme:** Package Sprint 204 evidence, final validation, and handoff for the
next Epic 18 sprint.
**Time estimate:** 11 hours

### Tasks

1. Create the Day 14 closeout artifact with item completion status for 204.1
   through 204.6.
2. Summarize the selected generated API policy, implemented guards, docs
   updates, validation commands, and residuals.
3. Update `WORKING_NOTES.md` with final evidence links, changed surfaces,
   validation results, and claim-boundary status.
4. Prepare retrospective inputs: completed work, deferred work, lessons,
   risks, and next-sprint handoff.
5. Confirm final git status excludes unintended generated API output.

### Deliverables

- Day 14 closeout artifact.
- Final Sprint 204 working-notes update.
- Retrospective input package.
- Handoff notes for Sprint 205.

### Completion Criteria

- Sprint 204 has complete evidence for the selected generated API policy.
- All implemented publication or guard behavior is documented and validated.
- The branch is ready for retrospective creation, commit, and PR review.
