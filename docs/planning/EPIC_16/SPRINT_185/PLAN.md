# Sprint 185 Plan: Large Test and Solver Review-Surface Reduction

**Sprint Duration:** 14 days
**Goal:** Reduce one large review surface by extracting helpers or proof-owner
files while preserving behavior and build registration.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 185 estimate in the Epic 16
project plan.

**Primary scope:** Select one large test or solver cluster with high review
cost and low refactor risk, design behavior-preserving helper boundaries,
extract the selected helpers or proof-owner files, update registration and
guards, document maintenance invariants, and run the required validation.

**Non-goals:** Rewriting solver behavior, broadening the sprint to multiple
clusters, changing public API contracts, changing numerical tolerances, adding
new solver features, or claiming performance improvements from refactoring.

---

## Day 1: Sprint Intake and Prior Evidence Review

**Title:** Review-Surface Intake
**Theme:** Establish Sprint 185 scope and inherited guardrails.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 185 section of the Epic 16 project plan and capture the
   acceptance boundaries for items 185.1 through 185.6.
2. Review Sprint 177 file-size inventory and any later sprint artifacts that
   mention large test/source review surfaces.
3. Inventory existing build source-list drift checks, Make/CMake registration
   conventions, and test registration patterns.
4. Identify candidate large test or solver clusters without selecting one yet.
5. Start `WORKING_NOTES.md` with sprint scope, candidate clusters, inherited
   guardrails, risks, and open questions.

### Deliverables

- Sprint intake notes tied to items 185.1 through 185.6.
- Initial candidate cluster inventory.
- Guardrail and risk list for behavior-preserving extraction.

### Completion Criteria

- Sprint scope is traceable to the project-plan items.
- Candidate clusters are known enough to begin review-cost comparison.
- Existing build and registration guard surfaces are identified.

---

## Day 2: Candidate Cluster Inventory

**Title:** Large Surface Baseline
**Theme:** Measure candidate review surfaces before selection.
**Time estimate:** 12 hours

### Tasks

1. Build a current file-size and responsibility inventory for candidate test
   and solver clusters.
2. Identify clusters with high line count, mixed responsibilities, repeated
   fixtures, large helper blocks, or unclear proof ownership.
3. Record each candidate's build registration, test binary ownership, helper
   dependencies, and validation path.
4. Estimate refactor risk for each candidate, including numerical behavior,
   fixture ownership, registration churn, and review complexity.
5. Narrow the candidate set to the strongest one or two clusters for Day 3.

### Deliverables

- Candidate cluster size and responsibility table.
- Registration and validation map for each candidate.
- Shortlist for final cluster selection.

### Completion Criteria

- Candidate review cost is based on concrete files and responsibilities.
- Refactor risk is explicit for each shortlisted cluster.
- Day 3 can select one cluster without further broad discovery.

---

## Day 3: Cluster Selection Decision

**Title:** Selected Cluster Decision
**Theme:** Select exactly one large review surface for Sprint 185.
**Time estimate:** 12 hours

### Tasks

1. Select one test or solver cluster using review cost, extraction feasibility,
   behavior-preservation confidence, and registration risk.
2. Document rejected alternatives and why they are deferred.
3. Capture the selected cluster's baseline files, test commands, build
   metadata, and proof-owner responsibilities.
4. Define the no-behavior-change contract for the selected cluster.
5. Draft the extraction checklist for Days 4 through 8.

### Deliverables

- Selected cluster decision record.
- Baseline file and registration inventory.
- No-behavior-change contract and extraction checklist.

### Completion Criteria

- Exactly one cluster is selected for Sprint 185.
- Deferred clusters are separated from active sprint scope.
- The selected cluster has clear baseline validation commands.

---

## Day 4: Extraction Boundary Design

**Title:** Helper Boundary Design
**Theme:** Define the extracted files before moving code.
**Time estimate:** 12 hours

### Tasks

1. Identify helper functions, fixtures, proof-owner routines, or repeated setup
   code that can move without changing behavior.
2. Decide whether extraction should target test helpers, solver-internal
   helpers, proof-owner files, or fixture headers.
3. Define new file names, internal declarations, include relationships, and
   ownership boundaries.
4. Record what must remain in the original file for readability and locality.
5. Design focused checks that can prove extraction preserved behavior.

### Deliverables

- Extraction boundary design for the selected cluster.
- Proposed file list and ownership model.
- Focused validation plan for the first extraction pass.

### Completion Criteria

- Helper boundaries are small enough for review.
- The design avoids circular includes and ambiguous ownership.
- Focused validation is ready before mechanical moves begin.

---

## Day 5: Build Registration and Guard Design

**Title:** Registration Guardrail Design
**Theme:** Plan source-list and test registration updates before extraction.
**Time estimate:** 12 hours

### Tasks

1. Audit Makefile, CMake, script, and test-runner registration for the selected
   cluster.
2. Identify source-list drift checks or registration guards that already cover
   the planned extraction.
3. Design any needed cluster-specific guard or extension to existing metadata
   checks.
4. Define expected build artifacts, test binaries, and generated files after
   extraction.
5. Record rollback criteria if registration changes expose unexpected risk.

### Deliverables

- Build and test registration design.
- Guard update plan or documented decision that existing guards are sufficient.
- Expected post-extraction source-list state.

### Completion Criteria

- Registration changes are known before code movement.
- Guard coverage can detect missing helper/proof-owner files.
- The build metadata plan is reviewable and bounded.

---

## Day 6: First Mechanical Extraction Pass

**Title:** Initial Helper Extraction
**Theme:** Move the lowest-risk helper surface first.
**Time estimate:** 12 hours

### Tasks

1. Extract the first approved helper, fixture, or proof-owner block from the
   selected cluster.
2. Add any required internal header declarations or test helper includes.
3. Update build registration for the new file if needed.
4. Preserve formatting, naming, static visibility, and existing ownership
   semantics.
5. Run the focused compile/test checks identified on Days 4 and 5.

### Deliverables

- First extracted helper or proof-owner file.
- Updated registration metadata if required.
- Focused validation results.

### Completion Criteria

- The selected extraction compiles and runs through focused checks.
- The original cluster is smaller without behavior changes.
- No unrelated files or clusters are modified.

---

## Day 7: Second Mechanical Extraction Pass

**Title:** Fixture and Setup Extraction
**Theme:** Reduce repeated setup or fixture ownership inside the selected
cluster.
**Time estimate:** 12 hours

### Tasks

1. Extract the next approved helper, fixture, or setup block from the selected
   cluster.
2. Keep fixture ownership explicit and avoid introducing shared mutable state.
3. Update includes, declarations, and registration metadata as needed.
4. Re-run focused validation for the selected cluster.
5. Record any extraction candidates that should be deferred.

### Deliverables

- Second extracted helper, fixture, or setup file.
- Updated ownership notes for moved fixtures.
- Focused validation evidence and deferred-candidate notes.

### Completion Criteria

- Review surface is smaller and responsibility boundaries are clearer.
- Extracted fixtures have explicit ownership and no behavior drift.
- Focused checks pass after registration updates.

---

## Day 8: Proof-Owner and Call-Site Cleanup

**Title:** Proof-Owner Cleanup
**Theme:** Make the selected cluster easier to review after extraction.
**Time estimate:** 12 hours

### Tasks

1. Review remaining large functions or proof-owner blocks in the selected
   cluster.
2. Move only the blocks approved by the Day 4 boundary design and Day 7
   findings.
3. Clean up local includes, duplicated declarations, stale comments, and
   helper ordering.
4. Preserve test names, solver behavior, fixture data, and validation
   tolerances.
5. Run focused validation and a source-list drift check.

### Deliverables

- Final mechanical extraction for the selected cluster.
- Cleaned call sites and helper ownership notes.
- Focused validation and source-list check results.

### Completion Criteria

- The selected cluster has a clearly reduced review surface.
- No validation tolerance, fixture, or solver behavior changed.
- Build registration remains synchronized.

---

## Day 9: Registration Guard Implementation

**Title:** Drift Guard Update
**Theme:** Add or extend protection for extracted files and registration.
**Time estimate:** 12 hours

### Tasks

1. Implement the planned Make/CMake metadata check or cluster-specific
   registration guard if existing checks are insufficient.
2. Ensure the guard catches missing extracted files, missing source-list
   entries, or broad registration drift relevant to the selected cluster.
3. Keep guard output actionable for future reviewers.
4. Run the guard in isolation and with existing registration checks.
5. Update `WORKING_NOTES.md` with guard coverage and limitations.

### Deliverables

- New or updated registration guard.
- Guard output captured in working notes.
- Guard limitation notes for future maintainers.

### Completion Criteria

- Registration drift for the selected extraction is mechanically protected.
- Guard failures are understandable and actionable.
- Guard checks pass in focused execution.

---

## Day 10: Maintenance Note Draft

**Title:** Maintenance Invariants
**Theme:** Document how to extend the selected cluster after extraction.
**Time estimate:** 12 hours

### Tasks

1. Draft a maintenance note for the selected cluster's new file boundaries,
   helper ownership, registration requirements, and validation commands.
2. Document invariants that future contributions must preserve.
3. Document how to add new fixtures, helpers, or proof-owner cases without
   re-growing the original review surface.
4. Link the note to relevant existing maintainer or test documentation.
5. Review the note against the actual extracted file layout.

### Deliverables

- Maintenance note draft for the selected cluster.
- Contribution guidance for future helper or fixture additions.
- Validation command list for future maintainers.

### Completion Criteria

- Future contributors can understand where new code belongs.
- Registration and validation requirements are explicit.
- The note matches the current file layout.

---

## Day 11: Documentation and Contributor Alignment

**Title:** Contributor Guidance Alignment
**Theme:** Integrate maintenance guidance with existing docs.
**Time estimate:** 12 hours

### Tasks

1. Add or update the selected maintenance note in the appropriate docs,
   planning, or test directory.
2. Cross-link from existing maintainer or testing documentation if useful.
3. Remove stale comments that conflict with the new file layout.
4. Confirm docs do not imply behavior, performance, or coverage changes from
   the refactor.
5. Run docs-focused checks and registration guards.

### Deliverables

- Final maintenance note and any cross-links.
- Stale-comment cleanup, if needed.
- Docs and guard validation results.

### Completion Criteria

- Maintenance guidance is discoverable.
- Documentation reflects the extracted review surface.
- No unsupported behavior or performance claims are introduced.

---

## Day 12: Focused Validation Pass

**Title:** Focused Cluster Validation
**Theme:** Exercise the selected cluster before the full quality gate.
**Time estimate:** 12 hours

### Tasks

1. Run all focused tests for the selected cluster.
2. Run source-list drift checks, registration guards, and docs checks relevant
   to the extraction.
3. Review the diff for accidental solver behavior changes, fixture changes,
   tolerance changes, or scope expansion.
4. Fix focused validation failures and rerun the failing checks.
5. Prepare the full validation command list for Day 13.

### Deliverables

- Focused validation results.
- Fixed registration, docs, or extraction issues found by focused checks.
- Full validation command list.

### Completion Criteria

- Focused checks pass before full validation begins.
- The diff remains limited to the selected cluster and required docs/guards.
- No unresolved behavior-preservation risks remain for full validation.

---

## Day 13: Full Quality Gate

**Title:** Full Extraction Validation
**Theme:** Run repository-level checks after source/test extraction.
**Time estimate:** 12 hours

### Tasks

1. Run `make format`.
2. Run `make lint`.
3. Run `make test`.
4. Run affected cluster tests, source-list checks, registration guards, docs
   checks, and `git diff --check`.
5. Fix any failures and rerun the failing command until all required checks
   pass.

### Deliverables

- Passing full quality-check results.
- Final validation notes for the selected extraction.
- Any final cleanup needed for review readiness.

### Completion Criteria

- `make format`, `make lint`, and `make test` pass.
- Source-list and registration guards pass.
- No unresolved validation failures remain.

---

## Day 14: Sprint Closeout and Handoff

**Title:** Review-Ready Handoff
**Theme:** Package the reduced review surface for retrospective and PR review.
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 185 diffs against project-plan items 185.1 through 185.6.
2. Confirm the selected cluster, extracted files, registration updates, guard
   coverage, and maintenance note are documented in `WORKING_NOTES.md`.
3. Prepare summary notes for the Sprint 185 retrospective, including decisions,
   validation, risks, and follow-up candidates.
4. Check for stale TODOs, unresolved open questions, generated artifacts, or
   accidental scope expansion.
5. Produce final review-ready notes describing the extraction and how to verify
   it.

### Deliverables

- Retrospective-ready Sprint 185 working notes.
- Review-ready change summary and validation summary.
- Follow-up list for deferred clusters or future extraction opportunities.

### Completion Criteria

- All Sprint 185 project-plan items have a documented outcome.
- The final state is ready for retrospective creation and PR preparation.
- Deferred work is explicitly separated from completed sprint scope.
