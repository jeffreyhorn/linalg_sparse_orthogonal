# Sprint 211 Plan: Large Review-Surface Reduction

**Sprint Duration:** 14 days
**Goal:** Reduce one large high-risk implementation, test, or tooling surface
with behavior-preserving extraction and ownership guards.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 211 estimate in the Epic 19
project plan.

**Primary scope:** Rank large source, test, and tooling surfaces; select one
high-value reduction candidate; document no-behavior-change invariants; design
and implement a focused helper/module extraction; add ownership, registration,
source-list, and regression guards; and close with focused plus required full
validation.

**Non-goals:** New solver behavior, algorithmic changes, numerical tolerance
changes, public API or ABI changes, package-manager support, broad platform
claims, performance claims, release claims, external-library parity, or
state-of-the-art claims. Any extracted surface must preserve observable
behavior unless a deviation is explicitly stopped and reviewed.

---

## Day 1: Surface Intake

**Title:** Surface Intake
**Theme:** Establish the Sprint 211 scope, candidate universe, and review
criteria before touching code.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 211 Epic 19 project-plan section and map items 211.1
   through 211.6 to expected artifacts, code changes, guard updates, and
   validation commands.
2. Review Epic 19 code review findings and prior large-surface reduction
   evidence, especially Sprint 201 selected SVD helper extraction.
3. Inventory large C tests, large C implementation files, large helper
   headers, and Python tooling with current line counts, ownership, churn risk,
   and validation surface.
4. Create `WORKING_NOTES.md` with an item checklist, candidate ledger, risk
   register, validation matrix, decision log, and changed-surface tracker.
5. Record sprint non-goals and behavior-preservation rules before ranking any
   extraction target.

### Deliverables

- Sprint 211 working-notes scaffold.
- Large-surface candidate inventory.
- Initial item-to-evidence traceability map.
- Initial risk register and validation matrix.

### Completion Criteria

- Every Sprint 211 item has a planned evidence path or artifact category.
- Candidate surfaces include source, test, and tooling options.
- Unsupported behavior, API, ABI, platform, performance, release, and
  state-of-the-art claims remain out of scope.

---

## Day 2: Candidate Ranking

**Title:** Candidate Ranking
**Theme:** Rank large review surfaces by risk, extraction value, validation
cost, and ownership clarity.
**Time estimate:** 12 hours

### Tasks

1. Define ranking criteria for line count, review burden, defect risk,
   ownership ambiguity, include dependency complexity, registration fragility,
   and validation availability.
2. Measure candidate surfaces with reproducible commands and record current
   line counts, public/private boundaries, and existing guards.
3. Score each candidate against extraction feasibility, behavior-preservation
   confidence, focused-test availability, and expected review-surface
   reduction.
4. Select one primary cluster and one fallback cluster with explicit rationale.
5. Write the Day 2 candidate-ranking artifact.

### Deliverables

- Ranked large-surface matrix.
- Selected primary and fallback cluster decision.
- Measurement command record.
- Day 2 ranking artifact.

### Completion Criteria

- Item 211.1 has evidence-backed candidate ranking.
- The selected primary cluster has a clear behavior-preserving extraction path.
- Fallback conditions are documented before design begins.

---

## Day 3: Cluster Boundary

**Title:** Cluster Boundary
**Theme:** Define the selected extraction boundary, ownership model, and
no-behavior-change invariants.
**Time estimate:** 12 hours

### Tasks

1. Trace selected-cluster entry points, helper dependencies, static functions,
   data fixtures, registration calls, and build/source-list ownership.
2. Identify observable behavior that must remain unchanged, including status
   codes, output values, stdout/stderr text, registration order, skip behavior,
   and generated artifacts where applicable.
3. Define the target owner file or helper/module boundary and the files that
   must not take ownership.
4. Record non-goals for unrelated refactors, algorithm changes, fixture
   rewrites, and claim expansion.
5. Write the Day 3 cluster-boundary artifact.

### Deliverables

- Selected-cluster boundary statement.
- No-behavior-change invariant list.
- Ownership and non-owner file list.
- Day 3 boundary artifact.

### Completion Criteria

- Item 211.2 has a concrete selected-cluster boundary.
- Every planned move has an owning destination and explicit non-owner scope.
- Observable behavior and validation expectations are documented before
  implementation.

---

## Day 4: Extraction Design

**Title:** Extraction Design
**Theme:** Design the helper/module split and dependency strategy before file
edits.
**Time estimate:** 12 hours

### Tasks

1. Design the extracted file shape, include guard or module-local visibility,
   static inline versus compiled-object strategy, and naming convention.
2. Map all dependencies that must move with the selected cluster and all
   dependencies that must remain in the original owner.
3. Plan Makefile, CMake, source-list, or script updates needed by the
   extraction.
4. Define guard checks for ownership, registration placement, order, source
   lists, include direction, and duplicate definitions.
5. Write the Day 4 extraction-design artifact.

### Deliverables

- Implementation-ready extraction design.
- Dependency and include-direction map.
- Build/source-list update plan.
- Guard strategy.

### Completion Criteria

- Item 211.3 has a concrete design ready for code edits.
- The design preserves the selected behavior boundary.
- Build and guard impacts are known before implementation starts.

---

## Day 5: Baseline Validation

**Title:** Baseline Validation
**Theme:** Capture pre-extraction behavior and focused validation output before
moving code.
**Time estimate:** 12 hours

### Tasks

1. Run focused tests, scripts, or inspection commands for the selected cluster
   before extraction.
2. Capture registration counts, order checks, output summaries, generated-file
   checks, or source-list counts relevant to the selected surface.
3. Record current line counts and review-surface measurements for the source
   and target files.
4. Identify any flaky, slow, platform-specific, or unavailable validation
   commands and document fallback checks.
5. Write the Day 5 baseline-validation artifact.

### Deliverables

- Pre-extraction validation record.
- Baseline line-count and review-surface metrics.
- Focused command output summary.
- Day 5 baseline artifact.

### Completion Criteria

- Baseline behavior is recorded before extraction edits.
- Any unavailable validation is explicitly documented with risk and fallback.
- The selected reduction has measurable before-state review-surface data.

---

## Day 6: Extraction Batch One

**Title:** Extraction Batch One
**Theme:** Move the first coherent portion of the selected cluster into its
new owner without behavior changes.
**Time estimate:** 12 hours

### Tasks

1. Create or update the selected helper/module owner file according to the Day
   4 design.
2. Move the first coherent helper, fixture, or implementation cluster while
   preserving names and behavior unless the design requires scoped renaming.
3. Update includes, declarations, source lists, or registration references
   needed for the first batch.
4. Run the smallest focused build or syntax check that proves the first batch
   compiles or parses.
5. Record changed files, moved symbols, and deviations in `WORKING_NOTES.md`.

### Deliverables

- First extraction batch.
- Updated includes/declarations/source references.
- Focused compile or syntax-check evidence.
- Working-notes implementation record.

### Completion Criteria

- Item 211.4 has begun with behavior-preserving edits.
- The selected surface still builds or parses under focused validation.
- No unrelated refactor is introduced.

---

## Day 7: Extraction Batch Two

**Title:** Extraction Batch Two
**Theme:** Move the remaining selected-cluster code and reconcile ownership
boundaries.
**Time estimate:** 12 hours

### Tasks

1. Move the remaining selected helpers, fixtures, local utilities, or module
   logic in the chosen cluster.
2. Remove duplicate declarations or obsolete local definitions from the
   original owner.
3. Verify include direction and ownership boundaries remain acyclic and
   localized.
4. Refresh review-surface line counts for the original and new owner files.
5. Write the Day 7 extraction-progress artifact.

### Deliverables

- Completed selected-cluster extraction.
- Duplicate-definition cleanup.
- Updated review-surface metrics.
- Day 7 extraction artifact.

### Completion Criteria

- The selected cluster lives in the planned owner file or module.
- The original large surface is measurably reduced.
- Behavior remains intended to be unchanged and ready for guard work.

---

## Day 8: Build And Source Wiring

**Title:** Build Wiring
**Theme:** Reconcile Makefile, CMake, source lists, and dependency metadata
after extraction.
**Time estimate:** 12 hours

### Tasks

1. Update Makefile prerequisites, build recipes, or generated source lists
   required by the extracted owner.
2. Update CMake source lists, test registration, install exclusions, or
   helper-header dependencies as applicable.
3. Verify source-list parity between build systems when the extraction touches
   compiled code or registered tests.
4. Add focused source-list or prerequisite checks if the new owner could be
   omitted silently.
5. Write the Day 8 build-wiring artifact.

### Deliverables

- Updated build/source wiring.
- Source-list or prerequisite parity evidence.
- Focused build metadata guard plan or implementation.
- Day 8 wiring artifact.

### Completion Criteria

- The extracted owner is reachable through the intended build/test paths.
- No build system silently omits the new owner.
- Source-list parity risks are documented and guarded where needed.

---

## Day 9: Ownership Guard

**Title:** Ownership Guard
**Theme:** Add guard coverage so extracted code cannot drift back into the
large owner or wrong helper.
**Time estimate:** 12 hours

### Tasks

1. Implement or update guard scripts/tests that assert extracted symbol,
   helper, registration, or include ownership.
2. Add negative fixtures or mutation-style checks for likely regressions:
   duplicate ownership, wrong file placement, omitted prerequisites, or
   commented-out registrations.
3. Ensure guards ignore comments and inert examples where relevant.
4. Wire the guard into an existing focused validation target or documented
   validation sequence.
5. Write the Day 9 ownership-guard artifact.

### Deliverables

- Ownership guard implementation.
- Regression fixtures for guard failure modes.
- Focused guard invocation record.
- Day 9 guard artifact.

### Completion Criteria

- Item 211.5 has active guard coverage for the extracted ownership boundary.
- Guard checks fail clearly for wrong ownership or missing registration.
- Guard behavior is documented with a runnable command.

---

## Day 10: Behavior Regression Coverage

**Title:** Behavior Coverage
**Theme:** Prove the extraction did not change selected behavior or
registration semantics.
**Time estimate:** 12 hours

### Tasks

1. Run focused tests for the selected extracted surface and compare against the
   Day 5 baseline.
2. Add or update regression tests if the extraction exposed missing coverage
   for registration order, helper placement, or output behavior.
3. Verify skip behavior, environment-variable behavior, generated-output
   behavior, or CLI diagnostics as applicable to the selected cluster.
4. Record any changed assertion counts or command outputs and explain whether
   they are expected.
5. Write the Day 10 behavior-regression artifact.

### Deliverables

- Focused behavior-regression evidence.
- Updated tests if coverage gaps were found.
- Baseline comparison record.
- Day 10 regression artifact.

### Completion Criteria

- Behavior-preservation evidence exists for the extracted surface.
- Any assertion-count or output change is explained and justified.
- Item 211.5 has focused regression coverage beyond ownership-only checks.

---

## Day 11: Documentation Calibration

**Title:** Documentation Calibration
**Theme:** Update maintainer and planning documentation to describe the new
ownership model without overstating behavior changes.
**Time estimate:** 12 hours

### Tasks

1. Update maintainer notes, README/INSTALL references, or planning docs that
   describe the selected large surface or helper ownership.
2. Replace stale path, line-count, or owner references with the new extracted
   ownership model.
3. Add explicit behavior-preservation and non-goal wording where the extraction
   could be misread as a feature or algorithm change.
4. Update `WORKING_NOTES.md` with changed-surface inventory and validation
   status.
5. Write the Day 11 documentation-calibration artifact.

### Deliverables

- Updated ownership documentation.
- Stale-reference cleanup.
- Changed-surface inventory.
- Day 11 documentation artifact.

### Completion Criteria

- Documentation names the correct owner files after extraction.
- No documentation claims new solver behavior, API, ABI, performance, platform,
  release, or state-of-the-art support.
- Planning evidence matches the actual changed surface.

---

## Day 12: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run the focused and required integrated validation suite for the
extracted surface.
**Time estimate:** 12 hours

### Tasks

1. Run all focused tests, ownership guards, source-list checks, and docs guards
   relevant to the extraction.
2. If any source or header file changed, run the required `make format`,
   `make lint`, and `make test` quality chain.
3. Run Make/CMake parity or CTest inspection checks when test registration or
   compiled sources changed.
4. Record command results, changed files, and residual risks in
   `WORKING_NOTES.md`.
5. Write the Day 12 integrated-validation artifact.

### Deliverables

- Integrated validation command log.
- Required full quality-chain result when applicable.
- Make/CMake/source-list parity evidence when applicable.
- Day 12 validation artifact.

### Completion Criteria

- Item 211.6 has current validation evidence.
- Required quality checks pass before closeout proceeds.
- Any unclear or failing validation stops the sprint for review.

---

## Day 13: Review Hardening

**Title:** Review Hardening
**Theme:** Inspect the diff as a reviewer and close likely guard, wording, and
inventory gaps.
**Time estimate:** 11 hours

### Tasks

1. Review the full diff for unrelated edits, behavior changes, stale comments,
   missing source-list entries, and incomplete guard coverage.
2. Re-run focused checks for any guard or documentation change made during
   hardening.
3. Verify all changed files are represented in Sprint 211 evidence inventories.
4. Update artifacts and working notes for any corrected assumptions or
   residual limitations.
5. Write the Day 13 review-hardening artifact.

### Deliverables

- Review-hardening findings and fixes.
- Updated changed-file inventories.
- Re-run focused validation evidence.
- Day 13 hardening artifact.

### Completion Criteria

- The diff is scoped to the selected extraction and its evidence.
- Inventories match the actual changed surface.
- Remaining risks are documented and not hidden as completed work.

---

## Day 14: Closeout Review

**Title:** Closeout Review
**Theme:** Finalize Sprint 211 evidence, status, and handoff for PR review.
**Time estimate:** 11 hours

### Tasks

1. Reconcile Sprint 211 items 211.1 through 211.6 against implemented
   artifacts, code changes, guards, and validation results.
2. Record final before/after review-surface metrics and ownership boundaries.
3. Update project-plan or working-notes status if the selected reduction was
   narrowed, deferred, or completed differently than planned.
4. Prepare closeout notes with validation commands, non-goals, residual risks,
   and expected PR review focus.
5. Write the Day 14 closeout-review artifact.

### Deliverables

- Sprint 211 closeout review artifact.
- Final item-status checklist.
- Before/after review-surface reduction summary.
- PR-ready validation and residual-risk summary.

### Completion Criteria

- Item 211.6 has final closeout evidence.
- The selected large review surface has a documented before/after reduction or
  a clearly recorded stop condition.
- The branch is ready for retrospective creation, final commit, push, and PR
  creation in a later step.
