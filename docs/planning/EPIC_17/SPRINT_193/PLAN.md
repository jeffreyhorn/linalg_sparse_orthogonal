# Sprint 193 Plan: Selected Large Review-Surface Reduction

**Sprint Duration:** 14 days
**Goal:** Reduce one high-risk implementation/test review surface while
preserving behavior and validation.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 193 estimate in the Epic 17
project plan.

**Primary scope:** Rank large source and test review surfaces, select exactly
one cluster, define no-behavior-change invariants, extract the selected cluster
behind focused helper/source boundaries, add ownership guards and maintainer
documentation, and complete full behavior-preserving validation.

**Non-goals:** Broad refactoring across unrelated files, algorithm changes,
public API changes, numerical tolerance changes, solver behavior changes,
test-case deletion, unreviewed generated source-list drift, performance claims,
or partial reductions across multiple clusters.

---

## Day 1: Sprint Intake and Review-Surface Baseline

**Title:** Review-Surface Intake
**Theme:** Establish Sprint 193 scope, candidate source/test surfaces, and the
no-behavior-change acceptance boundary.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 193 section of the Epic 17 project plan and map items
   193.1 through 193.6 to likely owner files, tests, scripts, source lists, and
   documentation.
2. Review Sprint 187 review-surface ranking artifacts and Sprint 185 helper
   extraction patterns for reusable selection and guard criteria.
3. Inventory large implementation and test files by line count, helper density,
   algorithmic risk, fixture ownership, source-list membership, and existing
   validation coverage.
4. Identify behavior-preservation constraints for candidate clusters, including
   public API stability, numerical tolerance stability, cleanup ownership, and
   process-global state restoration.
5. Create `WORKING_NOTES.md` with baseline findings, candidate list, risk
   register, source-list inventory, and Day 2 ranking questions.

### Deliverables

- Sprint 193 working-notes scaffold.
- Large source/test review-surface inventory.
- Candidate cluster list with initial risk tags.
- Initial no-behavior-change acceptance boundary.

### Completion Criteria

- Sprint scope is traceable to items 193.1 through 193.6.
- Candidate review surfaces are identified from current repository evidence,
  not memory or assumptions.
- No implementation edits begin before candidate ranking criteria are recorded.

---

## Day 2: Candidate Ranking

**Title:** Candidate Ranking
**Theme:** Rank large source/test clusters by size, risk, ownership clarity,
test coverage, and review-value payoff.
**Time estimate:** 12 hours

### Tasks

1. Score each candidate by line count, cyclomatic/branch density where easily
   measurable, helper duplication, cleanup complexity, process-global state
   risk, source-list coupling, and public behavior exposure.
2. Review current tests for each candidate cluster and identify which
   assertions would detect behavior drift after extraction.
3. Separate high-value behavior-preserving extractions from risky algorithmic
   rewrites or broad style cleanups.
4. Compare candidate clusters against Sprint 185 extraction and guard patterns
   to estimate implementation and review cost.
5. Produce a ranked candidate table with selected, alternate, rejected, and
   deferred clusters.

### Deliverables

- Candidate ranking artifact.
- Ranked cluster table with evidence-backed scores.
- Initial selected-cluster recommendation.
- Rejection rationale for broad or risky alternatives.

### Completion Criteria

- Exactly one cluster is recommended for Sprint 193 implementation.
- Ranking accounts for size, algorithm risk, helper ownership, current tests,
  and user-facing importance.
- Rejected candidates have concrete reasons that constrain later scope creep.

---

## Day 3: Cluster Selection and Invariant Contract

**Title:** Cluster Selection
**Theme:** Select the one review-surface cluster and define behavior invariants
before extraction design.
**Time estimate:** 12 hours

### Tasks

1. Confirm the selected cluster, owner files, source-list entries, tests, and
   documentation surfaces.
2. Define no-behavior-change invariants for inputs, outputs, status codes,
   memory ownership, cleanup paths, global override restoration, diagnostics,
   row ordering, and tolerance behavior.
3. Identify public, internal, and test-only boundaries for the selected cluster.
4. Record what must remain in the original file versus what can move into a
   helper source/header or test helper.
5. Create the Day 3 cluster-selection artifact with acceptance criteria and
   implementation constraints.

### Deliverables

- Selected cluster decision.
- No-behavior-change invariant contract.
- Owner file and source-list map.
- Extraction boundary constraints.

### Completion Criteria

- Only one cluster is selected for Sprint 193.
- Invariants are specific enough to drive tests and review.
- The selected cluster has a plausible extraction path without public API
  changes.

---

## Day 4: Extraction Boundary Design

**Title:** Boundary Design
**Theme:** Design helper/source boundaries, ownership rules, cleanup behavior,
and source-list updates.
**Time estimate:** 12 hours

### Tasks

1. Design the extracted helper API, file names, internal visibility, include
   relationships, and ownership comments.
2. Define cleanup ownership and error-path behavior for every moved helper or
   extracted block.
3. Identify any process-global override, registration, fixture, or temporary
   resource state that must be restored before early-return paths.
4. Map build-system source-list changes for Make, CMake, tests, and any
   generated or documented source inventory.
5. Record the extraction plan and review checkpoints before editing code.

### Deliverables

- Extraction design artifact.
- Proposed helper/source boundary map.
- Cleanup and global-state restoration checklist.
- Source-list update checklist.

### Completion Criteria

- The extraction plan preserves existing behavior by construction.
- Every new helper has a clear owner and caller contract.
- Build-system updates are identified before implementation begins.

---

## Day 5: Mechanical Extraction Scaffold

**Title:** Extraction Scaffold
**Theme:** Add the new helper/source scaffolding and wire it into local builds
without changing behavior.
**Time estimate:** 12 hours

### Tasks

1. Add the selected helper source/header or test-helper file using the approved
   boundary and local naming conventions.
2. Move declarations, includes, and static helper prototypes mechanically while
   preserving call sites and behavior.
3. Update Make/CMake/test source lists if the extracted files need build-system
   ownership.
4. Run the smallest build or compile check that proves the scaffold is wired
   correctly.
5. Update working notes with changed files, source-list edits, and unresolved
   compile or ownership questions.

### Deliverables

- New extraction scaffold.
- Updated source lists if required.
- Initial compile/build validation result.
- Day 5 implementation artifact.

### Completion Criteria

- The repository builds far enough to prove the new file boundary is visible.
- No behavior logic is intentionally changed.
- Source-list drift is documented immediately.

---

## Day 6: Helper Movement and Call-Site Preservation

**Title:** Helper Movement
**Theme:** Move the selected helper logic into the new boundary while keeping
call-site semantics stable.
**Time estimate:** 12 hours

### Tasks

1. Move selected helper implementations from the large source/test file into
   the new helper boundary in small, reviewable chunks.
2. Preserve function signatures or add minimal internal adapters where direct
   preservation is not possible.
3. Keep diagnostics, status propagation, allocation ordering, cleanup ordering,
   and tolerance handling identical unless the invariant contract permits a
   documented local improvement.
4. Run focused tests after each meaningful movement step.
5. Record before/after line-count and ownership changes in working notes.

### Deliverables

- First behavior-preserving helper movement.
- Focused validation output.
- Before/after review-surface metrics.
- Updated extraction risk notes.

### Completion Criteria

- Moved helpers retain the same observable behavior.
- Focused tests pass after movement.
- The original large file is measurably smaller or structurally simpler.

---

## Day 7: Cleanup and Error-Path Ownership

**Title:** Cleanup Ownership
**Theme:** Harden cleanup paths, early returns, resource ownership, and global
state restoration in the selected cluster.
**Time estimate:** 12 hours

### Tasks

1. Audit moved and remaining selected-cluster paths for early returns,
   allocation failure, cleanup ordering, double-free risk, and leak risk.
2. Convert fragile cleanup patterns to the smallest local ownership model that
   matches repository style.
3. Ensure process-global overrides, registration state, temporary files, and
   fixture state are restored before any assertion or early-return macro can
   exit.
4. Add or update focused regression tests for cleanup and restoration behavior
   where practical.
5. Document cleanup invariants and any intentionally unchanged risks.

### Deliverables

- Cleanup ownership artifact.
- Hardened early-return and restoration paths.
- Focused cleanup/restoration regression tests if needed.
- Updated no-behavior-change notes.

### Completion Criteria

- Selected-cluster cleanup behavior is explicit and reviewable.
- Failure paths cannot contaminate subsequent tests or calls through stale
  process-global state.
- Focused cleanup validation passes.

---

## Day 8: Registration and Source-List Guards

**Title:** Source-List Guards
**Theme:** Add ownership checks that keep the extracted boundary registered in
all required build and validation surfaces.
**Time estimate:** 12 hours

### Tasks

1. Add a cluster-specific guard that checks extracted files are present in the
   required source lists, test lists, or documentation inventories.
2. Cover missing-source, wrong-path, duplicate-entry, and stale-owner failure
   modes where they apply.
3. Integrate the guard into the smallest appropriate Make target or existing
   validation script.
4. Add tests for the guard's positive and negative cases.
5. Record guard ownership and expected remediation text.

### Deliverables

- Cluster-specific ownership guard.
- Guard regression tests.
- Make/script integration if appropriate.
- Day 8 guard artifact.

### Completion Criteria

- Source-list drift produces a clear failure before code review misses it.
- Guard scope is specific to the selected cluster and does not create broad
  repository policy accidentally.
- Guard tests pass.

---

## Day 9: Focused Behavior Regression Coverage

**Title:** Behavior Coverage
**Theme:** Strengthen tests around the selected cluster's public behavior and
edge cases without expanding feature scope.
**Time estimate:** 12 hours

### Tasks

1. Identify the smallest set of tests that prove extracted behavior still
   matches the pre-extraction contract.
2. Add or adjust tests for success paths, failure paths, cleanup paths,
   boundary inputs, numerical tolerances, and diagnostic/status preservation.
3. Avoid deleting tests or weakening assertions unless duplicate coverage is
   explicitly proven and documented.
4. Run focused test binaries or Python guards that own the selected cluster.
5. Record behavior coverage before and after the changes.

### Deliverables

- Focused behavior regression updates.
- Test coverage artifact for the selected cluster.
- Focused validation results.
- Assertion-preservation notes.

### Completion Criteria

- The selected cluster has tests covering extraction-sensitive behavior.
- No behavior-preserving claim relies only on compilation.
- Any test movement is justified by clearer ownership, not reduced coverage.

---

## Day 10: Maintainer Documentation

**Title:** Boundary Documentation
**Theme:** Document the new helper/source boundary and review expectations for
future maintainers.
**Time estimate:** 12 hours

### Tasks

1. Update maintainer documentation with the selected cluster boundary, owner
   files, guard command, and source-list expectations.
2. Document no-behavior-change invariants that future edits must preserve.
3. Explain cleanup ownership and global-state restoration expectations if they
   apply to the selected cluster.
4. Add documentation guard coverage if the repository uses executable docs
   markers for similar ownership boundaries.
5. Record documentation changes and retained non-goals in the Day 10 artifact.

### Deliverables

- Maintainer documentation update.
- Boundary and guard usage notes.
- Optional docs guard coverage.
- Day 10 documentation artifact.

### Completion Criteria

- Future maintainers can identify the extracted boundary and validation owner.
- Documentation does not imply algorithmic, API, or performance changes.
- Guard commands and source-list expectations are discoverable.

---

## Day 11: Integrated Build and Source-List Validation

**Title:** Build Integration
**Theme:** Run source-list, build-system, and focused integration checks after
the extraction is complete.
**Time estimate:** 12 hours

### Tasks

1. Run source-list guards, cluster-specific ownership guards, and any generated
   source inventory checks.
2. Run the focused test targets for the selected cluster and adjacent helper
   boundaries.
3. Run Make and CMake build checks required by any source-list changes.
4. Investigate and fix build, registration, include-order, or linkage failures
   without widening the selected cluster.
5. Record all integrated validation commands and results in working notes.

### Deliverables

- Integrated source-list validation record.
- Focused build/test result set.
- Fixed build-system drift, if any.
- Day 11 validation artifact.

### Completion Criteria

- All selected-cluster source-list and focused build checks pass.
- Any source-list changes are validated through both direct and guard coverage.
- No unrelated refactors are introduced to make validation pass.

---

## Day 12: Full C Quality Gate

**Title:** Full Quality Gate
**Theme:** Run formatting, linting, full tests, and CMake parity required for a
behavior-preserving C/source extraction.
**Time estimate:** 12 hours

### Tasks

1. Run `make format` and inspect the resulting diff for unintended churn.
2. Run `make lint` and resolve warnings or guard failures in the selected
   cluster without weakening diagnostics.
3. Run `make test` and investigate any failures against the invariant contract.
4. Run CMake parity or source-list validation if source lists changed.
5. Record complete validation output, changed files, and any residual risk.

### Deliverables

- Full C quality-gate validation record.
- Formatting/lint/test fixes if required.
- CMake/source-list parity result if applicable.
- Day 12 validation artifact.

### Completion Criteria

- `make format`, `make lint`, and `make test` pass.
- CMake/source-list parity passes if required by changed files.
- Remaining diffs are intentional and scoped to Sprint 193.

---

## Day 13: Review-Surface Audit

**Title:** Review-Surface Audit
**Theme:** Confirm the selected review surface is smaller, clearer, and still
behavior-preserving before closeout.
**Time estimate:** 12 hours

### Tasks

1. Measure before/after line count, helper count, source-list ownership, test
   coverage, and guard ownership for the selected cluster.
2. Review the final diff for accidental API changes, tolerance changes,
   diagnostics changes, generated artifact churn, or broad style refactors.
3. Verify maintainer docs, guards, tests, and source lists describe the same
   boundary.
4. Prepare review notes that explain why the extraction reduces review surface
   and what behavior was intentionally preserved.
5. Record residuals and deferred candidates for future sprints.

### Deliverables

- Review-surface audit artifact.
- Before/after metrics table.
- Final diff-risk register.
- Deferred candidate list.

### Completion Criteria

- The selected cluster has a concrete review-surface reduction.
- Behavior-preserving claims are backed by tests and validation.
- Residuals are documented without expanding Sprint 193 scope.

---

## Day 14: Closeout and Handoff

**Title:** Closeout
**Theme:** Complete final validation, summarize the extraction, and prepare the
next-sprint handoff.
**Time estimate:** 12 hours

### Tasks

1. Re-run the final required validation set, including focused tests, source
   guards, `make format`, `make lint`, `make test`, and CMake/source-list
   parity if applicable.
2. Confirm generated or ignored artifacts are not accidentally staged.
3. Update `WORKING_NOTES.md` and create the Day 14 closeout artifact with
   completed scope, changed files, validation evidence, and residuals.
4. Prepare retrospective inputs covering what worked, what was constrained,
   final metrics, closed claim, and next-sprint handoff.
5. Ensure the branch is ready for retrospective, commit, push, and pull request
   creation.

### Deliverables

- Final Sprint 193 closeout artifact.
- Final validation command log.
- Retrospective input notes.
- PR-ready implementation summary.

### Completion Criteria

- All required quality gates pass before closeout.
- Sprint 193 closes exactly one selected review-surface reduction claim.
- Remaining gaps are documented as residuals or future candidates.
