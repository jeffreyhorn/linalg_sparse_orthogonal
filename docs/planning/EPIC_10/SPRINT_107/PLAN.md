# Sprint 107 Plan: Residual Maintainability Debt & Proof-Owner Cleanup

**Sprint Duration:** 14 days
**Goal:** Resolve Sprint 106's explicitly deferred maintainability debt with
bounded proof-owner cleanup, fresh source/test boundaries, and no opportunistic
public API or install-header expansion. This sprint implements the Sprint 107
section of `docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 107 begins from:
- Sprint 106 extraction artifacts, working notes, retrospective, and residual
  deferred debt
- source-list and CMake reconciliation from Sprint 106
- the current reviewed test-count and helper-target constraints
- the explicit Sprint 106 non-goals for public API, install headers, broad
  rewrites, and central matrix shell extraction

The sprint must:
- convert Sprint 106 residual debt into a fresh boundary artifact
- extract one narrow `tests/test_ldlt_csc.c` proof helper
- clean repeated fixture builders in `tests/test_qr.c` and
  `tests/test_iterative.c`
- perform a bounded `tests/test_svd.c` proof-owner cleanup
- create a fresh `src/sparse_eigs.c` boundary and optionally split one
  low-risk helper
- document why `src/sparse_matrix.c` remains central API/compatibility
  territory
- close with validation evidence, updated maintainability metrics, and a
  residual handoff

**End State:** Sprint 107 leaves behind:
- a residual-debt boundary artifact for remaining large source and proof owners
- a smaller, more focused LDLT CSC proof owner
- cleaner QR, iterative, and SVD test fixture ownership
- an eigensolver source boundary and optional first low-risk split
- a documented central matrix shell non-extraction contract
- validation artifacts proving no accidental public API, install-header, or
  reviewed test-count drift

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 107 project-plan estimate.

---

## Day 1: Sprint 107 Scope & Residual Debt Intake

**Title:** Residual Intake
**Theme:** Convert Sprint 106 deferred debt into a bounded Sprint 107 work
package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 107 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 106 `WORKING_NOTES.md`, artifacts, and
   `RETROSPECTIVE.md`, with emphasis on the residual deferred debt section.
3. Inventory the six residual owners:
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
   - `tests/test_iterative.c`
   - `tests/test_svd.c`
   - `src/sparse_eigs.c`
   - `src/sparse_matrix.c`
4. Create Sprint 107 working notes and artifacts directory.
5. Record validation expectations for docs-only, test-touch, source-touch,
   build-system-touch, and mixed cleanup days.

### Deliverables
- Sprint 107 workstream inventory
- working-notes baseline
- initial artifacts directory structure
- validation expectation list

### Completion Criteria
- every Sprint 107 project-plan item has day-level ownership
- Sprint 106 residual debt is visible in working notes
- validation expectations are explicit before boundary or extraction work starts

---

## Day 2: Residual Boundary Re-rank

**Title:** Boundary Re-rank
**Theme:** Rank remaining proof-owner and source-owner debt from live evidence
**Time estimate:** 12 hours

### Tasks
1. Generate current size, symbol, helper, fixture, and assertion-density
   inventories for the six residual owners.
2. Compare those inventories against Sprint 106 before/after metrics.
3. Identify duplicate or already-solved items so Sprint 107 does not repeat
   Sprint 106 extraction work.
4. Rank each residual owner by risk reduction, failure-localization value,
   validation cost, and dependency ordering.
5. Write the residual boundary re-rank artifact.

### Deliverables
- residual source/test owner inventory
- duplicate-work exclusion notes
- ranked Sprint 107 cleanup queue
- first-pass validation-cost map

### Completion Criteria
- all deferred items from Sprint 106 have a Sprint 107 disposition
- no item duplicates a Sprint 106 completed extraction
- cleanup order is explicit and dependency-safe

---

## Day 3: LDLT CSC Proof Boundary

**Title:** LDLT Proof Boundary
**Theme:** Freeze the narrow `tests/test_ldlt_csc.c` proof-helper seam before
editing tests
**Time estimate:** 12 hours

### Tasks
1. Inspect `tests/test_ldlt_csc.c` row-adjacency assertions, residual checks,
   dense references, and oracle helper patterns.
2. Select one narrow row-adjacency assertion helper or residual/oracle helper
   candidate.
3. Define helper naming, placement, call sites, and failure-message behavior.
4. Verify the candidate does not require a new compiled test helper target.
5. Write the LDLT CSC proof-boundary artifact with focused validation commands.

### Deliverables
- LDLT CSC proof-helper boundary artifact
- selected extraction candidate
- no-new-target rationale
- focused validation command list

### Completion Criteria
- the selected helper is narrow enough to review safely
- direct CSC proof intent remains visible at call sites
- validation commands are known before the test edit starts

---

## Day 4: LDLT CSC Proof Helper Extraction

**Title:** LDLT Helper Extraction
**Theme:** Extract the selected `tests/test_ldlt_csc.c` helper without changing
solver behavior
**Time estimate:** 12 hours

### Tasks
1. Implement the selected helper extraction in the existing test owner or an
   appropriate local helper file.
2. Update only the call sites covered by the Day 3 boundary.
3. Preserve assertion specificity and failure-localization quality.
4. Run focused LDLT CSC validation and formatting for touched files.
5. Record before/after metrics and validation output in working notes.

### Deliverables
- bounded LDLT CSC proof-helper extraction
- updated focused tests
- before/after proof-owner metrics
- Day 4 validation notes

### Completion Criteria
- focused LDLT CSC checks pass
- no reviewed test-count or helper-target drift occurs
- `tests/test_ldlt_csc.c` loses a measurable repeated proof pattern

---

## Day 5: QR Fixture Boundary

**Title:** QR Fixture Boundary
**Theme:** Identify repeated `tests/test_qr.c` matrix/vector builders without
hiding solve or reconstruction intent
**Time estimate:** 12 hours

### Tasks
1. Inventory repeated QR matrix, vector, RHS, tolerance, and reconstruction
   setup patterns.
2. Classify candidates as safe builders, assertion helpers, or proof logic that
   should remain inline.
3. Select a bounded builder extraction batch.
4. Define helper naming, local placement, and call-site readability rules.
5. Write the QR fixture-boundary artifact.

### Deliverables
- QR fixture repetition inventory
- safe-builder extraction list
- inline-proof preservation notes
- focused QR validation plan

### Completion Criteria
- the selected QR cleanup avoids hiding core solve/reconstruction assertions
- helper placement does not require a new compiled target
- validation scope is defined before edits begin

---

## Day 6: QR Fixture Cleanup Batch

**Title:** QR Fixture Cleanup
**Theme:** Extract repeated QR setup builders while preserving proof clarity
**Time estimate:** 12 hours

### Tasks
1. Implement the selected QR fixture-builder cleanup.
2. Update only the call sites approved by the Day 5 boundary.
3. Keep solver assertions, rank checks, residual checks, and reconstruction
   checks visible at test sites.
4. Run focused QR validation and formatting for touched files.
5. Record before/after metrics and remaining QR proof-owner debt.

### Deliverables
- cleaned QR fixture builders
- updated QR call sites
- focused validation results
- QR residual debt notes

### Completion Criteria
- focused QR tests pass
- proof intent remains readable at edited call sites
- any remaining QR cleanup is explicitly deferred

---

## Day 7: Iterative Fixture Boundary

**Title:** Iterative Boundary
**Theme:** Select convergence-sensitive `tests/test_iterative.c` builders that
can move safely
**Time estimate:** 12 hours

### Tasks
1. Inventory repeated matrix, RHS, initial guess, option, preconditioner, and
   result-check setup in `tests/test_iterative.c`.
2. Separate safe reusable builders from convergence-sensitive assertions that
   must remain inline.
3. Select a first cleanup batch limited to matrix/RHS builders.
4. Define helper names and call-site readability criteria.
5. Write the iterative fixture-boundary artifact.

### Deliverables
- iterative fixture inventory
- safe matrix/RHS builder batch
- convergence-sensitive no-move list
- focused iterative validation plan

### Completion Criteria
- selected helpers cannot obscure convergence or residual behavior
- helper extraction does not change solver options or expected outcomes
- validation scope is documented before edits begin

---

## Day 8: Iterative Fixture Cleanup Batch

**Title:** Iterative Fixture Cleanup
**Theme:** Extract safe iterative matrix/RHS builders without changing
convergence proof behavior
**Time estimate:** 12 hours

### Tasks
1. Implement the selected `tests/test_iterative.c` builder cleanup.
2. Update approved call sites while preserving inline convergence assertions.
3. Run focused iterative tests and formatting for touched files.
4. Record before/after metrics and any remaining convergence-sensitive debt.
5. Update working notes with any validation nuance from iterative solver runs.

### Deliverables
- cleaned iterative matrix/RHS builders
- updated focused call sites
- focused validation results
- convergence-sensitive residual debt notes

### Completion Criteria
- focused iterative checks pass
- convergence behavior remains directly visible in tests
- remaining iterative cleanup is bounded and documented

---

## Day 9: SVD Proof-Owner Boundary

**Title:** SVD Boundary
**Theme:** Plan a dedicated `tests/test_svd.c` cleanup before moving rank or
oracle helpers
**Time estimate:** 12 hours

### Tasks
1. Inventory repeated rank, reconstruction, oracle, partial-SVD, low-rank, and
   condition-number proof patterns.
2. Identify helpers that are fixture setup versus proof interpretation.
3. Choose a cleanup batch that avoids moving rank/oracle interpretation before
   focused validation is defined.
4. Define helper names, placement, and no-move rules.
5. Write the SVD proof-owner boundary artifact.

### Deliverables
- SVD proof-owner inventory
- selected safe cleanup batch
- rank/oracle no-move list
- focused SVD validation plan

### Completion Criteria
- SVD proof interpretation remains visible in planned edits
- helper movement is constrained by validation evidence
- no broad SVD proof-owner rewrite is implied

---

## Day 10: SVD Proof-Owner Cleanup Batch

**Title:** SVD Cleanup
**Theme:** Perform the bounded SVD cleanup with focused validation before any
future rank/oracle movement
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 9 SVD cleanup batch.
2. Preserve rank, oracle, reconstruction, and residual interpretations at test
   sites unless the boundary explicitly permits helper movement.
3. Run focused SVD validation and formatting for touched files.
4. Record before/after metrics and validation output.
5. Update residual notes for any rank/oracle helper work that remains unsafe.

### Deliverables
- bounded SVD proof-owner cleanup
- focused SVD validation results
- before/after SVD metrics
- future rank/oracle helper deferral notes

### Completion Criteria
- focused SVD tests pass
- proof interpretation remains reviewable
- any further SVD cleanup has explicit prerequisites

---

## Day 11: Eigensolver Source Boundary

**Title:** Eigs Boundary
**Theme:** Create a fresh `src/sparse_eigs.c` boundary tied to Sprint 103
comparison surfaces
**Time estimate:** 12 hours

### Tasks
1. Inspect `src/sparse_eigs.c`, related internal headers, eigensolver tests,
   and Sprint 103 comparison artifacts.
2. Identify workspace, dispatch, refinement, and comparison-adjacent helper
   groups.
3. Select one optional low-risk split candidate, or document why no source edit
   should occur in Sprint 107.
4. Map Make, CMake, source-list, and reviewed parity follow-through if a split
   is selected.
5. Write the eigensolver source-boundary artifact.

### Deliverables
- eigensolver source-boundary artifact
- optional low-risk split candidate or no-split rationale
- build-system follow-through checklist
- focused eigensolver validation plan

### Completion Criteria
- `src/sparse_eigs.c` is not split without a fresh boundary
- Sprint 103 comparison semantics remain protected
- build-system implications are known before any source edit

---

## Day 12: Eigensolver First Split or Deferral

**Title:** Eigs Split or Deferral
**Theme:** Execute only the safest eigensolver helper split, or document
deferral if the boundary is not strong enough
**Time estimate:** 12 hours

### Tasks
1. If Day 11 selected a low-risk split, implement it with minimal logic
   movement and exact source-list/CMake updates.
2. If no split is safe, write the deferral artifact and perform no source
   extraction.
3. Run focused eigensolver validation and any required source-list/build checks.
4. Record before/after metrics or deferral rationale in working notes.
5. Update the residual queue for future eigensolver maintainability work.

### Deliverables
- optional eigensolver helper split, or explicit no-split deferral artifact
- focused validation results
- source-list/CMake updates if needed
- updated eigensolver residual queue

### Completion Criteria
- eigensolver behavior remains covered by focused tests
- build-system parity is exact if source files changed
- unsafe source movement is deferred rather than forced

---

## Day 13: Central Matrix Shell Deferral Contract

**Title:** Matrix Shell Contract
**Theme:** Document why `src/sparse_matrix.c` remains central API and
compatibility territory
**Time estimate:** 12 hours

### Tasks
1. Review `src/sparse_matrix.c` responsibilities, public headers, install
   headers, compatibility behavior, and Sprint 101 compressed-first notes.
2. Identify potential future split preconditions for allocation, mutation,
   lifecycle, import/export, and compatibility helpers.
3. Document risks that make opportunistic extraction inappropriate in Sprint
   107.
4. Add maintainer guidance or an artifact that fences future matrix-shell
   extraction behind explicit public-behavior review.
5. Reconcile Sprint 107 residual notes with the matrix-shell contract.

### Deliverables
- central matrix shell deferral contract
- future split preconditions
- public behavior and compatibility risk notes
- updated residual queue

### Completion Criteria
- `src/sparse_matrix.c` non-extraction is explicit and justified
- future split prerequisites are concrete
- no public API or install-header change is introduced

---

## Day 14: Validation, Metrics & Closeout

**Title:** Sprint Closeout
**Theme:** Validate Sprint 107 cleanup and publish maintainability evidence
**Time estimate:** 12 hours

### Tasks
1. Run the required checks for all touched files, including the full quality
   gate if any `.c` or `.h` files changed.
2. Verify no public API, install-header, reviewed test-count, or helper-target
   drift occurred unless explicitly approved and documented.
3. Generate final before/after maintainability metrics for touched source and
   proof owners.
4. Write the Sprint 107 validation and closeout artifact.
5. Update working notes with completed work, deferred items, and handoff
   guidance for Sprint 108.

### Deliverables
- final validation output
- before/after maintainability metrics
- public API/install-header/test-count drift check
- Sprint 107 closeout artifact
- Sprint 108 handoff notes

### Completion Criteria
- required checks pass for every touched file category
- Sprint 107 deliverables are linked from working notes
- residual debt is either resolved or explicitly handed forward with rationale
