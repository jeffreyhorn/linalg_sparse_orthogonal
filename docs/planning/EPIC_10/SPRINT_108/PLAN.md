# Sprint 108 Plan: Residual Proof-Owner & Source Boundary Follow-Through

**Sprint Duration:** 14 days
**Goal:** Convert Sprint 107's residual deferred debt into bounded
follow-through work for remaining proof-owner cleanup and source-owner boundary
planning without duplicating completed Sprint 107 helper extractions or changing
public support surfaces opportunistically. This sprint implements the Sprint 108
section of `docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 108 begins from:
- Sprint 107 residual-debt retrospective and boundary artifacts
- merged QR, iterative, SVD, and LDLT CSC helper extractions
- the `src/sparse_eigs.c` and `src/sparse_matrix.c` deferral contracts
- current reviewed constraints around public API, install headers, helper
  targets, reviewed test counts, and source-list parity

The sprint must:
- refresh the residual proof-owner boundary after Sprint 107
- explicitly exclude Sprint 107's completed helper extractions from new work
- extract at most one additional LDLT CSC proof helper if the boundary remains
  low risk
- perform bounded QR, iterative, and SVD residual proof-owner cleanup
- prepare an eigensolver source extraction feasibility plan
- review central sparse matrix shell public behavior and private dependencies
- close with validation evidence, maintainability metrics, and a residual queue

**End State:** Sprint 108 leaves behind:
- a refreshed residual proof-owner boundary artifact
- one additional bounded LDLT CSC proof helper if safe
- focused QR, iterative, and SVD cleanup artifacts and changes
- a source-list and cross-backend feasibility plan for `src/sparse_eigs.c`
- a public-behavior and private-header dependency review for
  `src/sparse_matrix.c`
- validation evidence proving no accidental public API, install-header,
  reviewed test-count, or helper-target drift

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 108 project-plan estimate.

---

## Day 1: Sprint 108 Scope & Carry-Forward Intake

**Title:** Carry-Forward Intake
**Theme:** Convert Sprint 107 residual deferred debt into a bounded Sprint 108
work package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 108 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 107 `WORKING_NOTES.md`, artifacts, and retrospective,
   emphasizing the residual deferred debt section.
3. Inventory Sprint 108 owners:
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
   - `tests/test_iterative.c`
   - `tests/test_svd.c`
   - `src/sparse_eigs.c`
   - `src/sparse_matrix.c`
4. Create Sprint 108 working notes and artifacts directory.
5. Record validation expectations for docs-only, test-touch, source-touch,
   build-system-touch, and mixed cleanup days.

### Deliverables
- Sprint 108 workstream inventory
- working-notes baseline
- initial artifacts directory structure
- validation expectation list

### Completion Criteria
- every Sprint 108 project-plan item has day-level ownership
- Sprint 107 completed extractions are marked as exclusions
- validation expectations are explicit before boundary or cleanup work starts

---

## Day 2: Residual Proof-Owner Boundary Refresh

**Title:** Boundary Refresh
**Theme:** Re-rank remaining proof owners after Sprint 107 from live evidence
**Time estimate:** 12 hours

### Tasks
1. Generate current size, helper, fixture, assertion-density, and repeated-setup
   inventories for the four residual test owners.
2. Compare the live inventories with Sprint 107 before/after notes.
3. Exclude already-completed row-adjacency, QR small-fixture, iterative
   matrix-free, and SVD diagonal/rank-1 cleanup.
4. Rank remaining proof-owner candidates by failure-localization value,
   reviewability, validation cost, and dependency order.
5. Write the residual proof-owner boundary refresh artifact.

### Deliverables
- refreshed proof-owner inventory
- duplicate-work exclusion list
- ranked Sprint 108 cleanup queue
- validation-cost map

### Completion Criteria
- all Sprint 107 residual proof-owner items have a Sprint 108 disposition
- no selected work duplicates a completed Sprint 107 extraction
- cleanup order is explicit and dependency-safe

---

## Day 3: LDLT CSC Oracle Boundary

**Title:** LDLT Oracle Boundary
**Theme:** Select at most one additional `tests/test_ldlt_csc.c` proof helper
without hiding direct-solver intent
**Time estimate:** 12 hours

### Tasks
1. Inspect remaining LDLT CSC assertion, residual, dense-oracle, and reference
   comparison patterns.
2. Identify candidates that are narrower than the surrounding proof logic.
3. Select at most one named helper candidate or explicitly defer all candidates.
4. Define call-site readability, assertion specificity, and helper placement
   rules.
5. Write the LDLT CSC oracle-boundary artifact and focused validation plan.

### Deliverables
- LDLT CSC oracle-boundary artifact
- selected helper candidate or explicit deferral
- no-new-target rationale
- focused validation command list

### Completion Criteria
- any selected helper is narrow and reviewable
- direct-solver proof intent remains visible at call sites
- validation commands are known before edits begin

---

## Day 4: LDLT CSC Oracle Helper Follow-Through

**Title:** LDLT Helper Follow-Through
**Theme:** Implement the approved LDLT CSC proof-helper cleanup or record the
deferral
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 3 approved helper extraction if the boundary selected one.
2. Update only the approved call sites.
3. Preserve assertion specificity, failure locality, and direct CSC proof
   readability.
4. Run focused LDLT CSC validation and formatting for touched files.
5. Record before/after metrics and any remaining LDLT CSC residual debt.

### Deliverables
- bounded LDLT CSC proof-helper change or explicit no-change deferral
- updated focused tests if code changed
- before/after proof-owner metrics
- Day 4 validation notes

### Completion Criteria
- focused LDLT CSC checks pass if code changed
- no reviewed test-count or helper-target drift occurs
- any remaining LDLT CSC cleanup is explicitly queued or rejected

---

## Day 5: QR Residual Fixture Boundary

**Title:** QR Residual Boundary
**Theme:** Identify QR generated-fixture and exact-RHS setup that can move
without hiding proof assertions
**Time estimate:** 12 hours

### Tasks
1. Inventory remaining generated fixtures, tall/economy builders,
   diagonal/singleton setup, and SuiteSparse exact-RHS setup in
   `tests/test_qr.c`.
2. Separate safe construction helpers from inline proof logic.
3. Define call-site readability rules for rank, solve, refinement,
   reconstruction, and residual assertions.
4. Select a bounded QR cleanup batch or explicitly defer unsafe candidates.
5. Write the QR residual fixture-boundary artifact.

### Deliverables
- QR residual fixture inventory
- safe-builder extraction list
- inline-proof preservation notes
- focused QR validation plan

### Completion Criteria
- selected QR work does not hide core proof assertions
- helper placement stays local unless a future target is explicitly approved
- validation scope is defined before edits begin

---

## Day 6: QR Fixture Follow-Through

**Title:** QR Follow-Through
**Theme:** Extract the approved QR residual fixture setup while preserving test
intent
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 5 approved QR fixture cleanup.
2. Check fixture construction errors at helper boundaries.
3. Update only approved call sites.
4. Keep rank, solve, residual, refinement, and reconstruction assertions
   visible.
5. Run focused QR validation and record metrics.

### Deliverables
- bounded QR fixture follow-through change
- updated QR call sites
- focused validation results
- remaining QR residual debt notes

### Completion Criteria
- focused QR tests pass
- fixture failures localize to fixture construction
- remaining QR cleanup is explicitly deferred or rejected

---

## Day 7: Iterative Convergence Boundary

**Title:** Iterative Boundary
**Theme:** Define safe cleanup boundaries for convergence-sensitive iterative
test setup
**Time estimate:** 12 hours

### Tasks
1. Inventory repeated matrix, RHS, option, restart, preconditioner, and result
   setup in `tests/test_iterative.c`.
2. Mark solver options, restarts, convergence flags, residuals, and direct
   comparisons that must stay visible.
3. Select one bounded cleanup batch that does not hide solver behavior.
4. Define focused validation for touched iterative solver families.
5. Write the iterative convergence-boundary artifact.

### Deliverables
- iterative setup inventory
- inline convergence-proof preservation notes
- selected cleanup batch
- focused iterative validation plan

### Completion Criteria
- selected work avoids hiding solver options or convergence evidence
- direct comparisons remain readable at call sites
- validation commands are ready before edits begin

---

## Day 8: Iterative Convergence Cleanup

**Title:** Iterative Cleanup
**Theme:** Apply the approved iterative cleanup without changing convergence
proof behavior
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 7 approved iterative cleanup.
2. Preserve visible solver options, restarts, preconditioners, convergence
   results, and direct comparisons.
3. Update only approved call sites.
4. Run focused iterative validation and formatting for touched files.
5. Record before/after metrics and residual iterative proof-owner debt.

### Deliverables
- bounded iterative cleanup change
- focused validation results
- before/after metrics
- residual convergence-proof queue

### Completion Criteria
- focused iterative checks pass
- no solver behavior or comparison surface changes unintentionally
- remaining iterative cleanup is explicitly deferred or rejected

---

## Day 9: SVD Validation Lane Boundary

**Title:** SVD Validation Lane
**Theme:** Create a dedicated validation lane before moving remaining SVD proof
logic
**Time estimate:** 12 hours

### Tasks
1. Inventory remaining rank, oracle, reconstruction, pseudoinverse, low-rank,
   partial-SVD, and condition-number proof logic in `tests/test_svd.c`.
2. Classify candidates by assertion visibility, oracle coupling, fixture
   construction, and validation cost.
3. Select one safe helper family or record a no-extraction deferral.
4. Define focused SVD validation for the selected proof surface.
5. Write the SVD oracle and reconstruction boundary artifact.

### Deliverables
- SVD proof-owner inventory
- dedicated SVD validation lane
- selected helper family or explicit deferral
- focused SVD validation plan

### Completion Criteria
- SVD proof movement has a validation lane before edits
- reconstruction and oracle assertions remain visible
- unsafe helper candidates are explicitly deferred

---

## Day 10: SVD Oracle and Reconstruction Cleanup

**Title:** SVD Proof Cleanup
**Theme:** Apply one bounded SVD helper-family cleanup under the Day 9
validation lane
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 9 approved SVD helper-family cleanup.
2. Preserve visible rank, oracle, reconstruction, pseudoinverse, low-rank,
   partial-SVD, and condition-number assertions.
3. Validate fixture construction failure handling where helpers build matrices.
4. Run focused SVD validation and formatting for touched files.
5. Record before/after metrics and remaining SVD proof-owner debt.

### Deliverables
- bounded SVD proof-owner cleanup
- focused validation results
- SVD before/after metrics
- residual SVD proof-owner queue

### Completion Criteria
- focused SVD tests pass
- no oracle or reconstruction proof is hidden behind opaque helpers
- remaining SVD cleanup is explicitly deferred or rejected

---

## Day 11: Eigensolver Source Feasibility Boundary

**Title:** Eigs Feasibility
**Theme:** Plan a future `src/sparse_eigs.c` extraction without landing a risky
split prematurely
**Time estimate:** 12 hours

### Tasks
1. Inspect `src/sparse_eigs.c`, eigensolver internal headers, Makefile source
   lists, CMake source lists, and source-list checker expectations.
2. Review Sprint 103 comparison surfaces and Sprint 107 eigensolver deferral
   notes.
3. Evaluate dense Jacobi feasibility and grow-m refinement boundaries as
   possible future seams.
4. Map required cross-backend spectral validation for any future split.
5. Write the eigensolver source feasibility boundary artifact.

### Deliverables
- eigensolver source ownership inventory
- dense Jacobi and grow-m refinement feasibility notes
- build-system/source-list impact map
- cross-backend validation plan

### Completion Criteria
- any future eigensolver split has explicit prerequisites
- no source split lands without evidence that it is low risk
- validation and build-system follow-through are mapped

---

## Day 12: Eigensolver Feasibility Closeout

**Title:** Eigs Plan Closeout
**Theme:** Convert eigensolver source feasibility into an actionable future
handoff
**Time estimate:** 12 hours

### Tasks
1. Decide whether Sprint 108 should land no source change or a very small
   preparatory documentation-only change.
2. Record Make/CMake/source-list requirements for any future eigensolver source
   owner.
3. Define focused and broad validation commands for a future extraction PR.
4. Update working notes with explicit non-claims and non-goals.
5. Produce the eigensolver feasibility closeout artifact.

### Deliverables
- eigensolver feasibility closeout artifact
- future extraction checklist
- explicit no-change or preparatory-change rationale
- residual eigensolver source queue

### Completion Criteria
- the eigensolver handoff is actionable without implying an unearned split
- cross-backend validation requirements are explicit
- source-list and build-system dependencies are not hidden

---

## Day 13: Matrix Shell Public-Behavior Review

**Title:** Matrix Shell Review
**Theme:** Review central `src/sparse_matrix.c` behavior before any future
shell extraction
**Time estimate:** 12 hours

### Tasks
1. Inventory public behavior in `src/sparse_matrix.c`: allocation, insertion,
   removal, permutation state, factor-state interactions, and compatibility
   expectations.
2. Map private headers, internal state ownership, and downstream source
   dependencies.
3. Identify public behavior tests that must guard any future shell extraction.
4. Define prerequisites for separating internal matrix shell ownership.
5. Write the matrix shell public-behavior review artifact.

### Deliverables
- central matrix public-behavior inventory
- private-header dependency map
- future shell extraction prerequisites
- validation guardrail list

### Completion Criteria
- public behavior and compatibility constraints are explicit
- any future shell extraction has named prerequisites
- no central matrix source move occurs without a guardrail plan

---

## Day 14: Validation, Metrics & Sprint 108 Closeout

**Title:** Sprint 108 Closeout
**Theme:** Validate all touched surfaces, update maintainability metrics, and
publish the residual queue
**Time estimate:** 12 hours

### Tasks
1. Run required checks based on touched files:
   - docs-only checks for documentation-only changes
   - focused tests and `make format && make lint && make test` if `.c` or `.h`
     files changed
   - source-list and CMake checks if build membership changed
2. Verify no accidental public API, install-header, helper-target, or reviewed
   test-count drift.
3. Capture before/after maintainability metrics for touched proof owners.
4. Update Sprint 108 working notes with validation evidence and residuals.
5. Write the Sprint 108 closeout artifact and retrospective input notes.

### Deliverables
- final validation evidence
- before/after maintainability metrics
- reviewed drift checks
- residual queue for Sprint 109 and beyond
- Sprint 108 closeout notes

### Completion Criteria
- required checks pass for the actual touched-file set
- no unsupported public/support-surface claims are introduced
- residual work is explicit, ordered, and non-duplicative
