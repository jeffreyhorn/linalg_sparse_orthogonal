# Sprint 113 Plan: Residual Behavior & Proof-Owner Closeout

**Sprint Duration:** 14 days
**Goal:** Resolve Sprint 110's remaining behavior-sensitive eigensolver and
proof-owner residuals in bounded, evidence-first batches before final Epic 10
integration and closeout. This sprint implements the Sprint 113 section of
`docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 113 begins from:
- the Sprint 110 Matrix builder and Matrix Market private source split
- the Sprint 110 eigensolver handle/workspace validation no-move contract
- the Sprint 110 direct/iterative and SVD proof-owner residual queues
- Sprint 111 user-facing documentation that does not depend on unstable
  internal source ownership
- Sprint 112 package/platform support truth, which must not be widened by
  behavior-owner or proof-owner cleanup

The sprint must:
- refresh Sprint 110 residual debt and explicitly exclude completed work
- select one behavior-sensitive eigensolver owner and prove it directly before
  any source movement
- perform a narrow eigensolver source movement only if the owner proof makes it
  low risk; otherwise publish a no-move contract
- perform one bounded direct/iterative proof-owner cleanup while keeping proof
  values visible at call sites
- perform one bounded SVD proof-owner cleanup after a fresh proof-boundary
  artifact
- capture before/after metrics and remaining non-claims
- close with validation evidence and a final residual handoff for Epic 10
  integration

**End State:** Sprint 113 leaves behind:
- a refreshed and deduplicated Sprint 110 residual-debt boundary
- direct eigensolver behavior-owner proof and either a source movement or a
  no-move contract
- one bounded direct/iterative proof-owner cleanup
- one bounded SVD proof-owner cleanup
- proof-owner metrics and non-claims for future work
- validation evidence showing no public API, install-header, helper-target, or
  reviewed CTest drift unless explicitly intended and proven

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 113 project-plan estimate.

---

## Day 1: Sprint 113 Residual Intake and Boundary Refresh

**Title:** Residual Intake
**Theme:** Establish the Sprint 110 residual-debt boundary and duplicate-work exclusions
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 113 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 110 retrospective residuals and closeout artifacts.
3. Explicitly exclude completed work:
   - Matrix builder ownership decision
   - Matrix builder private source implementation
   - Matrix Market private source split
   - Matrix Market focused validation
   - iterative CG exact-RHS setup cleanup
   - SVD rank-deficient setup cleanup
   - eigensolver handle/workspace validation no-move contract
4. Inventory remaining eigensolver behavior-owner candidates.
5. Inventory remaining direct/iterative and SVD proof-owner cleanup candidates.
6. Create Sprint 113 working notes and artifact directory.
7. Write the residual intake and boundary artifact.

### Deliverables
- residual-debt intake artifact
- duplicate-work exclusion list
- eigensolver behavior-owner candidate inventory
- proof-owner cleanup candidate inventory
- Sprint 113 working-notes baseline

### Completion Criteria
- no completed Sprint 110 work is reintroduced as unresolved debt
- every Sprint 113 item has an initial owner and evidence source
- downstream days can proceed without rediscovering residual scope

---

## Day 2: Eigensolver Behavior Owner Selection

**Title:** Eigen Owner
**Theme:** Select one behavior-sensitive eigensolver owner for direct proof
**Time estimate:** 12 hours

### Tasks
1. Review eigensolver behavior candidates:
   - defaults and option validation
   - backend dispatch
   - grow-m sizing and retry behavior
   - refinement defaults and budgets
   - shift-invert setup
   - shared Lanczos kernels
   - public handle/workspace movement
2. Inspect current eigensolver tests and source ownership.
3. Compare candidates by risk, coupling, testability, and source-movement
   value.
4. Select one owner for Sprint 113 proof.
5. Define required behavior invariants and focused tests for the selected
   owner.
6. Write the eigensolver behavior-owner selection artifact.

### Deliverables
- eigensolver candidate comparison
- selected behavior owner
- invariant list for selected owner
- focused test plan
- no-claim list for unselected eigensolver owners

### Completion Criteria
- exactly one eigensolver owner is selected for proof
- selected owner has direct observable behavior to test
- source movement remains blocked until proof lands

---

## Day 3: Eigensolver Behavior Proof Design

**Title:** Eigen Proof Design
**Theme:** Turn the selected eigensolver owner into focused validation
**Time estimate:** 12 hours

### Tasks
1. Locate the exact source functions, option paths, and tests that exercise the
   selected eigensolver owner.
2. Define focused test cases for expected behavior and boundary behavior.
3. Identify tolerances, iteration budgets, random seeds, matrix fixtures, and
   failure modes needed for stable proof.
4. Identify any helper extraction required for readability without hiding
   proof values.
5. Define the validation command set for Day 4.
6. Write the eigensolver proof design artifact.

### Deliverables
- focused eigensolver test design
- matrix and tolerance choices
- proof-value visibility rules
- validation command list
- Day 4 implementation checklist

### Completion Criteria
- tests are concrete enough to implement without changing public API
- proof design does not rely on source movement
- validation commands match the touched surface

---

## Day 4: Eigensolver Behavior Proof Implementation

**Title:** Eigen Proof
**Theme:** Add focused tests for the selected eigensolver behavior owner
**Time estimate:** 12 hours

### Tasks
1. Implement the focused eigensolver tests from Day 3.
2. Keep matrices, expected values, tolerances, and convergence assertions
   visible near the proof.
3. Avoid public API, install-header, helper-target, and reviewed CTest drift
   unless explicitly required and documented.
4. Run focused eigensolver validation.
5. Record failures, fixes, and final passing evidence.
6. Write the eigensolver behavior proof artifact.

### Deliverables
- focused eigensolver behavior tests
- focused validation output summary
- proof visibility notes
- drift assessment
- Day 4 proof artifact

### Completion Criteria
- selected behavior owner has direct tests
- focused validation passes
- source movement decision can be made from evidence

---

## Day 5: Eigensolver Movement Decision

**Title:** Move Decision
**Theme:** Decide whether the proven eigensolver owner can move safely
**Time estimate:** 12 hours

### Tasks
1. Review Day 4 proof and selected owner coupling.
2. Decide whether a narrow private source movement is low risk.
3. If movement is low risk, define exact file, source-list, Make/CMake, and
   CTest requirements.
4. If movement is not low risk, define a no-move contract and future proof
   requirements.
5. Identify validation commands required for the selected path.
6. Write the eigensolver movement or no-move decision artifact.

### Deliverables
- movement/no-move decision
- source-list and build parity plan if moving
- future proof requirements if not moving
- validation requirements
- explicit non-claims for unproven eigensolver owners

### Completion Criteria
- source movement is not attempted without direct proof
- decision is evidence-backed and reviewable
- Day 6 can execute the chosen path

---

## Day 6: Eigensolver Movement or No-Move Contract

**Title:** Eigen Boundary
**Theme:** Execute the eigensolver source movement or publish the no-move contract
**Time estimate:** 12 hours

### Tasks
1. Execute the Day 5 decision:
   - if moving, perform one narrow private source movement
   - if not moving, publish the no-move contract and future proof queue
2. Update Make/CMake/source-list metadata only if source movement occurs.
3. Preserve public API and install-header stability.
4. Run required focused and source-list/build checks for the chosen path.
5. Capture before/after source ownership metrics.
6. Write the eigensolver movement or no-move contract artifact.

### Deliverables
- private source movement or no-move contract
- source-list/build metadata updates if needed
- focused validation result
- source ownership metrics
- unproven eigensolver residual queue

### Completion Criteria
- chosen path is complete and validated
- no hidden public API or reviewed CTest drift occurs
- unproven eigensolver movement remains explicitly deferred

---

## Day 7: Direct/Iterative Proof-Owner Boundary Selection

**Title:** Direct/Iterative Boundary
**Theme:** Select one bounded direct or iterative proof-owner cleanup
**Time estimate:** 12 hours

### Tasks
1. Review remaining direct/iterative cleanup candidates:
   - QR sequential RHS setup
   - LDLT CSC external dense-reference oracle cleanup
   - CG preconditioner-specific exact-RHS setup
   - GMRES exact-RHS setup
   - BiCGSTAB exact-RHS setup
   - MINRES exact-RHS setup
2. Compare candidates by duplication, proof visibility, behavior risk, and
   validation cost.
3. Select one bounded cleanup target.
4. Define what must remain visible at call sites.
5. Define focused validation commands.
6. Write the direct/iterative proof-owner boundary artifact.

### Deliverables
- direct/iterative candidate comparison
- selected proof-owner cleanup target
- proof visibility rules
- validation command list
- explicit non-claims for unselected candidates

### Completion Criteria
- exactly one bounded cleanup target is selected
- selected cleanup does not hide solver proof values
- Day 8 can implement without broad cross-solver abstraction

---

## Day 8: Direct/Iterative Proof-Owner Cleanup

**Title:** Direct/Iterative Cleanup
**Theme:** Perform one bounded proof-owner cleanup while preserving proof clarity
**Time estimate:** 12 hours

### Tasks
1. Implement the selected Day 7 cleanup.
2. Keep solver calls, options, expected values, residual thresholds,
   convergence assertions, and printed evidence visible where required.
3. Avoid broad cross-solver helper abstractions.
4. Run focused validation for the touched test or solver family.
5. Capture before/after metrics for the touched proof owner.
6. Write the direct/iterative cleanup artifact.

### Deliverables
- bounded direct/iterative cleanup
- focused validation result
- before/after proof-owner metrics
- proof-value visibility notes
- remaining direct/iterative residual queue

### Completion Criteria
- cleanup reduces meaningful duplication or setup noise
- focused validation passes
- proof intent remains visible at call sites

---

## Day 9: SVD Proof Boundary Refresh

**Title:** SVD Boundary
**Theme:** Refresh SVD proof-owner candidates before cleanup
**Time estimate:** 12 hours

### Tasks
1. Review remaining SVD proof-owner cleanup candidates:
   - reconstruction helper movement
   - U/Vt orthogonality helper movement
   - Moore-Penrose product helper extraction
   - dense low-rank proof-loop cleanup
   - sparse low-rank proof-loop cleanup
   - partial-SVD vector/residual cleanup
   - condition-number proof cleanup
2. Inspect current SVD tests and proof-value locations.
3. Compare candidates by duplication, behavior risk, and proof clarity.
4. Select one bounded SVD cleanup target.
5. Define proof visibility and validation requirements.
6. Write the SVD proof-boundary artifact.

### Deliverables
- SVD candidate comparison
- selected SVD cleanup target
- proof visibility rules
- validation command list
- unselected SVD non-claims

### Completion Criteria
- exactly one SVD cleanup target is selected
- cleanup target is bounded and behavior-preserving
- Day 10 can implement without broad SVD proof abstraction

---

## Day 10: SVD Proof-Owner Cleanup

**Title:** SVD Cleanup
**Theme:** Perform one bounded SVD proof-owner cleanup
**Time estimate:** 12 hours

### Tasks
1. Implement the selected Day 9 SVD cleanup.
2. Keep ranks, residuals, orthogonality checks, expected values, and cleanup
   responsibilities visible where required.
3. Avoid broad SVD helper abstractions unless directly justified by the Day 9
   boundary artifact.
4. Run focused SVD validation.
5. Capture before/after metrics for the touched SVD proof owner.
6. Write the SVD cleanup artifact.

### Deliverables
- bounded SVD proof-owner cleanup
- focused SVD validation result
- before/after metrics
- proof-value visibility notes
- remaining SVD residual queue

### Completion Criteria
- cleanup preserves SVD behavior proof
- focused SVD validation passes
- unaddressed SVD proof owners remain explicitly deferred

---

## Day 11: Proof-Owner Metrics and Non-Claims

**Title:** Proof Metrics
**Theme:** Capture before/after metrics and remaining proof-owner boundaries
**Time estimate:** 12 hours

### Tasks
1. Capture before/after file metrics for touched source, test, and artifact
   files.
2. Capture any Make/CMake/source-list/CTest membership changes.
3. Identify remaining eigensolver, direct/iterative, and SVD proof-owner
   residuals.
4. Document why broad cross-solver and broad SVD proof abstractions remain
   unsupported unless directly proven.
5. Verify public API, install-header, helper-target, and reviewed CTest drift
   status.
6. Write the proof-owner metrics and non-claims artifact.

### Deliverables
- before/after file metrics
- membership drift table
- remaining proof-owner residual queue
- broad-abstraction non-claims
- public/API and reviewed-surface drift assessment

### Completion Criteria
- metrics are tied to concrete files and commands
- remaining non-claims are explicit
- Day 12 validation planning has accurate touched-surface scope

---

## Day 12: Integrated Validation Planning

**Title:** Validation Plan
**Theme:** Build the final validation matrix for touched surfaces
**Time estimate:** 12 hours

### Tasks
1. Review all code, test, build, source-list, docs, and artifact changes made
   during Sprint 113.
2. Map each touched surface to required validation commands.
3. Define focused rerun commands for eigensolver, direct/iterative, and SVD
   changes.
4. Define full quality checks required by `.c` or `.h` changes.
5. Define documentation hygiene and local link checks for Sprint artifacts.
6. Write the integrated validation plan artifact.

### Deliverables
- touched-surface inventory
- validation command matrix
- focused test rerun list
- full quality gate requirement decision
- Day 13 execution checklist

### Completion Criteria
- every touched surface has a validation owner
- no required quality check is omitted
- Day 13 can run validation without redesigning the matrix

---

## Day 13: Integrated Validation Execution

**Title:** Integrated Validation
**Theme:** Run required validation and capture final evidence
**Time estimate:** 12 hours

### Tasks
1. Run focused validation for eigensolver changes.
2. Run focused validation for direct/iterative cleanup.
3. Run focused validation for SVD cleanup.
4. Run source-list, Make/CMake, and CTest checks required by any build or
   source movement.
5. Run `make format && make lint && make test` if `.c` or `.h` files changed.
6. Run documentation hygiene checks for Sprint 113 artifacts.
7. Capture failures or final passing evidence in the integrated validation
   artifact.

### Deliverables
- integrated validation command log
- focused validation results
- full quality gate result if required
- source-list/build/CTest drift result
- documentation hygiene result

### Completion Criteria
- all required checks pass before closeout
- no public API/install-header/helper-target/reviewed CTest drift is hidden
- residual failures, if any, are blocking and escalated before closeout

---

## Day 14: Sprint 113 Closeout and Handoff

**Title:** Closeout Handoff
**Theme:** Close Sprint 113 with proof-owner truth and final residual queue
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 113 artifacts, working notes, code/test changes, metrics,
   and validation output.
2. Confirm all seven Sprint 113 project-plan items are complete or explicitly
   deferred.
3. Summarize eigensolver behavior-owner outcome and movement/no-move decision.
4. Summarize direct/iterative and SVD cleanup outcomes.
5. Record residual eigensolver, direct/iterative, SVD, validation, and
   proof-abstraction debt for final Epic integration.
6. Run final applicable documentation and hygiene checks.
7. Write the closeout and handoff artifact.

### Deliverables
- completed Sprint 113 item checklist
- eigensolver owner and movement/no-move summary
- direct/iterative cleanup summary
- SVD cleanup summary
- residual deferred-debt queue
- validation summary
- Day 14 closeout and handoff artifact

### Completion Criteria
- all Sprint 113 items are closed or explicitly deferred
- residuals are dependency-ordered and non-duplicative
- final checks pass
- final Epic 10 integration has a clear proof-owner handoff
