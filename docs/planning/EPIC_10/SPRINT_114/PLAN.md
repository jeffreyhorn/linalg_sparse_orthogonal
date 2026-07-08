# Sprint 114 Plan: Residual Eigensolver, Direct/Iterative & SVD Proof-Owner Follow-Through

**Sprint Duration:** 14 days
**Goal:** Convert Sprint 113's residual proof-owner debt into dependency-ordered
follow-through batches. Sprint 114 proves eigensolver movement prerequisites
before any source split, cleans up one bounded direct/iterative exact-RHS
batch, and performs an SVD proof-owner cleanup without introducing broad
abstractions, public API drift, helper-target drift, source-list drift, or
reviewed CTest membership changes unless explicitly proven.

**Starting Point:** Sprint 114 begins from:
- Sprint 113 grow-m behavior proof, eigensolver no-move contract, and residual
  proof-owner queue
- Sprint 113 LDLT CSC external dense-reference cleanup as the direct/iterative
  proof-owner example
- Sprint 113 partial-SVD residual helper cleanup as the SVD proof-owner example
- Sprint 113 metrics and non-claims for eigensolver movement, cross-solver
  proof abstraction, SVD proof abstraction, public API changes, helper target
  changes, source-list changes, and reviewed CTest membership changes

The sprint must:
- deduplicate Sprint 113 residual debt before implementing anything
- prove `lanczos_iterate_op` behavior across basic, thick-restart, and
  LOBPCG-adjacent dispatch paths
- add repeated/clustered spectrum proof before moving Ritz selection
- prove Ritz vector lifting and publication boundaries before extracting
  vector-publication helpers
- prove partial-result publication after `m_cap` exhaustion
- prove shift-invert grow-m conversion before any source movement
- decide whether one narrow eigensolver source movement is safe, or publish a
  continued no-move decision
- clean up QR, CG, GMRES, BiCGSTAB, and MINRES exact-RHS setup without hiding
  solver-specific proof values
- clean up bounded SVD proof owners while preserving storage, leading-dimension,
  product-dimension, low-rank, and condition-number evidence
- close with validation, metrics, and explicit non-claim handoff

**End State:** Sprint 114 leaves behind:
- a duplicate-fenced residual proof-owner intake artifact
- direct eigensolver behavior proofs for the required Lanczos, Ritz,
  publication, partial-result, and shift-invert grow-m paths
- an evidence-backed eigensolver movement or continued no-move decision
- direct/iterative exact-RHS cleanup with solver-specific proof values intact
- SVD proof-owner cleanup with storage and leading-dimension conventions
  preserved
- validation evidence and downstream non-claims for package, adoption, and
  final Epic 10 closeout work

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 114 project-plan estimate.

---

## Day 1: Residual Proof-Owner Intake and Duplicate Fence

**Title:** Residual Intake
**Theme:** Establish the Sprint 113 residual-debt boundary and implementation order
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 114 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 113 retrospective residual deferred debt and closeout
   artifacts.
3. Explicitly exclude work completed in Sprint 113:
   - grow-m behavior selection and proof
   - eigensolver no-move contract
   - LDLT CSC external dense-reference cleanup
   - partial-SVD residual helper cleanup
   - proof-owner metrics and validation artifacts
4. Build the dependency order for eigensolver proof, direct/iterative cleanup,
   SVD cleanup, and validation.
5. Create Sprint 114 working notes and artifact directory.
6. Write the residual intake and duplicate-fence artifact.

### Deliverables
- Sprint 114 working-notes baseline
- artifact directory
- residual-debt intake artifact
- duplicate-work exclusion list
- dependency-ordered implementation map

### Completion Criteria
- no completed Sprint 113 work is reintroduced as unresolved debt
- every Sprint 114 project-plan item has a day-level owner
- downstream proof days can proceed without rediscovering scope

---

## Day 2: Lanczos Iterate Behavior Proof Design

**Title:** Lanczos Design
**Theme:** Design direct proof for `lanczos_iterate_op` behavior across dispatch paths
**Time estimate:** 12 hours

### Tasks
1. Locate the basic Lanczos, thick-restart, and LOBPCG-adjacent paths that
   exercise `lanczos_iterate_op` or depend on its observable behavior.
2. Identify current tests that already prove adjacent behavior.
3. Define matrix fixtures, tolerances, iteration budgets, and expected
   observable results.
4. Define failure and boundary cases that show the helper behavior without
   exposing private internals.
5. Decide where the proof belongs in the current test files.
6. Write the Lanczos behavior proof design artifact.

### Deliverables
- dispatch-path inventory
- fixture and tolerance choices
- focused test checklist
- proof visibility rules
- validation command list

### Completion Criteria
- all three required paths have concrete test coverage targets
- the design avoids public API and source-list drift
- Day 3 can implement without changing the proof scope

---

## Day 3: Lanczos Iterate Behavior Proof Implementation

**Title:** Lanczos Proof
**Theme:** Add focused `lanczos_iterate_op` behavior tests
**Time estimate:** 12 hours

### Tasks
1. Implement focused tests for the basic Lanczos path.
2. Implement focused tests for the thick-restart path.
3. Implement focused tests for the LOBPCG-adjacent dispatch path.
4. Keep matrices, tolerances, expected values, and iteration budgets visible at
   the call sites.
5. Run focused eigensolver validation.
6. Write the implementation artifact with validation evidence and drift notes.

### Deliverables
- focused Lanczos behavior tests
- focused validation output summary
- proof-value visibility notes
- drift assessment
- Day 3 proof artifact

### Completion Criteria
- all targeted Lanczos behavior tests pass
- no unsupported source movement is introduced
- repeated/clustered Ritz proof can build on the new evidence

---

## Day 4: Repeated and Clustered Ritz Selection Proof Design

**Title:** Ritz Design
**Theme:** Design spectrum proof before moving Ritz selection logic
**Time estimate:** 12 hours

### Tasks
1. Inspect Ritz selection logic and existing eigensolver spectrum tests.
2. Identify repeated and clustered spectrum fixtures that are deterministic
   enough for regression coverage.
3. Define expected eigenvalue ordering, tolerances, and convergence assertions.
4. Identify any helper extraction that would improve readability without
   hiding proof values.
5. Define focused validation for the Ritz selection proof.
6. Write the Ritz selection proof design artifact.

### Deliverables
- repeated/clustered spectrum fixture plan
- expected ordering and tolerance table
- proof-value visibility rules
- focused validation list
- Day 5 implementation checklist

### Completion Criteria
- repeated and clustered spectrum cases are explicit and deterministic
- Ritz movement remains blocked until proof lands
- Day 5 has concrete tests to implement

---

## Day 5: Repeated and Clustered Ritz Selection Proof Implementation

**Title:** Ritz Proof
**Theme:** Add repeated/clustered spectrum proof for Ritz selection
**Time estimate:** 12 hours

### Tasks
1. Implement repeated-spectrum Ritz selection tests.
2. Implement clustered-spectrum Ritz selection tests.
3. Assert ordering, convergence counts, tolerances, and public result
   invariants.
4. Run focused eigensolver validation.
5. Record any instability and adjust fixtures without loosening claims.
6. Write the Ritz proof artifact.

### Deliverables
- repeated-spectrum Ritz tests
- clustered-spectrum Ritz tests
- focused validation output summary
- fixture stability notes
- Day 5 proof artifact

### Completion Criteria
- repeated and clustered Ritz selection behavior is directly tested
- proof values remain visible near assertions
- Ritz selection movement can be considered only from earned evidence

---

## Day 6: Ritz Vector Lifting and Publication Boundary Proof Design

**Title:** Vector Design
**Theme:** Design vector lifting and publication-boundary proof
**Time estimate:** 12 hours

### Tasks
1. Inspect current Ritz vector lifting and vector publication paths.
2. Identify public result fields and caller contracts that must remain stable.
3. Define tests that prove lifted vectors satisfy residual or orthogonality
   expectations.
4. Define publication-boundary checks for requested and converged vector counts.
5. Identify any shared vector-publication helper candidate and block movement
   until proof exists.
6. Write the vector lifting proof design artifact.

### Deliverables
- Ritz vector lifting path inventory
- publication-boundary invariant list
- helper-movement blocker list
- focused test checklist
- validation command list

### Completion Criteria
- vector lifting proof has concrete public assertions
- helper extraction remains blocked until Day 7 evidence exists
- Day 7 can implement from the artifact

---

## Day 7: Ritz Vector Lifting and Publication Boundary Proof Implementation

**Title:** Vector Proof
**Theme:** Add Ritz vector lifting and vector publication tests
**Time estimate:** 12 hours

### Tasks
1. Implement Ritz vector lifting proof tests.
2. Implement vector publication-boundary tests for requested and converged
   result counts.
3. Assert residual, normalization, and result-shape expectations where
   applicable.
4. Run focused eigensolver validation.
5. Record whether any vector-publication helper movement is now safe.
6. Write the vector proof artifact.

### Deliverables
- Ritz vector lifting tests
- vector publication-boundary tests
- focused validation output summary
- helper-movement safety assessment
- Day 7 proof artifact

### Completion Criteria
- lifted vector behavior is directly proven
- publication boundaries are covered before any helper extraction
- Day 8 can proceed to partial-result publication proof

---

## Day 8: Partial-Result Publication Exhaustion Proof

**Title:** Partial Results
**Theme:** Prove bounded publication after `m_cap` exhaustion
**Time estimate:** 12 hours

### Tasks
1. Identify current `m_cap` exhaustion paths and observable public results.
2. Design a deterministic matrix and budget that reaches bounded publication.
3. Implement partial-result publication tests.
4. Assert converged counts, result shapes, iteration counts, and non-overrun
   behavior.
5. Run focused eigensolver validation.
6. Write the partial-result publication artifact.

### Deliverables
- `m_cap` exhaustion fixture
- partial-result publication tests
- focused validation output summary
- result-publication invariant notes
- Day 8 proof artifact

### Completion Criteria
- bounded exits publish only documented converged state
- tests fail loudly on result-shape or count regressions
- shift-invert grow-m proof can proceed with publication behavior understood

---

## Day 9: Shift-Invert Grow-M Conversion Proof

**Title:** Shift-Invert
**Theme:** Prove shift-invert grow-m conversion before source ownership changes
**Time estimate:** 12 hours

### Tasks
1. Inspect shift-invert setup, grow-m conversion, and backend dispatch
   interactions.
2. Select deterministic shifted fixtures and sigma values.
3. Implement shift-invert grow-m conversion tests.
4. Assert backend, basis, convergence, and residual expectations through
   public results.
5. Run focused eigensolver validation.
6. Write the shift-invert proof artifact.

### Deliverables
- shift-invert grow-m fixture plan
- focused conversion tests
- validation output summary
- public-result invariant notes
- Day 9 proof artifact

### Completion Criteria
- shift-invert grow-m conversion is directly proven
- no source movement occurs before proof completion
- Day 10 can make an evidence-backed movement/no-move decision

---

## Day 10: Eigensolver Source Movement Decision or Narrow Movement

**Title:** Move Decision
**Theme:** Decide and optionally execute one narrow eigensolver source movement
**Time estimate:** 12 hours

### Tasks
1. Review Days 2-9 proof artifacts.
2. Decide whether one eigensolver owner has enough evidence for safe movement.
3. If safe, define exact files, build rules, source lists, and validation
   commands for one narrow private movement.
4. If not safe, publish a continued no-move decision and future proof
   requirements.
5. Execute only the narrow movement if the decision is low risk and fully
   bounded.
6. Write the movement or continued no-move artifact.

### Deliverables
- evidence-backed movement/no-move decision
- source-list and build-rule notes if movement occurs
- continued no-claim list if movement is deferred
- validation output summary for the chosen path
- Day 10 decision artifact

### Completion Criteria
- no broad eigensolver source split is claimed
- any movement is narrow, validated, and source-list coherent
- direct/iterative cleanup can proceed without depending on future movement

---

## Day 11: Direct/Iterative Exact-RHS Cleanup Design

**Title:** RHS Design
**Theme:** Plan solver-specific exact-RHS cleanup without broad abstraction
**Time estimate:** 12 hours

### Tasks
1. Inspect QR sequential RHS setup.
2. Inspect CG preconditioner-specific exact-RHS setup.
3. Inspect GMRES, BiCGSTAB, and MINRES exact-RHS setup.
4. Identify duplicated setup that can be cleaned without hiding solver-specific
   proof values.
5. Define a bounded cleanup order and focused validation commands.
6. Write the direct/iterative cleanup design artifact.

### Deliverables
- exact-RHS setup inventory
- solver-specific cleanup plan
- broad-abstraction blocker list
- focused validation command list
- Day 12 implementation checklist

### Completion Criteria
- all five solver areas have bounded cleanup targets
- no cross-solver oracle abstraction is attempted without evidence
- Day 12 can implement the batch safely

---

## Day 12: Direct/Iterative Exact-RHS Cleanup Implementation

**Title:** RHS Cleanup
**Theme:** Clean solver-specific exact-RHS setup while preserving proof values
**Time estimate:** 12 hours

### Tasks
1. Clean QR sequential RHS setup.
2. Clean CG preconditioner-specific exact-RHS setup.
3. Clean GMRES exact-RHS setup.
4. Clean BiCGSTAB exact-RHS setup.
5. Clean MINRES exact-RHS setup.
6. Run focused direct/iterative validation and write the cleanup artifact.

### Deliverables
- QR exact-RHS cleanup
- CG exact-RHS cleanup
- GMRES exact-RHS cleanup
- BiCGSTAB exact-RHS cleanup
- MINRES exact-RHS cleanup
- validation output summary

### Completion Criteria
- solver-specific proof values remain visible at call sites
- focused direct/iterative tests pass
- no broad direct/iterative proof abstraction is claimed

---

## Day 13: SVD Proof-Owner Cleanup Batch

**Title:** SVD Cleanup
**Theme:** Clean bounded SVD proof owners while preserving shape contracts
**Time estimate:** 12 hours

### Tasks
1. Split reconstruction helper movement by storage contract.
2. Split U/Vt orthogonality helper movement by economy/full leading-dimension
   convention.
3. Preserve Moore-Penrose product dimension proof while cleaning helpers.
4. Clean dense low-rank proof loops.
5. Clean sparse low-rank proof loops and condition-number proof logic.
6. Run focused SVD validation and write the SVD cleanup artifact.

### Deliverables
- storage-contract-aware reconstruction cleanup
- leading-dimension-aware U/Vt orthogonality cleanup
- Moore-Penrose dimension proof preservation notes
- dense and sparse low-rank cleanup
- condition-number proof cleanup
- focused validation output summary

### Completion Criteria
- SVD cleanup preserves all storage and dimension conventions
- focused SVD tests pass
- no broad SVD proof abstraction is claimed

---

## Day 14: Validation, Metrics, and Non-Claim Handoff

**Title:** Closeout
**Theme:** Validate the sprint and hand off remaining proof-owner truth
**Time estimate:** 12 hours

### Tasks
1. Run required focused and full quality checks for touched files.
2. Update proof-owner metrics for tests, helper ownership, source movement, and
   non-claims.
3. Verify no unsupported public API, install-header, helper-target,
   source-list, or reviewed CTest membership drift was introduced.
4. Capture remaining residual deferred debt in dependency order.
5. Write the Sprint 114 closeout and handoff artifact.
6. Update working notes with final validation evidence.

### Deliverables
- final validation evidence
- proof-owner metrics update
- non-claims checklist
- residual deferred debt handoff
- Sprint 114 closeout artifact
- completed working notes

### Completion Criteria
- required quality checks pass
- all Sprint 114 project-plan items have evidence or explicit residual status
- downstream package, adoption, and final Epic 10 closeout work receives a
  truthful handoff
