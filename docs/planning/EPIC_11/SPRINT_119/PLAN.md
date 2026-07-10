# Sprint 119 Plan: Eigensolver Source Boundary & Proof-Owner Follow-Through

**Sprint Duration:** 14 days
**Goal:** Convert the safest Epic 10 eigensolver residual movements into
validated source-boundary improvements without widening public claims.

**Starting Point:** Sprint 119 begins from:
- the Sprint 118 residual owner map
- the Sprint 118 product truth map and explicit non-claims
- the Sprint 118 hotspot owner handoff
- the Sprint 118 source-movement evidence template
- documented source-list, CMake parity, CTest count, and quality-gate
  expectations

The sprint must:
- re-rank eigensolver residual movement candidates before moving code
- define exact old/new files, internal headers, source-list updates, CMake
  updates, ownership contracts, and rollback plans
- move only the lowest-risk private eigensolver owner after focused consumer
  proof is defined
- move or explicitly defer `s20_select_indices` and
  `s20_lift_ritz_vectors` based on grow-m, thick-restart, and LOBPCG proof
- split or defer shift-invert setup/conversion only after proving LDLT
  lifecycle, operator selection, error propagation, and cleanup ownership
- validate source-list, Make/CMake parity, focused eigensolver behavior, CTest
  count, and required quality gates
- document what moved, what stayed, and why no broad eigensolver parity claim
  was created

**End State:** Sprint 119 leaves behind:
- an eigensolver source-boundary decision package
- validated source movement where safe
- source-list and CMake parity evidence
- focused eigensolver consumer proof
- explicit residuals and non-claims

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 119 project-plan estimate.

---

## Day 1: Sprint Intake and Evidence Setup

**Title:** Intake Setup
**Theme:** Establish Sprint 119 scope, inputs, artifact structure, and validation expectations
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 119 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Re-read Sprint 118 Day 6, Day 8, Day 10, Day 12, and Day 14 artifacts.
3. Create Sprint 119 working notes and artifact directories.
4. Map each Sprint 119 project-plan item to day-level owners.
5. Copy or reference the source-movement template fields required for this
   sprint.
6. Write the sprint intake artifact with scope boundaries and validation
   expectations.

### Deliverables
- Sprint 119 working-notes baseline
- artifact directory structure
- day-level owner map
- Sprint 118 input artifact inventory
- validation and non-claim boundary notes

### Completion Criteria
- every Sprint 119 project-plan item has a day-level owner
- required Sprint 118 handoff artifacts are identified
- no movement begins before feasibility, proof, and rollback expectations are recorded

---

## Day 2: Eigensolver Movement Candidate Inventory

**Title:** Candidate Inventory
**Theme:** Inventory all eigensolver residual movement candidates and their consumers
**Time estimate:** 12 hours

### Tasks
1. Inspect eigensolver source and internal helper files for private-owner
   boundaries.
2. Inventory `s20_select_indices`, `s20_lift_ritz_vectors`,
   shift-invert setup/conversion, and `lanczos_iterate_op`.
3. Map each candidate to grow-m Lanczos, thick-restart, LOBPCG, shift-invert,
   repeated-handle, and test consumers.
4. Record source-list, CMake, public-header, and internal-header touch points.
5. Write the movement candidate inventory artifact.

### Deliverables
- movement candidate list
- consumer map
- source-list and CMake touch-point notes
- public/private API impact notes
- Day 3 ranking checklist

### Completion Criteria
- Item 1 audit inputs are complete
- every movement candidate has named consumers
- no candidate is ranked before its consumer and build impacts are visible

---

## Day 3: Movement Feasibility Ranking

**Title:** Feasibility Ranking
**Theme:** Re-rank eigensolver residuals by proof risk, consumer breadth, and rollback cost
**Time estimate:** 12 hours

### Tasks
1. Rank private-owner movement candidates by behavior boundary clarity.
2. Rank `s20_select_indices` and `s20_lift_ritz_vectors` by grow-m,
   thick-restart, and LOBPCG dependency risk.
3. Rank shift-invert setup/conversion by LDLT lifecycle, operator selection,
   error propagation, and cleanup risk.
4. Rank `lanczos_iterate_op` by consumer breadth and compile-unit risk.
5. Identify the lowest-risk first movement batch and explicit defer candidates.
6. Write the movement feasibility artifact.

### Deliverables
- ranked movement feasibility table
- first movement batch recommendation
- move/defer recommendation list
- proof-risk notes
- rollback-risk notes

### Completion Criteria
- Item 1 is complete
- the first movement batch is evidence-ranked
- every high-risk candidate has a defer condition or proof prerequisite

---

## Day 4: Source Boundary Design

**Title:** Boundary Design
**Theme:** Define exact file, ownership, internal-header, and rollback design before movement
**Time estimate:** 12 hours

### Tasks
1. Define old/new files for the first movement batch.
2. Define internal header contracts and private API ownership.
3. Define source-list and CMake changes.
4. Define public API and public-claim impact.
5. Define rollback instructions and partial-move handling.
6. Write the source boundary design artifact.

### Deliverables
- exact old/new file plan
- internal header contract
- Makefile/source-list and CMake impact plan
- public API and claim-impact note
- rollback and defer plan

### Completion Criteria
- Item 2 design is complete for the first movement batch
- source-list and CMake impacts are explicit
- movement can proceed or be stopped from a documented design gate

---

## Day 5: Focused Consumer Proof Design

**Title:** Consumer Proof
**Theme:** Define focused tests and behavior invariants before moving code
**Time estimate:** 12 hours

### Tasks
1. Identify focused eigensolver tests for the first movement batch.
2. Define invariants for grow-m, thick-restart, LOBPCG, shift-invert, and
   repeated-handle consumers affected by the planned movement.
3. Define expected CTest count and focused rerun commands.
4. Fill the source-movement evidence template for the planned movement.
5. Write the focused consumer proof artifact.

### Deliverables
- focused consumer test list
- behavior invariant table
- expected CTest count note
- filled source-movement evidence draft
- Day 6 implementation checklist

### Completion Criteria
- movement has focused proof before code changes
- required quality gates are known
- no consumer path depends on implicit or undocumented behavior

---

## Day 6: First Movement Batch Implementation

**Title:** First Movement
**Theme:** Move the lowest-risk private eigensolver owner with build-system updates
**Time estimate:** 12 hours

### Tasks
1. Apply the first source movement batch from the Day 4-5 design.
2. Update internal headers and private declarations.
3. Update Makefile/source-list and CMake membership as needed.
4. Preserve public API and public docs unless the design explicitly requires
   bounded wording.
5. Run focused compile or source-list checks needed immediately after movement.
6. Record implementation notes.

### Deliverables
- first movement batch code change
- updated internal header/source-list/CMake surfaces as needed
- immediate compile/source-list notes
- implementation evidence note
- residual movement list

### Completion Criteria
- Item 3 implementation batch is complete or explicitly stopped
- build metadata matches moved files
- public API and claim boundaries are unchanged unless explicitly recorded

---

## Day 7: First Movement Batch Focused Validation

**Title:** Movement Proof
**Theme:** Validate the first movement batch against focused eigensolver consumers
**Time estimate:** 12 hours

### Tasks
1. Run focused eigensolver tests selected on Day 5.
2. Run source-list and CMake compile or registration checks as needed.
3. Confirm expected CTest count or record justified differences.
4. Run required quality checks for any `.c` or `.h` modifications.
5. Update the source-movement evidence artifact with observed results.
6. Fix or stop on any unclear failure.

### Deliverables
- focused consumer proof results
- source-list and CMake parity evidence
- CTest count evidence
- required quality-check output summary
- updated movement evidence artifact

### Completion Criteria
- Item 3 validation is complete
- required quality gates pass before continuing
- failures are fixed or treated as blockers

---

## Day 8: Selection and Lifting Proof Audit

**Title:** Selection Audit
**Theme:** Prove whether `s20_select_indices` and `s20_lift_ritz_vectors` can move safely
**Time estimate:** 12 hours

### Tasks
1. Inspect selection and lifting helper dependencies.
2. Map grow-m, thick-restart, and LOBPCG behavior affected by movement.
3. Define public-result invariants and partial-publication behavior.
4. Define compile-unit and consumer proof needed for movement.
5. Decide whether both helpers move together, separately, or defer.
6. Write the selection/lifting proof audit artifact.

### Deliverables
- selection/lifting dependency map
- grow-m/thick-restart/LOBPCG proof matrix
- move-together or split decision
- explicit defer conditions
- Day 9 implementation checklist

### Completion Criteria
- Item 4 proof audit is complete
- move/defer decision is evidence-backed
- public-result invariants are documented before any movement

---

## Day 9: Selection and Lifting Movement or Deferral

**Title:** Selection Move
**Theme:** Move or explicitly defer selection/lifting helpers based on Day 8 proof
**Time estimate:** 12 hours

### Tasks
1. Move `s20_select_indices` if Day 8 proof clears it, otherwise document
   deferral.
2. Move `s20_lift_ritz_vectors` if Day 8 proof clears it, otherwise document
   deferral.
3. Update internal headers, source-list, and CMake membership as needed.
4. Run immediate focused compile or source-list checks.
5. Record what moved, what deferred, and why.

### Deliverables
- selection/lifting movement or deferral decision
- code and build metadata changes where cleared
- focused compile/source-list notes
- residual helper movement list
- updated source-movement evidence

### Completion Criteria
- Item 4 implementation or deferral is complete
- movement does not proceed without proof
- explicit residuals exist for any deferred helper

---

## Day 10: Selection and Lifting Validation

**Title:** Selection Proof
**Theme:** Validate selection/lifting movement against eigensolver consumers
**Time estimate:** 12 hours

### Tasks
1. Run grow-m Lanczos focused tests.
2. Run thick-restart focused tests.
3. Run LOBPCG-adjacent focused tests if affected.
4. Run CMake/CTest count checks as needed.
5. Run required quality checks for `.c` or `.h` modifications.
6. Update selection/lifting movement evidence.

### Deliverables
- grow-m proof results
- thick-restart proof results
- LOBPCG-adjacent proof results
- CTest count and CMake evidence
- updated source-movement evidence

### Completion Criteria
- Item 4 validation is complete
- all required checks pass before shift-invert work begins
- no broad eigensolver parity claim is introduced

---

## Day 11: Shift-Invert Boundary Decision

**Title:** Shift-Invert Decision
**Theme:** Decide whether shift-invert setup/conversion can split or must defer
**Time estimate:** 12 hours

### Tasks
1. Inspect shift-invert setup and conversion ownership.
2. Prove LDLT lifecycle dependencies and selected backend behavior.
3. Prove operator selection, error propagation, and cleanup ownership needs.
4. Decide split or deferral with old/new file plan if cleared.
5. Record focused shift-invert tests and validation requirements.
6. Write the shift-invert boundary decision artifact.

### Deliverables
- shift-invert ownership map
- LDLT lifecycle dependency proof notes
- operator/error/cleanup proof notes
- split or defer decision
- validation checklist

### Completion Criteria
- Item 5 decision is complete
- shift-invert movement is not attempted without cleanup and error proof
- any deferral has an explicit reason and future owner

---

## Day 12: Shift-Invert Movement or Deferral Validation

**Title:** Shift-Invert Proof
**Theme:** Implement cleared shift-invert split or validate explicit deferral
**Time estimate:** 12 hours

### Tasks
1. Apply shift-invert split only if Day 11 cleared movement.
2. Update internal headers, source-list, and CMake membership as needed.
3. Run focused shift-invert tests and affected eigensolver tests.
4. Run CTest count checks and required quality gates.
5. If deferred, validate that no source/build surfaces changed and record the
   deferral proof.
6. Update movement evidence.

### Deliverables
- shift-invert movement or explicit deferral
- focused shift-invert proof results
- source-list/CMake/CTest count evidence
- required quality-check output summary
- updated residual list

### Completion Criteria
- Item 5 validation is complete
- movement or deferral is evidence-backed
- required checks pass before closeout work begins

---

## Day 13: Full Validation and Parity Package

**Title:** Validation Package
**Theme:** Run and package source-list, Make/CMake, focused eigensolver, CTest count, and quality evidence
**Time estimate:** 12 hours

### Tasks
1. Run source-list checks.
2. Run required Makefile quality checks for modified `.c` or `.h` files.
3. Run CMake configure/build and `ctest -N` count checks as needed.
4. Run full or focused CTest according to touched surfaces.
5. Capture command outputs, expected counts, and any skipped supplemental lanes.
6. Write the validation and parity package artifact.

### Deliverables
- source-list evidence
- Makefile quality evidence
- CMake parity evidence
- CTest count evidence
- focused eigensolver test evidence
- skipped-lane rationale

### Completion Criteria
- Item 6 is complete
- all required validation passes before closeout
- any mismatch is fixed or treated as a blocker

---

## Day 14: Closeout and Non-Claims

**Title:** Movement Closeout
**Theme:** Publish what moved, what stayed, residuals, validation, and non-claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Summarize every movement, deferral, validation result, and residual.
2. Update working notes and artifact index.
3. Document why no broad eigensolver, ARPACK, SciPy, LAPACK, or
   state-of-the-art parity claim was created.
4. Document Sprint 120 handoff risks and follow-up needs.
5. Write the Sprint 119 closeout artifact.

### Deliverables
- movement summary
- explicit residual and deferred-debt list
- non-claim register
- Sprint 120 handoff notes
- Sprint 119 closeout artifact

### Completion Criteria
- Item 7 is complete
- every moved or deferred candidate has evidence
- Sprint 120 receives clear prerequisites, validation outcomes, and non-claim boundaries
