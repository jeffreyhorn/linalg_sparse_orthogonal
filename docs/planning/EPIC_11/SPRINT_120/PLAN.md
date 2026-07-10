# Sprint 120 Plan: Direct/Iterative Oracle Architecture & Giant-Test Split

**Sprint Duration:** 14 days
**Goal:** Create a maintainable direct/iterative oracle architecture and reduce
giant test ownership in the highest-risk direct and iterative proof files.

**Starting Point:** Sprint 120 begins from:
- Sprint 118 evidence templates and product-truth guardrails
- Sprint 119 source-boundary validation lessons
- Sprint 119 proof-owner movement discipline and non-claim register
- current direct, iterative, and CSC test files with dense-reference,
  generated-RHS, progress-callback, lifecycle, and external-oracle proof blocks
- existing Make, CMake, source-list, and CTest count expectations

The sprint must:
- audit QR, CG, GMRES, BiCGSTAB, MINRES, LDLT, LU, and Cholesky oracle owners
- design shared direct/iterative fixture builders without hiding
  solver-specific tolerances or failure modes
- split selected direct proof blocks from `test_qr.c`, `test_ldlt.c`, or
  `test_ldlt_csc.c` only after proof and rollback expectations are recorded
- split selected iterative proof blocks from `test_iterative.c`,
  `test_bicgstab.c`, or `test_minres.c` only after ownership boundaries are
  clear
- add one bounded cross-solver oracle pilot for compatible generated-RHS or
  dense-reference comparison paths
- validate focused tests, source-list/CMake parity, CTest membership, and full
  quality gates whenever `.c` or `.h` files change
- document oracle interpretation, remaining residuals, and explicit non-claims

**End State:** Sprint 120 leaves behind:
- a direct/iterative oracle architecture artifact
- selected giant-test proof-owner reductions
- one bounded cross-solver oracle pilot
- focused and reviewed validation evidence
- a residual direct/iterative oracle queue and non-claim register

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 120 project-plan estimate.

---

## Day 1: Sprint Intake and Evidence Setup

**Title:** Oracle Intake
**Theme:** Establish Sprint 120 scope, inputs, artifact structure, and validation rules
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 120 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Re-read Sprint 118 evidence-template artifacts and Sprint 119 validation
   and closeout artifacts.
3. Create Sprint 120 working notes and artifact directories.
4. Map each Sprint 120 project-plan item to day-level owners.
5. Record validation requirements for documentation-only, test-only,
   source/header, build-system, and CMake/source-list changes.
6. Write the sprint intake artifact with scope boundaries and non-claim
   expectations.

### Deliverables
- Sprint 120 working-notes baseline
- artifact directory structure
- day-level owner map
- Sprint 118 and Sprint 119 input inventory
- validation and non-claim boundary notes

### Completion Criteria
- every Sprint 120 project-plan item has a day-level owner
- prior evidence templates and source-boundary lessons are identified
- no oracle split begins before audit, design, proof, and rollback expectations are recorded

---

## Day 2: Direct Oracle Ownership Audit

**Title:** Direct Audit
**Theme:** Inventory direct-solver generated-RHS, dense-reference, and lifecycle proof owners
**Time estimate:** 12 hours

### Tasks
1. Inspect `tests/test_qr.c`, `tests/test_ldlt.c`, and
   `tests/test_ldlt_csc.c` for generated-RHS and dense-reference proof blocks.
2. Identify Cholesky, LU, QR, LDLT, and CSC direct proof helpers already shared
   through local headers or static helpers.
3. Map fixture construction, expected residuals, tolerance choices, failure
   cases, lifecycle setup, and cleanup ownership.
4. Identify high-risk giant-test regions that could be split without changing
   behavior.
5. Write the direct oracle ownership audit artifact.

### Deliverables
- direct-solver oracle owner table
- giant-test hotspot inventory
- helper and fixture reuse map
- tolerance and failure-mode notes
- Day 4 shared-fixture design inputs

### Completion Criteria
- Item 1 direct audit inputs are complete
- every direct candidate has named proof owners and behavior boundaries
- no split candidate is selected before tolerance and failure-mode ownership is visible

---

## Day 3: Iterative Oracle Ownership Audit

**Title:** Iterative Audit
**Theme:** Inventory iterative generated-RHS, convergence, progress, and preconditioner proof owners
**Time estimate:** 12 hours

### Tasks
1. Inspect `tests/test_iterative.c`, `tests/test_bicgstab.c`, and
   `tests/test_minres.c` for generated-RHS, oracle, progress-callback, and
   preconditioner proof blocks.
2. Map CG, GMRES, BiCGSTAB, MINRES, block-solver, and callback proof owners.
3. Identify solver-specific convergence criteria, residual tolerances,
   iteration-count expectations, and failure paths that must stay visible.
4. Identify candidate proof blocks for focused split or helper ownership.
5. Write the iterative oracle ownership audit artifact.

### Deliverables
- iterative-solver oracle owner table
- progress and callback proof map
- convergence and tolerance notes
- giant-test hotspot inventory
- Day 4 shared-fixture design inputs

### Completion Criteria
- Item 1 iterative audit inputs are complete
- every iterative candidate has named proof owners and failure modes
- shared helper opportunities do not hide solver-specific convergence contracts

---

## Day 4: Shared Fixture Architecture Design

**Title:** Fixture Design
**Theme:** Design shared fixture builders while preserving solver-specific interpretation
**Time estimate:** 12 hours

### Tasks
1. Compare Day 2 and Day 3 oracle owner maps.
2. Define candidate shared fixture builders for generated RHS, dense reference,
   residual measurement, matrix construction, and cleanup.
3. Define what must remain solver-local: tolerances, convergence thresholds,
   expected failure classes, progress-callback semantics, and publication
   assertions.
4. Define file placement, helper-header boundaries, naming, and rollback
   instructions.
5. Write the shared direct/iterative fixture architecture artifact.

### Deliverables
- shared fixture architecture
- solver-local responsibility table
- helper placement plan
- rollback instructions
- Day 5 selection checklist

### Completion Criteria
- Item 2 shared fixture design is complete
- shared helpers have explicit non-goals
- direct and iterative split candidates can be evaluated against the same architecture

---

## Day 5: Split Candidate Ranking and Proof Plan

**Title:** Split Ranking
**Theme:** Rank direct and iterative split candidates by proof value, risk, and rollback cost
**Time estimate:** 12 hours

### Tasks
1. Rank direct proof-block split candidates from Day 2.
2. Rank iterative proof-block split candidates from Day 3.
3. Select one direct split batch and one iterative split batch, or explicitly
   defer either if proof risk is too high.
4. Define focused commands, expected CTest membership, source-list impact, and
   full-quality requirements for each selected batch.
5. Write the split ranking and proof plan artifact.

### Deliverables
- ranked split candidate table
- selected direct split batch
- selected iterative split batch
- focused validation command list
- rollback and defer criteria

### Completion Criteria
- Items 3 and 4 have evidence-ranked implementation candidates
- every selected candidate has focused tests and rollback instructions
- no implementation begins without an agreed validation lane

---

## Day 6: Direct Test Split Batch Design

**Title:** Direct Split Design
**Theme:** Prepare exact direct-test file boundaries and helper ownership before edits
**Time estimate:** 12 hours

### Tasks
1. Define exact direct test blocks to split from `test_qr.c`, `test_ldlt.c`,
   or `test_ldlt_csc.c`.
2. Define any new helper header/source or focused test file names.
3. Define Make, CMake, source-list, and CTest changes if new test owners are
   introduced.
4. Define focused direct validation commands and expected failure behavior.
5. Write the direct split implementation checklist.

### Deliverables
- direct split file plan
- helper/test ownership contract
- build and CTest impact notes
- direct focused validation checklist
- rollback checklist

### Completion Criteria
- Item 3 implementation can proceed from exact file boundaries
- any build-system impact is explicit
- direct-solver behavior preservation is measurable before and after the split

---

## Day 7: Direct Test Split Implementation

**Title:** Direct Split
**Theme:** Implement the selected direct proof-owner split and run focused proof
**Time estimate:** 12 hours

### Tasks
1. Apply the selected direct proof-owner split.
2. Keep solver-specific tolerances and failure assertions visible at the test
   owner boundary.
3. Update Make, CMake, source-list, and CTest metadata if required.
4. Run focused direct tests for touched files.
5. Write the direct split implementation artifact with diff and validation
   notes.

### Deliverables
- direct proof-owner split
- updated build metadata if needed
- focused direct validation evidence
- implementation artifact
- residual direct-split notes

### Completion Criteria
- selected direct split compiles and focused tests pass
- behavior-preserving movement is documented
- any new source/test owner is registered in all required build inventories

---

## Day 8: Direct Split Validation and Consolidation

**Title:** Direct Validation
**Theme:** Revalidate direct split behavior and consolidate residual direct oracle debt
**Time estimate:** 12 hours

### Tasks
1. Re-run focused direct tests from Day 7.
2. Run source-list and CMake membership checks if Day 7 changed build
   metadata.
3. Review direct diff for accidental tolerance, failure-mode, or public-claim
   changes.
4. Update the direct oracle residual queue.
5. Write the direct validation and consolidation artifact.

### Deliverables
- focused direct revalidation evidence
- source-list/CMake evidence when applicable
- direct residual queue
- no-claim and no-public-API notes
- Day 9 iterative readiness checklist

### Completion Criteria
- Item 3 is complete or explicitly deferred with evidence
- direct residuals are documented
- no unsupported direct-solver oracle claim is introduced

---

## Day 9: Iterative Test Split Batch Design

**Title:** Iterative Split Design
**Theme:** Prepare exact iterative-test file boundaries and helper ownership before edits
**Time estimate:** 12 hours

### Tasks
1. Define exact iterative proof blocks to split from `test_iterative.c`,
   `test_bicgstab.c`, or `test_minres.c`.
2. Define any new helper header/source or focused test file names.
3. Define Make, CMake, source-list, and CTest changes if new test owners are
   introduced.
4. Define focused iterative validation commands, convergence expectations, and
   progress-callback checks.
5. Write the iterative split implementation checklist.

### Deliverables
- iterative split file plan
- helper/test ownership contract
- build and CTest impact notes
- iterative focused validation checklist
- rollback checklist

### Completion Criteria
- Item 4 implementation can proceed from exact file boundaries
- convergence and progress-callback behavior remains solver-local and testable
- any build-system impact is explicit

---

## Day 10: Iterative Test Split Implementation

**Title:** Iterative Split
**Theme:** Implement the selected iterative proof-owner split and run focused proof
**Time estimate:** 12 hours

### Tasks
1. Apply the selected iterative proof-owner split.
2. Keep solver-specific convergence, tolerance, iteration, and callback
   assertions visible.
3. Update Make, CMake, source-list, and CTest metadata if required.
4. Run focused iterative tests for touched files.
5. Write the iterative split implementation artifact with diff and validation
   notes.

### Deliverables
- iterative proof-owner split
- updated build metadata if needed
- focused iterative validation evidence
- implementation artifact
- residual iterative-split notes

### Completion Criteria
- selected iterative split compiles and focused tests pass
- behavior-preserving movement is documented
- any new source/test owner is registered in all required build inventories

---

## Day 11: Cross-Solver Oracle Pilot Design

**Title:** Oracle Pilot Design
**Theme:** Design one bounded cross-solver oracle pilot without broad parity claims
**Time estimate:** 12 hours

### Tasks
1. Select one compatible generated-RHS or dense-reference comparison path from
   the direct and iterative audit results.
2. Define the pilot's accepted matrix fixtures, solver set, tolerances,
   residual interpretation, and failure handling.
3. Define where the pilot helper/test should live and whether it requires
   Make, CMake, source-list, or CTest metadata updates.
4. Define focused pilot validation commands and non-claim wording.
5. Write the cross-solver oracle pilot design artifact.

### Deliverables
- bounded oracle pilot design
- solver and fixture scope table
- tolerance and residual interpretation notes
- build/CTest impact notes
- non-claim boundaries

### Completion Criteria
- Item 5 has a bounded implementation plan
- the pilot cannot be mistaken for broad direct/iterative parity
- validation commands and rollback criteria are documented before edits

---

## Day 12: Cross-Solver Oracle Pilot Implementation

**Title:** Oracle Pilot
**Theme:** Implement the bounded cross-solver oracle pilot and run focused proof
**Time estimate:** 12 hours

### Tasks
1. Implement the selected bounded oracle pilot.
2. Preserve solver-specific interpretation in assertions and failure messages.
3. Update Make, CMake, source-list, and CTest metadata if required.
4. Run focused pilot tests and adjacent touched-surface tests.
5. Write the oracle pilot implementation artifact.

### Deliverables
- bounded cross-solver oracle pilot
- updated build metadata if needed
- focused pilot validation evidence
- implementation artifact
- residual pilot limitations

### Completion Criteria
- Item 5 implementation compiles and focused tests pass
- pilot scope and non-claims are visible next to evidence
- no broad direct/iterative oracle parity claim is introduced

---

## Day 13: Validation Package

**Title:** Validation Package
**Theme:** Run and package source-list, Make/CMake, focused test, CTest count, and quality evidence
**Time estimate:** 12 hours

### Tasks
1. Run source-list checks.
2. Run focused direct, iterative, and pilot tests for touched surfaces.
3. Run CMake configure/build and `ctest -N` count checks when build metadata or
   test owners changed.
4. Run required full quality checks if any `.c` or `.h` files changed.
5. Capture outputs, expected counts, skipped supplemental lanes, and any
   residual validation risk.
6. Write the Sprint 120 validation package artifact.

### Deliverables
- source-list evidence
- focused direct/iterative/pilot evidence
- CMake and CTest count evidence when applicable
- required full quality evidence when applicable
- skipped-lane rationale and residual validation risk

### Completion Criteria
- Item 6 is complete
- all required validation for touched surfaces passes before closeout
- skipped lanes have explicit, scope-based rationale

---

## Day 14: Closeout and Non-Claims

**Title:** Oracle Closeout
**Theme:** Publish what split, what stayed, residuals, validation, and non-claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Summarize every split, deferral, validation result, and residual.
2. Update working notes and artifact index.
3. Document why no broad direct/iterative parity, external-oracle, performance,
   or state-of-the-art claim was created.
4. Document Sprint 121 handoff risks and follow-up needs.
5. Write the Sprint 120 closeout artifact.

### Deliverables
- direct/iterative split summary
- explicit residual and deferred-debt list
- non-claim register
- Sprint 121 handoff notes
- Sprint 120 closeout artifact

### Completion Criteria
- Item 7 is complete
- every split or deferred candidate has evidence
- Sprint 121 receives clear prerequisites, validation outcomes, and non-claim boundaries
