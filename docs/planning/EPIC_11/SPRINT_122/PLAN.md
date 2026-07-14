# Sprint 122 Plan: SVD/QR External Oracle Residual Follow-Through

**Sprint Duration:** 14 days
**Goal:** Convert Sprint 121's residual SVD, QR, partial-SVD,
helper-ownership, and solver-selection claim gates into explicit owners before
broader corpus, reporting, and adoption documentation work proceeds.

**Starting Point:** Sprint 122 begins from:
- Sprint 121 SVD/QR/rank fixture taxonomy
- Sprint 121 bounded SVD external-reference pilot and non-claim register
- Sprint 121 residual deferred debt
- Sprint 120 fixture/oracle patterns
- the Epic 11 project plan for Sprints 118-128

The sprint must:
- convert Sprint 121 residual oracle debt into explicit owners and duplicate
  fences
- decide whether additional SVD external fixtures should be added beyond
  `svd_rect_fullrank_6x4`
- design or explicitly defer a QR external dense-reference lane with precise
  fixture, tolerance, skip, and failure semantics
- design or explicitly defer partial-SVD external parity separately from
  full-SVD parity
- revisit minimum-norm helper ownership and Bidiagonal/Golub-Kahan helper
  boundaries without hiding specialized semantics
- define the evidence threshold for any future solver-selection wording that
  references broader external or support-level evidence
- close with validation, non-claims, and handoff to corpus/report and adoption
  sprints

**End State:** Sprint 122 leaves behind:
- a Sprint 121 residual oracle owner map
- an SVD external fixture expansion decision
- a QR external dense-reference lane design or explicit deferral
- a partial-SVD external parity design or explicit deferral
- a helper ownership boundary decision package
- a solver-selection claim gate
- validation evidence and residual handoff package

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 122 project-plan estimate.

---

## Day 1: Sprint Intake and Residual Source Map

**Title:** Residual Intake
**Theme:** Establish Sprint 122 scope, inputs, artifact skeleton, and residual sources
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 122 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Re-read the Sprint 121 retrospective residual deferred debt section.
3. Inventory Sprint 121 artifacts that describe SVD, QR, partial-SVD,
   low-rank, pseudoinverse, helper, and external-reference decisions.
4. Create Sprint 122 working notes and artifact directories.
5. Map each Sprint 122 project-plan item to day-level owners.
6. Write the sprint intake and residual source map artifact.

### Deliverables
- Sprint 122 working-notes baseline
- artifact directory structure
- residual source map
- day-level owner map
- validation and non-claim boundary notes

### Completion Criteria
- every Sprint 122 project-plan item has a day-level owner
- all Sprint 121 residual inputs are identified
- no completed Sprint 121 audit, taxonomy, helper extraction, fixture
  expansion, or SVD pilot work is silently reopened

---

## Day 2: Residual Dedupe and Owner Map

**Title:** Owner Map
**Theme:** Convert Sprint 121 residuals into explicit owners with duplicate fences
**Time estimate:** 12 hours

### Tasks
1. Extract all Sprint 121 residual oracle, helper, and claim-gate candidates.
2. Mark completed work that must not be duplicated in Sprint 122.
3. Assign each remaining residual to SVD external, QR external, partial-SVD
   external, helper ownership, solver-selection claim, or future-sprint owners.
4. Identify dependencies among the residuals.
5. Define proof gates required before implementation or public wording changes.
6. Write the residual oracle owner map.

### Deliverables
- deduplicated residual list
- duplicate-fence table
- owner and dependency map
- proof-gate checklist
- future-sprint handoff candidates

### Completion Criteria
- Item 1 is complete
- every residual is owned, deferred, or explicitly rejected as duplicate
- no residual depends on later work without a documented prerequisite

---

## Day 3: SVD External Fixture Decision Inventory

**Title:** SVD Fixture Inventory
**Theme:** Identify candidate SVD external fixtures and decision criteria
**Time estimate:** 12 hours

### Tasks
1. Review the existing `svd_rect_fullrank_6x4` external-reference pilot.
2. Inventory SVD fixture classes already covered by deterministic internal
   tests.
3. Identify candidate external SVD fixture classes that would add new evidence
   without duplicating Sprint 121 work.
4. Define fixture-size, tolerance, skip behavior, and failure-interpretation
   criteria for any additional SVD external fixture.
5. Record non-claims around LAPACK, SciPy, NumPy, and broad dense-library
   parity.
6. Write the SVD external fixture decision inventory artifact.

### Deliverables
- current SVD external pilot summary
- candidate SVD external fixture table
- duplicate and non-claim filter
- SVD fixture decision criteria
- Day 4 decision checklist

### Completion Criteria
- SVD candidates are evaluated against existing Sprint 121 coverage
- any proposed external fixture has explicit trust boundaries
- no broad SVD external-library parity claim is introduced

---

## Day 4: SVD External Fixture Decision

**Title:** SVD Fixture Decision
**Theme:** Decide whether to add, defer, or reject additional SVD external fixtures
**Time estimate:** 12 hours

### Tasks
1. Review the Day 3 candidate inventory.
2. Decide whether Sprint 122 should add one or more additional SVD external
   fixture designs.
3. If accepted, define fixture key, matrix shape, expected values, tolerance,
   skip behavior, and failure semantics.
4. If deferred or rejected, document the reason and future owner.
5. Update the SVD non-claim register.
6. Write the SVD external fixture expansion decision artifact.

### Deliverables
- accepted/deferred/rejected SVD fixture decision
- fixture key and tolerance proposal if accepted
- skip and failure semantics
- SVD non-claim update
- residual handoff for any deferred fixture

### Completion Criteria
- Item 2 is complete
- decision is reproducible from named evidence
- accepted work has enough detail for implementation or a clear deferral owner

---

## Day 5: QR External Lane Requirements

**Title:** QR Lane Requirements
**Theme:** Define the QR external dense-reference lane problem before design
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 121 QR and least-squares evidence artifacts.
2. Inventory QR solve, rank, least-squares, and minimum-norm fixture coverage.
3. Identify QR external dense-reference candidate scenarios.
4. Define fixture-size, tolerance, skip behavior, and failure-interpretation
   requirements.
5. Identify dependencies on existing QR helper ownership and solver semantics.
6. Write the QR external lane requirements artifact.

### Deliverables
- QR evidence and fixture inventory
- QR external candidate scenario table
- tolerance and skip requirements
- failure-interpretation notes
- Day 6 design checklist

### Completion Criteria
- QR lane design inputs are explicit before any implementation decision
- candidates do not duplicate completed Sprint 121 QR fixture expansion
- QR external parity remains a non-claim unless separately earned

---

## Day 6: QR External Lane Design or Deferral

**Title:** QR Lane Design
**Theme:** Design or explicitly defer a bounded QR external dense-reference lane
**Time estimate:** 12 hours

### Tasks
1. Select the highest-value QR external scenario or decide to defer.
2. Define reference invocation, fixture builder, expected outputs, tolerance,
   and skip behavior for the selected lane.
3. Define failure interpretation and diagnostic expectations.
4. Identify affected test, helper, script, Makefile, CMake, and CTest surfaces.
5. Document implementation risk and rollback path.
6. Write the QR external dense-reference lane design or deferral artifact.

### Deliverables
- QR external lane design or explicit deferral
- fixture and reference protocol
- affected-surface matrix
- diagnostics and failure semantics
- implementation or future-owner handoff

### Completion Criteria
- Item 3 is complete
- QR fixture size, tolerance, skip behavior, and failure semantics are explicit
- no implementation is implied without a validation and support boundary

---

## Day 7: Partial-SVD External Semantics Inventory

**Title:** Partial-SVD Semantics
**Theme:** Separate partial-SVD external comparison semantics from full-SVD parity
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 121 partial-SVD, low-rank, vector, and reconstruction
   artifacts.
2. Inventory existing partial-SVD internal oracle coverage.
3. Identify why singular-vector, subspace, ordering, convergence, and tolerance
   semantics differ from full-SVD external comparisons.
4. Define candidate partial-SVD external evidence classes.
5. Record risks around degenerate spectra, sign flips, basis choices, and
   convergence budgets.
6. Write the partial-SVD semantics inventory artifact.

### Deliverables
- partial-SVD evidence inventory
- semantic-difference table versus full SVD
- candidate external evidence classes
- vector/subspace risk notes
- Day 8 design checklist

### Completion Criteria
- partial-SVD external work is not treated as full-SVD parity reuse
- vector/subspace and convergence risks are explicit
- no partial-SVD external parity claim is introduced

---

## Day 8: Partial-SVD External Design or Deferral

**Title:** Partial-SVD Design
**Theme:** Design or explicitly defer partial-SVD external parity
**Time estimate:** 12 hours

### Tasks
1. Review Day 7 semantics and candidate evidence classes.
2. Decide whether a bounded partial-SVD external comparison is viable in this
   sprint.
3. If viable, define fixture, reference behavior, ordering rules, tolerance,
   convergence budget, vector/subspace handling, and diagnostics.
4. If deferred, document the missing prerequisites and future owner.
5. Update partial-SVD non-claims.
6. Write the partial-SVD external parity design or deferral artifact.

### Deliverables
- partial-SVD external comparison design or explicit deferral
- ordering, tolerance, convergence, and vector/subspace semantics
- diagnostic and failure interpretation notes
- partial-SVD non-claim update
- future-owner handoff if deferred

### Completion Criteria
- Item 4 is complete
- partial-SVD design does not depend on unresolved QR or SVD decisions
- every unsupported partial-SVD claim remains explicitly fenced

---

## Day 9: Minimum-Norm Helper Ownership Review

**Title:** Minimum-Norm Helpers
**Theme:** Revisit minimum-norm helper ownership without premature migration
**Time estimate:** 12 hours

### Tasks
1. Inventory current QR solve, pseudoinverse, and minimum-norm helper ownership.
2. Identify duplicated setup, fixture, assertion, and diagnostic patterns.
3. Decide whether helper migration is safe now or should wait for a future QR
   solve/minimum-norm consolidation owner.
4. Define migration boundaries if accepted.
5. Define explicit deferral rationale if not accepted.
6. Write the minimum-norm helper ownership review artifact.

### Deliverables
- minimum-norm helper ownership inventory
- duplicate and risk table
- migration or deferral decision
- affected-surface notes
- validation implications

### Completion Criteria
- minimum-norm helper ownership has a concrete decision
- no helper movement hides QR solve or pseudoinverse semantics
- accepted migration has a validation path or deferred work has an owner

---

## Day 10: Bidiagonal/Golub-Kahan Helper Boundary Review

**Title:** Bidiag Boundary
**Theme:** Keep specialized bidiagonal and Golub-Kahan semantics separate unless consolidation is proven safe
**Time estimate:** 12 hours

### Tasks
1. Inventory current Bidiagonal and Golub-Kahan helper and test ownership.
2. Identify specialized transpose, shape, reconstruction, and orthogonality
   semantics that differ from general SVD helpers.
3. Evaluate safe extraction or consolidation candidates.
4. Decide whether helper extraction proceeds, is limited, or is deferred.
5. Define validation requirements for any future movement.
6. Write the Bidiagonal/Golub-Kahan helper boundary artifact.

### Deliverables
- Bidiagonal/Golub-Kahan helper inventory
- specialized semantics table
- consolidation risk assessment
- extraction or deferral decision
- future validation requirements

### Completion Criteria
- Item 5 helper-boundary work is complete
- specialized transpose and reconstruction semantics are preserved
- no general SVD helper absorbs behavior without proof-owner clarity

---

## Day 11: Solver-Selection Claim Gate Inventory

**Title:** Claim Gate Inventory
**Theme:** Define the evidence needed before public solver-selection wording expands
**Time estimate:** 12 hours

### Tasks
1. Audit current solver-selection, README, tutorial, example, and maintainer
   wording related to external evidence and support levels.
2. Compare wording against Sprint 121 and Sprint 122 oracle decisions.
3. Identify public statements that must remain non-claims.
4. Define evidence categories required before broader external or support-level
   wording can be added.
5. Write the solver-selection claim gate inventory artifact.

### Deliverables
- public/support wording inventory
- evidence-to-wording matrix
- non-claim list
- required-evidence categories
- Day 12 claim-gate checklist

### Completion Criteria
- candidate public wording is tied to evidence requirements
- unsupported external, support-level, performance, platform, and API claims
  remain fenced
- adoption and corpus/report sprints receive clear prerequisites

---

## Day 12: Solver-Selection Claim Gate Decision

**Title:** Claim Gate Decision
**Theme:** Publish explicit thresholds for future solver-selection wording
**Time estimate:** 12 hours

### Tasks
1. Review Day 11 inventory and evidence categories.
2. Define claim gates for SVD external evidence, QR external evidence,
   partial-SVD external evidence, helper ownership, and support-level wording.
3. Decide which wording remains unchanged in Sprint 122.
4. Identify future adoption-surface handoff requirements.
5. Write the solver-selection claim gate decision artifact.

### Deliverables
- solver-selection claim gate
- unchanged wording rationale
- external evidence threshold table
- adoption-surface handoff requirements
- non-claim update

### Completion Criteria
- Item 6 is complete
- future public wording has explicit evidence thresholds
- no current public docs are expanded beyond validated Sprint 122 evidence

---

## Day 13: Validation Package and Handoff

**Title:** Validation Package
**Theme:** Validate touched artifacts and package residual decisions for downstream sprints
**Time estimate:** 11 hours

### Tasks
1. Inventory all files touched by Sprint 122 work.
2. Select validation commands based on touched surfaces.
3. Run required docs, script, focused, or C quality checks as applicable.
4. Capture results, skips, and environment notes.
5. Package SVD, QR, partial-SVD, helper, and claim-gate residual decisions for
   Sprints 123 and 127.
6. Write the validation and handoff artifact.

### Deliverables
- touched-surface matrix
- validation command summary
- pass/fail/skip notes
- downstream residual handoff
- closeout checklist for Day 14

### Completion Criteria
- validation matches touched surfaces
- any required failure is investigated before closeout
- corpus/report and adoption sprints have clear handoff inputs

---

## Day 14: Sprint Closeout and Retrospective Prep

**Title:** Sprint Closeout
**Theme:** Close Sprint 122 with decisions, non-claims, and retrospective-ready evidence
**Time estimate:** 11 hours

### Tasks
1. Review all Sprint 122 artifacts for consistency.
2. Confirm each project-plan item has a completion disposition.
3. Update working notes with final decisions, validation, and residuals.
4. Publish the final Sprint 122 non-claim register.
5. Identify any residual deferred debt and future owners.
6. Write the sprint closeout artifact.

### Deliverables
- final Sprint 122 artifact index
- completed-item disposition table
- non-claim register
- residual deferred debt list
- retrospective-ready evidence summary

### Completion Criteria
- Item 7 is complete
- all Sprint 122 deliverables are present or explicitly deferred
- residuals are dependency-ordered and assigned to future owners
- no unsupported public, external-parity, support-level, or state-of-the-art
  claim is introduced
