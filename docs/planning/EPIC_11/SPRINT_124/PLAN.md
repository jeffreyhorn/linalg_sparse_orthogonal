# Sprint 124 Plan: Residual QR, Partial-SVD & Helper Oracle Follow-Through

**Sprint Duration:** 14 days
**Goal:** Convert Sprint 123's residual QR, partial-SVD, minimum-norm, and
Bidiagonal/Golub-Kahan deferred debt into bounded oracle decisions or explicit
future-owner packages before corpus, reporting, performance, package, and
adoption work consume the oracle truth.

**Starting Point:** Sprint 124 begins from:
- Sprint 121 SVD/QR/rank fixture taxonomy
- Sprint 122 and Sprint 123 bounded SVD, QR, and partial-SVD external oracle
  lanes
- Sprint 123 residual deferred debt, dependency ordering, and non-claim
  register
- existing QR, SVD, Bidiagonal, Golub-Kahan, and external-reference helper
  owners
- the Epic 11 project plan for Sprints 118-130

The sprint must:
- decide or implement rank-deficient QR external evidence only after
  rank-threshold, nullspace, pseudoinverse, tolerance, skip, and
  minimum-norm policies are explicit
- decide or implement QR minimum-norm external evidence under a
  behavior-specific owner
- decide or implement QR Q-basis and economy external evidence only after
  sign, orientation, projection, subspace, and economy-shape semantics are
  defined
- decide or implement partial-SVD vector/subspace evidence with
  sign-invariant residuals, projection metrics, tolerance rules, and failure
  interpretation
- decide or explicitly defer partial-SVD repeated-spectrum, clustered-spectrum,
  rank-deficient, convergence-budget, and low-rank optimality evidence without
  claiming broad partial-SVD parity
- revisit minimum-norm helper migration and Bidiagonal/Golub-Kahan extraction
  with behavior-specific helper names and dedicated ownership
- validate affected work, refresh maintainer evidence, and update
  solver-selection wording only if the evidence supports a new public claim

**End State:** Sprint 124 leaves behind:
- rank-deficient QR oracle decision or bounded implementation
- QR minimum-norm oracle decision or bounded implementation
- QR Q-basis/economy oracle decision package
- partial-SVD vector/subspace decision or bounded implementation
- partial-SVD repeated/clustered/rank-deficient/convergence/low-rank decision
  package
- minimum-norm and Bidiagonal/Golub-Kahan helper ownership follow-through
- validation, maintainer evidence, and solver-selection claim gate package

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 124 project-plan estimate.

---

## Day 1: Sprint Intake and Residual Dependency Map

**Title:** Residual Dependency Map
**Theme:** Establish Sprint 124 ownership, duplicate fences, and downstream
inputs
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 124 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Re-read the Sprint 123 retrospective residual deferred debt and non-claim
   register.
3. Inventory Sprint 121, Sprint 122, and Sprint 123 artifacts that define QR,
   partial-SVD, helper, and claim evidence boundaries.
4. Create Sprint 124 working notes and artifact directories.
5. Map each Sprint 124 project-plan item to day-level proof owners.
6. Write the residual dependency map and duplicate-fence artifact.

### Deliverables
- Sprint 124 working-notes baseline
- artifact directory structure
- residual dependency map
- duplicate-fence table
- validation and non-claim boundary notes

### Completion Criteria
- every Sprint 124 project-plan item has a day-level owner
- no completed Sprint 121-123 work is silently reopened
- downstream Sprint 125 corpus/report work has clear Sprint 124 inputs

---

## Day 2: Rank-Deficient QR Policy Design

**Title:** QR Rank Policy
**Theme:** Define rank-deficient QR oracle semantics before implementation
**Time estimate:** 12 hours

### Tasks
1. Inventory current QR rank-deficient, near-rank-deficient, and deterministic
   rank evidence.
2. Define rank-threshold, nullspace, pseudoinverse, tolerance, skip, and
   failure-interpretation policies.
3. Separate QR solve residual evidence from minimum-norm and basis-dependent
   evidence.
4. Identify affected QR, COLAMD, SVD-pseudoinverse, SuiteSparse, and
   external-reference owners.
5. Fence unsupported broad QR, dense-library, or SuiteSparse parity claims.
6. Write the rank-deficient QR policy artifact.

### Deliverables
- QR rank-deficient candidate table
- rank-threshold and nullspace policy
- pseudoinverse/tolerance/skip policy
- affected-owner map
- rank-deficient QR non-claim notes

### Completion Criteria
- Item 1 has explicit decision criteria
- rank-deficient QR evidence is not conflated with minimum-norm evidence
- every accepted candidate has a clear trust-boundary rationale

---

## Day 3: Rank-Deficient QR Decision or Bounded Batch

**Title:** QR Rank Decision
**Theme:** Implement or explicitly defer rank-deficient QR external evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 2 rank-deficient QR policy artifact.
2. Select the highest-value bounded rank-deficient QR evidence batch or decide
   to defer.
3. If accepted, define fixture keys, expected outputs, tolerances, diagnostics,
   and affected test owners.
4. If deferred, define promotion gates, future owner, and dependency blockers.
5. Update QR rank-deficient non-claims and validation expectations.
6. Write the rank-deficient QR decision artifact.

### Deliverables
- accepted/deferred rank-deficient QR decision
- fixture/reference protocol if accepted
- future-owner handoff if deferred
- validation checklist
- updated QR rank-deficient non-claim register

### Completion Criteria
- Item 1 is complete or explicitly deferred
- any accepted evidence is bounded and testable
- deferred work has clear promotion gates and owner

---

## Day 4: QR Minimum-Norm Behavior Contract

**Title:** Minimum-Norm Contract
**Theme:** Define QR minimum-norm evidence across solve, ordering, fallback,
and refinement paths
**Time estimate:** 12 hours

### Tasks
1. Inventory current QR minimum-norm, underdetermined, COLAMD, fallback,
   refinement, and SuiteSparse scenario coverage.
2. Define behavior-specific expected outputs and residual/norm comparison
   rules.
3. Define when SVD-pseudoinverse is an oracle, a fallback, or a non-claim.
4. Identify helper ownership that must remain scenario-local.
5. Define skip and tolerance behavior for optional SuiteSparse paths.
6. Write the QR minimum-norm behavior contract.

### Deliverables
- QR minimum-norm behavior matrix
- oracle/fallback/non-claim boundary table
- residual and norm comparison policy
- optional-backend skip policy
- helper ownership notes

### Completion Criteria
- Item 2 has behavior-specific acceptance criteria
- minimum-norm semantics are not hidden behind generic helper names
- optional backend behavior is explicitly fenced

---

## Day 5: QR Minimum-Norm Decision or Bounded Batch

**Title:** Minimum-Norm Decision
**Theme:** Implement or explicitly defer QR minimum-norm external evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 4 QR minimum-norm behavior contract.
2. Select a bounded minimum-norm evidence batch or decide to defer.
3. If accepted, define fixtures, reference values, tolerances, diagnostics, and
   scenario owners.
4. If deferred, write dependency blockers and promotion gates.
5. Update minimum-norm non-claims and affected docs notes.
6. Write the QR minimum-norm decision artifact.

### Deliverables
- accepted/deferred QR minimum-norm decision
- fixture/reference protocol if accepted
- future-owner handoff if deferred
- focused validation plan
- updated minimum-norm non-claim register

### Completion Criteria
- Item 2 is complete or explicitly deferred
- accepted evidence remains behavior-specific
- deferred evidence has clear future ownership

---

## Day 6: QR Q-Basis and Economy Semantics

**Title:** QR Basis Semantics
**Theme:** Define Q-basis, sign, projection, subspace, and economy-shape rules
**Time estimate:** 12 hours

### Tasks
1. Inventory current QR Q-basis, economy-mode, projection, and orthogonality
   evidence.
2. Define sign/orientation handling for basis-dependent comparisons.
3. Define projection/subspace metrics that avoid false vector-equality claims.
4. Define economy-shape expectations across compatible and rank-deficient
   cases.
5. Identify affected tests, helper scripts, and docs.
6. Write the QR Q-basis/economy semantics artifact.

### Deliverables
- Q-basis and economy candidate table
- sign/orientation policy
- projection and subspace metric policy
- economy-shape expectation table
- non-claim notes for basis-dependent QR evidence

### Completion Criteria
- Item 3 has explicit semantic gates
- basis equality is not claimed where only subspace equivalence is justified
- economy-mode expectations are visible before implementation

---

## Day 7: QR Q-Basis/Economy Decision Package

**Title:** QR Basis Decision
**Theme:** Implement or explicitly defer Q-basis and economy external evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 6 Q-basis/economy semantics artifact.
2. Select a bounded evidence batch or decide to defer.
3. If accepted, define fixture values, metrics, tolerances, diagnostics, and
   affected owners.
4. If deferred, define blockers, promotion gates, and future owner.
5. Update QR Q-basis/economy non-claims.
6. Write the QR Q-basis/economy decision package.

### Deliverables
- accepted/deferred Q-basis/economy decision
- metric and tolerance protocol if accepted
- future-owner handoff if deferred
- focused validation plan
- updated basis/economy non-claim register

### Completion Criteria
- Item 3 is complete or explicitly deferred
- sign and subspace semantics are represented in the decision
- accepted work cannot create unsupported vector-orientation claims

---

## Day 8: Partial-SVD Vector and Subspace Semantics

**Title:** Partial-SVD Semantics
**Theme:** Define vector, subspace, residual, tolerance, and failure
interpretation rules
**Time estimate:** 12 hours

### Tasks
1. Inventory current partial-SVD singular-value, vector, residual, and
   convergence evidence.
2. Define sign-invariant vector comparison rules.
3. Define projection and subspace metrics for singular-vector evidence.
4. Define residual, tolerance, skip, and failure-interpretation policies.
5. Separate top-k singular-value evidence from vector/subspace evidence.
6. Write the partial-SVD vector/subspace semantics artifact.

### Deliverables
- partial-SVD vector/subspace candidate table
- sign-invariant comparison policy
- projection/subspace metric policy
- residual and tolerance policy
- partial-SVD broad-parity non-claim notes

### Completion Criteria
- Item 4 has explicit semantic gates
- vector/subspace evidence is not reduced to singular-value agreement
- failure interpretation is visible before implementation

---

## Day 9: Partial-SVD Vector/Subspace Decision or Bounded Batch

**Title:** Partial-SVD Vector Decision
**Theme:** Implement or explicitly defer partial-SVD vector/subspace evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 8 partial-SVD vector/subspace semantics artifact.
2. Select a bounded vector/subspace evidence batch or decide to defer.
3. If accepted, define fixtures, expected metrics, tolerances, diagnostics, and
   affected test/script owners.
4. If deferred, define promotion gates, blockers, and future owner.
5. Update partial-SVD vector/subspace non-claims.
6. Write the partial-SVD vector/subspace decision artifact.

### Deliverables
- accepted/deferred partial-SVD vector/subspace decision
- metric/reference protocol if accepted
- future-owner handoff if deferred
- focused validation plan
- updated partial-SVD vector/subspace non-claim register

### Completion Criteria
- Item 4 is complete or explicitly deferred
- accepted evidence uses sign-invariant and subspace-aware metrics
- deferred work has clear promotion gates

---

## Day 10: Partial-SVD Residual Scenario Matrix

**Title:** Partial-SVD Residual Matrix
**Theme:** Decide repeated, clustered, rank-deficient, convergence, and
low-rank optimality evidence scope
**Time estimate:** 12 hours

### Tasks
1. Inventory candidate repeated-spectrum, clustered-spectrum, rank-deficient,
   convergence-budget, and low-rank optimality scenarios.
2. Define scenario-specific trust boundaries and expected diagnostics.
3. Decide which scenarios can be implemented now and which must be deferred.
4. Define tolerances, skip rules, and failure interpretation for each accepted
   scenario.
5. Preserve the broad partial-SVD parity non-claim.
6. Write the partial-SVD residual scenario matrix.

### Deliverables
- partial-SVD residual scenario matrix
- accepted/deferred scenario decisions
- tolerance and skip policy
- future-owner handoff for deferred scenarios
- updated broad partial-SVD non-claim notes

### Completion Criteria
- Item 5 has explicit scenario-level decisions
- no repeated/clustered/rank-deficient behavior is claimed without evidence
- low-rank optimality evidence has a bounded interpretation

---

## Day 11: Partial-SVD Residual Implementation or Deferral Package

**Title:** Partial-SVD Residual Package
**Theme:** Land accepted residual scenario work or publish explicit deferrals
**Time estimate:** 12 hours

### Tasks
1. Implement accepted partial-SVD residual scenario evidence or complete the
   explicit deferral package.
2. Update external-reference helpers only where the scenario matrix requires
   it.
3. Add focused tests only for accepted and bounded semantics.
4. Preserve package, ABI, platform, performance, and broad solver-family
   non-claims.
5. Run focused partial-SVD checks if code or scripts changed.
6. Write the partial-SVD residual implementation or deferral artifact.

### Deliverables
- bounded partial-SVD residual implementation or deferral package
- focused validation evidence or no-code rationale
- updated affected-surface matrix
- residual scenario handoff
- non-claim confirmation

### Completion Criteria
- Item 5 is complete or explicitly deferred
- any changed test/script path has focused validation evidence
- unsupported partial-SVD parity wording is not introduced

---

## Day 12: Helper Ownership Follow-Through

**Title:** Helper Ownership
**Theme:** Revisit minimum-norm and Bidiagonal/Golub-Kahan helper boundaries
**Time estimate:** 12 hours

### Tasks
1. Inventory minimum-norm helper candidates and Bidiagonal/Golub-Kahan helper
   extraction candidates.
2. Decide which helpers can move without hiding scenario-local assertions.
3. Rename or specify behavior-specific helper names where needed.
4. Implement safe helper movement or write explicit deferral packages.
5. Update maintainer evidence for any ownership changes.
6. Write the helper ownership follow-through artifact.

### Deliverables
- minimum-norm helper movement/deferral decision
- Bidiagonal/Golub-Kahan extraction movement/deferral decision
- helper naming and ownership table
- maintainer evidence updates
- focused validation checklist

### Completion Criteria
- Item 6 is complete or explicitly deferred
- helpers preserve behavior-specific meaning
- any ownership movement has source/test/list evidence

---

## Day 13: Validation, Maintainer Evidence, and Claim Gate

**Title:** Evidence and Claim Gate
**Theme:** Validate affected work and decide whether docs claims can change
**Time estimate:** 11 hours

### Tasks
1. Run focused checks for all accepted Sprint 124 implementation work.
2. Run required quality gates if any `.c` or `.h` files changed.
3. Update maintainer evidence tables for oracle lanes and helper ownership.
4. Review solver-selection and public docs for any claim changes.
5. Update docs only where evidence supports a user-facing claim.
6. Write the validation and claim-gate artifact.

### Deliverables
- focused validation evidence
- required quality-gate evidence or docs-only rationale
- maintainer evidence-table refresh
- solver-selection update or no-update rationale
- claim-boundary confirmation

### Completion Criteria
- Item 7 validation obligations are satisfied or clearly blocked
- public wording changes are evidence-backed
- no unsupported QR, partial-SVD, helper, or external-oracle claim is added

---

## Day 14: Sprint Closeout and Downstream Handoff

**Title:** Closeout Handoff
**Theme:** Publish final Sprint 124 decisions, residuals, and Sprint 125 inputs
**Time estimate:** 11 hours

### Tasks
1. Review all Sprint 124 artifacts for consistency with the project plan.
2. Consolidate accepted implementations, explicit deferrals, residuals, and
   non-claims.
3. Confirm every Sprint 124 deliverable has evidence or a named future owner.
4. Prepare downstream handoff notes for Sprint 125 corpus/report-index work.
5. Update working notes with validation results and unresolved risk.
6. Write the Sprint 124 closeout summary.

### Deliverables
- Sprint 124 closeout summary
- consolidated residual and future-owner queue
- final non-claim register
- Sprint 125 corpus/report handoff notes
- validation summary and unresolved-risk notes

### Completion Criteria
- all seven Sprint 124 project-plan items are complete or explicitly deferred
- every deferred item has owner, dependency, and promotion gate
- Sprint 125 has a stable oracle-truth input package
