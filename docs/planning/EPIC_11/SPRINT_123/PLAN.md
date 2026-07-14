# Sprint 123 Plan: Residual SVD/QR Oracle, Helper & Claim Evidence Follow-Through

**Sprint Duration:** 14 days
**Goal:** Promote Sprint 122's residual SVD, QR, partial-SVD, helper, and claim
debt into bounded implementation or explicit deferral packages before corpus,
report-index, performance, package, and adoption sprints consume the oracle
truth.

**Starting Point:** Sprint 123 begins from:
- Sprint 121 SVD/QR/rank fixture taxonomy
- Sprint 122 bounded SVD, QR, and partial-SVD external oracle lanes
- Sprint 122 residual deferred debt and non-claim register
- existing external-reference helper scripts and scenario tests
- the Epic 11 project plan for Sprints 118-129

The sprint must:
- decide whether to broaden the SVD external fixture matrix and either
  implement a bounded batch or defer it with proof gates
- add or explicitly defer QR external compatible, rank-deficient,
  underdetermined/minimum-norm, and Q/economy evidence
- expand or explicitly defer partial-SVD external semantics beyond top-k
  singular values
- revisit minimum-norm helper migration without hiding behavior-specific
  ownership
- revisit Bidiagonal/Golub-Kahan helper extraction only through a dedicated
  semantic owner
- refresh maintainer evidence tables with oracle lane ownership and trust
  boundaries
- refresh solver-selection wording only if the evidence supports user-facing
  claims

**End State:** Sprint 123 leaves behind:
- a broader SVD external fixture decision or bounded implementation
- a QR external behavior evidence decision or bounded implementation
- a partial-SVD external semantics decision or bounded implementation
- a minimum-norm helper migration decision package
- a Bidiagonal/Golub-Kahan helper extraction decision package
- a maintainer evidence-table refresh
- a solver-selection claim refresh or explicit no-update rationale

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 123 project-plan estimate.

---

## Day 1: Sprint Intake and Residual Proof Map

**Title:** Residual Proof Map
**Theme:** Establish Sprint 123 inputs, duplicate fences, and proof ownership
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 123 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Re-read the Sprint 122 retrospective residual deferred debt and non-claim
   register.
3. Inventory Sprint 121 and Sprint 122 artifacts that define SVD, QR,
   partial-SVD, helper, and claim evidence boundaries.
4. Create Sprint 123 working notes and artifact directories.
5. Map each Sprint 123 project-plan item to day-level proof owners.
6. Write the sprint intake and residual proof map artifact.

### Deliverables
- Sprint 123 working-notes baseline
- artifact directory structure
- residual proof-owner map
- duplicate-fence table
- validation and non-claim boundary notes

### Completion Criteria
- every Sprint 123 project-plan item has a day-level owner
- Sprint 122 completed work is not silently reopened
- downstream Sprint 124 corpus/report work has clear Sprint 123 inputs

---

## Day 2: SVD Fixture Taxonomy and Trust Model

**Title:** SVD Trust Model
**Theme:** Decide what broader SVD external evidence can safely mean
**Time estimate:** 12 hours

### Tasks
1. Inventory current SVD external fixtures and deterministic SVD fixture
   coverage.
2. Classify candidate fixture classes for rank, rectangularity, repeated
   spectra, pseudoinverse, low-rank, vector, and scale semantics.
3. Define the external reference trust model for any additional SVD fixture.
4. Define tolerance, skip, and failure-interpretation rules.
5. Fence unsupported LAPACK, NumPy, SciPy, and broad dense-library claims.
6. Write the SVD fixture taxonomy and trust-model artifact.

### Deliverables
- SVD fixture candidate table
- SVD trust-boundary matrix
- tolerance and skip policy
- broad-parity non-claim register
- Day 3 implementation/deferral decision inputs

### Completion Criteria
- Item 1 has explicit decision criteria
- every SVD fixture candidate has a trust-boundary rationale
- no external-library parity claim is introduced

---

## Day 3: SVD External Fixture Batch Decision

**Title:** SVD Batch Decision
**Theme:** Implement or explicitly defer the next bounded SVD external fixture batch
**Time estimate:** 12 hours

### Tasks
1. Review the Day 2 SVD fixture taxonomy and trust model.
2. Select the highest-value bounded SVD fixture batch or decide to defer.
3. If accepted, define fixture keys, matrix values, expected outputs,
   tolerances, skip behavior, and affected test owners.
4. If deferred, define promotion gates and the future owner.
5. Update the SVD oracle non-claim register.
6. Write the SVD external fixture batch decision artifact.

### Deliverables
- accepted/deferred SVD fixture batch decision
- fixture protocol and affected-surface matrix if accepted
- future-owner handoff if deferred
- updated SVD non-claims
- validation checklist for any accepted implementation

### Completion Criteria
- Item 1 is complete or explicitly deferred
- accepted SVD work is bounded and testable
- deferred SVD work has clear promotion gates

---

## Day 4: SVD External Fixture Implementation or Deferral Package

**Title:** SVD Fixture Package
**Theme:** Land accepted SVD fixture work or publish the complete deferral
**Time estimate:** 12 hours

### Tasks
1. Implement accepted SVD fixture additions or finish the explicit deferral
   package.
2. Update external-reference helper scripts only if the accepted design
   requires it.
3. Add focused SVD scenario tests only when the fixture semantics are bounded.
4. Preserve Makefile, CMake, CTest, package, ABI, platform, and public API
   non-claims unless explicitly touched.
5. Run focused SVD helper/script checks if code or script changes.
6. Write the SVD fixture implementation or deferral artifact.

### Deliverables
- bounded SVD fixture implementation or deferral artifact
- focused SVD validation evidence or no-code rationale
- updated affected-surface matrix
- SVD residual handoff
- non-claim confirmation

### Completion Criteria
- Day 3 decision is fully executed
- any code/script change has focused validation evidence
- no unsupported public SVD claim is added

---

## Day 5: QR Behavior Evidence Requirements

**Title:** QR Requirements
**Theme:** Define QR external behavior evidence before implementation
**Time estimate:** 12 hours

### Tasks
1. Inventory current QR external and deterministic QR evidence.
2. Separate compatible, rank-deficient, underdetermined/minimum-norm,
   Q-basis, and economy-mode evidence candidates.
3. Define behavior-specific fixture requirements and basis rules.
4. Define tolerance, skip, and failure-interpretation rules.
5. Preserve minimum-norm helper ownership boundaries.
6. Write the QR external behavior requirements artifact.

### Deliverables
- QR behavior candidate table
- basis and tolerance policy
- minimum-norm ownership notes
- QR non-claim register
- Day 6 decision checklist

### Completion Criteria
- QR candidates are behavior-specific
- basis-dependent evidence is not conflated with solve residual evidence
- QR external parity remains a non-claim unless separately earned

---

## Day 6: QR Compatible and Rank-Deficient Evidence Decision

**Title:** QR Rank Evidence
**Theme:** Decide compatible and rank-deficient QR external evidence
**Time estimate:** 12 hours

### Tasks
1. Review compatible and rank-deficient QR candidates from Day 5.
2. Decide whether to implement bounded external evidence or defer.
3. Define fixture keys, expected outputs, tolerances, and diagnostics for any
   accepted compatible or rank-deficient lane.
4. Identify affected QR solve, rank, helper, and script owners.
5. Define future promotion gates for deferred lanes.
6. Write the QR compatible/rank-deficient evidence decision artifact.

### Deliverables
- QR compatible evidence decision
- QR rank-deficient evidence decision
- fixture/reference protocol if accepted
- future-owner handoff if deferred
- focused validation plan

### Completion Criteria
- compatible and rank-deficient QR lanes are accepted or explicitly deferred
- basis/tolerance rules are visible
- no completed Sprint 121 or Sprint 122 QR work is duplicated

---

## Day 7: QR Minimum-Norm and Q/Economy Evidence Decision

**Title:** QR Basis Evidence
**Theme:** Decide underdetermined/minimum-norm and Q/economy external evidence
**Time estimate:** 12 hours

### Tasks
1. Review underdetermined/minimum-norm and Q/economy candidates from Day 5.
2. Decide whether each lane should be implemented or explicitly deferred.
3. Define minimum-norm, basis orientation, sign, and economy-shape semantics.
4. Identify affected QR helper and scenario owners.
5. Preserve QR/COLAMD/SVD-pseudoinverse/refinement/fallback/SuiteSparse
   ownership boundaries.
6. Write the QR minimum-norm and Q/economy decision artifact.

### Deliverables
- QR minimum-norm evidence decision
- Q/economy evidence decision
- sign/basis semantics
- helper ownership constraints
- residual handoff for deferred work

### Completion Criteria
- Item 2 is complete or explicitly deferred
- minimum-norm and Q/economy semantics are not hidden in generic helpers
- future implementation has behavior-specific proof gates

---

## Day 8: QR Evidence Implementation or Deferral Package

**Title:** QR Evidence Package
**Theme:** Land accepted QR evidence work or publish the complete deferral
**Time estimate:** 12 hours

### Tasks
1. Implement accepted QR external behavior lanes or finish deferral packages.
2. Update QR external-reference scripts only if accepted lanes require it.
3. Add focused QR scenario tests only for bounded accepted behavior.
4. Preserve CTest membership and platform/package claims unless intentionally
   changed.
5. Run focused QR helper/script checks if code or script changes.
6. Write the QR evidence implementation or deferral artifact.

### Deliverables
- bounded QR implementation or explicit deferral package
- focused QR validation evidence or no-code rationale
- affected-surface matrix
- QR residual handoff
- non-claim confirmation

### Completion Criteria
- Day 6 and Day 7 decisions are fully executed
- accepted QR lanes have diagnostics and validation
- unsupported broad QR external parity remains fenced

---

## Day 9: Partial-SVD External Semantics Design

**Title:** Partial-SVD Semantics
**Theme:** Define partial-SVD evidence beyond top-k singular values
**Time estimate:** 12 hours

### Tasks
1. Inventory current partial-SVD top-k, vector, subspace, low-rank, and
   convergence evidence.
2. Separate value, vector, subspace, repeated-spectrum, clustered-spectrum,
   rectangular, and rank-deficient semantics.
3. Define sign, subspace-angle, convergence-budget, tolerance, skip, and
   failure-interpretation rules.
4. Identify fixture candidates and duplicate fences.
5. Preserve broad partial-SVD parity and low-rank optimality non-claims.
6. Write the partial-SVD external semantics design artifact.

### Deliverables
- partial-SVD semantics matrix
- fixture candidate table
- sign/subspace/convergence policy
- duplicate and non-claim register
- Day 10 decision checklist

### Completion Criteria
- partial-SVD vector/subspace semantics are separate from value semantics
- convergence-budget evidence has explicit interpretation rules
- no low-rank global optimality claim is introduced

---

## Day 10: Partial-SVD Evidence Decision and Package

**Title:** Partial-SVD Evidence
**Theme:** Implement or explicitly defer bounded partial-SVD external semantics
**Time estimate:** 12 hours

### Tasks
1. Review the Day 9 semantics design.
2. Select accepted partial-SVD evidence lanes or defer them with proof gates.
3. Define fixture protocol, expected metrics, tolerances, and diagnostics for
   accepted lanes.
4. Implement accepted bounded work or publish deferral rationale.
5. Run focused partial-SVD helper/script checks if code or script changes.
6. Write the partial-SVD evidence decision and package artifact.

### Deliverables
- accepted/deferred partial-SVD evidence decision
- bounded implementation or deferral package
- focused validation evidence or no-code rationale
- future-owner handoff
- updated partial-SVD non-claims

### Completion Criteria
- Item 3 is complete or explicitly deferred
- any accepted lane has value/vector/subspace/convergence meaning stated
- broad partial-SVD external parity remains unsupported unless earned

---

## Day 11: Minimum-Norm Helper Migration Decision

**Title:** Minimum-Norm Helpers
**Theme:** Decide whether minimum-norm helper migration is safe
**Time estimate:** 12 hours

### Tasks
1. Inventory current QR, COLAMD, SVD-pseudoinverse, refinement, fallback, and
   SuiteSparse minimum-norm scenario ownership.
2. Identify duplicate helper opportunities and semantic risks.
3. Define behavior-specific helper names and tolerance inputs if migration is
   safe.
4. Implement a bounded migration or explicitly defer it.
5. Run focused tests if helper code changes.
6. Write the minimum-norm helper migration decision artifact.

### Deliverables
- minimum-norm ownership inventory
- migration or deferral decision
- helper naming and tolerance policy
- focused validation evidence or no-code rationale
- residual handoff

### Completion Criteria
- Item 4 is complete or explicitly deferred
- no minimum-norm scenario loses visible behavior ownership
- any helper movement has focused validation evidence

---

## Day 12: Bidiagonal/Golub-Kahan Helper Decision

**Title:** Bidiag/GK Helpers
**Theme:** Decide dedicated Bidiagonal/Golub-Kahan helper extraction boundaries
**Time estimate:** 12 hours

### Tasks
1. Inventory Bidiagonal/Golub-Kahan checks across SVD, bidiag, and partial-SVD
   tests.
2. Identify reusable fixture builders or measurement helpers that do not hide
   specialized semantics.
3. Define boundaries for wide-transpose, implicit Householder reconstruction,
   explicit `U`/`V` reconstruction, and bidiagonal QR iteration semantics.
4. Implement a bounded dedicated helper extraction or explicitly defer it.
5. Run focused SVD/bidiag tests if helper code changes.
6. Write the Bidiagonal/Golub-Kahan helper decision artifact.

### Deliverables
- Bidiagonal/Golub-Kahan ownership inventory
- extraction or deferral decision
- dedicated-helper boundary policy
- focused validation evidence or no-code rationale
- residual handoff

### Completion Criteria
- Item 5 is complete or explicitly deferred
- general SVD helpers do not absorb specialized Bidiagonal/GK semantics
- any extraction preserves reconstruction and iteration proof meaning

---

## Day 13: Maintainer Evidence Table Refresh

**Title:** Evidence Tables
**Theme:** Refresh maintainer evidence with oracle ownership and trust boundaries
**Time estimate:** 11 hours

### Tasks
1. Inventory maintainer-guide evidence tables and related support docs.
2. Add or update entries for Sprint 122 and Sprint 123 oracle lanes.
3. Record trust boundaries, validation commands, helper ownership, skip
   behavior, and non-claims.
4. Cross-check public docs for unsupported claim drift.
5. Run documentation hygiene and focused link/path checks.
6. Write the maintainer evidence refresh artifact.

### Deliverables
- maintainer evidence-table update or explicit no-update rationale
- trust-boundary and validation-command entries
- support-doc claim scan
- documentation hygiene evidence
- Sprint 123 residual queue draft

### Completion Criteria
- Item 6 is complete
- evidence entries match implemented or deferred Sprint 123 outcomes
- no public/support claim exceeds current proof

---

## Day 14: Solver-Selection Claim Gate and Closeout

**Title:** Claim Closeout
**Theme:** Refresh solver-selection claims only if evidence supports them
**Time estimate:** 11 hours

### Tasks
1. Review all Sprint 123 implementation and deferral outcomes.
2. Decide whether solver-selection wording can be refreshed or must remain
   unchanged.
3. Update solver-selection wording only for earned claims, or publish a
   no-update rationale.
4. Run final documentation hygiene and required focused checks.
5. Publish final non-claim register and dependency-ordered residual debt.
6. Write the Sprint 123 closeout artifact and retrospective inputs.

### Deliverables
- solver-selection claim refresh or no-update rationale
- final Sprint 123 non-claim register
- final validation summary
- dependency-ordered residual deferred debt
- retrospective-ready closeout package

### Completion Criteria
- Item 7 is complete
- all Sprint 123 items are complete or explicitly deferred
- every day remains at or below 12 hours
- total plan estimate remains 166 hours
- no unsupported public, support-level, package, platform, ABI, performance,
  or state-of-the-art claim is introduced
