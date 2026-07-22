# Sprint 129 Plan: QR Q-Basis, Economy & Helper Ownership Follow-Through

**Sprint Duration:** 14 days
**Goal:** Resolve the remaining Sprint 124 QR Q-basis/economy and helper
ownership debt in dependency order, preserving basis and helper semantics before
the corpus/index architecture consumes them. Sprint 129 does not continue
grinding Sprint 128 residual QR debt unless a Q/economy/helper item has a
distinct behavior-specific claim and satisfies the Sprint 128 promotion gate.

**Starting Point:** Sprint 129 begins from:
- Sprint 125-128 rank-deficient QR metric, tolerance, corpus, optional-large,
  minimum-norm, subspace, threshold, and claim-gate artifacts
- Sprint 124 Q-basis/economy semantic design and helper decision artifacts
- Sprint 128 end-of-epic deferred QR residual queue and no-reopen policy
- current `tests/test_qr.c`, `tests/test_qr_solve.c`,
  `tests/test_colamd.c`, `tests/test_svd.c`,
  `tests/qr_external_dense_reference.py`, QR helper files, and
  Bidiagonal/Golub-Kahan owners

The sprint must:
- refresh raw Q-column, sign, orientation, projection, economy-shape, skip, and
  corpus policies before extending QR evidence
- accept raw QR Q-column evidence only where fixture-local orientation and
  tolerance rules make equality meaningful
- add or explicitly defer rank-deficient Q/nullspace evidence using the
  Sprint 125-128 nullity and projection/subspace metric policies
- add or explicitly defer wide economy and sparse-mode Q/economy evidence with
  explicit shape, projection, and sparse-output interpretation
- add or explicitly defer SuiteSparse Q/economy evidence only after corpus
  availability, skip behavior, diagnostics, and support-tier wording are
  bounded
- revisit minimum-norm helper movement only with behavior-specific names,
  visible owner call-site tolerances, focused QR solve/COLAMD/SVD validation,
  and required full quality validation
- extract or explicitly defer Bidiagonal/Golub-Kahan helpers into a dedicated
  owner that preserves transpose, reconstruction, explicit `U`/`V`, wide skip,
  and QR-iteration semantics
- avoid reopening Sprint 128 residual QR debt unless it directly supports a
  Sprint 129 Q-basis, economy, or helper ownership claim

**End State:** Sprint 129 leaves behind:
- Q-basis/economy evidence policy refresh
- raw Q-column decision or bounded implementation
- rank-deficient Q/nullspace decision or bounded implementation
- wide economy and sparse-mode decision package
- SuiteSparse Q/economy decision package
- minimum-norm helper movement package
- Bidiagonal/Golub-Kahan helper extraction package
- no-reopen decision for Sprint 128 residual QR debt except where directly
  required by Q-basis, economy, or helper ownership

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 129 project-plan estimate.

---

## Day 1: Sprint Intake and No-Reopen Boundary

**Title:** Intake Boundary
**Theme:** Establish Sprint 129 scope, owners, duplicate fences, and the
Sprint 128 residual no-reopen rule
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 129 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Review Sprint 124 Q-basis/economy and helper decision artifacts.
3. Review Sprint 125-128 QR residual, nullspace/subspace, threshold,
   minimum-norm, corpus, and helper claim gates.
4. Create Sprint 129 working notes and artifact directory.
5. Map each Sprint 129 project-plan item to day-level owners, likely touched
   files, and validation requirements.
6. Write the Sprint 129 intake, duplicate-fence, and no-reopen boundary
   artifact.

### Deliverables
- Sprint 129 working-notes baseline
- artifact directory structure
- day-level owner map
- Sprint 128 residual no-reopen decision
- validation and claim-boundary notes

### Completion Criteria
- every Sprint 129 project-plan item has a day-level owner
- Sprint 128 residual QR debt is not silently reopened
- Q-basis, economy, and helper ownership dependencies are explicit before new
  evidence is accepted

---

## Day 2: Q-Basis and Economy Policy Refresh

**Title:** Basis Policy
**Theme:** Refresh raw Q, projection, sign/orientation, economy-shape, and
corpus rules before accepting evidence
**Time estimate:** 12 hours

### Tasks
1. Review current QR Q/economy evidence, including
   `qr_economy_projector_5x3`, economy tests, sparse-mode tests, and
   nullspace/subspace projector fixtures.
2. Inventory raw Q-column, projection, economy, sparse-mode, and SuiteSparse
   candidate surfaces.
3. Define when raw Q equality is meaningful versus when projection/subspace
   metrics are required.
4. Define economy output-shape, wide-shape, sparse-mode, and SuiteSparse
   interpretation rules.
5. Define skip, support-tier, diagnostics, and failure-interpretation
   requirements for corpus-backed Q/economy evidence.
6. Write the Q-basis/economy evidence policy refresh.

### Deliverables
- raw Q-column candidate table
- projection/subspace metric policy
- economy and sparse-mode output-shape policy
- SuiteSparse Q/economy support-tier policy
- non-claim boundary update

### Completion Criteria
- no Q/economy candidate can proceed without a metric and tolerance policy
- raw basis equality is allowed only for fixture-local deterministic cases
- economy and sparse-mode evidence cannot imply broad QR parity

---

## Day 3: Raw Q-Column Evidence Decision

**Title:** Raw Q Decision
**Theme:** Implement or explicitly defer raw QR Q-column evidence under the
Day 2 policy
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 2 raw Q policy to small deterministic QR fixture candidates.
2. Select at most one raw Q-column candidate that has stable sign/orientation,
   fixture-local tolerance, and distinct trust value.
3. Implement the accepted raw Q-column evidence, or document explicit deferral
   with future promotion gates.
4. Keep raw Q evidence separate from projection, economy, sparse-mode,
   nullspace, and residual claims.
5. Run focused helper/test checks for touched QR evidence.
6. Record diagnostics, validation, and non-claims.

### Deliverables
- raw Q-column implementation or explicit deferral
- fixture key, tolerance, and diagnostic notes
- focused validation results
- raw Q non-claim update

### Completion Criteria
- raw Q evidence either passes with fixture-local orientation rules or is
  explicitly deferred
- no raw Q result is described as unique-basis or broad Q parity evidence
- touched code/scripts have appropriate focused validation

---

## Day 4: Rank-Deficient Q/Nullspace Policy Gate

**Title:** Rank-Def Q Gate
**Theme:** Decide how Q-basis evidence may interact with existing
rank-deficient nullspace/subspace evidence
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 125-128 rank-deficient nullspace/subspace projector and
   threshold gates.
2. Identify rank-deficient Q/nullspace candidates that may add Q-specific
   trust without duplicating Sprint 128 residual/subspace debt.
3. Define whether each candidate needs raw Q equality, projection metrics,
   two-way projection residuals, or explicit deferral.
4. Pin rank, nullity, tolerance, diagnostics, and output-shape requirements.
5. Fence minimum-norm, residual-only, SuiteSparse, sparse-mode, and economy
   claims.
6. Write the rank-deficient Q/nullspace gate.

### Deliverables
- rank-deficient Q/nullspace candidate table
- metric and tolerance decision
- duplicate fence against Sprint 128 subspace work
- Day 5 implementation or deferral criteria

### Completion Criteria
- no rank-deficient Q/nullspace candidate duplicates completed projector work
- Q-specific proof value is explicit before implementation
- Sprint 128 residual debt remains in the end-of-epic queue unless directly
  required

---

## Day 5: Rank-Deficient Q/Nullspace Evidence

**Title:** Rank-Def Q Evidence
**Theme:** Implement or explicitly defer one bounded rank-deficient
Q/nullspace evidence lane
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 4 rank-deficient Q/nullspace gate to candidate fixtures.
2. Implement one accepted bounded evidence lane only if fixture key, metric,
   tolerance, diagnostics, and proof boundary are pinned.
3. Otherwise write an explicit deferral package with future-owner gates.
4. Update maintainer evidence only if a new accepted evidence lane changes the
   bounded QR evidence table.
5. Run focused QR/helper validation for touched files.
6. Record non-claims for raw basis, unique basis, residual, minimum-norm,
   sparse-mode, economy, and SuiteSparse behavior.

### Deliverables
- rank-deficient Q/nullspace implementation or explicit deferral
- diagnostics and validation package
- maintainer-guide update or no-update rationale
- non-claim register update

### Completion Criteria
- accepted evidence is behavior-specific and non-duplicative, or the lane is
  explicitly deferred
- validation is complete for every touched code or helper file
- no broad rank-deficient QR, nullspace, or Q-basis parity claim is added

---

## Day 6: Wide Economy and Sparse-Mode Policy

**Title:** Economy Policy
**Theme:** Define wide economy and sparse-mode Q/economy output semantics
before implementation
**Time estimate:** 12 hours

### Tasks
1. Review current economy, wide, sparse-mode, and SuiteSparse QR tests.
2. Identify wide economy and sparse-mode Q/economy candidates.
3. Define output-shape, projection, reconstruction, orthogonality, and sparse
   interpretation metrics.
4. Define what wide/economy evidence cannot imply about minimum-norm,
   residual-only behavior, raw basis, or unique basis.
5. Establish Day 7 implementation and deferral criteria.
6. Write the wide economy and sparse-mode policy artifact.

### Deliverables
- wide economy candidate table
- sparse-mode Q/economy candidate table
- output-shape and projection metric policy
- Day 7 acceptance checklist

### Completion Criteria
- wide economy and sparse-mode semantics are pinned before implementation
- no candidate can imply minimum-norm or residual-only behavior
- accepted metrics are compatible with existing QR/economy APIs

---

## Day 7: Wide Economy and Sparse-Mode Evidence

**Title:** Economy Evidence
**Theme:** Implement or explicitly defer bounded wide economy and sparse-mode
Q/economy evidence
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 6 policy to wide economy and sparse-mode candidates.
2. Implement one accepted bounded evidence lane if metric, shape, tolerance,
   diagnostics, and owner are pinned.
3. Otherwise document explicit deferral with future promotion criteria.
4. Keep sparse-mode, economy, raw Q, residual, and minimum-norm claims
   separate.
5. Run focused QR/economy validation for touched code or helper scripts.
6. Update maintainer evidence or record a no-update rationale.

### Deliverables
- wide economy or sparse-mode implementation or explicit deferral
- focused validation results
- maintainer evidence update or no-update rationale
- economy/sparse-mode non-claim update

### Completion Criteria
- accepted evidence has stable shape, metric, tolerance, and diagnostics
- sparse-mode evidence does not imply broad sparse QR parity
- touched files have focused validation and full quality gate when required

---

## Day 8: SuiteSparse Q/Economy Gate

**Title:** SuiteSparse Q Gate
**Theme:** Decide whether SuiteSparse Q/economy evidence has enough metadata
to proceed
**Time estimate:** 12 hours

### Tasks
1. Inventory checked-in and optional SuiteSparse QR controls relevant to
   Q/economy behavior.
2. Review corpus availability, support tier, skip behavior, diagnostics, and
   runtime expectations.
3. Identify SuiteSparse Q/economy candidates and required metrics.
4. Reject product-observed values as expected values unless independent
   metadata exists.
5. Define Day 9 promotion or explicit deferral criteria.
6. Write the SuiteSparse Q/economy gate.

### Deliverables
- SuiteSparse Q/economy corpus inventory
- candidate table with metadata status
- support-tier and skip-behavior policy
- Day 9 acceptance checklist

### Completion Criteria
- SuiteSparse evidence cannot proceed without support-tier and diagnostic
  metadata
- product output is not treated as an independent oracle
- runtime and optional-data expectations are explicit

---

## Day 9: SuiteSparse Q/Economy Evidence Decision

**Title:** SuiteSparse Q Decision
**Theme:** Implement or explicitly defer SuiteSparse Q/economy evidence
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 8 gate to SuiteSparse Q/economy candidates.
2. Implement one accepted SuiteSparse Q/economy lane only if corpus metadata,
   metric, support tier, skip behavior, diagnostics, and validation are
   complete.
3. Otherwise record explicit deferral with blocker and future owner.
4. Run focused SuiteSparse QR/economy diagnostics where useful as controls.
5. Preserve SuiteSparse corpus, optional-data, platform, and performance
   non-claims.
6. Update maintainer evidence or record a no-update rationale.

### Deliverables
- SuiteSparse Q/economy implementation or explicit deferral
- focused control diagnostics
- support-tier and skip-behavior notes
- non-claim update

### Completion Criteria
- SuiteSparse Q/economy work is either bounded and validated or explicitly
  deferred
- no broad SuiteSparse corpus or platform claim is introduced
- every deferred item has blocker and future-owner notes

---

## Day 10: Minimum-Norm Helper Ownership Gate

**Title:** Min-Norm Helpers
**Theme:** Decide whether minimum-norm helper movement is safe and
behavior-specific
**Time estimate:** 12 hours

### Tasks
1. Review current minimum-norm owners in QR solve, COLAMD, SVD-pseudoinverse,
   fallback, refinement, zero-row, and SuiteSparse smoke tests.
2. Inventory duplicated helper logic and call-site tolerance requirements.
3. Identify candidate helper names that encode behavior ownership rather than
   generic pseudoinverse or QR/SVD parity.
4. Define what must remain visible at owner call sites.
5. Decide whether Day 11 can move one helper safely or must explicitly defer.
6. Write the minimum-norm helper ownership gate.

### Deliverables
- minimum-norm helper ownership map
- helper movement candidate table
- call-site tolerance and diagnostic policy
- Day 11 implementation or deferral criteria

### Completion Criteria
- no helper candidate hides solver-specific behavior or tolerances
- QR solve, COLAMD, SVD, fallback, refinement, zero-row, and SuiteSparse lanes
  remain behavior-specific
- full validation requirements are explicit before any code movement

---

## Day 11: Minimum-Norm Helper Movement Decision

**Title:** Helper Movement
**Theme:** Move one accepted behavior-specific helper or explicitly defer
generic helper consolidation
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 10 helper ownership gate to candidate movements.
2. Move one helper only if behavior-specific name, owner, call-site tolerance,
   diagnostics, and validation plan are pinned.
3. Otherwise document explicit helper movement deferral.
4. Keep public headers and generic helper APIs unchanged unless a separate
   review gate is satisfied.
5. Run focused QR solve, COLAMD, and SVD checks when helper behavior is
   touched.
6. Run full quality validation if `.c` or `.h` files change.

### Deliverables
- helper movement implementation or explicit deferral
- focused validation logs
- public API no-change note or review-gated change note
- helper non-claim update

### Completion Criteria
- helper ownership is clearer than before, or deferral is explicit
- no generic QR/SVD/minimum-norm helper API is created accidentally
- all required focused and full quality checks pass

---

## Day 12: Bidiagonal/Golub-Kahan Helper Gate

**Title:** GK Helper Gate
**Theme:** Decide whether Bidiagonal/Golub-Kahan helpers can move into a
dedicated owner
**Time estimate:** 12 hours

### Tasks
1. Review Bidiagonal and Golub-Kahan helper usage across SVD tests and helper
   headers.
2. Inventory transpose, reconstruction, explicit `U`/`V`, wide skip, and
   bidiagonal QR-iteration semantics that must not be blurred.
3. Identify candidate helper files, static helpers, and ownership boundaries.
4. Define source-list, CMake, Makefile, include, rollback, and validation
   requirements for any movement.
5. Decide whether Day 13 can extract one bounded helper owner or must
   explicitly defer.
6. Write the Bidiagonal/Golub-Kahan helper gate.

### Deliverables
- Bidiagonal/Golub-Kahan helper inventory
- extraction candidate table
- source/build ownership requirements
- Day 13 implementation or deferral checklist

### Completion Criteria
- helper extraction cannot blur transpose, reconstruction, explicit vector, or
  QR-iteration semantics
- build/source-list impact is known before implementation
- validation scope is explicit

---

## Day 13: Bidiagonal/Golub-Kahan Helper Extraction Decision

**Title:** GK Helper Decision
**Theme:** Extract one accepted Bidiagonal/Golub-Kahan helper owner or
explicitly defer extraction
**Time estimate:** 11 hours

### Tasks
1. Apply the Day 12 extraction gate to candidate helper movements.
2. Extract one bounded helper owner only if source/build ownership, public
   boundary, rollback, and validation are ready.
3. Otherwise document explicit extraction deferral with future promotion gates.
4. Run focused Bidiagonal/SVD tests for touched owners.
5. Run source-list, CMake, and full quality validation if `.c` or `.h` files
   change.
6. Record helper ownership, non-claims, and validation results.

### Deliverables
- helper extraction implementation or explicit deferral
- source-list/build update or no-change rationale
- focused and required validation package
- helper ownership non-claim update

### Completion Criteria
- Bidiagonal/Golub-Kahan ownership is clearer or explicitly deferred
- no public API or broad SVD helper claim is introduced accidentally
- all required validation passes before closeout

---

## Day 14: Sprint Closeout and Handoff

**Title:** Sprint Closeout
**Theme:** Publish final Q/economy/helper evidence, deferrals, non-claims, and
Sprint 130 handoff
**Time estimate:** 11 hours

### Tasks
1. Reconcile all Sprint 129 items against the project-plan checklist.
2. Update working notes, artifact indexes, and final decision packages.
3. Confirm every accepted evidence or helper movement has validation and every
   deferred item has blocker, dependency, and future owner.
4. Refresh Q-basis, economy, sparse-mode, SuiteSparse, minimum-norm helper,
   Bidiagonal/Golub-Kahan helper, and Sprint 128 no-reopen non-claims.
5. Prepare Sprint 130 handoff notes for partial-SVD residual expansion and
   solver-selection claim gates.
6. Write the sprint closeout summary.

### Deliverables
- final Sprint 129 evidence and deferral index
- final validation package
- updated non-claim register
- Sprint 130 handoff notes
- sprint closeout summary

### Completion Criteria
- all Sprint 129 project-plan deliverables are present or explicitly deferred
- no unresolved item lacks a future owner and dependency statement
- Sprint 130 can begin without reopening Sprint 129 Q/economy/helper claim
  boundaries
