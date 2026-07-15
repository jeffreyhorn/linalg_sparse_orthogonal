# Sprint 125 Plan: Rank-Deficient QR & Minimum-Norm Residual Evidence

**Sprint Duration:** 14 days
**Goal:** Convert Sprint 124's rank-deficient QR and minimum-norm deferred debt
into behavior-specific evidence or explicit deferrals before broader corpus and
adoption work depend on these claims.

**Starting Point:** Sprint 125 begins from:
- Sprint 124 rank-deficient QR policy, minimum-norm behavior contract, and
  helper ownership decisions
- Sprint 124 residual deferred debt and explicit QR/minimum-norm non-claim
  register
- Sprint 121-124 SVD/QR/rank fixture taxonomy, external-reference scripts, and
  bounded QR evidence lanes
- current QR, QR-solve, SVD-pseudoinverse, COLAMD, SuiteSparse, and validation
  owners

The sprint must:
- map deferred Sprint 124 rank-deficient QR and minimum-norm work without
  duplicating completed intake, policy, fixture, or helper decisions
- add or explicitly defer residual-only rank-deficient QR evidence without
  implying nullspace or minimum-norm behavior
- define nullspace/subspace sign, ordering, nullity, projection metric, and
  fixture-local tolerance policies before adding basis-dependent evidence
- add or explicitly defer near-rank-deficient QR threshold evidence with
  threshold-family and stability boundaries
- add or explicitly defer SuiteSparse rank-deficient QR evidence only after
  optional corpus, platform skip, diagnostics, and support-tier behavior are
  explicit
- add or explicitly defer QR minimum-norm COLAMD, fallback, rank-deficient,
  refinement, QR-vs-SVD-pseudoinverse, and SuiteSparse evidence under
  behavior-specific owners
- validate affected work, update maintainer evidence, and preserve broad QR,
  nullspace, minimum-norm, backend, corpus, and dense-library non-claims

**End State:** Sprint 125 leaves behind:
- Sprint 124 deferred QR/minimum-norm dedupe map
- rank-deficient residual evidence or explicit deferral
- nullspace/subspace policy package
- near-rank-deficient threshold decision package
- SuiteSparse rank-deficient QR decision package
- minimum-norm behavior evidence package
- validation and QR non-claim register update

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `164` hours, matching the Sprint 125 project-plan estimate.

---

## Day 1: Sprint Intake and Deferred QR Dedupe Map

**Title:** Deferred QR Dedupe
**Theme:** Establish ownership, duplicate fences, and dependency order for
Sprint 124 carry-forward work
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 125 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Re-read Sprint 124 retrospective residual deferred debt, working notes, and
   day-level artifacts for rank-deficient QR and minimum-norm decisions.
3. Inventory Sprint 121-124 QR, QR-solve, SVD-pseudoinverse, external-reference,
   SuiteSparse, and helper evidence already completed.
4. Create Sprint 125 working notes and artifact directories.
5. Map each Sprint 125 project-plan item to day-level proof owners and
   validation expectations.
6. Write the deferred QR/minimum-norm dedupe map.

### Deliverables
- Sprint 125 working-notes baseline
- artifact directory structure
- deferred QR/minimum-norm dedupe map
- day-level owner map
- validation and non-claim boundary notes

### Completion Criteria
- every Sprint 125 project-plan item has a day-level owner
- completed Sprint 121-124 evidence is not duplicated or silently reopened
- dependency order is explicit before any new evidence is accepted

---

## Day 2: Residual-Only Rank-Deficient QR Trust Gate

**Title:** Residual Trust Gate
**Theme:** Decide when residual-only rank-deficient QR evidence adds trust
without widening claims
**Time estimate:** 12 hours

### Tasks
1. Review current rank-deficient QR deterministic tests and external-reference
   lanes.
2. Identify candidate residual-only fixtures and the exact behavior each can
   prove.
3. Define why each candidate does or does not add trust beyond existing
   deterministic evidence.
4. Fence nullspace, minimum-norm, pseudoinverse, and basis-dependent
   interpretations from residual-only checks.
5. Define diagnostics, tolerances, skip behavior, and failure interpretation for
   any accepted residual-only candidate.
6. Write the residual-only rank-deficient QR trust-gate artifact.

### Deliverables
- residual-only candidate table
- trust-beyond-deterministic rationale
- residual tolerance and diagnostic policy
- explicit non-claim notes for nullspace and minimum-norm behavior
- Day 3 implementation or deferral checklist

### Completion Criteria
- residual-only evidence has a clear proof boundary
- no accepted candidate implies nullspace, minimum-norm, or pseudoinverse
  behavior
- deferred candidates have explicit blockers and promotion gates

---

## Day 3: Rank-Deficient Residual Evidence Batch

**Title:** Residual Evidence
**Theme:** Implement or explicitly defer residual-only rank-deficient QR
evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 2 trust-gate artifact.
2. Select the highest-value residual-only rank-deficient QR batch or decide to
   defer.
3. If accepted, update fixture/reference data, tests, diagnostics, and local
   tolerance handling.
4. If deferred, write the future-owner handoff and dependency blockers.
5. Run focused QR checks required for the touched surfaces.
6. Write the rank-deficient residual evidence decision artifact.

### Deliverables
- accepted residual-only QR evidence or explicit deferral
- fixture/reference protocol updates if accepted
- focused validation notes
- future-owner handoff if deferred
- updated rank-deficient residual non-claims

### Completion Criteria
- Project-plan Item 2 is complete or explicitly deferred
- accepted code or script changes have focused validation evidence
- residual-only proof boundaries remain documented

---

## Day 4: Nullspace and Subspace Policy Design

**Title:** Nullspace Policy
**Theme:** Define sign, ordering, nullity, projection, and tolerance rules before
adding subspace evidence
**Time estimate:** 12 hours

### Tasks
1. Inventory current QR rank, Q-basis, nullspace-adjacent, and
   rank-deficient solve evidence.
2. Define nullity expectations and fixture-local rank thresholds.
3. Define sign and ordering rules for vectors that are not uniquely oriented.
4. Select projection or principal-angle style metrics for subspace evidence.
5. Define fixture-local tolerance policies and diagnostics for subspace
   failures.
6. Write the nullspace/subspace policy artifact.

### Deliverables
- nullity and rank-threshold policy
- sign and ordering policy
- projection/subspace metric policy
- fixture-local tolerance table
- nullspace/subspace non-claim register

### Completion Criteria
- Project-plan Item 3 has explicit acceptance criteria
- basis-dependent evidence cannot be mistaken for raw vector equality
- future tests have stable tolerance and diagnostic rules

---

## Day 5: Nullspace/Subspace Evidence Decision

**Title:** Subspace Decision
**Theme:** Apply the nullspace policy to rank-deficient QR evidence candidates
**Time estimate:** 12 hours

### Tasks
1. Review the Day 4 nullspace/subspace policy artifact.
2. Rank rank-deficient QR nullspace/subspace evidence candidates by trust value,
   fixture stability, and implementation risk.
3. Select a bounded implementation batch or explicitly defer all candidates.
4. If accepted, define fixture keys, expected nullity, metric thresholds, and
   diagnostics.
5. If deferred, write promotion gates and future owner requirements.
6. Update the Sprint 125 non-claim register.

### Deliverables
- rank-deficient nullspace/subspace decision table
- fixture and metric protocol if accepted
- future-owner handoff if deferred
- updated non-claim register
- Day 6 threshold-evidence inputs

### Completion Criteria
- nullspace/subspace evidence is accepted only under Day 4 metric rules
- unsupported raw Q-basis equality and unique-basis claims remain fenced
- deferred work has explicit dependencies and owner

---

## Day 6: Near-Rank-Deficient Threshold Family Design

**Title:** Threshold Families
**Theme:** Define near-rank-deficient QR thresholds, expected ranks, and
stability boundaries
**Time estimate:** 12 hours

### Tasks
1. Inventory current QR rank-threshold behavior and tolerance selection.
2. Define candidate near-rank-deficient matrix families and expected ranks.
3. Define threshold families, scale handling, perturbation size, and stability
   interpretation.
4. Separate fixture-local threshold claims from global rank policy claims.
5. Define diagnostics for threshold-sensitive rank changes.
6. Write the near-rank-deficient threshold-family artifact.

### Deliverables
- near-rank-deficient candidate matrix families
- expected-rank table
- threshold and scale policy
- stability and diagnostics policy
- non-global interpretation notes

### Completion Criteria
- Project-plan Item 4 has explicit fixture-family rules
- no threshold evidence creates a global rank-policy claim
- Day 7 can implement or defer without rediscovering semantics

---

## Day 7: Near-Rank-Deficient Threshold Evidence Decision

**Title:** Threshold Evidence
**Theme:** Implement or explicitly defer near-rank-deficient QR threshold
evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 6 threshold-family artifact.
2. Select the safest near-rank-deficient QR threshold evidence batch or decide
   to defer.
3. If accepted, add fixture/reference data, tests, diagnostics, and
   fixture-local tolerance checks.
4. If deferred, record blockers, promotion gates, and future-owner
   requirements.
5. Run focused checks required for touched QR/reference surfaces.
6. Write the near-rank-deficient threshold decision artifact.

### Deliverables
- accepted threshold evidence or explicit deferral
- fixture/reference updates if accepted
- focused validation notes
- future-owner handoff if deferred
- updated threshold non-claims

### Completion Criteria
- Project-plan Item 4 is complete or explicitly deferred
- threshold evidence remains fixture-local
- accepted implementation has focused validation evidence

---

## Day 8: SuiteSparse Rank-Deficient QR Corpus Policy

**Title:** SuiteSparse Policy
**Theme:** Bound optional corpus, platform skip, diagnostics, and support-tier
behavior before SuiteSparse evidence
**Time estimate:** 12 hours

### Tasks
1. Inventory available SuiteSparse fixtures, optional-data behavior, and
   platform skip conventions.
2. Identify rank-deficient or near-rank-deficient corpus candidates and their
   support-tier implications.
3. Define diagnostics for missing data, unsupported platforms, and numerical
   failures.
4. Decide what corpus evidence can prove without claiming broad SuiteSparse,
   backend, or performance parity.
5. Define validation commands and skip expectations.
6. Write the SuiteSparse rank-deficient QR corpus policy artifact.

### Deliverables
- SuiteSparse candidate and availability table
- platform skip and optional-corpus policy
- diagnostics and support-tier policy
- validation checklist
- SuiteSparse QR non-claim notes

### Completion Criteria
- Project-plan Item 5 has bounded corpus rules
- missing optional data and platform skips are not treated as failures
- broad SuiteSparse and backend parity claims remain fenced

---

## Day 9: SuiteSparse Rank-Deficient QR Evidence Decision

**Title:** SuiteSparse Decision
**Theme:** Implement or explicitly defer SuiteSparse rank-deficient QR evidence
**Time estimate:** 11 hours

### Tasks
1. Review the Day 8 SuiteSparse corpus policy.
2. Select one bounded SuiteSparse rank-deficient QR evidence batch or decide to
   defer.
3. If accepted, update tests, fixture metadata, diagnostics, and skip handling.
4. If deferred, write corpus blockers and future support-tier requirements.
5. Run focused SuiteSparse/QR checks that match the selected path.
6. Write the SuiteSparse rank-deficient QR decision artifact.

### Deliverables
- accepted SuiteSparse rank-deficient QR evidence or explicit deferral
- optional-corpus metadata updates if accepted
- focused validation notes
- future-owner handoff if deferred
- updated SuiteSparse non-claim register

### Completion Criteria
- Project-plan Item 5 is complete or explicitly deferred
- optional-corpus behavior is explicit and reproducible
- support-tier wording remains bounded by evidence

---

## Day 10: Minimum-Norm Behavior Owner Map

**Title:** Minimum-Norm Owners
**Theme:** Split QR minimum-norm evidence into behavior-specific owners before
implementation
**Time estimate:** 12 hours

### Tasks
1. Inventory QR minimum-norm, COLAMD, fallback, rank-deficient, refinement,
   SVD-pseudoinverse, and SuiteSparse coverage.
2. Define behavior-specific owner names, helper boundaries, and expected
   diagnostics.
3. Define residual, norm, and pseudoinverse comparison rules per behavior.
4. Separate QR solve evidence from SVD oracle evidence and fallback behavior.
5. Define validation expectations for each behavior-specific owner.
6. Write the minimum-norm behavior owner map.

### Deliverables
- minimum-norm scenario inventory
- behavior-specific owner map
- residual/norm/pseudoinverse comparison policy
- helper boundary notes
- validation checklist for Days 11-12

### Completion Criteria
- Project-plan Item 6 is decomposed into behavior-specific evidence lanes
- helper names do not hide QR, COLAMD, SVD, fallback, or SuiteSparse semantics
- validation expectations are known before implementation decisions

---

## Day 11: Minimum-Norm Core Evidence Batch

**Title:** Core Min-Norm Evidence
**Theme:** Implement or explicitly defer QR minimum-norm COLAMD, fallback,
rank-deficient, and refinement evidence
**Time estimate:** 11 hours

### Tasks
1. Review the Day 10 behavior owner map.
2. Select the safest core minimum-norm evidence batch or decide to defer.
3. If accepted, add or update QR solve fixtures, tests, helper calls,
   tolerances, and diagnostics.
4. If deferred, document blockers and future owners for each core behavior.
5. Run focused QR-solve checks required by touched code or scripts.
6. Write the core minimum-norm evidence artifact.

### Deliverables
- accepted core minimum-norm evidence or explicit deferrals
- fixture/test updates if accepted
- focused validation notes
- future-owner handoff for deferred behaviors
- updated core minimum-norm non-claims

### Completion Criteria
- COLAMD, fallback, rank-deficient, and refinement lanes have accepted or
  deferred dispositions
- accepted evidence remains behavior-specific
- focused validation matches touched surfaces

---

## Day 12: QR-vs-SVD and SuiteSparse Minimum-Norm Decision

**Title:** Oracle and Corpus Decision
**Theme:** Resolve QR-vs-SVD-pseudoinverse and SuiteSparse minimum-norm evidence
under bounded claim rules
**Time estimate:** 11 hours

### Tasks
1. Review Day 10 and Day 11 minimum-norm artifacts.
2. Decide whether QR-vs-SVD-pseudoinverse comparison is an accepted oracle lane,
   a bounded diagnostic, or a deferral.
3. Decide whether SuiteSparse minimum-norm evidence is accepted or deferred
   under optional-corpus rules.
4. If accepted, define fixture keys, metrics, tolerances, diagnostics, and skip
   behavior.
5. If deferred, write future-owner and promotion-gate requirements.
6. Update minimum-norm non-claims and validation expectations.

### Deliverables
- QR-vs-SVD-pseudoinverse decision package
- SuiteSparse minimum-norm decision package
- oracle/corpus fixture protocol if accepted
- future-owner handoff if deferred
- updated minimum-norm non-claim register

### Completion Criteria
- Project-plan Item 6 is complete or explicitly deferred
- QR and SVD oracle roles are not conflated
- SuiteSparse behavior follows optional-corpus support rules

---

## Day 13: Validation, Evidence Tables, and Claim Gate

**Title:** Validation Gate
**Theme:** Run required checks, refresh maintainer evidence, and preserve
unsupported non-claims
**Time estimate:** 12 hours

### Tasks
1. Inventory all files changed during Sprint 125 and determine required
   validation commands.
2. Run focused checks and required quality gates for touched docs, scripts,
   tests, or C sources.
3. Refresh maintainer evidence tables with accepted evidence, deferrals,
   validation commands, and trust boundaries.
4. Audit public/support docs for unsupported QR, nullspace, minimum-norm,
   backend, corpus, and dense-library claims.
5. Write the validation and claim-gate artifact.
6. Record any failed checks or unresolved blockers for Day 14.

### Deliverables
- validation command matrix and execution summary
- maintainer evidence table updates
- claim-boundary audit notes
- unresolved blocker list if any
- Day 14 closeout checklist

### Completion Criteria
- Project-plan Item 7 validation evidence is available
- public/support wording does not exceed accepted evidence
- any blocker is explicit before sprint closeout

---

## Day 14: Sprint Closeout and Handoff

**Title:** Closeout Handoff
**Theme:** Publish final Sprint 125 evidence, residuals, non-claims, and Sprint
126 inputs
**Time estimate:** 11 hours

### Tasks
1. Reconcile Days 1-13 artifacts against the Sprint 125 project-plan items.
2. Publish accepted implementations, explicit deferrals, validation evidence,
   and residual owner handoffs.
3. Update working notes with final command results, changed files, and claim
   decisions.
4. Prepare Sprint 126 inputs for QR Q-basis, economy, and helper ownership
   follow-through.
5. Write the Sprint 125 retrospective.
6. Confirm final docs hygiene and repository status.

### Deliverables
- final Sprint 125 working notes
- Sprint 125 retrospective
- residual owner handoff for Sprint 126+
- final non-claim register update
- final validation and repository-status notes

### Completion Criteria
- all Sprint 125 deliverables are accepted, explicitly deferred, or handed off
- Sprint 126 has clear inputs and no hidden dependencies
- final validation status and residual non-claims are documented
