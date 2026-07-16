# Sprint 126 Plan: Rank-Deficient QR Residual Corpus & Minimum-Norm Follow-Through

**Sprint Duration:** 14 days
**Goal:** Convert Sprint 125's remaining rank-deficient QR, nullspace,
threshold, SuiteSparse, and minimum-norm residual debt into bounded evidence or
explicit future-owner decisions before Q/economy, corpus-index, and adoption
work consume those truth boundaries.

**Starting Point:** Sprint 126 begins from:
- Sprint 125 residual QR, nullspace/subspace, threshold, SuiteSparse, and
  minimum-norm evidence policies
- Sprint 125 explicit broad QR, nullspace, minimum-norm, backend,
  external-library, helper, corpus, and performance non-claims
- Sprint 121-125 QR, QR-solve, SVD-pseudoinverse, external-reference,
  SuiteSparse, rank-threshold, and nullspace fixture evidence
- current `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_colamd.c`,
  `tests/qr_external_dense_reference.py`, and `tests/test_qr_helpers.h`
  ownership boundaries

The sprint must:
- map Sprint 125 deferred debt without duplicating completed residual,
  nullspace, threshold, minimum-norm, SuiteSparse, or helper evidence
- add or explicitly defer compatible zero-residual, dependent-row, and wide
  rank-deficient QR residual fixtures only when they add distinct trust
- expand nullspace/subspace evidence only through projector or two-way
  projection metrics with pinned rank and nullity metadata
- expand QR threshold families only with fixture-local expected ranks,
  diagnostics, and non-global interpretation
- gate SuiteSparse rank-deficient QR corpus evidence behind explicit metadata,
  support tier, diagnostics, skip behavior, and validation rules
- add or explicitly defer optional-large SuiteSparse, rank-deficient
  SuiteSparse, and larger underdetermined minimum-norm evidence with pinned
  residual, norm, rank, nullity, corpus metadata, and exact-value ownership
- add any QR-vs-SVD minimum-norm fixtures only as bounded cross-checks, not as
  a broad SVD-pseudoinverse oracle

**End State:** Sprint 126 leaves behind:
- Sprint 125 residual dedupe and dependency map
- compatible, dependent-row, and wide residual fixture decision package
- nullspace/subspace projector evidence or explicit deferrals
- QR threshold-family evidence or explicit deferrals
- SuiteSparse rank-deficient QR corpus gate
- SuiteSparse and underdetermined minimum-norm evidence decision package
- QR-vs-SVD minimum-norm cross-check gate and non-claim update

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 126 project-plan estimate.

---

## Day 1: Sprint Intake and Residual Dedupe Baseline

**Title:** Residual Dedupe
**Theme:** Establish Sprint 126 scope, duplicate fences, and dependency order
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 126 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Re-read Sprint 125 retrospective residual deferred debt, working notes, and
   day-level artifacts.
3. Inventory completed Sprint 121-125 QR residual, nullspace, threshold,
   SuiteSparse, minimum-norm, SVD-pseudoinverse, and helper evidence.
4. Create Sprint 126 working notes and artifact directories.
5. Map each Sprint 126 project-plan item to day-level owners, likely touched
   files, and validation requirements.
6. Write the initial residual dedupe baseline.

### Deliverables
- Sprint 126 working-notes baseline
- artifact directory structure
- residual dedupe baseline
- day-level owner map
- validation and non-claim boundary notes

### Completion Criteria
- every Sprint 126 project-plan item has a day-level owner
- completed Sprint 121-125 work is not duplicated or silently reopened
- dependency order is explicit before any new evidence is accepted

---

## Day 2: Residual Fixture Trust Policy

**Title:** Residual Trust Policy
**Theme:** Decide which compatible, dependent-row, and wide residual fixtures
add distinct trust
**Time estimate:** 12 hours

### Tasks
1. Review current rank-deficient QR residual-only evidence and deterministic
   rank fixtures.
2. Identify compatible zero-residual, dependent-row, and wide-shape residual
   candidates.
3. Define what each candidate proves and what it must not imply.
4. Fence nullspace, minimum-norm, pseudoinverse, and raw-basis interpretations.
5. Define fixture-local residual tolerances, diagnostics, skip behavior, and
   failure interpretation.
6. Write the residual fixture trust policy.

### Deliverables
- residual fixture candidate table
- trust-value and duplicate-risk analysis
- residual tolerance and diagnostic policy
- explicit non-claim list
- Day 3 implementation checklist

### Completion Criteria
- each accepted candidate has distinct trust value
- residual-only evidence cannot imply nullspace or minimum-norm behavior
- deferred candidates have explicit blockers and promotion gates

---

## Day 3: Compatible and Wide Residual Evidence Batch

**Title:** Residual Evidence
**Theme:** Implement or explicitly defer compatible, dependent-row, and wide
rank-deficient QR residual fixtures
**Time estimate:** 12 hours

### Tasks
1. Review the Day 2 residual fixture trust policy.
2. Select the safest residual fixture batch or explicitly defer all candidates.
3. If accepted, add focused fixtures, expected residual metadata, diagnostics,
   and assertions.
4. If deferred, document blockers and future-owner requirements.
5. Run focused QR checks for any touched tests or helper scripts.
6. Write the residual evidence decision artifact.

### Deliverables
- residual fixture implementation or explicit deferral
- fixture metadata and diagnostics if accepted
- focused validation notes
- future-owner handoff if deferred
- updated residual non-claims

### Completion Criteria
- Project-plan Item 2 is complete or explicitly deferred
- accepted fixtures prove only their documented residual behavior
- focused validation evidence is recorded for any code changes

---

## Day 4: Nullspace/Subspace Expansion Policy Refresh

**Title:** Subspace Policy
**Theme:** Extend Sprint 125 projection rules to multi-dimensional, wide,
near-threshold, dependent-row, and SuiteSparse candidates
**Time estimate:** 12 hours

### Tasks
1. Re-read Sprint 125 nullspace/subspace policy and implemented projector
   evidence.
2. Identify multi-dimensional, wide-shape, near-threshold, dependent-row, and
   SuiteSparse nullspace/subspace candidates.
3. Define pinned rank, nullity, threshold, and fixture metadata requirements.
4. Confirm projector or two-way projection metrics for each candidate class.
5. Define tolerances, diagnostics, skip behavior, and failure interpretation.
6. Write the expanded nullspace/subspace policy artifact.

### Deliverables
- expanded nullspace/subspace candidate matrix
- rank/nullity metadata requirements
- projector and two-way projection metric policy
- tolerance and diagnostic table
- unsupported raw-basis non-claims

### Completion Criteria
- no nullspace/subspace candidate proceeds without pinned rank and nullity
- raw vector equality, basis ordering, and unique-basis claims remain fenced
- SuiteSparse candidates have explicit support-tier and skip requirements

---

## Day 5: Nullspace/Subspace Evidence Batch

**Title:** Subspace Evidence
**Theme:** Implement or explicitly defer the first expanded nullspace/subspace
evidence batch
**Time estimate:** 12 hours

### Tasks
1. Review the Day 4 expanded policy.
2. Select a bounded nullspace/subspace implementation batch or explicitly defer
   the candidate set.
3. If accepted, add fixture keys, pinned metadata, projection metrics,
   diagnostics, and assertions.
4. If deferred, document blockers, dependencies, and future-owner expectations.
5. Run focused QR/nullspace checks for touched surfaces.
6. Write the nullspace/subspace evidence decision artifact.

### Deliverables
- nullspace/subspace evidence implementation or explicit deferral
- pinned rank/nullity metadata if accepted
- projection metric diagnostics
- focused validation notes
- updated nullspace/subspace non-claims

### Completion Criteria
- Project-plan Item 3 is complete or explicitly deferred
- accepted evidence uses only approved projection-style metrics
- broad raw Q-basis, orientation, and unique-nullspace claims remain absent

---

## Day 6: Threshold Family Expansion Policy

**Title:** Threshold Policy
**Theme:** Define scaled, perturbed, dependent-row, wide, and SuiteSparse QR
threshold families
**Time estimate:** 12 hours

### Tasks
1. Review current QR threshold fixtures and external reference metadata.
2. Identify candidate scaled diagonal, perturbed duplicate-column,
   dependent-row, wide, and SuiteSparse threshold families.
3. Define fixture-local expected ranks and threshold values.
4. Define diagnostics, tolerances, stability rules, and skip behavior.
5. Fence global QR rank-threshold and ecosystem-parity interpretations.
6. Write the threshold family expansion policy.

### Deliverables
- threshold family candidate matrix
- expected rank and threshold metadata table
- diagnostics and tolerance policy
- SuiteSparse threshold support-tier notes
- non-global rank-policy non-claims

### Completion Criteria
- each candidate has fixture-local expected rank metadata
- threshold evidence cannot be mistaken for a global rank policy
- Day 7 implementation inputs are explicit

---

## Day 7: Threshold Family Evidence Batch

**Title:** Threshold Evidence
**Theme:** Implement or explicitly defer expanded QR threshold-family evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 6 threshold policy.
2. Select the highest-value bounded threshold batch or explicitly defer the
   candidate set.
3. If accepted, add fixture metadata, expected ranks, diagnostics, assertions,
   and reference data.
4. If deferred, write blockers and promotion gates.
5. Run focused QR threshold tests and script checks for touched surfaces.
6. Write the threshold evidence decision artifact.

### Deliverables
- threshold-family implementation or explicit deferral
- fixture-local expected rank metadata
- focused validation notes
- future-owner handoff if deferred
- updated rank-threshold non-claims

### Completion Criteria
- Project-plan Item 4 is complete or explicitly deferred
- accepted threshold fixtures have pinned metadata and diagnostics
- broad external parity and global threshold claims remain fenced

---

## Day 8: SuiteSparse Rank-Deficient QR Corpus Gate

**Title:** SuiteSparse QR Gate
**Theme:** Decide whether and how SuiteSparse rank-deficient QR corpus evidence
can be accepted
**Time estimate:** 12 hours

### Tasks
1. Inventory checked-in and optional SuiteSparse matrices relevant to
   rank-deficient QR.
2. Define expected-rank metadata requirements and how metadata is generated or
   maintained.
3. Define support tier, optional-data behavior, platform skip behavior,
   diagnostics, and validation requirements.
4. Identify candidate corpus fixtures and duplicate risks against Sprint 125
   evidence.
5. Decide whether a bounded SuiteSparse rank-deficient QR batch can proceed.
6. Write the SuiteSparse rank-deficient QR corpus gate artifact.

### Deliverables
- SuiteSparse rank-deficient QR corpus inventory
- expected-rank metadata protocol
- support-tier and skip-behavior policy
- bounded candidate decision
- validation checklist

### Completion Criteria
- Project-plan Item 5 has explicit acceptance or deferral criteria
- optional corpus behavior is deterministic and documented
- no broad SuiteSparse corpus or platform support claim is introduced

---

## Day 9: SuiteSparse Rank-Deficient QR Evidence Decision

**Title:** SuiteSparse QR Evidence
**Theme:** Implement or explicitly defer SuiteSparse rank-deficient QR corpus
evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 8 corpus gate.
2. Select a bounded SuiteSparse QR evidence batch or explicitly defer the
   candidate set.
3. If accepted, add metadata, fixtures, diagnostics, skip handling, and tests.
4. If deferred, document exact missing metadata or support-tier blockers.
5. Run focused SuiteSparse QR checks for touched surfaces.
6. Write the SuiteSparse QR evidence decision artifact.

### Deliverables
- SuiteSparse QR evidence implementation or explicit deferral
- expected-rank metadata if accepted
- focused validation notes
- support-tier and skip-behavior evidence
- updated SuiteSparse/corpus non-claims

### Completion Criteria
- SuiteSparse rank-deficient QR work is accepted only under Day 8 rules
- checked-in and optional-data behavior is explicit
- broad corpus, platform, and performance claims remain absent

---

## Day 10: SuiteSparse Minimum-Norm Evidence Gate

**Title:** Minimum-Norm Corpus Gate
**Theme:** Define support-tier, residual, norm, rank, nullity, and metadata
requirements for SuiteSparse minimum-norm evidence
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 125 minimum-norm owner-local evidence and residual debt.
2. Identify optional-large SuiteSparse and rank-deficient SuiteSparse
   minimum-norm candidates.
3. Define pinned residual, norm, rank, nullity, shape, support-tier, and corpus
   metadata requirements.
4. Define diagnostics, skip behavior, tolerance rules, and failure
   interpretation.
5. Fence broad QR minimum-norm, pseudoinverse, corpus, and external-library
   parity interpretations.
6. Write the SuiteSparse minimum-norm corpus gate artifact.

### Deliverables
- SuiteSparse minimum-norm candidate inventory
- residual/norm/rank/nullity metadata protocol
- support-tier and skip-behavior policy
- diagnostics and tolerance table
- minimum-norm non-claim update

### Completion Criteria
- Project-plan Item 6 has clear SuiteSparse acceptance criteria
- optional-large behavior is documented before implementation
- no broad minimum-norm or external parity claim is introduced

---

## Day 11: SuiteSparse Minimum-Norm Evidence Batch

**Title:** Minimum-Norm Corpus Evidence
**Theme:** Implement or explicitly defer optional-large and rank-deficient
SuiteSparse minimum-norm evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 10 corpus gate.
2. Select a bounded SuiteSparse minimum-norm batch or explicitly defer all
   candidates.
3. If accepted, add metadata, diagnostics, skip behavior, and owner-local
   assertions.
4. If deferred, document missing support-tier, metadata, or validation blockers.
5. Run focused COLAMD, QR solve, minimum-norm, and SuiteSparse checks for
   touched surfaces.
6. Write the SuiteSparse minimum-norm evidence artifact.

### Deliverables
- SuiteSparse minimum-norm implementation or explicit deferral
- pinned residual/norm/rank/nullity metadata if accepted
- focused validation notes
- future-owner handoff if deferred
- updated minimum-norm and SuiteSparse non-claims

### Completion Criteria
- accepted evidence follows Day 10 metadata and support-tier rules
- assertions remain owner-local and behavior-specific
- broad corpus, performance, and external-library claims remain absent

---

## Day 12: Underdetermined Minimum-Norm and QR-vs-SVD Gate

**Title:** Cross-Check Gate
**Theme:** Decide exact-value underdetermined lanes and bounded QR-vs-SVD
minimum-norm cross-checks
**Time estimate:** 12 hours

### Tasks
1. Review current underdetermined minimum-norm fixtures and Sprint 125
   QR-vs-SVD bounded cross-check decision.
2. Identify larger underdetermined shapes that may deserve exact-value
   contracts.
3. Identify additional QR-vs-SVD minimum-norm fixtures that can remain bounded
   cross-checks.
4. Define fixture keys, exact-value ownership, tolerances, diagnostics, and
   non-oracle wording.
5. Decide which candidates proceed to Day 13 and which remain deferred.
6. Write the underdetermined and QR-vs-SVD cross-check gate artifact.

### Deliverables
- larger underdetermined candidate decision table
- QR-vs-SVD cross-check candidate table
- exact-value ownership and tolerance policy
- non-oracle wording
- Day 13 implementation checklist

### Completion Criteria
- Project-plan Items 6 and 7 have final implementation candidates or deferrals
- QR-vs-SVD checks remain bounded cross-checks, not SVD oracle claims
- exact-value contracts are assigned only to stable fixture shapes

---

## Day 13: Underdetermined and QR-vs-SVD Evidence Batch

**Title:** Cross-Check Evidence
**Theme:** Implement or explicitly defer larger underdetermined and QR-vs-SVD
minimum-norm evidence
**Time estimate:** 11 hours

### Tasks
1. Review the Day 12 gate artifact.
2. Implement the accepted bounded underdetermined and QR-vs-SVD evidence batch,
   or explicitly defer the candidate set.
3. Add fixture metadata, diagnostics, assertions, and reference data if
   accepted.
4. Run focused QR solve, COLAMD, SVD-pseudoinverse, and minimum-norm checks for
   touched surfaces.
5. Update the Sprint 126 non-claim register.
6. Write the Day 13 evidence decision artifact.

### Deliverables
- underdetermined minimum-norm evidence or explicit deferral
- QR-vs-SVD cross-check evidence or explicit deferral
- focused validation notes
- updated non-oracle wording and non-claims
- final validation checklist

### Completion Criteria
- Project-plan Item 7 is complete or explicitly deferred
- accepted cross-checks are fixture-keyed and bounded
- broad SVD-pseudoinverse, dense-library, and external parity claims remain
  absent

---

## Day 14: Full Validation, Claim Gate, and Handoff

**Title:** Validation and Handoff
**Theme:** Validate touched surfaces, update evidence tables, and close Sprint
126 with explicit residuals
**Time estimate:** 11 hours

### Tasks
1. Inventory all Sprint 126 code, script, fixture, and documentation changes.
2. Run required focused checks and full quality gates for touched C surfaces.
3. Run docs/script hygiene checks for touched non-C surfaces.
4. Update maintainer evidence and claim-boundary notes if Sprint 126 changed
   evidence surfaces.
5. Publish the Sprint 126 residual queue and explicit non-claim register.
6. Write closeout notes and handoff requirements for Sprint 127.

### Deliverables
- final validation package
- maintainer evidence and claim-boundary updates if needed
- Sprint 126 residual queue
- explicit QR/nullspace/minimum-norm/SuiteSparse non-claim register
- Sprint 127 handoff notes

### Completion Criteria
- every accepted implementation has validation evidence
- every deferred item has owner, blocker, and promotion-gate notes
- no public/support claim expands beyond earned Sprint 126 evidence
- Sprint 127 receives clear Q/economy/helper prerequisites
