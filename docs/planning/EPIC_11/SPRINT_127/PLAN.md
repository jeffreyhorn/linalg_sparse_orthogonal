# Sprint 127 Plan: QR Deferred Evidence Semantics & Corpus Follow-Through

**Sprint Duration:** 14 days
**Goal:** Convert Sprint 126's remaining rank-deficient QR residual,
nullspace/subspace, threshold-family, SuiteSparse corpus, optional-large, and
minimum-norm deferred debt into bounded evidence or explicit future-owner
decisions before Q/economy, partial-SVD, corpus-index, performance, package,
and adoption work consume those truth boundaries.

**Starting Point:** Sprint 127 begins from:
- Sprint 126 residual dependency map, fixture trust policies, corpus gates, and
  minimum-norm evidence decisions
- Sprint 126 broad QR, nullspace, minimum-norm, SuiteSparse, optional-data,
  helper, platform, and parity non-claims
- Sprint 121-126 QR residual, nullspace/subspace, threshold-family,
  SuiteSparse, minimum-norm, SVD-pseudoinverse, and helper evidence
- current `tests/test_qr.c`, `tests/test_qr_solve.c`,
  `tests/test_colamd.c`, `tests/qr_external_dense_reference.py`, and
  QR/SVD helper ownership boundaries

The sprint must:
- map Sprint 126 deferred debt without duplicating completed intake, fixture,
  policy, evidence, or claim-gate work
- add or explicitly defer compatible zero-residual and wide residual-only QR
  evidence only when output semantics and proof value are explicit
- expand nullspace/subspace evidence only after rank/nullity, projection
  metrics, support tier, tolerance, and sparse/economy semantics are pinned
- add or explicitly defer perturbed duplicate-column, dependent-row, wide,
  default-threshold, and SuiteSparse QR threshold families with fixture-local
  expected ranks and diagnostics
- gate SuiteSparse rank-deficient QR evidence behind expected-rank metadata,
  threshold semantics, support tier, skip behavior, runtime budget, and
  validation requirements
- gate SuiteSparse and optional-large QR/minimum-norm evidence behind corpus
  extraction rules, shape/nnz/RHS metadata, residual/norm metrics, missing-data
  skips, runtime expectations, and support-tier wording
- add larger exact underdetermined lanes and QR-vs-SVD minimum-norm
  cross-checks only as bounded behavior-specific evidence, not broad
  SVD-pseudoinverse or external-library parity

**End State:** Sprint 127 leaves behind:
- Sprint 126 deferred dedupe and dependency map
- compatible zero-residual and wide residual-only decision package
- QR nullspace/subspace expansion evidence or explicit deferrals
- QR threshold-family follow-through evidence or explicit deferrals
- SuiteSparse rank-deficient QR corpus evidence gate
- SuiteSparse and optional-large minimum-norm evidence decision package
- minimum-norm exact/cross-check/helper claim gate and non-claim update

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 127 project-plan estimate.

---

## Day 1: Sprint Intake and Deferred Dedupe Baseline

**Title:** Deferred Dedupe
**Theme:** Establish Sprint 127 scope, duplicate fences, and dependency order
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 127 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Re-read Sprint 126 retrospective residual deferred debt, working notes, and
   day-level artifacts.
3. Inventory completed Sprint 121-126 QR residual, nullspace/subspace,
   threshold, SuiteSparse, minimum-norm, SVD-pseudoinverse, and helper
   evidence.
4. Create Sprint 127 working notes and artifact directories.
5. Map each Sprint 127 project-plan item to day-level owners, likely touched
   files, and validation requirements.
6. Write the initial deferred dedupe baseline.

### Deliverables
- Sprint 127 working-notes baseline
- artifact directory structure
- deferred dedupe baseline
- day-level owner map
- validation and non-claim boundary notes

### Completion Criteria
- every Sprint 127 project-plan item has a day-level owner
- completed Sprint 121-126 work is not duplicated or silently reopened
- dependency order is explicit before any new evidence is accepted

---

## Day 2: Compatible and Wide Residual Semantics Policy

**Title:** Residual Semantics
**Theme:** Decide when compatible zero-residual and wide residual-only evidence
adds distinct trust
**Time estimate:** 12 hours

### Tasks
1. Review current compatible, dependent-row, and residual-only QR evidence.
2. Identify compatible zero-residual and wide residual-only candidates.
3. Define the exact output semantics required for underdetermined and wide
   residual-only fixtures.
4. Define what each candidate proves and what it cannot imply.
5. Fence nullspace, minimum-norm, pseudoinverse, raw-basis, and wide-solve
   interpretations.
6. Write the compatible and wide residual semantics policy.

### Deliverables
- compatible zero-residual candidate table
- wide residual-only candidate table
- output-semantics and proof-value policy
- residual tolerance and diagnostic policy
- explicit nullspace/minimum-norm non-claims

### Completion Criteria
- no residual candidate proceeds without distinct trust value
- wide residual-only fixtures have pinned output semantics or are deferred
- residual evidence cannot imply nullspace or minimum-norm behavior

---

## Day 3: Compatible and Wide Residual Evidence Decision

**Title:** Residual Decision
**Theme:** Implement or explicitly defer compatible zero-residual and wide
residual-only QR evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 2 residual semantics policy.
2. Select a bounded residual implementation batch or explicitly defer all
   candidates.
3. If accepted, add focused fixtures, expected residual metadata, diagnostics,
   and assertions.
4. If deferred, document blockers, dependencies, and future-owner gates.
5. Run focused QR checks for touched tests or helper scripts.
6. Write the residual evidence decision package.

### Deliverables
- compatible or wide residual fixture implementation or explicit deferral
- fixture metadata and diagnostics if accepted
- focused validation notes
- future-owner handoff if deferred
- updated residual non-claims

### Completion Criteria
- Project-plan Item 2 is complete or explicitly deferred
- accepted evidence proves only documented residual behavior
- focused validation evidence is recorded for any code or script changes

---

## Day 4: Nullspace/Subspace Expansion Policy

**Title:** Subspace Policy
**Theme:** Pin rank/nullity, projection metrics, and output semantics before
adding broader subspace evidence
**Time estimate:** 12 hours

### Tasks
1. Re-read Sprint 125 and Sprint 126 nullspace/subspace policies and evidence.
2. Identify wide-shape, dependent-row, near-threshold, and SuiteSparse
   nullspace/subspace candidates.
3. Define rank, nullity, threshold, and fixture metadata requirements.
4. Select projector or two-way projection residual metrics for each candidate
   class.
5. Define sparse/economy output semantics, support tier, skip behavior,
   tolerances, and diagnostics.
6. Write the nullspace/subspace expansion policy.

### Deliverables
- nullspace/subspace candidate matrix
- rank/nullity and threshold metadata requirements
- projector and two-way projection metric policy
- sparse/economy output semantics notes
- support-tier, skip, tolerance, and diagnostic table

### Completion Criteria
- no subspace candidate proceeds without pinned rank and nullity
- raw basis equality, basis ordering, and unique-basis claims remain fenced
- SuiteSparse candidates have explicit support-tier and skip requirements

---

## Day 5: Nullspace/Subspace Evidence Batch

**Title:** Subspace Evidence
**Theme:** Implement or explicitly defer wide, dependent-row, near-threshold,
and SuiteSparse nullspace/subspace evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 4 expansion policy.
2. Select the safest nullspace/subspace evidence batch or explicitly defer the
   candidate set.
3. If accepted, add fixture keys, pinned metadata, projection metrics,
   diagnostics, and assertions.
4. If deferred, document blockers, dependencies, and promotion gates.
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
- accepted evidence uses approved projection-style metrics
- broad raw Q-basis, orientation, and unique-nullspace claims remain absent

---

## Day 6: Threshold Family Follow-Through Policy

**Title:** Threshold Policy
**Theme:** Define perturbed, dependent-row, wide, default-threshold, and
SuiteSparse threshold-family gates
**Time estimate:** 12 hours

### Tasks
1. Review current QR threshold fixtures and external reference metadata.
2. Identify perturbed duplicate-column, dependent-row, wide, default-threshold,
   and SuiteSparse threshold-family candidates.
3. Define perturbation sizes, threshold values, expected ranks, and separation
   requirements.
4. Define diagnostics, tolerances, stability rules, support tiers, and skip
   behavior.
5. Fence global QR rank-threshold, default-threshold, and ecosystem-parity
   interpretations.
6. Write the threshold family follow-through policy.

### Deliverables
- threshold-family candidate table
- perturbation and threshold separation rules
- expected-rank metadata requirements
- diagnostics, support-tier, and skip behavior policy
- rank-policy and parity non-claims

### Completion Criteria
- perturbation sizes and thresholds are separated enough for stable claims
- each candidate has fixture-local expected-rank semantics
- no global default-threshold or numerical-rank policy claim is introduced

---

## Day 7: Threshold Family Evidence Batch

**Title:** Threshold Evidence
**Theme:** Implement or explicitly defer the next QR threshold-family evidence
batch
**Time estimate:** 12 hours

### Tasks
1. Review the Day 6 threshold policy.
2. Select a bounded threshold-family implementation batch or explicitly defer
   all candidates.
3. If accepted, add fixture/reference data, expected ranks, diagnostics, and
   assertions.
4. If deferred, record blockers and future-owner promotion gates.
5. Run focused QR threshold checks for touched tests or scripts.
6. Write the threshold-family evidence artifact.

### Deliverables
- threshold-family implementation or explicit deferral
- fixture-local expected-rank metadata if accepted
- diagnostic output notes
- focused validation evidence
- updated threshold non-claims

### Completion Criteria
- Project-plan Item 4 is complete or explicitly deferred
- accepted evidence has clear fixture-local threshold interpretation
- broad default-threshold and external parity claims remain absent

---

## Day 8: SuiteSparse Rank-Deficient QR Corpus Gate

**Title:** QR Corpus Gate
**Theme:** Decide whether SuiteSparse rank-deficient QR corpus evidence is ready
for bounded implementation
**Time estimate:** 12 hours

### Tasks
1. Review prior SuiteSparse QR corpus gates and deferrals.
2. Inventory candidate rank-deficient SuiteSparse matrices and metadata
   availability.
3. Define expected-rank metadata, threshold semantics, support tier, skip
   behavior, runtime budget, and diagnostics.
4. Decide default, optional-large, or explicit deferral status for each
   candidate.
5. Define validation requirements and platform expectations.
6. Write the SuiteSparse rank-deficient QR corpus gate artifact.

### Deliverables
- SuiteSparse rank-deficient QR candidate inventory
- expected-rank and threshold metadata policy
- default versus optional-large decision table
- runtime, skip, support-tier, and diagnostics policy
- Day 9 implementation or deferral checklist

### Completion Criteria
- no SuiteSparse QR corpus candidate proceeds without expected-rank metadata
- runtime and missing-data behavior are explicit
- corpus support wording remains bounded

---

## Day 9: SuiteSparse Rank-Deficient QR Evidence Decision

**Title:** QR Corpus Evidence
**Theme:** Implement or explicitly defer SuiteSparse rank-deficient QR corpus
evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 8 corpus gate artifact.
2. Select default-safe or optional-large SuiteSparse QR candidates, or defer.
3. If accepted, add metadata, skip behavior, diagnostics, tests, and focused
   validation.
4. If deferred, document missing metadata, runtime, platform, or support-tier
   blockers.
5. Verify that no broad SuiteSparse corpus or performance claim is introduced.
6. Write the SuiteSparse QR evidence decision artifact.

### Deliverables
- SuiteSparse QR corpus evidence implementation or explicit deferral
- expected-rank metadata and diagnostics if accepted
- focused validation notes
- optional-large/default registration decision
- updated corpus and performance non-claims

### Completion Criteria
- Project-plan Item 5 is complete or explicitly deferred
- accepted corpus evidence has deterministic skip and diagnostic behavior
- broad corpus, platform, runtime, and performance claims remain absent

---

## Day 10: SuiteSparse and Optional-Large Minimum-Norm Gate

**Title:** Minimum-Norm Corpus Gate
**Theme:** Define corpus extraction and support rules for SuiteSparse and
optional-large QR/minimum-norm evidence
**Time estimate:** 12 hours

### Tasks
1. Review existing QR minimum-norm, COLAMD, fallback, refinement, and
   QR-vs-SVD evidence.
2. Inventory additional SuiteSparse and optional-large QR/minimum-norm
   candidates.
3. Define extraction rule, shape, nnz, RHS, rank/nullity if claimed, residual
   metrics, norm metrics, skip behavior, and support tier.
4. Decide which candidates can be default-safe, optional-large, or deferred.
5. Define runtime and platform expectations before any default registration.
6. Write the SuiteSparse and optional-large minimum-norm gate artifact.

### Deliverables
- SuiteSparse minimum-norm candidate inventory
- extraction, shape, nnz, RHS, rank/nullity metadata policy
- residual and norm metric policy
- optional-large/default/deferred decision table
- runtime, skip, and support-tier notes

### Completion Criteria
- no corpus minimum-norm candidate proceeds without extraction and metric rules
- optional-large work has explicit missing-data and runtime behavior
- broad minimum-norm, SuiteSparse, and platform claims remain fenced

---

## Day 11: SuiteSparse and Optional-Large Minimum-Norm Evidence Decision

**Title:** Minimum-Norm Evidence
**Theme:** Implement or explicitly defer SuiteSparse and optional-large
QR/minimum-norm evidence
**Time estimate:** 12 hours

### Tasks
1. Review the Day 10 minimum-norm corpus gate.
2. Select a bounded evidence batch or explicitly defer the candidate set.
3. If accepted, add fixture metadata, RHS construction, residual/norm
   diagnostics, skip behavior, and assertions.
4. If deferred, document blockers and future-owner requirements.
5. Run focused QR solve/COLAMD/minimum-norm checks for touched surfaces.
6. Write the minimum-norm corpus evidence decision artifact.

### Deliverables
- SuiteSparse or optional-large minimum-norm evidence implementation or
  explicit deferral
- fixture metadata and diagnostics if accepted
- focused validation notes
- future-owner handoff if deferred
- updated minimum-norm corpus non-claims

### Completion Criteria
- Project-plan Item 6 is complete or explicitly deferred
- accepted evidence reports residual and norm metrics under pinned semantics
- broad SuiteSparse, optional-data, platform, and minimum-norm claims remain absent

---

## Day 12: Exact Minimum-Norm and QR-vs-SVD Cross-Check Gate

**Title:** Cross-Check Gate
**Theme:** Define larger exact underdetermined and QR-vs-SVD minimum-norm
cross-check boundaries
**Time estimate:** 12 hours

### Tasks
1. Review existing exact underdetermined and QR-vs-SVD minimum-norm evidence.
2. Identify larger non-duplicate exact underdetermined fixture candidates.
3. Identify bounded QR-vs-SVD cross-check candidates and fixture keys.
4. Define closed-form expected values, residual tolerances, value tolerances,
   norm tolerances, SVD tolerance, and non-oracle wording.
5. Revisit helper movement only through behavior-specific helper names and
   focused validation requirements.
6. Write the exact/cross-check/helper claim gate artifact.

### Deliverables
- larger exact underdetermined candidate table
- QR-vs-SVD cross-check candidate table
- residual/value/norm/SVD tolerance policy
- helper movement boundary notes
- explicit non-oracle and non-parity wording

### Completion Criteria
- no exact fixture proceeds without closed-form expected values
- QR-vs-SVD checks remain bounded cross-checks, not oracle claims
- helper movement has behavior-specific ownership and validation gates

---

## Day 13: Exact Minimum-Norm and Cross-Check Evidence Decision

**Title:** Cross-Check Evidence
**Theme:** Implement or explicitly defer larger exact minimum-norm and QR-vs-SVD
cross-check evidence
**Time estimate:** 11 hours

### Tasks
1. Review the Day 12 exact/cross-check/helper gate.
2. Select a bounded evidence batch or explicitly defer all candidates.
3. If accepted, add fixture data, expected values, residual/norm/value
   diagnostics, and assertions.
4. If deferred, record blockers and future-owner promotion criteria.
5. Run focused QR solve, SVD-pseudoinverse, and helper checks for touched
   surfaces.
6. Write the exact/cross-check evidence decision artifact.

### Deliverables
- exact minimum-norm or QR-vs-SVD cross-check implementation or explicit
  deferral
- expected-value and tolerance metadata if accepted
- focused validation notes
- helper movement decision notes
- updated SVD-pseudoinverse and minimum-norm non-claims

### Completion Criteria
- Project-plan Item 7 is complete or explicitly deferred
- accepted evidence remains behavior-specific and bounded
- broad SVD-pseudoinverse, helper API, and parity claims remain absent

---

## Day 14: Validation, Claim Gate, and Sprint Handoff

**Title:** Validation Handoff
**Theme:** Validate touched work, update claim boundaries, and prepare the
Sprint 128 handoff
**Time estimate:** 11 hours

### Tasks
1. Inventory all Sprint 127 code, script, documentation, and artifact changes.
2. Run focused checks and required quality gates for touched surfaces.
3. Recheck evidence artifacts against the Sprint 127 duplicate fences and
   non-claim register.
4. Update maintainer or planning notes only where new evidence changes support
   truth.
5. Publish final validation results, residual deferred debt, and Sprint 128
   handoff notes.
6. Prepare the retrospective input package.

### Deliverables
- final validation package
- updated evidence and non-claim register
- residual deferred debt queue
- Sprint 128 Q/economy/helper handoff notes
- retrospective input package

### Completion Criteria
- every Sprint 127 project-plan item is implemented or explicitly deferred
- required checks pass or blockers are documented before closeout
- no unsupported QR, nullspace, minimum-norm, SuiteSparse, optional-data,
  helper, platform, performance, or parity claim is introduced
