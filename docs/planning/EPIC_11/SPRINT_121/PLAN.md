# Sprint 121 Plan: SVD, QR & Rank-Deficient Numerical Oracle Expansion

**Sprint Duration:** 14 days
**Goal:** Strengthen SVD, QR, rank, pseudoinverse, and least-squares evidence
with reusable helpers while keeping LAPACK/SciPy parity as a non-claim.

**Starting Point:** Sprint 121 begins from:
- Sprint 120 fixture/oracle architecture and cross-solver oracle pilot
- existing SVD, partial SVD, QR, least-squares, and rank-deficient proof lanes
- current dense-reference, generated-RHS, reconstruction, residual, and
  orthogonality helpers
- known non-claim boundaries for LAPACK, SciPy, and broad state-of-the-art
  numerical parity
- existing Make, CMake, source-list, and CTest membership expectations

The sprint must:
- audit SVD, partial SVD, QR, rank-deficient, pseudoinverse, low-rank, and
  least-squares proof gaps
- design deterministic matrix taxonomies for rank, conditioning,
  rectangularity, sparsity, scaling, and expected-failure classes
- extract bounded SVD and QR proof helpers without hiding solver-specific
  tolerance and interpretation rules
- expand QR, least-squares, rank-deficient, pseudoinverse, and low-rank
  evidence using the shared taxonomy
- add one bounded dense-reference or external-process comparison lane for a
  high-value SVD/QR path
- validate focused tests, source-list/CMake parity, CTest membership, and full
  quality gates whenever `.c` or `.h` files change
- update maintainer and solver-selection guidance without claiming LAPACK or
  SciPy parity

**End State:** Sprint 121 leaves behind:
- an SVD/QR/rank fixture taxonomy
- reusable SVD/QR proof helpers
- expanded rank-deficient numerical evidence
- one bounded dense-reference or external comparison pilot
- updated trust-boundary and non-claim documentation

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 121 project-plan estimate.

---

## Day 1: Sprint Intake and Evidence Map

**Title:** Evidence Intake
**Theme:** Establish Sprint 121 scope, artifacts, validation rules, and proof-owner map
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 121 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Review Sprint 120 oracle architecture, cross-solver pilot, and closeout
   artifacts for reusable patterns.
3. Create Sprint 121 working notes and artifact directories.
4. Map each Sprint 121 project-plan item to day-level owners.
5. Record validation requirements for documentation-only, test-only,
   source/header, build-system, and CMake/source-list changes.
6. Write the sprint intake artifact with scope boundaries and non-claim
   expectations.

### Deliverables
- Sprint 121 working-notes baseline
- artifact directory structure
- day-level owner map
- Sprint 120 input inventory
- validation and non-claim boundary notes

### Completion Criteria
- every Sprint 121 project-plan item has a day-level owner
- prior oracle patterns and reusable validation lanes are identified
- no implementation begins before proof scope, non-claims, and validation
  expectations are recorded

---

## Day 2: SVD and Partial-SVD Evidence Audit

**Title:** SVD Audit
**Theme:** Inventory SVD, partial SVD, low-rank, rank, and pseudoinverse proof gaps
**Time estimate:** 12 hours

### Tasks
1. Inspect `tests/test_svd.c`, SVD helper headers, examples, and benchmarks for
   reconstruction, orthogonality, singular-value ordering, and low-rank proof
   owners.
2. Identify current full SVD, partial SVD, pseudoinverse, and low-rank fixture
   classes.
3. Map tolerances, rank expectations, dense-reference assumptions, and
   expected-failure behavior.
4. Identify repeated helper logic and candidate extraction boundaries.
5. Write the SVD evidence audit artifact.

### Deliverables
- SVD and partial-SVD proof-owner table
- low-rank and pseudoinverse gap inventory
- tolerance and failure-mode map
- helper extraction candidates
- Day 4 matrix taxonomy inputs

### Completion Criteria
- Item 1 SVD audit inputs are complete
- every SVD proof gap has a candidate owner or explicit deferral reason
- helper opportunities preserve solver-specific tolerance interpretation

---

## Day 3: QR and Rank-Deficient Evidence Audit

**Title:** QR Audit
**Theme:** Inventory QR, least-squares, rank-deficient, and rectangular proof gaps
**Time estimate:** 12 hours

### Tasks
1. Inspect `tests/test_qr.c`, `tests/test_qr_solve.c`, QR examples, and related
   helper code for least-squares and rank-deficient proof owners.
2. Identify current square, overdetermined, underdetermined, null-residual,
   rank-deficient, and reconstruction fixture coverage.
3. Map residual, rank, reconstruction, orthogonality, and solver-comparison
   tolerance expectations.
4. Identify repeated QR/least-squares helper logic and candidate extraction
   boundaries.
5. Write the QR and rank-deficient evidence audit artifact.

### Deliverables
- QR and least-squares proof-owner table
- rank-deficient and rectangular fixture gap inventory
- residual and reconstruction tolerance map
- helper extraction candidates
- Day 4 matrix taxonomy inputs

### Completion Criteria
- Item 1 QR audit inputs are complete
- every QR/rank-deficient gap has a candidate owner or explicit deferral reason
- candidate helpers do not hide rank, residual, or least-squares semantics

---

## Day 4: Matrix Taxonomy Design

**Title:** Taxonomy Design
**Theme:** Define deterministic matrix classes for SVD, QR, rank, and expected failures
**Time estimate:** 12 hours

### Tasks
1. Combine Day 2 and Day 3 audit findings into a shared fixture taxonomy.
2. Define deterministic classes for rank, conditioning, rectangularity,
   sparsity, scaling, duplicate columns, near-dependence, and expected failure.
3. Define fixture metadata: expected rank, singular-value shape, residual
   target, reconstruction tolerance, and non-claim notes.
4. Define helper placement and naming for taxonomy builders.
5. Write the matrix taxonomy design artifact.

### Deliverables
- SVD/QR/rank fixture taxonomy
- fixture metadata schema
- expected-failure class definitions
- helper placement plan
- implementation sequencing notes

### Completion Criteria
- Item 2 matrix taxonomy design is complete
- each fixture class has explicit expected behavior and tolerance ownership
- no taxonomy entry claims broad LAPACK/SciPy parity

---

## Day 5: Helper Extraction Plan

**Title:** Helper Plan
**Theme:** Prepare bounded SVD and QR proof-helper extraction before edits
**Time estimate:** 12 hours

### Tasks
1. Select SVD helper extraction candidates for reconstruction, orthogonality,
   rank, low-rank, and pseudoinverse checks.
2. Select QR helper extraction candidates for reconstruction, residual,
   least-squares, rank-deficient, and generated-RHS checks.
3. Define exact files, helper headers, static helper boundaries, and rollback
   expectations.
4. Define focused validation commands and expected CTest/build-system impact.
5. Write the helper extraction plan artifact.

### Deliverables
- SVD helper extraction checklist
- QR helper extraction checklist
- file-boundary and naming plan
- focused validation command list
- rollback instructions

### Completion Criteria
- Item 3 extraction can proceed from exact boundaries
- every selected helper has a named behavior owner
- validation and rollback commands are recorded before implementation

---

## Day 6: SVD Helper Extraction

**Title:** SVD Helpers
**Theme:** Extract bounded SVD proof helpers while preserving behavior
**Time estimate:** 12 hours

### Tasks
1. Implement selected SVD reconstruction and orthogonality helpers.
2. Implement selected rank, low-rank, and pseudoinverse helper paths.
3. Preserve existing SVD tolerances and expected failure semantics.
4. Run focused SVD validation commands.
5. Record implementation evidence and any deferred helper candidates.

### Deliverables
- extracted SVD proof helpers
- updated SVD tests using helper boundaries
- focused SVD validation evidence
- implementation notes
- residual helper queue

### Completion Criteria
- Item 3 SVD helper extraction has an implemented first batch
- focused SVD tests pass
- behavior-preservation evidence is recorded before any broader refactor

---

## Day 7: QR Helper Extraction

**Title:** QR Helpers
**Theme:** Extract bounded QR and least-squares proof helpers while preserving behavior
**Time estimate:** 12 hours

### Tasks
1. Implement selected QR reconstruction and residual helpers.
2. Implement selected rank-deficient, least-squares, and generated-RHS helper
   paths.
3. Preserve existing QR tolerances and expected failure semantics.
4. Run focused QR validation commands.
5. Record implementation evidence and any deferred helper candidates.

### Deliverables
- extracted QR proof helpers
- updated QR tests using helper boundaries
- focused QR validation evidence
- implementation notes
- residual helper queue

### Completion Criteria
- Item 4 QR helper extraction has an implemented first batch
- focused QR tests pass
- rank, residual, and least-squares interpretations remain visible

---

## Day 8: Rank-Deficient Fixture Expansion

**Title:** Rank Fixtures
**Theme:** Add deterministic rank-deficient and near-dependent evidence
**Time estimate:** 12 hours

### Tasks
1. Add or refine deterministic duplicate-column, near-dependent, and
   rank-deficient fixtures from the taxonomy.
2. Apply the fixtures to SVD rank, QR rank, least-squares, and pseudoinverse
   proof lanes where appropriate.
3. Record expected rank and tolerance reasoning for each fixture.
4. Run focused SVD/QR/rank validation commands.
5. Write the rank-deficient fixture expansion artifact.

### Deliverables
- expanded rank-deficient fixture coverage
- expected-rank metadata
- focused validation evidence
- tolerance rationale
- residual fixture queue

### Completion Criteria
- Item 4 rank-deficient evidence is expanded
- each new fixture has deterministic expected behavior
- focused validation passes before broader test integration

---

## Day 9: Least-Squares and Pseudoinverse Expansion

**Title:** LS Pinv
**Theme:** Expand least-squares, pseudoinverse, and rectangular matrix evidence
**Time estimate:** 12 hours

### Tasks
1. Add focused overdetermined and underdetermined least-squares cases from the
   taxonomy.
2. Add or refine pseudoinverse evidence for bounded deterministic fixtures.
3. Compare reported residuals with true residual helpers where available.
4. Record non-claims for numerical optimality and external library parity.
5. Run focused least-squares and pseudoinverse validation commands.

### Deliverables
- least-squares fixture expansion
- pseudoinverse proof expansion
- residual comparison evidence
- non-claim notes
- focused validation output

### Completion Criteria
- Item 4 least-squares and pseudoinverse evidence is expanded
- tests validate behavior without claiming broad optimality parity
- focused validation passes

---

## Day 10: Low-Rank and Partial-SVD Expansion

**Title:** Low-Rank Proofs
**Theme:** Expand bounded low-rank and partial-SVD evidence
**Time estimate:** 12 hours

### Tasks
1. Add deterministic low-rank fixtures from the taxonomy.
2. Add or refine partial-SVD reconstruction, singular-value ordering, and
   bounded approximation evidence.
3. Compare low-rank approximation behavior against internal dense references
   where feasible.
4. Record tolerance and non-claim boundaries.
5. Run focused partial-SVD and low-rank validation commands.

### Deliverables
- low-rank fixture expansion
- partial-SVD proof expansion
- dense-reference comparison notes
- tolerance and non-claim rationale
- focused validation output

### Completion Criteria
- Item 3 low-rank and partial-SVD helper use is demonstrated
- focused validation passes
- evidence remains bounded to deterministic fixtures

---

## Day 11: Dense Reference or External Pilot Design

**Title:** Reference Design
**Theme:** Design one bounded dense-reference or external-process SVD/QR comparison lane
**Time estimate:** 12 hours

### Tasks
1. Select one high-value SVD or QR lane for dense-reference or external-process
   comparison.
2. Define fixture size, determinism, expected output, timeout, skip behavior,
   and failure interpretation.
3. Define whether the pilot uses in-process dense reference or optional
   external tooling.
4. Define focused commands, environment gates, and CI non-claim wording.
5. Write the reference pilot design artifact.

### Deliverables
- selected dense-reference or external comparison lane
- fixture and timeout specification
- skip and failure policy
- focused validation command list
- non-claim wording

### Completion Criteria
- Item 5 pilot design is complete
- comparison scope is bounded and deterministic
- optional external behavior cannot become an implicit parity claim

---

## Day 12: Dense Reference or External Pilot Implementation

**Title:** Reference Pilot
**Theme:** Implement one bounded reference comparison lane
**Time estimate:** 12 hours

### Tasks
1. Implement the selected dense-reference or external-process comparison lane.
2. Add any required helper, test, Make, CMake, source-list, or CTest updates.
3. Ensure skip behavior and failure messages match the Day 11 design.
4. Run focused validation and any source-list/CMake parity checks.
5. Write the reference pilot implementation artifact.

### Deliverables
- bounded reference comparison implementation
- build-system and CTest updates if needed
- focused validation evidence
- implementation notes
- residual pilot queue

### Completion Criteria
- Item 5 pilot is implemented or explicitly deferred with evidence
- focused validation passes
- optional external comparison behavior is documented and bounded

---

## Day 13: Full Validation and Documentation Updates

**Title:** Validation Docs
**Theme:** Validate SVD/QR evidence and update trust-boundary guidance
**Time estimate:** 12 hours

### Tasks
1. Run focused SVD, QR, rank, least-squares, and reference-pilot validation
   commands.
2. Run `make format && make lint && make test` if any `.c` or `.h` files
   changed.
3. Update solver-selection, maintainer, or planning guidance affected by the
   new evidence.
4. Record explicit LAPACK/SciPy non-claims and trust boundaries.
5. Write the validation and documentation update artifact.

### Deliverables
- focused validation package
- full quality-chain evidence when required
- updated guidance or docs
- non-claim register update
- residual validation queue

### Completion Criteria
- Item 6 validation is complete
- Item 7 documentation updates are complete
- no code-affecting change is left without required quality evidence

---

## Day 14: Sprint Closeout and Residual Queue

**Title:** Oracle Closeout
**Theme:** Consolidate Sprint 121 evidence, residuals, and handoff notes
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 121 artifacts, working notes, code changes, and
   validation outputs.
2. Confirm each project-plan item has a completed, deferred, or blocked status.
3. Build the residual SVD/QR/rank/pseudoinverse queue for future sprints.
4. Record build-system, CTest, source-list, and non-claim evidence.
5. Write the sprint closeout artifact and prepare retrospective inputs.

### Deliverables
- Sprint 121 closeout artifact
- completed/deferred item table
- residual SVD/QR/rank queue
- validation and non-claim summary
- retrospective input notes

### Completion Criteria
- all Sprint 121 deliverables are accounted for
- residual debt is specific enough to schedule in later sprints
- Sprint 121 can close without unresolved validation or non-claim ambiguity
