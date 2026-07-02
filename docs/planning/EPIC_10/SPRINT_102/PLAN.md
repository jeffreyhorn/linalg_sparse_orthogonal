# Sprint 102 Plan: Direct Solver Robustness & External Oracle Expansion

**Sprint Duration:** 14 days
**Goal:** Deepen correctness evidence for direct solvers with named external
or dense-reference comparisons and cleaner family-local proof ownership. This
sprint implements the Sprint 102 section of
`docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 102 begins from:
- the Sprint 100 evidence templates and claim-boundary model
- `docs/planning/EPIC_10/SPRINT_100/artifacts/day9-solver-comparison-template.md`
- `docs/planning/EPIC_10/SPRINT_100/artifacts/templates/solver-comparison-evidence-template.md`
- `docs/planning/EPIC_10/SPRINT_101/artifacts/day13-validation-and-reconciliation.md`
- `docs/planning/EPIC_10/SPRINT_101/artifacts/day14-closeout-and-handoff.md`
- the stable Sprint 101 compressed-input ownership and lifecycle rules

The strongest Sprint 102 pressure is to expand direct-solver correctness
evidence without turning one or two fixture lanes into broad solver-family or
state-of-the-art claims. The sprint must:
- re-rank Cholesky, LDLT, LU, QR, SVD, and dispatch paths by oracle depth,
  fixture diversity, and failure-mode clarity
- define fixture classes before adding new comparisons
- extract reusable helper logic only where it reduces giant-test or
  proof-owner concentration
- add the highest-value CSC direct-family oracle expansion
- add the highest-value LU/QR/SVD oracle or failure-mode expansion
- update direct-solver guidance with clear trust boundaries
- preserve Sprint 101 non-claims around direct CSR/CSC solver APIs and broad
  compressed parity

**End State:** Sprint 102 leaves behind:
- a direct-solver gap audit and ranked evidence queue
- fixture taxonomy for symmetry, definiteness, rank, scaling, sparsity,
  ordering, and expected failures
- reusable direct-solver fixture or oracle helpers where justified
- expanded direct-solver oracle coverage for selected families
- clearer direct-solver failure-mode tests
- updated direct-solver selection and trust-boundary documentation
- validation artifacts and Sprint 103 handoff criteria

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 102 project-plan estimate.

---

## Day 1: Sprint 102 Scope & Evidence Baseline

**Title:** Scope Baseline
**Theme:** Convert Sprint 102 project-plan items and Sprint 100/101 handoffs
into one bounded direct-solver evidence package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 102 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 100 solver evidence templates and Sprint 101 closeout
   handoff.
3. Inventory required Sprint 102 workstreams:
   - direct-solver gap audit
   - fixture taxonomy
   - oracle helper extraction
   - LDLT/Cholesky expansion
   - LU/QR/SVD expansion
   - solver selection docs
   - validation and closeout
4. Create Sprint 102 working notes and an artifacts directory.
5. Record validation expectations for docs-only, helper-touch, test-touch,
   source-touch, and workflow-touch days.

### Deliverables
- Sprint 102 workstream inventory
- working-notes baseline
- initial artifacts directory structure
- validation expectation list

### Completion Criteria
- every Sprint 102 project-plan item has day-level ownership
- Sprint 100 evidence-template rules are visible in working notes
- Sprint 101 compressed-input non-claims remain explicit

---

## Day 2: Direct Solver Gap Audit

**Title:** Solver Gap Audit
**Theme:** Re-rank direct solver families by oracle depth, fixture diversity,
failure-mode clarity, and proof-owner concentration
**Time estimate:** 12 hours

### Tasks
1. Inventory Cholesky, LDLT, LU, QR, SVD, and direct-dispatch tests.
2. Classify existing oracle evidence as internal-only, dense-reference,
   external-helper, fixture-corpus, smoke, or failure-mode proof.
3. Identify giant-test and large-source ownership concentrations affecting
   direct-solver proof maintenance.
4. Score each family by user value, current evidence gap, implementation risk,
   and validation cost.
5. Write the direct-solver gap audit artifact.

### Deliverables
- direct-solver evidence inventory
- oracle-depth and fixture-diversity table
- proof-owner concentration notes
- ranked expansion queue

### Completion Criteria
- Cholesky, LDLT, LU, QR, SVD, and dispatch paths are all classified
- candidate expansion lanes are ranked before fixture design begins
- no new oracle or test implementation starts before the audit is recorded

---

## Day 3: Fixture Taxonomy Design

**Title:** Fixture Taxonomy
**Theme:** Define fixture classes and expected outcomes before adding solver
oracle coverage
**Time estimate:** 12 hours

### Tasks
1. Define fixture classes for symmetry, definiteness, rank, scaling, sparsity
   pattern, ordering, and conditioning.
2. Define expected-failure classes for singular, indefinite, rectangular,
   malformed, and numerically difficult inputs.
3. Map fixture classes to direct solver families and public trust boundaries.
4. Define naming, storage, and generation rules for reusable fixtures.
5. Write the fixture taxonomy artifact.

### Deliverables
- fixture class taxonomy
- expected-failure matrix
- solver-family fixture mapping
- fixture naming and storage rules

### Completion Criteria
- fixture classes are solver-neutral where possible and family-local where
  necessary
- expected failures are separated from correctness regressions
- new tests can cite a fixture taxonomy entry

---

## Day 4: Oracle Helper Boundary Freeze

**Title:** Helper Boundary
**Theme:** Decide which dense-reference or external-comparison helpers should
be extracted or reused
**Time estimate:** 12 hours

### Tasks
1. Review existing dense-reference and external-helper patterns in direct
   solver tests.
2. Identify duplicate fixture construction, residual checking, tolerance
   handling, and subprocess-helper logic.
3. Select the smallest helper extraction that improves proof ownership without
   widening the public API.
4. Define helper inputs, outputs, failure behavior, and validation commands.
5. Write the helper boundary artifact.

### Deliverables
- helper extraction decision record
- helper API and ownership notes
- reuse versus extraction table
- validation plan for helper changes

### Completion Criteria
- selected helper work has a clear family owner
- helper behavior is test-support only unless explicitly documented otherwise
- code-touch quality gates are known before implementation

---

## Day 5: Oracle Helper Extraction Batch 1

**Title:** Helper Batch 1
**Theme:** Extract or consolidate the first reusable direct-solver oracle
helper without changing solver behavior
**Time estimate:** 12 hours

### Tasks
1. Implement the selected helper extraction from Day 4.
2. Preserve existing direct-solver test behavior and public solver APIs.
3. Update affected tests to use the helper where it reduces duplication or
   proof-owner ambiguity.
4. Add focused helper self-checks or failure-path checks where appropriate.
5. Record implementation evidence and focused validation results.

### Deliverables
- first oracle helper extraction
- updated direct-solver tests using the helper
- focused helper validation notes
- Day 5 implementation artifact

### Completion Criteria
- existing test behavior is preserved
- helper failure behavior is deterministic and visible
- required focused checks pass before continuing

---

## Day 6: Helper Extraction Closeout & Rerank

**Title:** Helper Closeout
**Theme:** Validate the helper extraction and rerank solver expansion lanes
with the new proof ownership model
**Time estimate:** 12 hours

### Tasks
1. Run focused tests for every test file touched by the helper extraction.
2. Re-check direct-solver evidence gaps after the extraction.
3. Confirm whether LDLT/Cholesky or LU/QR/SVD work should consume the next
   implementation slot.
4. Record any helper follow-up that should be deferred rather than expanded.
5. Write the helper closeout and rerank artifact.

### Deliverables
- focused validation results
- updated expansion ranking
- helper residual queue
- Day 6 closeout artifact

### Completion Criteria
- helper extraction is validated before solver evidence expands
- remaining expansion lane is ranked by evidence value and implementation risk
- no broad family-comparison claim is introduced

---

## Day 7: LDLT/Cholesky Oracle Boundary Freeze

**Title:** CSC Oracle Boundary
**Theme:** Freeze the highest-value CSC direct-family oracle expansion before
adding new coverage
**Time estimate:** 12 hours

### Tasks
1. Re-read Cholesky and LDLT CSC evidence, Epic 9 lanes, and Sprint 98 proof
   artifacts.
2. Select the highest-value CSC direct-family expansion based on the Day 2 and
   Day 6 rankings.
3. Define fixture keys, expected solution/residual behavior, and tolerances.
4. Define failure-mode or skip behavior for unsupported fixture classes.
5. Write the LDLT/Cholesky oracle boundary artifact.

### Deliverables
- selected CSC direct-family oracle lane
- fixture and tolerance definition
- expected failure or unsupported-case notes
- focused validation plan

### Completion Criteria
- selected lane improves oracle depth beyond existing bounded evidence
- fixture expectations are explicit before implementation
- unsupported cases are not silently counted as passed correctness proof

---

## Day 8: LDLT/Cholesky Oracle Expansion Batch

**Title:** CSC Oracle Batch
**Theme:** Add the selected CSC direct-family oracle expansion and focused
failure-mode coverage
**Time estimate:** 12 hours

### Tasks
1. Implement the selected LDLT/Cholesky oracle or fixture expansion.
2. Reuse Day 5 helper work where it reduces duplicated comparison logic.
3. Add positive correctness checks with named fixture expectations.
4. Add failure-mode or unsupported-case checks if selected by Day 7.
5. Record implementation evidence and focused validation results.

### Deliverables
- LDLT/Cholesky oracle expansion
- named fixture tests
- focused failure-mode coverage where selected
- Day 8 implementation artifact

### Completion Criteria
- new direct-family evidence passes focused validation
- failure behavior is deterministic and documented
- no every-fixture or every-CSC-family claim is introduced

---

## Day 9: LDLT/Cholesky Closeout & LU/QR/SVD Rerank

**Title:** CSC Closeout
**Theme:** Close the CSC direct-family expansion and choose the LU/QR/SVD
evidence lane
**Time estimate:** 12 hours

### Tasks
1. Run focused LDLT/Cholesky tests and any helper checks touched by Day 8.
2. Compare new evidence against the Day 7 boundary criteria.
3. Update proof-owner and residual notes for the CSC direct-family lane.
4. Rerank LU, QR, and SVD candidates using Day 2 audit and Day 3 taxonomy.
5. Write the CSC closeout and LU/QR/SVD rerank artifact.

### Deliverables
- CSC direct-family validation summary
- proof-owner and residual notes
- LU/QR/SVD expansion choice
- Day 9 closeout artifact

### Completion Criteria
- CSC direct-family work is validated before changing another family
- LU/QR/SVD lane is selected from explicit criteria
- residual CSC work is deferred rather than folded into the next batch

---

## Day 10: LU/QR/SVD Oracle Boundary Freeze

**Title:** General Solver Boundary
**Theme:** Freeze the selected LU, QR, or SVD oracle and failure-mode expansion
before implementation
**Time estimate:** 12 hours

### Tasks
1. Re-read existing LU, QR, and SVD tests and public solver guidance.
2. Select the highest-value expansion lane from Day 9.
3. Define fixture keys, expected outputs, tolerances, and failure behavior.
4. Identify whether dense-reference, external-helper, or internal invariant
   comparison is appropriate.
5. Write the LU/QR/SVD oracle boundary artifact.

### Deliverables
- selected LU/QR/SVD oracle lane
- fixture and tolerance definition
- failure-mode expectation table
- focused validation plan

### Completion Criteria
- selected lane has a measurable correctness or failure-mode outcome
- fixture expectations are concrete enough to implement
- public trust-boundary implications are recorded

---

## Day 11: LU/QR/SVD Oracle Expansion Batch

**Title:** General Solver Batch
**Theme:** Add the selected LU/QR/SVD oracle or failure-mode coverage
**Time estimate:** 12 hours

### Tasks
1. Implement the selected LU/QR/SVD oracle expansion.
2. Reuse or extend helper logic only within the Day 10 boundary.
3. Add named positive and expected-failure fixture checks as selected.
4. Keep public solver APIs unchanged unless the boundary explicitly approved
   documentation-only clarification.
5. Record implementation evidence and focused validation results.

### Deliverables
- LU/QR/SVD oracle or failure-mode expansion
- named fixture tests
- focused validation results
- Day 11 implementation artifact

### Completion Criteria
- selected tests pass focused validation
- failure-mode behavior is deterministic
- no unsupported broad LU/QR/SVD reliability claim is introduced

---

## Day 12: Direct Solver Guidance Update

**Title:** Solver Guidance
**Theme:** Update direct-solver selection, capability, failure, and trust
boundary documentation from validated Sprint 102 evidence
**Time estimate:** 12 hours

### Tasks
1. Review public README, tutorial, examples, and maintainer documentation for
   direct-solver selection guidance.
2. Update wording only where Sprint 102 evidence supports the claim.
3. Document capability boundaries, expected failures, and trust levels for
   touched direct-solver families.
4. Preserve non-claims around broad solver superiority and direct compressed
   solver APIs.
5. Write the direct-solver guidance update artifact.

### Deliverables
- updated direct-solver guidance where justified
- trust-boundary wording table
- public/non-public claim notes
- Day 12 documentation artifact

### Completion Criteria
- documentation matches implemented and validated evidence
- no unsupported solver-family-wide claim is added
- future Sprint 103 comparison work has clear public wording inputs

---

## Day 13: Full Validation & Evidence Reconciliation

**Title:** Validation
**Theme:** Run final required validation and reconcile Sprint 102 evidence,
claims, and residuals before closeout
**Time estimate:** 12 hours

### Tasks
1. Run the full required quality chain if any `.c` or `.h` files changed:
   `make format && make lint && make test`.
2. Run focused helper, oracle, documentation, and CMake checks required by
   touched surfaces.
3. Reconcile new oracle evidence against the Day 3 fixture taxonomy and Sprint
   100 solver evidence template.
4. Record earned, deferred, and non-claim states for Sprint 102.
5. Write the validation and evidence reconciliation artifact.

### Deliverables
- full validation results
- focused direct-solver validation notes
- earned/deferred/non-claim state table
- Sprint 103 dependency notes

### Completion Criteria
- all required checks pass before closeout
- direct-solver claims are tied to named tests, fixtures, or helpers
- no Sprint 103 dependency is left implicit

---

## Day 14: Sprint 102 Closeout & Handoff

**Title:** Closeout
**Theme:** Close Sprint 102 with validated direct-solver oracle evidence and a
clear handoff to Sprint 103
**Time estimate:** 12 hours

### Tasks
1. Confirm every Sprint 102 project-plan item has a deliverable.
2. Write the Sprint 102 closeout artifact and artifact index.
3. Record Sprint 103 solver-family comparison prerequisites created or
   deferred by Sprint 102.
4. Record residual direct-solver, fixture, helper, documentation, or test
   follow-up.
5. Prepare retrospective inputs and final validation notes.

### Deliverables
- Sprint 102 closeout artifact
- complete artifact index
- Sprint 103 handoff requirements
- retrospective input and residual queue

### Completion Criteria
- Sprint 102 artifacts are complete and internally consistent
- validation requirements are satisfied or explicitly blocked
- Sprint 103 can start from named direct-solver oracle evidence and fixture
  taxonomy rules
