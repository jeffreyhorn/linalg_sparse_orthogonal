# Sprint 151 Plan: Partial-SVD Maintained Corpus Family Expansion

**Sprint Duration:** 14 days
**Goal:** Close a broader but still bounded partial-SVD corpus family beyond
the single Sprint 140 fixture. This sprint implements the Sprint 151 section of
`docs/planning/EPIC_13/PROJECT_PLAN.md`.

**Starting Point:** Sprint 151 begins from:
- Sprint 147 corpus evidence gate complete
- Sprint 140 partial-SVD fixture-local closure available
- Sprint 150 QR corpus/report lessons available and merged
- corpus schema, expected-result rows, report-index normalization, and
  generated-local oracle/report command patterns available
- existing partial-SVD implementation, tests, helpers, corpus rows, and claim
  boundaries available for review

The sprint must:
- select a small set of partial-SVD fixture families for complete closure
- define singular-value, projector, vector residual, ordering, tolerance,
  sparse-output, and convergence semantics before adding rows
- add source-controlled fixture, generator, expected, claim-scope,
  support-tier, and non-claim rows for the selected families
- add focused partial-SVD corpus proof-owner tests without broad raw-vector
  identity claims
- extend oracle/report generation and normalized report-index handling for the
  selected partial-SVD families
- align SVD docs, solver-selection docs, tutorial/cookbook references, corpus
  docs, and maintainer guidance with the bounded partial-SVD claim
- run corpus schema, focused SVD tests, oracle/report checks, and full quality
  gates if `.c` or `.h` files change
- leave Sprint 152 a clean generated-report handoff

**End State:** Sprint 151 leaves behind:
- selected and closed maintained partial-SVD corpus family
- source-controlled partial-SVD fixture metadata and comparison semantics
- focused partial-SVD corpus proof-owner tests
- partial-SVD oracle/report rows and normalized report-index updates
- updated partial-SVD claim boundaries and documentation
- Sprint 152 generated-report handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 151 project-plan estimate.

---

## Day 1: Sprint Intake And Partial-SVD Baseline

**Title:** SVD Intake
**Theme:** Establish Sprint 151 scope, artifact structure, and the current
partial-SVD corpus/test/report baseline
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 151 section of
   `docs/planning/EPIC_13/PROJECT_PLAN.md`.
2. Review Sprint 140 partial-SVD artifacts, Sprint 147 corpus evidence
   artifacts, and Sprint 150 QR corpus/report handoff notes.
3. Create Sprint 151 working notes and artifact directory structure.
4. Inventory existing partial-SVD source files, public headers, tests, helper
   files, examples, corpus manifests, generators, expected outputs, and report
   rows.
5. Identify current partial-SVD claim boundaries, tolerances, non-claims, and
   proof owners.
6. Define stop conditions for raw-vector identity claims, sign/orientation
   parity claims, stale generated reports, unowned fixtures, and unsupported
   platform/package/performance inference.

### Deliverables
- Sprint 151 working-notes baseline
- artifact directory structure
- partial-SVD source/test/corpus/report inventory
- current partial-SVD claim-boundary snapshot
- stop-condition register

### Completion Criteria
- Sprint 151 scope is tied to current repository files and prior sprint
  handoffs
- every current partial-SVD proof surface has an owner or is marked unowned
- stop conditions are explicit before fixture-family selection begins

---

## Day 2: Partial-SVD Family Candidate Audit

**Title:** Family Audit
**Theme:** Audit candidate partial-SVD fixture families and decide which ones
are bounded enough for complete closure
**Time estimate:** 12 hours

### Tasks
1. Inspect repeated-spectrum partial-SVD cases and current coverage.
2. Inspect rank-deficient rectangular partial-SVD cases and current coverage.
3. Inspect sparse low-rank output cases and current coverage.
4. Inspect convergence, tight-budget, and fail-closed partial-SVD cases and
   current coverage.
5. Identify generator, fixture, expected-output, tolerance, oracle, report, and
   documentation needs for each candidate family.
6. Score each family by closure value, implementation risk, report readiness,
   and claim-boundary clarity.

### Deliverables
- partial-SVD family candidate list
- per-family coverage and gap table
- generator/fixture/expected-output needs
- tolerance and oracle-readiness notes
- family selection risk register

### Completion Criteria
- candidate families are compared with concrete repository evidence
- each family has a closure/risk score
- family-selection inputs are ready for Day 3 without implementation bias

---

## Day 3: Family Selection And Claim Scope

**Title:** Family Selection
**Theme:** Select partial-SVD fixture families for complete closure and define
their claim scopes, non-claims, and rollback rules
**Time estimate:** 12 hours

### Tasks
1. Select a small set of partial-SVD fixture families for Sprint 151 closure.
2. Define claim scope for each selected family.
3. Define explicit non-claims, including raw singular-vector identity, sign,
   orientation, basis ordering, broad optimality, broad convergence, platform,
   package, ABI, performance, and state-of-the-art claims.
4. Map selected families to fixture rows, generator rows, expected rows,
   comparison semantics, proof-owner tests, report rows, and documentation
   updates.
5. Define rollback rules for unstable tolerances, ambiguous repeated-spectrum
   semantics, generator drift, convergence instability, or focused-test
   failures.
6. Write the family-selection and claim-scope artifact.

### Deliverables
- selected partial-SVD family list
- claim-scope and non-claim register
- implementation map for Days 4-11
- rollback criteria
- Sprint 151 family decision record

### Completion Criteria
- selected families can be fully closed within Sprint 151
- every claim has a matching proof owner and report/update owner
- unsupported partial-SVD claims are explicit before comparison-contract work
  starts

---

## Day 4: Comparison Contract Design

**Title:** Contract Design
**Theme:** Define subspace-safe comparison semantics before metadata or tests
expand the claim surface
**Time estimate:** 12 hours

### Tasks
1. Review current partial-SVD expected-result row kinds and oracle comparison
   behavior.
2. Define singular-value comparison semantics and ordering assumptions for the
   selected families.
3. Define projector and subspace-distance semantics for repeated or
   degenerate singular subspaces.
4. Define vector residual semantics for `A v ~= sigma u`, `A^T u ~= sigma v`,
   and orthogonality checks.
5. Define sparse-output comparison semantics for low-rank sparse output
   fixtures.
6. Define convergence/fail-closed semantics for tight-budget and recovery
   cases.

### Deliverables
- singular-value comparison contract
- projector/subspace comparison contract
- vector residual and orthogonality contract
- sparse-output comparison contract
- convergence/fail-closed comparison contract

### Completion Criteria
- comparison semantics are numerically meaningful and bounded
- raw-vector identity and sign/orientation parity are explicitly rejected
- each selected family has a concrete expected-result kind and tolerance

---

## Day 5: Fixture Metadata Design

**Title:** Metadata Design
**Theme:** Design source-controlled fixture, generator, expected, tolerance,
support-tier, claim-scope, and non-claim rows for the selected families
**Time estimate:** 12 hours

### Tasks
1. Inspect current corpus manifest schemas and partial-SVD-related rows.
2. Design fixture rows for the selected partial-SVD families.
3. Design generator rows and deterministic regeneration commands.
4. Design expected-result rows and tolerance fields based on the Day 4
   comparison contract.
5. Design claim-scope, support-tier, and non-claim rows for each selected
   family.
6. Verify the design fits existing corpus schemas or identify narrowly scoped
   schema extensions.

### Deliverables
- fixture-row design
- generator-row design
- expected-output and tolerance design
- claim-scope/support-tier/non-claim design
- schema compatibility notes

### Completion Criteria
- metadata design fits existing corpus schemas or names required schema changes
- each selected family has complete planned metadata coverage
- regeneration and validation commands are identified before rows are edited

---

## Day 6: Fixture Metadata Batch

**Title:** Metadata Batch
**Theme:** Add the selected partial-SVD family metadata rows and deterministic
corpus entries
**Time estimate:** 12 hours

### Tasks
1. Add fixture rows for the selected partial-SVD families.
2. Add generator rows and deterministic command metadata.
3. Add expected-output rows and tolerance metadata.
4. Add claim-scope, support-tier, and non-claim rows.
5. Update deterministic corpus generator validation as needed.
6. Run corpus schema validation and normalize any affected report indexes.

### Deliverables
- source-controlled partial-SVD fixture metadata rows
- partial-SVD generator metadata rows
- partial-SVD expected-output and tolerance rows
- partial-SVD claim-scope/support-tier/non-claim rows
- schema validation result

### Completion Criteria
- selected partial-SVD families have complete source-controlled metadata
- corpus schema validation passes
- report-index normalization remains stable or planned updates are explicit

---

## Day 7: Oracle And Expected Data Implementation

**Title:** Oracle Data
**Theme:** Implement deterministic partial-SVD oracle inputs and expected data
for the selected families
**Time estimate:** 12 hours

### Tasks
1. Update or add partial-SVD oracle-generation inputs for the selected
   families.
2. Generate or refresh expected outputs deterministically.
3. Verify singular values, projectors, vector residuals, sparse-output values,
   and convergence/fail-closed expected data according to the Day 4 contract.
4. Record generation commands and any seed/tolerance decisions.
5. Run schema/report checks affected by expected-data changes.
6. Write the oracle data implementation artifact.

### Deliverables
- generated or refreshed partial-SVD expected data
- recorded generation commands
- seed and tolerance notes
- schema/report validation result
- oracle implementation artifact

### Completion Criteria
- expected data is deterministic and source-controlled as needed
- generation commands are reproducible
- expected data matches the selected comparison semantics

---

## Day 8: Proof-Owner Test Design

**Title:** Test Design
**Theme:** Design focused partial-SVD corpus proof-owner tests and helper
cleanup without broad raw-vector claims
**Time estimate:** 12 hours

### Tasks
1. Inspect current `tests/test_svd_partial_corpus.c`,
   `tests/test_svd.c`, and partial-SVD helper files.
2. Decide whether to extend the existing focused corpus proof owner or split a
   narrower helper/test surface.
3. Design focused tests for singular-value, projector/subspace, vector
   residual, sparse-output, and convergence/fail-closed semantics.
4. Design fixture-key-oriented diagnostics for failure triage.
5. Identify helper cleanup needed to avoid duplication or monolithic test
   growth.
6. Map each selected expected-result row to an executable proof assertion.

### Deliverables
- focused proof-owner test design
- helper cleanup plan
- fixture-key-oriented diagnostic plan
- expected-row-to-assertion map
- affected-test validation plan

### Completion Criteria
- every selected claim has a focused executable proof design
- raw-vector identity and sign/orientation parity remain non-claims
- implementation can proceed without broad monolithic test expansion

---

## Day 9: Proof-Owner Test Implementation

**Title:** Test Implementation
**Theme:** Implement focused partial-SVD corpus tests and helper cleanup
**Time estimate:** 12 hours

### Tasks
1. Implement focused partial-SVD corpus proof-owner tests.
2. Implement helper cleanup or extraction selected on Day 8.
3. Add fixture-key-oriented diagnostics for selected families.
4. Run focused partial-SVD corpus tests and affected SVD tests.
5. Run schema/oracle/report checks affected by test implementation.
6. If `.c` or `.h` files changed, run `make format && make lint && make test`.

### Deliverables
- focused partial-SVD corpus proof-owner tests
- helper cleanup changes
- focused test output
- affected-test validation result
- quality-gate status

### Completion Criteria
- focused tests pass locally
- selected expected-result rows have executable proof coverage
- required full C quality gates pass when code/header files change

---

## Day 10: Report Integration Design

**Title:** Report Design
**Theme:** Design partial-SVD oracle/report rows and normalized report-index
handling for the selected families
**Time estimate:** 12 hours

### Tasks
1. Inspect current report-family contracts and partial-SVD report rows.
2. Define generated-local partial-SVD oracle row expectations for selected
   fixtures.
3. Define source-controlled report-index expectations for corpus fixture,
   generator, expected, and generated-local oracle rows.
4. Define freshness rules and stale generated-output detection.
5. Define non-claims for generated-local report rows.
6. Write the report integration design artifact.

### Deliverables
- partial-SVD report row design
- normalized report-index expectations
- generated-local freshness rules
- stale-output handling plan
- report non-claim register

### Completion Criteria
- selected families have report/index coverage design
- generated-local rows remain bounded to local command evidence
- stale generated-output risks are addressed before implementation

---

## Day 11: Report Integration Implementation

**Title:** Report Integration
**Theme:** Extend oracle/report generation and normalized report-index handling
for the selected partial-SVD families
**Time estimate:** 12 hours

### Tasks
1. Update partial-SVD oracle/report generation for selected families.
2. Reuse or extend generated-output cleanup so stale ignored `build/` files do
   not contaminate report-index normalization.
3. Generate local partial-SVD oracle/report outputs.
4. Run normalized report-index checks for corpus and oracle families.
5. Run freshness checks and record advisory warning boundaries.
6. Update report-family contracts only if selected rows require it.

### Deliverables
- partial-SVD oracle/report generation updates
- generated-output cleanup confirmation
- local report manifest expectations
- normalized report-index validation result
- freshness validation result

### Completion Criteria
- selected partial-SVD families generate expected local report rows
- report-index normalization passes
- generated-local evidence is recorded without becoming source-controlled
  release proof

---

## Day 12: Documentation Alignment

**Title:** Docs Alignment
**Theme:** Align SVD docs, solver-selection docs, corpus docs, cookbook
guidance, and maintainer guidance with the bounded partial-SVD claim
**Time estimate:** 12 hours

### Tasks
1. Update README and solver-selection guidance for the selected partial-SVD
   family boundary.
2. Update SVD algorithm documentation with comparison semantics, proof owners,
   oracle commands, report expectations, and non-claims.
3. Update cookbook/tutorial references to use bounded fixture-local wording.
4. Update corpus docs and oracle-schema docs with selected families,
   expected-result kinds, generated-local rows, and validation commands.
5. Update maintainer guidance with proof owners, stale-report rules, and
   future update rules.
6. Run stale-claim and whitespace searches over current docs.

### Deliverables
- updated README/solver-selection docs
- updated SVD algorithm guidance
- updated cookbook/tutorial references
- updated corpus and oracle-schema docs
- updated maintainer guidance
- stale-claim and whitespace search results

### Completion Criteria
- docs name the selected partial-SVD family scope accurately
- raw-vector identity, broad convergence, platform/package/ABI, performance,
  and state-of-the-art claims remain non-claims
- maintainer guidance identifies proof owners and future update rules

---

## Day 13: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run schema, focused proof-owner, oracle/report, documentation, and
quality-gate validation before closeout
**Time estimate:** 12 hours

### Tasks
1. Run corpus schema validation.
2. Run focused partial-SVD proof-owner tests and affected existing SVD tests.
3. Run oracle generation or validation commands.
4. Run report-index normalization checks.
5. Run documentation whitespace and stale-claim searches.
6. Run `make format && make lint && make test` if `.c` or `.h` files changed.

### Deliverables
- integrated validation log
- schema validation result
- focused partial-SVD test result
- oracle/report validation result
- quality-gate status

### Completion Criteria
- all required local validation commands and results are recorded
- any `.c` or `.h` change has full quality-gate evidence
- unresolved failures are fixed or explicitly escalated before closeout

---

## Day 14: Closeout And Sprint 152 Handoff

**Title:** Closeout
**Theme:** Finalize Sprint 151 artifacts, residuals, validation status, and
the Sprint 152 generated-report handoff
**Time estimate:** 12 hours

### Tasks
1. Finalize `WORKING_NOTES.md` with day-by-day completion notes and validation
   status.
2. Finalize all Sprint 151 artifacts and ensure links point to current paths.
3. Prepare Sprint 151 retrospective inputs: completed partial-SVD families,
   validation, claim changes, residuals, and follow-up risks.
4. Write the Sprint 152 generated-report handoff.
5. Run final `git status`, whitespace, stale-reference, schema, and
   report-index checks.
6. Record closeout summary.

### Deliverables
- finalized Sprint 151 working notes
- complete Sprint 151 artifact set
- Sprint 151 residual and validation summary
- Sprint 152 generated-report handoff
- final closeout checklist

### Completion Criteria
- Sprint 151 partial-SVD family closure is ready for retrospective
- residuals are explicit and assigned to later sprint candidates
- branch is clean except for intentional Sprint 151 changes
- generated-report evidence boundary is clear
- Sprint 152 generated-report handoff is prepared
