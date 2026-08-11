# Sprint 150 Plan: QR Maintained Corpus Family Expansion

**Sprint Duration:** 14 days
**Goal:** Close a broader but still bounded QR corpus family beyond the single
Sprint 139 fixture. This sprint implements the Sprint 150 section of
`docs/planning/EPIC_13/PROJECT_PLAN.md`.

**Starting Point:** Sprint 150 begins from:
- Sprint 147 corpus evidence gate complete
- Sprint 139 QR fixture-local closure available
- Sprint 149 Windows CMake install/downstream boundary documented and merged
- corpus schema, report-index normalization, and oracle/report command patterns
  available
- existing QR implementation, tests, examples, corpus rows, and claim
  boundaries available for review

The sprint must:
- select two or three QR fixture families for complete closure
- add source-controlled fixture, generator, expected, claim-scope, tolerance,
  and non-claim rows for the selected QR families
- define residual, rank, nullspace, minimum-norm, and subspace-safe oracle
  semantics without raw-basis identity claims
- add focused QR corpus proof-owner tests instead of enlarging the largest
  monolithic QR test file
- extend oracle/report generation and normalized report-index handling for the
  selected QR family
- align README, corpus docs, solver-selection docs, tutorial/cookbook
  references, and maintainer guidance with the bounded QR claim
- run corpus schema, focused QR tests, oracle/report checks, and full C gates if
  `.c` or `.h` files change
- leave Sprint 151 a clean partial-SVD handoff

**End State:** Sprint 150 leaves behind:
- selected and closed maintained QR corpus family
- source-controlled QR fixture metadata and oracle semantics
- focused QR corpus proof-owner tests
- QR oracle/report rows and normalized report-index updates
- updated QR claim boundaries and documentation
- Sprint 151 partial-SVD handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 150 project-plan estimate.

---

## Day 1: Sprint Intake And QR Baseline

**Title:** QR Intake
**Theme:** Establish Sprint 150 scope, artifact structure, and the current QR
corpus/test/report baseline
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 150 section of
   `docs/planning/EPIC_13/PROJECT_PLAN.md`.
2. Review Sprint 139 QR artifacts, Sprint 147 corpus evidence artifacts, and
   Sprint 149 handoff notes.
3. Create Sprint 150 working notes and artifact directory structure.
4. Inventory existing QR source files, public headers, tests, examples, corpus
   manifests, generators, expected outputs, and report rows.
5. Identify current QR claim boundaries, tolerances, non-claims, and proof
   owners.
6. Define stop conditions for raw-basis identity claims, unowned fixtures,
   stale reports, and unsupported Windows/package inference.

### Deliverables
- Sprint 150 working-notes baseline
- artifact directory structure
- QR source/test/corpus/report inventory
- current QR claim-boundary snapshot
- stop-condition register

### Completion Criteria
- Sprint 150 scope is tied to current repository files and prior sprint
  handoffs
- every current QR proof surface has an owner or is marked unowned
- stop conditions are explicit before fixture-family selection begins

---

## Day 2: QR Family Candidate Audit

**Title:** Family Audit
**Theme:** Audit candidate QR fixture families and decide which ones are
bounded enough for complete closure
**Time estimate:** 12 hours

### Tasks
1. Inspect candidate rank-deficient rectangular QR cases and current coverage.
2. Inspect candidate underdetermined minimum-norm QR cases and current coverage.
3. Inspect candidate reorder/COLAMD QR paths and current coverage.
4. Identify generator, fixture, expected-output, tolerance, and oracle needs
   for each candidate family.
5. Score each family by closure value, implementation risk, report readiness,
   and claim-boundary clarity.
6. Write the QR family candidate audit artifact.

### Deliverables
- QR family candidate list
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
**Theme:** Select the QR fixture families for complete closure and define their
claim scopes, non-claims, and rollback rules
**Time estimate:** 12 hours

### Tasks
1. Select two or three QR fixture families for Sprint 150 closure.
2. Define claim scope for each selected family.
3. Define explicit non-claims, including raw-basis identity and broad QR
   optimality claims.
4. Map selected families to fixture rows, generator rows, expected rows,
   oracle semantics, proof-owner tests, report rows, and documentation updates.
5. Define rollback rules for unstable tolerances, ambiguous rank/nullspace
   semantics, generator drift, or focused-test failures.
6. Write the family-selection and claim-scope artifact.

### Deliverables
- selected QR family list
- claim-scope and non-claim register
- implementation map for Days 4-10
- rollback criteria
- Sprint 150 family decision record

### Completion Criteria
- selected families can be fully closed within Sprint 150
- every claim has a matching proof owner and report/update owner
- unsupported QR claims are explicit before metadata work starts

---

## Day 4: Fixture Metadata Design

**Title:** Metadata Design
**Theme:** Design source-controlled fixture, generator, expected, tolerance,
claim-scope, and non-claim rows for the selected QR families
**Time estimate:** 12 hours

### Tasks
1. Inspect current corpus manifest schemas and QR-related rows.
2. Design fixture rows for selected QR families.
3. Design generator rows and deterministic regeneration commands.
4. Design expected-output rows and tolerance fields.
5. Design claim-scope and non-claim rows for each selected family.
6. Write the fixture metadata design artifact.

### Deliverables
- fixture-row design
- generator-row design
- expected-output and tolerance design
- claim-scope/non-claim row design
- schema compatibility notes

### Completion Criteria
- metadata design fits existing corpus schemas
- each selected family has complete planned metadata coverage
- regeneration and validation commands are identified before rows are edited

---

## Day 5: Fixture Metadata Batch

**Title:** Metadata Batch
**Theme:** Add the selected QR family metadata rows and deterministic corpus
entries
**Time estimate:** 12 hours

### Tasks
1. Add fixture rows for the selected QR families.
2. Add generator rows and deterministic command metadata.
3. Add expected-output rows and tolerance metadata.
4. Add claim-scope and non-claim rows.
5. Run corpus schema validation and normalize any affected report indexes.
6. Write the fixture metadata implementation artifact.

### Deliverables
- source-controlled QR fixture metadata rows
- QR generator metadata rows
- QR expected-output and tolerance rows
- QR claim-scope/non-claim rows
- schema validation result

### Completion Criteria
- selected QR families have complete source-controlled metadata
- corpus schema validation passes
- report-index normalization remains stable or planned updates are explicit

---

## Day 6: Oracle Semantics Design

**Title:** Oracle Design
**Theme:** Define numerical oracle semantics for residual, rank, nullspace,
minimum-norm, and subspace-safe QR comparisons
**Time estimate:** 12 hours

### Tasks
1. Review existing QR oracle behavior and report-generation commands.
2. Define residual checks for each selected QR family.
3. Define rank and nullspace checks without raw-basis identity assumptions.
4. Define minimum-norm checks for underdetermined cases.
5. Define subspace-safe comparison rules and tolerances.
6. Write the QR oracle semantics design artifact.

### Deliverables
- QR residual oracle rules
- QR rank/nullspace oracle rules
- minimum-norm oracle rules
- subspace-safe comparison rules
- tolerance rationale

### Completion Criteria
- oracle semantics are numerically meaningful and bounded
- raw-basis identity claims are explicitly rejected
- each selected family has a concrete oracle rule and tolerance

---

## Day 7: Oracle And Expected Data Implementation

**Title:** Oracle Data
**Theme:** Implement or update deterministic QR oracle generation and expected
data for the selected families
**Time estimate:** 12 hours

### Tasks
1. Update or add QR oracle-generation inputs for the selected families.
2. Generate or refresh expected outputs deterministically.
3. Verify residual, rank, nullspace, minimum-norm, and subspace-safe expected
   data according to Day 6 rules.
4. Record generation commands and any seed/tolerance decisions.
5. Run schema/report checks affected by expected-data changes.
6. Write the oracle data implementation artifact.

### Deliverables
- generated or refreshed QR expected data
- recorded generation commands
- seed and tolerance notes
- schema/report validation result
- oracle implementation artifact

### Completion Criteria
- expected data is deterministic and source-controlled as needed
- generation commands are reproducible
- expected data matches the selected oracle semantics

---

## Day 8: Proof-Owner Test Design

**Title:** Test Design
**Theme:** Design focused QR corpus proof-owner tests for the selected families
without expanding the monolithic QR test surface unnecessarily
**Time estimate:** 12 hours

### Tasks
1. Inspect existing QR tests and identify focused-test insertion points.
2. Define one or more proof-owner tests for selected QR families.
3. Define fixture loading, expected-data loading, tolerance assertions, and
   diagnostics.
4. Define failure messages for residual, rank, nullspace, minimum-norm, and
   subspace comparison failures.
5. Define platform and CI registration expectations.
6. Write the proof-owner test design artifact.

### Deliverables
- focused QR proof-owner test design
- fixture/expected-data loading plan
- assertion and diagnostic plan
- CMake/Make registration plan
- validation plan

### Completion Criteria
- proof-owner tests are scoped to selected QR families
- diagnostics identify the failing family and oracle condition
- registration plan preserves current platform and CI boundaries

---

## Day 9: Proof-Owner Test Implementation

**Title:** Test Implementation
**Theme:** Implement focused QR corpus proof-owner tests and register them in
the local build/test surfaces
**Time estimate:** 12 hours

### Tasks
1. Add focused QR corpus proof-owner test source or extend the selected focused
   test target.
2. Implement fixture and expected-data loading.
3. Implement residual, rank, nullspace, minimum-norm, and subspace-safe
   assertions as needed for selected families.
4. Register the proof-owner test in Make/CMake as appropriate.
5. Run focused build and test commands.
6. Write the proof-owner implementation artifact.

### Deliverables
- focused QR proof-owner test implementation
- build/test registration updates
- focused validation result
- implementation artifact

### Completion Criteria
- selected QR families have executable proof-owner coverage
- focused QR tests pass locally
- failures produce actionable family/oracle diagnostics

---

## Day 10: Report Integration Design

**Title:** Report Design
**Theme:** Design QR oracle/report generation and normalized report-index
updates for the selected families
**Time estimate:** 12 hours

### Tasks
1. Inspect existing QR report rows, report generators, and normalized indexes.
2. Define report outputs for the selected QR families.
3. Define report freshness and proof-owner expectations.
4. Define normalized report-index rows and non-claim wording.
5. Identify documentation surfaces that must reference the new QR report rows.
6. Write the report integration design artifact.

### Deliverables
- QR report-output design
- report freshness/proof-owner rules
- report-index row design
- non-claim wording for reports
- documentation update map

### Completion Criteria
- report integration has source-controlled owners
- report rows do not imply freshness without generated evidence
- normalized index changes are planned before implementation

---

## Day 11: Report Integration Implementation

**Title:** Report Implementation
**Theme:** Implement QR report generation, report rows, and normalized indexes
for the selected families
**Time estimate:** 12 hours

### Tasks
1. Add or update QR report generation commands and outputs.
2. Add or update normalized report-index rows.
3. Add report-family ownership and non-claim wording.
4. Run report generation or report validation commands.
5. Run report-index normalization checks.
6. Write the report integration implementation artifact.

### Deliverables
- QR report generation updates
- QR report rows
- normalized report-index updates
- report validation results
- implementation artifact

### Completion Criteria
- selected QR families have report/index coverage
- report commands and normalized rows validate locally
- report wording remains bounded to source-controlled evidence

---

## Day 12: Documentation Alignment

**Title:** Docs Alignment
**Theme:** Align corpus docs, solver-selection docs, README, tutorial/cookbook
references, and maintainer guidance with the new QR family boundary
**Time estimate:** 12 hours

### Tasks
1. Update corpus documentation for selected QR families and oracle semantics.
2. Update solver-selection documentation for QR family boundaries and
   non-claims.
3. Update README and tutorial/cookbook references as needed.
4. Update maintainer guidance for QR corpus ownership, report freshness, and
   future-family additions.
5. Run focused stale-claim and raw-basis identity wording searches.
6. Write the documentation alignment artifact.

### Deliverables
- updated corpus documentation
- updated solver-selection documentation
- updated README/tutorial/cookbook references
- updated maintainer guidance
- stale-claim search results

### Completion Criteria
- docs name the selected QR family scope accurately
- raw-basis identity and broad QR claims remain non-claims
- maintainer guidance identifies proof owners and future update rules

---

## Day 13: Integrated Validation

**Title:** Validation
**Theme:** Run integrated schema, focused QR, oracle/report, documentation, and
quality gates
**Time estimate:** 12 hours

### Tasks
1. Run corpus schema validation.
2. Run focused QR proof-owner tests and any affected existing QR tests.
3. Run oracle generation or validation commands.
4. Run report-index normalization checks.
5. Run documentation whitespace/stale-claim searches.
6. Run `make format && make lint && make test` if `.c` or `.h` files changed.

### Deliverables
- integrated validation log
- schema validation result
- focused QR test result
- oracle/report validation result
- quality-gate status

### Completion Criteria
- all required local validation commands and results are recorded
- any `.c` or `.h` change has full quality-gate evidence
- unresolved failures are fixed or explicitly escalated before closeout

---

## Day 14: Closeout And Sprint 151 Handoff

**Title:** Closeout
**Theme:** Finalize Sprint 150 artifacts, residuals, validation status, and the
Sprint 151 partial-SVD handoff
**Time estimate:** 12 hours

### Tasks
1. Finalize `WORKING_NOTES.md` with day-by-day completion notes and validation
   status.
2. Finalize all Sprint 150 artifacts and ensure links point to current paths.
3. Prepare Sprint 150 retrospective inputs: completed QR families, validation,
   claim changes, residuals, and follow-up risks.
4. Write the Sprint 151 partial-SVD handoff.
5. Run final `git status`, whitespace, stale-reference, and report-index
   checks.
6. Record closeout summary.

### Deliverables
- finalized Sprint 150 working notes
- complete Sprint 150 artifact set
- Sprint 150 residual and validation summary
- Sprint 151 partial-SVD handoff
- final closeout checklist

### Completion Criteria
- Sprint 150 QR family closure is ready for retrospective
- residuals are explicit and assigned to later sprint candidates
- branch is clean except for intentional Sprint 150 changes
