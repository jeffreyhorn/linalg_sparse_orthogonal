# Sprint 140 Plan: Partial-SVD Edge-Case & Convergence Residual Closure

**Sprint Duration:** 14 days
**Goal:** Completely close the selected partial-SVD residual with edge-case
fixtures, convergence-budget proof, oracle semantics, and docs. This sprint
implements the Sprint 140 section of
`docs/planning/EPIC_12/PROJECT_PLAN.md`.

**Starting Point:** Sprint 140 begins from:
- the Sprint 138 maintained corpus architecture, schema, oracle/report command,
  and skip/defer semantics
- the Sprint 139 QR fixture-local closure pattern, proof-owner split,
  residual/subspace-safe comparison rules, and stale-report guidance
- existing SVD and partial-SVD tests, helpers, examples, solver docs, and
  maintainer claim boundaries
- the Sprint 139 handoff for partial-SVD clustered/repeated singular-value and
  rank-deficient range-projector follow-through

The sprint must:
- re-audit partial-SVD residuals and choose one bounded residual to close fully
- add deterministic edge-case corpus fixtures for the selected residual family
- implement comparison semantics for singular values, vectors, subspaces,
  ordering, tolerances, convergence budgets, and skips
- add focused convergence-budget tests without masking non-convergence
- clean up only the proof-owner/helper surfaces needed for maintainability
- run focused SVD/partial-SVD/corpus validation and required quality gates
- update SVD docs, solver-selection wording, non-claims, and Sprint 141
  report-index handoff requirements

**End State:** Sprint 140 leaves behind:
- closed priority partial-SVD residual
- partial-SVD edge-case corpus fixture batch
- partial-SVD oracle comparison row(s)
- convergence-budget proof owner
- updated partial-SVD docs and claim boundaries
- validation evidence for touched surfaces
- Sprint 141 report normalization handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 140 project-plan estimate.

---

## Day 1: Partial-SVD Residual Intake

**Title:** Residual Intake
**Theme:** Establish Sprint 140 scope, inherited SVD/partial-SVD evidence, and
closure criteria before selecting the priority residual
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 140 section of
   `docs/planning/EPIC_12/PROJECT_PLAN.md`.
2. Review Sprint 138 corpus/oracle artifacts and Sprint 139 QR closure,
   validation, and handoff artifacts.
3. Inventory current SVD and partial-SVD tests, helpers, examples,
   solver-selection docs, algorithm docs, corpus rows, and maintainer guidance.
4. Create Sprint 140 working notes and artifact directory structure.
5. Map Sprint 140 Items 1-7 to day-level owners.
6. Record initial claim boundaries, non-claims, validation expectations, and
   stop conditions for partial-SVD work.

### Deliverables
- Sprint 140 working-notes baseline
- artifact directory structure
- inherited partial-SVD evidence inventory
- item-to-day owner map
- initial closure criteria, non-claim register, and validation expectations

### Completion Criteria
- every Sprint 140 project-plan item has a day-level owner
- inherited SVD/partial-SVD/corpus evidence is visible before fixture or code
  changes begin
- closure criteria distinguish earned fixture-local evidence from broad
  partial-SVD non-claims

---

## Day 2: Partial-SVD Residual Reaudit

**Title:** Residual Reaudit
**Theme:** Re-rank partial-SVD residual candidates across numerical risk,
comparison ambiguity, convergence behavior, and closure feasibility
**Time estimate:** 12 hours

### Tasks
1. Review partial-SVD residual candidates for rank, repeated singular values,
   clustered singular values, near-zero values, rectangularity, convergence
   budget, ordering, vectors, and subspaces.
2. Compare each candidate against current tests, helper fixtures, corpus
   architecture, docs, examples, and known non-claims.
3. Score candidates by user-facing risk, determinism, fixture complexity,
   validation cost, and ability to close fully in Sprint 140.
4. Identify residuals that require broad external-library parity, optional
   SuiteSparse data, performance proof, or report-index normalization and keep
   them out of scope.
5. Draft the residual ranking artifact.
6. Record the selected closure candidate, backup candidate, and defer reasons.

### Deliverables
- partial-SVD residual ranking table
- evidence and gap map for each candidate
- selected priority residual
- backup candidate notes
- out-of-scope residual list

### Completion Criteria
- the selected residual can be closed without broadening unsupported
  partial-SVD claims
- lower-priority residuals have explicit defer reasons
- fixture and oracle design can proceed from one bounded behavior family

---

## Day 3: Closure Design

**Title:** Closure Design
**Theme:** Define the selected partial-SVD behavior, edge-case fixture class,
comparison semantics, and proof-owner boundary
**Time estimate:** 12 hours

### Tasks
1. Define the selected residual's success, diagnostic failure, tolerance, and
   convergence-budget boundaries.
2. Choose deterministic fixture dimensions, sparsity patterns, rank profile,
   repeated or clustered singular values, scaling, and rectangularity.
3. Decide whether comparisons use singular values, vector residuals,
   projectors, subspace angles, ordering constraints, or mixed metrics.
4. Define how sign, scale, basis rotation, repeated singular-value ambiguity,
   and partial convergence are interpreted.
5. Define the focused proof owner and helper ownership boundary.
6. Write the closure design artifact.

### Deliverables
- selected residual closure contract
- fixture class and expected behavior definition
- comparison semantics and tolerance model
- convergence-budget interpretation rules
- proof-owner boundary and non-claim list

### Completion Criteria
- the selected residual has unambiguous fixture-local success and failure
  semantics
- comparison metrics avoid raw vector identity when basis ambiguity is valid
- proof ownership is scoped before implementation begins

---

## Day 4: Edge-Case Fixture Batch Design

**Title:** Fixture Design
**Theme:** Specify deterministic partial-SVD edge-case fixtures and expected
rows for the selected residual family
**Time estimate:** 12 hours

### Tasks
1. Design deterministic generated or stored fixture rows for the selected
   partial-SVD residual family.
2. Define expected singular values, rank, vector residuals, projector/subspace
   metrics, convergence-budget outcomes, and skip/defer rows as needed.
3. Assign fixture keys, generator keys, expected-result row IDs, support tier,
   tolerance kinds, and claim scopes.
4. Define canonical text, hash, seed, and regeneration policy.
5. Review fixtures for overclaiming risk, basis ambiguity, and portability.
6. Write the fixture batch design artifact.

### Deliverables
- partial-SVD edge-case fixture batch specification
- expected-result row design
- generator and hash policy
- fixture-local claim and non-claim wording
- implementation checklist

### Completion Criteria
- every planned fixture has precise expected partial-SVD behavior
- expected rows are ready for source-controlled corpus implementation
- no fixture implies broad SVD, partial-SVD, SuiteSparse, or performance parity

---

## Day 5: Edge-Case Fixture Batch Implementation

**Title:** Fixture Implementation
**Theme:** Add the deterministic partial-SVD fixture rows, generator metadata,
and expected-result rows under the maintained corpus architecture
**Time estimate:** 12 hours

### Tasks
1. Add or update corpus fixture manifest rows for the selected partial-SVD
   fixtures.
2. Add or update generator metadata, canonical hashes, and expected-result
   rows.
3. Update corpus schemas or validation logic only if new fields are genuinely
   required.
4. Preserve optional-data skip/defer semantics and generated-output ignore
   policy.
5. Run corpus schema and TSV consistency checks.
6. Write the fixture implementation artifact.

### Deliverables
- source-controlled partial-SVD fixture rows
- generator metadata and expected-result rows
- any required schema/validator updates
- corpus validation evidence
- fixture implementation artifact

### Completion Criteria
- corpus fixtures validate through maintained schema checks
- expected rows encode only fixture-local targets
- generated reports or oracle rows are not committed as pass evidence

---

## Day 6: Comparison Semantics Design

**Title:** Comparison Design
**Theme:** Define oracle comparison behavior for singular values, vectors,
subspaces, ordering, tolerances, convergence budget, and skips
**Time estimate:** 12 hours

### Tasks
1. Choose the oracle reference approach for the selected partial-SVD residual.
2. Define singular-value comparison, vector residual comparison,
   projector/subspace comparison, and ordering interpretation.
3. Define convergence-budget statuses, failure classes, partial-result
   handling, and stale-report implications.
4. Define tolerance scaling for repeated, clustered, near-zero, rectangular,
   and rank-deficient fixture cases.
5. Define support-tier and optional-data interpretation rules.
6. Write the comparison semantics design artifact.

### Deliverables
- partial-SVD comparison semantics design
- oracle row field and status mapping
- convergence-budget and partial-result policy
- tolerance and ordering rules
- non-claim and skip/defer interpretation notes

### Completion Criteria
- oracle semantics support the selected residual without broad claims
- convergence-budget failures cannot be misread as pass evidence
- repeated/clustered singular-vector ambiguity is handled explicitly

---

## Day 7: Comparison Semantics Implementation

**Title:** Oracle Implementation
**Theme:** Implement partial-SVD oracle comparison row(s) and report output for
the selected residual
**Time estimate:** 12 hours

### Tasks
1. Update the corpus oracle runner or add a focused partial-SVD oracle path for
   the selected residual.
2. Emit oracle rows with expected/observed values, tolerance semantics,
   comparison statuses, failure classes, support tier, and non-claims.
3. Add report-index rows and manifest metadata for the new partial-SVD lane.
4. Keep default behavior backward-compatible unless the plan explicitly
   requires otherwise.
5. Run schema validation, oracle generation, report metadata checks, and
   script compile checks.
6. Write the oracle implementation artifact.

### Deliverables
- partial-SVD oracle comparison row(s)
- report-index rows and manifest metadata
- updated oracle runner or focused helper
- script and corpus validation evidence
- oracle implementation artifact

### Completion Criteria
- oracle rows are mechanically parseable and reproducible
- passing rows reflect only the selected fixture-local partial-SVD behavior
- generated reports include provenance and freshness context

---

## Day 8: Convergence-Budget Proof Design

**Title:** Proof Design
**Theme:** Design the focused partial-SVD convergence-budget proof owner and
helper split without weakening existing coverage
**Time estimate:** 12 hours

### Tasks
1. Review current `tests/test_svd.c`, partial-SVD helper headers, and existing
   budget/failure tests for ownership boundaries.
2. Decide whether to add a focused proof owner or extend an existing
   partial-SVD owner.
3. Define tests for success within budget, fail-closed behavior, diagnostic
   status, partial-result handling, and retry/recovery behavior.
4. Define helper extraction or reuse needed for maintainability.
5. Confirm existing SVD and partial-SVD coverage remains intact.
6. Write the convergence-budget proof design artifact.

### Deliverables
- convergence-budget proof-owner design
- helper ownership map
- focused test list
- build-system touch-point map
- validation plan

### Completion Criteria
- proof-owner scope is narrower than broad partial-SVD correctness
- existing SVD coverage is not weakened or silently bypassed
- build and validation requirements are explicit before implementation

---

## Day 9: Convergence-Budget Proof Implementation

**Title:** Proof Implementation
**Theme:** Implement focused partial-SVD convergence-budget tests and connect
them to maintained build/test surfaces
**Time estimate:** 12 hours

### Tasks
1. Add or update focused partial-SVD tests for the selected residual and
   convergence-budget behavior.
2. Add or extract helper functions only where they reduce real duplication or
   clarify ownership.
3. Register new tests in Make and CMake if a new proof owner is added.
4. Preserve existing SVD/partial-SVD tests and failure diagnostics.
5. Run focused SVD/partial-SVD tests and build-surface checks required by
   touched files.
6. Write the proof implementation artifact.

### Deliverables
- focused convergence-budget proof owner or updated partial-SVD tests
- helper cleanup
- Make/CMake registration updates if needed
- focused test evidence
- proof implementation artifact

### Completion Criteria
- the selected partial-SVD residual has a focused proof owner
- convergence-budget behavior is tested without masking non-convergence
- existing SVD/partial-SVD coverage remains present or explicitly transferred

---

## Day 10: Proof-Owner Cleanup

**Title:** Ownership Cleanup
**Theme:** Keep partial-SVD helper and test ownership maintainable without
unrelated refactors
**Time estimate:** 12 hours

### Tasks
1. Review new and existing helper/test boundaries after proof implementation.
2. Remove meaningful duplication introduced by fixture, oracle, or proof work.
3. Clarify helper names, fixture ownership, and test scope where needed.
4. Preserve local style, public API boundaries, and claim fences.
5. Run focused tests and formatting checks required by touched files.
6. Write the proof-owner cleanup artifact.

### Deliverables
- focused helper/test ownership cleanup
- updated ownership notes
- focused validation evidence
- cleanup artifact

### Completion Criteria
- helper ownership is easier to maintain than before Sprint 140 implementation
- cleanup does not broaden behavior or public API claims
- touched tests still pass through focused validation

---

## Day 11: Documentation Update

**Title:** Documentation Update
**Theme:** Update SVD-facing solver, algorithm, cookbook, corpus, and
maintainer documentation with earned partial-SVD wording and preserved
non-claims
**Time estimate:** 12 hours

### Tasks
1. Review solver-selection, SVD algorithm, cookbook, README, corpus, and
   maintainer guidance for partial-SVD wording.
2. Add fixture-local earned wording for the selected partial-SVD residual.
3. Update examples or cookbook notes that reference the selected behavior.
4. Preserve non-claims for broad SVD/partial-SVD correctness, raw vector
   identity, broad external-library parity, optional data, platform,
   performance, package, ABI, and state-of-the-art claims.
5. Update links to corpus/oracle/report artifacts.
6. Write the documentation update artifact.

### Deliverables
- updated partial-SVD solver documentation
- updated algorithm/cookbook/example notes if needed
- updated corpus and maintainer guidance
- non-claim wording updates
- documentation update artifact

### Completion Criteria
- public docs state only the earned fixture-local partial-SVD behavior
- unsupported behavior remains fenced by explicit non-claims
- documentation points to reproducible fixture/oracle/proof evidence

---

## Day 12: Focused Validation

**Title:** Validation Pass
**Theme:** Execute focused SVD/partial-SVD, corpus, documentation, and build
validation for all touched surfaces
**Time estimate:** 12 hours

### Tasks
1. Run corpus schema validation and partial-SVD oracle/report generation
   checks.
2. Run focused SVD/partial-SVD tests and any newly introduced test target.
3. Run source-list, Make, CMake, or build parity checks required by code or
   test ownership changes.
4. Run documentation link, whitespace, TSV consistency, generated-artifact, and
   stale-report hygiene checks.
5. If `.c` or `.h` files changed, run the required full quality gates:
   `make format && make lint && make test`.
6. Record command results, skips, failures, and rerun requirements.

### Deliverables
- focused validation command log
- partial-SVD test evidence
- corpus/oracle/report validation evidence
- docs and TSV hygiene evidence
- full quality-gate evidence when required

### Completion Criteria
- all required checks for touched surfaces pass
- any skipped supplemental check has an explicit reason and non-claim
- generated artifacts are not accidentally promoted without freshness context

---

## Day 13: Claim Closure & Sprint 141 Handoff

**Title:** Claim Closure
**Theme:** Publish the closed partial-SVD claim, remaining non-claims, and
Sprint 141 report-index handoff requirements
**Time estimate:** 12 hours

### Tasks
1. Summarize the selected residual, evidence added, and claim now closed.
2. Summarize fixture, oracle, proof-owner, convergence-budget, and
   documentation deliverables.
3. Publish remaining SVD/partial-SVD non-claims and residuals.
4. Define Sprint 141 report-index normalization, freshness, and stale-report
   handoff requirements.
5. Confirm validation evidence supports the final wording.
6. Write the claim closure and handoff artifact.

### Deliverables
- closed partial-SVD claim summary
- remaining SVD/partial-SVD non-claim list
- Sprint 141 report-index handoff
- validation-to-claim traceability table
- closeout readiness notes

### Completion Criteria
- the closed claim is traceable to fixture, oracle, proof, and validation
  evidence
- remaining non-claims are visible and not contradicted by docs
- Sprint 141 can start without rediscovering report/freshness dependencies

---

## Day 14: Sprint 140 Closeout

**Title:** Closeout
**Theme:** Finalize Sprint 140 artifacts, validation summary, and Sprint 141
handoff package
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 140 artifacts for consistency with the project-plan
   items.
2. Confirm all deliverables have source, test, corpus, documentation, or
   residual evidence.
3. Re-run required final validation for touched surfaces or cite the latest
   passing command set.
4. Update working notes with final decisions, deferred work, and validation
   status.
5. Prepare the retrospective input summary for the sprint closeout request.
6. Write the final closeout validation summary artifact.

### Deliverables
- final Sprint 140 artifact inventory
- validation summary
- residual and deferred-work summary
- retrospective input notes
- Sprint 141 handoff confirmation

### Completion Criteria
- Sprint 140 Items 1-7 are complete or explicitly deferred with reasons
- validation status is current and tied to touched surfaces
- the branch is ready for retrospective creation and pull-request packaging
