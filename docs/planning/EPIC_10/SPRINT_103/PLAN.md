# Sprint 103 Plan: Iterative, Eigensolver & SVD External Comparison

**Sprint Duration:** 14 days
**Goal:** Raise evidence quality for iterative solvers, eigensolvers, and SVD
without claiming broad parity with mature external packages prematurely. This
sprint implements the Sprint 103 section of
`docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 103 begins from:
- the Sprint 100 evidence templates, claim map, and non-claim discipline
- `docs/planning/EPIC_10/SPRINT_100/artifacts/day9-solver-comparison-template.md`
- `docs/planning/EPIC_10/SPRINT_100/artifacts/templates/solver-comparison-evidence-template.md`
- the Sprint 102 oracle helper patterns and direct-solver comparison lessons
- `docs/planning/EPIC_10/SPRINT_102/artifacts/day14-closeout-and-handoff.md`

The strongest Sprint 103 pressure is to expand comparison evidence for
iterative, eigen, and SVD families without turning bounded residual checks into
unearned package-wide parity claims. The sprint must:
- audit CG, MINRES, BiCGSTAB, eigen, thick-restart, LOBPCG, and SVD paths by
  comparison weakness and user impact
- define convergence, stagnation, tolerance, restart, preconditioning,
  residual, and rank fixture classes before adding new comparisons
- reuse Sprint 102 helper patterns where they fit, but avoid broadening helper
  scope beyond test evidence ownership
- add the highest-value iterative oracle or deterministic-reference batch
- add the highest-value eigensolver and LOBPCG comparison evidence
- extend SVD comparison evidence where fixture or reporting infrastructure is
  shared with eigensolver work
- document residual interpretation, convergence-profile limits, and explicit
  non-claims for broad external parity

**End State:** Sprint 103 leaves behind:
- an iterative/eigen/SVD comparison gap audit and ranked evidence queue
- a reusable convergence fixture taxonomy
- focused iterative solver comparison artifacts and tests
- focused eigensolver, thick-restart, and LOBPCG comparison artifacts and tests
- bounded SVD comparison follow-through where infrastructure overlaps
- clearer convergence, residual, and external-parity documentation
- validation artifacts and Sprint 104 handoff criteria

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 103 project-plan estimate.

---

## Day 1: Sprint 103 Scope & Comparison Baseline

**Title:** Scope Baseline
**Theme:** Convert the Sprint 103 project-plan section and Sprint 100/102
handoffs into one bounded comparison-evidence package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 103 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 100 evidence templates and Sprint 102 oracle helper
   closeout notes.
3. Inventory required Sprint 103 workstreams:
   - solver-family audit
   - convergence fixture design
   - iterative oracle batch
   - eigensolver oracle batch
   - SVD comparison follow-through
   - reporting and documentation
   - validation and closeout
4. Create Sprint 103 working notes and an artifacts directory.
5. Record validation expectations for docs-only, helper-touch, test-touch,
   source-touch, and workflow-touch days.

### Deliverables
- Sprint 103 workstream inventory
- working-notes baseline
- initial artifacts directory structure
- validation expectation list

### Completion Criteria
- every Sprint 103 project-plan item has day-level ownership
- Sprint 100 evidence-template rules are visible in working notes
- Sprint 102 helper reuse boundaries remain explicit

---

## Day 2: Solver Family Comparison Audit

**Title:** Family Audit
**Theme:** Rank CG, MINRES, BiCGSTAB, eigen, thick-restart, LOBPCG, and SVD
paths by comparison weakness, user impact, and validation cost
**Time estimate:** 12 hours

### Tasks
1. Inventory iterative solver tests for CG, MINRES, BiCGSTAB, GMRES-adjacent
   dependencies, preconditioning, stagnation, and tolerance behavior.
2. Inventory eigen, thick-restart, LOBPCG, and SVD tests for residual and
   external-comparison evidence.
3. Classify current evidence as internal consistency, deterministic reference,
   dense-reference, external helper, fixture corpus, property, or smoke.
4. Score each family by user value, current evidence gap, numerical risk, and
   implementation cost.
5. Write the solver-family comparison audit artifact.

### Deliverables
- solver-family evidence inventory
- comparison weakness and user-impact table
- validation-cost notes by family
- ranked Sprint 103 expansion queue

### Completion Criteria
- all named Sprint 103 solver families are classified
- expansion candidates are ranked before fixture design starts
- broad external-parity claims remain out of scope

---

## Day 3: Convergence Fixture Taxonomy

**Title:** Fixture Taxonomy
**Theme:** Define convergence, stagnation, tolerance, restart,
preconditioning, residual, and rank fixture classes before adding tests
**Time estimate:** 12 hours

### Tasks
1. Define matrix families for SPD, symmetric indefinite, nonsymmetric,
   ill-conditioned, rank-deficient, clustered-spectrum, and low-rank behavior.
2. Define convergence-profile classes for fast convergence, slow convergence,
   stagnation, tolerance sensitivity, restart sensitivity, and preconditioner
   effectiveness.
3. Map fixture classes to CG, MINRES, BiCGSTAB, eigensolver, thick-restart,
   LOBPCG, and SVD paths.
4. Define expected residual, orthogonality, singular-value, rank, and skip
   criteria for each fixture class.
5. Write the convergence fixture taxonomy artifact.

### Deliverables
- convergence fixture taxonomy
- solver-family fixture mapping
- residual and orthogonality criteria table
- expected skip and failure-mode rules

### Completion Criteria
- new comparison tests can cite a fixture taxonomy entry
- convergence behavior is separated from correctness regressions
- fixture classes support both iterative and spectral workstreams

---

## Day 4: Oracle Helper & Reporting Boundary

**Title:** Helper Boundary
**Theme:** Decide what Sprint 102 helper patterns, external helpers, and
reporting utilities are safe to reuse for iterative and spectral evidence
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 102 helper patterns for subprocess output, status handling,
   reason strings, fixture loading, and dense-reference comparison.
2. Identify repeated residual, orthogonality, convergence-profile, and
   tolerance-reporting logic in iterative/eigen/SVD tests.
3. Select the smallest helper or reporting extraction that improves evidence
   ownership without widening public APIs.
4. Define helper inputs, outputs, failure behavior, skip behavior, and focused
   validation commands.
5. Write the helper and reporting boundary artifact.

### Deliverables
- helper reuse and extraction decision record
- convergence reporting contract
- skip/error behavior table
- focused validation plan for helper changes

### Completion Criteria
- helper scope is test-support only unless explicitly documented otherwise
- status and reason behavior follows Sprint 102 conventions where reused
- implementation scope is frozen before code changes begin

---

## Day 5: Iterative Oracle Batch Design

**Title:** Iterative Design
**Theme:** Select the highest-value iterative solver comparison batch and
freeze fixtures, tolerances, and non-claims
**Time estimate:** 12 hours

### Tasks
1. Select the highest-value iterative paths from the Day 2 ranking.
2. Bind each selected path to Day 3 fixture taxonomy entries.
3. Define deterministic-reference or external-helper comparison criteria for
   residuals, iteration counts, stagnation, and preconditioner effects.
4. Define file-level implementation ownership and focused validation commands.
5. Write the iterative oracle batch design artifact.

### Deliverables
- selected iterative comparison batch
- fixture and tolerance matrix
- implementation ownership notes
- validation command list

### Completion Criteria
- iterative implementation scope fits the sprint budget
- each selected test has an explicit expected outcome
- unselected iterative follow-ups are deferred rather than silently dropped

---

## Day 6: Iterative Oracle Batch Implementation

**Title:** Iterative Batch
**Theme:** Add focused iterative solver comparisons for the highest-value
paths selected on Day 5
**Time estimate:** 12 hours

### Tasks
1. Implement the selected iterative comparison tests or helper updates.
2. Preserve existing public solver behavior and solver option semantics.
3. Add or reuse fixtures for convergence, stagnation, tolerance, restart, or
   preconditioning scenarios as designed.
4. Run focused validation for every touched iterative test file.
5. Record implementation evidence and validation results.

### Deliverables
- iterative comparison implementation
- focused iterative validation results
- updated fixture or helper notes
- Day 6 implementation artifact

### Completion Criteria
- selected iterative comparison tests pass locally
- failure and skip behavior is deterministic and documented
- no broad iterative external-parity claim is introduced

---

## Day 7: Iterative Batch Closeout & Rerank

**Title:** Iterative Closeout
**Theme:** Validate iterative comparison coverage and rerank remaining
eigen/SVD work using the updated evidence map
**Time estimate:** 12 hours

### Tasks
1. Run focused and affected-family validation for iterative changes.
2. Re-check the Day 2 evidence ranking after the iterative batch.
3. Identify any helper, fixture, or reporting debt created by iterative work.
4. Update the expansion queue for eigen, thick-restart, LOBPCG, and SVD work.
5. Write the iterative closeout and rerank artifact.

### Deliverables
- iterative evidence closeout
- updated solver-family ranking
- residual follow-up queue
- Day 7 validation artifact

### Completion Criteria
- iterative comparison work is validated before spectral implementation starts
- remaining work is ranked against actual evidence gained
- deferred iterative items have explicit ownership

---

## Day 8: Eigensolver Oracle Batch Design

**Title:** Eigen Design
**Theme:** Select focused eigen, thick-restart, and LOBPCG comparison cases
with bounded residual and orthogonality expectations
**Time estimate:** 12 hours

### Tasks
1. Select highest-value eigen, thick-restart, and LOBPCG comparison gaps from
   the updated ranking.
2. Bind each selected path to clustered-spectrum, shifted, preconditioned,
   restart, and orthogonality fixture classes where appropriate.
3. Define comparison criteria for eigenvalue error, residual norms,
   eigenvector orthogonality, convergence status, and skip behavior.
4. Define file-level implementation ownership and focused validation commands.
5. Write the eigensolver oracle batch design artifact.

### Deliverables
- selected eigen/thick-restart/LOBPCG comparison batch
- residual and orthogonality criteria matrix
- fixture ownership notes
- validation command list

### Completion Criteria
- selected spectral cases are bounded and explainable
- expected outcomes are fixture-specific rather than package-wide
- SVD overlap opportunities are identified before spectral implementation

---

## Day 9: Eigensolver Oracle Batch Implementation

**Title:** Eigen Batch
**Theme:** Add focused eigensolver, thick-restart, and LOBPCG comparison
evidence for the Day 8 cases
**Time estimate:** 12 hours

### Tasks
1. Implement selected eigen, thick-restart, and LOBPCG comparison tests or
   helper updates.
2. Preserve existing eigensolver public API behavior and option defaults.
3. Add or reuse fixtures for clustered spectra, shift-invert, restart,
   preconditioning, and orthogonality checks.
4. Run focused validation for every touched spectral test file.
5. Record implementation evidence and validation results.

### Deliverables
- spectral comparison implementation
- focused eigensolver validation results
- updated fixture or helper notes
- Day 9 implementation artifact

### Completion Criteria
- selected spectral comparison tests pass locally
- orthogonality and residual thresholds are documented
- implementation does not claim broad eigensolver parity

---

## Day 10: Spectral Closeout & SVD Scope Freeze

**Title:** Spectral Closeout
**Theme:** Validate spectral comparison work and freeze the SVD follow-through
scope that shares fixture or reporting infrastructure
**Time estimate:** 12 hours

### Tasks
1. Run focused and affected-family validation for spectral changes.
2. Re-check residual, orthogonality, and convergence evidence after the eigen
   batch.
3. Identify which SVD comparison gaps can reuse spectral fixtures or reporting
   infrastructure.
4. Freeze the SVD implementation scope and define focused validation commands.
5. Write the spectral closeout and SVD scope artifact.

### Deliverables
- spectral evidence closeout
- SVD overlap and reuse table
- selected SVD comparison scope
- Day 10 validation artifact

### Completion Criteria
- spectral implementation is validated before SVD work begins
- SVD scope is limited to shared infrastructure or highest-value gaps
- remaining spectral follow-ups are explicitly deferred

---

## Day 11: SVD Comparison Follow-Through

**Title:** SVD Follow-Through
**Theme:** Extend SVD comparison evidence where it shares fixture or reporting
infrastructure with eigensolver work
**Time estimate:** 12 hours

### Tasks
1. Implement the selected SVD comparison tests, fixture reuse, or reporting
   updates from Day 10.
2. Check singular values, reconstruction residuals, orthogonality, and
   rank-sensitive behavior according to the fixture taxonomy.
3. Preserve existing SVD public API behavior and option defaults.
4. Run focused validation for every touched SVD test file.
5. Record implementation evidence and validation results.

### Deliverables
- SVD comparison follow-through implementation
- focused SVD validation results
- fixture and reporting reuse notes
- Day 11 implementation artifact

### Completion Criteria
- selected SVD comparison tests pass locally
- SVD evidence is tied to explicit fixture classes
- SVD docs and tests avoid broad external-parity claims

---

## Day 12: Reporting & Documentation Update

**Title:** Reporting Docs
**Theme:** Document convergence-profile interpretation, residual criteria,
comparison scope, and explicit non-claims
**Time estimate:** 12 hours

### Tasks
1. Update public or maintainer documentation for convergence profiles,
   residual interpretation, and comparison evidence boundaries.
2. Explain how iterative, eigensolver, and SVD residuals should be interpreted
   across fixture classes.
3. Record where external comparisons are helper-backed, deterministic,
   dense-reference, internal consistency, or still absent.
4. Add non-claim wording for broad external package parity where evidence is
   intentionally bounded.
5. Run documentation-focused validation and record results.

### Deliverables
- updated convergence and residual documentation
- comparison evidence boundary notes
- non-claim language for broad external parity
- Day 12 documentation artifact

### Completion Criteria
- docs distinguish evidence types and solver families clearly
- public narrative does not overstate Sprint 103 proof
- docs validation passes before closeout begins

---

## Day 13: Validation & Evidence Reconciliation

**Title:** Evidence Reconciliation
**Theme:** Reconcile implemented comparisons, documentation, and claim
boundaries before sprint closeout
**Time estimate:** 12 hours

### Tasks
1. Run required quality checks for all touched `.c`, `.h`, documentation, and
   helper files.
2. Reconcile Day 2 rankings with implemented iterative, spectral, and SVD
   comparison evidence.
3. Update working notes with validation results, remaining gaps, and Sprint
   104 candidates.
4. Confirm that every implemented comparison has a matching artifact and
   documented claim boundary.
5. Write the validation and evidence reconciliation artifact.

### Deliverables
- complete validation results
- reconciled evidence map
- remaining gap and deferred-work list
- Sprint 104 candidate queue

### Completion Criteria
- required checks pass for all touched code and documentation
- every Sprint 103 deliverable is traceable to an artifact or documented
  deferral
- no unresolved claim-boundary mismatch remains

---

## Day 14: Closeout & Handoff

**Title:** Closeout Handoff
**Theme:** Package Sprint 103 artifacts, final validation, and Sprint 104
handoff criteria
**Time estimate:** 12 hours

### Tasks
1. Build the Sprint 103 artifact index and closeout summary.
2. Confirm all working notes, implemented evidence, documentation updates, and
   validation records are complete.
3. Identify Sprint 104 prerequisites, risks, and deferred comparison work.
4. Run final required checks for the branch state.
5. Write the closeout and handoff artifact.

### Deliverables
- Sprint 103 artifact index
- closeout and handoff artifact
- final validation record
- Sprint 104 prerequisite and risk list

### Completion Criteria
- Sprint 103 has a complete artifact trail from audit through closeout
- final validation passes
- Sprint 104 can start from explicit prerequisites and deferred-work ownership
