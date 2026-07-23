# Sprint 130 Plan: Partial-SVD Residual Expansion & Solver-Selection Claim Gate

**Sprint Duration:** 14 days
**Goal:** Expand or explicitly defer Sprint 124 partial-SVD residual evidence
under dedicated metric policies, then refresh public solver-selection wording
only where earned.

**Starting Point:** Sprint 130 begins from:
- Sprint 124 partial-SVD vector/subspace semantics and residual scenario matrix
- Sprint 125-129 QR and helper claim gates
- existing partial-SVD helper, fixture, dense-reference, corpus, and
  convergence-budget tests
- current public solver-selection wording and maintainer evidence tables

The sprint must:
- map completed `partial_svd_vector_residual_diag6_k2` evidence against all
  deferred partial-SVD residual lanes before adding new tests
- define rectangular, nonsymmetric, repeated-spectrum, clustered-spectrum,
  rank-deficient, SuiteSparse, low-rank optimality, and convergence-budget
  metrics before implementation
- use subspace, projection, rank/nullity, residual, and optimality metrics
  only where they are mathematically meaningful for the scenario
- treat optional corpus availability, skip behavior, diagnostics, and support
  tier as part of the evidence contract
- avoid broad solver-selection, parity, or optimality claims unless the
  bounded evidence explicitly earns them
- publish no-update rationale when solver-selection wording should remain
  workflow guidance rather than a stronger user-facing claim

**End State:** Sprint 130 leaves behind:
- partial-SVD deferred-evidence dedupe map
- rectangular and nonsymmetric residual decision package
- repeated and clustered spectrum decision package
- rank-deficient subspace decision package
- SuiteSparse and low-rank optimality decision package
- convergence-budget decision package
- solver-selection wording update or explicit no-update rationale
- final validation, non-claim, and Sprint 131 handoff notes

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 130 project-plan estimate.

---

## Day 1: Sprint Intake and Residual Dedupe Baseline

**Title:** Dedupe Intake
**Theme:** Establish Sprint 130 scope, owners, duplicate fences, and the
partial-SVD residual evidence map
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 130 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Review Sprint 124 partial-SVD vector/subspace semantics and residual
   scenario artifacts.
3. Review Sprint 125-129 QR/helper claim gates for reusable evidence-gate
   patterns and non-claim wording.
4. Inventory completed `partial_svd_vector_residual_diag6_k2` evidence and
   every deferred residual lane named in Sprint 130.
5. Create Sprint 130 working notes and artifact directory.
6. Write the sprint intake and duplicate-fence baseline.

### Deliverables
- Sprint 130 working-notes baseline
- artifact directory structure
- completed-versus-deferred residual evidence map
- item-to-day owner map
- duplicate and non-claim boundary notes

### Completion Criteria
- every Sprint 130 project-plan item has a day-level owner
- completed Sprint 124 evidence is not duplicated silently
- deferred rectangular, spectral, subspace, corpus, optimality, and convergence
  lanes are visible before new evidence is accepted

---

## Day 2: Partial-SVD Metric Policy Map

**Title:** Metric Map
**Theme:** Pin metric, tolerance, oracle, and failure-interpretation policies
before expanding partial-SVD evidence
**Time estimate:** 12 hours

### Tasks
1. Classify each deferred lane by shape, spectrum, rank, corpus, optimality,
   convergence, and solver-selection impact.
2. Define when vector residuals are meaningful versus when subspace or
   projection metrics are required.
3. Define tolerance policy for singular values, residual norms, projector
   differences, rank/nullity, and partial-result reporting.
4. Identify independent oracle requirements for dense references, analytic
   fixtures, and corpus-backed cases.
5. Define failure interpretation for numerical mismatch, skipped optional
   data, unsupported shapes, and convergence-budget exhaustion.
6. Write the partial-SVD dedupe and metric-map artifact.

### Deliverables
- residual scenario metric matrix
- tolerance and oracle policy
- failure-interpretation policy
- deferred-lane promotion checklist
- no-claim boundary for broad partial-SVD parity

### Completion Criteria
- no evidence lane can proceed without a metric, tolerance, and oracle policy
- vector equality is not used where basis orientation or multiplicity makes it
  unstable
- failure outcomes are classified before tests or docs are changed

---

## Day 3: Rectangular Residual Gate

**Title:** Rectangular Gate
**Theme:** Decide how rectangular partial-SVD residual evidence should be
measured and bounded
**Time estimate:** 12 hours

### Tasks
1. Inventory tall and wide rectangular partial-SVD candidates from current
   tests, helpers, and dense-reference scripts.
2. Separate shape behavior from symmetric, square, corpus, and convergence
   claims.
3. Define residual, singular-value, reconstruction, and optional subspace
   metrics for each candidate.
4. Select one bounded rectangular lane for Day 4 implementation, or define
   explicit deferral criteria.
5. Identify touched files, expected diagnostics, and focused validation.
6. Write the rectangular residual decision gate.

### Deliverables
- rectangular candidate table
- shape-specific metric and tolerance policy
- Day 4 implementation or deferral checklist
- validation and diagnostics plan

### Completion Criteria
- accepted rectangular evidence has distinct proof value beyond Sprint 124
- tall and wide interpretations are not conflated
- no rectangular result implies broad nonsymmetric or solver-selection parity

---

## Day 4: Rectangular Residual Evidence

**Title:** Rectangular Evidence
**Theme:** Implement or explicitly defer bounded rectangular partial-SVD
residual evidence
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 3 gate to the selected rectangular candidate.
2. Implement one bounded evidence lane only if metric, fixture, tolerance,
   oracle, and diagnostics are pinned.
3. Otherwise publish an explicit deferral package with blocker and future
   owner.
4. Keep residual, reconstruction, singular-value, and subspace claims separate.
5. Run focused partial-SVD validation for touched code, fixtures, or scripts.
6. Record evidence results and non-claims.

### Deliverables
- rectangular residual implementation or explicit deferral
- fixture and diagnostic notes
- focused validation results
- non-claim update

### Completion Criteria
- rectangular evidence is bounded, validated, and non-duplicative, or
  explicitly deferred
- touched files have focused validation
- no broad rectangular or nonsymmetric partial-SVD claim is introduced

---

## Day 5: Nonsymmetric Rectangular Gate

**Title:** Nonsym Gate
**Theme:** Decide whether nonsymmetric rectangular evidence can add trust
beyond rectangular shape coverage
**Time estimate:** 12 hours

### Tasks
1. Inventory nonsymmetric rectangular candidates and current helper support.
2. Identify which candidates need left/right singular-vector, residual,
   reconstruction, or subspace metrics.
3. Define orientation, sign, and multiplicity rules for any vector-level
   comparisons.
4. Define skip and failure interpretation for unsupported or numerically
   unstable candidates.
5. Select one bounded Day 6 lane or document why all candidates should defer.
6. Write the nonsymmetric rectangular gate.

### Deliverables
- nonsymmetric rectangular candidate table
- left/right vector and residual metric policy
- orientation and multiplicity notes
- Day 6 acceptance or deferral criteria

### Completion Criteria
- nonsymmetric behavior is not treated as already covered by rectangular shape
  evidence
- vector comparisons are allowed only under stable orientation rules
- unsupported candidates have explicit deferral blockers

---

## Day 6: Nonsymmetric Rectangular Evidence

**Title:** Nonsym Evidence
**Theme:** Implement or explicitly defer nonsymmetric rectangular partial-SVD
residual evidence
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 5 gate to the selected nonsymmetric candidate.
2. Implement one accepted evidence lane if fixture, oracle, metrics,
   tolerance, diagnostics, and non-claims are ready.
3. Otherwise write an explicit deferral package.
4. Preserve separation between nonsymmetric, rectangular, rank-deficient,
   corpus, optimality, and convergence claims.
5. Run focused partial-SVD and dense-reference validation for touched files.
6. Update evidence notes or record a no-update rationale.

### Deliverables
- nonsymmetric rectangular implementation or explicit deferral
- dense-reference or analytic oracle notes
- focused validation package
- evidence-table update or no-update rationale

### Completion Criteria
- accepted nonsymmetric evidence validates the chosen bounded behavior
- no vector or solver-selection claim exceeds the metric policy
- every deferred candidate has blocker and future-owner notes

---

## Day 7: Repeated and Clustered Spectrum Policy

**Title:** Spectrum Policy
**Theme:** Define subspace-safe metrics for repeated and clustered singular
values
**Time estimate:** 12 hours

### Tasks
1. Review repeated-spectrum and clustered-spectrum partial-SVD candidates.
2. Identify where vector equality is invalid because bases are non-unique or
   numerically unstable.
3. Define projector, principal-angle, residual, singular-value gap, and
   cluster-tolerance metrics.
4. Define diagnostics needed to distinguish repeated, clustered, and merely
   close spectra.
5. Choose Day 8 implementation or explicit deferral paths.
6. Write the repeated/clustered spectrum policy artifact.

### Deliverables
- repeated and clustered candidate table
- subspace metric policy
- gap and tolerance diagnostics
- Day 8 implementation or deferral checklist

### Completion Criteria
- no repeated or clustered case relies on raw vector equality
- subspace metrics are pinned before implementation
- spectral-gap diagnostics are required for accepted evidence

---

## Day 8: Repeated and Clustered Spectrum Evidence

**Title:** Spectrum Evidence
**Theme:** Implement or explicitly defer repeated-spectrum and
clustered-spectrum partial-SVD evidence
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 7 policy to repeated and clustered candidates.
2. Implement one bounded spectrum evidence lane only if subspace metrics,
   tolerance, fixture, oracle, and diagnostics are complete.
3. Otherwise document explicit deferral with blocker and future promotion gate.
4. Keep spectral evidence separate from rectangular, rank-deficient,
   SuiteSparse, optimality, and convergence claims.
5. Run focused SVD/helper validation for touched files.
6. Record diagnostics, validation, and non-claims.

### Deliverables
- repeated or clustered spectrum implementation or explicit deferral
- projector or principal-angle diagnostics
- focused validation results
- spectrum non-claim update

### Completion Criteria
- accepted evidence uses subspace-safe metrics
- basis non-uniqueness is explicit in the decision package
- no broad clustered-spectrum or partial-SVD parity claim is added

---

## Day 9: Rank-Deficient Subspace Gate

**Title:** Rank-Def Gate
**Theme:** Define rank, nullity, projection, and tolerance policies for
rank-deficient partial-SVD subspace evidence
**Time estimate:** 12 hours

### Tasks
1. Inventory rank-deficient partial-SVD candidates and existing rank/nullity
   helpers.
2. Define expected numerical rank, nullity, retained subspace, and residual
   semantics for each candidate.
3. Define projector, nullspace, reconstruction, and singular-value threshold
   metrics.
4. Separate rank-deficient evidence from repeated-spectrum, minimum-norm, and
   solver-selection claims.
5. Select one Day 10 lane or define explicit deferral blockers.
6. Write the rank-deficient subspace gate.

### Deliverables
- rank-deficient candidate table
- rank/nullity and threshold policy
- projection and residual metric policy
- Day 10 acceptance or deferral checklist

### Completion Criteria
- rank and nullity expectations are explicit before implementation
- projection metrics are preferred where bases are non-unique
- no rank-deficient evidence implies broad solver robustness

---

## Day 10: Rank-Deficient Subspace Evidence

**Title:** Rank-Def Evidence
**Theme:** Implement or explicitly defer bounded rank-deficient partial-SVD
subspace evidence
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 9 gate to the selected rank-deficient candidate.
2. Implement one accepted evidence lane if rank, nullity, projection,
   tolerance, diagnostics, and validation are ready.
3. Otherwise publish an explicit deferral package.
4. Update evidence tables only if the new lane adds non-duplicative trust.
5. Run focused partial-SVD validation for touched code or fixtures.
6. Record rank-deficient non-claims and future-owner notes.

### Deliverables
- rank-deficient subspace implementation or explicit deferral
- rank/nullity diagnostics
- focused validation results
- evidence-table update or no-update rationale

### Completion Criteria
- accepted evidence validates bounded rank-deficient subspace behavior
- no raw basis uniqueness or broad optimality claim is introduced
- every deferral has blocker, dependency, and future owner

---

## Day 11: SuiteSparse and Corpus Evidence Gate

**Title:** Corpus Gate
**Theme:** Decide whether SuiteSparse partial-SVD evidence has enough corpus,
oracle, and skip metadata to proceed
**Time estimate:** 12 hours

### Tasks
1. Inventory checked-in and optional SuiteSparse matrices relevant to
   partial-SVD residual behavior.
2. Classify candidates by shape, symmetry, rank behavior, size, conditioning,
   and runtime.
3. Define optional-data skip behavior, support tier, diagnostics, and failure
   interpretation.
4. Reject product-observed values as independent expected values unless
   external metadata exists.
5. Select Day 12 corpus evidence or explicit deferral paths.
6. Write the SuiteSparse corpus gate.

### Deliverables
- SuiteSparse candidate inventory
- support-tier and optional-data policy
- oracle and diagnostics requirements
- Day 12 acceptance or deferral checklist

### Completion Criteria
- corpus evidence cannot proceed without skip and diagnostic metadata
- expected values are independent or explicitly bounded as smoke diagnostics
- runtime and platform expectations are explicit

---

## Day 12: SuiteSparse and Low-Rank Optimality Evidence

**Title:** Corpus Evidence
**Theme:** Implement or explicitly defer SuiteSparse corpus and low-rank
optimality evidence under bounded claim language
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 11 corpus gate to SuiteSparse candidates.
2. Define low-rank optimality metrics, including residual, reconstruction
   error, singular-value truncation, and comparison boundary.
3. Implement one accepted corpus or low-rank optimality lane only if oracle,
   metric, diagnostics, skip behavior, and support tier are complete.
4. Otherwise publish explicit deferrals for corpus and optimality lanes.
5. Run focused partial-SVD, corpus, and helper validation for touched files.
6. Record bounded claim language and non-claims.

### Deliverables
- SuiteSparse corpus implementation or explicit deferral
- low-rank optimality implementation or explicit deferral
- support-tier and skip-behavior notes
- focused validation package

### Completion Criteria
- corpus and optimality evidence are bounded and validated, or explicitly
  deferred
- no broad SuiteSparse, platform, or best-rank approximation claim is added
- optional-data behavior is visible to maintainers

---

## Day 13: Convergence-Budget Evidence

**Title:** Budget Evidence
**Theme:** Add or explicitly defer convergence-budget evidence without
implying broad partial-SVD parity
**Time estimate:** 11 hours

### Tasks
1. Inventory convergence-budget candidates, including iteration limit,
   tolerance, stagnation, and partial-result scenarios.
2. Define diagnostics for iteration count, achieved tolerance, returned
   singular values, residuals, and partial-result semantics.
3. Implement one bounded convergence-budget lane if metric, tolerance,
   failure interpretation, and validation are ready.
4. Otherwise write explicit deferral with blocker and future owner.
5. Keep convergence evidence separate from optimality, corpus, and
   solver-selection claims.
6. Run focused validation and record convergence non-claims.

### Deliverables
- convergence-budget implementation or explicit deferral
- iteration and tolerance diagnostics
- partial-result semantics notes
- focused validation results

### Completion Criteria
- accepted evidence reports bounded convergence behavior only
- partial results do not imply broad parity or success guarantees
- every deferred convergence lane has blocker and promotion criteria

---

## Day 14: Solver-Selection Gate and Sprint Closeout

**Title:** Claim Closeout
**Theme:** Refresh public solver-selection wording only where Sprint 130
evidence earns a stronger claim
**Time estimate:** 11 hours

### Tasks
1. Reconcile every Sprint 130 item against the project-plan checklist.
2. Review all accepted evidence and deferrals for solver-selection impact.
3. Update public solver-selection wording only if evidence supports a bounded
   user-facing claim beyond current workflow guidance.
4. Otherwise publish an explicit no-update rationale.
5. Finalize working notes, artifact indexes, validation logs, and non-claim
   register.
6. Write Sprint 130 closeout and Sprint 131 handoff notes.

### Deliverables
- solver-selection wording update or explicit no-update rationale
- final Sprint 130 evidence and deferral index
- final validation package
- updated non-claim register
- Sprint 131 handoff notes

### Completion Criteria
- all Sprint 130 deliverables are present or explicitly deferred
- public solver-selection wording matches only earned evidence
- no unresolved item lacks blocker, dependency, and future-owner notes
