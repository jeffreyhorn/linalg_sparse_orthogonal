# Sprint 161 Plan: Partial-SVD Comparison Publication Closure

**Sprint Duration:** 14 days
**Goal:** Publish the first bounded partial-SVD comparison family with
subspace-safe metrics and generated freshness checks. This sprint implements
the Sprint 161 section of `docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Starting Point:** Sprint 161 begins from:
- Sprint 159 hosted generated evidence path available;
- Sprint 151 partial-SVD corpus families available;
- Sprint 160 descriptor-backed comparison runner pattern available;
- Sprint 160 selected comparison freshness, focused runner tests, normalizer
  row-state tests, documentation non-claims, and partial-SVD handoff
  available;
- public claims remain constrained to fixture-local generated evidence and
  explicit non-claims.

The sprint must:
- select one partial-SVD fixture family with stable subspace-safe behavior;
- define singular-value, projector/subspace, residual, orthogonality,
  convergence, fail-closed, recovery, tolerance, and skip/defer fields before
  implementation;
- extend the comparison runner and source-controlled report metadata for the
  selected partial-SVD family;
- add focused tests for runner output, normalizer freshness, and proof-owner
  behavior only where touched;
- normalize generated partial-SVD comparison rows and require freshness for
  the selected family;
- update SVD, corpus, maintainer, README, and non-claim documentation;
- leave Sprint 162 with a Windows package handoff grounded in the new
  comparison publication lessons.

**End State:** Sprint 161 leaves behind:
- first bounded partial-SVD comparison family;
- subspace-safe comparison contract;
- generated freshness proof;
- focused runner and normalizer tests;
- updated SVD comparison non-claims;
- Sprint 162 Windows package handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 161 project-plan estimate.

---

## Day 1: Sprint Intake And Handoff Review

**Title:** Sprint Intake
**Theme:** Establish Sprint 161 scope, artifact layout, and partial-SVD
comparison handoff inputs
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 161 section of
   `docs/planning/EPIC_14/PROJECT_PLAN.md`.
2. Review Sprint 151 partial-SVD corpus artifacts and Sprint 160 comparison
   handoff.
3. Create Sprint 161 working notes and artifact directory structure.
4. Inventory current SVD/partial-SVD corpus, oracle, comparison, report-index,
   and documentation surfaces.
5. Record explicit non-goals for broad partial-SVD correctness, raw
   singular-vector identity, external-library parity, platform/package/ABI,
   performance, release, and state-of-the-art claims.
6. Write the Day 1 sprint-intake artifact.

### Deliverables
- Sprint 161 working-notes baseline
- artifact directory structure
- partial-SVD comparison surface inventory
- non-goal and assumption register
- Day 1 sprint-intake artifact

### Completion Criteria
- Sprint 161 scope is tied to the Epic 14 project plan
- Sprint 151 and Sprint 160 handoffs have been reviewed
- unsupported broad partial-SVD claims are blocked before target selection

---

## Day 2: Partial-SVD Target Family Selection

**Title:** Target Selection
**Theme:** Select one bounded partial-SVD fixture family for comparison
publication
**Time estimate:** 12 hours

### Tasks
1. Review maintained partial-SVD corpus fixtures and dense reference helper
   targets.
2. Identify candidate families with stable singular-value, residual,
   orthogonality, and subspace behavior.
3. Exclude candidates that depend on raw singular-vector identity, unstable
   ordering, repeated-spectrum vector identity, or unsupported parity claims.
4. Select one target family with clear subspace-safe or diagonal-safe
   semantics.
5. Map the selected family to source fixtures, expected outputs, report rows,
   and documentation owners.
6. Write the Day 2 target-selection artifact.

### Deliverables
- selected partial-SVD comparison family
- rejected-candidate register
- fixture and owner map
- raw-vector-identity non-claim notes
- Day 2 target-selection artifact

### Completion Criteria
- selected target is narrow enough to close in one sprint
- selection criteria are evidence-based and reproducible
- rejected candidates have documented blockers or deferrals

---

## Day 3: Subspace-Safe Metric Contract

**Title:** Metric Contract
**Theme:** Define comparison fields and tolerance semantics before code changes
**Time estimate:** 12 hours

### Tasks
1. Define singular-value, residual, orthogonality, projector/subspace, and
   recovery fields required for the selected target.
2. Define convergence, fail-closed, and partial-result interpretation where
   relevant.
3. Define tolerance fields for project output, reference output, row
   freshness, vector sign/order handling, and subspace-safe deltas.
4. Define skip, defer, stale, missing, malformed, non-pass, and failure
   semantics.
5. Identify which metrics are claim-bearing and which remain diagnostic.
6. Write the Day 3 metric-contract artifact.

### Deliverables
- partial-SVD comparison metric contract
- tolerance and row-state semantics
- claim-bearing versus diagnostic metric map
- raw-vector-identity and ordering non-claim notes
- Day 3 metric-contract artifact

### Completion Criteria
- implementation work has exact metric fields and tolerances
- skip/defer behavior cannot be mistaken for passing evidence
- partial-SVD wording is bounded by subspace-safe measurable rows

---

## Day 4: Harness Extension Design

**Title:** Harness Design
**Theme:** Design comparison-runner, expected-row, dependency-status, and
metadata changes
**Time estimate:** 12 hours

### Tasks
1. Review the external comparison runner descriptor model from Sprint 160.
2. Decide where the selected partial-SVD family should be wired into the
   harness.
3. Define source-controlled contract rows, expected fixture names, output
   paths, and artifact names.
4. Map failure diagnostics for solver, reference, parse, tolerance, freshness,
   convergence, and fail-closed failures.
5. Identify build-system, script, test, manifest, and docs surfaces touched by
   the change.
6. Write the Day 4 harness-design artifact.

### Deliverables
- harness extension design
- contract-row and output-path map
- failure diagnostic matrix
- touched-surface validation plan
- Day 4 harness-design artifact

### Completion Criteria
- harness changes are scoped to the selected partial-SVD family
- generated outputs have stable names and reviewable diagnostics
- validation requirements are known before implementation begins

---

## Day 5: Comparison Harness Implementation

**Title:** Harness Implementation
**Theme:** Extend the comparison runner and contract rows for the selected
partial-SVD family
**Time estimate:** 12 hours

### Tasks
1. Implement selected partial-SVD family wiring in the comparison runner.
2. Add or update source-controlled contract rows and expected metadata.
3. Emit required metric fields using the Day 3 contract.
4. Preserve existing QR comparison behavior and non-selected family semantics.
5. Run focused local comparison commands for the selected family.
6. Record the Day 5 implementation artifact.

### Deliverables
- comparison-runner changes
- contract-row updates
- generated metric output for selected family
- focused local command output
- Day 5 harness-implementation artifact

### Completion Criteria
- selected partial-SVD family can be generated locally
- existing QR comparison rows are not weakened
- failures produce actionable diagnostics

---

## Day 6: Expected Rows And Dependency Semantics

**Title:** Expected Rows
**Theme:** Tie selected partial-SVD comparison rows to maintained metadata and
dependency policy
**Time estimate:** 12 hours

### Tasks
1. Add or update report-family metadata needed by the selected partial-SVD
   comparison family.
2. Ensure fixture naming, row IDs, support tier, artifact pattern, and claim
   scope are consistent.
3. Verify dependency status rows keep optional external package behavior as
   skip/defer context.
4. Document any selected-family skips, deferrals, or unsupported states.
5. Run focused schema and generator checks.
6. Write the Day 6 expected-rows artifact.

### Deliverables
- report-family metadata updates
- fixture and row-ID alignment
- dependency skip/defer notes
- focused schema/generator validation output
- Day 6 expected-rows artifact

### Completion Criteria
- selected comparison rows are traceable to maintained metadata
- dependency defers cannot become pass evidence
- metadata supports the comparison claim boundary

---

## Day 7: Focused Proof-Owner Test Design

**Title:** Test Design
**Theme:** Design targeted tests for touched partial-SVD comparison behavior
**Time estimate:** 12 hours

### Tasks
1. Identify script tests needed for comparison target dispatch and row
   generation.
2. Identify normalizer tests needed for selected partial-SVD comparison
   freshness.
3. Identify C proof-owner tests only if solver behavior or fixture helpers are
   touched.
4. Define failure cases for stale, missing, unexpected, duplicate, skipped,
   deferred, tolerance-failing, malformed, and valid rows.
5. Map validation commands by changed-file type.
6. Write the Day 7 test-design artifact.

### Deliverables
- focused test plan
- script, normalizer, and C proof-owner decision
- row-state failure-case list
- validation command matrix
- Day 7 test-design artifact

### Completion Criteria
- test scope matches touched behavior
- row-state failures are covered before freshness promotion
- C/header quality gates are reserved for actual C/header changes

---

## Day 8: Focused Tests Implementation

**Title:** Focused Tests
**Theme:** Add targeted runner, normalizer, and optional proof-owner tests
**Time estimate:** 12 hours

### Tasks
1. Implement focused script tests for selected partial-SVD generated rows.
2. Implement focused normalizer tests for required selected comparison
   freshness.
3. Add targeted C tests only if Day 7 identified touched implementation
   behavior.
4. Cover valid, stale, missing, unexpected, duplicate, skipped, deferred, and
   tolerance-failing row states as appropriate.
5. Run targeted tests locally.
6. Record the Day 8 focused-tests artifact.

### Deliverables
- focused comparison tests
- focused normalizer tests
- optional C proof-owner tests
- targeted validation output
- Day 8 focused-tests artifact

### Completion Criteria
- selected family has direct regression coverage
- stale or invalid selected rows fail clearly
- existing QR comparison coverage remains intact

---

## Day 9: Report Integration Design

**Title:** Report Design
**Theme:** Define normalized rows and freshness requirements for the selected
partial-SVD family
**Time estimate:** 12 hours

### Tasks
1. Review report-index and comparison freshness expectations from Sprints 159
   and 160.
2. Define normalized row names, family names, statuses, freshness fields, and
   artifact diagnostics.
3. Decide whether the selected row is reviewed hosted, supplemental hosted, or
   local-only advisory.
4. Define deterministic summary output needed for reviewers.
5. Identify docs and support-tier wording affected by report integration.
6. Write the Day 9 report-design artifact.

### Deliverables
- normalized row design
- freshness requirement decision
- hosted/supplemental/local classification
- deterministic summary expectations
- Day 9 report-design artifact

### Completion Criteria
- generated rows have stable normalized identifiers
- freshness status is tied to the correct evidence tier
- reviewers can inspect the selected family without local reproduction

---

## Day 10: Report Integration Implementation

**Title:** Report Integration
**Theme:** Normalize generated partial-SVD comparison rows and enforce selected
freshness
**Time estimate:** 12 hours

### Tasks
1. Implement report-index or comparison-output changes for normalized
   partial-SVD rows.
2. Wire selected freshness checks for the new partial-SVD comparison family.
3. Preserve selected QR comparison behavior and non-promoted families.
4. Add or update focused normalizer tests as needed.
5. Run comparison freshness and normalizer checks locally.
6. Record the Day 10 report-integration artifact.

### Deliverables
- normalized partial-SVD comparison report rows
- selected freshness wiring
- normalizer test updates
- local freshness validation output
- Day 10 report-integration artifact

### Completion Criteria
- selected partial-SVD comparison rows appear in normalized reports
- selected freshness fails on stale, missing, or invalid rows
- non-selected families retain documented support-tier behavior

---

## Day 11: Documentation Alignment

**Title:** Docs Alignment
**Theme:** Align SVD docs, corpus docs, maintainer guidance, README, and
non-claim wording
**Time estimate:** 12 hours

### Tasks
1. Update SVD or partial-SVD documentation for the selected comparison family.
2. Update maintainer guidance for generating and validating the new rows.
3. Update corpus documentation with row meaning, support tier, and artifact
   paths.
4. Update README or solver-selection wording with earned wording only.
5. Preserve explicit non-claims for broad partial-SVD correctness,
   raw-vector identity, ordering identity, external parity, platform, package,
   ABI, performance, release, and state-of-the-art claims.
6. Write the Day 11 documentation-alignment artifact.

### Deliverables
- SVD/partial-SVD documentation updates
- maintainer validation guidance
- corpus and README wording updates
- non-claim preservation notes
- Day 11 docs-alignment artifact

### Completion Criteria
- docs match the implemented comparison evidence
- no broad partial-SVD or external parity claims are introduced
- support-tier wording matches generated freshness behavior

---

## Day 12: Local Validation Pass

**Title:** Local Validation
**Theme:** Run focused comparison, corpus, report, docs, and quality checks
**Time estimate:** 12 hours

### Tasks
1. Run selected partial-SVD comparison generation and freshness commands.
2. Run selected QR comparison freshness to protect Sprint 160 behavior.
3. Run focused corpus, schema, and report-index checks.
4. Run targeted script and normalizer tests for row semantics.
5. Run docs checks and whitespace hygiene.
6. If `.c` or `.h` files changed, run `make format`, `make lint`, and
   `make test`.

### Deliverables
- partial-SVD comparison freshness validation output
- QR comparison regression validation output
- corpus and report-index check output
- targeted test output
- Day 12 validation artifact

### Completion Criteria
- selected comparison evidence passes locally
- quality gates match the changed-file surface
- remaining failures, if any, are understood before closeout

---

## Day 13: Evidence And Claim Review

**Title:** Evidence Review
**Theme:** Review comparison rows, freshness behavior, docs, and claims as one
surface
**Time estimate:** 12 hours

### Tasks
1. Trace each selected partial-SVD comparison claim to fixture, generated row,
   test, and documentation evidence.
2. Confirm selected freshness behavior matches support-tier wording.
3. Confirm skip/defer rows cannot be read as passing evidence.
4. Review diffs for unsupported broad claims, stale paths, and ambiguous
   terminology.
5. Finalize Sprint 162 Windows package handoff.
6. Write the Day 13 evidence-review artifact.

### Deliverables
- claim-to-evidence trace
- support-tier consistency checklist
- skip/defer interpretation notes
- Sprint 162 Windows package handoff
- Day 13 evidence-review artifact

### Completion Criteria
- selected partial-SVD comparison evidence is reviewable end to end
- public wording is bounded by generated evidence
- Sprint 162 handoff is ready to start Windows package follow-through

---

## Day 14: Closeout And Retrospective Prep

**Title:** Closeout
**Theme:** Finalize Sprint 161 artifacts, validation record, and retrospective
inputs
**Time estimate:** 12 hours

### Tasks
1. Re-run final targeted checks required by the changed-file surface.
2. Update Sprint 161 working notes with final decisions, commands, and outputs.
3. Finalize closeout artifacts for selected rows, deferred rows, validation,
   and Sprint 162 handoff.
4. Review changed files for claim wording, stale paths, and unsupported
   evidence assertions.
5. Prepare retrospective inputs from artifacts and working notes.
6. Record the Day 14 closeout artifact.

### Deliverables
- final validation record
- selected/deferred row closeout notes
- complete working notes
- retrospective input set
- Day 14 closeout artifact

### Completion Criteria
- Sprint 161 deliverables are complete and traceable
- validation status is recorded with exact commands
- Sprint 162 Windows package handoff is ready
