# Sprint 160 Plan: QR Comparison Family Closure

**Sprint Duration:** 14 days
**Goal:** Add one bounded QR comparison family beyond the current minimum-norm
seed and publish normalized freshness evidence. This sprint implements the
Sprint 160 section of `docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Starting Point:** Sprint 160 begins from:
- Sprint 159 hosted generated evidence path available;
- reviewed hosted oracle/comparison freshness semantics available for selected
  rows;
- maintained QR corpus fixtures and external comparison harness available;
- QR comparison reporting already has a minimum-norm seed and explicit
  non-claim boundaries;
- public claims remain constrained to reviewed hosted evidence and explicit
  non-claims.

The sprint must:
- select one bounded QR fixture family with stable comparison metrics;
- define residual, rank, nullspace/projector, minimum-norm, tolerance, and
  skip/defer fields before implementation;
- extend the comparison runner and source-controlled contract rows for the
  selected family;
- add focused tests only for touched behavior and proof ownership;
- normalize generated comparison rows and require freshness for the selected
  family;
- update QR corpus, maintainer, solver-selection, and public non-claim wording;
- leave Sprint 161 with a partial-SVD comparison handoff grounded in the new
  comparison lessons.

**End State:** Sprint 160 leaves behind:
- one new bounded QR comparison family;
- normalized QR comparison report rows;
- refreshed QR comparison freshness evidence;
- updated QR corpus and public claim-boundary documentation;
- targeted validation records;
- Sprint 161 partial-SVD comparison handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 160 project-plan estimate.

---

## Day 1: Sprint Intake And Handoff Review

**Title:** Sprint Intake
**Theme:** Establish Sprint 160 scope, artifact layout, and QR comparison
handoff inputs
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 160 section of
   `docs/planning/EPIC_14/PROJECT_PLAN.md`.
2. Review Sprint 159 working notes, hosted freshness artifacts, and QR
   comparison handoff.
3. Create Sprint 160 working notes and artifact directory structure.
4. Inventory current QR comparison, corpus, oracle, and report-index surfaces.
5. Record explicit non-goals for broad QR parity, broad external-comparison
   claims, and basis-identity assertions.
6. Write the Day 1 sprint-intake artifact.

### Deliverables
- Sprint 160 working-notes baseline
- artifact directory structure
- QR comparison surface inventory
- non-goal and assumption register
- Day 1 sprint-intake artifact

### Completion Criteria
- Sprint 160 scope is tied to the Epic 14 project plan
- Sprint 159 hosted freshness handoff has been reviewed
- unsupported broad QR claims are blocked before target selection begins

---

## Day 2: QR Target Family Selection

**Title:** Target Selection
**Theme:** Select one bounded QR fixture family for comparison closure
**Time estimate:** 12 hours

### Tasks
1. Review maintained QR corpus fixtures and current comparison report rows.
2. Identify candidate families with stable residual, rank, and solution
   behavior.
3. Exclude candidates that depend on basis identity, unstable ordering, or
   unsupported external parity claims.
4. Select one target family with clear non-basis-identity semantics.
5. Map the selected family to source fixtures, expected outputs, and
   documentation owners.
6. Write the Day 2 target-selection artifact.

### Deliverables
- selected QR comparison family
- rejected-candidate register
- fixture and owner map
- basis-identity non-claim notes
- Day 2 target-selection artifact

### Completion Criteria
- selected target is narrow enough to close in one sprint
- selection criteria are evidence-based and reproducible
- rejected candidates have documented blockers or deferrals

---

## Day 3: Metric Contract Draft

**Title:** Metric Contract
**Theme:** Define the comparison fields and tolerance semantics before code
changes
**Time estimate:** 12 hours

### Tasks
1. Define required residual, rank, nullspace/projector, and minimum-norm fields.
2. Define tolerance fields for solver output, reference output, and row
   freshness.
3. Define skip, defer, stale, missing, and failure semantics for the selected
   family.
4. Identify which metrics are claim-bearing and which remain diagnostic.
5. Draft expected row schema changes or fixture additions.
6. Write the Day 3 metric-contract artifact.

### Deliverables
- QR comparison metric contract
- tolerance and row-state semantics
- claim-bearing versus diagnostic metric map
- expected row/schema update list
- Day 3 metric-contract artifact

### Completion Criteria
- implementation work has exact metric fields and tolerances
- skip and defer behavior cannot be mistaken for passing evidence
- claim-bearing QR wording is bounded by measurable rows

---

## Day 4: Harness Extension Design

**Title:** Harness Design
**Theme:** Design comparison-runner and contract-row changes for the selected
family
**Time estimate:** 12 hours

### Tasks
1. Review the external comparison runner and generated report paths.
2. Decide where the selected QR family should be wired into the harness.
3. Define source-controlled contract rows, expected fixture names, and output
   paths.
4. Map failure diagnostics for solver, reference, parse, tolerance, and
   freshness failures.
5. Identify any build-system, script, or docs surfaces touched by the change.
6. Write the Day 4 harness-design artifact.

### Deliverables
- harness extension design
- contract-row and output-path map
- failure diagnostic matrix
- touched-surface validation plan
- Day 4 harness-design artifact

### Completion Criteria
- harness changes are scoped to the selected QR family
- generated outputs have stable names and reviewable diagnostics
- validation requirements are known before implementation begins

---

## Day 5: Comparison Harness Implementation

**Title:** Harness Implementation
**Theme:** Extend the comparison runner and contract rows for the selected
family
**Time estimate:** 12 hours

### Tasks
1. Implement selected QR family wiring in the comparison runner.
2. Add or update source-controlled contract rows and expected metadata.
3. Emit required metric fields using the Day 3 contract.
4. Preserve existing minimum-norm seed behavior and non-selected family
   semantics.
5. Run focused local comparison commands for the selected family.
6. Record the Day 5 implementation artifact.

### Deliverables
- comparison-runner changes
- contract-row updates
- generated metric output for selected family
- focused local command output
- Day 5 harness-implementation artifact

### Completion Criteria
- selected family can be generated locally
- existing comparison rows are not weakened
- failures produce actionable diagnostics

---

## Day 6: Fixture And Corpus Integration

**Title:** Corpus Integration
**Theme:** Tie the selected comparison family to maintained QR corpus fixtures
**Time estimate:** 12 hours

### Tasks
1. Add or update QR corpus metadata needed by the selected comparison family.
2. Ensure fixture naming, manifests, and expected rows are consistent.
3. Verify fixture dimensions, rank conditions, and tolerance expectations.
4. Document any fixture-specific skips or deferrals.
5. Run focused corpus or fixture checks.
6. Write the Day 6 corpus-integration artifact.

### Deliverables
- QR corpus metadata updates
- fixture manifest alignment
- rank and tolerance notes
- focused corpus validation output
- Day 6 corpus-integration artifact

### Completion Criteria
- selected comparison rows are traceable to maintained fixtures
- fixture metadata is deterministic and documented
- corpus validation supports the comparison claim boundary

---

## Day 7: Focused Proof-Owner Test Design

**Title:** Test Design
**Theme:** Design targeted tests for touched QR comparison behavior
**Time estimate:** 12 hours

### Tasks
1. Identify script tests needed for comparison row generation and normalization.
2. Identify C proof-owner tests only if solver behavior or fixture helpers are
   touched.
3. Define failure cases for stale, missing, skipped, tolerance-failing, and
   valid rows.
4. Preserve existing QR test ownership and avoid broad test-file expansion
   unless required.
5. Map validation commands by changed-file type.
6. Write the Day 7 test-design artifact.

### Deliverables
- focused test plan
- script and C test ownership decision
- row-state failure-case list
- validation command matrix
- Day 7 test-design artifact

### Completion Criteria
- test scope matches touched behavior
- row-state failures are covered before freshness gating
- C/header quality gates are reserved for actual C/header changes

---

## Day 8: Focused Tests Implementation

**Title:** Focused Tests
**Theme:** Add targeted script and proof-owner tests for the selected family
**Time estimate:** 12 hours

### Tasks
1. Implement focused script tests for generated comparison rows.
2. Add targeted C tests only if Day 7 identified required touched behavior.
3. Cover valid, stale, missing, skipped, and tolerance-failing row states.
4. Preserve existing minimum-norm seed and unrelated comparison behavior.
5. Run targeted tests locally.
6. Record the Day 8 test-implementation artifact.

### Deliverables
- focused comparison tests
- optional C proof-owner tests
- targeted validation output
- row-state coverage notes
- Day 8 focused-tests artifact

### Completion Criteria
- selected family has direct regression coverage
- stale or invalid selected rows fail clearly
- existing comparison seed coverage remains intact

---

## Day 9: Report Integration Design

**Title:** Report Design
**Theme:** Define normalized rows and freshness requirements for the selected
family
**Time estimate:** 12 hours

### Tasks
1. Review report-index and comparison freshness expectations from Sprint 159.
2. Define normalized row names, family names, statuses, and freshness fields.
3. Decide whether the selected row is reviewed hosted, supplemental hosted, or
   local-only advisory.
4. Define deterministic summary output needed for reviewers.
5. Identify docs and support-tier wording affected by report integration.
6. Write the Day 9 report-integration design artifact.

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
**Theme:** Normalize generated comparison rows and enforce selected freshness
**Time estimate:** 12 hours

### Tasks
1. Implement report-index or comparison-output changes for normalized rows.
2. Wire selected freshness checks for the new QR comparison family.
3. Preserve local-only behavior for non-promoted comparison families.
4. Add or update focused normalizer tests as needed.
5. Run comparison freshness and normalizer checks locally.
6. Record the Day 10 report-integration artifact.

### Deliverables
- normalized comparison report rows
- selected freshness wiring
- normalizer test updates
- local freshness validation output
- Day 10 report-integration artifact

### Completion Criteria
- selected QR comparison rows appear in normalized reports
- selected freshness fails on stale, missing, or invalid rows
- non-selected families retain documented support-tier behavior

---

## Day 11: Documentation Alignment

**Title:** Docs Alignment
**Theme:** Align QR corpus, maintainer, solver-selection, and public non-claim
wording
**Time estimate:** 12 hours

### Tasks
1. Update QR corpus documentation for the selected comparison family.
2. Update maintainer guidance for generating and validating the new rows.
3. Update solver-selection or QR documentation with earned wording only.
4. Preserve explicit non-claims for broad external parity and basis identity.
5. Draft Sprint 161 partial-SVD comparison handoff notes.
6. Write the Day 11 documentation-alignment artifact.

### Deliverables
- QR corpus documentation updates
- maintainer validation guidance
- solver-selection or QR wording updates
- non-claim preservation notes
- Day 11 docs-alignment artifact

### Completion Criteria
- docs match the implemented comparison evidence
- no broad QR or external parity claims are introduced
- Sprint 161 handoff has concrete lessons and residuals

---

## Day 12: Local Validation Pass

**Title:** Local Validation
**Theme:** Run focused comparison, corpus, report, docs, and quality checks
**Time estimate:** 12 hours

### Tasks
1. Run selected QR comparison generation and freshness commands.
2. Run focused corpus and report-index checks.
3. Run targeted script tests for row normalization and freshness semantics.
4. Run docs checks and whitespace hygiene.
5. If `.c` or `.h` files changed, run `make format`, `make lint`, and
   `make test`.
6. Record the Day 12 validation artifact.

### Deliverables
- comparison freshness validation output
- corpus and report-index check output
- targeted test output
- quality-check record
- Day 12 validation artifact

### Completion Criteria
- selected comparison evidence passes locally
- quality gates match the changed-file surface
- remaining failures, if any, are understood before closeout

---

## Day 13: Evidence And Claim Review

**Title:** Evidence Review
**Theme:** Review comparison rows, hosted evidence, docs, and claims as one
surface
**Time estimate:** 12 hours

### Tasks
1. Trace each selected QR comparison claim to fixture, generated row, test, and
   documentation evidence.
2. Confirm hosted or selected freshness behavior matches support-tier wording.
3. Confirm skip/defer rows cannot be read as passing evidence.
4. Review diffs for unsupported broad claims, stale paths, and ambiguous
   terminology.
5. Finalize Sprint 161 partial-SVD comparison handoff.
6. Write the Day 13 evidence-review artifact.

### Deliverables
- claim-to-evidence trace
- support-tier consistency checklist
- skip/defer interpretation notes
- finalized Sprint 161 handoff
- Day 13 evidence-review artifact

### Completion Criteria
- selected QR comparison evidence is reviewable end to end
- public wording is bounded by generated evidence
- Sprint 161 handoff is ready to start partial-SVD comparison work

---

## Day 14: Closeout And Retrospective Prep

**Title:** Closeout
**Theme:** Finalize Sprint 160 artifacts, validation record, and retrospective
inputs
**Time estimate:** 12 hours

### Tasks
1. Re-run final targeted checks required by the changed-file surface.
2. Update Sprint 160 working notes with final decisions, commands, and outputs.
3. Finalize closeout artifacts for selected rows, deferred rows, validation,
   and Sprint 161 handoff.
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
- Sprint 160 deliverables are complete and traceable
- validation status is recorded with exact commands
- Sprint 161 partial-SVD comparison handoff is ready
