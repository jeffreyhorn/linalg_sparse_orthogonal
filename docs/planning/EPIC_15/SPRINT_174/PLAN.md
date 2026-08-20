# Sprint 174 Plan: Additional Bounded External Comparison Family

**Sprint Duration:** 14 days
**Goal:** Add one more complete bounded external comparison family with
generated report, freshness checks, and claim-safe documentation. This sprint
implements the Sprint 174 section of
`docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md`, but the active merged Sprint 174
project-plan section lives in `docs/planning/EPIC_15/PROJECT_PLAN.md` and has
the title "Sprint 174: Additional Bounded External Comparison Family".

**Starting Point:** Sprint 174 begins from:

- the existing external comparison harness and report-index architecture;
- Sprint 167 evidence-ledger and claim-gate conventions;
- Sprint 173 generated API local-only freshness and publication boundary;
- current QR and partial-SVD comparison/oracle freshness infrastructure;
- package-manager, static package, shared-library ABI, platform, performance,
  and state-of-the-art non-claim guards inherited from prior Epic 15 sprints.

The sprint must:

- select exactly one additional bounded external comparison family;
- define fixtures, external comparator, tolerances, report rows, and claim
  boundaries before implementation;
- extend the comparison runner and generated outputs for the selected family;
- integrate generated report/index/freshness checks;
- update documentation with exact solver, fixture, comparator, and tolerance
  scope;
- run comparison generation, freshness, focused tests, and relevant deferral
  guards.

**End State:** Sprint 174 leaves behind:

- one complete bounded external comparison family;
- generated and indexed comparison report support for that family;
- freshness checks that fail closed when report outputs are stale or missing;
- claim-safe public and maintainer documentation;
- Sprint 174 working notes, daily artifacts, and closeout records.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 174 project-plan estimate.

---

## Day 1: Sprint Intake And Comparison Boundary

**Title:** Comparison Intake
**Theme:** Establish Sprint 174 scope, inherited evidence rules, and external
comparison non-claims
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 174 section of
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.
2. Review Sprint 167 evidence-ledger guidance and prior comparison freshness
   artifacts.
3. Review Sprint 173 generated API local-only handoff to avoid mixing
   generated docs with comparison evidence.
4. Inventory existing external comparison families and their proof-owner rows.
5. Create Sprint 174 working notes and artifact directory structure.
6. Write the Day 1 comparison-intake artifact.

### Deliverables

- Sprint 174 working-notes baseline
- artifact directory structure
- inherited comparison and claim-boundary summary
- source artifact note
- Day 1 comparison-intake artifact

### Completion Criteria

- Sprint 174 scope is tied to the active Epic 15 project plan
- existing comparison families and report owners are visible
- unsupported platform, performance, package, ABI, and state-of-the-art claims
  remain protected

---

## Day 2: Candidate Family Inventory

**Title:** Family Inventory
**Theme:** Inventory candidate solver families, external comparators, fixtures,
and report integration costs
**Time estimate:** 12 hours

### Tasks

1. Inventory solver families that do not yet have the next bounded external
   comparison proof.
2. Review existing tests, fixtures, dense-reference helpers, report-index rows,
   and maintainer guide support tables for each candidate.
3. Identify candidate external comparators and whether they are
   source-controlled, local-only, or generated.
4. Rank candidates by user value, harness fit, implementation risk, and claim
   containment.
5. Identify families that should be rejected or deferred this sprint.
6. Write the Day 2 candidate-family inventory artifact.

### Deliverables

- candidate comparison-family matrix
- comparator availability notes
- implementation-risk ranking
- rejected/deferred candidate list
- Day 2 candidate-family inventory artifact

### Completion Criteria

- candidate families are comparable before selection
- no harness implementation starts before selecting one family
- unsupported broad parity claims remain excluded

---

## Day 3: Family Selection Decision

**Title:** Family Selection
**Theme:** Select exactly one bounded external comparison family and define the
initial claim surface
**Time estimate:** 12 hours

### Tasks

1. Select the solver family for Sprint 174.
2. Select the external comparator and explain why it is suitable for bounded
   fixture-local comparison.
3. Select initial fixture shapes, matrix properties, expected outputs, and
   tolerances.
4. Define explicit non-claims for broad solver parity, raw basis/vector
   identity, performance, platform, package, ABI, and state-of-the-art scope.
5. Define implementation acceptance criteria for Days 4 through 12.
6. Write the Day 3 family-selection decision artifact.

### Deliverables

- selected comparison family
- selected external comparator
- initial fixture/tolerance table
- bounded claim and non-claim statement
- Day 3 family-selection artifact

### Completion Criteria

- exactly one comparison family is selected
- selected fixtures and tolerances are reviewable before implementation
- non-claims are explicit enough to guide docs and tests

---

## Day 4: Fixture Design

**Title:** Fixture Design
**Theme:** Design bounded matrices and expected comparison outputs for the
selected family
**Time estimate:** 12 hours

### Tasks

1. Design fixture matrices that exercise the selected family without broad
   unsupported coverage.
2. Define fixture-local expected outputs and diagnostics.
3. Define tolerance strategy and failure messages.
4. Decide whether fixtures live in source-controlled helpers, generated
   reports, tests, or planning artifacts.
5. Identify edge cases that must remain deferred.
6. Write the Day 4 fixture-design artifact.

### Deliverables

- fixture matrix specification
- expected-output specification
- tolerance and diagnostics policy
- deferred edge-case list
- Day 4 fixture-design artifact

### Completion Criteria

- fixture design is precise enough for implementation
- expected outputs are bounded to selected fixtures
- deferred edge cases are documented before code changes

---

## Day 5: Comparator Output Design

**Title:** Comparator Design
**Theme:** Design external comparator invocation, output schema, and stale-data
handling
**Time estimate:** 12 hours

### Tasks

1. Review existing external dense-reference helpers and generated comparison
   report schemas.
2. Design the selected comparator output fields and diagnostic rows.
3. Define stale-output detection and regeneration behavior.
4. Define how comparator failures should fail closed.
5. Plan integration with report-index normalization and freshness checks.
6. Write the Day 5 comparator-output design artifact.

### Deliverables

- comparator command/input design
- output schema
- stale-data and failure semantics
- report-index integration plan
- Day 5 comparator-output artifact

### Completion Criteria

- comparator output is schema-stable before implementation
- stale or missing comparator output has clear failure behavior
- report integration requirements are known

---

## Day 6: Fixture Implementation

**Title:** Fixture Implementation
**Theme:** Implement selected fixtures and source-controlled expected-reference
inputs
**Time estimate:** 12 hours

### Tasks

1. Add or update fixture definitions for the selected family.
2. Add source-controlled comparator input or expected-reference helper data
   where appropriate.
3. Add focused tests for fixture construction and basic selected-family
   behavior.
4. Keep fixture names and diagnostics stable for report rows.
5. Run focused tests for changed fixture surfaces.
6. Write the Day 6 fixture-implementation artifact.

### Deliverables

- implemented selected-family fixtures
- expected-reference inputs or helper data
- focused fixture tests
- validation output summary
- Day 6 fixture-implementation artifact

### Completion Criteria

- selected fixtures build and run locally
- fixture names are stable enough for report integration
- implementation does not widen solver-family claims

---

## Day 7: Harness Extension Design

**Title:** Harness Design
**Theme:** Design the runner, CLI, generated files, and report rows for the
selected comparison family
**Time estimate:** 12 hours

### Tasks

1. Locate existing comparison runner entry points and report generation paths.
2. Define the selected-family runner option, fixture selection, and output
   path conventions.
3. Define report row fields, support tier, claim boundary, and diagnostics.
4. Define generated-output cleanup behavior.
5. Identify Make targets or script entry points that should own the run.
6. Write the Day 7 harness-extension design artifact.

### Deliverables

- harness extension design
- CLI/Make target plan
- generated-output path plan
- report-row schema additions
- Day 7 harness-design artifact

### Completion Criteria

- harness implementation has a narrow target
- generated output paths are ignored or source-controlled intentionally
- report rows have bounded claim wording before implementation

---

## Day 8: Harness Extension Implementation

**Title:** Harness Implementation
**Theme:** Implement selected-family comparison generation and local outputs
**Time estimate:** 12 hours

### Tasks

1. Extend the comparison runner for the selected family.
2. Add CLI or Make target wiring for the selected comparison generation path.
3. Emit generated comparison artifacts using the Day 7 schema.
4. Ensure stale generated outputs are cleared or overwritten predictably.
5. Run the selected comparison generation locally.
6. Write the Day 8 harness-implementation artifact.

### Deliverables

- implemented comparison runner extension
- selected comparison generation command
- generated comparison artifacts
- local generation validation summary
- Day 8 harness-implementation artifact

### Completion Criteria

- selected comparison artifacts can be generated locally
- generated outputs follow the planned schema
- stale outputs do not masquerade as fresh evidence

---

## Day 9: Report Index Integration

**Title:** Report Integration
**Theme:** Add selected-family rows to normalized report-index and proof-owner
surfaces
**Time estimate:** 12 hours

### Tasks

1. Add or update report-index rows for the selected comparison family.
2. Add proof-owner metadata for the selected generated comparison path.
3. Update normalization logic only if the existing schema cannot represent the
   selected family.
4. Run report-index normalization checks.
5. Verify generated artifacts map to source-controlled proof-owner rows.
6. Write the Day 9 report-integration artifact.

### Deliverables

- report-index row updates
- proof-owner row updates
- normalization/freshness validation output
- generated artifact mapping
- Day 9 report-integration artifact

### Completion Criteria

- selected comparison report rows normalize cleanly
- proof owners point to source-controlled commands and claim boundaries
- no broad report-family claim is introduced

---

## Day 10: Freshness Gate Implementation

**Title:** Freshness Gate
**Theme:** Add or extend checks that prove selected comparison reports are
fresh and complete
**Time estimate:** 12 hours

### Tasks

1. Add or extend the selected comparison freshness checker.
2. Ensure missing, stale, malformed, or mismatched generated report artifacts
   fail closed.
3. Add Make target wiring for the selected freshness path where appropriate.
4. Run positive freshness validation.
5. Run at least one negative or stale-output proof where practical.
6. Write the Day 10 freshness-gate artifact.

### Deliverables

- freshness checker implementation
- Make target or script entry point
- positive validation evidence
- stale/missing-output failure evidence
- Day 10 freshness-gate artifact

### Completion Criteria

- selected comparison freshness can be checked repeatably
- stale or missing artifacts fail clearly
- freshness wording stays fixture-local

---

## Day 11: Claim Documentation Update

**Title:** Claim Documentation
**Theme:** Update public and maintainer documentation with exact comparator,
fixture, tolerance, and non-claim scope
**Time estimate:** 12 hours

### Tasks

1. Update README, maintainer guide, and relevant report/comparison docs with
   selected-family scope.
2. Document exact fixtures, comparator, tolerances, diagnostics, and freshness
   command.
3. Preserve non-claims for broad solver parity, performance, platform,
   package, ABI, runtime-loader, external-library parity, and state-of-the-art
   support.
4. Run targeted claim scans over changed documentation.
5. Run package/static deferral guards if wording touches those surfaces.
6. Write the Day 11 claim-documentation artifact.

### Deliverables

- updated public/maintainer documentation
- selected-family claim table entries
- claim-scan results
- deferral-guard results if needed
- Day 11 claim-documentation artifact

### Completion Criteria

- users can find the selected comparison and understand its boundary
- docs name the freshness command and fixture-local scope
- unsupported claims remain excluded

---

## Day 12: Integrated Comparison Validation

**Title:** Integrated Validation
**Theme:** Run selected-family generation, freshness, tests, report checks, and
claim guards together
**Time estimate:** 12 hours

### Tasks

1. Run selected comparison generation from a clean local output state.
2. Run selected comparison freshness checks.
3. Run focused tests for changed solver, fixture, comparator, and report
   surfaces.
4. Run report-index normalization and freshness checks.
5. Run targeted claim scans and relevant package/static deferral guards.
6. Write the Day 12 integrated-validation artifact.

### Deliverables

- integrated validation command log
- focused test results
- report-index validation results
- claim/deferral guard results
- Day 12 integrated-validation artifact

### Completion Criteria

- selected comparison family validates end to end
- report artifacts are generated and freshness-checked
- unsupported claims remain bounded after docs and report updates

---

## Day 13: Integrated Claim Review

**Title:** Claim Review
**Theme:** Reconcile selected-family evidence, docs, reports, freshness, and
non-claims before closeout
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 174 artifacts and working notes.
2. Reconcile selected-family fixtures, comparator, tolerances, report rows,
   freshness checks, and documentation wording.
3. Confirm generated-output staging behavior matches the selected report
   architecture.
4. Re-run targeted claim scans and relevant deferral guards.
5. Identify Sprint 175 handoff needs.
6. Write the Day 13 integrated-claim-review artifact.

### Deliverables

- selected-family claim review
- generated-output staging review
- claim-scan and deferral-guard results
- Sprint 175 handoff list
- Day 13 claim-review artifact

### Completion Criteria

- selected-family comparison claim is internally coherent
- docs, reports, and freshness gates agree
- broad parity, performance, package, ABI, platform, and state-of-the-art
  claims remain unselected

---

## Day 14: Sprint Closeout And Sprint 175 Handoff

**Title:** Sprint Closeout
**Theme:** Reconcile Sprint 174 deliverables, final validation, and handoff to
cross-platform report freshness work
**Time estimate:** 12 hours

### Tasks

1. Reconcile Sprint 174 outcomes against project-plan items 174.1 through
   174.6.
2. Confirm selected-family fixtures, harness, report integration, freshness
   gates, docs, and validation artifacts are complete.
3. Re-run final hygiene and required focused checks.
4. Confirm generated comparison outputs are staged or ignored according to the
   selected architecture.
5. Prepare Sprint 175 handoff notes for cross-platform report freshness
   promotion.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables

- final Sprint 174 validation record
- project-plan item reconciliation
- generated-output staging check
- Sprint 175 handoff
- Day 14 sprint-closeout artifact

### Completion Criteria

- one bounded external comparison family is complete and validated
- generated report/index/freshness behavior is source-controlled and bounded
- Sprint 175 can begin from clear selected-report freshness boundaries
