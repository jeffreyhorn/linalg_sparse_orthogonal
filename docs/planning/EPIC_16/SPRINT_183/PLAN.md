# Sprint 183 Plan: Additional Bounded External Comparison Family

**Sprint Duration:** 14 days
**Goal:** Add one fully maintained external comparison family with fixtures,
metrics, report freshness, selected-target metadata, and scoped claims. This
sprint implements the Sprint 183 section of
`docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Source Artifact Note:** This plan lives under
`docs/planning/EPIC_16/SPRINT_183/PLAN.md` and implements the Sprint 183
section of `docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Starting Point:** Sprint 183 begins from:

- the Sprint 181 selected target manifest at
  `tests/corpus/manifests/selected_report_targets.tsv`;
- the existing external comparison runner in
  `scripts/run_external_comparison.py`;
- selected QR, partial-SVD, and LU comparison families with manifest-backed
  report freshness;
- existing selected comparison workflow guards for Linux/macOS selected
  report freshness and Windows deferral;
- Sprint 182's formal Windows report freshness deferral and handoff notes;
- README, solver-selection, maintainer-guide, corpus, and report-index docs
  that preserve bounded comparison claims and non-claims.

The sprint must:

- select exactly one additional bounded external comparison family by claim
  risk, user value, fixture stability, and comparator availability;
- define source-controlled fixtures, expected rows, metrics, tolerances,
  skip/defer behavior, and non-parity wording;
- extend comparison runner logic and focused tests for the selected family;
- generate, index, freshness-check, and register the selected comparison
  report in selected-target metadata;
- align README, solver-selection docs, maintainer guide, corpus/report docs,
  workflow guards, and claim boundaries;
- validate comparison generation, selected freshness checks, script tests,
  relevant C tests, package/ABI deferral guards, and whitespace hygiene.

**End State:** Sprint 183 leaves behind:

- one additional bounded external comparison family;
- manifest-backed selected comparison report metadata and freshness checks;
- focused runner/helper tests and any relevant C fixture tests;
- updated public, maintainer, corpus, report-index, and claim-boundary docs;
- validation records, residual risks, retrospective inputs, and Sprint 184
  handoff notes.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 183 project-plan estimate.

---

## Day 1: Comparison Family Intake

**Title:** Family Intake
**Theme:** Establish Sprint 183 scope, inherited comparison authority,
candidate criteria, and artifact structure
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 183 section of
   `docs/planning/EPIC_16/PROJECT_PLAN.md`.
2. Review Sprint 181 selected target manifest authority and Sprint 182
   Windows report freshness handoff boundaries.
3. Review existing selected comparison families for QR minimum-norm, QR
   compatible least-squares, partial-SVD diagonal top-k, and LU nonsymmetric
   square solve.
4. Create Sprint 183 working notes and artifact directory structure.
5. Define candidate evaluation criteria for user value, fixture stability,
   comparator availability, implementation cost, validation cost, and claim
   risk.
6. Write the Day 1 comparison-family-intake artifact.

### Deliverables

- Sprint 183 working-notes baseline
- artifact directory structure
- inherited selected comparison authority notes
- candidate evaluation criteria
- Day 1 comparison-family-intake artifact

### Completion Criteria

- Sprint 183 scope is tied to the Epic 16 project plan
- existing selected comparison families and non-claims are explicit
- candidate selection starts from shared criteria rather than preference

---

## Day 2: Existing Comparison Surface Audit

**Title:** Surface Audit
**Theme:** Audit current comparison runner, selected rows, artifacts,
freshness checks, and workflow guard behavior
**Time estimate:** 12 hours

### Tasks

1. Inspect `scripts/run_external_comparison.py` target registration,
   generator paths, project probes, baseline helper execution, and output
   writers.
2. Inspect current comparison helper tests, report-index tests, and selected
   workflow guard tests.
3. Inspect `tests/corpus/manifests/selected_report_targets.tsv` comparison
   rows, selected row IDs, required files, workflow metadata, support tiers,
   claim scopes, and non-claims.
4. Inspect current generated comparison artifact layout under
   `build/comparison/` if present, without staging generated outputs.
5. Record invariants that any new family must preserve.
6. Write the Day 2 existing-comparison-surface-audit artifact.

### Deliverables

- current comparison runner inventory
- selected manifest row inventory
- artifact and freshness-check invariant list
- workflow guard interaction notes
- Day 2 existing-comparison-surface-audit artifact

### Completion Criteria

- new family design starts from the existing selected comparison contract
- artifact, row-count, and guard invariants are explicit
- generated local outputs remain unstaged

---

## Day 3: Candidate Family Inventory

**Title:** Candidate Inventory
**Theme:** Inventory possible bounded comparison families and reject broad or
claim-risky candidates early
**Time estimate:** 12 hours

### Tasks

1. Inventory candidate solver or factorization families not already covered
   by selected comparison rows.
2. For each candidate, identify source-controlled fixtures, baseline helper
   availability, expected metrics, optional dependency needs, and C test
   coverage.
3. Score candidates for user value, fixture stability, comparator
   availability, implementation size, maintenance cost, and claim risk.
4. Reject candidates that require broad parity, external package guarantees,
   unstable numerics, or unbounded performance claims.
5. Shortlist one or two families for Day 4 fixture and metric design.
6. Write the Day 3 candidate-family-inventory artifact.

### Deliverables

- candidate family inventory table
- reject/shortlist rationale
- comparator availability notes
- claim-risk notes
- Day 3 candidate-family-inventory artifact

### Completion Criteria

- at least one feasible bounded family is shortlisted
- broad or unstable candidates have explicit rejection reasons
- family selection remains tied to defensible evidence

---

## Day 4: Family Selection

**Title:** Family Selection
**Theme:** Select the next bounded external comparison family and define the
closed claim
**Time estimate:** 12 hours

### Tasks

1. Compare shortlisted families against Sprint 183 item 183.1 criteria.
2. Select exactly one family for implementation.
3. Define the intended closed claim, explicit non-claims, and support tier.
4. Identify existing C fixtures or new C test coverage needed to support the
   selected comparison family.
5. Identify required Python helper, runner, manifest, report-index, workflow,
   and documentation changes.
6. Write the Day 4 family-selection artifact.

### Deliverables

- selected family decision
- closed-claim statement
- non-claim and support-boundary list
- required implementation surface map
- Day 4 family-selection artifact

### Completion Criteria

- exactly one family is selected
- selected family claim is bounded and testable
- implementation scope is narrow enough for the remaining sprint

---

## Day 5: Fixture And Metric Contract

**Title:** Fixture Contract
**Theme:** Define source-controlled fixture inputs, expected rows, metrics,
tolerances, skip/defer behavior, and non-parity wording
**Time estimate:** 12 hours

### Tasks

1. Define the selected fixture identity, matrix shape, solver mode, expected
   status, and source-controlled provenance.
2. Define project and baseline metrics, row IDs, tolerances, and exact row
   meanings.
3. Define optional dependency skip/defer behavior and diagnostics.
4. Define expected output files and selected report row count.
5. Draft manifest row metadata for support tier, freshness policy, claim
   scope, non-claims, owner, and provenance.
6. Write the Day 5 fixture-and-metric-contract artifact.

### Deliverables

- selected fixture contract
- expected row ID and metric table
- tolerance and skip/defer rules
- draft selected manifest metadata
- Day 5 fixture-and-metric-contract artifact

### Completion Criteria

- fixture and metric contract is complete before runner implementation
- selected rows can be validated deterministically
- non-parity wording is explicit before docs are updated

---

## Day 6: Helper And Fixture Implementation

**Title:** Helper Implementation
**Theme:** Implement or extend source-controlled helper logic and fixture tests
for the selected family
**Time estimate:** 12 hours

### Tasks

1. Add or extend the source-controlled dense-reference helper needed by the
   selected comparison family.
2. Add focused helper tests for success, expected values, and failure/skip
   diagnostics.
3. Add or extend C fixture tests if the selected family needs additional
   source-controlled project-side proof.
4. Ensure helper output is deterministic and suitable for report generation.
5. Update working notes with implementation decisions and any fixture
   compromises.
6. Write the Day 6 helper-and-fixture-implementation artifact.

### Deliverables

- helper implementation or extension
- focused helper tests
- relevant C fixture tests if needed
- implementation notes
- Day 6 helper-and-fixture-implementation artifact

### Completion Criteria

- selected fixture can be evaluated by source-controlled project and baseline
  logic
- helper diagnostics are clear for missing optional dependencies or invalid
  inputs
- any C changes have an identified validation path

---

## Day 7: Runner Extension Design

**Title:** Runner Design
**Theme:** Design the comparison runner extension, output schema, and
validation behavior for the selected family
**Time estimate:** 12 hours

### Tasks

1. Map the selected family into `scripts/run_external_comparison.py` target
   registration and target-specific configuration.
2. Design project probe generation, compilation/execution, baseline helper
   invocation, output parsing, and error handling.
3. Define generated file set: project observations, baseline observations,
   dependency status, study TSV, summary, and manifest.
4. Define self-check behavior and focused runner tests.
5. Confirm output row IDs and counts match the Day 5 contract.
6. Write the Day 7 runner-extension-design artifact.

### Deliverables

- runner extension design
- output file contract
- runner test plan
- self-check plan
- Day 7 runner-extension-design artifact

### Completion Criteria

- runner implementation can proceed without revisiting the fixture contract
- generated output shape matches existing selected comparison patterns
- failure behavior is defined before code changes

---

## Day 8: Runner Implementation

**Title:** Runner Implementation
**Theme:** Extend comparison runner logic and tests for the selected family
**Time estimate:** 12 hours

### Tasks

1. Implement the selected target in `scripts/run_external_comparison.py`.
2. Add focused runner tests for target registration, generated rows, row IDs,
   dependency status, and summary/manifest behavior.
3. Add or update self-check coverage for the selected family.
4. Generate local comparison output for inspection without staging generated
   artifacts.
5. Fix implementation issues found by focused tests.
6. Write the Day 8 runner-implementation artifact.

### Deliverables

- comparison runner extension
- focused runner tests
- local generated output inspection notes
- Day 8 runner-implementation artifact

### Completion Criteria

- selected family can generate deterministic comparison output locally
- runner tests cover success and important failure paths
- generated local artifacts remain ignored and unstaged

---

## Day 9: Report Integration

**Title:** Report Integration
**Theme:** Register selected report metadata, report-family semantics, and
freshness integration for the new family
**Time estimate:** 12 hours

### Tasks

1. Add selected target metadata to
   `tests/corpus/manifests/selected_report_targets.tsv`.
2. Update `tests/corpus/manifests/report_families.tsv` if a new comparison
   subfamily row is required.
3. Update normalizer or freshness checks only if the selected comparison
   contract needs new behavior.
4. Update selected manifest and report-index tests for the new row count,
   expected row IDs, required files, and artifact metadata.
5. Run selected comparison freshness locally and inspect diagnostics.
6. Write the Day 9 report-integration artifact.

### Deliverables

- selected target manifest row
- report-family manifest row if needed
- updated report-index and selected manifest tests
- selected freshness integration notes
- Day 9 report-integration artifact

### Completion Criteria

- selected report metadata is manifest-backed
- expected row IDs, row counts, required files, and artifact names validate
- selected freshness checks include the new family

---

## Day 10: Freshness Gate And Workflow Guard Update

**Title:** Freshness Gate
**Theme:** Ensure selected freshness commands, workflow guard tests, and
hosted artifact expectations include the new family safely
**Time estimate:** 12 hours

### Tasks

1. Update `make report-index-comparison-freshness` behavior if the new target
   needs explicit inclusion.
2. Update Linux/macOS selected comparison workflow expectations if the new
   selected report artifact participates in hosted lanes.
3. Update `tests/test_selected_comparison_workflow.py` for selected target
   lists, artifact paths, row counts, and fail-closed uploads.
4. Confirm Windows report freshness remains formally deferred unless this
   sprint explicitly changes that boundary.
5. Add drift tests for missing artifact paths, wrong row counts, or stale
   workflow metadata as needed.
6. Write the Day 10 freshness-gate-and-workflow-guard artifact.

### Deliverables

- freshness command integration
- workflow guard updates
- hosted artifact expectation updates
- Windows non-promotion confirmation
- Day 10 freshness-gate-and-workflow-guard artifact

### Completion Criteria

- selected freshness gate includes the new family where intended
- hosted workflow guard behavior matches selected manifest metadata
- Windows report freshness non-claim remains intact unless deliberately
  superseded

---

## Day 11: Documentation Alignment

**Title:** Documentation Alignment
**Theme:** Update public, maintainer, solver-selection, corpus, and
report-index docs with bounded comparison claims
**Time estimate:** 12 hours

### Tasks

1. Update README comparison and solver-family claim wording for the selected
   family.
2. Update solver-selection documentation if the selected family affects solver
   guidance.
3. Update `docs/maintainer_guide.md` with regeneration, freshness, row ID,
   support tier, and non-claim guidance.
4. Update corpus/report-index docs and schema docs for the selected family.
5. Scan docs for broad parity, package, platform, performance, release, or
   state-of-the-art wording that would overclaim the new evidence.
6. Write the Day 11 documentation-alignment artifact.

### Deliverables

- README updates
- solver-selection updates if applicable
- maintainer guide updates
- corpus/report-index doc updates
- claim-boundary scan notes
- Day 11 documentation-alignment artifact

### Completion Criteria

- users and maintainers can identify the new bounded comparison claim
- docs do not imply broad external-library, platform, package, performance,
  release, or state-of-the-art claims
- selected manifest remains the source of truth for target metadata

---

## Day 12: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run focused comparison, report freshness, script, manifest,
workflow, docs, and relevant C validation
**Time estimate:** 12 hours

### Tasks

1. Run helper and runner unit tests for the selected family.
2. Run comparison generation and selected comparison freshness checks.
3. Run schema, selected manifest, report-index, and workflow guard tests.
4. Run relevant C fixture tests if C files or solver-facing fixtures changed.
5. Run package-manager and static package/shared ABI deferral guards if docs
   or claim wording touched those surfaces.
6. Run formatting, linting, and whitespace checks required by the changed
   surface.

### Deliverables

- validation command output summary
- fixed validation failures if any
- generated-artifact staging check
- residual risk notes
- Day 12 integrated-validation artifact

### Completion Criteria

- all feasible local checks pass or blockers are explicit
- generated local comparison outputs remain unstaged
- validation covers both new evidence and retained non-claims

---

## Day 13: Claim Review And Hardening

**Title:** Claim Review
**Theme:** Reconcile implementation, reports, manifests, docs, validation,
and residual risks before closeout
**Time estimate:** 12 hours

### Tasks

1. Reconcile Sprint 183 project-plan items against produced artifacts and
   changed files.
2. Confirm generated rows, selected manifest metadata, workflow behavior,
   documentation wording, and tests describe the same bounded family claim.
3. Review broad-claim scans and non-claim wording for stale or overbroad
   statements.
4. Harden diagnostics if failures do not name the target, row, artifact, or
   command to fix.
5. Prepare retrospective inputs and Sprint 184 handoff notes.
6. Write the Day 13 claim-review-and-hardening artifact.

### Deliverables

- project-plan reconciliation
- claim-boundary review notes
- diagnostic hardening changes if needed
- retrospective inputs
- Day 13 claim-review-and-hardening artifact

### Completion Criteria

- new comparison claim is internally consistent across code, docs, manifests,
  reports, and tests
- retained non-claims are explicit
- remaining risks are clear enough for Sprint 184 planning

---

## Day 14: Closeout And Handoff

**Title:** Closeout
**Theme:** Finalize Sprint 183 closeout records, validation summary, handoff
notes, and PR-ready claim-boundary summary
**Time estimate:** 12 hours

### Tasks

1. Re-read all Sprint 183 artifacts and working notes for consistency.
2. Finalize closed claim, validation summary, changed-surface summary, and
   handoff notes.
3. Confirm no generated build/report artifacts are staged.
4. Confirm final C/header change status and whether full C quality gate was
   required and run.
5. Record final validation commands, residual risks, and sequential-only
   checks.
6. Write the Day 14 closeout-and-handoff artifact.

### Deliverables

- Sprint 183 closeout artifact
- final validation and changed-surface summary
- Sprint 184 handoff notes
- retrospective inputs
- PR-ready risk and claim-boundary summary

### Completion Criteria

- Sprint 183 deliverables are complete and internally consistent
- working notes and artifacts are ready for retrospective creation
- branch is ready for final validation, retrospective, commit, push, and PR
