# Sprint 202 Plan: Hosted Selected Benchmark Freshness on One Additional Platform

**Sprint Duration:** 14 days
**Goal:** Add one hosted selected benchmark freshness lane outside the current
Linux-only selected performance proof, without claiming portable performance.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 202 estimate in the Epic 18
project plan.

**Primary scope:** Select exactly one additional hosted platform and one
benchmark row, define methodology-bound benchmark metadata, add the workflow
lane and selected artifact freshness validation, expand missing/stale/duplicate
and malformed artifact tests, calibrate docs to avoid portable performance
claims, and validate the lane and evidence.

**Non-goals:** Portable performance claims, broad benchmark matrix expansion,
Linux benchmark methodology rewrites, Homebrew/package-manager work, public API
or ABI changes, solver algorithm changes, benchmark threshold promotion, release
readiness claims, or state-of-the-art performance claims.

---

## Day 1: Benchmark Freshness Intake

**Title:** Freshness Intake
**Theme:** Establish Sprint 202 scope, inherited methodology, and current
selected benchmark evidence.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 202 Epic 18 project-plan section and map items 202.1
   through 202.6 to expected artifacts.
2. Review Sprint 192 selected benchmark methodology, freshness checks, and
   threshold-free interpretation boundaries.
3. Inventory current selected benchmark artifacts, manifest rows, workflow
   lanes, validator scripts, and documentation claim surfaces.
4. Create `WORKING_NOTES.md` with an item checklist, evidence map, risk
   register, validation matrix, and open questions.
5. Record non-goals so the sprint cannot be mistaken for portable performance
   support.

### Deliverables

- Sprint 202 working-notes scaffold.
- Current selected benchmark freshness inventory.
- Item-to-artifact traceability map.
- Initial risk register and validation matrix.

### Completion Criteria

- Every Sprint 202 item has an initial evidence path or artifact category.
- Current Linux-only selected benchmark proof is understood before platform
  selection.
- No workflow or validator implementation begins before scope is recorded.

---

## Day 2: Platform And Row Candidate Ranking

**Title:** Platform Ranking
**Theme:** Rank candidate hosted platform and benchmark row pairs by evidence
value, runtime budget, and claim risk.
**Time estimate:** 12 hours

### Tasks

1. List candidate hosted platforms available in current CI and their compiler,
   shell, path, artifact, and runtime constraints.
2. List candidate selected benchmark rows and identify the row with the highest
   evidence value for cross-platform freshness.
3. Score platform/row pairs by stability, runtime cost, existing methodology
   fit, freshness diagnosability, and claim-safety risk.
4. Identify pairs that should be deferred because they imply broad performance
   or platform-support claims.
5. Record the ranked candidate table and preferred platform/row shortlist.

### Deliverables

- Platform/row candidate ranking artifact.
- Deferred candidate ledger.
- Preferred selected benchmark freshness lane shortlist.
- Item 202.1 ranking evidence.

### Completion Criteria

- Item 202.1 has a documented candidate ranking.
- The shortlist contains exactly bounded hosted lane candidates.
- Deferred candidates include explicit claim or runtime reasons.

---

## Day 3: Selected Platform And Row Decision

**Title:** Lane Selection
**Theme:** Select exactly one hosted platform and one benchmark row for Sprint
202 implementation.
**Time estimate:** 12 hours

### Tasks

1. Select one platform/row pair from the Day 2 shortlist and document why it
   closes the highest-value freshness gap.
2. Define the selected benchmark artifact path, target key, workflow job, and
   freshness validator scope.
3. Freeze out-of-scope platforms, rows, benchmark suites, and artifact families.
4. Record the no-portable-performance-claim and no-threshold-claim boundaries.
5. Identify focused validation commands and hosted evidence that must pass.

### Deliverables

- Selected platform/row decision record.
- In-scope and out-of-scope lane map.
- Claim-boundary statement.
- Focused validation checklist.

### Completion Criteria

- Item 202.1 is complete with exactly one selected platform/row pair.
- The selected lane can be reviewed independently from broader benchmarking.
- Public performance and platform-support claims remain explicitly unexpanded.

---

## Day 4: Methodology Metadata Contract

**Title:** Metadata Contract
**Theme:** Define the platform-bound benchmark metadata and interpretation
contract before workflow implementation.
**Time estimate:** 12 hours

### Tasks

1. Define required metadata fields for platform, runner, compiler, build flags,
   benchmark command, repeat policy, timestamps, and artifact scope.
2. Decide how selected row identity maps to existing manifest or report-index
   semantics.
3. Define threshold-free interpretation wording for hosted selected benchmark
   freshness.
4. Record how deferred, missing, malformed, duplicate, and stale artifacts must
   be diagnosed.
5. Identify docs and validator surfaces that must consume the same metadata
   vocabulary.

### Deliverables

- Methodology metadata contract artifact.
- Freshness diagnostic vocabulary.
- Selected artifact schema notes.
- Item 202.2 evidence record.

### Completion Criteria

- Item 202.2 has a documented metadata contract.
- Metadata does not imply benchmark comparability across platforms.
- Freshness diagnostics are defined before tests or workflow code change.

---

## Day 5: Validator And Manifest Design

**Title:** Validator Design
**Theme:** Design the selected benchmark freshness validator changes and tests
without broadening artifact semantics.
**Time estimate:** 12 hours

### Tasks

1. Locate the current benchmark report, manifest, normalizer, and freshness
   validation code paths.
2. Design the minimal selected-platform freshness filter for the chosen
   platform/row pair.
3. Define path-normalization behavior for hosted artifact paths.
4. Map every planned diagnostic to a fixture or current-tree test.
5. Record registration, source-list, CMake, and docs impacts before editing.

### Deliverables

- Validator implementation design artifact.
- Fixture and regression matrix.
- Path-normalization decision record.
- Registration impact checklist.

### Completion Criteria

- The implementation path for item 202.4 is clear before code edits.
- The design covers missing, stale, duplicate, malformed, deferred, and
  path-normalized artifacts.
- Planned changes remain selected-platform and selected-row scoped.

---

## Day 6: Freshness Validator Implementation

**Title:** Validator Implementation
**Theme:** Implement selected-platform benchmark freshness diagnostics and
artifact matching.
**Time estimate:** 12 hours

### Tasks

1. Add the selected benchmark freshness filter and metadata handling to the
   chosen validator or normalizer surface.
2. Preserve existing Linux selected benchmark freshness behavior.
3. Add path normalization for hosted platform artifact paths where needed.
4. Keep diagnostics threshold-free and selected-row scoped.
5. Run focused validator checks after implementation.

### Deliverables

- Selected-platform freshness validator changes.
- Path-normalized artifact matching.
- Initial focused validator run notes.
- Item 202.3 and 202.4 implementation evidence.

### Completion Criteria

- The new validator path accepts the selected platform/row artifact.
- Existing selected benchmark freshness tests still pass.
- No portable performance or broad platform claim is introduced.

---

## Day 7: Freshness Regression Fixtures

**Title:** Freshness Fixtures
**Theme:** Add targeted tests for selected benchmark freshness diagnostics.
**Time estimate:** 12 hours

### Tasks

1. Add a passing selected-platform benchmark artifact fixture for the chosen
   row.
2. Add missing-artifact and stale-artifact fixtures with clear diagnostics.
3. Add duplicate-artifact and malformed-metadata fixtures.
4. Add deferred-artifact and path-normalized artifact fixtures.
5. Ensure tests cover only the selected platform/row lane.

### Deliverables

- Freshness regression fixture updates.
- Missing, stale, duplicate, malformed, deferred, and path-normalized tests.
- Focused test output record.

### Completion Criteria

- Item 202.4 has direct regression coverage for every required diagnostic.
- Negative fixtures fail for the intended reason.
- Existing Linux selected benchmark freshness behavior remains covered.

---

## Day 8: Hosted Workflow Lane Implementation

**Title:** Hosted Lane
**Theme:** Add the selected hosted workflow job and artifact upload path.
**Time estimate:** 12 hours

### Tasks

1. Add or update the hosted workflow job for the selected platform/row pair.
2. Wire benchmark command execution with the Day 4 metadata fields.
3. Upload only the selected benchmark artifact required for freshness.
4. Run the selected freshness check in the hosted lane.
5. Keep runtime budget and job naming scoped to one additional platform.

### Deliverables

- Hosted workflow lane.
- Selected artifact upload path.
- Freshness check wiring.
- Runtime-budget notes.

### Completion Criteria

- Item 202.3 has workflow evidence.
- The workflow lane generates and checks exactly the selected artifact scope.
- CI wording does not imply broad hosted benchmark coverage.

---

## Day 9: Workflow Guard And Local Simulation

**Title:** Workflow Guard
**Theme:** Add guard coverage or local simulation for workflow and artifact
contract drift.
**Time estimate:** 12 hours

### Tasks

1. Add or update guard checks for selected workflow job naming, command flags,
   metadata fields, artifact upload path, and freshness invocation.
2. Add fixture-based tests for missing workflow step, wrong platform, wrong
   artifact path, and missing selected-target wiring.
3. Run local simulation or static workflow validation where hosted execution is
   unavailable locally.
4. Document any environment residuals that require hosted CI confirmation.
5. Keep the guard selected-lane scoped.

### Deliverables

- Workflow guard or validation test updates.
- Local simulation evidence.
- Hosted residual checklist.
- Drift diagnostics for workflow wiring.

### Completion Criteria

- Workflow wiring has local or static validation before hosted CI review.
- Missing or wrong selected-lane fields fail clearly.
- Hosted-only residuals are explicitly recorded.

---

## Day 10: Documentation Calibration

**Title:** Claim Calibration
**Theme:** Update benchmark and support documentation with selected-scope,
threshold-free language.
**Time estimate:** 12 hours

### Tasks

1. Update benchmark documentation to describe the selected hosted freshness
   lane, metadata contract, and threshold-free interpretation.
2. Update README, INSTALL, support matrix, and maintainer guide claim surfaces
   as needed.
3. Add explicit non-claims for portable performance, broad platform support,
   benchmark superiority, bottles/package distribution, and release readiness.
4. Cross-check terminology with Sprint 192 methodology wording.
5. Record documentation changes and claim boundaries in working notes.

### Deliverables

- Claim-safe benchmark documentation updates.
- README/INSTALL/support matrix updates if applicable.
- Maintainer guidance updates.
- Item 202.5 evidence record.

### Completion Criteria

- Item 202.5 documentation is complete.
- Documentation states hosted selected freshness without implying comparable
  performance across platforms.
- Non-claims are explicit and consistent across public-facing surfaces.

---

## Day 11: Focused Freshness Validation

**Title:** Focused Validation
**Theme:** Run focused freshness, manifest, workflow, and docs checks after the
implementation and documentation updates.
**Time estimate:** 12 hours

### Tasks

1. Run selected benchmark freshness tests and validator regression tests.
2. Run selected manifest/report-index tests affected by the implementation.
3. Run workflow guard or local workflow validation.
4. Run docs checks and stale-claim text scans.
5. Capture pass/fail output and fix any focused failure before proceeding.

### Deliverables

- Focused validation artifact.
- Freshness test results.
- Workflow guard results.
- Docs and claim-scan results.

### Completion Criteria

- Focused validation passes locally.
- Any failed diagnostic is fixed before integrated validation.
- No `.c` or `.h` quality gate is skipped if those files changed.

---

## Day 12: Integrated Validation And Hosted Evidence Review

**Title:** Integrated Validation
**Theme:** Run integrated checks and review hosted CI evidence for the selected
benchmark freshness lane.
**Time estimate:** 12 hours

### Tasks

1. Run the required local integrated validation suite for changed scripts,
   docs, manifests, workflows, and tests.
2. Run `make format && make lint && make test` if any `.c` or `.h` file changed.
3. Review hosted CI logs for the selected platform/row freshness lane.
4. Confirm artifact upload, metadata, freshness check, and diagnostics match the
   Day 4 contract.
5. Record hosted evidence residuals if the platform lane cannot be fully proven
   locally.

### Deliverables

- Integrated validation artifact.
- Hosted CI evidence review.
- Full C quality-gate record if applicable.
- Residual and rerun checklist.

### Completion Criteria

- Item 202.6 validation evidence is complete.
- Required local checks pass.
- Hosted evidence either passes or has a clearly bounded residual requiring
  user/CI follow-up.

---

## Day 13: Review Hardening

**Title:** Review Hardening
**Theme:** Audit the selected lane for unnecessary breadth, stale claims, and
freshness evidence gaps.
**Time estimate:** 11 hours

### Tasks

1. Review the diff for accidental broad benchmark, platform, workflow, or
   documentation expansion.
2. Verify every selected benchmark freshness diagnostic has a test or hosted
   evidence record.
3. Re-run targeted stale-claim and stale-path scans.
4. Confirm the residual queue names deferred platforms, rows, and benchmark
   claims.
5. Tighten docs, tests, or guard messages where review would otherwise be
   ambiguous.

### Deliverables

- Review-hardening artifact.
- Final claim-safety scan.
- Residual queue updates.
- Diff-scope audit.

### Completion Criteria

- The branch changes exactly one additional hosted selected freshness lane.
- Reviewers can trace every claim to a validation artifact.
- Deferred broad performance work remains explicit.

---

## Day 14: Closeout And Retrospective Prep

**Title:** Closeout Prep
**Theme:** Finalize Sprint 202 evidence, close residuals, and prepare
retrospective inputs.
**Time estimate:** 11 hours

### Tasks

1. Re-run final required checks for changed files and capture command output.
2. Update `WORKING_NOTES.md` with final item status, validation summary, and
   residuals.
3. Update Epic 18 project-plan and residual-queue status for Sprint 202 if
   required.
4. Prepare retrospective inputs: completed work, validation, claim boundaries,
   deferred items, and follow-up recommendations.
5. Confirm no generated files, local absolute paths, or hosted-only secrets are
   included.

### Deliverables

- Final Sprint 202 working-notes status.
- Closeout review artifact.
- Retrospective input checklist.
- Final validation and residual summary.

### Completion Criteria

- Sprint 202 artifacts are ready for retrospective creation.
- Item 202.6 has final validation evidence.
- No untracked generated output or accidental environment-specific path remains
  in the planned change set.

