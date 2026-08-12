# Sprint 154 Plan: External Comparison Harness And First Narrow Study

**Sprint Duration:** 14 days
**Goal:** Build the first direct external comparison harness and publish one
narrow evidence-backed comparison study without overclaiming ecosystem parity.
This sprint implements the Sprint 154 section of
`docs/planning/EPIC_13/PROJECT_PLAN.md`.

**Starting Point:** Sprint 154 begins from:
- Sprint 150 QR maintained corpus family expansion available;
- Sprint 151 partial-SVD maintained corpus family expansion available;
- Sprint 152 generated report freshness policy available;
- Sprint 153 static-first package and shared-library ABI decision available;
- current report-index normalization and freshness semantics available;
- current documentation explicitly rejects broad state-of-the-art,
  shared-library, package-manager, and ecosystem parity claims.

The sprint must:
- select one narrow comparison target with maintained fixtures, tolerances, and
  claim boundaries;
- define dependency pinning, optional external tool rules, skip/defer
  semantics, and provenance expectations;
- define comparison output schema and report-index meaning;
- implement the first comparison harness for the selected target;
- generate and publish one narrow comparison study artifact;
- update docs and non-claims so comparison evidence is not confused with broad
  ecosystem parity;
- run focused validation and full quality gates if `.c` or public `.h` files
  change;
- leave Sprint 155 a clean adoption/API handoff.

**End State:** Sprint 154 leaves behind:
- external comparison target decision;
- dependency and skip/defer policy;
- comparison output schema;
- first narrow comparison harness;
- first narrow comparison study;
- generated comparison report rows or a documented deferral if rows are not
  ready for normalization;
- updated documentation and public non-claims;
- explicit residuals and Sprint 155 adoption/API handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 154 project-plan estimate.

---

## Day 1: Sprint Intake And Comparison Boundary

**Title:** Intake Boundary
**Theme:** Establish Sprint 154 scope, artifact structure, comparison
non-claims, and starting evidence
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 154 section of
   `docs/planning/EPIC_13/PROJECT_PLAN.md`.
2. Review Sprint 150, Sprint 151, Sprint 152, and Sprint 153 handoff
   artifacts.
3. Create Sprint 154 working notes and artifact directory structure.
4. Inventory current corpus fixtures, expected rows, oracle/report commands,
   report-family metadata, and comparison-related documentation.
5. Define stop conditions for external-library parity, performance,
   package-manager, shared-library, hosted CI, and state-of-the-art claims.
6. Write the Day 1 comparison-boundary artifact.

### Deliverables
- Sprint 154 working-notes baseline
- artifact directory structure
- starting evidence inventory
- comparison non-claim register
- Day 1 comparison-boundary artifact

### Completion Criteria
- Sprint 154 scope is tied to maintained corpus and report evidence
- unsupported comparison claims are explicitly blocked before target selection
- Sprint 153 static-first package boundary is preserved

---

## Day 2: Target Candidate Audit

**Title:** Candidate Audit
**Theme:** Compare QR and partial-SVD candidate families for first narrow
external study feasibility
**Time estimate:** 12 hours

### Tasks
1. Audit Sprint 150 QR fixtures, expected rows, tolerances, and proof owners.
2. Audit Sprint 151 partial-SVD fixtures, expected rows, tolerances, and proof
   owners.
3. Identify external libraries or tools that can plausibly run the selected
   fixtures locally.
4. Score candidates for dependency availability, portability, metric clarity,
   report integration cost, and overclaim risk.
5. Identify candidate-specific skip/defer blockers.
6. Write the target candidate audit artifact.

### Deliverables
- QR comparison candidate assessment
- partial-SVD comparison candidate assessment
- external baseline candidate list
- risk and feasibility scorecard
- Day 2 target candidate audit artifact

### Completion Criteria
- at least two candidate families have comparable feasibility notes
- external dependency risks are visible before target selection
- the first study can be selected without broad parity claims

---

## Day 3: Comparison Target Selection

**Title:** Target Selection
**Theme:** Select one narrow comparison target and freeze its fixtures,
metrics, tolerances, and non-claims
**Time estimate:** 12 hours

### Tasks
1. Review Day 2 candidate scoring.
2. Select one narrow comparison target, likely QR or partial-SVD.
3. Select exact fixtures and expected metrics for the first study.
4. Define accepted metric types, tolerances, caveats, and failure statuses.
5. Record explicitly deferred comparison targets and why they remain deferred.
6. Write the comparison target decision artifact.

### Deliverables
- selected comparison target
- fixture and metric selection
- tolerance and caveat policy
- deferred-target register
- Day 4 dependency-policy handoff

### Completion Criteria
- the sprint has one selected comparison target
- selected target is small enough to finish in the remaining days
- non-selected comparison claims are explicitly deferred

---

## Day 4: Dependency Pinning Policy

**Title:** Dependency Policy
**Theme:** Define dependency versions, discovery paths, optional behavior,
provenance, and skip/defer semantics
**Time estimate:** 12 hours

### Tasks
1. Identify the external baseline dependency or tool for the selected target.
2. Define version capture, executable/library discovery, and local
   installation assumptions.
3. Define optional dependency behavior and what counts as skip versus failure.
4. Define provenance fields for command, version, platform, compiler, fixture,
   and artifact path.
5. Define security and reproducibility boundaries for externally installed
   tools.
6. Write the dependency pinning policy artifact.

### Deliverables
- dependency version policy
- local discovery policy
- optional skip/defer semantics
- provenance field list
- Day 5 schema handoff

### Completion Criteria
- missing external dependencies fail or skip with explicit policy
- dependency output can be traced to a version and command
- no package-manager support claim is implied by dependency discovery

---

## Day 5: Comparison Output Schema Design

**Title:** Output Schema
**Theme:** Design row fields and status semantics for the comparison harness
and report-index integration
**Time estimate:** 12 hours

### Tasks
1. Inventory existing report-index schemas and family metadata.
2. Define comparison row fields for library, version, platform, compiler,
   fixture, metric, tolerance, status, caveat, and artifact path.
3. Define pass/fail/skip/defer/error status semantics.
4. Define how local generated comparison rows should be normalized or kept
   artifact-only.
5. Define stale-output and unsupported-claim checks.
6. Write the output schema design artifact.

### Deliverables
- comparison row schema proposal
- status and caveat semantics
- report-index integration design
- stale-output policy
- Day 6 harness-design handoff

### Completion Criteria
- comparison rows have enough provenance for audit
- skipped/deferred rows cannot be counted as comparison proof
- report integration cannot widen claims by accident

---

## Day 6: Harness Architecture Design

**Title:** Harness Design
**Theme:** Design scripts, helpers, command flow, fixtures, outputs, and
failure messages for the first comparison harness
**Time estimate:** 12 hours

### Tasks
1. Inspect existing corpus generators, oracle scripts, report-index scripts,
   and test helpers.
2. Design project-side command execution for the selected fixtures.
3. Design external-baseline command execution and output parsing.
4. Design fixture selection, temporary output paths, reproducibility metadata,
   and cleanup behavior.
5. Design failure messages for missing dependencies, malformed output,
   tolerance misses, and unsupported scope.
6. Write the harness architecture artifact.

### Deliverables
- harness command-flow design
- project-side runner design
- external-baseline runner design
- output and cleanup policy
- Day 7 implementation checklist

### Completion Criteria
- implementation can proceed without ad hoc parsing decisions
- missing dependencies and malformed output have planned diagnostics
- generated artifacts remain local unless intentionally source-controlled

---

## Day 7: Harness Implementation Batch 1

**Title:** Runner Batch
**Theme:** Implement the project-side runner, fixture selection, metadata
capture, and output scaffold
**Time estimate:** 12 hours

### Tasks
1. Implement the selected fixture enumeration for the comparison harness.
2. Implement project-side command execution or direct project metric
   extraction.
3. Implement version, platform, compiler, command, and fixture metadata
   capture.
4. Implement output directory and artifact naming conventions.
5. Add focused smoke checks for fixture selection and project-side output.
6. Record Day 7 validation results.

### Deliverables
- project-side comparison runner scaffold
- fixture selection implementation
- metadata capture implementation
- focused smoke validation
- Day 8 implementation handoff

### Completion Criteria
- project-side output is deterministic for selected fixtures
- metadata is present before external-baseline integration
- failures are readable and scoped to the selected target

---

## Day 8: Harness Implementation Batch 2

**Title:** Baseline Batch
**Theme:** Implement external-baseline discovery, execution, parsing, and
skip/defer behavior
**Time estimate:** 12 hours

### Tasks
1. Implement external dependency discovery according to Day 4 policy.
2. Implement external-baseline command execution for selected fixtures.
3. Implement output parsing and metric extraction for the selected comparison.
4. Implement skip/defer behavior for missing optional dependencies.
5. Add focused smoke checks for dependency discovery and skip behavior.
6. Record Day 8 validation results.

### Deliverables
- external-baseline discovery implementation
- external-baseline execution path
- parsed metric output
- skip/defer diagnostics
- Day 9 comparison-logic handoff

### Completion Criteria
- missing external dependencies do not produce false failures or false proof
- external output is versioned and traceable
- selected metrics can be compared against project output

---

## Day 9: Comparison Logic Implementation

**Title:** Compare Batch
**Theme:** Implement metric comparison, tolerance evaluation, row emission, and
artifact summaries
**Time estimate:** 12 hours

### Tasks
1. Implement metric matching between project output and external baseline
   output.
2. Implement tolerance evaluation and status assignment.
3. Emit comparison rows using the Day 5 schema.
4. Emit a human-readable narrow study summary artifact.
5. Add focused tests or smoke checks for pass, fail, skip, and malformed-output
   cases.
6. Record Day 9 validation results.

### Deliverables
- comparison evaluator
- generated comparison rows
- narrow study summary scaffold
- focused status-semantics checks
- Day 10 report-integration handoff

### Completion Criteria
- comparison rows are deterministic for selected fixtures
- skipped/deferred rows are not counted as proof
- tolerance failures are clear and reproducible

---

## Day 10: Report Integration Design

**Title:** Report Design
**Theme:** Decide how comparison rows enter report-index semantics without
promoting broad external-library parity
**Time estimate:** 12 hours

### Tasks
1. Review generated comparison rows from Day 9.
2. Decide whether to add a new report family, extend an existing family, or
   keep the first study artifact-only.
3. Define normalization, freshness, row-count, stale-output, and caveat
   behavior for the selected path.
4. Define report-index non-claims for local-only and optional dependency
   evidence.
5. Write the report integration design artifact.
6. Prepare Day 11 implementation checklist.

### Deliverables
- report integration product decision
- comparison row normalization policy
- freshness and caveat policy
- non-claim register
- Day 11 implementation checklist

### Completion Criteria
- generated comparison evidence has an explicit report status
- local-only comparison output cannot be cited as hosted CI proof
- broad parity claims remain blocked

---

## Day 11: Report Integration Implementation

**Title:** Report Batch
**Theme:** Implement selected report-index, schema, metadata, command, or
artifact-only behavior for comparison output
**Time estimate:** 12 hours

### Tasks
1. Implement the selected report-family or artifact-only integration.
2. Add schema, metadata, or normalizer updates needed for comparison rows.
3. Add focused tests for comparison row normalization and freshness semantics.
4. Add maintained local command or documented manual command as selected.
5. Run focused report-index validation.
6. Record Day 11 validation results.

### Deliverables
- implemented report integration or artifact-only policy
- schema/metadata updates if selected
- focused report-index tests
- maintained comparison command or documented command
- Day 12 documentation handoff

### Completion Criteria
- report checks pass for the selected comparison integration
- generated comparison output has clear freshness meaning
- optional dependency absence cannot create pass evidence

---

## Day 12: Documentation Alignment

**Title:** Docs Alignment
**Theme:** Align maintainer, report, solver, README, and public claim wording
with the first narrow comparison study
**Time estimate:** 12 hours

### Tasks
1. Update maintainer guidance for running and interpreting the comparison
   harness.
2. Update report-index or benchmark/report documentation for comparison rows
   or artifact-only policy.
3. Update solver or algorithm docs for the selected narrow study.
4. Update README or public docs with bounded comparison wording if appropriate.
5. Search for stale or overbroad external-library, parity, state-of-the-art,
   package-manager, shared-library, and performance wording.
6. Write the documentation alignment artifact.

### Deliverables
- updated maintainer comparison guidance
- updated report/solver/public docs as applicable
- stale wording search results
- documentation alignment artifact
- Day 13 validation handoff

### Completion Criteria
- docs explain how to run and interpret the comparison harness
- docs preserve static-first and local-only evidence boundaries
- no active wording claims broad ecosystem or state-of-the-art parity

---

## Day 13: Integrated Validation And Study Publication

**Title:** Study Publish
**Theme:** Run the first narrow study, publish its artifact, and validate the
affected harness, report, docs, and code surfaces
**Time estimate:** 12 hours

### Tasks
1. Run the comparison harness for the selected target.
2. Publish the first narrow comparison study artifact with caveats and
   provenance.
3. Run focused harness tests, report-index checks, stale wording checks, and
   whitespace validation.
4. Determine whether `.c` or public `.h` files changed and run the required
   full quality gate if needed.
5. Record residual comparative gaps and failed/deferred lanes.
6. Write the integrated validation and study publication artifact.

### Deliverables
- first narrow comparison study artifact
- integrated validation results
- residual comparative gap register
- quality-gate decision
- Day 14 closeout handoff

### Completion Criteria
- the study artifact is reproducible from recorded commands and versions
- all required focused or full quality gates pass
- residual comparative gaps are explicit and not hidden in docs

---

## Day 14: Closeout And Sprint 155 Handoff

**Title:** Closeout
**Theme:** Finalize Sprint 154 artifacts, evidence boundaries, residuals, and
Sprint 155 adoption/API handoff
**Time estimate:** 12 hours

### Tasks
1. Finalize `WORKING_NOTES.md` with day-by-day completion notes and validation
   status.
2. Finalize all Sprint 154 artifacts and ensure links point to current paths.
3. Prepare Sprint 154 retrospective inputs: selected target, dependency
   policy, schema, harness, report integration, study, validation, and
   residuals.
4. Write the Sprint 155 adoption/API handoff.
5. Run final status, whitespace, stale-reference, comparison/report, and
   quality checks required by changed files.
6. Record closeout summary.

### Deliverables
- finalized Sprint 154 working notes
- complete Sprint 154 artifact set
- first narrow comparison study summary
- Sprint 155 adoption/API handoff
- final closeout checklist

### Completion Criteria
- Sprint 154 has a complete artifact and validation trail
- comparison evidence is bounded to the selected target and fixtures
- Sprint 155 can begin without rediscovering comparison non-claims
