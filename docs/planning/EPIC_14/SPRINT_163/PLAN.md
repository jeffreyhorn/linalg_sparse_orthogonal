# Sprint 163 Plan: Methodology-Bound Performance Publication

**Sprint Duration:** 14 days
**Goal:** Publish a methodology-bound performance/report artifact for selected
canonical benchmark and sentinel rows while preserving non-superiority claims.
This sprint implements the Sprint 163 section of
`docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references the older Epic 12 project-plan
path, but the current Sprint 163 project-plan section lives in
`docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Starting Point:** Sprint 163 begins from:
- existing benchmark governance and sentinel semantics;
- hosted/report publication lessons from Sprint 159;
- Sprint 162 package decision closed with Windows CMake-first package proof,
  metadata-only Windows `sparse.pc` inspection, and retained Windows
  Makefile/`pkg-config` non-claims;
- static-first package and ABI non-claims preserved;
- public claims constrained away from portable superiority,
  state-of-the-art, package-manager, shared-library, dynamic ABI,
  runtime-loader, and broad platform claims.

The sprint must:
- select canonical benchmark and sentinel rows narrow enough to publish with
  methodology rather than superiority claims;
- define platform, compiler, build, thread, fixture, repeat, variance,
  threshold, caveat, and provenance fields before changing reports;
- update benchmark/report scripts only where methodology fields are missing;
- classify hard timing gates, threshold-free reports, local-only rows,
  supplemental rows, advisory rows, and hosted publication candidates;
- align README, benchmark docs, maintainer guide, and report-index wording;
- run selected benchmark/report/sentinel commands and focused script checks;
- leave Sprint 164 with an API-header handoff grounded in performance
  evidence boundaries.

**End State:** Sprint 163 leaves behind:
- methodology-bound performance report;
- updated benchmark/report schema or documentation;
- clear performance non-superiority boundaries;
- validation record for selected benchmark/report/sentinel commands;
- Sprint 164 API-header handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 163 project-plan estimate.

---

## Day 1: Sprint Intake And Performance Surface Inventory

**Title:** Sprint Intake
**Theme:** Establish Sprint 163 scope, artifact layout, and current
performance evidence surfaces
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 163 section of
   `docs/planning/EPIC_14/PROJECT_PLAN.md`.
2. Review Sprint 159 hosted/report publication artifacts and Sprint 162
   package-boundary handoff.
3. Create Sprint 163 working notes and artifact directory structure.
4. Inventory benchmark, sentinel, report-index, generated report,
   documentation, CI, and maintainer-guide performance surfaces.
5. Record explicit non-goals for portable superiority, state-of-the-art,
   broad platform performance, package evidence reuse, package-manager,
   shared-library ABI, dynamic-loader, and runtime-backend superiority claims.
6. Write the Day 1 sprint-intake artifact.

### Deliverables
- Sprint 163 working-notes baseline
- artifact directory structure
- performance surface inventory
- non-goal and assumption register
- Day 1 sprint-intake artifact

### Completion Criteria
- Sprint 163 scope is tied to the Epic 14 project plan
- current benchmark/report proof owners are identified
- performance work is separated from Sprint 162 package proof

---

## Day 2: Canonical Row Candidate Inventory

**Title:** Row Inventory
**Theme:** Inventory candidate benchmark and sentinel rows for publication
**Time estimate:** 12 hours

### Tasks
1. Inspect benchmark targets, generated report scripts, sentinel scripts, and
   report-index rows for candidate publication surfaces.
2. Identify rows with stable fixture provenance, repeat semantics,
   thresholds, and local or hosted execution ownership.
3. Separate benchmark timing rows from correctness, package, corpus,
   comparison, and generated-report freshness rows.
4. Reject rows that cannot carry methodology fields or that imply broad
   performance superiority.
5. Build the candidate row register with owners, commands, current artifacts,
   blockers, and publication risk.
6. Write the Day 2 row-inventory artifact.

### Deliverables
- canonical row candidate register
- rejected-candidate notes
- command and owner map
- performance versus non-performance evidence separation
- Day 2 row-inventory artifact

### Completion Criteria
- candidate rows are source-backed and reproducible
- non-performance rows are excluded from performance publication
- row blockers are documented before selection

---

## Day 3: Surface Selection

**Title:** Surface Selection
**Theme:** Select the canonical benchmark and sentinel rows to publish
**Time estimate:** 12 hours

### Tasks
1. Score candidate rows by reproducibility, methodology completeness,
   maintenance cost, runtime cost, hosted suitability, user value, and claim
   risk.
2. Select the smallest row set that can close a methodology-bound publication
   claim in one sprint.
3. Define which rows are published, local-only, supplemental, advisory, or
   deferred.
4. Map selected rows to commands, generated outputs, report-index rows,
   documentation owners, and validation requirements.
5. Define stop conditions if selected rows are flaky or missing methodology
   fields.
6. Write the Day 3 surface-selection artifact.

### Deliverables
- selected canonical benchmark and sentinel rows
- row classification table
- selected command map
- deferred row register
- Day 3 surface-selection artifact

### Completion Criteria
- selected row set is narrow enough to close
- selected rows can be validated without broad performance claims
- deferred rows have explicit blockers

---

## Day 4: Methodology Contract

**Title:** Methodology Contract
**Theme:** Define required methodology fields and caveats before report edits
**Time estimate:** 12 hours

### Tasks
1. Define platform, operating system, compiler, compiler version, build flags,
   thread count, backend/runtime settings, fixture, matrix size, repeat count,
   warmup, variance, threshold, and date/provenance fields.
2. Define which fields are required for published rows, supplemental rows,
   advisory rows, and local-only rows.
3. Define missing, stale, malformed, skipped, deferred, and failed row
   semantics.
4. Define threshold semantics for hard timing gates versus threshold-free
   methodology reports.
5. Define public caveat wording for non-superiority, non-portability, and
   non-state-of-the-art claims.
6. Write the Day 4 methodology-contract artifact.

### Deliverables
- methodology field contract
- row-state semantics
- threshold and variance rules
- public caveat wording
- Day 4 methodology-contract artifact

### Completion Criteria
- implementation has exact report fields
- gate rows and publication rows are distinguished
- unsupported performance claims are blocked before edits

---

## Day 5: Report Schema And Script Gap Analysis

**Title:** Schema Gap Analysis
**Theme:** Compare current benchmark/report scripts against the methodology
contract
**Time estimate:** 12 hours

### Tasks
1. Inspect benchmark report generators, sentinel scripts, report normalizers,
   schema files, and generated report indexes.
2. Identify missing methodology fields for selected rows.
3. Identify stale or ambiguous output fields that could imply broad
   superiority or portable performance.
4. Define exact script, schema, fixture, report-index, and documentation edits
   required.
5. Define focused tests or self-checks needed for changed scripts.
6. Write the Day 5 schema-gap artifact.

### Deliverables
- methodology field gap table
- script and schema change list
- focused test map
- unsupported-wording risk register
- Day 5 schema-gap artifact

### Completion Criteria
- report enhancement work is source-backed
- edits are scoped to selected rows and fields
- required validation commands are known

---

## Day 6: Report Enhancement Implementation I

**Title:** Report Implementation I
**Theme:** Add missing methodology fields and row classification plumbing
**Time estimate:** 12 hours

### Tasks
1. Update selected benchmark/report scripts or metadata to emit missing
   methodology fields.
2. Add row classification fields for hard gate, threshold-free report,
   local-only, supplemental, advisory, hosted, skipped, and deferred states.
3. Preserve existing benchmark command behavior for unselected rows.
4. Add focused script self-checks or fixture tests where the implementation
   changes report behavior.
5. Run focused local checks for changed scripts.
6. Write the Day 6 implementation artifact.

### Deliverables
- first report enhancement patch set
- methodology field output updates
- row classification plumbing
- focused local check output
- Day 6 implementation artifact

### Completion Criteria
- selected reports can emit required methodology fields
- unselected rows are not silently promoted
- focused script checks pass or failures are documented

---

## Day 7: Report Enhancement Implementation II

**Title:** Report Implementation II
**Theme:** Complete report output, diagnostics, and stale-row handling
**Time estimate:** 12 hours

### Tasks
1. Complete remaining report-script, schema, fixture, or normalizer changes.
2. Add diagnostics for missing methodology fields, stale outputs, malformed
   rows, unsupported thresholds, and ambiguous publication states.
3. Ensure generated reports distinguish benchmark data from package, corpus,
   comparison, and correctness evidence.
4. Verify selected rows carry caveat fields and row-state semantics.
5. Re-run focused script and metadata checks.
6. Write the Day 7 implementation-completion artifact.

### Deliverables
- completed report enhancement patch set
- diagnostics for methodology and stale-row failures
- focused report/script test output
- Day 7 implementation-completion artifact

### Completion Criteria
- selected report behavior is complete locally
- diagnostics are reviewable and actionable
- unsupported performance claims fail or remain explicit non-claims

---

## Day 8: Gate Classification And Publication Policy

**Title:** Gate Classification
**Theme:** Separate hard timing gates, reports, local-only evidence, and
advisory rows
**Time estimate:** 12 hours

### Tasks
1. Classify selected rows into hard timing gates, threshold-free reports,
   local-only rows, supplemental rows, advisory rows, and deferred rows.
2. Define which rows are allowed in source-controlled reports and which remain
   generated or local-only.
3. Define CI or hosted publication expectations for selected rows.
4. Define policy for variance, drift, runtime budget, flaky timing, skipped
   rows, and stale generated outputs.
5. Update report-index or planning evidence wording as needed.
6. Write the Day 8 gate-classification artifact.

### Deliverables
- gate classification matrix
- publication policy
- local-only and hosted-only row notes
- stale/skip/defer rules
- Day 8 gate-classification artifact

### Completion Criteria
- hard gates are not confused with methodology reports
- publication eligibility is explicit
- local-only evidence cannot be cited as hosted proof

---

## Day 9: Documentation Alignment I

**Title:** Benchmark Docs I
**Theme:** Align benchmark README and report documentation with methodology
fields
**Time estimate:** 12 hours

### Tasks
1. Update benchmark README or equivalent benchmark docs for selected commands,
   methodology fields, row states, and caveats.
2. Document how to regenerate selected benchmark/report artifacts.
3. Document hard-gate versus threshold-free-report semantics.
4. Add warnings against portable superiority, broad platform, and
   state-of-the-art claims.
5. Link selected rows to evidence owners and validation commands.
6. Write the Day 9 benchmark-docs artifact.

### Deliverables
- updated benchmark/report documentation
- regeneration command notes
- gate versus report wording
- non-superiority caveats
- Day 9 docs artifact

### Completion Criteria
- users can reproduce selected reports
- docs explain methodology fields and caveats
- unsupported performance claims remain blocked

---

## Day 10: Documentation Alignment II

**Title:** Public Docs II
**Theme:** Align README, maintainer guide, and report-index wording
**Time estimate:** 12 hours

### Tasks
1. Update README performance-summary wording for the selected methodology-bound
   publication.
2. Update maintainer guide instructions for benchmark/report publication,
   freshness, row states, and validation commands.
3. Update report-index schema or wording where selected rows require new
   fields or states.
4. Ensure docs do not reuse Sprint 162 package proof as performance evidence.
5. Scan for state-of-the-art, superiority, package-manager, shared-library,
   dynamic ABI, runtime-loader, and broad platform wording drift.
6. Write the Day 10 public-docs artifact.

### Deliverables
- updated README and maintainer guidance
- report-index wording or schema notes
- unsupported-claim scan notes
- package/performance separation notes
- Day 10 docs artifact

### Completion Criteria
- public docs match selected performance evidence
- package and performance evidence remain separate
- non-superiority boundaries are explicit

---

## Day 11: Selected Benchmark And Sentinel Validation

**Title:** Selected Validation
**Theme:** Run selected benchmark, report, sentinel, and focused script checks
**Time estimate:** 12 hours

### Tasks
1. Run selected benchmark/report commands for the canonical row set.
2. Run selected sentinel commands or freshness checks.
3. Run focused script tests, schema checks, or normalizer checks touched by
   implementation.
4. Capture output summaries, row states, failures, skips, and local-only
   limitations.
5. Fix validation-driven script or documentation issues.
6. Write the Day 11 selected-validation artifact.

### Deliverables
- selected benchmark/report command output
- sentinel command output
- focused script/schema check output
- validation-driven fix notes
- Day 11 validation artifact

### Completion Criteria
- selected local validation passes or failures are explicitly blocked
- generated rows match the methodology contract
- local-only limitations are documented

---

## Day 12: Cross-Surface Validation And Quality Gate

**Title:** Cross-Surface Validation
**Theme:** Re-run affected report, docs, package-boundary, and code quality
checks
**Time estimate:** 12 hours

### Tasks
1. Re-run selected benchmark/report/sentinel checks after Day 11 fixes.
2. Run report-index, schema, freshness, or generated-artifact checks affected
   by this sprint.
3. Run documentation hygiene and unsupported-claim scans.
4. Run package-boundary checks if docs mention package or platform support.
5. Run `git diff --check` and trailing-whitespace scans.
6. If `.c` or `.h` files changed, run `make format`, `make lint`, and
   `make test`.

### Deliverables
- cross-surface validation record
- changed-file quality-gate decision
- hosted-only verification checklist
- failure or residual notes
- Day 12 validation artifact

### Completion Criteria
- validation matches the changed-file surface
- required checks pass before closeout
- hosted-only expectations are explicit

---

## Day 13: Evidence And Claim Review

**Title:** Evidence Review
**Theme:** Review performance claims, report rows, docs, and residuals as one
evidence surface
**Time estimate:** 12 hours

### Tasks
1. Trace each positive performance claim to selected benchmark/report/sentinel
   commands, generated rows, docs, and validation evidence.
2. Trace each retained non-claim to unsupported-surface guards or explicit
   documentation.
3. Review report rows and docs for portable superiority, state-of-the-art,
   broad platform, package, ABI, runtime-loader, and backend-superiority
   wording.
4. Review diffs for stale outputs, ambiguous timing terminology, and
   unsupported evidence assertions.
5. Finalize Sprint 164 API-header handoff.
6. Write the Day 13 evidence-review artifact.

### Deliverables
- claim-to-evidence trace
- retained non-claim trace
- report and docs wording review
- Sprint 164 API-header handoff
- Day 13 evidence-review artifact

### Completion Criteria
- performance publication is reviewable end to end
- positive wording is bounded by actual methodology evidence
- Sprint 164 handoff is ready

---

## Day 14: Closeout And Retrospective Prep

**Title:** Closeout
**Theme:** Finalize Sprint 163 artifacts, validation record, residuals, and
retrospective inputs
**Time estimate:** 12 hours

### Tasks
1. Re-run final targeted checks required by the changed-file surface.
2. Update Sprint 163 working notes with final decisions, commands, outputs,
   and hosted-only expectations.
3. Finalize closeout artifacts for selected rows, methodology contract,
   report changes, documentation, validation, residuals, and Sprint 164
   handoff.
4. Review changed files for claim wording, stale generated outputs, and
   unsupported performance evidence assertions.
5. Prepare retrospective inputs from artifacts and working notes.
6. Record the Day 14 closeout artifact.

### Deliverables
- final validation record
- selected performance publication closeout notes
- residual queue
- complete working notes
- retrospective input set
- Day 14 closeout artifact

### Completion Criteria
- Sprint 163 deliverables are complete and traceable
- validation status is recorded with exact commands
- Sprint 164 API-header handoff is ready
