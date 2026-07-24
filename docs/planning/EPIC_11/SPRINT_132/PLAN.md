# Sprint 132 Plan: Performance Sentinel & Backend Runtime Governance

**Sprint Duration:** 14 days
**Goal:** Strengthen local performance and backend/runtime governance without
turning local measurements into portable performance claims.

**Starting Point:** Sprint 132 begins from:
- Sprint 131 report-index model, freshness policy, ownership map, and residual
  assurance queue
- current benchmark, sentinel, large-matrix guardrail, coverage, and dead-code
  non-claims
- existing benchmark scripts, performance sentinel scripts, guardrail reports,
  backend selection surfaces, OpenMP build modes, and maintainer guidance
- current local report artifacts under `build/bench-reports/` when generated
- Sprint 131 decision that supplemental large-matrix lanes need runtime and
  support-tier policy before recurring validation or promotion

The sprint must:
- inventory hot compressed, direct, iterative, eigensolver, SVD, reorder, and
  backend/runtime paths with current sentinel/report coverage
- define builtin and optional dense backend observability, fallback, OpenMP,
  thread-count, and nested-runtime boundaries
- design bounded local performance sentinels with non-portable interpretation
- add or refine selected sentinel/report lanes and generated metadata only
  when owner, freshness, runtime cost, and claim boundary are explicit
- improve benchmark interpretation and report-index handoff without changing
  benchmark claims silently
- run focused benchmark/report checks and full C quality gates if code changes
- publish validation evidence, residual runtime gaps, and performance
  non-claims

**End State:** Sprint 132 leaves behind:
- hot-path and sentinel coverage inventory
- backend/runtime contract artifact
- sentinel design and implementation decision package
- updated local sentinel bundle or explicit deferrals
- benchmark/report-index documentation cleanup or no-update rationale
- validation package for affected code, scripts, reports, and docs
- performance non-claim register and Sprint 133 handoff notes

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 132 project-plan estimate.

---

## Day 1: Sprint Intake and Runtime Governance Baseline

**Title:** Runtime Intake
**Theme:** Establish Sprint 132 scope, artifact structure, validation lanes,
and non-claim fences around local performance evidence
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 132 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Review Sprint 131 report-index, ownership, freshness, residual, and
   closeout artifacts.
3. Inventory candidate source areas for benchmarks, sentinels, dense backend
   selection, OpenMP/runtime behavior, guardrails, and maintainer docs.
4. Map Sprint 132 project-plan Items 1-7 to day-level owners.
5. Create the Sprint 132 working-notes baseline and artifact directory.
6. Record duplicate fences so local timing, backend availability, thread
   counts, and supplemental reports are not promoted into portable claims.

### Deliverables
- Sprint 132 working-notes baseline
- artifact directory structure
- source-area intake list
- item-to-day owner map
- validation-lane and non-claim boundary notes

### Completion Criteria
- every Sprint 132 project-plan item has a day-level owner
- Sprint 131 report-index and non-claim boundaries are preserved
- benchmark, sentinel, backend, OpenMP, and guardrail surfaces are visible
  before design or implementation begins

---

## Day 2: Hot Path Inventory

**Title:** Hot Path Map
**Theme:** Inventory compressed, direct, iterative, eigensolver, SVD, reorder,
and backend/runtime paths with current sentinel or benchmark coverage
**Time estimate:** 12 hours

### Tasks
1. Inventory benchmark binaries and report scripts that exercise hot sparse
   paths.
2. Map compressed-format, direct-factorization, iterative/preconditioner,
   eigensolver, SVD/partial-SVD, graph/reorder, and backend dispatch paths.
3. Record current sentinel, canonical benchmark, large-matrix guardrail, and
   direct benchmark-local coverage for each path.
4. Separate reviewed structural/report lanes from supplemental and exploratory
   timing lanes.
5. Identify hot paths that lack current sentinel/report visibility.
6. Write the hot-path inventory artifact.

### Deliverables
- hot-path inventory table
- current sentinel/report coverage map
- reviewed versus supplemental lane notes
- missing-sentinel queue
- owner and validation surface map

### Completion Criteria
- every high-value hot path has current coverage status or explicit unknown
  status
- timing rows are not confused with correctness or portable performance proof
- missing coverage is recorded as an owner queue, not left implicit

---

## Day 3: Sentinel Coverage Gap Ranking

**Title:** Sentinel Gaps
**Theme:** Rank missing or weak sentinel coverage by runtime risk, user-facing
workflow, backend sensitivity, and claim impact
**Time estimate:** 12 hours

### Tasks
1. Review the Day 2 hot-path inventory and current report/sentinel coverage.
2. Rank gaps by user-facing workflow, expected runtime cost, regression risk,
   backend sensitivity, OpenMP sensitivity, corpus availability, and report
   claim impact.
3. Identify gaps suitable for local bounded sentinels versus benchmark-only or
   supplemental reports.
4. Identify paths where timing variance or platform sensitivity blocks hard
   thresholds.
5. Define candidate owners and validation commands for the highest-value gaps.
6. Write the sentinel coverage gap ranking artifact.

### Deliverables
- sentinel gap ranking table
- runtime-risk rubric
- candidate owner map
- threshold-suitability notes
- residual hot-path queue

### Completion Criteria
- sentinel gaps are ranked by risk, not convenience
- threshold-hostile paths are marked as report-only or supplemental
- every high-priority gap has an owner or explicit blocker

---

## Day 4: Backend Runtime Contract

**Title:** Runtime Contract
**Theme:** Define builtin and optional dense backend observability, fallback,
OpenMP, thread-count, and nested-runtime boundaries
**Time estimate:** 12 hours

### Tasks
1. Inventory dense backend selection, optional backend availability,
   environment variables, compile-time flags, and runtime fallback paths.
2. Inventory OpenMP build modes, thread-count controls, and nested-runtime
   behavior visible to benchmarks or sentinels.
3. Define observability fields needed in benchmark/sentinel reports for
   backend, compiler, OpenMP, thread count, and fallback state.
4. Define what backend fallback means and what it does not claim.
5. Define unsupported, unknown, and unavailable backend states for report
   rows.
6. Write the backend/runtime contract artifact.

### Deliverables
- backend/runtime contract
- fallback and unavailable-state policy
- OpenMP and thread-count boundary notes
- observability field list
- backend non-claim register

### Completion Criteria
- builtin and optional backend states are distinguishable
- OpenMP and nested-runtime boundaries are explicit
- backend observability does not imply portable performance or backend parity

---

## Day 5: Backend Runtime Metadata Design

**Title:** Backend Metadata
**Theme:** Design report metadata for backend, compiler, OpenMP, thread,
fallback, and platform context without changing benchmark claims
**Time estimate:** 12 hours

### Tasks
1. Review current benchmark, sentinel, canonical report, and guardrail
   metadata fields.
2. Compare current metadata against the Day 4 backend/runtime contract.
3. Decide which metadata fields are required, optional, deferred, or
   intentionally omitted for each report family.
4. Define row semantics for builtin backend, optional backend, unavailable
   backend, fallback, unknown backend, OpenMP mode, and thread count.
5. Identify script, documentation, and validation touch points for a metadata
   implementation batch.
6. Write the backend metadata design artifact.

### Deliverables
- backend metadata schema proposal
- report-family field matrix
- fallback row semantics
- implementation touch-point list
- metadata deferral queue

### Completion Criteria
- metadata fields trace to Day 4 contract decisions
- no metadata field creates a backend parity or portable timing claim
- implementation touch points and blockers are explicit

---

## Day 6: Sentinel Design Policy

**Title:** Sentinel Design
**Theme:** Design bounded local sentinels for high-value paths with explicit
runtime, fixture, threshold, and non-portable interpretation rules
**Time estimate:** 12 hours

### Tasks
1. Select candidate sentinel lanes from the Day 3 gap ranking.
2. Define fixture, command, metric, runtime budget, baseline, threshold,
   variance tolerance, and support tier for each candidate.
3. Separate hard local wall-check gates from threshold-free report lanes.
4. Define skip, unavailable-backend, supplemental, and stale-report behavior.
5. Decide which candidate lanes are ready for implementation and which remain
   design-only.
6. Write the sentinel design policy artifact.

### Deliverables
- candidate sentinel lane table
- metric and threshold policy
- reviewed versus supplemental sentinel split
- implementation-ready lane list
- design-only deferral list

### Completion Criteria
- every proposed sentinel has explicit metric and non-claim semantics
- threshold-free lanes are not treated as hard performance gates
- implementation candidates have validation commands and runtime budgets

---

## Day 7: Sentinel Implementation Plan

**Title:** Implementation Plan
**Theme:** Choose the sentinel/report implementation batch and define exact
edits, validation, rollback, and deferral criteria
**Time estimate:** 12 hours

### Tasks
1. Re-check Day 5 metadata design and Day 6 sentinel design against source
   and script touch points.
2. Select a scoped implementation batch for sentinel/report metadata or lane
   refinement.
3. Decide which changes require C edits, script edits, docs edits, generated
   report refresh, or no implementation.
4. Define validation commands for each touched surface, including full C
   quality gates if `.c` or `.h` files change.
5. Record rollback criteria and explicit deferral criteria.
6. Write the implementation plan artifact.

### Deliverables
- selected implementation batch
- touched-file forecast
- validation command plan
- rollback and deferral criteria
- Day 8 implementation checklist

### Completion Criteria
- implementation scope is small enough to validate in Sprint 132
- every potential code/script/docs touch point has a required check
- deferred lanes have blocker, dependency, and future owner

---

## Day 8: Sentinel Implementation Batch

**Title:** Sentinel Batch
**Theme:** Add or refine selected local sentinel/report lanes and generated
metadata without broadening performance claims
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 7 selected implementation batch.
2. Update benchmark/sentinel scripts, generated metadata, docs, or C code only
   where the implementation plan requires it.
3. Keep hard thresholds limited to lanes with accepted baseline and variance
   policy.
4. Preserve supplemental and report-only interpretation for threshold-free
   lanes.
5. Run the focused validation commands required by touched files.
6. Write the implementation batch artifact.

### Deliverables
- implemented sentinel/report changes or explicit implementation deferral
- generated metadata update notes
- focused validation output
- touched-file list
- unchanged-claims statement

### Completion Criteria
- implemented changes match the Day 7 plan
- no benchmark, backend, coverage, or public performance claim changes
  silently
- focused validation passes or the sprint stops with a blocker

---

## Day 9: Benchmark Documentation Cleanup

**Title:** Benchmark Docs
**Theme:** Improve benchmark interpretation and report-index handoff without
changing benchmark claims
**Time estimate:** 12 hours

### Tasks
1. Review `benchmarks/README.md`, `docs/maintainer_guide.md`, and related
   planning artifacts for sentinel/backend interpretation.
2. Apply documentation updates only if Day 4-8 accepted decisions require
   clearer local performance, backend, or report-index wording.
3. Otherwise publish a no-update rationale.
4. Cross-check wording against non-claims for portable performance,
   scalability, memory, backend parity, and benchmark portability.
5. Record docs validation and claim-boundary scan.
6. Write the benchmark documentation cleanup artifact.

### Deliverables
- benchmark docs update or no-update rationale
- report-index handoff wording notes
- non-claim scan results
- documentation validation log
- residual wording queue

### Completion Criteria
- benchmark docs match only earned Sprint 132 decisions
- no local sentinel becomes a portable performance claim
- every skipped docs update has a rationale and future owner

---

## Day 10: Report Index Handoff and Metadata Validation

**Title:** Metadata Check
**Theme:** Validate generated benchmark/sentinel metadata and handoff fields
against Sprint 131 report-index rules
**Time estimate:** 12 hours

### Tasks
1. Regenerate or inspect affected sentinel, benchmark, or guardrail report
   artifacts from the implementation batch.
2. Validate metadata fields for backend, compiler, platform, OpenMP,
   thread-count, fallback, freshness, and support tier where applicable.
3. Verify stale, missing, skipped, supplemental, and unavailable-backend
   behavior remains explicit.
4. Compare generated or curated rows against Sprint 131 report-index
   requirements.
5. Record gaps that block recurring index integration.
6. Write the report-index handoff and metadata validation artifact.

### Deliverables
- metadata validation results
- generated report/index inspection notes
- stale/missing/skip behavior notes
- Sprint 131 index-handoff alignment table
- residual metadata queue

### Completion Criteria
- affected generated metadata is reproducible or explicitly deferred
- report rows preserve support-tier and freshness boundaries
- every index-integration gap has blocker, dependency, and owner

---

## Day 11: Focused Benchmark and Runtime Validation

**Title:** Runtime Validation
**Theme:** Run focused benchmark, sentinel, backend, and report checks for all
touched surfaces
**Time estimate:** 12 hours

### Tasks
1. Run focused sentinel and benchmark commands required by implementation and
   documentation changes.
2. Run backend/runtime observability checks required by touched scripts or
   metadata.
3. Run supplemental report commands only when they were explicitly touched or
   promoted.
4. Run `make format && make lint && make test` if any `.c` or `.h` files
   changed.
5. Record pass/fail results, skipped checks, runtime notes, and blockers.
6. Write the focused runtime validation artifact.

### Deliverables
- focused validation command log
- benchmark/sentinel report results
- backend/runtime check results
- skipped-check rationale
- blocker or pass summary

### Completion Criteria
- all required checks pass before proceeding
- skipped checks are justified by support tier or untouched surfaces
- validation evidence is reproducible for closeout

---

## Day 12: Performance Non-Claim Register

**Title:** Non-Claims
**Theme:** Publish local performance, backend, runtime, and report non-claims
with owner and future-promotion criteria
**Time estimate:** 12 hours

### Tasks
1. Consolidate non-claims from Sprint 131, Day 4 backend contract, Day 6
   sentinel policy, implementation, docs cleanup, and validation artifacts.
2. Classify non-claims by performance portability, backend parity, runtime
   scalability, memory, thread-count, corpus breadth, and report freshness.
3. Define criteria for promoting any local sentinel/report row into stronger
   reviewed or recurring evidence.
4. Assign future owners to residual non-claim and promotion queues.
5. Decide whether maintainer-facing wording needs a final update.
6. Write the performance non-claim register artifact.

### Deliverables
- performance non-claim register
- backend/runtime non-claim table
- supplemental-to-reviewed promotion criteria
- future-owner queue
- maintainer wording update or no-update rationale

### Completion Criteria
- every performance/backend/runtime non-claim has owner and trigger
- promotion criteria are stricter than local measurement existence
- maintainer wording is updated only when evidence supports it

---

## Day 13: Final Validation Batch and Runtime Residual Queue

**Title:** Final Validation
**Theme:** Run final affected checks and publish the residual runtime,
sentinel, backend, and report queue
**Time estimate:** 12 hours

### Tasks
1. Run all affected docs, script, benchmark, sentinel, report-generation, and
   hygiene checks required by Sprint 132 changes.
2. Re-run full C quality gates if any C/header changes landed.
3. Reconcile all residual hot-path, backend, sentinel, benchmark-doc,
   report-index, and non-claim gaps.
4. Classify each residual by blocker, dependency, support tier, claim impact,
   validation status, and future owner.
5. Prepare Day 14 closeout inputs.
6. Write the final validation and runtime residual queue artifact.

### Deliverables
- final validation command log
- affected-check results
- residual runtime queue
- support-tier and claim-impact classification
- Day 14 closeout inputs

### Completion Criteria
- required checks have passed or the sprint stops with a blocker
- every residual gap has blocker, dependency, and future owner
- validation evidence is sufficient for closeout

---

## Day 14: Sprint Closeout and Backend Governance Handoff

**Title:** Runtime Closeout
**Theme:** Publish final performance sentinel/backend governance outcomes,
validation package, no-claim boundaries, and Sprint 133 handoff
**Time estimate:** 12 hours

### Tasks
1. Reconcile every Sprint 132 item against the project-plan checklist.
2. Review all accepted hot-path, backend, sentinel, metadata, documentation,
   validation, and non-claim outcomes for public or maintainer claim impact.
3. Publish final runtime governance ownership and residual assurance gaps.
4. Update maintainer-facing wording only if accepted evidence supports a
   bounded claim beyond current guidance.
5. Otherwise publish an explicit no-update rationale.
6. Write Sprint 132 closeout, retrospective inputs, and Sprint 133 handoff
   notes.

### Deliverables
- final Sprint 132 closeout artifact
- performance/backend ownership summary
- residual runtime assurance queue
- validation package
- maintainer wording update or no-update rationale
- Sprint 133 handoff notes

### Completion Criteria
- all Sprint 132 deliverables are present or explicitly deferred
- public and maintainer wording matches only earned evidence
- no unresolved performance, backend, sentinel, report, or runtime item lacks
  blocker, dependency, and future-owner notes
