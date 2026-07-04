# Sprint 105 Plan: Reordering, Graph & Large-Matrix Scalability

**Sprint Duration:** 14 days
**Goal:** Improve reorder, graph, and large-matrix evidence with clearer fill,
ordering, runtime, and memory interpretation. This sprint implements the
Sprint 105 section of `docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 105 begins from:
- the Sprint 100 evidence templates, claim boundaries, and validation contract
- the Sprint 101 compressed-first storage and API front-door baseline
- the Sprint 102 direct solver oracle and robustness work
- the Sprint 103 iterative, eigensolver, and SVD comparison surfaces
- the Sprint 104 runtime, backend, OpenMP, and performance-sentinel contract
- existing AMD, COLAMD, nested-dissection, quotient-graph, graph partition,
  fill-report, and large-matrix proof surfaces that need stronger evidence and
  clearer ownership

The strongest Sprint 105 pressure is to improve scalability evidence without
turning local runtime or fill measurements into broad performance claims. The
sprint must:
- re-rank reorder, graph, fill, and partition gaps from the live tree
- define canonical fill, runtime, memory, and fixture fields for reorder
  artifacts
- expand named-matrix and generated-family evidence in bounded ways
- add deterministic guardrails for large-matrix memory and runtime behavior
- clean graph/reorder proof owners where touched history has accumulated
- align reports, docs, and maintainer guidance with the evidence model
- close with validation, generated artifacts, and a Sprint 106 handoff queue

**End State:** Sprint 105 leaves behind:
- a reorder/graph audit and prioritized gap queue
- a documented fill, runtime, memory, and fixture naming contract
- expanded named-matrix and generated graph-family evidence
- deterministic large-matrix scalability guardrails where appropriate
- cleaner graph/reorder ownership and helper structure
- reporting and documentation aligned with maintained evidence
- validation artifacts and Sprint 106 handoff criteria

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 105 project-plan estimate.

---

## Day 1: Sprint 105 Scope & Scalability Baseline

**Title:** Scalability Baseline
**Theme:** Convert the Sprint 105 project-plan section and prior Epic 10
handoffs into one bounded reorder/graph scalability package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 105 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 100 evidence templates and Sprint 104 runtime/benchmark
   closeout notes for measurement and non-claim constraints.
3. Inventory the Sprint 105 workstreams:
   - reorder and graph audit
   - fill metric contract
   - named-matrix comparison expansion
   - generated graph-family expansion
   - large-matrix guardrails
   - graph/reorder ownership cleanup
   - reporting, docs, validation, and closeout
4. Create Sprint 105 working notes and artifacts directory.
5. Record validation expectations for docs-only, script-touch, benchmark-touch,
   test-touch, and source-touch days.

### Deliverables
- Sprint 105 workstream inventory
- working-notes baseline
- initial artifacts directory structure
- validation expectation list

### Completion Criteria
- every Sprint 105 project-plan item has day-level ownership
- Sprint 100 evidence rules and Sprint 104 runtime limits are visible in
  working notes
- validation expectations are explicit before audit work begins

---

## Day 2: Reorder and Graph Surface Audit

**Title:** Surface Audit
**Theme:** Re-rank AMD, COLAMD, nested dissection, quotient graph, graph
partition, and fill-report gaps from the live repository
**Time estimate:** 12 hours

### Tasks
1. Inventory reorder and graph source, test, benchmark, documentation, and
   artifact owners.
2. Classify current evidence by algorithm family:
   - AMD and quotient-graph AMD
   - COLAMD
   - nested dissection
   - graph partition and separator paths
   - fill and runtime report surfaces
3. Identify missing fixtures, ambiguous metrics, stale comments, and unclear
   proof ownership.
4. Rank gaps by user-visible value, determinism, validation cost, and claim
   risk.
5. Write the reorder/graph surface audit artifact.

### Deliverables
- reorder and graph owner inventory
- ranked gap list
- current proof and benchmark surface map
- initial fix-now vs defer queue

### Completion Criteria
- AMD, COLAMD, nested dissection, quotient graph, and graph partition surfaces
  are represented
- gaps are ranked from live evidence, not stale assumptions
- deferred items include explicit reasons

---

## Day 3: Fill and Fixture Contract Design

**Title:** Fill Contract
**Theme:** Define canonical fill, runtime, memory, and fixture naming fields
for reorder and graph artifacts
**Time estimate:** 12 hours

### Tasks
1. Review current benchmark CSV fields, wall-check output, fill reporting, and
   generated artifacts.
2. Define canonical fields for:
   - matrix or graph fixture identity
   - ordering algorithm and mode
   - symbolic fill counts and ratios
   - reorder runtime
   - memory proxy or allocation guardrail
   - skipped or unavailable lanes
3. Define fixture naming rules for named matrices, generated grids, generated
   graphs, and synthetic stress cases.
4. Identify where the contract should appear in docs, scripts, benchmarks, or
   maintainer guidance.
5. Write the fill and fixture contract artifact.

### Deliverables
- fill metric contract
- fixture naming contract
- skip and unavailable-lane rules
- implementation checklist for reporting surfaces

### Completion Criteria
- fill, runtime, memory, and fixture fields have clear semantics
- naming rules support aggregation across pass, report, and skip rows
- no metric is framed as a portable performance claim

---

## Day 4: Evidence Boundary and Matrix Selection

**Title:** Evidence Boundary
**Theme:** Select the named matrices and generated graph families that Sprint
105 will use for bounded scalability evidence
**Time estimate:** 12 hours

### Tasks
1. Compare the Day 2 audit against the Day 3 metric contract.
2. Select named matrices already present or practical to maintain in the repo.
3. Select generated families for grids, separators, quotient graphs, and large
   sparse patterns.
4. Define size tiers for smoke, reviewed, supplemental, and local-only lanes.
5. Write the evidence boundary artifact with exact commands and expected
   outputs.

### Deliverables
- selected named-matrix evidence list
- selected generated-family evidence list
- size-tier and ownership table
- command and artifact plan

### Completion Criteria
- Sprint 105 implementation lanes are frozen before source or script edits
- reviewed lanes are deterministic and affordable
- large or noisy lanes are explicitly supplemental or local-only

---

## Day 5: Reorder/Fill Reporting Batch 1

**Title:** Reporting Batch 1
**Theme:** Implement the first reporting updates for canonical fill and fixture
fields
**Time estimate:** 12 hours

### Tasks
1. Update the selected reorder or fill reporting path with Day 3 field names.
2. Preserve existing consumers or document any deliberate output change.
3. Add focused parsing or smoke coverage for the updated reporting format.
4. Regenerate the bounded sample artifact selected on Day 4.
5. Record validation output in working notes.

### Deliverables
- updated reorder/fill reporting path
- focused reporting tests or smoke proof
- regenerated sample artifact
- Day 5 validation notes

### Completion Criteria
- output fields match the fill and fixture contract
- existing maintained consumers remain coherent
- focused validation passes before expanding evidence

---

## Day 6: Named-Matrix Evidence Expansion

**Title:** Named Matrices
**Theme:** Add or refresh reorder and fill comparisons on selected named
matrices
**Time estimate:** 12 hours

### Tasks
1. Run or update comparison lanes for the selected named matrices.
2. Capture AMD, COLAMD, nested-dissection, quotient-graph, and baseline
   behavior where applicable.
3. Record fill counts, fill ratios, runtime context, and skipped-lane reasons.
4. Add focused checks or fixtures for deterministic named-matrix behavior.
5. Write the named-matrix evidence artifact.

### Deliverables
- named-matrix reorder/fill evidence
- deterministic fixture or test updates where needed
- skip and limitation notes
- focused validation output

### Completion Criteria
- named-matrix evidence uses the canonical metric contract
- skipped or unavailable lanes are explicit
- local timing context remains bounded and non-portable

---

## Day 7: Generated Graph Family Expansion

**Title:** Graph Families
**Theme:** Add or refresh generated-family evidence for graph partition,
separator, quotient-graph, and ordering behavior
**Time estimate:** 12 hours

### Tasks
1. Implement or refresh generated graph-family inputs selected on Day 4.
2. Cover at least two structural families, such as grids, banded graphs,
   block-like graphs, or separator-heavy patterns.
3. Compare ordering and graph behavior with canonical fixture names.
4. Add deterministic smoke tests or artifacts for generated-family coverage.
5. Write the generated graph-family evidence artifact.

### Deliverables
- generated graph-family evidence
- deterministic generated inputs or helper usage
- focused test or benchmark proof
- Day 7 validation notes

### Completion Criteria
- generated families are reproducible without external downloads
- evidence exposes structural behavior, not broad superiority claims
- artifacts are small enough for maintained review or clearly local-only

---

## Day 8: Large-Matrix Guardrail Design

**Title:** Guardrail Design
**Theme:** Define deterministic large-matrix memory and runtime guardrails that
fit reviewed or supplemental validation lanes
**Time estimate:** 12 hours

### Tasks
1. Identify large-matrix risks from Days 2, 6, and 7:
   - memory growth
   - integer overflow
   - recursion depth or stack pressure
   - pathological fill
   - runtime cliffs
2. Select guardrails suitable for tests, benchmarks, scripts, or docs.
3. Define thresholds as structural bounds or smoke limits rather than portable
   timing claims.
4. Define skip behavior for local-only large lanes.
5. Write the large-matrix guardrail design artifact.

### Deliverables
- large-matrix risk list
- guardrail design artifact
- reviewed vs supplemental lane classification
- validation plan

### Completion Criteria
- guardrails target deterministic failure modes
- local-only lanes are separated from reviewed checks
- thresholds are justified and maintainable

---

## Day 9: Scalability Guardrail Implementation

**Title:** Guardrail Batch
**Theme:** Add deterministic large-matrix guardrails where suitable for
reviewed or supplemental checks
**Time estimate:** 12 hours

### Tasks
1. Implement the selected guardrail tests, scripts, or benchmark checks.
2. Keep reviewed lanes bounded enough for normal development validation.
3. Add supplemental or local-only commands for larger evidence where needed.
4. Update working notes with exact commands and expected pass/fail behavior.
5. Run focused validation for touched code, scripts, or tests.

### Deliverables
- implemented scalability guardrails
- reviewed and supplemental command list
- focused validation output
- residual large-matrix queue

### Completion Criteria
- guardrails detect meaningful large-matrix regressions deterministically
- supplemental lanes do not become hidden CI requirements
- focused validation passes

---

## Day 10: Graph/Reorder Ownership Cleanup

**Title:** Ownership Cleanup
**Theme:** Remove touched history-heavy comments and extract helpers from graph
and reorder proof owners where useful
**Time estimate:** 12 hours

### Tasks
1. Identify touched graph/reorder files with stale sprint-history comments,
   duplicated helpers, or unclear proof ownership.
2. Extract helpers only where they reduce real duplication or clarify the
   maintained proof surface.
3. Remove or rewrite comments that describe history instead of current
   behavior.
4. Preserve public behavior and existing validation expectations.
5. Run focused source and test validation for touched owners.

### Deliverables
- graph/reorder ownership cleanup
- extracted helper or comment cleanup batch
- focused validation notes
- updated residual cleanup queue

### Completion Criteria
- touched comments describe current behavior, not implementation history
- helper extraction reduces actual maintenance cost
- behavior-preserving validation passes

---

## Day 11: Reporting and Documentation Alignment

**Title:** Docs Alignment
**Theme:** Consolidate reorder, fill, graph, and scalability reporting guidance
for users and maintainers
**Time estimate:** 12 hours

### Tasks
1. Update user-facing documentation for reorder/fill metrics and claim
   boundaries.
2. Update maintainer guidance for generated artifacts, reviewed lanes, and
   supplemental large-matrix lanes.
3. Align benchmark documentation with the Day 3 contract and Day 8 guardrail
   design.
4. Ensure examples or command lists point at current targets and outputs.
5. Run documentation formatting or focused doc checks where available.

### Deliverables
- updated reorder/fill documentation
- maintainer guidance updates
- benchmark/reporting interpretation notes
- docs validation notes

### Completion Criteria
- users can interpret fill and runtime outputs without overreading them
- maintainers know which lanes are reviewed, supplemental, or local-only
- documentation matches the implemented artifacts

---

## Day 12: Integrated Evidence Reconciliation

**Title:** Evidence Reconciliation
**Theme:** Reconcile named-matrix, generated-family, guardrail, and reporting
artifacts into one coherent Sprint 105 evidence package
**Time estimate:** 12 hours

### Tasks
1. Re-run the selected named-matrix and generated-family evidence commands.
2. Re-run implemented scalability guardrails.
3. Compare artifacts against the metric and fixture contract.
4. Identify any remaining contradictions in docs, reports, or tests.
5. Write the evidence reconciliation artifact with fix candidates.

### Deliverables
- integrated Sprint 105 evidence table
- regenerated artifact checklist
- contradiction and fix-candidate list
- validation output summary

### Completion Criteria
- evidence artifacts use consistent metric and fixture fields
- contradictions are fixed immediately or assigned to Day 13/14 residuals
- no local runtime result is framed as portable performance evidence

---

## Day 13: Final Fix Batch and Validation Sweep

**Title:** Final Validation
**Theme:** Land the final bounded fix batch and run the required validation
surface for Sprint 105
**Time estimate:** 12 hours

### Tasks
1. Resolve the highest-priority contradictions from Day 12.
2. Re-run focused validation for every touched source, test, script, and docs
   surface.
3. Run the broader required validation gate if `.c` or `.h` files were
   modified.
4. Regenerate final artifacts that changed because of the fix batch.
5. Record the final validation sweep in working notes.

### Deliverables
- final bounded fix batch
- final validation command output summary
- regenerated final artifacts
- remaining residual queue

### Completion Criteria
- all required checks pass before closeout begins
- final artifacts match the implemented behavior
- residual items are explicit and not hidden in prose

---

## Day 14: Sprint 105 Closeout and Handoff

**Title:** Closeout Handoff
**Theme:** Close Sprint 105 with validated artifacts, documentation, and a
clear Sprint 106 handoff queue
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 105 artifacts, working notes, docs, tests, scripts, and
   source changes.
2. Ensure every Sprint 105 project-plan item has a completed, deferred, or
   non-claim status.
3. Write the closeout and handoff artifact.
4. Prepare the Sprint 105 retrospective input list.
5. Re-run final focused checks required by any closeout edits.

### Deliverables
- Sprint 105 closeout artifact
- Sprint 106 handoff queue
- retrospective input list
- final validation notes

### Completion Criteria
- Sprint 105 closes from validated evidence rather than aspirational claims
- every project-plan item has an explicit closeout status
- Sprint 106 has a clear queue of remaining scalability, reporting, or
  ownership work
