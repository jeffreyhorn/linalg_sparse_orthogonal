# Sprint 98 Plan: Assurance, External Comparison & Coverage Architecture Phase 3

**Sprint Duration:** 14 days
**Goal:** Broaden the maintained external comparison and proof architecture so
the repo can support stronger competitive claims than the bounded Epic 8 lane.
This sprint implements the Sprint 98 section of
`docs/planning/EPIC_9/PROJECT_PLAN.md`.

**Starting Point:** Sprint 98 begins from:
- the Sprint 90 comparison contract
- the Sprint 94 capability-surface modernization baseline
- the Sprint 96 maintainability extraction and proof-owner cleanup work
- the Sprint 97 build/package convergence and source-list proof architecture
- a repo with bounded external comparison evidence, but not yet enough
  maintained correctness, runtime, fill, and coverage topology evidence to
  support broader competitive calibration

The strongest Sprint 98 pressure is not to make broad performance claims. It
is to widen the maintained assurance architecture by:
- re-ranking the next external correctness and performance comparison lanes
- defining a bounded architecture for differential proof and runtime/fill
  comparison
- landing one high-value external correctness expansion beyond the current SPD
  path
- adding bounded runtime/fill comparison artifacts for meaningful workloads
- reducing fragmentation in proof-owner and comparison-owner topology
- aligning workflows, docs, and maintainer ownership with the widened model
- closing from full validation and a clear residual comparison queue

**End State:** Sprint 98 leaves behind:
- a refreshed comparison-surface ranking
- a bounded proof/comparison architecture for future assurance expansion
- one maintained external correctness expansion batch
- bounded runtime/fill comparison evidence on selected workloads
- cleaner coverage and proof-owner topology for widened comparison work
- workflow, docs, and maintainer guidance aligned with the assurance model
- a validated Sprint 98 closeout package and Sprint 99 handoff queue

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 98 project-plan estimate.

---

## Day 1: Sprint 98 Scope & Assurance Baseline

**Title:** Assurance Baseline
**Theme:** Turn the Sprint 98 project-plan section and prior comparison work
into one bounded assurance-expansion package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 98 section of
   `docs/planning/EPIC_9/PROJECT_PLAN.md`.
2. Re-read the Sprint 90 comparison contract, Sprint 94 capability artifacts,
   Sprint 96 maintainability closeout, and Sprint 97 build/package closeout.
3. Inventory current assurance and comparison surfaces:
   - maintained external comparison tests
   - benchmark and runtime reporting surfaces
   - fill/compression or structural comparison artifacts
   - coverage and proof-owner topology
   - CI workflow lanes and local reviewed validation gates
4. Separate correctness evidence, runtime/fill evidence, coverage topology,
   documentation claims, and workflow ownership.
5. Open Sprint 98 working notes and record validation expectations for docs,
   scripts, benchmark, test, and code-touch days.

### Deliverables
- Sprint 98 scope inventory
- current assurance/comparison surface map
- starting validation and working-notes baseline

### Completion Criteria
- Sprint 98 starts from the merged Sprint 97 end state
- current external comparison and proof-owner surfaces are visible before
  reranking
- validation requirements are explicit before comparison architecture changes

---

## Day 2: Comparison-Surface Rerank

**Title:** Comparison Rerank
**Theme:** Rank the highest-value next correctness, runtime, and fill
comparison lanes
**Time estimate:** 12 hours

### Tasks
1. Inspect existing comparison fixtures, benchmark inputs, and external
   reference paths.
2. Rank candidate external correctness lanes by:
   - user-visible algorithm value
   - availability of a trusted external reference
   - maintenance cost
   - deterministic local reproducibility
   - CI suitability
3. Rank candidate runtime/fill comparison lanes by workload relevance,
   reporting clarity, and risk of misleading claims.
4. Identify which candidate lanes belong in Sprint 98 and which should remain
   in the residual queue.
5. Write the Day 2 comparison-surface rerank artifact.

### Deliverables
- ranked correctness comparison candidate list
- ranked runtime/fill comparison candidate list
- fix-now vs residual assurance queue

### Completion Criteria
- Sprint 98 has one authoritative comparison ranking
- selected lanes are tied to maintainable proof value, not broad marketing
  claims
- lower-value or unstable lanes are explicitly deferred

---

## Day 3: Proof/Comparison Architecture Design

**Title:** Architecture Design
**Theme:** Define the bounded architecture for differential proof,
runtime/fill comparison, and proof-owner topology cleanup
**Time estimate:** 12 hours

### Tasks
1. Use the Day 2 ranking to select the first correctness and runtime/fill
   implementation lanes.
2. Define how external references, fixture ownership, benchmark ownership, and
   CI ownership should be separated.
3. Decide where comparison artifacts should live:
   - test fixtures
   - benchmark inputs
   - generated reports
   - docs or maintainer-only artifacts
   - workflow assertions
4. Define claim boundaries for correctness, runtime, fill, and coverage
   language.
5. Write the proof/comparison architecture artifact and validation plan.

### Deliverables
- proof/comparison architecture artifact
- selected correctness expansion boundary
- selected runtime/fill comparison boundary
- validation plan for implementation days

### Completion Criteria
- no comparison lane is widened before ownership boundaries are written
- selected architecture preserves bounded, reproducible proof
- competitive language is constrained by maintained evidence

---

## Day 4: External Correctness Boundary Freeze

**Title:** Correctness Boundary
**Theme:** Freeze the highest-value maintained external correctness expansion
before implementation
**Time estimate:** 12 hours

### Tasks
1. Re-read the selected correctness lane against the Day 3 architecture.
2. Identify exact algorithms, fixtures, reference outputs, and tests to touch.
3. Define acceptable tolerances, deterministic inputs, and skip behavior for
   unavailable optional external dependencies.
4. Identify local and CI-equivalent validation commands for the lane.
5. Write the Day 4 correctness boundary artifact with landing sequence and
   rollback notes.

### Deliverables
- external correctness boundary artifact
- fixture/reference ownership checklist
- validation and rollback checklist

### Completion Criteria
- the correctness expansion can be implemented without widening scope
- optional dependency behavior is explicit before edits begin
- validation commands are specific enough for a proof-lane change

---

## Day 5: External Correctness Expansion Batch 1

**Title:** Correctness Batch 1
**Theme:** Land the first half of the selected maintained external correctness
comparison lane
**Time estimate:** 12 hours

### Tasks
1. Implement the initial fixture, reference, or harness changes from Day 4.
2. Keep external dependency handling bounded and deterministic.
3. Add or update tests only where they strengthen maintained comparison
   evidence.
4. Run focused correctness and build checks during development.
5. Record implementation notes and any discovered residual comparison risks.

### Deliverables
- initial external correctness comparison batch
- updated fixtures, harnesses, or tests
- implementation notes and residual risk list

### Completion Criteria
- the selected lane has a working maintained comparison path
- focused checks pass for touched correctness surfaces
- no unsupported competitive claim is introduced

---

## Day 6: External Correctness Expansion Batch 2

**Title:** Correctness Batch 2
**Theme:** Complete the correctness lane and reconcile proof ownership
**Time estimate:** 12 hours

### Tasks
1. Finish remaining correctness-lane implementation from Day 5.
2. Tighten tolerance, fixture naming, and failure messages for maintainability.
3. Update adjacent docs or maintainer notes only where the comparison contract
   changed.
4. Re-run the targeted proof commands from Day 4.
5. Write the Day 6 correctness expansion closeout artifact.

### Deliverables
- completed external correctness expansion
- updated proof-owner notes
- correctness expansion closeout artifact

### Completion Criteria
- the expanded correctness lane is maintainable and reproducible
- proof ownership is clear across tests, fixtures, and docs
- targeted correctness validation passes

---

## Day 7: Runtime/Fill Comparison Boundary Freeze

**Title:** Runtime Boundary
**Theme:** Freeze the bounded runtime/fill comparison workload and reporting
contract
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 2 runtime/fill ranking and Day 3 architecture.
2. Select the workload, metrics, and artifact shape for Sprint 98 runtime/fill
   comparison.
3. Define what the runtime/fill comparison can and cannot claim.
4. Identify benchmark scripts, generated artifacts, docs, and workflow checks
   that may be touched.
5. Write the Day 7 runtime/fill boundary artifact with validation commands.

### Deliverables
- runtime/fill comparison boundary artifact
- selected workload and metric contract
- claim-boundary and validation checklist

### Completion Criteria
- runtime/fill implementation scope is fixed before edits begin
- reporting avoids broad or unstable performance claims
- validation and artifact ownership are explicit

---

## Day 8: Runtime/Fill Comparison Batch 1

**Title:** Runtime Batch 1
**Theme:** Add the initial bounded runtime/fill comparison artifact on the
selected workload
**Time estimate:** 12 hours

### Tasks
1. Implement the initial runtime/fill comparison script, fixture, or report
   path from Day 7.
2. Preserve deterministic inputs and stable metric definitions.
3. Ensure output is useful for maintainers without implying universal
   performance superiority.
4. Run focused benchmark or artifact-generation checks.
5. Record observed workload behavior and reporting limitations.

### Deliverables
- initial runtime/fill comparison artifact
- stable metric/reporting path
- observed limitation notes

### Completion Criteria
- selected workload produces a bounded comparison artifact
- metric definitions are repeatable and documented
- focused runtime/fill checks pass

---

## Day 9: Runtime/Fill Comparison Batch 2

**Title:** Runtime Batch 2
**Theme:** Complete runtime/fill evidence and align it with maintained
assurance language
**Time estimate:** 12 hours

### Tasks
1. Finish remaining runtime/fill comparison work from Day 8.
2. Add guardrails, labels, or maintainer notes needed to prevent claim drift.
3. Reconcile generated artifacts with docs, benchmark references, and workflow
   ownership.
4. Re-run targeted runtime/fill validation commands.
5. Write the Day 9 runtime/fill comparison closeout artifact.

### Deliverables
- completed runtime/fill comparison batch
- reporting guardrails or maintainer notes
- runtime/fill closeout artifact

### Completion Criteria
- runtime/fill evidence is useful but bounded
- docs and artifacts use consistent comparison language
- targeted runtime/fill validation passes

---

## Day 10: Coverage-Topology Audit

**Title:** Coverage Audit
**Theme:** Identify the highest-value proof-owner and comparison-owner topology
cleanup
**Time estimate:** 12 hours

### Tasks
1. Inventory proof-owner files touched or referenced by Sprint 98 work.
2. Identify fragmentation in:
   - correctness comparison owners
   - benchmark and runtime/fill owners
   - coverage-related docs or scripts
   - workflow labels and validation targets
3. Separate naming cleanup from structural cleanup and deferred refactoring.
4. Select the highest-value topology cleanup that supports widened assurance.
5. Write the Day 10 coverage-topology audit artifact.

### Deliverables
- coverage/proof-owner topology audit
- selected cleanup target
- deferred topology cleanup queue

### Completion Criteria
- cleanup is tied to real Sprint 98 maintainability pressure
- proof ownership is not weakened by consolidation
- deferred cleanup is explicit and bounded

---

## Day 11: Coverage-Topology Cleanup Batch

**Title:** Topology Cleanup
**Theme:** Reduce fragmentation or rename the selected proof owners so the
widened comparison story is easier to maintain
**Time estimate:** 12 hours

### Tasks
1. Implement the selected Day 10 topology cleanup.
2. Keep changes limited to naming, ownership, documentation, or small
   structure improvements justified by the audit.
3. Update references in workflows, docs, tests, or scripts only where ownership
   changed.
4. Run focused validation for all renamed or moved proof surfaces.
5. Write the Day 11 topology cleanup artifact.

### Deliverables
- coverage/proof-owner topology cleanup
- updated references for renamed or clarified owners
- topology cleanup artifact

### Completion Criteria
- selected proof-owner fragmentation is reduced
- all touched references are coherent
- focused validation passes for changed proof surfaces

---

## Day 12: CI & Support-Surface Alignment

**Title:** CI Alignment
**Theme:** Reconcile workflows, docs, and maintainer ownership with the widened
assurance model
**Time estimate:** 12 hours

### Tasks
1. Review CI workflows and local reviewed gates against the new correctness,
   runtime/fill, and topology changes.
2. Update workflow labels, comments, docs, or maintainer guidance where the
   assurance model changed.
3. Preserve staged exclusions and platform claim fences where evidence remains
   bounded.
4. Reconcile public docs with maintainer-only proof details.
5. Write the Day 12 CI/support alignment artifact.

### Deliverables
- CI/support-surface alignment changes
- updated maintainer or workflow guidance
- platform and claim-fence reconciliation notes

### Completion Criteria
- workflows and docs describe the widened assurance model consistently
- public claims remain bounded by maintained evidence
- local and CI validation ownership is clear

---

## Day 13: Validation Sweep & Residual Queue

**Title:** Validation Sweep
**Theme:** Run the strongest practical validation set and convert remaining
assurance work into a bounded queue
**Time estimate:** 12 hours

### Tasks
1. Re-run focused checks for every Sprint 98 touched surface.
2. Run broader validation required by code, header, build, workflow, or docs
   changes.
3. Inspect generated comparison artifacts and docs for stale or overstated
   claims.
4. Build the residual queue for external comparison, runtime/fill evidence,
   coverage topology, and CI/support alignment.
5. Write the Day 13 validation and residual-queue artifact.

### Deliverables
- Sprint 98 validation results
- residual assurance/comparison queue
- stale-claim and artifact review notes

### Completion Criteria
- validation status is explicit before closeout
- residual work is ranked and bounded
- no known overstated claim remains in touched surfaces

---

## Day 14: Sprint 98 Closeout & Handoff

**Title:** Closeout
**Theme:** Close Sprint 98 with artifacts, validation evidence, and a Sprint 99
handoff queue
**Time estimate:** 12 hours

### Tasks
1. Reconcile Sprint 98 artifacts against all seven project-plan items.
2. Finalize working notes with landed changes, validation results, and
   deferred work.
3. Prepare the Sprint 98 closeout artifact and Sprint 99 handoff queue.
4. Run final docs hygiene and any required reruns from Day 13.
5. Confirm the sprint plan, artifacts, and repository state are coherent for
   review.

### Deliverables
- Sprint 98 closeout package
- final validation summary
- Sprint 99 handoff queue
- updated working notes

### Completion Criteria
- all Sprint 98 project-plan items have explicit closeout status
- validation results and residual risks are documented
- Sprint 99 can start from a clear assurance/comparison handoff
