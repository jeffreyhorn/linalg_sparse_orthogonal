# Sprint 200 Plan: Additional Allocation-Failure Owner Proof

**Sprint Duration:** 14 days
**Goal:** Prove one additional selected allocation-failure owner with
deterministic cleanup, stale-output suppression, and retry evidence.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 200 estimate in the Epic 18
project plan.

**Primary scope:** Select exactly one uncovered allocation-failure owner,
record its cleanup and publication invariants before implementation, extend
deterministic failure injection only where needed, add failed-allocation,
cleanup, stale-output, retry, and caller-owned-input regression tests, wire a
focused reliability gate, and calibrate reliability documentation against the
evidence.

**Non-goals:** Broad allocation-failure ownership across all modules, unrelated
solver rewrites, new public API behavior, package-manager work, benchmark
promotion, broad Windows freshness promotion, or state-of-the-art reliability
claims beyond the selected owner.

---

## Day 1: Owner Candidate Intake

**Title:** Candidate Intake
**Theme:** Establish the Sprint 200 reliability scope and collect eligible
allocation-failure owner candidates before choosing one.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 200 Epic 18 project-plan section and map items 200.1
   through 200.6 to expected artifacts.
2. Review Sprint 195 symbolic Cholesky allocation-failure proof artifacts for
   reusable evidence structure, harness constraints, and claim language.
3. Inventory currently covered allocation-failure gates and identify owners
   that remain uncovered.
4. Build a candidate ledger for symbolic LU, analyze lifecycle, direct-solver
   output publication, and any other locally visible candidate owners.
5. Create `WORKING_NOTES.md` with item checklist, candidate ledger,
   validation matrix, risk register, and open questions.

### Deliverables

- Sprint 200 working-notes scaffold.
- Candidate owner ledger.
- Existing allocation-failure gate inventory.
- Item-to-artifact traceability map.

### Completion Criteria

- Every Sprint 200 item has an initial owner artifact.
- Candidate owners are ranked against existing coverage and sprint feasibility.
- No implementation begins before the selected-owner boundary is explicit.

---

## Day 2: Selection Decision

**Title:** Owner Selection
**Theme:** Choose exactly one allocation-failure owner and document why that
owner is the sprint target.
**Time estimate:** 12 hours

### Tasks

1. Score candidates by review value, failure-path risk, harness reachability,
   stale-output exposure, retry semantics, and implementation size.
2. Exclude owners already covered by existing deterministic allocation-failure
   gates.
3. Select exactly one owner for Sprint 200 and freeze the owner boundary.
4. Record rejected candidates with deferral reasons and future follow-up
   notes.
5. Define the owner proof checklist for cleanup, stale-output suppression,
   retry behavior, caller-owned input preservation, and unsupported breadth.

### Deliverables

- Selected-owner decision record.
- Rejected-candidate deferral table.
- Owner proof checklist.
- Sprint scope boundary statement.

### Completion Criteria

- Item 200.1 has a documented owner selection.
- The selected owner is not already covered by an existing proof gate.
- Deferred candidates cannot be mistaken for Sprint 200 commitments.

---

## Day 3: Lifecycle Trace

**Title:** Lifecycle Trace
**Theme:** Trace the selected owner’s allocation, publication, cleanup, and
retry lifecycle before making code edits.
**Time estimate:** 12 hours

### Tasks

1. Read the selected owner implementation and identify all allocation sites,
   partially initialized outputs, publication points, cleanup paths, and retry
   entry points.
2. Identify caller-owned inputs and confirm which objects must remain
   unchanged on allocation failure.
3. Map stale-output risks for output pointers, metadata fields, cached state,
   and generated artifacts.
4. Record unsupported breadth, including failure modes intentionally left
   outside the selected proof.
5. Add lifecycle diagrams or tables to the working notes.

### Deliverables

- Selected-owner lifecycle trace.
- Allocation and publication point map.
- Caller-owned input preservation table.
- Unsupported-breadth record.

### Completion Criteria

- Item 200.2 has concrete implementation references.
- Cleanup and publication responsibilities are known before harness changes.
- Unsupported breadth is explicit and claim-safe.

---

## Day 4: Invariant Record

**Title:** Invariant Contract
**Theme:** Convert the lifecycle trace into pre-edit invariants for the proof
and tests.
**Time estimate:** 12 hours

### Tasks

1. Write cleanup invariants for every allocation-failure point in the selected
   owner boundary.
2. Write stale-output suppression invariants for all published outputs and
   status fields.
3. Write retry invariants that describe expected behavior after deterministic
   allocation failure is disabled.
4. Write caller-owned input invariants for matrices, vectors, options, handles,
   and buffers used by the selected owner.
5. Review invariants against Sprint 195 wording and keep the new proof
   selected-owner scoped.

### Deliverables

- Pre-edit invariant record.
- Cleanup, stale-output, retry, and caller-owned input checklist.
- Claim boundary wording for documentation and tests.

### Completion Criteria

- Item 200.2 is complete before code edits begin.
- Every planned test maps to a named invariant.
- The invariant record avoids broad reliability claims.

---

## Day 5: Harness Reachability Design

**Title:** Harness Design
**Theme:** Design the minimum deterministic failure-injection changes needed
to reach the selected owner.
**Time estimate:** 12 hours

### Tasks

1. Inspect existing deterministic allocation-failure harness APIs, counters,
   scopes, and registration guards.
2. Identify whether the selected owner can be reached with existing hooks or
   needs a narrowly scoped extension.
3. Define injection points and expected failure indices for the selected owner.
4. Plan cleanup for harness state so retries cannot inherit stale injection
   configuration.
5. Record harness risks and mitigation checks in the working notes.

### Deliverables

- Harness reachability design.
- Injection point and failure-index map.
- Harness cleanup checklist.
- Registration guard plan.

### Completion Criteria

- Item 200.3 has a minimal implementation design.
- Harness changes are limited to the selected owner’s proof needs.
- Retry contamination risks have named checks.

---

## Day 6: Harness Integration

**Title:** Harness Integration
**Theme:** Implement deterministic failure-injection reachability for the
selected owner.
**Time estimate:** 12 hours

### Tasks

1. Extend existing failure-injection plumbing only where the Day 5 design
   requires it.
2. Add selected-owner registration or discovery helpers if needed by the proof
   gate.
3. Ensure harness state is reset after success, failure, and skipped paths.
4. Add comments only where the selected-owner injection path would otherwise be
   hard to audit.
5. Build a focused local sanity check for harness compilation or script
   execution.

### Deliverables

- Minimal harness integration changes.
- Selected-owner reachability support.
- Harness reset behavior.
- Initial focused sanity-check result.

### Completion Criteria

- Item 200.3 implementation is locally reachable.
- No broad harness behavior changes are introduced.
- Harness state cannot intentionally leak across tests.

---

## Day 7: Failed-Allocation Tests

**Title:** Allocation Failure Tests
**Theme:** Add deterministic tests that prove allocation failures are surfaced
without publishing partial success.
**Time estimate:** 12 hours

### Tasks

1. Add failed-allocation tests for each selected-owner injection point.
2. Assert the expected error code or failure status for each injected failure.
3. Assert that public outputs remain unset or retain documented safe values.
4. Confirm partially initialized internal objects are not exposed as success.
5. Record failure-index coverage in the working notes.

### Deliverables

- Failed-allocation regression tests.
- Failure-index coverage table.
- Stale-output assertions for failed calls.
- Updated working-notes evidence.

### Completion Criteria

- Item 200.4 has direct failed-allocation coverage.
- Every reachable injection point has an expected result.
- Failed calls cannot be mistaken for successful publication.

---

## Day 8: Cleanup Proof Tests

**Title:** Cleanup Proof
**Theme:** Prove deterministic cleanup for the selected owner’s failure paths.
**Time estimate:** 12 hours

### Tasks

1. Add cleanup assertions for all selected-owner failure paths that allocate
   temporary or partially owned resources.
2. Use existing allocation accounting or leak-detection helpers when available.
3. Add focused teardown checks for harness state and selected-owner outputs.
4. Verify cleanup behavior for early failures and late failures.
5. Document any cleanup behavior that is indirectly proven by existing helpers.

### Deliverables

- Cleanup regression tests.
- Early- and late-failure cleanup evidence.
- Harness teardown checks.
- Cleanup proof notes.

### Completion Criteria

- Item 200.4 covers cleanup invariants.
- Cleanup assertions map back to the Day 4 invariant record.
- No selected-owner allocation failure leaves retained test-visible state.

---

## Day 9: Retry and Input Preservation Tests

**Title:** Retry Proof
**Theme:** Prove retry behavior and caller-owned input preservation after
injected allocation failure.
**Time estimate:** 12 hours

### Tasks

1. Add retry tests that disable injection after failure and rerun the selected
   owner successfully.
2. Assert caller-owned inputs retain expected structure, dimensions, indices,
   and values across failed calls.
3. Assert retry output is fresh and not contaminated by stale failed-call
   state.
4. Add repeated failure-then-success cases if the owner has multiple
   publication stages.
5. Record retry evidence and caller-owned input snapshots in the working notes.

### Deliverables

- Retry regression tests.
- Caller-owned input preservation assertions.
- Fresh-output retry evidence.
- Updated proof checklist.

### Completion Criteria

- Item 200.4 covers retry and caller-owned input invariants.
- A failed allocation does not prevent later successful use.
- Retry success does not depend on stale output from the failed attempt.

---

## Day 10: Focused Gate Wiring

**Title:** Reliability Gate
**Theme:** Register a focused gate for the selected owner and make it easy to
run independently.
**Time estimate:** 12 hours

### Tasks

1. Add Make or CTest wiring for the selected reliability proof.
2. Add labels or target names that identify the selected owner and avoid broad
   reliability claims.
3. Add registration guards so the proof cannot silently drop out of the
   focused gate.
4. Update source-list or test-list expectations if new test files are added.
5. Run the focused gate and record the exact command and result.

### Deliverables

- Focused reliability gate.
- Registration guard.
- Source/test-list alignment.
- Focused gate evidence.

### Completion Criteria

- Item 200.5 is implemented.
- The selected-owner proof can be run without the full suite.
- Missing proof registration fails clearly.

---

## Day 11: Claim Documentation

**Title:** Claim Calibration
**Theme:** Update documentation so reliability claims match the selected-owner
proof and retained non-goals.
**Time estimate:** 12 hours

### Tasks

1. Update maintainer and user-facing reliability documentation with the
   selected-owner proof boundary.
2. Record that Sprint 200 proves one additional owner, not broad
   allocation-failure ownership across the library.
3. Link focused gate commands, artifacts, and invariant records from the
   documentation.
4. Update planning artifacts with the item status and residual risks.
5. Ensure unsupported breadth and deferred owner candidates remain visible.

### Deliverables

- Claim-safe documentation updates.
- Selected-owner proof references.
- Updated working-notes item status.
- Residual-risk and deferred-candidate records.

### Completion Criteria

- Item 200.6 documentation work is complete.
- Documentation does not overstate broad reliability coverage.
- Proof claims are linked to runnable evidence.

---

## Day 12: Integrated Local Validation

**Title:** Validation Pass
**Theme:** Run focused and required local validation before closeout.
**Time estimate:** 12 hours

### Tasks

1. Run the focused selected-owner reliability gate.
2. Run source-list and docs checks that apply to the changed files.
3. Run `make format`.
4. Run `make lint`.
5. Run `make test` and record results, failures, or environmental residuals.

### Deliverables

- Integrated validation transcript summary.
- Focused gate result.
- Format, lint, and test result record.
- Updated risk register.

### Completion Criteria

- Item 200.6 validation evidence is recorded.
- Required quality checks have run or have explicit environment residuals.
- Any failures are fixed before closeout proceeds.

---

## Day 13: Review-Surface Hardening

**Title:** Review Hardening
**Theme:** Audit the selected-owner proof for review clarity, narrowness, and
evidence consistency.
**Time estimate:** 12 hours

### Tasks

1. Review all code, tests, docs, and planning diffs for unnecessary breadth.
2. Confirm each new test assertion maps to a Day 4 invariant.
3. Confirm the focused gate names, labels, and documentation use selected-owner
   vocabulary.
4. Re-run any focused checks affected by review hardening changes.
5. Prepare closeout notes with completed items and residual non-goals.

### Deliverables

- Review-hardening notes.
- Final invariant-to-test traceability table.
- Cleaned-up focused proof surface.
- Closeout checklist draft.

### Completion Criteria

- The review surface is limited to the selected owner and required supporting
  gates.
- Evidence, docs, and tests use consistent claim vocabulary.
- Remaining residuals are explicit and not hidden in completion claims.

---

## Day 14: Closeout and Retrospective Inputs

**Title:** Closeout Evidence
**Theme:** Finalize Sprint 200 evidence, item status, and retrospective inputs.
**Time estimate:** 12 hours

### Tasks

1. Reconcile items 200.1 through 200.6 against deliverables and validation
   evidence.
2. Update `WORKING_NOTES.md` with final item status, command results, known
   residuals, and follow-up candidates.
3. Create a Day 14 closeout artifact summarizing selected-owner proof
   completion and claim boundaries.
4. Prepare retrospective inputs: completed work, validation, deviations,
   deferred breadth, and recommendations.
5. Confirm the working tree contains only Sprint 200 intended changes.

### Deliverables

- Final Sprint 200 working-notes status.
- Day 14 closeout artifact.
- Retrospective input summary.
- Final validation and residual-risk ledger.

### Completion Criteria

- Sprint 200 has enough evidence for a later retrospective.
- All six Sprint 200 items have clear completion or residual status.
- The selected allocation-failure owner proof is documented without broad
  unsupported claims.
