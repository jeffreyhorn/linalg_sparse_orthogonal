# Sprint 195 Plan: Selected Reliability and Failure-Path Proof

**Sprint Duration:** 14 days
**Goal:** Add deterministic reliability evidence for one selected
allocation-heavy or failure-prone owner beyond prior proof lanes.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 195 estimate in the Epic 17
project plan.

**Primary scope:** Select exactly one reliability owner using current
allocation-density, cleanup-complexity, user-impact, and testability evidence;
document its failure-path invariants; extend deterministic failure injection or
owner-local fail-at-count control; add regression coverage for allocation
failure, cleanup, stale-output suppression, and successful retry behavior; add
a focused validation gate; and update claim-safe documentation with exact
boundaries.

**Non-goals:** Broad reliability proof across multiple owners, allocator
redesign, public API or ABI changes, solver behavior changes, numerical
tolerance changes, performance claims, platform support expansion, release
claims, state-of-the-art claims, or cleanup of unrelated failure paths outside
the selected owner.

---

## Day 1: Sprint Intake and Reliability Candidate Inventory

**Title:** Reliability Intake
**Theme:** Establish Sprint 195 scope, candidate owners, prior proof patterns,
and selection evidence.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 195 section of the Epic 17 project plan and map items
   195.1 through 195.6 to likely owner files, tests, harnesses, documentation,
   and validation targets.
2. Review Sprint 187 reliability candidate artifacts and Sprint 193
   review-surface changes for current candidate owners and ownership
   boundaries.
3. Inventory allocation-heavy and failure-prone owners by allocation density,
   cleanup complexity, user-facing impact, source-list coupling, and existing
   negative-path tests.
4. Identify reusable deterministic failure-injection patterns, fail-at-count
   helpers, cleanup assertions, and retry expectations already present in the
   repository.
5. Create `WORKING_NOTES.md` with the candidate inventory, prior-pattern map,
   risk register, and Day 2 scoring questions.

### Deliverables

- Sprint 195 working-notes scaffold.
- Reliability candidate inventory.
- Prior deterministic failure-proof pattern map.
- Initial risk and selection-question register.

### Completion Criteria

- Sprint scope is traceable to items 195.1 through 195.6.
- Candidate owners are identified from current repository evidence.
- No implementation edits begin before selection criteria are recorded.

---

## Day 2: Owner Selection Scoring

**Title:** Owner Scoring
**Theme:** Score candidate owners by reliability payoff, proof feasibility,
cleanup risk, and review cost.
**Time estimate:** 12 hours

### Tasks

1. Score each candidate by allocation density, number of cleanup exits, stale
   output risk, retry semantics, public-user impact, and current test gaps.
2. Inspect current success-path and failure-path coverage for each candidate
   and record which assertions already constrain cleanup behavior.
3. Separate owners that can receive deterministic proof in one sprint from
   owners that require broader allocator, fixture, or API redesign.
4. Identify the selected owner, one fallback owner, rejected candidates, and
   the reason each rejection is out of scope for Sprint 195.
5. Produce the Day 2 selection artifact with item 195.1 acceptance evidence.

### Deliverables

- Ranked reliability-owner table.
- Selected owner and fallback owner decision.
- Rejected-candidate rationale.
- Item 195.1 acceptance notes.

### Completion Criteria

- Exactly one owner is selected for Sprint 195 implementation.
- Selection is backed by allocation, cleanup, user-impact, and testability
  evidence.
- The selected owner has a feasible deterministic proof path within the sprint.

---

## Day 3: Selected Owner Invariant Record

**Title:** Invariant Record
**Theme:** Define cleanup, publication, retry, and unsupported-breadth
invariants before harness or test changes.
**Time estimate:** 12 hours

### Tasks

1. Trace selected-owner control flow for allocation sites, publication points,
   cleanup exits, ownership transfer, output initialization, and error returns.
2. Define invariants for failed allocations, partially initialized state,
   stale-output suppression, successful retry, caller-owned inputs, and
   callee-owned temporaries.
3. Identify unsupported breadth, including untested allocators, unrelated
   owners, platform-specific behavior, and non-deterministic failure modes that
   remain outside the proof.
4. Map each invariant to an existing or planned assertion and note any required
   fixture or harness support.
5. Create the Day 3 invariant artifact with item 195.2 acceptance criteria.

### Deliverables

- Selected-owner invariant record.
- Allocation and cleanup path map.
- Publication and retry semantics statement.
- Unsupported-breadth and non-claim list.

### Completion Criteria

- Item 195.2 is complete enough to drive implementation and review.
- Every planned regression test maps to a recorded invariant.
- Unsupported breadth is explicit before documentation claims are updated.

---

## Day 4: Harness Design

**Title:** Harness Design
**Theme:** Design deterministic failure injection or owner-local fail-at-count
control for the selected proof.
**Time estimate:** 12 hours

### Tasks

1. Review existing allocation-failure hooks, fail-at-count tests, fixture
   cleanup helpers, and process-global state restoration patterns.
2. Choose whether to extend the existing deterministic failure harness or add a
   narrowly scoped owner-local fail-at-count control.
3. Define harness API, reset requirements, nesting behavior, thread-safety
   assumptions, and error reporting for the selected owner.
4. Identify source, header, test, Make, CMake, and source-list updates needed
   to wire the harness into focused validation.
5. Record the Day 4 harness design and review risks before editing code.

### Deliverables

- Harness extension design.
- Reset and ownership checklist.
- Build/source-list update map.
- Item 195.3 implementation plan.

### Completion Criteria

- Harness behavior is deterministic and bounded to the selected proof.
- Reset behavior is defined for both success and early-return paths.
- Required build and test integration points are known before implementation.

---

## Day 5: Harness Scaffold

**Title:** Harness Scaffold
**Theme:** Add the minimum harness scaffolding and prove it builds without
changing selected-owner behavior.
**Time estimate:** 12 hours

### Tasks

1. Add or extend the deterministic failure-injection helper according to the
   Day 4 design.
2. Wire test-only declarations, fixture setup, fixture teardown, and reset
   helpers using existing local conventions.
3. Update Make, CMake, source-list, or test-list ownership if the new harness
   requires build-system visibility.
4. Add smoke-level harness tests that prove fail-at-count selection and reset
   behavior without depending on the selected owner yet.
5. Run the smallest focused build or test command that proves the scaffold is
   wired correctly.

### Deliverables

- Deterministic harness scaffold.
- Reset helpers and smoke assertions.
- Updated build or source lists if needed.
- Day 5 validation notes.

### Completion Criteria

- Harness scaffold compiles and runs in a focused path.
- Reset behavior is covered before selected-owner regression tests rely on it.
- No selected-owner behavior is intentionally changed.

---

## Day 6: Selected Owner Harness Integration

**Title:** Owner Integration
**Theme:** Connect the deterministic harness to the selected owner while
preserving normal success behavior.
**Time estimate:** 12 hours

### Tasks

1. Add harness-controlled allocation or failure checkpoints at the selected
   owner boundaries identified in the invariant record.
2. Preserve normal success-path allocation ordering, status codes, output
   publication, cleanup ordering, and diagnostics unless the invariant record
   permits a local correction.
3. Ensure all harness state is restored before early returns, assertion
   failures, fixture teardown, and retry checks.
4. Run selected-owner success tests and harness smoke tests after integration.
5. Update working notes with changed files, checkpoint ordering, and any
   invariant adjustments discovered during implementation.

### Deliverables

- Harness-integrated selected owner.
- Preserved success-path validation notes.
- Updated checkpoint ordering record.
- Early-return restoration checklist.

### Completion Criteria

- Selected-owner success behavior remains unchanged.
- Harness checkpoints are deterministic and reviewable.
- Process-global or fixture-local harness state cannot leak across tests.

---

## Day 7: Failed Allocation Regression Tests

**Title:** Allocation Failures
**Theme:** Add deterministic tests for selected-owner failed allocation paths.
**Time estimate:** 12 hours

### Tasks

1. Add fail-at-count regression cases for every selected-owner allocation site
   or every representative allocation class recorded in the invariant map.
2. Assert expected status codes, absence of published stale outputs, cleanup of
   partial state, and stable caller-owned inputs.
3. Add fixture helpers to keep assertions compact without hiding ownership or
   cleanup semantics.
4. Run the focused selected-owner failure tests and fix any legitimate cleanup
   or publication defects they expose.
5. Record Day 7 coverage with allocation-site mapping and any deferred cases.

### Deliverables

- Failed allocation regression tests.
- Allocation-site coverage map.
- Cleanup and stale-output assertions.
- Focused test results.

### Completion Criteria

- Item 195.4 covers deterministic failed allocation behavior for the selected
  owner.
- Tests fail for leaked partial publication or wrong status behavior.
- Deferred allocation breadth, if any, is documented with rationale.

---

## Day 8: Cleanup and Stale-Output Proof

**Title:** Cleanup Proof
**Theme:** Strengthen assertions for cleanup completeness and stale-output
suppression after failure.
**Time estimate:** 12 hours

### Tasks

1. Add or refine assertions for nulling outputs, preserving caller-owned data,
   freeing partial temporaries, and avoiding double-free or reused stale state.
2. Exercise cleanup exits after early allocations, mid-construction
   allocations, late publication-adjacent allocations, and validation failures
   where applicable.
3. Add leak-sensitive or allocator-counter assertions when existing harnesses
   expose reliable counters.
4. Run failure-path tests under the available local diagnostic mode that best
   constrains cleanup behavior.
5. Update the invariant record with final cleanup proof coverage.

### Deliverables

- Cleanup-specific regression coverage.
- Stale-output suppression assertions.
- Updated invariant-to-test traceability.
- Diagnostic validation notes.

### Completion Criteria

- Cleanup behavior is asserted at each selected failure-path class.
- Stale outputs cannot survive selected-owner deterministic failures.
- Any unavailable cleanup diagnostics are documented as residuals.

---

## Day 9: Successful Retry Proof

**Title:** Retry Proof
**Theme:** Prove the selected owner can recover after deterministic failure and
complete a later successful call.
**Time estimate:** 12 hours

### Tasks

1. Add retry tests that intentionally fail the selected owner, reset harness
   state, rerun the same operation, and validate successful output.
2. Assert that retry results match an oracle, baseline, or existing success
   fixture within the selected owner's established tolerances.
3. Verify that caller-visible state after the failed attempt does not alter the
   successful retry path.
4. Run the selected-owner success, failure, cleanup, and retry tests together
   to detect ordering sensitivity.
5. Record Day 9 retry evidence and any ordering assumptions.

### Deliverables

- Successful retry regression tests.
- Retry output validation against existing oracle or baseline.
- Ordering-sensitivity check results.
- Item 195.4 retry acceptance notes.

### Completion Criteria

- A failed selected-owner call can be followed by a successful retry.
- Retry success does not depend on test ordering or leaked harness state.
- Retry assertions use existing numerical and ownership conventions.

---

## Day 10: Focused Gate Definition

**Title:** Focused Gate
**Theme:** Add a reviewable Make or CTest target for the selected
reliability proof.
**Time estimate:** 12 hours

### Tasks

1. Define the focused validation command that runs the selected owner success,
   failure, cleanup, retry, harness reset, and relevant source-list checks.
2. Add Make, CTest, or script wiring using the repository's existing target
   naming and guard conventions.
3. Ensure the focused gate is narrow enough for reviewers to run locally and
   broad enough to catch the Sprint 195 reliability regressions.
4. Add tests or guard checks that fail if the focused gate loses the selected
   proof coverage.
5. Run the focused gate and record results in working notes.

### Deliverables

- Focused reliability validation target.
- Gate coverage description.
- Guard against accidental coverage removal.
- Day 10 focused-gate result.

### Completion Criteria

- Item 195.5 has a runnable focused gate.
- The gate includes all selected reliability proof tests.
- The gate does not imply reliability proof for unrelated owners.

---

## Day 11: Documentation and Claim Boundaries

**Title:** Claim Boundaries
**Theme:** Update README, maintainer, and evidence wording with exact selected
owner reliability claims and non-claims.
**Time estimate:** 12 hours

### Tasks

1. Update user-facing or maintainer-facing documentation to describe the
   selected reliability proof, focused gate, and supported interpretation.
2. Add exact non-claim wording for unselected owners, unsupported allocator
   breadth, platform breadth, concurrency breadth, and broad library-wide
   reliability claims.
3. Link the invariant record or sprint artifact from maintainer documentation
   without making sprint notes the primary user contract.
4. Check existing README, maintainer guide, support matrix, and planning docs
   for wording that now needs calibration.
5. Record Day 11 documentation changes and retained residuals.

### Deliverables

- Claim-safe reliability documentation.
- Focused-gate usage wording.
- Updated residual and unsupported-breadth notes.
- Item 195.5 documentation acceptance notes.

### Completion Criteria

- Documentation claims match the selected proof exactly.
- Broad reliability, allocator, platform, and state-of-the-art claims are
  avoided.
- Maintainers can find the focused gate and invariant evidence.

---

## Day 12: Focused Validation and Source Ownership

**Title:** Focused Validation
**Theme:** Run focused reliability, source-list, formatting, and documentation
checks before full validation.
**Time estimate:** 12 hours

### Tasks

1. Run the focused reliability gate added on Day 10 and fix any selected-owner
   or harness regressions it exposes.
2. Run source-list and build-list checks affected by new tests, harness files,
   or validation targets.
3. Run formatting checks for modified C, header, script, and documentation
   surfaces.
4. Run relevant documentation or claim-boundary checks for updated README,
   maintainer, and planning text.
5. Update working notes with commands, results, fixes, and remaining risks.

### Deliverables

- Focused gate pass.
- Source-list and formatting validation results.
- Documentation check results.
- Day 12 fix log.

### Completion Criteria

- Focused Sprint 195 evidence passes locally.
- Source and build ownership are synchronized.
- Documentation wording passes available claim-boundary checks.

---

## Day 13: Full Quality Gate

**Title:** Full Validation
**Theme:** Run the full repository quality path and resolve any sprint-caused
failures.
**Time estimate:** 12 hours

### Tasks

1. Run `make format` and inspect any formatting changes before keeping them.
2. Run `make lint` and fix warnings or style defects caused by Sprint 195
   edits.
3. Run `make test` and fix selected-owner, harness, retry, or unrelated
   ordering failures caused by the new proof.
4. Run additional docs or focused gates if full validation does not cover
   Sprint 195 claim-boundary checks.
5. Record final full-gate commands and results in working notes.

### Deliverables

- Passing `make format` result.
- Passing `make lint` result.
- Passing `make test` result.
- Final validation log.

### Completion Criteria

- Item 195.6 full validation passes.
- No known Sprint 195 regression remains unresolved.
- Any environmental limitation is documented with exact command and failure
  context.

---

## Day 14: Closeout and Review Package

**Title:** Sprint Closeout
**Theme:** Package Sprint 195 evidence for review with traceability from plan
items to implementation, tests, docs, and validation.
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 195 changed files for scope control, no unrelated cleanup,
   stable wording, and exact claim boundaries.
2. Ensure items 195.1 through 195.6 each have evidence in artifacts, tests,
   docs, validation logs, or working notes.
3. Update `WORKING_NOTES.md` with final selected-owner proof summary,
   residuals, validation results, and review notes.
4. Prepare a concise review checklist covering selected owner, harness,
   failure tests, retry tests, focused gate, documentation, and non-claims.
5. Confirm there are no unstaged generated files, missing source-list entries,
   or unrecorded validation residuals.

### Deliverables

- Sprint 195 closeout notes.
- Item-to-evidence traceability checklist.
- Final residual and non-claim list.
- Review-ready change package.

### Completion Criteria

- Sprint 195 deliverables are complete and traceable.
- Reviewers can reproduce focused and full validation from documented
  commands.
- The change package proves one selected owner without overclaiming broader
  reliability.
