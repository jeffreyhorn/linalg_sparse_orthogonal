# Sprint 32 Plan: Test-Suite Truthfulness & Dormant Scaffold Resolution

**Sprint Duration:** 14 days  
**Goal:** Make the test suite accurately represent what is executed and protected today by resolving dormant scaffolding, formalizing slow/experimental checks, and reducing warning noise from test code that does not actually run. This sprint implements the Sprint 32 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`.

**Starting Point:** Sprint 31 closed the benchmark/example warning queue and left the authoritative Apple Clang serialized CMake full-tree inventory at `98` warnings: `src` `0`, `tests` `98`, `benchmarks` `0`, `examples` `0`. The remaining warning debt is now explicitly test-only, led by `tests/test_reorder_nd.c` dormant-scaffold concerns, high-volume designated-initializer drift in files such as `tests/test_ldlt.c`, `tests/test_colamd.c`, `tests/test_chol_csc.c`, and `tests/test_cholesky.c`, and residual mechanical `-Wdouble-promotion` warnings in integration and numeric test files.

**End State:** Sprint 32 leaves behind a test suite whose active protection surface is honest and documented, removes or formalizes dormant scaffold in `tests/test_reorder_nd.c`, reduces the named test-warning queue substantially, defines the project’s slow/experimental test policy, and preserves a green standard validation flow after the structural cleanup.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 128 hours, matching the Sprint 32 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 32 Scope Audit & Baseline

**Title:** Test Baseline  
**Theme:** Convert the Sprint 31 handoff into a precise Sprint 32 test-cleanup inventory  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 32 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`, the Sprint 31 handoff, and the Sprint 31 retrospective so the work stays tied to the explicit deferred queue.
2. Inspect `tests/test_reorder_nd.c` and the named high-volume warning files called out for Sprint 32.
3. Reproduce the current authoritative Apple Clang serialized CMake warning state and isolate the remaining test warnings by file and warning class.
4. Record the current state of dormant or historical-only scaffold in `tests/test_reorder_nd.c`, including which helpers are compiled, which checks run, and which code appears to be advisory only.
5. Open Sprint 32 working notes and record the file-by-file queue grouped by dormant scaffold, designated-initializer drift, mechanical double-promotion, and documentation/policy gaps.

### Deliverables
- Sprint 32 file-by-file work inventory
- Current test-warning snapshot for the Sprint 32 target files
- Baseline notes for `tests/test_reorder_nd.c` execution truthfulness

### Completion Criteria
- Every named Sprint 32 target file has a documented problem statement
- The remaining `tests/` warning sites are separated by category before edits begin
- The sprint scope is explicitly limited to test-suite truthfulness and test-warning cleanup

---

## Day 2: Dormant Scaffold Audit — `test_reorder_nd.c`

**Title:** ND Audit  
**Theme:** Define exactly which parts of `tests/test_reorder_nd.c` are active, dormant, historical, or misleading  
**Time estimate:** 8 hours

### Tasks
1. Audit `tests/test_reorder_nd.c` for compiled-but-unexecuted helpers, commented-out `RUN_TEST` sites, historical experiment code, and implicit “expected fail” scaffold.
2. Compare the file’s active `RUN_TEST` surface against the helpers and notes around the dormant sections.
3. Identify which unused functions are simply dead, which are recoverable as real tests, and which belong in docs or artifacts instead of the main suite.
4. Check whether the file also contains designated-initializer drift or warning sites that should be resolved in the same pass.
5. Write a short design note capturing the intended end state before structural edits begin.

### Deliverables
- Audited map of active vs dormant `test_reorder_nd.c` code
- Design note for the dormant-scaffold resolution approach
- Initial decision list for delete vs formalize vs keep-active paths

### Completion Criteria
- No ambiguity remains about which `test_reorder_nd.c` code is actually part of the current active suite
- The main structural choices are documented before edits begin
- The dormant-scaffold problem is defined narrowly enough to implement without scope creep

---

## Day 3: Dormant Scaffold Resolution Design & Policy Inputs

**Title:** Truthfulness Design  
**Theme:** Choose how the project should represent slow, experimental, and historical tests going forward  
**Time estimate:** 10 hours

### Tasks
1. Review the current test harness conventions to determine the least invasive way to distinguish normal, slow opt-in, and experimental/historical checks.
2. Decide when deletion is the correct answer versus when an explicit skip/experimental mechanism is justified.
3. Define the naming or policy rules Sprint 32 will follow for historical evidence that should not pretend to be active protection.
4. Capture the policy implications for `tests/test_reorder_nd.c` and any similar future files.
5. Write the Sprint 32 policy/design note that Day 4 and Day 5 will implement.

### Deliverables
- Slow/experimental/historical test policy note
- Chosen representation for non-default checks
- Decision framework for dormant scaffold resolution

### Completion Criteria
- The project-level policy is specific enough to implement in code and docs
- The chosen approach fits the current test harness instead of bypassing it
- `test_reorder_nd.c` has a clear structural end state before code changes begin

---

## Day 4: Framework & Convention Support

**Title:** Harness Support  
**Theme:** Add the minimal support needed to represent slow or experimental checks honestly  
**Time estimate:** 8 hours

### Tasks
1. Implement the smallest needed framework extension, helper pattern, or naming convention chosen on Day 3.
2. Ensure the mechanism is easy to audit in future test files and does not silently hide active failures.
3. Add focused tests or self-checks for the new support path if code changes are involved.
4. Document how maintainers should use the new convention in Sprint 32 notes.
5. Verify that the extension does not disturb the current active Makefile/CTest test flow.

### Deliverables
- Minimal harness/policy support for skipped or experimental checks
- Notes documenting the new usage pattern
- Validation evidence that the active suite behavior is preserved

### Completion Criteria
- The new support is real and testable, not just documented aspiration
- Active tests still run the same way by default
- The extension is small enough to be maintainable

---

## Day 5: `test_reorder_nd.c` Structural Cleanup

**Title:** ND Cleanup I  
**Theme:** Apply the chosen truthfulness model to `tests/test_reorder_nd.c`  
**Time estimate:** 10 hours

### Tasks
1. Remove truly dead historical-only scaffold from `tests/test_reorder_nd.c`, or move it into docs/artifacts if it remains useful as evidence.
2. Convert any justified non-default checks to the explicit slow/experimental mechanism chosen earlier in the sprint.
3. Eliminate the file’s `-Wunused-function` debt through real structural cleanup rather than suppression.
4. Resolve any obvious designated-initializer drift in the same file while the structure is open.
5. Rebuild and run focused test validation after each small batch to catch harness regressions immediately.

### Deliverables
- Structural cleanup landed in `tests/test_reorder_nd.c`
- `-Wunused-function` debt in the file removed or explicitly formalized
- Focused validation notes for the edited test file

### Completion Criteria
- `tests/test_reorder_nd.c` now honestly represents what the active suite executes
- Dormant scaffold is removed or formalized, not left implicit
- The file no longer carries stale unused-function debt

---

## Day 6: Coverage-Honesty Docs & Notes

**Title:** Honesty Docs  
**Theme:** Document the active vs opt-in test split introduced by the structural cleanup  
**Time estimate:** 8 hours

### Tasks
1. Update planning/docs notes to explain which test classes are active by default, which are opt-in, and which historical checks are retained only as evidence.
2. Cross-check that the docs match the actual `tests/test_reorder_nd.c` end state after Day 5.
3. Record how future contributors should avoid reintroducing dormant scaffold into normal test files.
4. Capture any open questions or residual out-of-scope cases that Sprint 32 should defer rather than silently ignore.
5. Update Sprint 32 notes with the before/after truthfulness model.

### Deliverables
- Documentation for active, slow, and experimental test categories
- Updated Sprint 32 notes for the `test_reorder_nd.c` structural cleanup
- Deferred-notes list for any residual out-of-scope truthfulness issues

### Completion Criteria
- Documentation matches the executed protection surface
- Future contributors have a clear rule for historical or experimental checks
- No silent ambiguity remains about the Day 5 structural decisions

---

## Day 7: Initializer Warning Inventory & Batch Design

**Title:** Initializer Audit  
**Theme:** Turn the remaining test initializer warnings into a sequenced implementation plan  
**Time estimate:** 10 hours

### Tasks
1. Audit the high-volume designated-initializer warning files named for Sprint 32, starting with `tests/test_ldlt.c`, `tests/test_colamd.c`, `tests/test_chol_csc.c`, `tests/test_cholesky.c`, `tests/test_sprint12_integration.c`, and `tests/test_reorder.c`.
2. Confirm which warning sites are the same trailing-field growth pattern already fixed in Sprint 31 tooling.
3. Group the files into implementation batches that can be validated incrementally without opening too many large test files at once.
4. Decide whether any companion files such as `tests/test_sprint18_integration.c`, `tests/test_sprint19_integration.c`, `tests/test_sprint20_integration.c`, or `tests/test_etree.c` belong in the same cleanup batch.
5. Write the Sprint 32 batch-design note for the initializer cleanup pass.

### Deliverables
- Audited list of test-side initializer warning sites
- Chosen batch order for the initializer cleanup
- Notes documenting the reuse of the Sprint 31 designated-init pattern

### Completion Criteria
- The remaining initializer queue is fully enumerated before edits begin
- The cleanup batches are sequenced tightly enough to validate incrementally
- No ambiguity remains about the intended coding pattern

---

## Day 8: Designated Initializers — Batch I

**Title:** Initializer Batch I  
**Theme:** Remove the first half of the high-volume test initializer warning queue  
**Time estimate:** 8 hours

### Tasks
1. Replace brittle positional options-struct initialization with designated initializers in the first Sprint 32 batch of high-volume files.
2. Keep field ordering and readability consistent with the Sprint 31 benchmark/example cleanup.
3. Update nearby comments or helper examples in the touched tests if they still teach the brittle pattern.
4. Rebuild after each small batch and record warning deltas per file.
5. Update Sprint 32 notes with the first initializer-cleanup results.

### Deliverables
- First test initializer cleanup batch landed
- Incremental rebuild evidence for the touched files
- Notes documenting the warning reduction for Batch I

### Completion Criteria
- The first batch no longer relies on brittle positional initialization
- Warning reduction is measured file by file
- No behavior regressions are introduced in the active tests

---

## Day 9: Designated Initializers — Batch II

**Title:** Initializer Batch II  
**Theme:** Finish the remaining Sprint 32 initializer cleanup queue  
**Time estimate:** 10 hours

### Tasks
1. Apply the same designated-initializer cleanup to the remaining Sprint 32 test files in scope.
2. Sweep the full touched set for consistent field ordering and explicitness.
3. Rebuild after each small batch and record the warning delta for the final initializer queue.
4. Identify any residual initializer sites that should be deferred to a later sprint rather than half-cleaned.
5. Update Sprint 32 notes with the consolidated before/after initializer summary.

### Deliverables
- Remaining Sprint 32 initializer cleanup completed
- Consolidated initializer warning summary
- Deferred residual-initializer notes if anything remains out of scope

### Completion Criteria
- The Sprint 32 initializer queue is closed or explicitly deferred by named file
- The safer designated-init style is consistent across all touched test files
- The cleanup leaves behind a durable pattern future sprints can reuse

---

## Day 10: Double-Promotion Inventory & Batch Design

**Title:** Promotion Audit  
**Theme:** Turn the residual test-side `-Wdouble-promotion` warnings into a precise implementation plan  
**Time estimate:** 8 hours

### Tasks
1. Audit the residual mechanical `-Wdouble-promotion` files, including `tests/test_sprint20_integration.c`, `tests/test_svd.c`, `tests/test_sprint6_integration.c`, `tests/test_sprint18_integration.c`, `tests/test_bidiag.c`, and smaller companion files.
2. Confirm which warnings are simple constant-suffix or macro-return issues versus anything more subtle.
3. Group the warning sites into implementation batches that can be fixed and validated incrementally.
4. Decide on the standard replacement idioms Sprint 32 will use for these mechanical sites.
5. Write the batch-design note for the double-promotion cleanup pass.

### Deliverables
- Audited list of residual test-side `-Wdouble-promotion` sites
- Chosen cleanup idioms for the mechanical fixes
- Batch order for the double-promotion work

### Completion Criteria
- Every Sprint 32 double-promotion site has a documented fix strategy
- The fixes are scoped as mechanical cleanup rather than open-ended numeric refactoring
- The batch order is clear before edits begin

---

## Day 11: Double-Promotion Cleanup — Batch I

**Title:** Promotion Batch I  
**Theme:** Remove the first half of the residual mechanical `-Wdouble-promotion` queue  
**Time estimate:** 10 hours

### Tasks
1. Apply the chosen mechanical cleanup idioms to the first batch of residual `-Wdouble-promotion` files.
2. Rebuild and run focused tests after each small batch to verify no behavioral regressions.
3. Confirm that the fixes actually remove the intended warning sites rather than shifting them elsewhere.
4. Record per-file warning deltas in Sprint 32 notes.
5. Identify any surprising sites that are not as mechanical as the audit suggested.

### Deliverables
- First double-promotion cleanup batch landed
- Incremental validation evidence for the touched files
- Notes documenting the per-file warning reduction

### Completion Criteria
- The first batch’s `-Wdouble-promotion` sites are removed
- Active tests remain green on the touched paths
- Any non-mechanical outliers are explicitly called out

---

## Day 12: Double-Promotion Cleanup — Batch II & Reconciliation

**Title:** Promotion Batch II  
**Theme:** Finish the residual double-promotion queue and reconcile the remaining test warning state  
**Time estimate:** 8 hours

### Tasks
1. Apply the remaining mechanical `-Wdouble-promotion` fixes in Sprint 32 scope.
2. Rebuild after each small batch and capture the final warning delta for the class.
3. Reconcile the current test-warning inventory against the Day 1 baseline to see what remains.
4. Record any still-deferred warnings by named file and class rather than leaving a vague residual backlog.
5. Update Sprint 32 notes with the consolidated double-promotion cleanup summary.

### Deliverables
- Remaining Sprint 32 double-promotion cleanup completed
- Reconciled warning inventory for the remaining test queue
- Named-file deferred list for any unresolved residual warnings

### Completion Criteria
- The targeted `-Wdouble-promotion` queue is closed or explicitly deferred by file
- Warning-count reconciliation is updated before final validation
- No residual Sprint 32 warning debt is left implicit

---

## Day 13: Full Validation Sweep

**Title:** Validation Pass  
**Theme:** Prove the cleaned-up active test suite still passes and measure the final warning delta  
**Time estimate:** 10 hours

### Tasks
1. Re-run the authoritative Apple Clang serialized CMake warning reproduction path and compare the test-warning counts against the Sprint 31 starting baseline.
2. Run the full standard validation flow, including the active Makefile and CTest paths that Sprint 32 is supposed to preserve.
3. Verify that the truthfulness cleanup did not accidentally remove active protection without documenting it.
4. Record final before/after counts by file and warning class in Sprint 32 notes.
5. Identify any remaining warning debt that should route to later Epic 3 work rather than being silently carried forward.

### Deliverables
- Final Sprint 32 warning-count comparison
- Full validation evidence from the current branch state
- Residual deferred queue summary for later sprints

### Completion Criteria
- The active suite remains green after the structural cleanup
- The test-warning delta is measured on the authoritative path
- Remaining cleanup, if any, is explicitly routed forward

---

## Day 14: Closeout, Handoff & Deferred Queue Update

**Title:** Sprint Closeout  
**Theme:** Turn Sprint 32’s structural and warning-cleanup results into durable handoff inputs  
**Time estimate:** 12 hours

### Tasks
1. Write the Sprint 32 closeout notes summarizing which truthfulness issues were fully resolved and which were deferred.
2. Update the deferred queue for Sprint 33 and later so any remaining test-warning or dormant-scaffold debt stays attached to named files and classes.
3. Record the final warning delta relative to the Sprint 31 starting state.
4. Confirm that the slow/experimental test policy, test-honesty documentation, and validation results are all reflected in the sprint notes.
5. Prepare the inputs needed for the Sprint 32 retrospective and any later Epic 3 planning updates.

### Deliverables
- Sprint 32 closeout notes and deferred-queue update
- Final test-warning delta summary
- Handoff inputs for the next Epic 3 sprint

### Completion Criteria
- Sprint 32 results are documented in a form the next sprint can use directly
- Any remaining cleanup is explicitly routed rather than left implicit
- The sprint ends with a clear before/after state for test-suite truthfulness and test warning debt
