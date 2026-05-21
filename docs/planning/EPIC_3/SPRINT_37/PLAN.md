# Sprint 37 Plan: Auxiliary-Code Cleanup & Maintainability Refactor

**Sprint Duration:** 14 days  
**Goal:** Improve maintainability in the repo's non-core layers by reducing duplication, clarifying helper ownership, simplifying large auxiliary files, and tightening maintainer workflow guidance. This sprint implements the Sprint 37 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`.

**Starting Point:** Sprint 36 closed with a validated cross-platform quality-parity baseline, explicit enforced/staged/supplemental CI expectations, and a documented sanitizer-build-tree caveat for maintainers. Sprint 37 starts from that validated platform contract and shifts attention to maintainability debt in auxiliary code: duplicated test and benchmark helpers, sprawling Makefile quality-target layout, large support files that are becoming harder to reason about, and stale maintainer-facing workflow notes.

**End State:** Sprint 37 leaves behind a lower-maintenance auxiliary surface: shared test and benchmark helper logic is consolidated where appropriate, quality targets are easier to understand and safer to rerun after sanitizer paths, large auxiliary files are easier to navigate, and maintainer workflow documentation matches the reviewed-quality contract that the repo now enforces.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 148 hours, matching the Sprint 37 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 37 Scope Audit & Baseline

**Title:** Maintainability Baseline  
**Theme:** Convert the Sprint 37 project-plan items into a concrete maintainability audit scope  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 37 section of `docs/planning/EPIC_3/PROJECT_PLAN.md` plus the Sprint 32 and Sprint 36 handoff/retrospective docs so the sprint stays anchored to the documented prerequisites.
2. Confirm the validated baseline that must remain true through Sprint 37: reviewed Makefile wrappers pass, reviewed CMake parity remains auditable, cross-platform expectations stay explicit, and the Sprint 36 sanitizer caveat is preserved.
3. Inventory the likely auxiliary-code maintainability hotspots: test helper duplication, benchmark helper duplication, large helper-heavy files, quality-target sprawl, and maintainer-doc drift.
4. Record the file map and likely implementation batches before any edits begin.
5. Open Sprint 37 working notes and capture the initial command set, baseline assumptions, and likely risk surfaces.

### Deliverables
- Sprint 37 maintainability baseline
- Initial auxiliary-surface inventory
- Named first-pass audit targets for tests, benchmarks, Makefile targets, and maintainer docs

### Completion Criteria
- Sprint 37 starts from a documented Sprint 36 validated baseline
- The maintainability scope is separated from new feature work and from old resolved warning debt
- Likely high-value auxiliary cleanup surfaces are identified before edits begin

---

## Day 2: Test-Helper Consolidation Audit

**Title:** Test Helper Audit  
**Theme:** Identify duplicated or drifting helper logic across the maintained test tree  
**Time estimate:** 8 hours

### Tasks
1. Audit the test tree for duplicated fixture setup, matrix construction, assertion, and utility helper patterns that have grown across multiple files.
2. Distinguish between genuine consolidation candidates and locally-scoped helpers that are clearer left in place.
3. Pay special attention to Sprint 32 truthfulness work so helper cleanup does not blur active-vs-opt-in test intent.
4. Record the strongest consolidation candidates, likely landing files, and any contract risks.
5. Write the audit note that defines the test-helper cleanup queue.

### Deliverables
- Test-helper consolidation audit note
- Ranked list of real consolidation candidates
- Initial keep/extract/defer classification for test-side helper code

### Completion Criteria
- The test-helper queue is explicit before edits begin
- Local clarity is separated from real duplication
- The later implementation batch is bounded clearly enough to execute safely

---

## Day 3: Benchmark-Helper Consolidation Audit

**Title:** Benchmark Helper Audit  
**Theme:** Map duplicated setup and reporting logic across benchmark-side helper code  
**Time estimate:** 10 hours

### Tasks
1. Audit benchmark and example-adjacent support code for duplicated timing setup, CLI parsing patterns, result formatting, reorder labeling, and matrix-loading helpers.
2. Separate true shared-helper candidates from benchmark-specific logic that should stay local to preserve readability.
3. Cross-check the Sprint 31 benchmark-contract work so any consolidation preserves the current reorder, reporting, and portability behavior.
4. Record the benchmark-side candidate queue, likely shared-home files, and any risk to benchmark comparability.
5. Write the audit note that defines the benchmark-helper cleanup queue.

### Deliverables
- Benchmark-helper consolidation audit note
- Ranked benchmark/helper duplication queue
- Initial keep/extract/defer classification for benchmark-side helper code

### Completion Criteria
- The benchmark-helper queue is explicit before implementation begins
- Shared-helper opportunities are separated from intentional local behavior
- The benchmark cleanup batch is bounded enough to implement deliberately

---

## Day 4: Quality-Target Normalization Design

**Title:** Target Design  
**Theme:** Define a clearer quality-target layout before changing the Makefile structure  
**Time estimate:** 10 hours

### Tasks
1. Audit the current reviewed-quality, dead-code, coverage, and sanitizer-related targets in the Makefile for naming overlap, dependency sprawl, and operator ambiguity.
2. Decide which targets are true maintained operator entry points versus lower-level helper plumbing.
3. Fold the Sprint 36 sanitizer-built `build/` tree caveat into the design so direct and reviewed sweeps do not silently reuse stale instrumented artifacts.
4. Choose the target-ownership model for normalization: wrappers, prerequisite structure, cleanup expectations, and documentation boundaries.
5. Write the design note that defines the Makefile normalization batch.

### Deliverables
- Quality-target normalization design note
- Defined entry-point vs helper-target layout
- Explicit sanitizer-build-tree cleanup expectation for later implementation

### Completion Criteria
- The target-normalization contract is chosen before edits begin
- The sanitizer caveat is handled as a first-class design constraint
- Later Makefile changes have a concrete topology rather than ad hoc edits

---

## Day 5: Test-Helper Consolidation Design & Narrow Batch

**Title:** Test Helper Batch I  
**Theme:** Convert the test-helper audit into the safest first consolidation slice  
**Time estimate:** 12 hours

### Tasks
1. Choose the first test-helper consolidation batch from the Day 2 audit, limited to the clearest shared patterns with low semantic risk.
2. Implement the helper extraction or simplification in the most maintainable landing location, keeping names and ownership explicit.
3. Keep the batch narrow enough that any affected tests remain easy to reason about in code review.
4. Validate touched test files and helper behavior directly.
5. Record the residual test-helper queue that remains for later cleanup or deliberate deferral.

### Deliverables
- First test-helper consolidation batch
- Residual test-helper queue
- Updated notes describing the new shared-helper ownership

### Completion Criteria
- The safest high-value test-helper duplication is reduced
- Touched tests remain readable and behaviorally unchanged
- Remaining test-helper work is narrowed explicitly rather than left implicit

---

## Day 6: Benchmark-Helper Consolidation Design & Narrow Batch

**Title:** Benchmark Batch I  
**Theme:** Convert the benchmark-helper audit into the safest first consolidation slice  
**Time estimate:** 10 hours

### Tasks
1. Choose the first benchmark/helper consolidation batch from the Day 3 audit, limited to repetitive support logic with stable behavior contracts.
2. Implement the shared-helper extraction or simplification without changing benchmark result semantics or CLI expectations.
3. Recheck the touched benchmark/example surfaces against the Sprint 31 naming and portability decisions.
4. Validate touched benchmark binaries or support paths directly.
5. Record the residual benchmark-helper queue for later cleanup or deliberate deferral.

### Deliverables
- First benchmark-helper consolidation batch
- Residual benchmark-helper queue
- Updated notes describing benchmark-helper ownership after the cleanup

### Completion Criteria
- The safest high-value benchmark/helper duplication is reduced
- Touched benchmark behavior remains stable and reviewable
- Remaining benchmark-helper work is narrowed explicitly

---

## Day 7: Quality-Target Normalization Implementation

**Title:** Target Batch I  
**Theme:** Implement the reviewed-quality and sanitizer-safety normalization in the Makefile  
**Time estimate:** 12 hours

### Tasks
1. Implement the first Makefile normalization batch based on the Day 4 design, preserving current reviewed entry-point behavior while clarifying ownership and flow.
2. Encode the sanitizer-build-tree cleanup expectation so direct and reviewed sweeps stop tripping over stale instrumented artifacts.
3. Reduce duplication or ambiguity in the maintained target layout where the design justified it.
4. Validate the touched target paths directly, including both reviewed wrappers and the affected lower-level commands.
5. Record any remaining quality-target layout debt that still belongs to a later pass.

### Deliverables
- First quality-target normalization batch
- Sanitizer-safe direct/reviewed rerun behavior
- Residual target-layout queue

### Completion Criteria
- Maintained quality targets are easier to reason about after the batch
- The Sprint 36 sanitizer caveat is materially reduced or made explicit in operator flow
- Remaining target-layout work is bounded clearly

---

## Day 8: Large-File Maintainability Audit

**Title:** Large-File Audit  
**Theme:** Identify the auxiliary files whose size and structure now hurt maintainability most  
**Time estimate:** 12 hours

### Tasks
1. Inventory large auxiliary files across tests, benchmarks, scripts, docs, and Makefile-adjacent surfaces that are becoming difficult to navigate or review.
2. Separate files that are merely large from files whose size reflects real maintainability problems such as mixed concerns, helper drift, repeated local patterns, or stale notes.
3. Cross-check prior sprint artifacts so already-intentional long files are not split mechanically without benefit.
4. Choose the highest-value one or two maintainability targets for the sprint’s large-file pass.
5. Write the audit note that defines the large-file cleanup batch.

### Deliverables
- Large-file maintainability audit note
- Ranked large-file cleanup candidates
- Defined high-value batch for the later refactor pass

### Completion Criteria
- Large-file cleanup is driven by maintainability evidence, not line count alone
- The large-file target set is explicit before edits begin
- The later refactor batch is bounded to files that materially benefit from cleanup

---

## Day 9: Large-File Maintainability Refactor

**Title:** Large-File Batch I  
**Theme:** Simplify the highest-value auxiliary file(s) without changing behavior  
**Time estimate:** 10 hours

### Tasks
1. Implement the chosen large-file maintainability refactor from the Day 8 audit, focusing on structure, helper extraction, sectioning, or dead-note removal rather than semantics.
2. Keep the cleanup narrow enough that ownership and behavior remain clear after the change.
3. Revalidate any touched support paths directly.
4. Record any residual file-structure debt that still belongs to later follow-on work.
5. Update notes with before/after maintainability rationale grounded in the actual file shape.

### Deliverables
- First large-file maintainability refactor batch
- Residual large-file queue
- Updated notes with structural before/after rationale

### Completion Criteria
- The chosen large auxiliary surface is materially easier to navigate
- Behavior stays unchanged while structure improves
- Remaining large-file debt is explicit and smaller

---

## Day 10: Comment & Documentation Debt Audit

**Title:** Comment Audit  
**Theme:** Identify stale or duplicative maintainer-facing comments and support docs created by earlier sprint layers  
**Time estimate:** 10 hours

### Tasks
1. Audit auxiliary comments, support docs, workflow notes, and inline maintainer guidance for stale historical wording, duplicated explanations, or contradictions introduced across prior sprints.
2. Focus especially on areas touched by Sprint 34 through Sprint 36: reviewed wrappers, cross-platform parity language, dead-code staged/excluded expectations, and sanitizer caveats.
3. Separate comments/docs that need deletion, wording refresh, or pointer-style simplification.
4. Choose the highest-value cleanup batch that improves maintainer clarity without rewriting authoritative contracts unnecessarily.
5. Write the audit note that defines the comment/documentation cleanup batch.

### Deliverables
- Comment/documentation debt audit note
- Ranked maintainer-facing wording cleanup queue
- Defined cleanup batch for final implementation

### Completion Criteria
- Stale or duplicative maintainer guidance is mapped before edits begin
- Cleanup targets are chosen by clarity impact, not breadth
- Later documentation edits have a concrete bounded scope

---

## Day 11: Comment & Maintainer-Workflow Documentation Cleanup

**Title:** Workflow Docs Batch  
**Theme:** Refresh maintainer-facing docs so they match the current quality and platform contract  
**Time estimate:** 12 hours

### Tasks
1. Implement the comment/support-doc cleanup batch chosen on Day 10.
2. Update maintainer workflow documentation so it preserves the Sprint 36 enforced/staged/supplemental platform contract explicitly.
3. Document the clean-build expectation after sanitizer paths and the current staged/excluded limits on macOS and Windows without overstating parity.
4. Reduce duplicated guidance by pointing support docs back to the maintained authoritative surfaces where appropriate.
5. Record any residual maintainer-doc debt that still belongs to later sprints.

### Deliverables
- Maintainer workflow documentation refresh
- Updated comments/support docs aligned to the current platform contract
- Residual maintainer-doc queue, if any

### Completion Criteria
- Maintainer workflow docs match the shipped Sprint 34-Sprint 36 contract
- Sanitizer cleanup expectations and staged limits are explicit
- Duplicated or stale workflow wording is materially reduced

---

## Day 12: Focused Validation & Reconciliation

**Title:** Focused Recheck  
**Theme:** Validate the touched auxiliary surfaces before the full sprint sweep  
**Time estimate:** 12 hours

### Tasks
1. Re-run the focused validation paths for every touched maintainability batch: test helpers, benchmark helpers, quality-target normalization, large-file refactors, and maintainer-doc support paths.
2. Confirm that reviewed target behavior, sanitizer-safe reruns, and any extracted helpers still align with the intended contracts.
3. Reconcile any drift between implementation notes and the actual landed file layout.
4. Fix any small residual mismatch discovered during focused validation.
5. Record the resulting near-close state and remaining full-sweep requirements.

### Deliverables
- Focused validation record across all touched maintainability batches
- Reconciled sprint notes aligned to landed behavior
- Final full-sweep command list for Day 13

### Completion Criteria
- Every touched auxiliary surface has been rechecked directly
- Notes and code/doc state agree before the full validation sweep
- Only full-sweep verification remains before closeout

---

## Day 13: Full Validation Sweep

**Title:** Full Validation  
**Theme:** Re-run the maintained project quality contract after the auxiliary cleanup work  
**Time estimate:** 12 hours

### Tasks
1. Run the full maintained validation set required by the touched code and helper surfaces.
2. Reconfirm reviewed Makefile wrappers, reviewed CMake parity, dead-code flow, and the maintained direct gates after the Sprint 37 cleanup work.
3. Confirm the sanitizer-build-tree caveat is now either reduced operationally or documented correctly in the maintained operator flow.
4. Record timing, pass/fail status, and any important reconciliation details from the sweep.
5. Prepare the closeout summary grounded in the fully validated end state.

### Deliverables
- Full Sprint 37 validation sweep record
- Reconfirmed maintained quality baseline after auxiliary cleanup
- Closeout-ready summary of changed auxiliary surfaces and residual limits

### Completion Criteria
- The maintained validation contract passes after Sprint 37 changes
- Auxiliary cleanup did not regress reviewed-quality behavior
- The sprint is ready for closeout based on an explicit validated end state

---

## Day 14: Closeout, Handoff & Forward Queue

**Title:** Closeout  
**Theme:** Capture the maintainability gains, residual limits, and future-sprint follow-through  
**Time estimate:** 10 hours

### Tasks
1. Write the Sprint 37 handoff and retrospective grounded in the Day 13 validated end state.
2. Summarize the concrete maintainability gains across tests, benchmarks, Makefile targets, large auxiliary files, and maintainer docs.
3. Name any bounded residual cleanup or later-sprint maturation work that should be routed forward.
4. Update the Epic 3 project plan if Sprint 37 surfaces any new deferred work that later sprints must not miss.
5. Record the final clean-sprint state in the working notes.

### Deliverables
- Sprint 37 handoff
- Sprint 37 retrospective
- Project-plan updates for any real deferred follow-through

### Completion Criteria
- Sprint 37 closes with a documented validated end state
- Maintainability gains and residual limits are explicit
- Any later-sprint work is routed forward instead of left implicit
