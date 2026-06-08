# Sprint 59 Plan: Quality/Platform Follow-Through, Final Integration & Epic 5 Closeout

**Sprint Duration:** 14 days  
**Goal:** Reassess the remaining staged quality/platform limits, land only the
bounded follow-through that is still justified, run the final cross-surface
integration sweep, and close Epic 5 from a measured validated baseline. This
sprint implements the Sprint 59 section of
`docs/planning/EPIC_5/PROJECT_PLAN.md`.

**Starting Point:** Sprint 58 closed with the implementation, giant-test,
public-surface, and documentation cleanup work already landed:
- Sprint 50-58 feature and maintainability work complete
- repeated-run, factor-many, CSC, iterative-handle, and eigensolver-handle
  stories already validated
- the remaining queue now centers on staged quality/platform residuals,
  final cross-surface reconciliation, and Epic 5 closeout

The next highest-value work is not new solver capability. It is bounded
follow-through on the remaining quality/platform surfaces plus final integrated
truthfulness across:
- API/header wording
- docs/tutorial/examples/benchmarks
- maintained quality gates and parity anchors
- Epic 5 summary and residual-journal closeout

**End State:** Sprint 59 leaves behind one final measured Epic 5 validation
baseline, a smaller and more explicit residual quality/platform queue, a fully
reconciled caller-facing story across API/docs/examples/benchmarks/tests, and
the final Epic 5 closeout/handoff package.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
132 hours, matching the Sprint 59 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 59 Scope Audit & Final-Sprint Baseline

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 59 project-plan items plus the Sprint 58 close
state into a bounded final Epic 5 quality/platform and closeout map  
**Time estimate:** 9 hours

### Tasks
1. Re-read the Sprint 59 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`,
   the Sprint 58 closeout, and the latest Epic 5 retrospective/closeout notes.
2. Reconfirm the preserved Sprint 59 constraints:
   - no broad new feature work
   - no public API redesign
   - no fake platform closure claims without measured evidence
   - bounded follow-through only where still justified
3. Define the Sprint 59 workstreams explicitly:
   - quality/platform residual audit
   - bounded quality follow-through batch
   - cross-surface compatibility sweep
   - full validation sweep
   - Epic 5 summary/handoff
   - project-plan/residual-journal finalization
4. Record the strongest live Sprint 59 surfaces likely to be touched:
   - maintained quality wrappers
   - parity/truthfulness anchors
   - caller-facing docs/examples/benchmarks
   - Epic 5 closeout artifacts
5. Open Sprint 59 working notes and record the intended landing order and
   validation expectations.

### Deliverables
- Sprint 59 scope inventory
- Final-sprint baseline notes
- Working-notes starting assumptions

### Completion Criteria
- Sprint 59 starts from the Sprint 58 validated state rather than reopening
  design questions
- The final-sprint compatibility and scope fences are explicit before follow-
  through work begins
- The quality/platform and closeout workstreams are named before any edits land

---

## Day 2: Validation Baseline & Truthfulness Anchor Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the maintained reviewed baseline and exact rerun set the
final Epic 5 branch must preserve  
**Time estimate:** 8 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity counts
   - current truthfulness-anchor wording
2. Reconfirm the mandatory gate for any later `*.c` / `*.h` days:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial quality/platform code
   follow-through:
   - `make quality-review-full`
4. Refresh the targeted Sprint 59 rerun set most likely to be needed during
   final integration and validation:
   - direct/public lifecycle proofs
   - representative examples
   - representative benchmark drivers
   - parity/count anchors
5. Record the authoritative validation boundary for docs-only days versus code-
   touching days.

### Deliverables
- Refreshed validation/truthfulness notes
- Sprint 59 rerun list
- Final-sprint code-day validation checklist

### Completion Criteria
- Sprint 59 uses the same baseline wording and parity anchors as the live repo
- The authoritative rerun set is explicit before follow-through work begins
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Quality/Platform Residual Audit

**Title:** Residual Audit  
**Theme:** Reduce the remaining quality/platform problem to concrete staged
residuals before landing any follow-through batch  
**Time estimate:** 10 hours

### Tasks
1. Audit the remaining staged quality/platform surfaces named in the project
   plan:
   - serialized dead-code execution
   - macOS staging
   - Windows reviewed-wrapper parity
   - Windows dead-code exclusion
   - coverage calibration
2. Separate each residual into concrete classes:
   - already acceptable as deferred residual
   - needs bounded follow-through now
   - needs measurement before any change
   - no longer justified by the current repo state
3. Rank the residuals by:
   - truthfulness risk
   - maintenance burden
   - platform visibility
   - implementation cost
4. Reject generic “platform cleanup” expansion that would widen Sprint 59
   beyond a bounded final follow-through batch.
5. Write the residual-audit artifact and ranked landing order.

### Deliverables
- Quality/platform residual audit
- Ranked follow-through candidates
- Proposed first implementation boundary

### Completion Criteria
- The residual queue is reduced to named, defensible items
- At least one bounded implementation seam is justified by measurement or
  truthfulness risk
- Sprint 59 can start follow-through from a concrete map instead of a generic
  cleanup bucket

---

## Day 4: Quality Follow-Through Design

**Title:** Follow-Through Design  
**Theme:** Freeze the first bounded quality/platform landing boundary before
editing maintained build or validation surfaces  
**Time estimate:** 9 hours

### Tasks
1. Select the highest-value residual seam from Day 3.
2. Define the exact landing boundary across:
   - touched build or validation surfaces
   - expected parity/truthfulness changes
   - explicitly preserved non-goals
3. Define the invariants the batch must preserve:
   - reviewed baseline wording
   - Makefile/CMake parity truthfulness
   - platform-story honesty
   - stable local developer workflow
4. Define the cleanup policy for the touched surfaces:
   - minimize blast radius
   - prefer measurement-backed wording or logic
   - avoid broad platform abstraction redesign
5. Record the design artifact and landing checklist.

### Deliverables
- Quality follow-through design
- Landing boundary map
- Preserved invariants and checklist

### Completion Criteria
- The first follow-through boundary is explicit before code or build edits land
- Ownership is defined by one bounded residual seam, not vague cleanup intent
- Truthfulness expectations are fixed before high-signal quality surfaces move

---

## Day 5: Bounded Quality Follow-Through Batch I

**Title:** Quality Batch I  
**Theme:** Land the first bounded quality/platform follow-through patch  
**Time estimate:** 12 hours

### Tasks
1. Implement the selected Day 4 follow-through seam.
2. Keep the touched logic, wrapper, or platform wording bounded to the agreed
   residual target.
3. Reconcile the touched behavior with the maintained reviewed baseline and
   platform-story expectations.
4. Add or refresh narrowly scoped proof/guard coverage if needed.
5. Run the required validation gate for the touched file types and capture the
   key parity/truthfulness anchors.

### Deliverables
- Landed quality/platform follow-through patch
- Narrow proof/guard coverage as needed
- Updated validation record

### Completion Criteria
- The landed batch resolves a real residual without broadening Sprint 59 scope
- The maintained reviewed baseline still passes from the landed tree
- No new platform-story contradiction is introduced

---

## Day 6: Bounded Quality Follow-Through Batch II / Reconciliation

**Title:** Quality Batch II  
**Theme:** Finish the strongest remaining justified follow-through work without
turning Sprint 59 into a platform-redesign sprint  
**Time estimate:** 10 hours

### Tasks
1. Re-audit the landed Day 5 surfaces.
2. Either land one more bounded residual fix or explicitly record why the
   remaining items should stay deferred.
3. Normalize touched quality/platform wording so the final story reads as a
   stable maintained baseline rather than staged sprint-local caveats.
4. Record the consciously deferred residuals that should remain outside Sprint
   59.
5. Run the required validation gate if any code/build logic was touched.

### Deliverables
- Follow-through reconciliation patch or explicit defer decision
- Normalized quality/platform wording
- Updated deferred-residual note

### Completion Criteria
- The remaining quality/platform queue is smaller and more concrete after Day 6
- The touched quality story is more stable and less staged
- No unjustified implementation surface has been widened

---

## Day 7: Final Cross-Surface Compatibility Audit

**Title:** Compatibility Audit  
**Theme:** Reduce the final integration problem to explicit caller-story drift
classes before the last reconciliation batch lands  
**Time estimate:** 9 hours

### Tasks
1. Audit the strongest cross-surface story surfaces:
   - public headers
   - `README.md`
   - `docs/tutorial.md`
   - examples
   - benchmark docs
   - representative proof surfaces
2. Separate the remaining drift into concrete classes:
   - API/docs wording mismatch
   - example/docs mismatch
   - benchmark/docs mismatch
   - test/story mismatch
3. Rank the remaining reconciliation targets by:
   - caller visibility
   - confusion risk
   - ease of truthful correction
4. Reject broad cleanup that would turn the final integration pass into another
   full documentation sprint.
5. Write the compatibility-audit artifact and ranked landing order.

### Deliverables
- Cross-surface compatibility audit
- Ranked final reconciliation targets
- Proposed integration landing boundary

### Completion Criteria
- The final integration problem is reduced to named drift classes
- The first reconciliation target is justified by caller value, not only file
  size
- Sprint 59 can start the final integration batch from a concrete map

---

## Day 8: Final Integration Reconciliation Batch I

**Title:** Integration Batch I  
**Theme:** Land the first bounded final caller-story reconciliation patch  
**Time estimate:** 9 hours

### Tasks
1. Reconcile the highest-value drift seam from Day 7 across the touched
   surfaces.
2. Keep the changes bounded to final caller-story truthfulness:
   - API/docs consistency
   - example alignment
   - benchmark alignment
3. Remove any remaining stale final-sprint ambiguity from the touched sections.
4. Recheck the touched wording against the maintained proof/example/benchmark
   surfaces.
5. Run the required validation gate if any `*.c` / `*.h` files were touched.

### Deliverables
- Landed final integration reconciliation patch
- Updated truthfulness/alignment record

### Completion Criteria
- The strongest caller-story drift is removed without broadening scope
- The touched surfaces now agree more clearly than before
- No contradiction is introduced across the final public story

---

## Day 9: Final Integration Reconciliation Batch II

**Title:** Integration Batch II  
**Theme:** Finish the strongest remaining cross-surface drift without turning
the sprint into another broad cleanup pass  
**Time estimate:** 8 hours

### Tasks
1. Re-audit the landed integration surfaces after Day 8.
2. Land one more bounded reconciliation patch if the residual drift remains
   significant.
3. Normalize terminology so the final Epic 5 story prefers stable workflow and
   validation categories over sprint-local phrasing.
4. Record any intentionally deferred cross-surface density that remains outside
   Sprint 59.
5. Run targeted sanity checks across the touched surfaces.

### Deliverables
- Follow-through integration patch
- Normalized final caller-story wording
- Updated deferred-integration note

### Completion Criteria
- The remaining cross-surface drift is smaller and more concrete after Day 9
- Final caller-story language is more stable and less sprint-local
- No broad rewrite has been introduced late in the sprint

---

## Day 10: Epic 5 Closeout Input Audit

**Title:** Closeout Audit  
**Theme:** Gather the measured inputs needed for the final Epic 5 summary,
handoff, and residual-journal closeout  
**Time estimate:** 10 hours

### Tasks
1. Re-audit the final Epic 5 deliverables across Sprints 50-59:
   - direct lifecycle and CSC completion
   - iterative/eigensolver repeated-run support
   - large-source decomposition
   - giant-test refactor
   - public-surface simplification
   - quality/platform follow-through
2. Identify the final metrics, anchors, and residuals that need to appear in
   the closeout package.
3. Separate:
   - validated closed work
   - consciously deferred residuals
   - future-facing queue items
4. Reject summary inflation that overclaims closure beyond the measured
   baseline.
5. Write the closeout-input audit artifact.

### Deliverables
- Epic 5 closeout input audit
- Final metrics/residual inventory
- Ranked closeout-writing queue

### Completion Criteria
- The final closeout package has a concrete measured input set
- Deferred residuals are explicit before the summary writing begins
- No fake “everything is perfect” closeout language is required

---

## Day 11: Epic 5 Summary & Handoff Batch

**Title:** Summary Batch  
**Theme:** Land the main Epic 5 summary and handoff writing from the measured
Sprint 59 state  
**Time estimate:** 8 hours

### Tasks
1. Write the main Epic 5 summary/handoff artifact from the Day 10 input set.
2. Make the final validated baseline explicit:
   - maintained quality gates
   - parity anchors
   - targeted follow-on reruns
3. Make the preserved compatibility fence explicit:
   - no broad API redesign
   - stable public workflow boundary
   - consciously bounded residual queue
4. Keep the summary high-signal and measured rather than historical for its own
   sake.
5. Recheck the touched closeout surfaces for internal consistency.

### Deliverables
- Landed Epic 5 summary/handoff draft
- Final validated-baseline summary
- Preserved compatibility-fence wording

### Completion Criteria
- The main Epic 5 handoff story is written from measured evidence
- The validated baseline and residual queue are both explicit
- The summary does not rely on vague or inflated closure language

---

## Day 12: Project-Plan / Residual Journal Finalization

**Title:** Residual Finalization  
**Theme:** Reconcile final residual limits and project-level planning artifacts
from the landed Sprint 59 state  
**Time estimate:** 8 hours

### Tasks
1. Re-read `docs/planning/EPIC_5/PROJECT_PLAN.md` and the current residual
   journal/closeout notes.
2. Update only the project-level plan and residual artifacts that now need
   final measured closure or defer-state wording.
3. Make the future-facing queue explicit without reopening solved Sprint 50-59
   scope.
4. Reconcile any touched project-level wording against the final Epic 5 summary
   and handoff draft.
5. Record the finalization artifact and remaining non-goals.

### Deliverables
- Updated project-level summary/residual wording
- Residual-journal finalization artifact
- Final non-goals note

### Completion Criteria
- Project-level closeout wording matches the landed Sprint 59 state
- Residuals are explicit and future-facing rather than ambiguous
- No solved scope is reopened late in the sprint

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full maintained quality gates, truthfulness anchors, and
targeted follow-ons from the final Epic 5 state  
**Time estimate:** 12 hours

### Tasks
1. Run the full required baseline:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm reviewed CMake parity and test-count truthfulness anchors.
3. Rerun the targeted Sprint 59 follow-ons fixed from the final tree:
   - representative direct lifecycle surfaces
   - representative iterative/eigensolver surfaces
   - representative examples
   - representative benchmarks
4. Record representative retained outputs that support the final Epic 5 story.
5. Write the full-validation artifact from the landed state.

### Deliverables
- Full validation artifact
- Final parity/truthfulness anchor record
- Representative retained-output set

### Completion Criteria
- All required validation passes from the final Sprint 59 tree
- Reviewed parity remains exact
- No blocker-level drift remains before final closeout

---

## Day 14: Sprint 59 Closeout & Epic 5 Handoff

**Title:** Closeout  
**Theme:** Close Sprint 59 from the validated baseline and finalize the Epic 5
handoff state  
**Time estimate:** 10 hours

### Tasks
1. Summarize the final Sprint 59 deliverables across:
   - quality/platform follow-through
   - final cross-surface reconciliation
   - full validation baseline
   - Epic 5 summary/residual finalization
2. Record the preserved compatibility fence and the final validated baseline.
3. Capture the explicit deferred queue without hiding it inside the done state.
4. Recheck whether `docs/planning/EPIC_5/PROJECT_PLAN.md` needs any final
   correction from the landed state.
5. Write the final Sprint 59 closeout/handoff artifact and final working-notes
   summary.

### Deliverables
- Sprint 59 closeout artifact
- Final Epic 5 handoff state
- Final working-notes summary

### Completion Criteria
- Sprint 59 closes from the validated Day 13 baseline
- The final Epic 5 handoff state is explicit and measured
- Remaining work is clearly future-facing rather than hidden Sprint 59 debt
