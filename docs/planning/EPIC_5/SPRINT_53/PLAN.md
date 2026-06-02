# Sprint 53 Plan: CSC Direct-Solver Completion & Dispatch Follow-Through

**Sprint Duration:** 14 days  
**Goal:** Close the most important deferred CSC direct-solver follow-through
items so the direct-solver lifecycle story is more complete, especially on the
analysis-aware indefinite LDL^T path and the Cholesky / LDL^T dispatch seams.
This sprint implements the Sprint 53 section of
`docs/planning/EPIC_5/PROJECT_PLAN.md`.

**Starting Point:** Sprint 52 closed with a stronger public
analysis/factor/refactor workflow, tighter repeated-run semantics, refreshed
factor-many benchmark proof, and expanded regression coverage. The remaining
high-value direct-solver gaps are now concentrated in the CSC-specific
follow-through work that earlier sprints intentionally deferred:
- analysis-aware indefinite LDL^T completion
- more transparent LDL^T CSC/native/supernodal dispatch
- stronger indefinite factor-many proof
- clearer Cholesky / LDL^T dispatch documentation and behavior
- targeted CSC regression and benchmark refresh

**End State:** Sprint 53 leaves behind a more complete and better proved CSC
direct-solver package: the analysis-aware indefinite LDL^T path is audited and
strengthened, dispatch behavior is easier to reason about, high-signal CSC
benchmarks and regressions are refreshed, and the sprint closes from a fully
validated reviewed baseline.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
144 hours, matching the Sprint 53 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 53 Scope Audit & CSC Baseline

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 53 project-plan items plus the Sprint 52 closeout
package into a bounded CSC follow-through map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 53 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`,
   the Sprint 52 closeout artifact, the Sprint 52 retrospective, and the most
   relevant deferred CSC notes from prior sprints.
2. Reconfirm the preserved Sprint 53 constraints:
   - keep the Sprint 50-52 direct-lifecycle compatibility fence intact
   - deepen and clarify the existing CSC paths rather than redesigning the
     public direct-solver model
   - preserve one-shot LU / Cholesky / LDL^T first-class entry points
   - preserve the strongest local reviewed baseline and truthfulness anchors
3. Define the Sprint 53 workstreams explicitly:
   - analysis-aware indefinite path audit
   - LDL^T dispatch follow-through
   - indefinite factor-many proof
   - Cholesky / LDL^T dispatch reconciliation
   - CSC benchmark and regression refresh
   - validation and closeout
4. Record the highest-risk seams for the sprint:
   - analysis-aware indefinite fallback or hidden rebuilds
   - dispatch behavior that is correct but difficult to reason about
   - benchmark claims outpacing measured CSC evidence
   - regression gaps around indefinite repeated-run workflows
5. Open Sprint 53 working notes and record starting assumptions, landing
   order, and touched-surface expectations.

### Deliverables
- Sprint 53 scope inventory
- CSC baseline notes
- Working-notes starting assumptions

### Completion Criteria
- Sprint 53 starts from the Sprint 52 validated implementation state rather
  than reopening public lifecycle design
- Preserved compatibility and validation constraints are explicit before CSC
  code work begins
- The CSC completion workstreams are named before audits or patches land

---

## Day 2: Validation Baseline & Touched-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed local baseline and the exact CSC rerun set
Sprint 53 code days must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the maintained reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity
   - current truthfulness-anchor counts
2. Reconfirm the mandatory gate for later `*.c` / `*.h` CSC batches:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial shared direct-solver or CSC
   dispatch batches:
   - `make quality-review-full`
4. Refresh the targeted CSC follow-on binaries Sprint 53 is most likely to
   need:
   - `./build/bench_refactor_csc`
   - `./build/test_chol_csc`
   - `./build/test_ldlt_csc`
   - `./build/test_cholesky`
   - `./build/test_ldlt`
   - `./build/test_etree`
   - `./build/test_integration`
   - `./build/example_analysis`
5. Record the smallest authoritative validation boundary for docs-only days
   versus CSC code-touch days.

### Deliverables
- Refreshed validation/truthfulness notes
- Sprint 53 CSC rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 53 uses the same baseline wording and parity anchors as the live repo
- The authoritative CSC rerun set is explicit before implementation work
  begins
- No validation ambiguity remains around dispatch or indefinite-path landing
  days

---

## Day 3: Analysis-Aware Indefinite Path Audit

**Title:** Path Audit  
**Theme:** Audit the live analysis-aware LDL^T indefinite CSC path before
trying to extend or rely on it more heavily  
**Time estimate:** 10 hours

### Tasks
1. Audit the live analysis-aware indefinite CSC path across:
   - `include/sparse_analysis.h`
   - `src/sparse_analysis.c`
   - `src/sparse_ldlt.c`
   - `src/sparse_ldlt_csc.c`
   - related internal CSC headers
2. Identify where `ldlt_csc_from_sparse_with_analysis(...)` already satisfies
   the intended repeated-run story and where it still falls back or rebuilds
   too much state.
3. Separate acceptable indefinite-family specifics from avoidable CSC
   integration drift.
4. Rank the highest-value Phase 3 CSC targets and explicitly reject broader
   redesign surfaces.
5. Write the audit artifact and the ranked implementation target list.

### Deliverables
- Analysis-aware indefinite path audit
- Fallback/drift inventory
- Ranked CSC implementation targets

### Completion Criteria
- The main indefinite CSC problem is reduced to named seams instead of a
  generic “complete the path” instruction
- The existing analysis/factors public boundary remains explicit
- Sprint 53 can start implementation from a concrete audit

---

## Day 4: Analysis-Aware LDL^T Integration Batch I

**Title:** Indefinite Batch I  
**Theme:** Strengthen the highest-value analysis-aware indefinite LDL^T CSC
path without broadening scope  
**Time estimate:** 12 hours

### Tasks
1. Land the first bounded CSC indefinite-path patch on the strongest shared
   seam identified on Day 3.
2. Keep the repeated-run direct story analysis-centric:
   - analyze once
   - factor / solve
   - refactor / solve many
3. Preserve accepted indefinite-family specifics where the shared path should
   not absorb them.
4. Add focused maintainership comments only where the new path would otherwise
   be unclear.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- First analysis-aware indefinite integration patch
- Preserved one-shot compatibility behavior
- Validation output for batch I

### Completion Criteria
- A high-value analysis-aware indefinite seam is strengthened or simplified
- The public lifecycle contract remains intact
- All required validation passes before the next CSC batch

---

## Day 5: Analysis-Aware LDL^T Integration Batch II

**Title:** Indefinite Batch II  
**Theme:** Extend the CSC indefinite follow-through to the next highest-value
seam without reopening larger redesign questions  
**Time estimate:** 12 hours

### Tasks
1. Land the second bounded CSC indefinite patch on the next ranked seam.
2. Reconfirm that the patch strengthens reuse of symbolic/permutation setup
   and does not overpromise reuse of stale numeric or pivot state.
3. Preserve accepted linked-list or non-CSC family behavior where the shared
   CSC path still should not absorb it.
4. Add focused regression coverage if the touched seam would otherwise be
   under-proved.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched CSC follow-ons justified by the batch

### Deliverables
- Second analysis-aware indefinite integration patch
- Updated proof for the touched seam
- Validation output for batch II

### Completion Criteria
- A second high-value indefinite CSC seam is removed or reduced
- Reuse semantics remain honestly bounded
- Required validation passes before dispatch-focused work begins

---

## Day 6: Transparent LDL^T Dispatch Batch I

**Title:** Dispatch Batch I  
**Theme:** Make LDL^T CSC/native/supernodal dispatch easier to reason about on
the highest-value path  
**Time estimate:** 12 hours

### Tasks
1. Audit the live LDL^T dispatch flow against the Day 3 findings.
2. Land the first bounded dispatch-follow-through patch.
3. Preserve existing behavior where it is already correct and intentionally
   family-specific.
4. Keep public behavior additive and clarifying rather than redesigning the
   surface.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- First LDL^T dispatch reconciliation patch
- Updated dispatch reasoning notes
- Validation output for dispatch batch I

### Completion Criteria
- LDL^T dispatch behavior becomes materially easier to reason about
- No compatibility regression is introduced on one-shot or lifecycle paths
- Validation passes before further dispatch work begins

---

## Day 7: Transparent LDL^T Dispatch Batch II

**Title:** Dispatch Batch II  
**Theme:** Extend dispatch follow-through to the next highest-value LDL^T CSC
seam while keeping the patch set bounded  
**Time estimate:** 10 hours

### Tasks
1. Land the second bounded LDL^T dispatch patch on the next ranked seam.
2. Reconfirm the relationship between:
   - CSC path
   - native path
   - supernodal path
   - explicit/implicit dispatch decisions
3. Preserve any valid indefinite-family exceptions instead of flattening them
   away.
4. Add focused proof if the touched seam would otherwise remain under-tested.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched follow-ons justified by the batch

### Deliverables
- Second LDL^T dispatch reconciliation patch
- Updated dispatch proof
- Validation output for dispatch batch II

### Completion Criteria
- A second dispatch seam is clarified or simplified
- The LDL^T CSC dispatch story is more uniform with Cholesky where it should
  be
- Required validation passes before factor-many proof work begins

---

## Day 8: Indefinite Factor-Many Benchmark Proof

**Title:** Benchmark Proof  
**Theme:** Refresh and strengthen the measured indefinite CSC factor-many story
on the intended workloads  
**Time estimate:** 12 hours

### Tasks
1. Audit the current indefinite CSC benchmark surfaces and identify the
   highest-value proof gap.
2. Land the bounded benchmark refresh on the strongest CSC indefinite
   repeated-run workload.
3. Keep the benchmark claims narrower than the measured evidence.
4. Update benchmark-local documentation only if the live benchmark contract
   changes.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `./build/bench_refactor_csc`
   - any additional targeted CSC benchmark reruns justified by the batch

### Deliverables
- Refreshed indefinite CSC factor-many benchmark proof
- Updated benchmark-local notes if needed
- Validation output for the benchmark batch

### Completion Criteria
- Sprint 53 has measured indefinite CSC factor-many evidence rather than only
  inferred reasoning
- Benchmark claims remain truthful and reproducible
- All required validation passes before documentation reconciliation begins

---

## Day 9: Cholesky / LDL^T Dispatch Reconciliation Audit

**Title:** Reconciliation Audit  
**Theme:** Identify the smallest high-signal Cholesky / LDL^T dispatch
reasoning gaps that still remain after the CSC batches  
**Time estimate:** 12 hours

### Tasks
1. Audit the live dispatch story across:
   - `README.md`
   - direct headers
   - CSC-specific tests
   - benchmark notes
2. Separate real reasoning drift from acceptable family-local differences.
3. Rank the smallest useful documentation or behavior follow-through targets.
4. Explicitly defer any tutorial-scale or broad docs rewrite work that exceeds
   Sprint 53 scope.
5. Write the audit artifact and the ranked reconciliation target list.

### Deliverables
- Dispatch-reconciliation audit
- Ranked high-signal follow-through targets
- Deferred non-goal list

### Completion Criteria
- Remaining reconciliation work is reduced to named targets instead of generic
  “clean up docs” wording
- Acceptable family-local differences remain explicit
- Sprint 53 can land only the smallest useful follow-through batch next

---

## Day 10: Dispatch Reconciliation Batch

**Title:** Reconciliation Batch  
**Theme:** Land the smallest high-signal Cholesky / LDL^T dispatch
clarification batch  
**Time estimate:** 10 hours

### Tasks
1. Land the bounded reconciliation batch on the highest-value Day 9 targets.
2. Keep public wording aligned with actual dispatch behavior and measured CSC
   evidence.
3. Avoid broad documentation or API churn outside the identified targets.
4. Re-run only the focused follow-ons justified by the touched surfaces if the
   batch is docs-only; otherwise use the full code-day gate.
5. Record the resulting public dispatch story and any still-intentional
   differences.

### Deliverables
- Cholesky / LDL^T dispatch reconciliation patch
- Updated public/maintainer wording
- Validation output appropriate to the touched surfaces

### Completion Criteria
- High-signal dispatch reasoning drift is materially reduced
- The docs and behavior tell the same CSC story
- No unnecessary scope expansion occurs late in the sprint

---

## Day 11: Benchmark and Regression Updates

**Title:** Regression Batch  
**Theme:** Add or refresh the most valuable CSC regression proof still missing
after the main implementation work  
**Time estimate:** 8 hours

### Tasks
1. Add or refresh the smallest high-value CSC regression coverage still
   missing after Days 4-10.
2. Prefer direct proof of:
   - indefinite analysis-aware path behavior
   - dispatch invariants
   - factor-many correctness under same-pattern updates
3. Keep the patch focused on real proof gaps rather than test-suite bulk
   growth.
4. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched CSC follow-ons justified by the batch
5. Record what proof gap is now closed and what remains intentionally deferred.

### Deliverables
- Focused CSC regression expansion
- Updated proof notes
- Validation output for the regression batch

### Completion Criteria
- The most important remaining CSC proof gap is closed
- Regression scope stays focused and defensible
- Required validation passes before final audit and validation sweep

---

## Day 12: Post-Landing Compatibility Audit

**Title:** Compatibility Audit  
**Theme:** Verify that the landed Sprint 53 branch still matches the Sprint
50-52 compatibility fence  
**Time estimate:** 8 hours

### Tasks
1. Audit the landed Sprint 53 branch against the preserved compatibility rules:
   - one-shot direct APIs remain first-class
   - repeated direct runs remain analysis/factors-centric
   - CSC follow-through did not broaden into raw storage exposure or generic
     direct-handle redesign
2. Reconfirm that reuse/refactor and dispatch claims remain honestly bounded.
3. Reconfirm that benchmark and README language do not outrun the measured
   evidence.
4. Fix the Day 13 pre-validation checklist from the landed state.
5. Record any remaining future-facing queue that is not a Sprint 53 blocker.

### Deliverables
- Post-landing compatibility audit
- Day 13 validation checklist
- Future-facing deferred queue notes

### Completion Criteria
- No blocker-level compatibility drift remains before final validation
- The preserved scope fence is still explicit and intact
- Day 13 validation can run from a clean, audited state

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Re-run the full reviewed baseline plus the highest-value CSC
follow-ons from the final landed branch  
**Time estimate:** 10 hours

### Tasks
1. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm reviewed CMake parity and current truthfulness anchors.
3. Run the targeted Sprint 53 follow-ons:
   - `./build/bench_refactor_csc`
   - `./build/test_chol_csc`
   - `./build/test_ldlt_csc`
   - `./build/test_cholesky`
   - `./build/test_ldlt`
   - `./build/test_etree`
   - `./build/test_integration`
   - `./build/example_analysis`
4. Record representative measured CSC results that support the final sprint
   story.
5. Stop and fix any failing truthfulness or validation gap before Day 14.

### Deliverables
- Full Sprint 53 validation record
- Updated measured CSC evidence
- Final proof that the reviewed baseline still holds

### Completion Criteria
- All required validation passes from the final Sprint 53 branch
- Reviewed parity and truthfulness anchors remain exact
- No unresolved blocker remains for sprint closeout

---

## Day 14: Closeout and Handoff

**Title:** Closeout  
**Theme:** Convert the validated Sprint 53 result into a clean handoff package
for the next Epic 5 sprint  
**Time estimate:** 8 hours

### Tasks
1. Summarize the final Sprint 53 outcome:
   - analysis-aware indefinite LDL^T completion state
   - LDL^T dispatch follow-through state
   - indefinite factor-many benchmark state
   - Cholesky / LDL^T dispatch reconciliation state
   - benchmark/regression proof state
2. Record the final validated baseline and representative CSC evidence.
3. Capture any explicit remaining queue for the next sprint without expanding
   it into replanning.
4. Check whether Sprint 53 surfaced any required `PROJECT_PLAN.md` update.
5. Write the closeout artifact and Sprint 54 handoff notes.

### Deliverables
- Sprint 53 closeout artifact
- Sprint 54 handoff notes
- Explicit final validated baseline summary

### Completion Criteria
- Sprint 53 closes from a validated and well-scoped CSC completion state
- The remaining queue is explicit and future-facing rather than a hidden
  closeout defect
- The sprint hands off a coherent next-step package to Sprint 54
