# Sprint 44 Plan: Graph / ND Subsystem Decomposition Phase 2 & Large-Test Maintainability Batch

**Sprint Duration:** 14 days  
**Goal:** Finish the second graph / ND subsystem decomposition phase by extracting FM refinement, separator lifting, and remaining runtime-strategy glue from the residual monolith, then use the same bounded structural style to start the first maintainability cleanup batch for the largest test binaries. This sprint implements the Sprint 44 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`.

**Starting Point:** Sprint 43 closed with a validated Phase-1 graph decomposition package: graph ownership moved to `src/sparse_graph_core.c`, hierarchy/coarsening moved to `src/sparse_graph_coarsen.c`, coarse bisection moved to `src/sparse_graph_bisect.c`, and the remaining `src/sparse_graph.c` was narrowed to FM refinement, separator lifting, and top-level orchestration. Sprint 44 begins from that baseline plus the Sprint 40 hotspot inventory, which still identifies the largest test binaries as the next maintainability concentration after the graph monolith.

**End State:** Sprint 44 leaves behind a much smaller graph orchestration layer with explicit module ownership for FM refinement and separator lifting, cleaner runtime-strategy parsing, and the first real helper/fixture consolidation batch in the largest test binaries without changing the one-binary-per-test model or reopening public API work.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 148 hours, matching the Sprint 44 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 44 Scope Audit & Baseline Refresh

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 44 project-plan items into a bounded graph Phase-2 and large-test maintainability execution map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 44 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`, the Sprint 43 closeout artifacts, and the Sprint 40/41/42 validation and prep-rule artifacts.
2. Reconfirm the preserved constraints Sprint 44 must not reopen:
   - internal-first graph refactoring
   - no public API churn
   - Sprint 40 validation anchor
   - Sprint 41 shared safety-helper reuse
   - Sprint 42 lifecycle compatibility rules
   - Sprint 43 Phase-1 graph ownership boundaries
3. Define the Sprint 44 workstreams explicitly:
   - FM refinement extraction
   - separator lifting extraction
   - runtime strategy parsing cleanup
   - final orchestration cleanup
   - large-test helper audit
   - first test-helper consolidation batch
   - validation closeout
4. Record the highest-risk remaining graph seams:
   - FM gain bucket / pass control flow
   - edge-to-vertex separator lifting and policy selection
   - environment/config parsing interleaved with partition hot paths
   - top-level retry/orchestration glue
5. Open Sprint 44 working notes and record scope, assumptions, and initial landing order.

### Deliverables
- Sprint 44 scope inventory
- Graph Phase-2 and test-maintainability workstream map
- Working-notes baseline assumptions

### Completion Criteria
- Sprint 44 starts from the documented Epic 4 baseline rather than ad hoc code motion
- The preserved constraints are explicit before Phase-2 extraction begins
- The graph and large-test targets are named before implementation starts

---

## Day 2: Residual Graph Seam Inventory Refresh

**Title:** Residual Graph Inventory  
**Theme:** Re-map the remaining `src/sparse_graph.c` seams after Sprint 43 before choosing the Phase-2 extraction order  
**Time estimate:** 10 hours

### Tasks
1. Refresh the live seam inventory for the residual `src/sparse_graph.c` and classify the remaining implementation regions:
   - FM refinement core
   - separator lifting / policy selection
   - top-level partition orchestration
   - runtime strategy and override parsing
   - retry / fallback glue
2. Mark which helpers are stable extraction candidates now and which still span multiple phases.
3. Separate:
   - extract-now FM support logic
   - extract-now separator support logic
   - runtime/config parsing that can move without behavior change
   - orchestration glue that should remain until after the extractions land
4. Record any shared declarations that will need header promotion versus translation-unit-local retention.
5. Write the residual graph seam inventory artifact.

### Deliverables
- Refreshed residual graph seam inventory
- Stable-seam vs still-coupled classification
- Initial Phase-2 extraction order notes

### Completion Criteria
- The remaining graph monolith is reduced to named Phase-2 seams
- Extraction-ready logic is separated from orchestration glue
- Later implementation order is grounded in the live residual file

---

## Day 3: FM Refinement Module Boundary Design

**Title:** FM Boundary Design  
**Theme:** Define the ownership, local-helper rules, and interface surface for FM refinement extraction  
**Time estimate:** 10 hours

### Tasks
1. Design the target FM refinement module boundary:
   - candidate implementation unit name
   - owned helper clusters
   - owned bucket/control-flow state
   - declarations that must become shared
2. Decide which FM-adjacent helpers remain local to orchestration versus move with the refinement core.
3. Define naming and ownership expectations for:
   - gain-bucket helpers
   - pass/rollback helpers
   - local score/update helpers
4. Record dependencies on `src/sparse_graph_fm_buckets.h` and the shared internal graph header.
5. Write the FM boundary design artifact.

### Deliverables
- FM refinement extraction design
- Shared vs local helper ownership map
- Header dependency notes

### Completion Criteria
- The FM extraction target is explicit before code edits begin
- Bucket/control-flow ownership is documented
- The sprint stays bounded away from broad algorithm redesign

---

## Day 4: Separator, Runtime, and Large-Test Design

**Title:** Separator and Test Design  
**Theme:** Bound the separator-lifting seam, runtime parsing cleanup seam, and large-test maintainability target set before implementation  
**Time estimate:** 10 hours

### Tasks
1. Design the target separator-lifting module boundary:
   - edge-to-vertex separator extraction
   - policy selection helpers
   - local scoring / conversion helpers
2. Define the runtime-strategy parsing cleanup seam:
   - environment/config parsing candidates
   - stable parser ownership
   - behavior-preserving extraction rules
3. Audit the highest-volume test binaries at a planning level:
   - `tests/test_chol_csc.c`
   - `tests/test_svd.c`
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
4. Record likely helper/fixture seams in those tests without committing to full-file splits.
5. Write the combined separator/runtime/test design artifact.

### Deliverables
- Separator-lifting boundary design
- Runtime parsing cleanup notes
- Large-test target shortlist

### Completion Criteria
- Separator extraction and runtime cleanup seams are explicit before code motion
- The large-test batch is bounded to helper/fixture consolidation rather than file splitting
- Day 11 and Day 12 targets are named early

---

## Day 5: FM Refinement Extraction Batch I

**Title:** FM Batch I  
**Theme:** Move the FM refinement core and its direct support logic into a dedicated module  
**Time estimate:** 12 hours

### Tasks
1. Extract the first bounded FM refinement slice into a dedicated implementation unit.
2. Move only the helpers and state that are clearly FM-owned from the Day 3 design.
3. Update shared declarations and include wiring as needed.
4. Keep the batch narrow:
   - no separator-lifting extraction yet
   - no runtime-strategy cleanup yet
   - no algorithmic behavior change
5. Run the required code-quality gate and graph-focused checks justified by the touched seam.

### Deliverables
- First extracted FM refinement module
- Updated shared declarations and build wiring
- Validation result for the first FM batch

### Completion Criteria
- A real FM subsystem file exists outside `src/sparse_graph.c`
- The moved FM logic no longer lives only in the residual monolith
- The required validation passes

---

## Day 6: Separator-Lifting Extraction Batch I

**Title:** Separator Batch I  
**Theme:** Move separator lifting and direct edge-to-vertex conversion support into a dedicated module  
**Time estimate:** 12 hours

### Tasks
1. Extract the first bounded separator-lifting slice into a dedicated implementation unit.
2. Move only the helpers and policy-selection logic clearly owned by separator lifting.
3. Preserve current orchestration flow and behavior while reducing residual monolith concentration.
4. Update shared declarations and build wiring as needed.
5. Run the required code-quality gate and graph-focused checks justified by the touched seam.

### Deliverables
- First extracted separator-lifting module
- Updated declarations and build wiring
- Validation result for the first separator batch

### Completion Criteria
- A real separator subsystem file exists outside `src/sparse_graph.c`
- Separator conversion logic is no longer only monolithic
- The required validation passes

---

## Day 7: Runtime Strategy Parsing Audit

**Title:** Runtime Audit  
**Theme:** Audit the remaining runtime/config parsing and orchestration coupling after the first FM and separator extractions  
**Time estimate:** 10 hours

### Tasks
1. Review the post-Day-6 graph state and identify remaining runtime/config parsing mixed into hot paths.
2. Separate:
   - parser logic ready for direct extraction or consolidation
   - parser logic still coupled to orchestration
   - retry/fallback logic that should stay until final cleanup
3. Confirm the bounded Day 8 target set for runtime cleanup and orchestration simplification.
4. Record any internal-header cleanup needed before the final graph pass.
5. Write the runtime-strategy parsing audit artifact.

### Deliverables
- Runtime parsing and orchestration seam map
- Day 8 cleanup target list
- Internal-header cleanup notes

### Completion Criteria
- The remaining graph cleanup queue is concrete rather than generic
- Parsing ownership is separated from orchestration where possible
- Final graph cleanup targets are explicit before the next batch

---

## Day 8: Runtime Parsing and Orchestration Cleanup

**Title:** Orchestration Cleanup  
**Theme:** Simplify the remaining graph orchestration path after the Phase-2 extractions land  
**Time estimate:** 12 hours

### Tasks
1. Extract or consolidate the bounded runtime/config parsing helpers identified on Day 7.
2. Simplify the top-level partition orchestration path after FM and separator ownership move out.
3. Keep behavior unchanged while making ownership clearer:
   - retry policy
   - override/config plumbing
   - phase handoff glue
4. Update internal declarations, comments, and build/include wiring as needed.
5. Run the required code-quality gate and graph-focused checks justified by the touched paths.

### Deliverables
- Cleaner residual graph orchestration layer
- Consolidated runtime/config parsing ownership
- Validation result for the cleanup batch

### Completion Criteria
- The residual `src/sparse_graph.c` reads as orchestration rather than a mixed monolith
- Runtime parsing is less entangled with algorithm hot paths
- The required validation passes

---

## Day 9: Graph Residual Audit and Focused Test Design

**Title:** Graph Residual Audit  
**Theme:** Audit the post-cleanup graph state and define the focused subsystem tests needed to protect the new boundaries  
**Time estimate:** 10 hours

### Tasks
1. Review the post-Day-8 graph subsystem and confirm the live ownership split.
2. Identify the highest-value targeted graph regressions needed to protect:
   - FM extraction seams
   - separator-lifting seams
   - runtime/fallback behavior
   - residual orchestration contracts
3. Separate stable new tests from tests that would overfit to private implementation details.
4. Define the bounded Day 10 graph-test batch.
5. Write the residual graph audit and test-design artifact.

### Deliverables
- Post-cleanup graph ownership audit
- Focused graph seam test plan
- Day 10 targeted test list

### Completion Criteria
- The new graph module boundaries are reviewed before the sprint shifts toward test maintainability
- The focused graph test batch is concrete and bounded
- The sprint stays behavior-oriented rather than implementation-detail heavy

---

## Day 10: Focused Graph Seam Tests

**Title:** Graph Seam Tests  
**Theme:** Add the targeted graph regressions needed to protect the Sprint 44 extraction boundaries  
**Time estimate:** 10 hours

### Tasks
1. Implement the bounded graph-focused tests selected on Day 9.
2. Prefer behavior-level coverage that locks in:
   - FM path dispatch / refinement outcomes
   - separator lifting contracts
   - runtime override / fallback behavior
   - residual orchestration expectations
3. Keep tests aligned with the one-binary-per-test model already in place.
4. Add only load-bearing comments where a test protects a subtle refactor seam.
5. Run the required code-quality gate and targeted graph test reruns justified by the touched paths.

### Deliverables
- Focused graph seam regression tests
- Targeted graph rerun results
- Validation result for the graph-test batch

### Completion Criteria
- The Phase-2 graph extraction boundaries have explicit regression protection
- Tests stay readable and behavior-oriented
- The required validation passes

---

## Day 11: Large-Test Helper Audit

**Title:** Large-Test Audit  
**Theme:** Audit the largest test binaries for true helper/fixture extraction opportunities  
**Time estimate:** 12 hours

### Tasks
1. Audit the largest test binaries in depth:
   - `tests/test_chol_csc.c`
   - `tests/test_svd.c`
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
2. Classify real maintainability seams:
   - repeated fixture builders
   - repeated residual/assertion helpers
   - repeated matrix setup / teardown helpers
   - repeated data-table or scenario setup
3. Separate:
   - good helper-extraction candidates
   - cases better left local
   - cases that would need later structural work outside Sprint 44
4. Choose the bounded Day 12 target set.
5. Write the large-test helper audit artifact.

### Deliverables
- Large-test maintainability seam inventory
- Good helper-extraction candidate list
- Bounded Day 12 consolidation target set

### Completion Criteria
- The large-test batch is grounded in real duplication, not generic file-size discomfort
- Helper candidates are separated from later structural refactors
- Day 12 has a concrete, bounded landing set

---

## Day 12: First Test-Helper Consolidation Batch

**Title:** Test Helper Batch  
**Theme:** Extract the first shared helpers/fixtures from the largest tests while preserving the existing test-binary model  
**Time estimate:** 10 hours

### Tasks
1. Implement the bounded helper/fixture consolidation chosen on Day 11.
2. Prefer small, high-signal extractions:
   - repeated fixture builders
   - repeated assertion helpers
   - repeated setup/teardown helpers
3. Keep the batch bounded:
   - no large file splits
   - no broad test-framework redesign
   - no behavior changes
4. Update test includes/declarations as needed while preserving readability.
5. Run the required code-quality gate and targeted reruns for the touched large-test binaries.

### Deliverables
- First large-test helper/fixture consolidation batch
- Updated touched test binaries
- Validation result for the maintainability batch

### Completion Criteria
- At least one real maintainability seam is removed from the largest tests
- The one-binary-per-test model remains clear
- The required validation passes

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Re-run the full maintained gate and targeted subsystem checks across the Sprint 44 change surface  
**Time estimate:** 10 hours

### Tasks
1. Run the full required validation stack:
   - `make format`
   - `make lint`
   - `make test`
2. Run the stronger local reviewed baseline:
   - `make quality-review-full`
3. Run targeted follow-on reruns justified by Sprint 44’s touched surfaces:
   - graph / ND focused binaries
   - any large test binaries touched by Day 12
4. Record exact results, timings, and any required reruns.
5. Write the full validation sweep artifact.

### Deliverables
- Full validation sweep report
- Targeted graph and large-test rerun results
- Final pre-closeout issue list, if any

### Completion Criteria
- The required gate passes completely
- The stronger reviewed baseline also passes
- Sprint 44 ends from a measured validated baseline rather than assumption

---

## Day 14: Closeout and Handoff

**Title:** Closeout  
**Theme:** Synthesize Sprint 44 outcomes, residual deferred work, and the handoff into later Epic 4 graph and maintainability phases  
**Time estimate:** 10 hours

### Tasks
1. Summarize Sprint 44 outcomes across:
   - FM extraction
   - separator extraction
   - runtime/orchestration cleanup
   - graph seam tests
   - large-test helper consolidation
2. Record the validated end-state and preserved constraints.
3. Identify the residual deferred queue for later Epic 4 work:
   - any deeper graph-phase cleanup still intentionally left
   - any remaining large-test structural refactors not suitable for Sprint 44
4. Update working notes and write the closeout/handoff artifact.
5. Confirm whether `PROJECT_PLAN.md` needs a follow-up note for any newly surfaced deferred work.

### Deliverables
- Sprint 44 closeout and handoff artifact
- Updated working-notes closeout summary
- Explicit residual deferred-work list

### Completion Criteria
- Sprint 44 closes from the Day 13 validated baseline
- The residual queue is explicit instead of implied
- Later Epic 4 sprints inherit a clear graph/test-maintainability handoff
