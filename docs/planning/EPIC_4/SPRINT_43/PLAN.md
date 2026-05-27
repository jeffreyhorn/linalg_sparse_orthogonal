# Sprint 43 Plan: Graph / ND Subsystem Decomposition Phase 1

**Sprint Duration:** 14 days  
**Goal:** Break `src/sparse_graph.c` into explicit subsystem slices so graph partitioning, hierarchy building, coarsening, coarse bisection, and supporting runtime strategy logic can evolve without one monolithic implementation unit carrying every heuristic and mode switch. This sprint implements the Sprint 43 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`.

**Starting Point:** Sprint 42 closed with a validated internal lifecycle-groundwork package, shared matrix-state guard helpers, compatibility-preserving factor-path normalization, and the preserved Sprint 40 architecture contract. Sprint 43 begins from that baseline plus Sprint 40's hotspot inventory, which already identified `src/sparse_graph.c` as the largest and most concentrated implementation hotspot in the repo.

**End State:** Sprint 43 leaves behind the first real decomposition of the graph / ND subsystem, with explicit module boundaries for graph ownership, hierarchy/coarsening, and coarse bisection, updated build wiring, focused subsystem tests, and a validated baseline that makes later FM-refinement and separator-lifting extractions safer.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 152 hours, matching the Sprint 43 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 43 Scope Audit & Graph Baseline

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 43 project-plan items into a bounded graph-subsystem decomposition inventory  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 43 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`, the Sprint 40 hotspot inventory, and the Sprint 41/42 prep-rule and validation-anchor artifacts.
2. Reconfirm the preserved constraints Sprint 43 must not reopen:
   - internal-first implementation
   - public API stability
   - Sprint 40 validation anchor
   - Sprint 41 shared safety helper reuse where relevant
   - Sprint 42 compatibility-preserving refactor style
3. Define the Sprint 43 workstreams explicitly:
   - graph-module boundary design
   - graph ownership / construction extraction
   - hierarchy / coarsening extraction
   - coarse-bisection extraction
   - build/include cleanup
   - focused graph tests
   - validation closeout
4. Record the highest-risk seams already implied by the monolith:
   - graph construction / teardown ownership
   - multilevel hierarchy building
   - heavy-edge matching and coarsening
   - coarse-level bisection dispatch
   - runtime strategy parsing and mode selection
5. Open Sprint 43 working notes and record scope, assumptions, and initial migration order.

### Deliverables
- Sprint 43 scope inventory
- Graph-subsystem workstream map
- Working-notes baseline assumptions

### Completion Criteria
- Sprint 43 starts from the documented Epic 4 baseline rather than ad hoc code movement
- The preserved refactor constraints are explicit before file splitting begins
- The graph decomposition targets are named before implementation starts

---

## Day 2: Monolith Seam Inventory Refresh

**Title:** Seam Inventory  
**Theme:** Re-map the internal seams inside `src/sparse_graph.c` before choosing extraction boundaries  
**Time estimate:** 10 hours

### Tasks
1. Refresh the live seam inventory for `src/sparse_graph.c` and classify the main implementation regions:
   - graph construction / ownership
   - hierarchy and coarse-graph lifecycle
   - matching / coarsening
   - coarse bisection
   - FM refinement
   - separator lifting / final partition projection
   - runtime strategy parsing and option dispatch
2. Mark cross-cutting state that currently travels through too many phases.
3. Record which helper clusters are safe extraction candidates now and which should stay local in Sprint 43 Phase 1.
4. Distinguish “stable module seam” candidates from “still too entangled” regions.
5. Write the seam-refresh inventory artifact.

### Deliverables
- Refreshed `sparse_graph.c` seam inventory
- Stable-seam vs entangled-region classification
- Initial extraction order notes

### Completion Criteria
- The monolith is reduced to named subsystem seams rather than treated as one opaque file
- Phase-1 extraction candidates are separated from later FM/separator work
- The later implementation order is grounded in current code structure

---

## Day 3: Graph Module Boundary Design

**Title:** Boundary Design  
**Theme:** Define the target Phase-1 file split and interface ownership before moving code  
**Time estimate:** 10 hours

### Tasks
1. Design the target Phase-1 file layout for:
   - graph ownership / construction
   - hierarchy / coarsening
   - coarse bisection
   - shared internal graph types / declarations
2. Decide what remains in `src/sparse_graph.c` after Phase 1:
   - FM refinement
   - separator lifting
   - top-level orchestration
   - runtime strategy glue that still spans multiple phases
3. Define internal header boundaries, ownership rules, and naming expectations.
4. Record stable interface expectations for the extracted modules.
5. Write the module-boundary design artifact.

### Deliverables
- Phase-1 file-layout design
- Internal header / ownership map
- Keep-in-monolith vs extract-now notes

### Completion Criteria
- The Phase-1 decomposition target is explicit before code edits begin
- Internal interface boundaries are documented
- The sprint stays bounded away from full graph-subsystem completion

---

## Day 4: Build and Include Strategy Design

**Title:** Wiring Design  
**Theme:** Design the include and build-system adjustments needed for a multi-file graph subsystem  
**Time estimate:** 10 hours

### Tasks
1. Define how `Makefile` and `CMakeLists.txt` should absorb new graph implementation units.
2. Decide which declarations belong in a shared internal graph header versus local translation-unit scope.
3. Identify include-order and dependency risks:
   - graph internals versus reorder front-ends
   - graph buckets / helper structs
   - test visibility and internal linkage expectations
4. Record naming and include hygiene rules for the extracted files.
5. Write the build/include strategy artifact.

### Deliverables
- Build and include wiring design
- Shared-header boundary notes
- Include-risk checklist

### Completion Criteria
- Build-system changes are planned before file extraction begins
- Shared declarations are separated from translation-unit-local helpers
- Include hygiene expectations are explicit

---

## Day 5: Graph Ownership / Construction Extraction Batch I

**Title:** Ownership Batch  
**Theme:** Move graph construction and ownership helpers into a dedicated module  
**Time estimate:** 12 hours

### Tasks
1. Extract the graph-construction and ownership helpers selected by Days 2-3 into a dedicated implementation unit.
2. Add or update the corresponding internal header declarations.
3. Keep the batch narrow:
   - no FM refinement movement
   - no separator-lifting movement
   - no algorithmic behavior changes
4. Update the build wiring for the new module.
5. Run the required code-quality gate and any targeted graph tests justified by the touched seam.

### Deliverables
- First extracted graph ownership / construction module
- Updated internal declarations and build wiring
- Validation result for the first extraction batch

### Completion Criteria
- A real subsystem file exists outside `src/sparse_graph.c`
- Ownership and construction helpers no longer live only in the monolith
- The required validation passes

---

## Day 6: Hierarchy / Coarsening Extraction Batch I

**Title:** Coarsening Batch I  
**Theme:** Start moving hierarchy-building and heavy-edge matching logic behind a dedicated subsystem seam  
**Time estimate:** 12 hours

### Tasks
1. Extract the first bounded hierarchy / coarsening slice:
   - hierarchy-building helpers
   - coarse-graph ownership transitions
   - selected heavy-edge matching helpers
2. Preserve current state flow and behavior while reducing monolithic concentration.
3. Keep algorithm-specific comments concise and load-bearing.
4. Update shared internal declarations and build wiring as needed.
5. Run the required code-quality gate and graph-focused checks justified by the touched paths.

### Deliverables
- First hierarchy / coarsening extraction
- Updated graph internal declarations
- Validation result for the batch

### Completion Criteria
- The multilevel/coarsening seam exists in a real extracted module
- Behavior remains unchanged
- The required validation passes

---

## Day 7: Residual Coarsening / Hierarchy Audit

**Title:** Coarsening Audit  
**Theme:** Audit the remaining hierarchy/coarsening seam before the second extraction push  
**Time estimate:** 10 hours

### Tasks
1. Review the post-Day-6 state and identify what still belongs in the coarsening module versus what remains tied to later phases.
2. Separate:
   - ready-for-direct extraction helpers
   - helpers still coupled to FM refinement or separator lifting
   - runtime strategy glue better left for later
3. Confirm the bounded Day 8 target set.
4. Record any interface cleanup needed before coarse-bisection extraction begins.
5. Write the residual coarsening audit artifact.

### Deliverables
- Residual coarsening seam map
- Day 8 extraction target list
- Interface cleanup notes

### Completion Criteria
- The second coarsening batch has a concrete landing order
- Sprint 43 stays bounded away from FM/separator churn
- Interface cleanup needs are explicit before more extraction

---

## Day 8: Hierarchy / Coarsening Extraction Batch II

**Title:** Coarsening Batch II  
**Theme:** Complete the planned first-phase hierarchy/coarsening extraction  
**Time estimate:** 12 hours

### Tasks
1. Extract the remaining bounded hierarchy/coarsening helpers chosen on Day 7.
2. Consolidate any duplicated local declarations created by the earlier split.
3. Keep state handoff and ownership flow behavior-preserving.
4. Update build/include wiring if the second extraction changes dependencies.
5. Run the required code-quality gate and targeted graph checks justified by the touched paths.

### Deliverables
- Completed Phase-1 hierarchy/coarsening extraction
- Consolidated internal declarations
- Validation result for the second batch

### Completion Criteria
- The planned Phase-1 coarsening seam is no longer monolithic
- Shared declarations stay coherent after the second split
- The required validation passes

---

## Day 9: Coarse-Bisection Extraction Batch I

**Title:** Bisection Batch I  
**Theme:** Extract the coarse-level bisection logic from the remaining graph monolith  
**Time estimate:** 12 hours

### Tasks
1. Move the bounded coarse-bisection logic into its own implementation unit:
   - brute-force coarse search
   - spectral coarse split helpers
   - coarse-level dispatch helpers chosen in the design
2. Preserve runtime strategy behavior and current mode selection semantics.
3. Keep FM refinement, separator lifting, and top-level orchestration in place for now.
4. Update internal declarations and build wiring for the new module.
5. Run the required code-quality gate and targeted graph tests justified by the touched paths.

### Deliverables
- First coarse-bisection extraction
- Updated internal declarations and build wiring
- Validation result for the batch

### Completion Criteria
- Coarse-bisection logic is no longer only embedded in `src/sparse_graph.c`
- Runtime behavior is preserved
- The required validation passes

---

## Day 10: Runtime Strategy / Glue Reconciliation

**Title:** Glue Reconciliation  
**Theme:** Reconcile top-level graph orchestration after the Phase-1 extractions  
**Time estimate:** 10 hours

### Tasks
1. Clean up the remaining top-level orchestration in `src/sparse_graph.c` after the ownership, coarsening, and coarse-bisection splits.
2. Reduce internal include / declaration drift created by the new modules.
3. Keep runtime strategy parsing and mode selection behavior stable while clarifying which layer owns each decision.
4. Record any deliberate Phase-2 deferrals:
   - FM refinement
   - separator lifting
   - deeper runtime strategy cleanup
5. Run the required code-quality gate and targeted graph checks if the touched surface justifies it.

### Deliverables
- Reconciled top-level orchestration surface
- Reduced graph-internal glue drift
- Explicit Phase-2 deferral notes

### Completion Criteria
- The extracted modules integrate cleanly with the remaining monolith
- Top-level ownership is clearer than before the sprint
- Validation passes if code changed

---

## Day 11: Focused Graph Test Design

**Title:** Test Design  
**Theme:** Define the graph-focused seam tests needed to pin the new subsystem shape  
**Time estimate:** 10 hours

### Tasks
1. Audit existing graph / ND tests for current coverage of:
   - graph construction and ownership
   - hierarchy/coarsening behavior
   - coarse-bisection paths
2. Identify the highest-value seam tests still missing after the refactor.
3. Separate:
   - tests needed now for extracted-module safety
   - tests better deferred until FM or separator extraction
4. Define the bounded Day 12 implementation batch.
5. Write the focused graph-test design artifact.

### Deliverables
- Graph seam-test inventory
- Day 12 test implementation plan
- Now-vs-later graph test boundary notes

### Completion Criteria
- The new module seams have an explicit test plan
- Sprint 43 does not drift into an open-ended graph test rewrite
- The Day 12 test batch is concrete

---

## Day 12: Focused Graph Test Batch

**Title:** Test Batch  
**Theme:** Add or adapt tests that pin the extracted graph subsystem seams  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded graph-focused tests selected on Day 11.
2. Prefer seam-protection and behavior-preservation checks over broad new algorithm coverage.
3. Keep the batch focused on extracted ownership/coarsening/coarse-bisection boundaries.
4. Update any supporting test helpers only if the touched batch requires it.
5. Run the required code-quality gate and any direct graph-focused test reruns justified by the changes.

### Deliverables
- Focused graph subsystem seam tests
- Any minimal supporting test updates
- Validation result for the test batch

### Completion Criteria
- The new module boundaries are pinned by tests
- The batch stays bounded to Sprint 43 Phase-1 seams
- The required validation passes

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full Sprint 40/41/42 validation anchor against the decomposed graph subsystem  
**Time estimate:** 12 hours

### Tasks
1. Run the mandatory code-change floor:
   - `make format`
   - `make lint`
   - `make test`
2. Run the default strong proof for a substantial refactor:
   - `make quality-review-full`
3. Run additional graph-focused checks if the touched surface justifies them:
   - selected graph tests
   - any maintained graph-adjacent benchmark or smoke path needed for confidence
4. Reconfirm the Sprint 40 truthfulness anchors:
   - `ctest -N --test-dir build/quality-review-cmake` remains `53`
   - Makefile/CMake parity remains explicit
5. Record timings, outcomes, and any operational caveats in the validation artifact.

### Deliverables
- Full validation sweep record
- Updated graph-focused validation notes
- Final measured Sprint 43 baseline

### Completion Criteria
- Full validation passes
- The graph decomposition is proven against the maintained local reviewed baseline
- Any caveats are documented clearly

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Synthesize the Sprint 43 decomposition outcome and hand off the residual graph work cleanly  
**Time estimate:** 10 hours

### Tasks
1. Summarize what Phase 1 actually accomplished in the graph / ND subsystem:
   - extracted ownership / construction seam
   - extracted hierarchy/coarsening seam
   - extracted coarse-bisection seam
   - updated build/include shape
   - focused tests and validated baseline
2. Record what intentionally remains for later graph phases:
   - FM refinement extraction
   - separator lifting extraction
   - deeper runtime strategy simplification
3. Confirm whether `PROJECT_PLAN.md` needs any adjustment based on real Sprint 43 outcomes.
4. Update working notes with final outcomes, residual risks, and next-sprint prerequisites.
5. Write the closeout and handoff artifact.

### Deliverables
- Sprint 43 closeout artifact
- Residual graph-phase handoff notes
- Final working-notes synthesis

### Completion Criteria
- Sprint 43 ends with one coherent decomposition handoff rather than scattered implementation notes
- Residual graph work is routed cleanly to later phases
- Any needed planning updates are recorded explicitly
