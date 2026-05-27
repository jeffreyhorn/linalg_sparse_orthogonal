# Sprint 45 Plan: Iterative Solver Workspace Reuse & Repeated-Solve Efficiency

**Sprint Duration:** 14 days  
**Goal:** Add reusable workspace support for iterative solvers so repeated solve workloads can avoid repeated heap allocation while preserving the current simple public APIs. This sprint implements the Sprint 45 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`.

**Starting Point:** Sprint 44 closed with the Epic 4 structural groundwork already in place: Sprint 41 landed shared allocation helpers in `src/sparse_alloc_internal.{h,c}`, Sprint 42 established internal lifecycle/compatibility scaffolding, and the current iterative solver implementation remains concentrated in `src/sparse_iterative.c` with one-shot allocation bundles across CG, GMRES, matrix-free CG, block CG, and MINRES paths. Sprint 45 begins from that validated baseline and turns the iterative solver surface into a repeated-solve efficiency target without reopening public API shape changes.

**End State:** Sprint 45 leaves behind a reusable internal workspace model for the main iterative solver families, compatibility-preserving one-shot wrappers, repeated-solve benchmark evidence for reduced allocator churn, and a validation record showing the stronger iterative implementation still satisfies the maintained local reviewed baseline.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 140 hours, matching the Sprint 45 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 45 Scope Audit & Baseline Refresh

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 45 project-plan items into a bounded iterative-workspace execution map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 45 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`, the Sprint 40 validation anchor, the Sprint 41 prep-rule artifact, and the Sprint 42 compatibility notes.
2. Reconfirm the preserved constraints Sprint 45 must not reopen:
   - no public API churn first
   - keep one-shot iterative entry points
   - preserve Sprint 40 validation truthfulness
   - reuse Sprint 41 allocation helpers instead of adding a second helper layer
   - respect Sprint 42 lifecycle/compatibility boundaries where factors or preconditioners are involved
3. Define the Sprint 45 workstreams explicitly:
   - iterative workspace seam inventory
   - reusable workspace API design
   - shared workspace-backed internal helper layer
   - CG / GMRES migration
   - block iterative migration
   - wrapper preservation
   - repeated-solve benchmark batch
   - validation closeout
4. Record the highest-risk remaining iterative seams:
   - packed workspace slicing in CG and GMRES
   - block and multi-RHS workspace sizing
   - matrix-free and preconditioned repeated-solve paths
   - ownership/reset semantics for reusable buffers
5. Open Sprint 45 working notes and record scope, assumptions, and initial landing order.

### Deliverables
- Sprint 45 scope inventory
- Iterative-workspace workstream map
- Working-notes baseline assumptions

### Completion Criteria
- Sprint 45 starts from the documented Epic 4 baseline rather than ad hoc iterative changes
- Preserved compatibility and validation constraints are explicit before implementation begins
- The iterative workspace and repeated-solve targets are named before code changes start

---

## Day 2: Iterative Workspace Seam Inventory Refresh

**Title:** Workspace Inventory  
**Theme:** Re-map the iterative solver allocation and reuse seams before choosing the workspace landing order  
**Time estimate:** 10 hours

### Tasks
1. Refresh the live seam inventory for `src/sparse_iterative.c` and classify the main allocation/workspace regions:
   - scalar CG
   - matrix-free CG
   - GMRES
   - block CG / multi-RHS support
   - MINRES and smaller shared support paths
2. Separate:
   - shared work-buffer patterns that are clear extraction targets
   - solver-specific state that should stay solver-local
   - optional/preconditioner-dependent buffers
   - one-shot wrapper logic versus reusable-core logic
3. Record the strongest common allocation shapes:
   - graph-sized vector bundles
   - Hessenberg / Arnoldi scratch
   - block `(n * nrhs)` bundles
   - stagnation / history tracking state
4. Identify which paths should be first-phase workspace adopters versus later follow-ons.
5. Write the iterative workspace seam inventory artifact.

### Deliverables
- Refreshed iterative allocation/workspace seam inventory
- Shared-vs-solver-local classification
- First migration-order notes

### Completion Criteria
- The iterative solver surface is reduced to named workspace seams
- Shared allocation patterns are distinguished from solver-specific logic
- Later implementation order is grounded in the live iterative file

---

## Day 3: Reusable Workspace API Design

**Title:** Workspace API Design  
**Theme:** Define the internal reusable workspace objects, ownership, sizing, and reset rules  
**Time estimate:** 10 hours

### Tasks
1. Design the internal workspace object model for the first Sprint 45 landing:
   - shared iterative buffer object(s)
   - CG-specific view/state
   - GMRES-specific view/state
   - block-path view/state
2. Define ownership and lifecycle rules:
   - create / destroy
   - reset between solves
   - resize or reject on mismatched dimensions
   - rules for optional buffers
3. Decide what remains internal-only in Sprint 45 versus what must be exposed to wrappers.
4. Record expected interaction rules with preconditioners, matrix-free operators, and repeated solves on stable dimensions.
5. Write the workspace API design artifact.

### Deliverables
- Internal iterative workspace design
- Ownership and reset contract
- Internal-only vs wrapper-facing boundary notes

### Completion Criteria
- Sprint 45 has a concrete internal workspace model before code edits
- Reset/reuse rules are explicit instead of implicit
- The design stays bounded away from public API redesign

---

## Day 4: Shared Buffer Layer Design & Validation Plan

**Title:** Shared Buffer Design  
**Theme:** Bound the common workspace-backed helper layer and the first required validation shape  
**Time estimate:** 10 hours

### Tasks
1. Design the shared internal helper layer for common iterative allocation patterns:
   - checked sizing
   - one-allocation packed buffers
   - typed slicing into solver views
   - reset / zeroing expectations
2. Decide which helpers belong in shared iterative internals versus solver-local code.
3. Define the initial validation shape for the implementation days:
   - full required gate for all `*.c` / `*.h` changes
   - targeted touched-binary reruns for iterative solvers
   - repeated-solve benchmark checks once the workspace path exists
4. Confirm the first landing order:
   - shared layer first
   - CG / GMRES second
   - block paths next
   - wrappers and benchmarks after the internal paths are stable
5. Write the shared-buffer design artifact.

### Deliverables
- Shared iterative buffer-layer design
- Shared-vs-local helper ownership map
- Implementation-day validation plan

### Completion Criteria
- The common helper layer is explicit before implementation begins
- The first migration order is fixed up front
- The sprint has a clear validation contract for the workspace rollout

---

## Day 5: Shared Iterative Buffer Layer Batch I

**Title:** Shared Buffer Batch I  
**Theme:** Land the reusable shared iterative buffer infrastructure that later solver migrations will consume  
**Time estimate:** 12 hours

### Tasks
1. Add the first bounded shared iterative workspace-backed internal layer using the Day 3 / Day 4 design.
2. Move only clearly shared sizing, allocation, and packed-buffer slicing logic into the new seam.
3. Keep the batch narrow:
   - no public API changes
   - no broad solver migration yet
   - no benchmark work yet
4. Update build/include wiring and internal declarations as needed.
5. Run the required code-quality gate and targeted iterative checks justified by the touched seam.

### Deliverables
- First reusable internal iterative workspace layer
- Updated declarations/build wiring
- Validation result for the first shared-buffer batch

### Completion Criteria
- A real shared iterative workspace seam exists outside the one-shot solver bodies
- Shared allocation logic is no longer only duplicated inside solver entry points
- The required validation passes

---

## Day 6: CG / GMRES Migration Batch I

**Title:** CG/GMRES Batch I  
**Theme:** Convert the primary scalar iterative paths to the new reusable internal workspace-backed model  
**Time estimate:** 12 hours

### Tasks
1. Migrate the main CG path to the shared reusable internal workspace seam.
2. Migrate the main GMRES path to the shared reusable internal workspace seam.
3. Preserve current one-shot public behavior while routing internal work through reusable buffers.
4. Keep the batch bounded:
   - do not broaden to block paths yet
   - do not add new public APIs yet
5. Run the required code-quality gate and targeted iterative tests justified by the touched solver paths.

### Deliverables
- Workspace-backed CG internals
- Workspace-backed GMRES internals
- Validation result for the first primary-solver migration batch

### Completion Criteria
- The main repeated-solve iterative paths no longer depend only on per-call heap bundles
- One-shot user-facing behavior remains intact
- The required validation passes

---

## Day 7: Primary Workspace Landing Audit

**Title:** Primary Path Audit  
**Theme:** Audit the post-Day-6 state to confirm what remains for block paths, wrappers, and repeated-solve measurement  
**Time estimate:** 10 hours

### Tasks
1. Review the post-Day-6 iterative state and identify remaining allocation churn in:
   - block CG / multi-RHS paths
   - matrix-free variants
   - smaller support paths
2. Separate:
   - clear block-workspace migration candidates
   - wrappers that already naturally compose over the new seam
   - remaining solver-local state that should stay local in Sprint 45
3. Confirm the bounded Day 8 target set for block iterative migration.
4. Record any internal-header cleanup needed before wrapper/benchmark work.
5. Write the primary-workspace landing audit artifact.

### Deliverables
- Post-primary-path audit
- Bounded block-path migration target list
- Wrapper/benchmark follow-on notes

### Completion Criteria
- The remaining iterative queue is concrete rather than generic
- Block-path targets are explicit before the next batch
- Wrapper and benchmark work are sequenced from the live code state

---

## Day 8: Block Iterative Migration Batch

**Title:** Block Path Batch  
**Theme:** Extend the reusable workspace model to the multi-RHS and block iterative paths  
**Time estimate:** 12 hours

### Tasks
1. Migrate the bounded block / multi-RHS iterative slice identified on Day 7.
2. Reuse the shared workspace model rather than adding a separate block-only allocation framework.
3. Preserve current one-shot behavior and algorithm choices while reducing repeated allocation churn.
4. Update internal declarations and helper ownership as needed.
5. Run the required code-quality gate and targeted touched-solver checks.

### Deliverables
- Workspace-backed block iterative internals
- Updated internal workspace declarations
- Validation result for the block migration batch

### Completion Criteria
- The multi-RHS/block path participates in the reusable workspace model
- The sprint still uses one coherent workspace design instead of divergent submodels
- The required validation passes

---

## Day 9: Compatibility Wrapper Landing

**Title:** Wrapper Compatibility  
**Theme:** Make the current one-shot public iterative APIs explicit convenience wrappers over the reusable internals  
**Time estimate:** 10 hours

### Tasks
1. Normalize the one-shot iterative entry points so they clearly delegate through the new internal workspace-capable model.
2. Keep wrapper behavior and signatures unchanged.
3. Tighten any remaining cleanup or reset rules needed for wrapper-owned temporary workspaces.
4. Record any remaining documentation or benchmark assumptions required by the compatibility layer.
5. Run the required code-quality gate and targeted iterative checks justified by the touched wrapper paths.

### Deliverables
- Compatibility-preserving one-shot iterative wrappers
- Cleanup/reset alignment for wrapper-owned temporary workspaces
- Validation result for the wrapper batch

### Completion Criteria
- The current public iterative APIs remain available and behaviorally stable
- The wrapper relationship to the reusable internals is explicit in code
- The required validation passes

---

## Day 10: Repeated-Solve Benchmark Design & Audit

**Title:** Benchmark Design  
**Theme:** Define the smallest useful repeated-solve benchmark slice that can show allocation-churn reduction  
**Time estimate:** 8 hours

### Tasks
1. Audit the current benchmark surface for iterative repeated-solve suitability.
2. Choose a bounded repeated-solve benchmark set:
   - scalar iterative repeat case
   - GMRES repeat case
   - optional block repeat case only if it stays small
3. Define the benchmark comparison model:
   - one-shot repeated calls
   - reusable workspace-backed repeated calls
   - timing and allocation-churn reporting notes
4. Record guardrails:
   - no broad benchmark framework rewrite
   - no unstable system-dependent claims
5. Write the repeated-solve benchmark design artifact.

### Deliverables
- Repeated-solve benchmark target set
- Comparison methodology notes
- Bounded Day 11 benchmark implementation plan

### Completion Criteria
- Sprint 45 has a concrete measurement slice before benchmark code changes
- The benchmark scope stays narrow and comparable
- Allocation-churn evidence targets are explicit

---

## Day 11: Repeated-Solve Benchmark Batch

**Title:** Benchmark Batch  
**Theme:** Add or update the bounded repeated-solve benchmarks for the new reusable iterative workspace paths  
**Time estimate:** 10 hours

### Tasks
1. Implement the repeated-solve benchmark slice designed on Day 10.
2. Keep the batch narrow and directly tied to the migrated iterative paths.
3. Record benchmark outputs or measurement notes in the sprint artifacts.
4. Avoid broader benchmark harness churn beyond what the repeated-solve comparison needs.
5. Run the required code-quality gate if production/test `*.c` / `*.h` files change, plus the benchmark-focused follow-on runs justified by the touched surface.

### Deliverables
- Repeated-solve benchmark updates
- First measured allocation/runtime comparison notes
- Validation result for the benchmark batch

### Completion Criteria
- Sprint 45 has direct repeated-solve measurement evidence
- The benchmark work remains bounded to iterative workspace reuse
- Required validation and targeted benchmark runs pass

---

## Day 12: Documentation, Prep Rules, and Residual Audit

**Title:** Docs and Residual Audit  
**Theme:** Capture the new internal workspace contract and audit any remaining iterative repeated-allocation seams  
**Time estimate:** 8 hours

### Tasks
1. Document the internal workspace contract and maintainer-facing expectations for repeated-solve iterative work.
2. Audit the residual iterative surface for any obvious still-unmigrated allocation seams worth routing to later sprints.
3. Record what Sprint 45 intentionally does not solve:
   - eigensolver workspace reuse
   - public explicit workspace APIs
   - broader documentation/public tutorial refresh
4. Write the docs/residual audit artifact.
5. Update working notes with the intended Day 13 validation sweep shape.

### Deliverables
- Maintainer-facing workspace contract notes
- Residual iterative repeated-allocation audit
- Day 13 validation plan notes

### Completion Criteria
- The new internal workspace model is documented for later Epic 4 work
- Residual iterative seams are explicitly classified rather than left implicit
- The sprint’s non-goals remain clear before closeout

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the authoritative full validation pass for the Sprint 45 iterative workspace and repeated-solve changes  
**Time estimate:** 10 hours

### Tasks
1. Run the full required code-quality gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the stronger local reviewed baseline:
   - `make quality-review-full`
3. Run the targeted iterative and benchmark follow-on checks justified by the touched surfaces.
4. Record measured validation results, touched-binary outcomes, and any benchmark notes in the sprint artifact set.
5. Confirm that the maintained truthfulness anchors remain exact, including reviewed CMake parity.

### Deliverables
- Full validation sweep artifact
- Recorded iterative/benchmark follow-on results
- Confirmed maintained baseline metrics

### Completion Criteria
- All required quality gates pass
- The stronger reviewed local baseline passes
- The iterative workspace sprint closes from a measured, documented validation state

---

## Day 14: Closeout and Handoff

**Title:** Closeout and Handoff  
**Theme:** Consolidate the Sprint 45 reusable-workspace outcome and hand off the next repeated-run efficiency work cleanly  
**Time estimate:** 8 hours

### Tasks
1. Summarize the final Sprint 45 shipped state:
   - shared iterative workspace layer
   - migrated primary and block paths
   - compatibility-preserving one-shot wrappers
   - repeated-solve benchmark evidence
   - Day 13 validated baseline
2. Record what later sprints inherit next, especially the eigensolver repeated-run efficiency queue.
3. Check whether Sprint 45 surfaced any `PROJECT_PLAN.md` adjustments that must be recorded immediately.
4. Write the closeout/handoff artifact and update working notes.
5. Confirm the branch is clean and the sprint can be handed off without an unresolved queue inside Sprint 45 itself.

### Deliverables
- Sprint 45 closeout and handoff artifact
- Final working-notes synthesis
- Explicit next-sprint handoff notes

### Completion Criteria
- Sprint 45 ends with one coherent iterative-workspace package rather than disconnected edits
- Later repeated-run work is handed forward explicitly
- The sprint closes from the Day 13 validated baseline with no hidden queue
