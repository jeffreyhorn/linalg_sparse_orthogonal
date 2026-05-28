# Sprint 46 Plan: Eigensolver Workspace Reuse & Advanced Repeated-Run Efficiency

**Sprint Duration:** 14 days  
**Goal:** Add reusable workspace/state support for eigensolvers so repeated Lanczos, thick-restart Lanczos, and LOBPCG workloads can avoid repeated large-buffer allocation while preserving the current simple public APIs. This sprint implements the Sprint 46 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`.

**Starting Point:** Sprint 45 closed with the reusable-workspace model already established for iterative solvers, including shared allocation helpers from Sprint 41 and a compatibility-preserving internal-first rollout pattern from Sprint 42. The current eigensolver implementation remains concentrated in `src/sparse_eigs.c`, with repeated large-buffer allocation across grow-m Lanczos, thick-restart Lanczos, and LOBPCG paths. Sprint 46 begins from that validated baseline and turns the eigensolver surface into a repeated-run efficiency target without reopening public API shape changes.

**End State:** Sprint 46 leaves behind a reusable internal workspace/state model for the main eigensolver families, compatibility-preserving one-shot wrappers, repeated-run benchmark evidence for reduced allocation churn, maintainer-facing memory-behavior guidance, and a validation record showing the stronger eigensolver implementation still satisfies the maintained local reviewed baseline.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 148 hours, matching the Sprint 46 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 46 Scope Audit & Baseline Refresh

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 46 project-plan items into a bounded eigensolver-workspace execution map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 46 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`, the Sprint 40 validation anchor, the Sprint 41 shared-helper notes, the Sprint 42 compatibility rules, and the Sprint 45 iterative-workspace closeout.
2. Reconfirm the preserved constraints Sprint 46 must not reopen:
   - no public API churn first
   - keep current one-shot eigensolver entry points
   - preserve Sprint 40 validation truthfulness
   - reuse Sprint 41 / Sprint 45 helper patterns instead of adding a second allocation framework
   - keep the work internal-first even where multiple eigensolver backends exist
3. Define the Sprint 46 workstreams explicitly:
   - eigensolver seam inventory
   - reusable workspace/state design
   - grow-m / thick-restart migration
   - LOBPCG migration
   - wrapper preservation
   - repeated-run benchmark batch
   - memory-behavior documentation
   - validation closeout
4. Record the highest-risk remaining eigensolver seams:
   - Lanczos basis / tridiagonal scratch
   - thick-restart state carryover buffers
   - LOBPCG block and dense-intermediate storage
   - ownership/reset semantics for reusable eigensolver buffers
5. Open Sprint 46 working notes and record scope, assumptions, and initial landing order.

### Deliverables
- Sprint 46 scope inventory
- Eigensolver-workspace workstream map
- Working-notes baseline assumptions

### Completion Criteria
- Sprint 46 starts from the documented Epic 4 baseline rather than ad hoc eigensolver changes
- Preserved compatibility and validation constraints are explicit before implementation begins
- The eigensolver workspace and repeated-run targets are named before code changes start

---

## Day 2: Eigensolver Workspace Seam Inventory Refresh

**Title:** Workspace Inventory  
**Theme:** Re-map the eigensolver allocation and reuse seams before choosing the migration order  
**Time estimate:** 10 hours

### Tasks
1. Refresh the live seam inventory for `src/sparse_eigs.c` and classify the main allocation/workspace regions:
   - grow-m Lanczos
   - thick-restart Lanczos
   - LOBPCG
   - shared spectral helper/support paths
2. Separate:
   - shared work-buffer patterns that are clear extraction targets
   - solver/back-end-specific state that should stay local
   - optional/preconditioner-dependent buffers
   - one-shot wrapper logic versus reusable-core logic
3. Record the strongest common allocation shapes:
   - basis / vector bundles
   - tridiagonal / Ritz / restart scratch
   - block `(n * k)` bundles
   - dense subproblem and projected-operator intermediates
4. Identify which paths should be first-phase workspace adopters versus later follow-ons.
5. Write the eigensolver workspace seam inventory artifact.

### Deliverables
- Refreshed eigensolver allocation/workspace seam inventory
- Shared-vs-solver-local classification
- First migration-order notes

### Completion Criteria
- The eigensolver surface is reduced to named workspace seams
- Shared allocation patterns are distinguished from solver-specific logic
- Later implementation order is grounded in the live eigensolver file

---

## Day 3: Reusable Eigensolver Workspace/State API Design

**Title:** Workspace API Design  
**Theme:** Define the internal reusable eigensolver workspace/state objects, ownership, sizing, and reset rules  
**Time estimate:** 10 hours

### Tasks
1. Design the internal workspace/state object model for the first Sprint 46 landing:
   - shared eigensolver buffer owner
   - grow-m Lanczos view/state
   - thick-restart view/state
   - LOBPCG view/state
2. Define ownership and lifecycle rules:
   - create / destroy
   - reset between repeated runs
   - resize or reject on mismatched `(n, k, restart/block)` dimensions
   - rules for optional/preconditioner-dependent buffers
3. Decide what remains internal-only in Sprint 46 versus what must be visible to one-shot wrappers.
4. Record expected interaction rules with shift-invert, preconditioners, and repeated stable-dimension eigensolver workloads.
5. Write the reusable eigensolver workspace/state design artifact.

### Deliverables
- Internal eigensolver workspace/state design
- Ownership and reset contract
- Internal-only vs wrapper-facing boundary notes

### Completion Criteria
- Sprint 46 has a concrete internal workspace/state model before code edits
- Reset/reuse rules are explicit instead of implicit
- The design stays bounded away from public API redesign

---

## Day 4: Shared Buffer Layer & Validation Design

**Title:** Shared Buffer Design  
**Theme:** Bound the common eigensolver buffer-backed helper layer and the first required validation shape  
**Time estimate:** 10 hours

### Tasks
1. Design the shared internal helper layer for common eigensolver allocation patterns:
   - checked sizing
   - one-allocation packed buffers
   - typed slicing into eigensolver views
   - reset / zeroing expectations
2. Decide which helpers belong in shared eigensolver internals versus grow-m / thick-restart / LOBPCG-local code.
3. Define the initial validation shape for implementation days:
   - full required gate for all `*.c` / `*.h` changes
   - targeted touched-binary reruns for eigensolvers
   - repeated-run benchmark checks once the workspace path exists
4. Confirm the first landing order:
   - shared layer first
   - grow-m / thick-restart second
   - LOBPCG next
   - wrappers and benchmarks after the internal paths are stable
5. Write the shared-buffer design artifact.

### Deliverables
- Shared eigensolver buffer-layer design
- Shared-vs-local helper ownership map
- Implementation-day validation plan

### Completion Criteria
- The common helper layer is explicit before implementation begins
- The first migration order is fixed up front
- The sprint has a clear validation contract for the eigensolver workspace rollout

---

## Day 5: Shared Eigensolver Buffer Layer Batch I

**Title:** Shared Buffer Batch I  
**Theme:** Land the reusable shared eigensolver buffer/state infrastructure that later migrations will consume  
**Time estimate:** 12 hours

### Tasks
1. Add the first bounded shared eigensolver workspace/state-backed internal layer using the Day 3 / Day 4 design.
2. Move only clearly shared sizing, allocation, and packed-buffer slicing logic into the new seam.
3. Keep the batch narrow:
   - no public API changes
   - no broad algorithm migration yet
   - no benchmark work yet
4. Update build/include wiring and internal declarations as needed.
5. Run the required code-quality gate and targeted eigensolver checks justified by the touched seam.

### Deliverables
- First reusable internal eigensolver workspace/state layer
- Updated declarations/build wiring
- Validation result for the first shared-buffer batch

### Completion Criteria
- A real shared eigensolver workspace/state seam exists outside the one-shot solver bodies
- Shared allocation logic is no longer only duplicated inside eigensolver entry points
- The required validation passes

---

## Day 6: Grow-m / Thick-Restart Migration Batch I

**Title:** Lanczos Batch I  
**Theme:** Convert the main grow-m and thick-restart Lanczos paths to the new reusable internal workspace/state model  
**Time estimate:** 12 hours

### Tasks
1. Migrate the main grow-m Lanczos path to the shared reusable internal workspace/state seam.
2. Migrate the thick-restart path to the shared reusable internal workspace/state seam.
3. Preserve current one-shot public behavior while routing internal work through reusable buffers/state.
4. Keep the batch bounded:
   - do not broaden to LOBPCG yet
   - do not add new public APIs yet
5. Run the required code-quality gate and targeted eigensolver tests justified by the touched solver paths.

### Deliverables
- Workspace-backed grow-m Lanczos internals
- Workspace-backed thick-restart internals
- Validation result for the first primary eigensolver migration batch

### Completion Criteria
- The main repeated-run Lanczos paths no longer depend only on per-call heap bundles
- One-shot user-facing behavior remains intact
- The required validation passes

---

## Day 7: Primary Workspace Landing Audit

**Title:** Primary Path Audit  
**Theme:** Audit the post-Day-6 state to confirm what remains for LOBPCG, wrappers, and repeated-run measurement  
**Time estimate:** 10 hours

### Tasks
1. Review the post-Day-6 eigensolver state and identify remaining allocation churn in:
   - LOBPCG block/dense-intermediate paths
   - smaller shared support paths
   - one-shot wrapper edges
2. Separate:
   - clear LOBPCG workspace migration candidates
   - wrappers that already naturally compose over the new seam
   - remaining back-end-local state that should stay local in Sprint 46
3. Confirm the bounded Day 8 target set for LOBPCG migration.
4. Record any internal-header cleanup needed before wrapper/benchmark work.
5. Write the primary-workspace landing audit artifact.

### Deliverables
- Post-primary-path audit
- Bounded LOBPCG migration target list
- Wrapper/benchmark follow-on notes

### Completion Criteria
- The remaining eigensolver queue is concrete rather than generic
- LOBPCG targets are explicit before the next batch
- Wrapper and benchmark work are sequenced from the live code state

---

## Day 8: LOBPCG Workspace Migration Batch

**Title:** LOBPCG Batch  
**Theme:** Extend the reusable workspace/state model to the block eigensolver path and its dense intermediates  
**Time estimate:** 12 hours

### Tasks
1. Migrate the bounded LOBPCG slice identified on Day 7.
2. Reuse the shared workspace/state model rather than adding a separate block-only allocation framework.
3. Preserve current one-shot behavior and algorithm choices while reducing repeated allocation churn.
4. Update internal declarations and helper ownership as needed.
5. Run the required code-quality gate and targeted touched-eigensolver checks.

### Deliverables
- Workspace-backed LOBPCG internals
- Updated internal helper ownership
- Validation result for the LOBPCG migration batch

### Completion Criteria
- LOBPCG no longer depends only on per-call allocation bundles for its main repeated-run buffers
- The migration stays within the shared internal workspace/state model
- The required validation passes

---

## Day 9: Compatibility Wrapper Batch

**Title:** Wrapper Compatibility  
**Theme:** Normalize the one-shot public eigensolver entry points as compatibility wrappers over workspace-capable internals  
**Time estimate:** 10 hours

### Tasks
1. Review and normalize the current public eigensolver entries so the one-shot wrapper model is explicit.
2. Keep the batch bounded to compatibility/composition cleanup:
   - no new public explicit workspace API
   - no broad benchmark changes yet
3. Ensure the touched wrapper paths preserve current defaults, null handling, and result/reporting behavior.
4. Record any remaining private-header cleanup needed before benchmark work.
5. Run the required code-quality gate and targeted wrapper-focused eigensolver reruns.

### Deliverables
- Compatibility-preserving one-shot wrapper cleanup
- Wrapper-vs-internal ownership notes
- Validation result for the wrapper batch

### Completion Criteria
- The public one-shot eigensolver APIs clearly behave as wrappers over reusable internal paths
- No public API redesign is introduced
- The required validation passes

---

## Day 10: Repeated-Run Benchmark Design

**Title:** Benchmark Design  
**Theme:** Define the smallest honest repeated-run benchmark slice for the new eigensolver workspace/state model  
**Time estimate:** 10 hours

### Tasks
1. Audit the current eigensolver benchmark surface and identify the best A/B repeated-run comparison shape.
2. Select the narrow repeated-run target set:
   - grow-m Lanczos
   - thick-restart Lanczos
   - LOBPCG if the batch stays bounded
3. Decide on stable `(n, k, restart/block)` benchmark cases that are repeated-run meaningful but not over-claimed.
4. Record the measurement rules:
   - wall time / repeated-run comparison
   - behavior-level convergence parity
   - no universal speedup claims
5. Write the repeated-run benchmark design artifact.

### Deliverables
- Repeated-run eigensolver benchmark design
- Selected benchmark cases
- Measurement/claim-scope notes

### Completion Criteria
- Day 11 has a concrete, bounded repeated-run benchmark target
- The benchmark shape is tied directly to the migrated eigensolver paths
- The benchmark claim scope is explicit before implementation

---

## Day 11: Repeated-Run Benchmark Batch

**Title:** Benchmark Batch  
**Theme:** Add or update the bounded repeated-run benchmarks for the new reusable eigensolver workspace/state paths  
**Time estimate:** 12 hours

### Tasks
1. Implement the repeated-run benchmark slice designed on Day 10.
2. Keep the batch narrow and directly tied to the migrated eigensolver paths.
3. Record benchmark outputs or measurement notes in the sprint artifacts.
4. Avoid broader benchmark harness churn beyond what the repeated-run comparison needs.
5. Run the required code-quality gate if production/test `*.c` / `*.h` files change, plus the benchmark-focused follow-on runs justified by the touched surface.

### Deliverables
- Repeated-run benchmark updates
- First measured allocation/runtime comparison notes
- Validation result for the benchmark batch

### Completion Criteria
- Sprint 46 has direct repeated-run measurement evidence
- The benchmark work remains bounded to eigensolver workspace/state reuse
- Required validation and targeted benchmark runs pass

---

## Day 12: Documentation, Memory Behavior, and Residual Audit

**Title:** Docs and Residual Audit  
**Theme:** Capture the new internal eigensolver workspace/state contract and audit the remaining repeated-allocation seams  
**Time estimate:** 8 hours

### Tasks
1. Document the internal workspace/state contract and maintainer-facing expectations for repeated-run eigensolver work.
2. Audit the residual eigensolver surface for any obvious still-unmigrated allocation seams worth routing to later sprints.
3. Record what Sprint 46 intentionally does not solve:
   - public explicit workspace APIs
   - broad benchmark CLI redesign
   - broad public docs/tutorial refresh
4. Write the docs/residual audit artifact.
5. Update working notes with the intended Day 13 validation sweep shape.

### Deliverables
- Maintainer-facing eigensolver workspace/state contract notes
- Residual repeated-allocation audit
- Day 13 validation plan notes

### Completion Criteria
- The new internal eigensolver workspace/state model is documented for later Epic 4 work
- Residual eigensolver seams are explicitly classified rather than left implicit
- The sprint’s non-goals remain clear before closeout

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the authoritative full validation pass for the Sprint 46 eigensolver workspace/state and repeated-run changes  
**Time estimate:** 10 hours

### Tasks
1. Run the full required code-quality gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the stronger local reviewed baseline:
   - `make quality-review-full`
3. Run the targeted eigensolver and benchmark follow-on checks justified by the touched surfaces.
4. Record measured validation results, touched-binary outcomes, and any benchmark notes in the sprint artifact set.
5. Confirm that the maintained truthfulness anchors remain exact, including reviewed CMake parity.

### Deliverables
- Full validation sweep artifact
- Recorded eigensolver/benchmark follow-on results
- Confirmed maintained baseline metrics

### Completion Criteria
- All required quality gates pass
- The stronger reviewed local baseline passes
- The eigensolver workspace sprint closes from a measured, documented validation state

---

## Day 14: Closeout and Handoff

**Title:** Closeout and Handoff  
**Theme:** Consolidate the Sprint 46 reusable eigensolver workspace/state outcome and hand off the next repeated-run efficiency work cleanly  
**Time estimate:** 12 hours

### Tasks
1. Summarize the final Sprint 46 shipped state:
   - shared eigensolver workspace/state layer
   - migrated primary and LOBPCG paths
   - compatibility-preserving one-shot wrappers
   - repeated-run benchmark evidence
   - Day 13 validated baseline
2. Record what later sprints inherit next, especially any outward-facing API or benchmark CLI work still intentionally deferred.
3. Check whether Sprint 46 surfaced any `PROJECT_PLAN.md` adjustments that must be recorded immediately.
4. Write the closeout/handoff artifact and update working notes.
5. Confirm the branch is clean and the sprint can be handed off without an unresolved queue inside Sprint 46 itself.

### Deliverables
- Sprint 46 closeout and handoff artifact
- Final working-notes synthesis
- Explicit next-sprint handoff notes

### Completion Criteria
- Sprint 46 ends with one coherent eigensolver-workspace package rather than disconnected edits
- Later repeated-run work is handed forward explicitly
- The sprint closes from the Day 13 validated baseline with no hidden queue
