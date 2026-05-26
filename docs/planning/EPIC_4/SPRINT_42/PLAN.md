# Sprint 42 Plan: Matrix Lifecycle Refactor Phase 1 - Internal Handles & Compatibility Scaffolding

**Sprint Duration:** 14 days  
**Goal:** Start reducing the hidden mutable `SparseMatrix` lifecycle by introducing explicit internal handle boundaries, shared matrix-state guard helpers, first-phase LU/Cholesky payload separation, and compatibility scaffolding, while preserving the current public API and the Sprint 40 internal-first contract. This sprint implements the Sprint 42 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`.

**Starting Point:** Sprint 41 closed with a validated shared internal allocation/overflow layer, broader `src/` safety alignment, bounded example-side helper alignment, and compact prep rules for later Epic 4 work. Sprint 42 begins from that preserved baseline plus Sprint 40's lifecycle taxonomy, handle-model design, migration-risk audit, and validation anchor.

**End State:** Sprint 42 leaves behind internal lifecycle-handle scaffolding, shared matrix-state validation helpers, first normalized factor-entry paths, clarified cancellation and mutation expectations, compatibility bridge planning around `sparse_factors_t`, stronger lifecycle misuse tests, and a validated compatibility-preserving baseline for the next lifecycle phases.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 144 hours, matching the Sprint 42 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 42 Scope Audit & Lifecycle Baseline

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 42 project-plan items into a bounded internal-handle and compatibility work inventory  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 42 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`, the Sprint 40 architecture-contract synthesis, the Sprint 40 lifecycle artifacts, and the Sprint 41 closeout notes.
2. Reconfirm the preserved contracts Sprint 42 must not reopen:
   - internal-first implementation
   - current public API compatibility
   - Sprint 40 validation anchor
   - Sprint 41 shared helper-layer reuse
3. Define the Sprint 42 workstreams explicitly:
   - lifecycle seam inventory refresh
   - internal handle scaffolding
   - matrix-state guard helpers
   - factor-path normalization
   - cancellation-contract normalization
   - compatibility bridge planning
   - focused lifecycle tests
   - validation closeout
4. Record the primary high-risk seams already handed forward:
   - LU factorization internals
   - Cholesky factorization internals
   - `sparse_factors_t`
   - lifecycle-sensitive precondition checks across QR/SVD/analysis/direct-factor paths
5. Open Sprint 42 working notes and record scope, assumptions, and migration order.

### Deliverables
- Sprint 42 scope inventory
- Initial lifecycle-refactor workstream map
- Working-notes baseline assumptions

### Completion Criteria
- Sprint 42 starts from the documented Epic 4 baseline rather than ad hoc implementation
- The preserved compatibility and validation constraints are explicit up front
- The lifecycle-refactor surfaces are named before code changes begin

---

## Day 2: Lifecycle Seam Refresh Inventory

**Title:** Seam Inventory  
**Theme:** Re-map the concrete implementation seams where hidden mutable `SparseMatrix` state still leaks through  
**Time estimate:** 8 hours

### Tasks
1. Refresh the lifecycle seam inventory for the Day 5-10 implementation targets:
   - LU
   - Cholesky
   - LDLT / analyze-once bridge
   - QR/SVD original-matrix guard paths
2. Classify each seam by:
   - matrix mutation boundary
   - factor payload ownership
   - cancellation behavior
   - public compatibility exposure
3. Record which seams need new internal object boundaries versus only shared precondition logic.
4. Separate immediate Sprint 42 landing seams from later public-handle or broader lifecycle work.
5. Write the refreshed lifecycle-seam baseline artifact.

### Deliverables
- Refreshed lifecycle seam inventory
- Immediate-vs-later seam classification
- Implementation landing order for internal handle work

### Completion Criteria
- The internal-handle work is grounded in concrete current seams
- Immediate Sprint 42 targets are separated from later lifecycle phases
- The factor-payload and cancellation seams are explicit before design coding

---

## Day 3: Internal Handle Scaffolding Design

**Title:** Handle Design  
**Theme:** Define the first internal handle and ownership boundaries before implementing them  
**Time estimate:** 10 hours

### Tasks
1. Design the first internal handle-family scaffolding for:
   - LU numeric payload ownership
   - Cholesky numeric payload ownership
   - bridge payload normalization around `sparse_factors_t`
2. Define how the new internal objects relate to existing `SparseMatrix`-centric code without changing the public API yet.
3. Decide the target file layout, naming, and ownership rules for the first handle scaffolding.
4. Record which state remains in `SparseMatrix` for Sprint 42 and which state moves behind internal boundaries.
5. Write the internal handle scaffolding design artifact.

### Deliverables
- Internal handle scaffolding design
- Ownership and file-layout plan
- Explicit keep-in-matrix vs move-behind-handle notes

### Completion Criteria
- The first internal handle layer is designed concretely before code lands
- LU/Cholesky and bridge ownership seams are explicit
- Public API expansion remains out of scope for this sprint

---

## Day 4: Matrix-State Guard Helper Design

**Title:** Guard Design  
**Theme:** Define the shared internal precondition helpers for matrix-lifecycle-sensitive paths  
**Time estimate:** 10 hours

### Tasks
1. Design the shared validation helper layer for:
   - original-state required
   - identity permutations required
   - factored-state required
   - compatibility-friendly error reporting
2. Map the initial adoption set:
   - LU
   - Cholesky
   - LDLT / analysis bridge
   - QR
   - SVD
3. Decide which checks should become shared helpers versus stay local because they are algorithm-specific.
4. Record how the guard helpers will preserve current semantics while reducing bespoke drift.
5. Write the matrix-state guard design artifact.

### Deliverables
- Shared guard-helper design
- Initial adoption matrix
- Shared-vs-local validation boundary notes

### Completion Criteria
- The precondition-helper layer is designed before coding
- Shared checks are distinguished from algorithm-specific checks
- Compatibility-preserving semantics are explicit

---

## Day 5: Internal Handle Scaffolding Batch I

**Title:** Handle Batch I  
**Theme:** Land the first internal lifecycle-handle scaffolding in the LU/Cholesky seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the first internal handle/payload scaffolding based on Day 3's design.
2. Insert the initial LU- and/or Cholesky-adjacent ownership boundary without changing the public API shape.
3. Keep the batch narrow: no broad lifecycle redesign outside the chosen seam.
4. Add succinct internal comments where the ownership split would otherwise be difficult to infer.
5. Run the required code-quality gate and any targeted checks justified by the touched factorization paths.

### Deliverables
- First internal handle scaffolding in-tree
- Initial LU/Cholesky ownership-boundary proof
- Validation result for the first handle batch

### Completion Criteria
- A real internal-handle seam exists in live code
- The public API remains behavior-compatible
- The required validation passes

---

## Day 6: Matrix-State Guard Helper Implementation

**Title:** Guard Implementation  
**Theme:** Add the shared state-validation helper layer and first live uses  
**Time estimate:** 12 hours

### Tasks
1. Implement the shared matrix-state guard helper layer from Day 4's design.
2. Migrate the first low-risk users onto the shared helpers.
3. Preserve current error returns and user-visible semantics while removing duplicated guard drift.
4. Record any validation paths that still need algorithm-specific checks to remain local.
5. Run the required code-quality gate and any targeted checks justified by the touched paths.

### Deliverables
- Shared matrix-state guard helper layer
- First live adoption points
- Shared-vs-local residual-check notes

### Completion Criteria
- The shared guard layer exists and is used in live code
- User-visible semantics remain stable
- The required validation passes

---

## Day 7: Factor-Path Landing Audit

**Title:** Factor Audit  
**Theme:** Choose the bounded factor-entry paths for the main normalization batches  
**Time estimate:** 10 hours

### Tasks
1. Audit the broader factor-entry set for immediate shared-guard and internal-handle adoption:
   - LU
   - Cholesky
   - LDLT / analyze-once bridge
   - QR
   - SVD
2. Separate:
   - ready-for-direct adoption paths
   - bridge paths needing minor local adapters
   - later/separate lifecycle paths
3. Confirm where `sparse_factors_t` normalization can start safely in Sprint 42.
4. Record the migration order for Days 8-10.
5. Write the factor-path landing audit artifact.

### Deliverables
- Prioritized factor-path landing map
- `sparse_factors_t` readiness notes
- Day 8-10 migration order

### Completion Criteria
- The main implementation batches have a concrete landing order
- Bridge normalization pressure is visible before code moves
- Later/deferred lifecycle work is separated from Sprint 42 scope

---

## Day 8: Factor-Path Normalization Batch I

**Title:** Factor Batch I  
**Theme:** Start replacing bespoke lifecycle checks in the highest-value factor-entry paths  
**Time estimate:** 10 hours

### Tasks
1. Migrate the highest-priority factor-entry paths chosen on Day 7 onto the shared guard helpers.
2. Tighten the touched internal lifecycle boundaries where the new handle scaffolding already supports it.
3. Keep the batch behavior-preserving and avoid opportunistic broader refactors.
4. Record any bridge-path adapters needed for the next batch.
5. Run the required code-quality gate and targeted follow-on checks justified by the touched paths.

### Deliverables
- First factor-path normalization batch
- Reduced bespoke lifecycle-check drift
- Bridge-adapter notes for the next batch

### Completion Criteria
- The highest-value factor-entry paths adopt the shared guard layer
- The internal handle scaffolding proves useful in live code
- Validation passes

---

## Day 9: Factor-Path Normalization Batch II

**Title:** Factor Batch II  
**Theme:** Extend normalization into the bridge and secondary lifecycle-sensitive paths  
**Time estimate:** 10 hours

### Tasks
1. Complete the main Sprint 42 factor-path normalization set chosen on Day 7.
2. Start the bounded `sparse_factors_t` payload normalization or bridge cleanup that is safe for this sprint.
3. Reconcile any helper or ownership drift surfaced by Day 8.
4. Record what remains intentionally deferred to later lifecycle phases.
5. Run the required code-quality gate and targeted follow-on checks justified by the touched paths.

### Deliverables
- Main Sprint 42 factor-path normalization pass
- Initial `sparse_factors_t` bridge normalization result
- Explicit deferred-lifecycle list

### Completion Criteria
- The planned factor-path normalization set is landed or explicitly bounded
- `sparse_factors_t` starts moving toward the future bridge role
- Validation passes

---

## Day 10: Cancellation & Mutation Contract Normalization

**Title:** Contract Normalization  
**Theme:** Reduce cancellation and in-place-mutation drift across touched lifecycle paths  
**Time estimate:** 10 hours

### Tasks
1. Audit the touched LU/Cholesky/bridge paths for cancellation and mutation-contract drift between code, comments, and tests.
2. Normalize implementation-side behavior or internal comments where minor drift can be resolved safely in Sprint 42.
3. Record any remaining public-doc or later-phase lifecycle wording work without reopening broad docs churn yet.
4. Keep the batch focused on touched lifecycle semantics rather than general documentation cleanup.
5. Run the required code-quality gate and targeted follow-on checks justified by the touched paths.

### Deliverables
- Cancellation/mutation normalization batch
- Residual later-phase wording notes
- Validation result for the contract batch

### Completion Criteria
- Touched lifecycle paths express clearer, more consistent cancellation expectations
- Mutation boundaries are more explicit in code or tests where safe
- Validation passes

---

## Day 11: Compatibility Bridge & Focused-Test Design

**Title:** Compatibility Design  
**Theme:** Define the compatibility shim expectations and the focused misuse-test additions before the final batch  
**Time estimate:** 10 hours

### Tasks
1. Write the compatibility-wrapper and bridge plan for how current one-shot APIs continue to sit over the evolving internal-handle model.
2. Define how `sparse_factors_t` remains the preserve-and-evolve bridge during later payload normalization.
3. Audit current tests for lifecycle misuse, copy-before-use requirements, and cancellation behavior.
4. Choose the bounded focused-test additions for Day 12.
5. Write the compatibility and focused-test design artifact.

### Deliverables
- Compatibility wrapper/bridge plan
- `sparse_factors_t` preserve-and-evolve notes
- Focused lifecycle-test addition plan

### Completion Criteria
- The compatibility story is explicit before later public-handle work
- The focused test batch is defined concretely
- Later lifecycle phases inherit a clear bridge contract

---

## Day 12: Focused Lifecycle Tests

**Title:** Lifecycle Tests  
**Theme:** Add or tighten tests around lifecycle misuse, copy-before-use requirements, and cancellation  
**Time estimate:** 10 hours

### Tasks
1. Add or tighten the focused tests chosen on Day 11.
2. Cover the highest-value lifecycle misuse surfaces:
   - original-state requirements
   - identity-permutation requirements
   - cancellation behavior
   - copy-before-use compatibility expectations
3. Keep the test batch tightly aligned with the Sprint 42 code changes.
4. Record any remaining test gaps that belong to later public-handle phases.
5. Run the required code-quality gate and any targeted follow-on checks justified by the touched tests.

### Deliverables
- Focused lifecycle misuse/cancellation tests
- Residual later-phase test-gap notes
- Validation result for the test batch

### Completion Criteria
- The highest-value touched lifecycle contracts have stronger test coverage
- The tests reinforce compatibility-preserving behavior
- Validation passes

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Re-run the strongest maintained validation baseline over the Sprint 42 landing set  
**Time estimate:** 12 hours

### Tasks
1. Run the mandatory code-change validation floor:
   - `make format`
   - `make lint`
   - `make test`
2. Run the default strong proof for substantial refactors:
   - `make quality-review-full`
3. Run any targeted follow-on checks justified by the touched lifecycle surfaces.
4. Reconfirm the Sprint 40/Sprint 41 truthfulness anchors:
   - `ctest -N --test-dir build/quality-review-cmake`
   - Makefile/CMake test-count parity
   - behavior-compatible public API surface
5. Record the full validation results and any final reconciliation notes.

### Deliverables
- Full validation sweep record
- Updated measured baseline for Sprint 42
- Final reconciliation notes if needed

### Completion Criteria
- The full maintained validation baseline passes
- The measured parity/quality anchors remain truthful
- No unresolved behavior-compatibility regressions remain

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Synthesize Sprint 42 into a clean handoff for later lifecycle phases  
**Time estimate:** 12 hours

### Tasks
1. Summarize what Sprint 42 landed in architecture terms:
   - internal handle scaffolding
   - shared matrix-state guards
   - first factor-path normalization
   - compatibility bridge direction
   - strengthened lifecycle tests
2. Record what remains explicitly deferred to Sprint 43 and later lifecycle phases.
3. Confirm whether any `PROJECT_PLAN.md` updates are needed so new deferred items are not lost.
4. Write the closeout and handoff artifact for Sprint 42.
5. Ensure the sprint ends from the validated Day 13 baseline rather than open implementation drift.

### Deliverables
- Sprint 42 closeout and handoff artifact
- Explicit next-sprint prerequisite list
- `PROJECT_PLAN.md` follow-up note if needed

### Completion Criteria
- Sprint 42 hands off a coherent lifecycle-groundwork package, not isolated refactors
- Deferred work is routed explicitly if anything new surfaced
- The handoff is grounded in the measured validation baseline
