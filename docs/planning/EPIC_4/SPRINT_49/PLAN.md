# Sprint 49 Plan: Lifecycle API Exposure, Final Integration, Validation & Epic 4 Closeout

**Sprint Duration:** 14 days  
**Goal:** Land the final public lifecycle/workspace-facing API refinements that are now safe after the groundwork sprints, then run the final Epic 4 compatibility, residual-review, validation, and closeout passes. This sprint implements the Sprint 49 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`.

**Starting Point:** Sprint 48 closed with the documentation/policy redistribution complete, while Sprint 42 already established the lifecycle scaffolding and Sprint 45-46 already established the internal reusable workspace seams. Epic 4 now has the structural prerequisites needed to expose the final explicit lifecycle-facing API improvements without reopening the large decomposition or reuse groundwork.

**End State:** Sprint 49 leaves behind a compatibility-preserving public lifecycle/workspace package, migration-path documentation for existing callers, a full cross-surface compatibility audit, a final disposition of Epic 4 residual review findings, a fully revalidated integrated baseline, and the final Epic 4 closeout artifact set.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 132 hours, matching the Sprint 49 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 49 Scope Audit & Final-Prerequisite Refresh

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 49 project-plan items into a bounded public-lifecycle and Epic 4 closeout execution map  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 49 section of `docs/planning/EPIC_4/PROJECT_PLAN.md` plus the key prerequisite closeouts:
   - Sprint 42 lifecycle groundwork
   - Sprint 45 iterative workspace reuse
   - Sprint 46 eigensolver workspace reuse
   - Sprint 48 documentation/policy redistribution
2. Reconfirm the preserved constraints Sprint 49 must not violate:
   - maintain compatibility wrappers for existing one-shot usage where required
   - avoid reopening large internal decomposition work already closed in Sprints 43-48
   - preserve Sprint 40 truthfulness around validation and warning authority
   - treat public API exposure as bounded finalization, not broad API redesign
3. Define the Sprint 49 workstreams explicitly:
   - public lifecycle API landing
   - migration-path documentation
   - cross-surface compatibility sweep
   - final residual review
   - full integration validation
   - Epic 4 summary artifacts
   - closeout and handoff
4. Record the highest-risk final seams:
   - compatibility drift between old one-shot paths and new explicit lifecycle/workspace usage
   - examples/benchmarks/docs disagreeing on the final model
   - exposing lifecycle/workspace affordances without sufficiently clear caller guidance
5. Open Sprint 49 working notes and record scope, assumptions, and intended landing order.

### Deliverables
- Sprint 49 scope inventory
- Final-prerequisite refresh notes
- Working-notes baseline assumptions

### Completion Criteria
- Sprint 49 starts from the documented Epic 4 baseline rather than ad hoc API exposure
- Preserved compatibility and validation constraints are explicit before implementation begins
- The final Epic 4 workstreams are named before edits start

---

## Day 2: Public Lifecycle/API Surface Inventory

**Title:** Surface Inventory  
**Theme:** Re-map the public lifecycle, workspace, compatibility-wrapper, and caller-facing documentation seams before choosing the landing order  
**Time estimate:** 8 hours

### Tasks
1. Refresh the live seam inventory for:
   - public headers
   - internal lifecycle/workspace support seams
   - compatibility wrappers
   - examples and benchmark drivers that present solver usage patterns
2. Classify the remaining public-surface work into bounded buckets:
   - explicit lifecycle handle exposure
   - compatibility-preserving one-shot wrappers
   - caller guidance and migration rules
   - cross-surface documentation alignment
   - final residual-review bookkeeping
3. Identify which surfaces are the real first landing targets versus later verification surfaces.
4. Confirm the first implementation order:
   - API design
   - bounded public header / source landing
   - migration docs
   - cross-surface compatibility sweep
   - residual review
   - final validation and closeout
5. Write the public-surface inventory artifact.

### Deliverables
- Refreshed public lifecycle/workspace seam inventory
- Shared-vs-wrapper classification
- First landing-order notes

### Completion Criteria
- The remaining Epic 4 API exposure work is reduced to named seams
- Public-facing and compatibility-wrapper responsibilities are explicit
- Later implementation order is grounded in the live repo surface

---

## Day 3: Public Lifecycle API Landing Design

**Title:** API Design  
**Theme:** Define the bounded public lifecycle/workspace model that can now be safely exposed after the groundwork sprints  
**Time estimate:** 8 hours

### Tasks
1. Design the intended public lifecycle/workspace exposure:
   - which handles/types/functions become public
   - what stays internal-only
   - what compatibility wrappers remain first-class
2. Define the public contract for:
   - initialization / prepare
   - solve / run
   - reuse / reset expectations
   - teardown / free
3. Decide how option structs, result structs, and designated-initializer norms should appear in the final public shape.
4. Record explicit non-goals:
   - broad solver API redesign
   - removal of compatibility wrappers in Sprint 49
   - publicizing every internal helper
5. Write the lifecycle API design artifact.

### Deliverables
- Public lifecycle/workspace API design
- Compatibility and non-goal notes
- Public contract boundary definition

### Completion Criteria
- Sprint 49 has a concrete public lifecycle target before code movement begins
- Compatibility rules are explicit before implementation starts
- The design stays bounded away from broad API churn

---

## Day 4: Landing and Validation Design

**Title:** Landing Design  
**Theme:** Bound the lifecycle API landing batches and the final validation shape before implementation begins  
**Time estimate:** 8 hours

### Tasks
1. Define the validation contract for Sprint 49 implementation days:
   - any `*.c` / `*.h` change requires `make format`, `make lint`, and `make test`
   - substantial public-API batches should also default to `make quality-review-full`
2. Decide the targeted follow-on binaries/tests most likely to matter after public lifecycle landing:
   - touched examples
   - touched benchmarks
   - solver-specific regression binaries where appropriate
3. Bound the intended landing order:
   - header/API surface
   - implementation/wrapper integration
   - migration docs
   - compatibility sweep
   - residual review
   - final validation
4. Record explicit out-of-scope items:
   - post-Epic-4 feature expansion
   - large new benchmark framework work
   - new solver families
   - broad tutorial rewrite unrelated to the final lifecycle shape
5. Write the landing/validation design artifact.

### Deliverables
- Validation-plan artifact
- Mid-sprint landing order
- Explicit out-of-scope notes

### Completion Criteria
- The sprint has a clear validation contract before public API edits begin
- The landing batches are sequenced from the live compatibility map
- Scope boundaries are explicit before implementation starts

---

## Day 5: Public Lifecycle API Landing Batch I

**Title:** API Batch I  
**Theme:** Land the first bounded public header/API exposure batch with compatibility preserved  
**Time estimate:** 12 hours

### Tasks
1. Add the first bounded public lifecycle/workspace declarations to the relevant headers.
2. Keep the exposure narrowly aligned with the Day 3 design:
   - expose the minimal intended handle/type/function surface
   - avoid leaking internal-only helpers
3. Preserve existing one-shot caller behavior through compatibility wrappers or unchanged public entry points where needed.
4. Update nearby comments/docblocks so the newly public surface is self-consistent.
5. Run the full required gate plus targeted touched-surface follow-ons.

### Deliverables
- First public lifecycle header/API landing
- Compatibility-preserving public declarations
- Validation notes for the touched batch

### Completion Criteria
- The first public lifecycle/workspace surface is landed without broad API churn
- Compatibility wrappers remain intact where the old one-shot model still needs to work
- The required validation gate passes

---

## Day 6: Public Lifecycle API Landing Batch II

**Title:** API Batch II  
**Theme:** Land the corresponding implementation and wrapper integration behind the newly exposed public lifecycle surface  
**Time estimate:** 12 hours

### Tasks
1. Wire the public lifecycle/workspace declarations to the intended implementation paths.
2. Make the compatibility wrappers explicitly delegate through the final public lifecycle model where appropriate.
3. Preserve result initialization, error-path safety, and free/reset semantics.
4. Keep the batch bounded:
   - no new solver-family expansion
   - no unrelated refactors
5. Run the full required gate, the stronger reviewed baseline, and targeted touched-surface follow-ons.

### Deliverables
- Public lifecycle implementation landing
- Compatibility-wrapper integration
- Validation notes for the touched batch

### Completion Criteria
- The exposed lifecycle/workspace API is backed by working implementation
- Legacy usage remains supported through bounded compatibility handling
- Validation passes at the required and reviewed-baseline levels

---

## Day 7: Post-Landing API Audit

**Title:** API Audit  
**Theme:** Re-audit the public and compatibility surfaces after the first landing to narrow the remaining queue  
**Time estimate:** 8 hours

### Tasks
1. Inspect the live post-landing state across:
   - public headers
   - implementation/wrapper paths
   - touched examples/benchmarks/docs
2. Confirm what is complete versus what still needs explicit follow-on work:
   - migration docs
   - compatibility sweep
   - residual-review mapping
3. Identify any remaining naming, comment, or ownership drift caused by the landing.
4. Bound the Day 8/9 follow-on work so it stays focused on migration-path clarity and surface agreement.
5. Write the post-landing audit artifact.

### Deliverables
- Post-landing public-surface audit
- Narrowed remaining Sprint 49 queue
- Follow-on work boundary notes

### Completion Criteria
- The remaining Sprint 49 queue is concrete instead of generic
- Migration and compatibility work is bounded before more edits start
- No unintended API-sprawl is left ambiguous

---

## Day 8: Migration-Path Documentation Batch

**Title:** Migration Docs  
**Theme:** Document how existing callers can stay on the old one-shot path and when the explicit lifecycle/workspace path is preferable  
**Time estimate:** 10 hours

### Tasks
1. Write or update the primary migration-path documentation around:
   - old one-shot usage
   - explicit lifecycle/workspace usage
   - when reuse is worth it
   - when compatibility wrappers remain appropriate
2. Keep the guidance concrete and scoped to the final Epic 4 public shape.
3. Cross-reference the touched examples, README/tutorial material, or maintainer guide as needed.
4. Avoid broad tutorial redesign beyond what is necessary for clarity.
5. Run targeted doc sanity checks on the touched migration-path surfaces.

### Deliverables
- Migration-path documentation
- Explicit old-vs-new usage guidance
- Targeted doc sanity-check notes

### Completion Criteria
- Existing callers have a clear compatibility path
- The new explicit lifecycle/workspace style has a concrete “when to use it” explanation
- The documentation stays aligned with the landed public API

---

## Day 9: Cross-Surface Compatibility Sweep Design and Audit

**Title:** Compatibility Audit  
**Theme:** Reconcile the final lifecycle/workspace model across examples, benchmarks, tests, and docs before the final sweep lands  
**Time estimate:** 8 hours

### Tasks
1. Audit the touched cross-surface callers:
   - examples
   - benchmarks
   - tests
   - docs
2. Classify remaining agreement issues into bounded buckets:
   - stale one-shot-only examples
   - stale wording in docs
   - benchmarks presenting outdated usage assumptions
   - tests not reflecting the final public contract
3. Decide the smallest high-signal sweep to land on Day 10.
4. Keep the sweep behavior-level and avoid gratuitous churn.
5. Write the compatibility-audit artifact.

### Deliverables
- Cross-surface compatibility audit
- Final sweep target list
- Bounded Day 10 landing plan

### Completion Criteria
- Cross-surface drift is mapped before cleanup lands
- The remaining compatibility sweep is narrowed to the most valuable surfaces
- The sprint avoids broad unfocused doc/example churn

---

## Day 10: Cross-Surface Compatibility Sweep Batch

**Title:** Compatibility Sweep  
**Theme:** Land the smallest coherent batch that makes examples, benchmarks, tests, and docs agree on the final lifecycle/workspace model  
**Time estimate:** 10 hours

### Tasks
1. Update the bounded set of surfaces identified on Day 9.
2. Preserve behavior while reconciling:
   - usage examples
   - benchmark descriptions or invocation assumptions
   - tests or comments reflecting the public contract
   - nearby docs
3. Keep the batch tightly scoped to final model agreement.
4. Run the required validation gate if any code or headers changed; otherwise run targeted touched-surface checks.
5. Record any remaining true residuals rather than widening the batch.

### Deliverables
- Cross-surface compatibility cleanup
- Agreement between examples/benchmarks/tests/docs on the final model
- Validation notes for the touched batch

### Completion Criteria
- The final lifecycle/workspace model reads consistently across the key surfaces
- The batch stays bounded and avoids unrelated cleanup
- Any true residuals are named rather than silently carried

---

## Day 11: Final Residual Review

**Title:** Residual Review  
**Theme:** Revisit every Epic 4 review finding and classify it against the final landed state  
**Time estimate:** 10 hours

### Tasks
1. Re-read `review-codex-2026-05-21.md` and any later tracked residual queues relevant to Epic 4.
2. Classify each item as:
   - fixed
   - intentionally deferred
   - accepted tradeoff
3. Tie each disposition to the specific sprint/day or final residual rationale.
4. Identify whether any true blocking gap remains before final validation.
5. Write the final residual-review artifact.

### Deliverables
- Final residual-review classification
- Fixed/deferred/accepted-tradeoff mapping
- Blocking-gap check

### Completion Criteria
- Every tracked Epic 4 review finding has a final disposition
- Any remaining residuals are explicit and justified
- Sprint 49 has a clear pre-validation residual state

---

## Day 12: Epic 4 Summary Artifact Batch

**Title:** Summary Batch  
**Theme:** Prepare the final integrated Epic 4 summary artifacts before the authoritative validation closeout  
**Time estimate:** 8 hours

### Tasks
1. Draft the final Epic 4 summary framing:
   - what Epic 4 changed structurally
   - what public-facing lifecycle/workspace contract now exists
   - what residual risks remain
2. Record the final post-remediation contract:
   - compatibility expectations
   - validation authority
   - deferred work boundaries
3. Keep the summary grounded in the actual landed state and measured validation anchors.
4. Prepare the explicit Day 13 validation checklist.
5. Write the summary artifact batch.

### Deliverables
- Epic 4 summary artifact draft
- Final contract framing
- Day 13 validation checklist

### Completion Criteria
- The final summary package is ready before the authoritative validation sweep
- The final contract is explicit rather than implied
- Validation inputs are fixed before Day 13 begins

---

## Day 13: Full Integration Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the authoritative final Epic 4 validation pass from the integrated public lifecycle/workspace end state  
**Time estimate:** 12 hours

### Tasks
1. Run the full required baseline:
   - `make format`
   - `make lint`
   - `make test`
2. Run the stronger reviewed baseline:
   - `make quality-review-full`
3. Reconfirm maintained truthfulness anchors:
   - `ctest -N --test-dir build/quality-review-cmake`
   - Makefile/CMake parity
   - full reviewed CMake `ctest`
4. Run targeted Sprint 49 follow-ons justified by the final lifecycle landing:
   - touched examples
   - touched benchmarks
   - touched solver regression binaries
5. Record measured results, note any failures, and resolve any bounded issues that arise.

### Deliverables
- Full validation-sweep artifact
- Final measured baseline results
- Targeted follow-on validation notes

### Completion Criteria
- The complete Sprint 49 / Epic 4 integrated baseline passes
- Truthfulness anchors remain explicit and measured
- No unresolved blocking issue remains before closeout

---

## Day 14: Epic 4 Closeout and Handoff

**Title:** Closeout  
**Theme:** Finalize Sprint 49 and Epic 4 closeout, route true residuals forward, and leave the final handoff package coherent  
**Time estimate:** 10 hours

### Tasks
1. Write the Sprint 49 closeout and handoff artifact.
2. Write the final Epic 4 closeout synthesis:
   - delivered package
   - measured validation baseline
   - final residuals / accepted tradeoffs
   - future-planning handoff
3. Determine whether `PROJECT_PLAN.md` needs any final status update after Epic 4 closeout.
4. Confirm the retrospective inputs are complete for both Sprint 49 and Epic 4 end-state reporting.
5. Record the final working-notes synthesis and leave the branch in a clean handoff state.

### Deliverables
- Sprint 49 closeout and handoff artifact
- Final Epic 4 closeout synthesis
- Residual-routing / future-planning notes

### Completion Criteria
- Sprint 49 closes from the measured Day 13 baseline
- Epic 4 has a coherent final handoff package
- True residuals are explicitly routed rather than left implicit
