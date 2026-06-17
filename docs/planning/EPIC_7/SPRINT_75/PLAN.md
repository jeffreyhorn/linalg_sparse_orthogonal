# Sprint 75 Plan: Performance Backend Architecture Phase 2

**Sprint Duration:** 14 days  
**Goal:** Deepen the backend/performance architecture beyond the first bounded
Cholesky lane while preserving the self-contained default build. This sprint
implements the Sprint 75 section of
`docs/planning/EPIC_7/PROJECT_PLAN.md`.

**Starting Point:** Sprint 74 closed with a validated first-phase capability
package and a ranked Sprint 75 carry-forward queue:
- the strongest local reviewed baseline remains `make quality-review-full`
- Sprint 70 fixed the Epic 7 performance target and truthfulness fence
- Sprint 64 completed the first backend/performance phase without widening the
  default product claim beyond the shipped self-contained build
- Sprint 74 reduced the strongest capability contradictions without reopening
  backend, platform, or packaging scope
- the strongest remaining backend/performance pressure is now concentrated in:
  - dense-kernel hotspot ownership
  - callback and runtime parity on widened backend-aware paths
  - benchmark proof that makes backend behavior measurable without turning it
    into a pass/fail timing gate
- the preserved Epic 7 non-goal fence is still in force:
  - no fake shared-library or external-backend maturity claim
  - no backend widening that weakens the self-contained default build
  - no benchmark timing thresholds masquerading as portable proof
  - no widened reviewed-platform claim detached from maintained evidence

The highest-value Sprint 75 work is therefore not a broad performance rewrite.
It is a bounded second backend phase on the strongest current hotspot seam,
with focused callback/runtime follow-through, benchmark-proof refresh, and
regression/fallback coverage that makes the next backend-aware slice real
without overstating what the library supports today.

**End State:** Sprint 75 leaves behind a smaller and more coherent backend gap:
- a ranked, evidence-based hotspot map for the next backend-aware lane
- one explicit bounded backend/policy contract for the touched solver/kernel
  path
- one shipped second backend-aware integration slice on the highest-value
  touched seam
- tighter callback and runtime observability parity on that widened path
- refreshed canonical benchmark proof for the landed backend behavior
- focused regression and fallback proof for the touched backend surface
- a validated Sprint 75 closeout package that lets Sprint 76 start from a
  real second backend-aware landing rather than a speculative architecture goal

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
154 hours, staying within the day-cap limit while covering the Sprint 75
implementation scope.

---

## Day 1: Sprint 75 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 75 project-plan scope plus the Sprint 70, Sprint
64, and Sprint 74 handoff into a bounded backend/performance sprint  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 75 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Sprint 70 performance target, the Sprint 64 retrospective, and the
   Sprint 74 closeout artifact.
2. Reconfirm the preserved Sprint 75 constraints:
   - no fake external-backend maturity story
   - no shared-library maturity claim
   - no performance-threshold pass/fail reinterpretation
   - no widened reviewed-platform claim
3. Define the Sprint 75 workstreams explicitly:
   - hotspot re-audit
   - backend/policy design
   - kernel integration batch
   - callback/runtime follow-through
   - benchmark proof refresh
   - regression/fallback proof
   - validation and closeout
4. Record the strongest likely Sprint 75 touch surfaces:
   - backend-aware solver and dense-kernel seams
   - callback/runtime policy seams
   - canonical benchmark proof surfaces
   - highest-value proof-owner tests
5. Open Sprint 75 working notes and record the intended landing order,
   required artifacts, and validation expectations.

### Deliverables
- Sprint 75 scope inventory
- Backend/performance workstream map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 75 starts from the Sprint 64, Sprint 70, and Sprint 74 contract
  rather than reopening broad Epic 7 planning
- The implementation workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or landing work proceeds

---

## Day 2: Validation Baseline & Truth-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the implementation-day validation contract and the
highest-signal rerun surfaces Sprint 75 must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 75 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial backend or architecture batches default to
     `make quality-review-full`
   - docs-only audit/design days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 75 is most likely to stress:
   - backend-aware solver proof owners
   - callback and cancellation parity tests
   - representative examples
   - maintained benchmark/reporting surfaces
   - install/package proof scripts
4. Refresh the targeted rerun set most likely to matter in Sprint 75.
5. Record the authoritative validation split in the working notes.

### Deliverables
- Refreshed validation notes
- Sprint 75 rerun checklist
- Preserved proof-surface checklist

### Completion Criteria
- Sprint 75 uses the same reviewed/truthfulness reading fixed in Sprint 70
- The code-day validation contract is explicit before backend work starts
- No rerun ambiguity remains around the likely touched solver, callback, and
  benchmark seams

---

## Day 3: Hotspot Re-audit

**Title:** Hotspot Audit  
**Theme:** Re-rank dense-kernel, CSC/QR/SVD, and callback-parity hotspots by
   current user value and implementation leverage  
**Time estimate:** 12 hours

### Tasks
1. Audit the current backend/performance hotspot surface across:
   - dense kernels
   - CSC-backed direct paths
   - QR/SVD supporting kernels
   - progress/cancel callback parity
2. Classify the strongest remaining burdens:
   - largest runtime hotspots
   - weakest backend-owner seams
   - weakest observability parity seams
   - highest proof-cost benchmark surfaces
3. Record where the backend surface is already coherent and where it still
   behaves like a narrow first-phase lane.
4. Rank the strongest contradiction centers by:
   - runtime leverage
   - caller value
   - proof cost
   - bounded Sprint 75 payoff
5. Write the first hotspot re-audit artifact.

### Deliverables
- Hotspot re-audit
- Ranked contradiction list
- First backend-modernization hotspot map

### Completion Criteria
- The broad Sprint 75 backend problem is reduced to a concrete seam ranking
- The highest-value hotspots are explicit
- Day 4 can proceed from a real current-state ranking

---

## Day 4: First Backend Boundary

**Title:** Boundary Freeze  
**Theme:** Refine the ranking and freeze the first backend/policy fence for
the sprint  
**Time estimate:** 11 hours

### Tasks
1. Re-rank the Day 3 backend hotspots against:
   - runtime leverage
   - proof cost
   - compatibility risk
   - likely bounded cleanup payoff
2. Separate:
   - first-batch landing surfaces
   - support surfaces that move only if the first batch forces it
   - later or explicitly deferred backend work
3. Identify the strongest first Sprint 75 fence:
   - solver/kernel integration lane
   - callback/runtime parity lane
   - benchmark proof refresh lane
4. Record the strongest non-goals:
   - no broad backend abstraction layer rewrite
   - no fake optional-backend maturity claim
   - no benchmark-threshold portability story
5. Fix the Day 4 modernization boundary in writing.

### Deliverables
- Refined backend ranking
- First landing boundary
- Deferred/support-surface map

### Completion Criteria
- The first Sprint 75 landing fence is explicit before design begins
- Lower-value or higher-risk backend work is clearly separated from the first
  lane
- Support surfaces are bounded rather than assumed

---

## Day 5: Backend / Policy Design

**Title:** Backend Design  
**Theme:** Define the bounded implementation contract for the first
backend-aware landing before edits begin  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 70 performance target and non-goal fences against the
   first-batch surfaces.
2. Decide the ownership split for the strongest remaining backend work:
   - touched solver/kernel owner
   - runtime/backend observability owner
   - fallback and self-contained default-build owner
3. Define the guarantees the batch must preserve:
   - self-contained default build remains the main product path
   - touched callback and cancellation behavior becomes clearer, not weaker
   - benchmark surfaces remain reporting/proof, not timing gates
4. Fix the exact first-batch non-touch set:
   - unrelated solver families
   - broader packaging/platform redesign
   - broad docs cleanup spill
   - capability-surface reopening
5. Record the Day 5 design artifact.

### Deliverables
- Backend/policy design
- First-batch non-touch set
- Preserved compatibility checklist

### Completion Criteria
- The first backend batch is explicitly designed before code edits begin
- The landing is bounded to backend ownership clarity, not generic refactoring
- Compatibility and non-goal fences are fixed in writing

---

## Day 6: Design Freeze & Proof Map

**Title:** Proof Map  
**Theme:** Finalize the exact code/proof ownership split for the first landing
before the first backend edits  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Day 5 design against the likely proof-owner tests and runtime
   callback seams.
2. Map the first backend batch to:
   - implementation owners
   - callback/runtime follow-through owners
   - benchmark proof owners
   - regression/fallback proof owners
3. Confirm which docs and headers stay support-only for the first landing.
4. Write the explicit Day 7 implementation fence and Day 8 audit criteria.
5. Record the finalized proof and ownership map.

### Deliverables
- First-landing proof map
- Finalized Day 7 touch set
- Support-only follow-through list

### Completion Criteria
- The first backend landing can proceed without ambiguity about proof ownership
- Callback/runtime and benchmark follow-through are pre-bounded
- The code batch can start without reopening design scope

---

## Day 7: Kernel Integration Batch II

**Title:** Kernel Batch  
**Theme:** Land the next backend-aware solver/kernel slice on the
highest-value touched path  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded backend-aware kernel or solver-path integration on
   the exact Day 6 touch set.
2. Preserve the self-contained default build and existing fallback behavior.
3. Add or update focused proof in the right owners for:
   - backend-aware behavior
   - fallback behavior
   - touched callback/runtime expectations if directly affected
4. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
5. Record the landed behavior, constraints preserved, and proof results.

### Deliverables
- Landed backend-aware integration slice
- Focused regression updates
- Day 7 implementation artifact

### Completion Criteria
- One real second backend-aware landing is in the tree
- The landing stays inside the Day 6 fence
- Validation passes and the default-build/fallback contract remains intact

---

## Day 8: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining Sprint 75 queue after the first backend-aware
landing  
**Time estimate:** 10 hours

### Tasks
1. Audit the Day 7 landing against the original hotspot ranking.
2. Determine whether the strongest remaining contradiction is now:
   - callback/runtime parity
   - benchmark proof refresh
   - residual support-surface drift
3. Fix the exact Day 9 design center and support-only follow-through list.
4. Record what no longer needs to move in Sprint 75.
5. Write the post-landing rerank artifact.

### Deliverables
- Post-landing audit
- Re-ranked remaining queue
- Exact Day 9 target set

### Completion Criteria
- The strongest remaining Sprint 75 seam is explicit after the code landing
- The next batch is narrowed to one real follow-through center
- No unnecessary second code batch is assumed

---

## Day 9: Callback / Runtime Policy Design

**Title:** Runtime Design  
**Theme:** Define the bounded callback/runtime parity batch needed after the
kernel landing  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Day 7 landing and the Day 8 rerank against callback/runtime
   policy seams.
2. Decide the ownership split for:
   - progress callback parity
   - cancellation semantics
   - backend/runtime observability
   - support-only benchmark or docs follow-through
3. Preserve:
   - current default behavior where no new touched runtime path is active
   - truthful backend/runtime reporting
   - fallback semantics under unsupported or unavailable paths
4. Fix the exact Day 10 touch set and non-touch list.
5. Record the callback/runtime design artifact.

### Deliverables
- Callback/runtime design
- Day 10 touch set
- Preserved parity checklist

### Completion Criteria
- The callback/runtime batch is explicitly designed before edits begin
- Observability and cancellation work is bounded to the touched lane
- No broader runtime-policy rewrite is implied

---

## Day 10: Callback / Runtime Follow-Through Batch

**Title:** Runtime Batch  
**Theme:** Land the bounded callback, cancellation, or backend-observability
follow-through required by the widened backend-aware surface  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 9 callback/runtime follow-through on the exact touch set.
2. Add or update focused proof for:
   - progress/cancel parity
   - backend/runtime reporting
   - fallback behavior on the touched path
3. Re-run:
   - `make format`
   - `make lint`
   - `make test`
4. Decide whether `make quality-review-full` must be rerun immediately based
   on the effective touched risk.
5. Record the landed runtime-parity result and preserved fences.

### Deliverables
- Landed callback/runtime follow-through
- Focused parity proof updates
- Day 10 implementation artifact

### Completion Criteria
- The widened backend surface has truthful runtime and callback behavior
- The batch stays inside the Day 9 fence
- Required validation passes

---

## Day 11: Benchmark Proof Refresh

**Title:** Benchmark Proof  
**Theme:** Refresh canonical proof/report fields so the new backend-aware
behavior is measurable and reviewable  
**Time estimate:** 10 hours

### Tasks
1. Audit the current benchmark and canonical-report surfaces touched by the
   Day 7 and Day 10 landings.
2. Add or tighten only the reporting needed to make the new backend-aware
   behavior measurable.
3. Preserve the benchmark-governance fence:
   - reporting remains threshold-free
   - benchmark output remains proof context, not a pass/fail gate
4. Update support docs only if the landed benchmark story truly moved.
5. Record the benchmark proof refresh artifact.

### Deliverables
- Refreshed benchmark/reporting proof surface
- Updated benchmark-proof notes
- Day 11 artifact

### Completion Criteria
- The widened backend surface has measurable canonical proof
- The benchmark story remains truthful and threshold-free
- No fake portability or timing-gate story is introduced

---

## Day 12: Regression & Fallback Proof Alignment

**Title:** Proof Alignment  
**Theme:** Close any remaining regression or fallback proof gaps without
adding redundant tests  
**Time estimate:** 10 hours

### Tasks
1. Re-read the landed code and benchmark/runtime follow-through against the
   live proof-owner map.
2. Decide whether any focused regression or fallback proof is still missing.
3. Add only the proof required for:
   - backend-aware fallback behavior
   - callback/runtime parity
   - benchmark/reporting interpretation if now test-owned somewhere
4. Fix the exact Day 13 validation queue around the touched surfaces.
5. Record the final proof-alignment artifact.

### Deliverables
- Final proof-owner map
- Any last focused regression additions if truly needed
- Exact Day 13 validation queue

### Completion Criteria
- No real proof gap remains for the landed backend package
- No redundant test expansion is added where no gap exists
- Day 13 validation can run from an explicit queue

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Validate the landed Sprint 75 branch from the strongest reviewed
baseline and the exact focused follow-on queue  
**Time estimate:** 12 hours

### Tasks
1. Run the standard code-day gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed baseline:
   - `make quality-review-full`
3. Recheck the reviewed CMake parity anchors:
   - `ctest -N --test-dir build/quality-review-cmake`
   - Makefile/CMake test-count parity
   - full reviewed CMake `ctest`
4. Run the exact Day 12 follow-ons:
   - touched proof-owner tests
   - representative examples
   - canonical benchmark/reporting surfaces
   - install/package proof scripts if still in the active truth surface
5. Record retained representative outputs and any non-blocking validation note.

### Deliverables
- Full validation record
- Reviewed-anchor summary
- Retained output sample set

### Completion Criteria
- The full Sprint 75 branch passes the standard gate and strongest reviewed
  baseline
- The touched backend, callback, benchmark, and fallback surfaces all re-pass
- Day 14 can close from a validated state instead of a speculative one

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 75 from the Day 13 validated baseline with one
explicit backend/performance handoff package  
**Time estimate:** 10 hours

### Tasks
1. Summarize the landed Sprint 75 package across:
   - hotspot re-audit
   - backend/policy contract
   - kernel integration batch
   - callback/runtime follow-through
   - benchmark proof refresh
   - regression/fallback proof
   - validated close state
2. Rank the strongest Sprint 76+ carry-forward items for:
   - later backend/performance maturity
   - later capability-dependent algorithm breadth
   - later permanent-surface cleanup
3. Recheck whether `docs/planning/EPIC_7/PROJECT_PLAN.md` needs a Sprint 75
   correction.
4. Confirm final Sprint 75 branch state and validation footprint.
5. Record final handoff notes and closeout artifact.

### Deliverables
- Sprint 75 closeout artifact
- Ranked carry-forward queue
- Final handoff notes

### Completion Criteria
- Sprint 75 closes from a validated backend/performance package
- The preserved truthfulness fence remains explicit
- Sprint 76 can start from a real landed backend-aware surface rather than a
  speculative target
