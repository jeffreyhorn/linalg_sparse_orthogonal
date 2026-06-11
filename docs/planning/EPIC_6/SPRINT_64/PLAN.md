# Sprint 64 Plan: Performance Backend Architecture Phase 1

**Sprint Duration:** 14 days  
**Goal:** Introduce a bounded backend/performance abstraction for the
highest-value dense-kernel and supernodal hot paths while preserving the
self-contained default build and the reviewed truthfulness contract. This
sprint implements the Sprint 64 section of
`docs/planning/EPIC_6/PROJECT_PLAN.md`.

**Starting Point:** Sprint 63 closed with the direct-lifecycle follow-through
package landed and validated:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- one-shot direct wrappers remain first-class/default peer entry points
- the explicit repeated-run direct lifecycle remains the canonical reuse path
- large-`n` CSC-backed Cholesky failure-preserve semantics are now proved on
  the public repeated-run direct path
- the next Epic 6 priority is no longer direct-lifecycle coherence; it is a
  bounded performance/backend architecture sprint centered on hotspot ranking,
  kernel abstraction, fallback correctness, benchmark proof, and validated
  closeout

The next highest-value work is not broad platform expansion or packaging work.
It is a bounded backend-architecture sprint focused on the dense-kernel and
supernodal paths that matter most, the option/build surface needed to expose
them safely, and the proof burden required to preserve the self-contained
default path.

**End State:** Sprint 64 leaves behind the first bounded Epic 6
backend/performance architecture package:
- a ranked hotspot audit for dense-kernel, supernodal, and related execution
  paths
- an explicit backend abstraction contract for the highest-value selected
  kernels
- the first backend-aware integration on selected hot paths
- preserved default-build and fallback correctness with explicit proof
- refreshed benchmark and regression surfaces that demonstrate the new path
- full validation and closeout from the landed state

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
142 hours, staying within the Sprint 64 estimate and below the 168-hour limit.

---

## Day 1: Sprint 64 Scope Audit & Backend Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 64 project-plan scope plus the Sprint 63 validated
close into a bounded backend-architecture implementation map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 64 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 63 retrospective, and
   the strongest Sprint 63 closeout artifacts.
2. Reconfirm the preserved Sprint 64 constraints:
   - no broad framework rewrite
   - no fake platform closure beyond reviewed evidence
   - no backend widening that weakens the self-contained default build
   - no benchmark-governance sprawl disguised as kernel work
3. Define the Sprint 64 workstreams explicitly:
   - hotspot audit
   - backend abstraction design
   - kernel integration batch
   - build/options wiring
   - benchmark proof refresh
   - regression and fallback checks
   - validation and closeout
4. Record the strongest likely Sprint 64 touch surfaces:
   - dense-kernel and supernodal implementation seams
   - public options and build/config surfaces
   - benchmark and validation proof surfaces
5. Open Sprint 64 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 64 scope inventory
- Backend baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 64 starts from the Sprint 63 validated close rather than reopening
  lifecycle or productization target-definition work
- The backend-architecture workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or code edits land

---

## Day 2: Validation Baseline & Performance Rerun Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed baseline and rerun set that Sprint 64 kernel
and backend changes must preserve  
**Time estimate:** 8 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity counts
   - current quality/truthfulness wording
2. Reconfirm the mandatory gate for later `*.c` / `*.h` days:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial backend, performance, or
   build-option work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Sprint 64:
   - direct and CSC proof surfaces touching dense-kernel paths
   - representative examples and high-signal benchmark binaries
   - adjacent iterative/eigensolver sentinels that should not drift
5. Record the authoritative validation split for docs-only, bounded code-day,
   and substantial backend-architecture days.

### Deliverables
- Refreshed validation notes
- Sprint 64 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 64 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before implementation work begins
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Performance Hotspot Audit

**Title:** Hotspot Audit I  
**Theme:** Re-rank the dense-kernel, supernodal, and related execution hot
paths by payoff, proof cost, and fallback risk  
**Time estimate:** 11 hours

### Tasks
1. Inventory the strongest performance-sensitive implementation surfaces
   across:
   - dense-kernel helpers
   - CSC supernodal paths
   - solve/factor hot loops
   - threading-sensitive helper seams
2. Classify each hotspot by:
   - runtime payoff potential
   - correctness sensitivity
   - fallback complexity
   - build-surface implications
3. Rank the strongest Sprint 64 kernel candidates by:
   - user-visible value
   - boundedness of the integration
   - proof burden
   - compatibility risk
4. Identify the strongest first abstraction cut line for Sprint 64.
5. Write the audit artifact with the explicit ranked hotspot map.

### Deliverables
- Live hotspot inventory
- Ranked Sprint 64 kernel candidate list
- Risk/payoff classification draft

### Completion Criteria
- The broad “backend architecture” claim is reduced to a concrete ranked list
- The strongest first kernel targets are explicit before design begins
- Day 4 can proceed from a real migration target instead of generic
  performance ambitions

---

## Day 4: Hotspot Audit Follow-Through & Residual Rerank

**Title:** Hotspot Audit II  
**Theme:** Separate the must-touch Phase 1 kernel seams from later
architecture, packaging, and benchmark-governance work  
**Time estimate:** 10 hours

### Tasks
1. Re-rank the Day 3 candidates against the Epic 6 state-of-the-art target.
2. Separate:
   - must-touch Phase 1 backend seams
   - important but later Phase 2 backend seams
   - non-goal or deferred architecture work
3. Confirm which hotspots belong in:
   - kernel abstraction work
   - build/options work
   - benchmark-only proof refresh
   - later packaging/platform effort
4. Fix the first selected Sprint 64 landing surface in writing.
5. Record the residual queue that Sprint 64 should not absorb.

### Deliverables
- Refined hotspot ranking
- First selected landing target
- Deferred backend residual map

### Completion Criteria
- The Sprint 64 target set is smaller and sharper than the original epic-level
  review
- The first backend-aware integration target is explicit before abstraction
  design begins
- Later-phase work is clearly separated from the bounded Phase 1 sprint

---

## Day 5: Backend Abstraction Contract Design

**Title:** Abstraction Design  
**Theme:** Define the bounded backend layer for selected kernels, fallback
rules, and preserved default-build behavior  
**Time estimate:** 12 hours

### Tasks
1. Design the bounded backend abstraction for the selected Sprint 64 hot path.
2. Define exact ownership across:
   - public option surfaces
   - internal backend dispatch
   - kernel fallback behavior
   - telemetry or proof surfaces
3. Define the preserved compatibility rules:
   - default self-contained build stays authoritative
   - backend-aware acceleration remains optional and bounded
   - fallback correctness takes priority over opportunistic widening
4. Decide which controls belong in:
   - build-time switches
   - internal typed policy
   - public options, only if justified
5. Record the explicit safety contract for the first implementation batch.

### Deliverables
- Backend abstraction design artifact
- Explicit safety/compatibility contract
- First implementation contract

### Completion Criteria
- The backend layer is explicit before implementation edits start
- Public and internal ownership are separated clearly enough to prevent drift
- Default-path, optional-path, and fallback-path behavior are defined tightly
  enough to support regression work

---

## Day 6: Build/Option Surface Design

**Title:** Option Design  
**Theme:** Convert the backend contract into an exact build/options wiring
plan without widening the public surface unnecessarily  
**Time estimate:** 9 hours

### Tasks
1. Identify the exact build-system and option surfaces to touch.
2. Decide how the selected backend path is enabled, disabled, or forced.
3. Define the minimum viable public-facing wording and option exposure.
4. Define the minimum viable internal helper or config additions.
5. Record the Day 7-10 touched-file fence and explicit non-goals.

### Deliverables
- Build/options wiring plan
- Exact touched-surface map
- Day 7-10 implementation fence

### Completion Criteria
- The kernel batch has an explicit build/options plan
- The landing is bounded tightly enough to preserve the self-contained default
  path
- Non-goals are clear before header, build, or implementation edits start

---

## Day 7: Kernel Integration Landing Design

**Title:** Landing Design  
**Theme:** Convert the abstraction and option design into an exact touched-file
and proof plan for the first code batch  
**Time estimate:** 10 hours

### Tasks
1. Identify the exact implementation seams to touch in the first batch.
2. Identify the exact proof home for the first backend-aware path:
   - regression tests
   - representative benchmarks
   - optional example touch only if required
3. Define the minimum viable fallback-preserve behavior.
4. Define the minimum viable benchmark signal required to prove the landing.
5. Record the Day 8 code batch fence and Day 9-12 proof follow-through plan.

### Deliverables
- Kernel landing design
- Proof-surface plan
- Day 8-12 implementation/proof fence

### Completion Criteria
- The first code batch has an explicit touched-file plan
- Benchmark proof and regression proof are separated clearly enough to prevent
  accidental scope growth
- The first landing is bounded tightly enough to preserve momentum and safety

---

## Day 8: Kernel Integration Batch I

**Title:** Kernel Batch I  
**Theme:** Land the first backend-aware integration on the highest-value
selected hot path  
**Time estimate:** 12 hours

### Tasks
1. Implement the first selected backend-aware kernel path.
2. Preserve the default self-contained path and explicit fallback behavior.
3. Keep the touched surface inside the Day 7 fence.
4. Add or adjust the minimum required regression coverage for the selected
   path.
5. Run the required code-day validation gate and targeted performance/path
   follow-ons.

### Deliverables
- First backend-aware kernel integration batch
- First bounded fallback-preserve proof
- Validation results for the batch

### Completion Criteria
- At least one high-value hot path is materially modernized
- The default path and fallback path remain stable where promised
- Required validation passes before proof follow-through begins

---

## Day 9: Regression & Safety Proof Design

**Title:** Safety Design  
**Theme:** Define the exact correctness, fallback, and regression expansion
needed after the first kernel landing  
**Time estimate:** 10 hours

### Tasks
1. Audit the landed Day 8 branch state for correctness and fallback seams.
2. Identify the highest-value missing proof around:
   - backend selection
   - fallback preservation
   - error-path behavior
   - output/telemetry truthfulness
3. Decide which proof belongs in:
   - unit/integration tests
   - family-local tests
   - benchmark output verification
4. Fix the exact Day 10-12 proof queue in writing.
5. Record any bounded docs/header follow-through that the landed semantics now
   require.

### Deliverables
- Post-landing safety audit
- Ranked proof-expansion queue
- Day 10-12 follow-through design

### Completion Criteria
- The remaining Sprint 64 queue is smaller and more concrete after the first
  code landing
- The highest-value missing proof is explicit before additional code or docs
  move
- Later follow-through is bounded to real gaps rather than generic polish

---

## Day 10: Build/Option Wiring & Fallback Follow-Through

**Title:** Wiring Follow-Through  
**Theme:** Land the smallest remaining build/config or dispatch follow-through
needed to make the first backend path production-coherent  
**Time estimate:** 11 hours

### Tasks
1. Implement the selected build/options or dispatch follow-through slice.
2. Tighten fallback behavior or configuration truthfulness where required by
   the Day 9 audit.
3. Preserve the default build contract and explicit fallback semantics.
4. Add or adjust regression coverage if the landed wiring changes observable
   behavior.
5. Run the required code-day validation gate and targeted follow-ons.

### Deliverables
- Build/options follow-through batch
- Fallback-truthfulness tightening
- Validation results for the batch

### Completion Criteria
- The first backend-aware path is coherent across implementation and option
  surfaces
- Default-path and fallback-path behavior remain truthful and explicit
- Required validation passes before benchmark proof refresh begins

---

## Day 11: Benchmark Proof Refresh

**Title:** Benchmark Refresh  
**Theme:** Refresh the benchmark proof surface so the new backend-aware path is
measurable without widening into broad governance work  
**Time estimate:** 10 hours

### Tasks
1. Identify the exact benchmark surfaces to refresh for the landed path.
2. Add or adjust the minimum output fields needed to prove:
   - selected path used
   - fallback path preserved
   - performance signal remains comparable
3. Keep the work inside the bounded Sprint 64 benchmark fence.
4. Re-run the selected benchmark proof set.
5. Record representative retained outputs and any caveats.

### Deliverables
- Refreshed benchmark proof surface
- Representative output sample set
- Benchmark-proof artifact

### Completion Criteria
- The new backend-aware path is measurable from the maintained benchmark
  surface
- Benchmark outputs remain interpretable without broad benchmark-policy churn
- The proof set stays inside the bounded Sprint 64 benchmark scope

---

## Day 12: Regression, Docs, and Maintainer Follow-Through

**Title:** Proof Follow-Through  
**Theme:** Close the remaining bounded regression, header, and maintainer-story
gaps after the backend landing  
**Time estimate:** 9 hours

### Tasks
1. Land the remaining bounded regression additions identified in Day 9.
2. Update touched public/header wording only if the landed semantics require
   it.
3. Update maintainer guidance so the backend-aware path and fallback contract
   are documented truthfully.
4. Recheck benchmark/docs/header alignment on the touched surfaces.
5. Record the final residual deferred queue for Sprint 64.

### Deliverables
- Final bounded regression expansion
- Header/docs/maintainer follow-through
- Explicit residual queue

### Completion Criteria
- No important contradiction remains across touched implementation, test, and
  maintained docs surfaces
- Remaining residuals are explicit and future-facing rather than hidden drift
- Sprint 64 is ready for final validation

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full reviewed validation set plus targeted backend-path
follow-ons and capture the retained evidence  
**Time estimate:** 12 hours

### Tasks
1. Run the required full validation set:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm reviewed CMake parity counts and any authoritative timing notes.
3. Re-run the targeted Sprint 64 proof set:
   - direct and CSC proof binaries
   - selected examples
   - refreshed benchmark binaries
4. Capture representative retained outputs for the landed backend-aware path
   and fallback path.
5. Record any non-blocking warnings or caveats without diluting the pass/fail
   baseline.

### Deliverables
- Full validation record
- Targeted proof rerun results
- Representative retained output set

### Completion Criteria
- All required validation passes from the landed Sprint 64 state
- Reviewed parity anchors remain exact or any change is explained concretely
- The branch is ready for closeout from a validated baseline

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Convert the validated Sprint 64 branch into a clear handoff package
for Sprint 65 and the remaining Epic 6 backend work  
**Time estimate:** 8 hours

### Tasks
1. Summarize the landed backend/performance architecture outcomes.
2. Record the preserved compatibility and truthfulness fence:
   - self-contained default build remains authoritative
   - backend-aware path remains bounded and optional
   - fallback correctness remains explicit
3. Record the validated Day 13 baseline and the strongest retained benchmark
   and regression evidence.
4. Define the remaining deferred queue for later backend, packaging, or
   benchmark-governance work.
5. Write the closeout artifact and update working notes for retrospective
   creation.

### Deliverables
- Sprint 64 closeout artifact
- Final working-notes summary
- Explicit Sprint 65 handoff queue

### Completion Criteria
- Sprint 64 ends with one coherent backend-architecture Phase 1 package
- The preserved default-path and fallback-path contract is explicit in writing
- The next sprint inherits a ranked queue instead of a generic backend backlog
