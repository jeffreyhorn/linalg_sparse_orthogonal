# Sprint 62 Plan: Direct-Solver Usability & Lifecycle Coherence

**Sprint Duration:** 14 days  
**Goal:** Reduce mutable-matrix surprise and make the one-shot and explicit
lifecycle direct-solver stories more coherent. This sprint implements the
Sprint 62 section of `docs/planning/EPIC_6/PROJECT_PLAN.md`.

**Starting Point:** Sprint 61 closed with the first Phase 1 typed
configuration package landed and validated:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- the repeated-run workflow fence is already fixed and must not widen
- the highest-value analysis/reorder configuration seam now has a shipped
  typed front door
- the next Epic 6 priority is direct-solver usability and lifecycle coherence
  rather than further configuration-first work

The next highest-value work is no longer broad productization audit or
configuration-surface debate. It is a bounded usability sprint centered on
one-shot direct-solver mutation surprises, wrapper/lifecycle coherence,
helper-state hardening, higher-signal docs/examples, regression proof, and
validated closeout.

**End State:** Sprint 62 leaves behind one coherent direct-solver usability
and lifecycle package:
- a ranked live audit of one-shot direct-solver usability pain points
- explicit bounded design for lifecycle/wrapper convergence
- landed hardening on the highest-value one-shot direct paths
- clearer relationship between one-shot and explicit lifecycle direct usage
- strengthened direct lifecycle, cancellation, and mutation-surprise proof
- updated docs/examples that match the refined direct-solver story
- full validation and closeout from the landed state

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
150 hours, staying within the Sprint 62 estimate and below the 168-hour limit.

---

## Day 1: Sprint 62 Scope Audit & Direct-Usability Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 62 project-plan scope plus the Sprint 60-61 frozen
contracts into a bounded direct-usability implementation map  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Sprint 62 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 61 retrospective, and
   the strongest Sprint 61 closeout artifacts.
2. Reconfirm the preserved Sprint 62 constraints:
   - no reopening the repeated-run workflow fence
   - no broad configuration-surface rewrite in the same batch
   - no packaging/platform widening disguised as usability work
   - no fake convergence between one-shot and lifecycle direct APIs that
     breaks explicit ownership or compatibility
3. Define the Sprint 62 workstreams explicitly:
   - direct-usability audit
   - lifecycle/wrapper coherence design
   - one-shot hardening
   - explicit lifecycle convergence
   - example/docs adoption
   - regression expansion
   - validation and closeout
4. Record the strongest likely Sprint 62 touch surfaces:
   - public direct-solver headers
   - one-shot wrapper and factorization entry seams
   - lifecycle and factor-state implementation seams
   - highest-value direct proof surfaces
5. Open Sprint 62 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 62 scope inventory
- Direct-usability baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 62 starts from the Sprint 60-61 frozen contracts rather than
  reopening direct-usage target-definition debates
- The implementation workstreams are explicit before deeper investigation
  begins
- The sprint non-goal fence is fixed before design or code edits land

---

## Day 2: Validation Baseline & Direct-Path Rerun Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed baseline and rerun set that Sprint 62 direct
usability changes must preserve  
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
3. Reconfirm the stronger default for substantial direct-control or
   architecture-sensitive work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Sprint 62:
   - direct lifecycle proofs
   - CSC direct solver proofs
   - representative direct examples and repeated-run benchmarks
   - adjacent iterative/eigensolver regression sentinels that should not drift
5. Record the authoritative validation split for:
   - docs-only days
   - bounded code-touching days
   - substantial direct-control days

### Deliverables
- Refreshed validation notes
- Sprint 62 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 62 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before implementation work begins
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Direct One-Shot Usability Audit

**Title:** Usability Audit  
**Theme:** Re-rank the live one-shot direct-solver pain points by mutation
surprise, lifecycle ambiguity, and caller-risk  
**Time estimate:** 12 hours

### Tasks
1. Inventory the one-shot direct-solver public surfaces and their mutation or
   ownership implications:
   - LU
   - Cholesky
   - LDL^T
   - QR where relevant to shared lifecycle expectations
2. Classify each direct usability seam by:
   - mutable-matrix surprise risk
   - lifecycle story ambiguity
   - cancellation/progress interaction risk
   - wrapper/helper hardening opportunity
3. Rank the strongest Sprint 62 problems by:
   - caller-facing value
   - compatibility sensitivity
   - implementation leverage
   - regression burden
4. Identify the strongest first hardening cut line for Sprint 62.
5. Write the audit artifact with the explicit ranked direct-usability map.

### Deliverables
- Live direct-usability inventory
- Ranked Sprint 62 candidate list
- Mutation/lifecycle pain-point classification draft

### Completion Criteria
- The broad “direct usability needs work” claim is reduced to a concrete
  ranked list
- The strongest Phase 1 hardening targets are explicit before design begins
- Day 4 can proceed from a real migration target instead of generic cleanup
  goals

---

## Day 4: Lifecycle Coherence Design & Safety Contract

**Title:** Coherence Design  
**Theme:** Define the bounded lifecycle/wrapper hardening model and exact
preserved compatibility rules before code changes begin  
**Time estimate:** 12 hours

### Tasks
1. Design the bounded Sprint 62 hardening model for the highest-value
   one-shot direct paths.
2. Define exact preserved rules across:
   - one-shot wrappers
   - explicit `analysis` / `factors` lifecycle
   - progress/cancellation behavior
   - matrix ownership and copy discipline
3. Decide which improvements belong in:
   - public wrapper behavior
   - internal factor-state hardening
   - lifecycle helper plumbing
   - docs/examples only
4. Define the explicit compatibility behavior:
   - what caller-visible semantics stay unchanged
   - what usability signals become clearer without API widening
   - what lifecycle convergence is not justified in Sprint 62
5. Record the landing fence for the first implementation batch.

### Deliverables
- Lifecycle coherence design artifact
- Explicit safety/compatibility contract
- Phase 1 landing fence

### Completion Criteria
- The future direct-usage behavior is explicit before implementation edits
- Public and internal ownership are separated clearly enough to prevent drift
- Compatibility behavior is defined tightly enough to support regression work

---

## Day 5: Public/Header and Internal Landing Design

**Title:** Landing Design  
**Theme:** Convert the lifecycle design into a precise touched-file and
API/impl boundary plan  
**Time estimate:** 10 hours

### Tasks
1. Identify the exact public header and docs surfaces to widen or normalize.
2. Identify the exact internal implementation seams to touch first:
   - one-shot direct wrappers
   - factor-state and lifecycle helper seams
   - cancellation/progress-safe cleanup paths
3. Define the minimum viable Sprint 62 public-surface adjustments.
4. Define the minimum viable internal helper additions or refactors.
5. Record the Day 6-8 code landing boundary and explicit non-goals.

### Deliverables
- Header/internal landing design
- Exact touched-surface map
- Day 6-8 implementation fence

### Completion Criteria
- The implementation batch has an explicit touched-file plan
- The first landing is bounded tightly enough to preserve momentum and safety
- Non-goals are clear before public-header or implementation edits start

---

## Day 6: One-Shot Hardening Batch I

**Title:** Hardening I  
**Theme:** Land the first bounded direct one-shot usability reductions on the
highest-value path  
**Time estimate:** 12 hours

### Tasks
1. Implement the first selected one-shot hardening slice.
2. Reduce the highest-value mutable-matrix or lifecycle-surprise behavior
   where justified.
3. Preserve existing default compatibility where explicit behavior changes are
   not promised.
4. Add or adjust regression coverage for the selected path.
5. Run the required code-day validation gate and targeted direct follow-ons.

### Deliverables
- First one-shot direct hardening batch
- First bounded direct-usability reduction
- Validation results for the batch

### Completion Criteria
- At least one high-value direct usability pain point is materially reduced
- Existing default behavior remains stable where promised
- Required validation passes before Day 7 proceeds

---

## Day 7: One-Shot Hardening Batch II

**Title:** Hardening II  
**Theme:** Complete the bounded first usability batch and tighten the touched
control flow around it  
**Time estimate:** 12 hours

### Tasks
1. Land the remaining bounded one-shot hardening work selected in Day 4.
2. Tighten defaulting, cleanup, and state-preservation handling around the
   new direct path.
3. Normalize any touched public/internal naming or comments needed for
   clarity.
4. Add or expand regression coverage around the landed path.
5. Run the required gate and any targeted reruns driven by the touched
   surfaces.

### Deliverables
- Completed first direct hardening batch
- Expanded regression support
- Validation results for the second batch

### Completion Criteria
- The selected one-shot Phase 1 surface is fully landed
- Cleanup and lifecycle behavior are no longer ambiguous on the landed path
- Validation remains clean after the full first integration batch

---

## Day 8: Lifecycle/Wrapper Convergence Audit

**Title:** Convergence Audit  
**Theme:** Re-audit the remaining wrapper/lifecycle seams after the first
hardening landing  
**Time estimate:** 9 hours

### Tasks
1. Re-read the landed control flow after Days 6-7.
2. Identify the next strongest one-shot versus explicit-lifecycle mismatch
   still present.
3. Separate:
   - changes that should still move in Sprint 62
   - changes that should stay compatibility-only for now
   - changes that should defer to a later sprint
4. Define the exact bounded Day 9 target.
5. Record any new risks exposed by the landed direct path.

### Deliverables
- Post-landing lifecycle/wrapper audit
- Ranked next convergence slice
- Day 9 landing target

### Completion Criteria
- The post-Day-7 queue is smaller and more concrete than the original Sprint
  62 scope
- The next integration slice is explicit before more code moves
- No accidental broadening of scope is required to proceed

---

## Day 9: Lifecycle/Wrapper Convergence Design

**Title:** Convergence Design  
**Theme:** Convert the remaining justified lifecycle/wrapper improvements into
a bounded implementation plan  
**Time estimate:** 10 hours

### Tasks
1. Define the exact wrapper/lifecycle convergence subset to move.
2. Design the public/internal plumbing for that subset.
3. Define exact preserved semantics and compatibility behavior for the Day 10
   batch.
4. Confirm which usability issues remain explicitly deferred.
5. Record the Day 10 code landing fence and regression obligations.

### Deliverables
- Lifecycle/wrapper convergence design
- Explicit Day 10 implementation fence
- Deferred-usability list

### Completion Criteria
- The next code batch is precise rather than generic
- Compatibility behavior is explicit before the batch lands
- Deferred usability issues are named rather than silently dropped

---

## Day 10: Lifecycle/Wrapper Convergence Batch

**Title:** Hardening III  
**Theme:** Land the bounded lifecycle/wrapper coherence changes selected in
Day 9  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 9 lifecycle/wrapper convergence batch.
2. Preserve stable defaults on untouched caller paths.
3. Land the bounded helper/state-hardening still justified.
4. Add or adjust regression coverage for the new path.
5. Run the required code-day validation gate and targeted direct follow-ons.

### Deliverables
- Lifecycle/wrapper convergence batch
- Helper/state-hardening updates
- Validation results for the batch

### Completion Criteria
- The selected direct-lifecycle coherence changes are available on the landed
  path
- Stable defaults and backward behavior remain intact where promised
- Required validation passes before closeout work begins

---

## Day 11: Regression Sweep & Compatibility Tightening

**Title:** Regression Sweep  
**Theme:** Tighten the direct-lifecycle proof surface and explicitly verify
the refined usability contract  
**Time estimate:** 11 hours

### Tasks
1. Review the full landed direct usability path:
   - one-shot wrapper usage
   - explicit lifecycle usage
   - cancellation/progress behavior
   - preserved mutation/copy expectations
2. Add or tighten regression tests around:
   - lifecycle coherence
   - mutation-surprise reduction
   - stable default behavior
   - cancellation or error-preservation behavior where touched
3. Remove or clarify any stale wording/comments around the old one-shot story.
4. Record the post-landing compatibility state.
5. Run the required validation gate and targeted reruns.

### Deliverables
- Direct-lifecycle regression expansion
- Compatibility-layer cleanup
- Post-landing usability notes

### Completion Criteria
- The refined direct usage story is explicitly proven
- Remaining surprising behavior is bounded and intentional
- No stale wording implies broader or different direct behavior than shipped

---

## Day 12: Example & Documentation Adoption

**Title:** Docs Follow-Through  
**Theme:** Align caller-facing and maintainer-facing surfaces with the refined
direct usability and lifecycle story  
**Time estimate:** 9 hours

### Tasks
1. Update the highest-value public docs and examples for the landed direct
   usage story.
2. Update maintainer guidance around:
   - one-shot versus explicit lifecycle usage
   - preserved compatibility behavior
   - preferred caller path for repeated direct usage
3. Update benchmark/example references only if needed for truthfulness.
4. Record the exact residual deferred direct-usability queue.
5. Run docs/workflow sanity checks against the touched surfaces.

### Deliverables
- Updated public/maintainer direct-usage docs
- Example-story alignment
- Explicit deferred direct-usability queue

### Completion Criteria
- The highest-value docs/examples match the landed direct behavior
- Maintainers have one clear home for the residual direct-usability queue
- No caller-facing contradiction remains on the touched Sprint 62 surfaces

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Revalidate the full Sprint 62 landed state from the strongest
reviewed baseline  
**Time estimate:** 12 hours

### Tasks
1. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Recheck reviewed CMake parity counts / Makefile-CMake parity / full
   reviewed CMake `ctest`.
3. Run the targeted Sprint 62 follow-ons:
   - direct lifecycle proofs
   - CSC direct proofs
   - representative direct examples
   - representative repeated-run and direct benchmark drivers
   - adjacent iterative/eigensolver sentinels if still part of the rerun set
4. Record representative retained behavior/results.
5. Capture non-blocking warnings honestly if any remain.

### Deliverables
- Full validation artifact
- Final retained direct-usability measurements
- Explicit validated close baseline

### Completion Criteria
- The strongest local reviewed baseline passes from the landed Sprint 62 tree
- The targeted rerun set also passes from the landed state
- The sprint can close from one explicit validated baseline

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Package Sprint 62 into a clean validated direct-usability handoff
for the next Epic 6 implementation sprint  
**Time estimate:** 8 hours

### Tasks
1. Summarize the Sprint 62 shipped direct-usability and lifecycle outcomes.
2. Record the preserved compatibility fence and explicit deferred queue.
3. Reconfirm whether `PROJECT_PLAN.md` needs any correction from the landed
   sprint state.
4. Write the final Day 14 working-notes synthesis.
5. Write the Day 14 closeout artifact and final working-notes summary.

### Deliverables
- Sprint 62 closeout artifact
- Final Sprint 62 handoff notes
- Explicit next-sprint starting queue

### Completion Criteria
- Sprint 62 closes from the Day 13 validated baseline
- The landed usability story and deferred queue are explicit
- The next Epic 6 sprint can start without reopening Sprint 62 scope or
  validation decisions
