# Sprint 63 Plan: Direct-Lifecycle Uniformity & CSC/LU Follow-Through

**Sprint Duration:** 14 days  
**Goal:** Reduce the remaining internal heterogeneity behind the public direct
lifecycle model, focusing on LU and the hardest CSC seams. This sprint
implements the Sprint 63 section of
`docs/planning/EPIC_6/PROJECT_PLAN.md`.

**Starting Point:** Sprint 62 closed with the first bounded direct-usability
package landed and validated:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- one-shot direct wrappers remain first-class/default peer entry points
- the explicit repeated-run direct lifecycle remains the canonical reuse path
- the highest-value reordered LU and Cholesky cancel/failure seams now preserve
  the caller matrix
- the next Epic 6 priority is reducing remaining direct-lifecycle
  heterogeneity, especially around LU and CSC-backed direct paths

The next highest-value work is no longer broad direct-usability audit or
configuration-surface expansion. It is a bounded lifecycle-uniformity sprint
centered on LU lifecycle follow-through, CSC repeated-run coherence,
solve/refactor semantics alignment, proof refresh, and validated closeout.

**End State:** Sprint 63 leaves behind one more uniform direct repeated-run
implementation package:
- a ranked live audit of the remaining uneven direct-lifecycle seams
- explicit bounded design for LU and CSC follow-through
- landed lifecycle and CSC uniformity reductions on the highest-value internal
  paths
- clearer solve/refactor semantics across the touched direct families
- refreshed proof, benchmark, and example surfaces where lifecycle behavior
  matters
- full validation and closeout from the landed state

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
144 hours, staying within the Sprint 63 estimate and below the 168-hour limit.

---

## Day 1: Sprint 63 Scope Audit & Lifecycle Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 63 project-plan scope plus the Sprint 62 validated
close into a bounded direct-lifecycle implementation map  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Sprint 63 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 62 retrospective, and
   the strongest Sprint 62 closeout artifacts.
2. Reconfirm the preserved Sprint 63 constraints:
   - no reopening the repeated-run workflow fence
   - no broad configuration-surface rewrite in the same batch
   - no packaging/platform widening disguised as lifecycle work
   - no fake convergence that erases real family-local ownership or
     cancellation semantics
3. Define the Sprint 63 workstreams explicitly:
   - internal path audit
   - LU lifecycle follow-through
   - CSC repeated-run uniformity
   - solve/refactor semantics alignment
   - benchmark/example proof refresh
   - regression expansion
   - validation and closeout
4. Record the strongest likely Sprint 63 touch surfaces:
   - public direct-lifecycle headers and docs
   - LU repeated-run and factor-state implementation seams
   - CSC-backed Cholesky and LDL^T lifecycle seams
   - highest-value lifecycle proof and benchmark surfaces
5. Open Sprint 63 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 63 scope inventory
- Direct-lifecycle baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 63 starts from the Sprint 62 validated close rather than reopening
  usability or configuration target-definition debates
- The implementation workstreams are explicit before deeper investigation
  begins
- The sprint non-goal fence is fixed before design or code edits land

---

## Day 2: Validation Baseline & Lifecycle Rerun Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed baseline and rerun set that Sprint 63
lifecycle changes must preserve  
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
3. Reconfirm the stronger default for substantial direct-lifecycle or CSC
   architecture-sensitive work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Sprint 63:
   - direct lifecycle proofs
   - CSC direct solver proofs
   - representative direct examples and factor-many benchmarks
   - adjacent iterative/eigensolver regression sentinels that should not drift
5. Record the authoritative validation split for docs-only, bounded code-day,
   and substantial lifecycle/CSC days.

### Deliverables
- Refreshed validation notes
- Sprint 63 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 63 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before implementation work begins
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Internal Path Audit

**Title:** Path Audit  
**Theme:** Re-rank the remaining direct repeated-run implementation seams by
caller-risk, lifecycle inconsistency, and CSC leverage  
**Time estimate:** 12 hours

### Tasks
1. Inventory the remaining uneven direct-lifecycle paths across:
   - LU
   - Cholesky
   - LDL^T
   - CSC-backed repeated-run analysis/factors flows
2. Classify each seam by:
   - public story drift
   - factor-state/result semantics drift
   - CSC dispatch/lifecycle asymmetry
   - regression burden
3. Rank the strongest Sprint 63 problems by:
   - caller-facing value
   - compatibility sensitivity
   - implementation leverage
   - proof cost
4. Identify the strongest first follow-through cut line for Sprint 63.
5. Write the audit artifact with the explicit ranked lifecycle map.

### Deliverables
- Live direct-lifecycle inventory
- Ranked Sprint 63 candidate list
- Lifecycle/CSC asymmetry classification draft

### Completion Criteria
- The broad “direct lifecycle is still heterogeneous” claim is reduced to a
  concrete ranked list
- The strongest LU and CSC follow-through targets are explicit before design
  begins
- Day 4 can proceed from a real migration target instead of generic cleanup
  goals

---

## Day 4: Lifecycle Uniformity Design & Safety Contract

**Title:** Uniformity Design  
**Theme:** Define the bounded lifecycle-follow-through model and preserved
compatibility rules before code changes begin  
**Time estimate:** 11 hours

### Tasks
1. Design the bounded Sprint 63 follow-through model for the highest-value
   LU and CSC lifecycle seams.
2. Define exact preserved rules across:
   - one-shot wrappers versus explicit lifecycle entry points
   - analysis/factors ownership
   - solve/refactor result semantics
   - CSC dispatch and state retention
3. Decide which improvements belong in:
   - public lifecycle behavior
   - internal factor-state hardening
   - CSC dispatch/helper plumbing
   - docs/examples only
4. Define the explicit compatibility behavior:
   - what caller-visible semantics stay unchanged
   - what lifecycle behavior becomes more uniform
   - what deeper convergence is not justified in Sprint 63
5. Record the landing fence for the first implementation batch.

### Deliverables
- Lifecycle uniformity design artifact
- Explicit safety/compatibility contract
- Phase 1 landing fence

### Completion Criteria
- The future direct-lifecycle behavior is explicit before implementation edits
- Public and internal ownership are separated clearly enough to prevent drift
- Compatibility behavior is defined tightly enough to support regression work

---

## Day 5: Public/Header and Internal Landing Design

**Title:** Landing Design  
**Theme:** Convert the lifecycle design into a precise touched-file and
API/impl boundary plan  
**Time estimate:** 10 hours

### Tasks
1. Identify the exact public header and docs surfaces to normalize.
2. Identify the exact internal implementation seams to touch first:
   - LU lifecycle helpers and factor-state behavior
   - CSC analysis/factors dispatch seams
   - solve/refactor result-state publication paths
3. Define the minimum viable Sprint 63 public-surface adjustments.
4. Define the minimum viable internal helper additions or refactors.
5. Record the Day 6-10 code landing boundary and explicit non-goals.

### Deliverables
- Header/internal landing design
- Exact touched-surface map
- Day 6-10 implementation fence

### Completion Criteria
- The implementation batch has an explicit touched-file plan
- The first landing is bounded tightly enough to preserve momentum and safety
- Non-goals are clear before public-header or implementation edits start

---

## Day 6: LU Lifecycle Follow-Through Batch I

**Title:** LU Follow-Through I  
**Theme:** Land the first bounded lifecycle-uniformity reductions on the
highest-value LU repeated-run seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the first selected LU lifecycle follow-through slice.
2. Reduce the highest-value factor-state or result-semantics inconsistency
   where justified.
3. Preserve existing compatibility where explicit behavior changes are not
   promised.
4. Add or adjust regression coverage for the selected LU path.
5. Run the required code-day validation gate and targeted direct follow-ons.

### Deliverables
- First LU lifecycle follow-through batch
- First bounded lifecycle-uniformity reduction
- Validation results for the batch

### Completion Criteria
- At least one high-value LU lifecycle inconsistency is materially reduced
- Existing default behavior remains stable where promised
- Required validation passes before Day 7 proceeds

---

## Day 7: CSC Repeated-Run Uniformity Batch I

**Title:** CSC Uniformity I  
**Theme:** Land the first bounded CSC lifecycle/dispatch uniformity reduction
behind the public analysis/factors path  
**Time estimate:** 12 hours

### Tasks
1. Implement the first selected CSC repeated-run uniformity slice.
2. Reduce the highest-value CSC lifecycle or dispatch asymmetry where
   justified.
3. Preserve the public direct-lifecycle boundary and existing reviewed
   behavior.
4. Add or adjust regression coverage for the selected CSC path.
5. Run the required code-day validation gate and targeted CSC/direct follow-ons.

### Deliverables
- First CSC uniformity batch
- First bounded CSC lifecycle reduction
- Validation results for the batch

### Completion Criteria
- At least one high-value CSC lifecycle asymmetry is materially reduced
- Public lifecycle expectations stay intact
- Required validation passes before the sprint moves to follow-through audit

---

## Day 8: Post-Landing Audit & Residual Queue Re-Rank

**Title:** Follow-Through Audit  
**Theme:** Reassess the remaining Sprint 63 lifecycle queue after the first
LU/CSC landings  
**Time estimate:** 8 hours

### Tasks
1. Audit the landed Day 6-7 behavior against the Day 4 contract.
2. Re-rank the remaining lifecycle, CSC, and solve/refactor semantics seams.
3. Decide what must still land in Sprint 63 versus what should remain
   consciously deferred.
4. Fix the exact Day 9-10 target based on the landed branch shape.
5. Record the residual queue and updated implementation boundary.

### Deliverables
- Post-landing audit artifact
- Updated ranked residual queue
- Day 9-10 narrowed target

### Completion Criteria
- The remaining Sprint 63 queue is smaller and more concrete than the opening
  audit implied
- The next batch is selected from the landed branch state, not from stale
  pre-landing assumptions
- Deferred items are explicit rather than silently dropped

---

## Day 9: Solve/Refactor Semantics Alignment Design

**Title:** Semantics Design  
**Theme:** Design the bounded solve/refactor result-state alignment batch
before the second implementation pass  
**Time estimate:** 10 hours

### Tasks
1. Identify the strongest remaining solve/refactor semantics mismatches across
   touched direct families.
2. Decide what alignment belongs in:
   - result/error propagation
   - factor retention or invalidation
   - analysis/factors consistency checks
   - docs-only clarification
3. Define the exact touched-file fence for the second implementation pass.
4. Define required proof additions for semantics alignment.
5. Record the Day 10 landing contract and explicit non-goals.

### Deliverables
- Solve/refactor semantics design artifact
- Exact Day 10 landing fence
- Regression-expansion plan for the second batch

### Completion Criteria
- The second implementation batch has an explicit semantics target
- Proof burden is known before code changes resume
- The batch remains bounded to Sprint 63 goals

---

## Day 10: Solve/Refactor Semantics Alignment Batch

**Title:** Semantics Batch  
**Theme:** Land the second bounded lifecycle-uniformity slice on the
strongest remaining solve/refactor semantics seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the selected solve/refactor semantics alignment slice.
2. Normalize the strongest remaining result-state inconsistency where
   justified.
3. Preserve caller-visible compatibility where no explicit change was promised.
4. Add or adjust regression coverage for the aligned semantics.
5. Run the required code-day validation gate and targeted follow-ons.

### Deliverables
- Solve/refactor semantics alignment batch
- Additional lifecycle-uniformity reduction
- Validation results for the batch

### Completion Criteria
- The strongest remaining result-state inconsistency is materially reduced
- Regression coverage proves the intended lifecycle behavior
- Required validation passes before Day 11 proceeds

---

## Day 11: Regression Expansion & Compatibility Sweep

**Title:** Compatibility Sweep  
**Theme:** Tighten proof and compatibility coverage around the landed LU and
CSC lifecycle changes  
**Time estimate:** 11 hours

### Tasks
1. Expand the highest-signal regression proof for the touched lifecycle and
   CSC seams.
2. Sweep touched public headers and comments for stale semantics wording.
3. Tighten compatibility/default behavior where the landed code requires more
   explicit proof.
4. Keep proof growth bounded to the highest-signal direct homes.
5. Run the required code-day validation gate and targeted lifecycle follow-ons.

### Deliverables
- Expanded lifecycle/CSC regression coverage
- Compatibility-layer follow-through
- Validation results for the sweep

### Completion Criteria
- The landed Sprint 63 behavior is explicitly regression-proven on the
  highest-value paths
- Touched docs/header commentary matches the shipped implementation
- Required validation passes before docs and benchmark/example follow-through

---

## Day 12: Benchmark, Example, and Docs Follow-Through

**Title:** Proof Follow-Through  
**Theme:** Align the highest-signal docs, examples, and benchmarks with the
landed lifecycle-uniformity story  
**Time estimate:** 8 hours

### Tasks
1. Update the highest-signal docs and header wording that should reflect the
   landed Sprint 63 behavior.
2. Refresh representative example or benchmark wording where lifecycle or CSC
   interpretation changed materially.
3. Keep the touched surfaces bounded to the sprint’s highest-signal adoption
   path.
4. Record any consciously deferred documentation or benchmark density cleanup.
5. Run targeted sanity checks appropriate to the touched surface.

### Deliverables
- Docs/header/example/benchmark follow-through
- Updated maintainer and adoption wording
- Deferred wording queue, if any

### Completion Criteria
- The public and maintainer story matches the shipped Sprint 63 behavior
- The batch stays bounded and does not widen into general docs cleanup
- The branch is ready for final validation

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Revalidate the full Sprint 63 landed state from the strongest
reviewed baseline  
**Time estimate:** 11 hours

### Tasks
1. Run the full required validation gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Run the targeted Sprint 63 rerun set:
   - direct lifecycle proofs
   - CSC direct proofs
   - representative examples
   - representative direct and adjacent workflow benchmarks
3. Capture representative retained outputs where they carry lifecycle or CSC
   signal.
4. Record any non-blocking validation notes.
5. Confirm branch cleanliness and final validation status.

### Deliverables
- Full validation artifact
- Final Sprint 63 metrics
- Clean validated branch state

### Completion Criteria
- The strongest local reviewed baseline passes end to end
- The targeted Sprint 63 follow-ons pass
- Sprint 63 is ready to close from a validated branch state

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Package Sprint 63 into a clean validated lifecycle-uniformity
handoff for the next Epic 6 implementation sprint  
**Time estimate:** 8 hours

### Tasks
1. Summarize the Sprint 63 shipped LU and CSC lifecycle outcomes.
2. Record the preserved compatibility fence and explicit deferred queue.
3. Reconfirm whether `PROJECT_PLAN.md` needs any correction from the landed
   sprint state.
4. Write the final Day 14 working-notes synthesis.
5. Write the Day 14 closeout artifact and final working-notes summary.

### Deliverables
- Sprint 63 closeout artifact
- Final Sprint 63 handoff notes
- Explicit next-sprint starting queue

### Completion Criteria
- Sprint 63 closes from the Day 13 validated baseline
- The landed lifecycle-uniformity story and deferred queue are explicit
- The next Epic 6 sprint can start without reopening Sprint 63 scope or
  validation decisions
