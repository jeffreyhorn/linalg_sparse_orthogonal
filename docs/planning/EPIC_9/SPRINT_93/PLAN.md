# Sprint 93 Plan: Runtime Scalability, Threading & ND Convergence Phase 2

**Sprint Duration:** 14 days  
**Goal:** Reduce the strongest reviewed runtime long pole and tighten the
product-wide threading/runtime model without weakening proof quality. This
sprint implements the Sprint 93 section of
`docs/planning/EPIC_9/PROJECT_PLAN.md`.

**Starting Point:** Sprint 93 begins from:
- the Sprint 90 runtime/measurement contract
- the Sprint 91 compressed-first product convergence baseline
- the Sprint 92 backend split between dense/backend maturity and
  reorder/runtime costs
- a still-visible reviewed runtime long pole centered on the ND/reorder lane
- a still-bounded threading/runtime story that needs sharper public and
  proof-surface interpretation

The strongest Sprint 93 pressure is no longer generic performance work. It is
one bounded runtime-and-threading convergence package centered on:
- reviewed `test_reorder_nd` runtime concentration
- ND/reorder runtime reduction on the highest-value path
- runtime/threading control cleanup where the model is still too loose
- proof-surface rebalancing only where it reduces reviewed-runtime cost
  without weakening trust
- bounded runtime evidence follow-through for the touched seam

**End State:** Sprint 93 leaves behind:
- one fresh reviewed-runtime audit from the live Epic 9 tree
- one explicit threading/runtime and ND convergence contract
- one bounded ND runtime reduction landing
- one bounded runtime-control cleanup landing
- one focused proof-surface rebalancing pass
- one bounded runtime-evidence follow-through package
- one validated Sprint 93 close baseline and Sprint 94 handoff queue

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, which stays within the 14-day cap and remains close
to the Sprint 93 project-plan `~166` hour scope.

---

## Day 1: Sprint 93 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 93 project-plan section into one bounded runtime,
threading, and ND convergence package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 93 section of
   `docs/planning/EPIC_9/PROJECT_PLAN.md` and the Sprint 92 prerequisites it
   depends on.
2. Reconfirm the preserved starting assumptions:
   - Sprint 90 runtime/measurement contract remains authoritative
   - Sprint 92 separated dense/backend concerns from reorder/runtime concerns
   - Sprint 93 targets the reviewed long pole rather than generic speed work
3. Define the Sprint 93 workstreams explicitly:
   - reviewed runtime audit
   - threading/runtime contract design
   - ND runtime reduction batch
   - runtime-control cleanup
   - proof-surface rebalancing
   - runtime evidence follow-through
4. Record the strongest likely Sprint 93 touch surfaces:
   - `tests/test_reorder_nd.c`
   - reorder/graph runtime owners
   - threading/runtime control seams
   - touched benchmark/reporting and support surfaces
5. Open Sprint 93 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 93 scope inventory
- baseline/setup working notes
- explicit workstream map

### Completion Criteria
- Sprint 93 starts from the preserved Epic 9 runtime contract
- the runtime/threading convergence problem is explicit before deeper audit
  work begins
- the initial non-goal fence is visible before design or implementation widens

---

## Day 2: Validation & Maintained Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, benchmark, install/export, and
workflow truth split before runtime work begins  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest implementation-day and substantial-batch gates:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm the maintained owners Sprint 93 must treat as authoritative:
   - reviewed CMake parity
   - reviewed runtime-heavy tests
   - canonical benchmark reporting owner
   - install/export proof owner
   - workflow and support surfaces
3. Recheck the current ownership split so Sprint 93 does not blur:
   - reviewed runtime proof owners
   - reorder/runtime benchmark owners
   - threading/runtime control owners
   - support and workflow wording owners
4. Fix the authoritative rerun set most likely to matter throughout Sprint 93.
5. Record the validation and maintained-surface split in working notes and a
   Day 2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- maintained surface ownership map
- authoritative rerun list

### Completion Criteria
- the strongest local validation contract is explicit before audit findings are
  written
- proof ownership across reviewed tests, benchmarks, install/export checks,
  and support surfaces is fixed in writing
- later Sprint 93 days have no ambiguity about required truth surfaces

---

## Day 3: Reviewed Runtime Audit

**Title:** Runtime Audit  
**Theme:** Reduce the broad runtime problem to one ranked map of the highest-
value reviewed runtime costs  
**Time estimate:** 12 hours

### Tasks
1. Profile the current reviewed baseline around the strongest long pole:
   - `test_reorder_nd`
   - adjacent reorder/graph runtime owners
   - representative benchmark slices
2. Decompose the runtime concentration into actionable cause classes:
   - algorithmic cost
   - proof topology cost
   - setup/repeated-work cost
   - runtime-control complexity
3. Separate:
   - costs that should drive Sprint 93 implementation
   - costs that remain measurement-only
   - costs already materially reduced by prior work
4. Identify the highest-value source, test, benchmark, and support owners tied
   to those costs.
5. Write the ranked Day 3 runtime audit artifact.

### Deliverables
- ranked reviewed-runtime audit artifact
- cause-class split
- strongest owner-surface map

### Completion Criteria
- the broad runtime problem is reduced to one ranked live map
- the strongest long-pole causes are explicit before the first implementation
  fence is frozen
- lower-value runtime concerns are separated from the main Sprint 93 lane

---

## Day 4: Threading & Runtime Contract Design

**Title:** Contract Design  
**Theme:** Decide how much remaining runtime debt is algorithmic, proof-side,
or runtime-control complexity  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 audit against the Sprint 93 project-plan contract.
2. Define the runtime/threading interpretation Sprint 93 should preserve:
   - what counts as algorithmic runtime debt
   - what counts as proof concentration debt
   - what counts as unsafe or unclear runtime-control complexity
3. Decide how far Sprint 93 should go on threading/runtime cleanup:
   - public runtime knobs
   - internal overrides
   - env / compile-time seams
4. Decide what Sprint 93 will not claim:
   - fake broad scaling victory
   - fake threading maturity
   - broad benchmark supremacy
5. Write the Day 4 contract artifact.

### Deliverables
- threading/runtime contract artifact
- debt-classification map
- explicit non-claim fence

### Completion Criteria
- the repo has one explicit runtime/threading contract before code moves
- algorithmic, proof, and runtime-control debt are separated in writing
- later implementation days can stay bounded to the highest-value seam

---

## Day 5: First Implementation Boundary

**Title:** Boundary Freeze  
**Theme:** Fix one bounded first landing so Sprint 93 starts with the
highest-value ND runtime seam instead of generic runtime churn  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 and Day 4 artifacts against the Sprint 93 project-plan
   contract.
2. Decide the required first landing center:
   - one ND/reorder runtime-reduction seam
3. Decide which adjacent surfaces are directly forced support-only follow-
   through and which are explicitly later:
   - runtime-control cleanup
   - proof-surface rebalancing
   - benchmark/reporting widening
   - support/workflow wording
4. Freeze what Sprint 93 will not do in the first batch:
   - broad graph/reorder rewrites
   - generic multithreading everywhere
   - detached benchmark-only tuning
5. Write the Day 5 boundary artifact and update working notes.

### Deliverables
- first-landing boundary artifact
- required-owner vs support-only map
- explicit deferral list

### Completion Criteria
- Sprint 93 has one explicit first implementation fence
- the first landing is small enough to validate cleanly
- later runtime-control, proof, and support work is clearly sequenced

---

## Day 6: ND Runtime Reduction Design

**Title:** Runtime Design  
**Theme:** Define the bounded implementation contract for the touched
reorder/ND runtime seam  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 5 fence against the runtime audit.
2. Decide the exact runtime reduction target:
   - repeated work to remove
   - proof cost to avoid
   - correctness/ordering invariants to preserve
3. Decide the failure boundaries:
   - what must remain deterministic
   - what must remain parity-checked
   - what runtime behavior must remain support-only or non-claimed
4. Decide the strongest directly forced proof owners and benchmark owners.
5. Write the Day 6 implementation-design artifact.

### Deliverables
- runtime-reduction design artifact
- invariant and risk fence
- directly forced proof-owner map

### Completion Criteria
- Day 7 has one exact implementation center
- runtime reduction intent is explicit before code moves
- proof and benchmark obligations are frozen before the landing

---

## Day 7: ND Runtime Reduction Batch

**Title:** Runtime Batch  
**Theme:** Land the highest-value runtime/scalability improvement on the
reviewed ND lane  
**Time estimate:** 12 hours

### Tasks
1. Implement the required Day 6 runtime-reduction landing.
2. Keep the batch bounded to the highest-value ND/reorder seam.
3. Add directly forced proof or benchmark follow-through only where the
   landing requires it.
4. Run the required implementation-day validation.
5. Record the landing, exact changes, and observed runtime effect in working
   notes and a Day 7 artifact.

### Deliverables
- landed ND runtime-reduction batch
- directly forced proof/benchmark updates if needed
- implementation-day validation results

### Completion Criteria
- one real runtime improvement is landed on the reviewed long-pole seam
- validation passes cleanly
- the landed change does not widen into unrelated threading or support work

---

## Day 8: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining Sprint 93 queue after the runtime landing and
choose the strongest second move  
**Time estimate:** 12 hours

### Tasks
1. Re-audit the touched runtime seam after the Day 7 landing.
2. Determine whether the strongest remaining contradiction is now:
   - runtime-control cleanup
   - proof-surface rebalancing
   - benchmark/reporting follow-through
3. Decide whether a second touched implementation batch is still required.
4. Fix the exact Day 9 or Day 10 design center from the post-landing state.
5. Write the Day 8 rerank artifact and update working notes.

### Deliverables
- post-landing rerank artifact
- updated contradiction order
- fixed next design center

### Completion Criteria
- the Day 7 landing is evaluated against live evidence
- the strongest remaining seam is explicit before the next design pass
- Sprint 93 does not drift into stale pre-landing assumptions

---

## Day 9: Runtime-Control Cleanup Design

**Title:** Control Design  
**Theme:** Freeze the bounded cleanup contract for the riskiest
threading/runtime control seam  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 8 rerank against the Sprint 93 contract.
2. Decide the exact runtime-control seam to clean up:
   - env override
   - internal global/thread-local seam
   - unsupported or misleading control path
3. Decide what support-only wording or proof owners are directly forced.
4. Freeze what will remain deferred after this cleanup.
5. Write the Day 9 design artifact.

### Deliverables
- runtime-control cleanup design artifact
- exact implementation center
- support-only follow-through map

### Completion Criteria
- the second implementation contract is explicit before code moves
- the cleanup stays bounded to the riskiest control seam
- later proof and benchmark work remains clearly separated

---

## Day 10: Runtime-Control Cleanup Batch

**Title:** Control Batch  
**Theme:** Remove or bound the riskiest touched runtime-control seam without
weakening the runtime model  
**Time estimate:** 12 hours

### Tasks
1. Implement the required Day 9 cleanup landing.
2. Add directly forced support-only or proof follow-through only where the
   cleanup truly changes interpretation.
3. Run the required implementation-day validation.
4. Confirm the cleaned-up runtime model still matches the preserved Sprint 93
   contract.
5. Record the landing, validation, and remaining open seams in a Day 10
   artifact and working notes.

### Deliverables
- landed runtime-control cleanup batch
- directly forced support/proof updates if needed
- implementation-day validation results

### Completion Criteria
- the touched runtime-control seam is cleaner or more bounded
- validation passes cleanly
- the landing does not widen into generic runtime or API churn

---

## Day 11: Proof-Surface Rebalancing & Runtime Evidence Design

**Title:** Proof Design  
**Theme:** Decide how to reduce reviewed-runtime concentration and what bounded
runtime evidence must be added before closeout  
**Time estimate:** 12 hours

### Tasks
1. Re-read the live post-Day 10 state against the remaining Sprint 93 queue.
2. Decide whether proof-surface rebalancing is needed to reduce runtime
   concentration without weakening trust.
3. Decide the exact bounded runtime-evidence follow-through shape:
   - reviewed reruns
   - benchmark slices
   - reporting artifact changes
4. Freeze the required Day 12 center and any directly forced support-only
   follow-through.
5. Write the Day 11 design artifact.

### Deliverables
- proof-surface and runtime-evidence design artifact
- fixed Day 12 center
- frozen reporting/rerun shape

### Completion Criteria
- the remaining Sprint 93 gap is explicit before the final implementation or
  support batch
- proof rebalancing and runtime evidence are separated from generic doc churn
- the Day 12 queue is exact and bounded

---

## Day 12: Proof/Evidence Follow-Through Batch

**Title:** Evidence Batch  
**Theme:** Land the bounded proof-surface rebalancing or runtime-evidence
follow-through required by the live Sprint 93 state  
**Time estimate:** 12 hours

### Tasks
1. Implement the required Day 11 batch.
2. Keep the landing bounded to:
   - proof-surface rebalancing actually needed
   - runtime evidence or reporting actually needed
3. Run the required validation based on the touched surfaces.
4. Reconfirm the final Sprint 93 owner split and freeze the Day 13 queue.
5. Record the landing and owner split in working notes and a Day 12 artifact.

### Deliverables
- landed proof/evidence follow-through batch
- final owner split for Sprint 93
- frozen Day 13 validation queue

### Completion Criteria
- the remaining runtime proof/evidence seam is materially closed
- validation passes cleanly
- the final validation queue is explicit before the full sweep

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Validate the full Sprint 93 runtime/threading/ND package from the
live branch state  
**Time estimate:** 12 hours

### Tasks
1. Run the full Sprint 93 validation queue from the frozen Day 12 plan.
2. Rerun the strongest reviewed runtime and touched proof owners explicitly.
3. Rerun the bounded runtime evidence/reporting surfaces touched by Sprint 93.
4. Record exact pass/fail totals, runtime notes, and any residual non-blocking
   observations.
5. Write the Day 13 validation artifact and update working notes.

### Deliverables
- full validation-sweep artifact
- exact reviewed/runtime evidence results
- explicit residual non-blocking notes

### Completion Criteria
- the Sprint 93 package passes the required validation queue
- runtime and proof owners are rechecked from the live branch state
- any remaining long-pole or threading notes are written as bounded residuals

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 93 from the validated baseline and hand off the next
Epic 9 queue cleanly  
**Time estimate:** 12 hours

### Tasks
1. Summarize the final Sprint 93 outcomes against the original project-plan
   section.
2. Record what contradictions were closed, partially reduced, or deliberately
   deferred.
3. Freeze the validated baseline and residual runtime/threading queue.
4. Fix the next-sprint handoff order for capability widening and later Epic 9
   work.
5. Write the Day 14 closeout/handoff artifact and final working-notes entry.

### Deliverables
- Sprint 93 closeout artifact
- validated final baseline summary
- fixed Sprint 94 handoff queue

### Completion Criteria
- Sprint 93 closes from a validated branch state
- the runtime/threading/ND results are summarized in one bounded final reading
- the next queue is explicit enough to start the next sprint without
  re-auditing Sprint 93 intent
