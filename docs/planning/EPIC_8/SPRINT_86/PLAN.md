# Sprint 86 Plan: Reordering Scalability & Reviewed Runtime Convergence

**Sprint Duration:** 14 days  
**Goal:** Reduce the strongest reviewed runtime long pole and improve
reordering / nested-dissection scalability after Sprint 85’s bounded
maintainability cleanup. This sprint implements the Sprint 86 section of
`docs/planning/EPIC_8/PROJECT_PLAN.md`.

**Starting Point:** Sprint 85 closed from a validated maintainability baseline
with clearer ownership on the iterative lane, the dense LDL^T backend seam,
and the largest retained Cholesky CSC registration block. The strongest
remaining first-tier Epic 8 contradiction is now reviewed runtime and
reordering-scalability cost:
- `test_reorder_nd` still dominates the reviewed runtime long pole
- the strongest current runtime pressure is concentrated on the ND /
  reordering proof lane rather than on generic solver coverage
- Sprint 85 already reduced the maintainability ambiguity that would have made
  runtime work harder to review
- Sprint 86 should not reopen generic source-decomposition work as its first
  implementation center
- the strongest local reviewed baseline remains `make quality-review-full`

The highest-value Sprint 86 work is therefore not generic “make tests faster.”
It is one bounded reviewed-runtime and scalability-modernization package that:
- audits the reviewed runtime long pole from the current tree
- separates algorithmic, fixture-organization, and reviewed-surface causes
- lands one bounded ND / reorder runtime reduction batch
- lands one bounded proof-surface rebalancing batch where runtime
  concentration is avoidable without weakening correctness ownership
- adds bounded benchmark / comparison evidence for the touched runtime seam
- reconciles the reviewed validation path and runtime expectations with the
  landed change

**End State:** Sprint 86 leaves behind:
- a refreshed reviewed-runtime and ND-scalability cause map
- one explicit algorithm / proof runtime design contract
- one landed runtime or scalability reduction batch on the ND lane
- one landed proof-surface rebalancing batch
- one bounded benchmark / comparison follow-through package
- one CI / reviewed-path alignment package
- one validated closeout baseline and later Epic 8 handoff

**Time budget:** Each day is capped at 12 hours as requested. Because that cap
allows at most `168` hours across 14 days, this day-by-day plan totals `168`
hours rather than the higher project-plan estimate of `~176` hours, while
preserving the Sprint 86 scope and ordering.

---

## Day 1: Sprint 86 Scope Audit & Reviewed Runtime Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 86 project-plan section and Sprint 85 closeout into
one bounded reviewed-runtime execution package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 86 section of
   `docs/planning/EPIC_8/PROJECT_PLAN.md`, the Sprint 85 closeout artifact, and
   the Sprint 85 retrospective.
2. Reconfirm the preserved Sprint 86 starting assumptions:
   - Sprint 80 already fixed the performance-contract premise
   - Sprint 85 already reduced the strongest ownership ambiguity on the
     touched hotspot lanes
   - Sprint 86 should not reopen generic maintainability decomposition as its
     first implementation center
   - Sprint 86 should not weaken correctness proof quality while chasing
     runtime wins
3. Define the Sprint 86 workstreams explicitly:
   - reviewed runtime audit
   - algorithm / proof design
   - ND runtime reduction
   - proof-surface rebalancing
   - benchmark / comparison follow-through
   - CI / reviewed-path alignment
   - validation and closeout
4. Record the strongest likely Sprint 86 touch surfaces:
   - `tests/test_reorder_nd.c`
   - relevant reorder / ND implementation owners
   - reviewed proof-owner tests and representative benchmark surfaces
   - reviewed-path workflow or maintainer wording only if truly forced
5. Open Sprint 86 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 86 scope inventory
- runtime/scalability workstream map
- starting working-notes baseline

### Completion Criteria
- Sprint 86 starts from the validated Sprint 85 end state
- the first runtime contradiction is explicit before deeper audit begins
- the non-goal fence is visible before design or implementation work

---

## Day 2: Validation & Reviewed-Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, proof-owner, benchmark, and
install/export validation split before runtime/scalability changes begin  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline and implementation-day gate:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm reviewed CMake parity, focused reviewed test ownership, canonical
   benchmark-report ownership, and install/export proof ownership.
3. Recheck the representative reviewed proof-owner and runtime surfaces most
   likely to move during Sprint 86:
   - `test_reorder_nd`
   - reorder-adjacent proof-owner tests
   - representative examples
   - canonical benchmark/reporting command surfaces
   - install/export proof scripts
4. Fix the authoritative rerun list most likely to matter throughout Sprint
   86.
5. Record the validation / reviewed-surface split in working notes and a Day 2
   artifact.

### Deliverables
- refreshed validation-baseline artifact
- preserved reviewed proof-owner map
- authoritative Sprint 86 rerun list

### Completion Criteria
- the strongest local validation contract is explicit before implementation
  work lands
- proof ownership across reviewed tests, benchmarks, and install/export
  surfaces is fixed in writing
- later code days have no ambiguity about the required validation gate

---

## Day 3: Reviewed Runtime Long-Pole Audit

**Title:** Runtime Audit  
**Theme:** Decompose the current reviewed long pole into one ranked live cause
map  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the highest-signal reviewed runtime surfaces:
   - `test_reorder_nd`
   - reorder-adjacent proof-owner tests
   - ND / reorder implementation owners
   - benchmark/reporting surfaces that can inform but not own correctness
2. Capture where the current reviewed runtime ceiling is strongest:
   - algorithmic ND or reorder work
   - fixture size / count concentration
   - repeated proof work that may be reorganizable
   - reviewed-path architecture that may be over-concentrated
3. Separate:
   - strongest first-batch implementation center
   - second-tier runtime and proof follow-through seams
   - support-only maintainer/docs surfaces
   - deliberate non-goals
4. Reconcile the rerank against Sprint 80’s performance contract and Sprint
   85’s close handoff.
5. Write the ranked runtime-cause artifact.

### Deliverables
- ranked reviewed-runtime artifact
- first-tier vs deferred cause map
- Sprint 80/85 carry-forward reconciliation notes

### Completion Criteria
- Sprint 86’s broad runtime problem is reduced to one ranked live map
- the strongest implementation center is explicit before design
- lower-value spillover work is clearly separated from the first lane

---

## Day 4: First Runtime / Scalability Boundary Freeze

**Title:** Boundary Freeze  
**Theme:** Fix the first bounded Sprint 86 implementation fence and the
allowed proof/runtime movement  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 runtime ranking against the Sprint 86 project-plan scope.
2. Decide the required first implementation center:
   - ND / reorder runtime reduction first
   - proof-surface rebalancing second
   - benchmark / CI follow-through only where the first batch truly forces it
3. Decide which support surfaces move only if forced:
   - reorder proof-owner tests
   - maintainer wording
   - support docs
   - package/runtime surfaces
4. Fix the preserved non-goal fence for the first landing:
   - no weakening of correctness proof quality
   - no generic maintainability decomposition restart
   - no benchmark-governance or example-governance drift into correctness
     ownership
   - no support-surface churn detached from a real landed runtime seam
5. Record the first implementation fence in working notes and a Day 4
   artifact.

### Deliverables
- first runtime/scalability-boundary artifact
- required vs support-only touch set
- preserved first-batch non-goal fence

### Completion Criteria
- Sprint 86 has one explicit first landing boundary
- support-only surfaces are clearly separated from the batch center
- Day 5 can design one runtime contract instead of a broad optimization rewrite

---

## Day 5: Algorithm / Proof Runtime Architecture Design

**Title:** Runtime Design  
**Theme:** Define the bounded runtime/scalability contract Sprint 86 will
actually land first  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 boundary and the strongest reviewed-runtime
   contradictions.
2. Define the ownership split for:
   - ND / reorder runtime reduction seams
   - retained proof-owner boundaries after the runtime change
   - benchmark / comparison evidence owners
   - CI / reviewed-path expectation surfaces
3. Decide how the first landing preserves Sprint 80’s performance-truth fence
   and Sprint 85’s clearer ownership map while reducing runtime concentration.
4. Fix the touch fence for code, tests, docs, and package/runtime surfaces.
5. Write the Day 5 architecture artifact and working-notes design summary.

### Deliverables
- algorithm / proof runtime architecture contract
- ownership split for touched seams
- preserved bounded-runtime and non-goal fence

### Completion Criteria
- Sprint 86 has one explicit implementation contract
- ownership between runtime reduction and proof preservation is clear
- Day 6 can implement one bounded landing without reopening design questions

---

## Day 6: ND Runtime Reduction Batch 1

**Title:** Runtime Batch  
**Theme:** Land the highest-value bounded runtime or scalability improvement
on the reorder / ND lane  
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-value runtime/scalability seam from the Day 5
   contract.
2. Keep the landing bounded to the required first implementation center.
3. Preserve the existing proof-owner split and avoid accidental contract or
   correctness drift outside the touched runtime seam.
4. Reconcile only the directly forced proof-owner, maintainer, or reviewed-path
   follow-through.
5. Run the required implementation-day validation gate and record the batch
   result.

### Deliverables
- landed bounded runtime/scalability batch
- any directly forced proof-owner/support follow-through
- Day 6 artifact and working-notes implementation record

### Completion Criteria
- one real ND / reorder runtime contradiction is reduced in code
- proof ownership remains explicit after the change
- the batch stays inside the Day 5 fence and passes the required validation

---

## Day 7: Post-Landing Runtime Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining Sprint 86 contradiction map after the first
runtime/scalability landing  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 landing and measure what contradiction it actually
   removed.
2. Re-rank the remaining Sprint 86 queue:
   - proof-surface rebalancing
   - benchmark / comparison follow-through
   - CI / reviewed-path alignment
3. Decide whether the strongest next move is still algorithmic or has shifted
   to reviewed-surface concentration.
4. Fix the exact Day 8 design center in writing.
5. Record the rerank in working notes and a Day 7 artifact.

### Deliverables
- post-landing rerank artifact
- updated remaining contradiction map
- fixed Day 8 design center

### Completion Criteria
- Sprint 86’s second implementation center is explicit after the Day 6 result
- lower-value spillover work remains clearly separated
- Day 8 can design the right next seam instead of assuming the original rank
  order survived unchanged

---

## Day 8: Proof-Surface Rebalancing Design

**Title:** Proof Design  
**Theme:** Define the next bounded reviewed-surface cleanup that reduces
unnecessary runtime concentration without weakening proof quality  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank and the strongest remaining reviewed runtime
   concentration seam.
2. Decide the exact proof-surface or fixture-organization cleanup to land
   next:
   - fixture decomposition
   - registration / grouping rebalance
   - retained proof-owner rerun reshaping
3. Fix which support-only surfaces and benchmark evidence move only if forced.
4. Separate the Day 9 required center from lower-value adjacent cleanup.
5. Write the Day 8 design artifact and working-notes summary.

### Deliverables
- proof-surface rebalancing design artifact
- exact Day 9 touch fence
- preserved proof-quality fence

### Completion Criteria
- Sprint 86 has one explicit second implementation contract
- the next reviewed-surface cleanup is bounded and reviewable
- Day 9 can land without weakening correctness ownership

---

## Day 9: Proof-Surface Rebalancing Batch

**Title:** Proof Batch  
**Theme:** Reduce unnecessary reviewed-runtime concentration while preserving
the strongest correctness proof  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 8 proof-surface rebalancing seam.
2. Preserve the authoritative proof-owner contract while reducing reviewed
   runtime concentration.
3. Reconcile only the directly forced helper, benchmark, or maintainer
   follow-through.
4. Run the required implementation-day validation gate.
5. Record the landed batch in working notes and a Day 9 artifact.

### Deliverables
- landed proof-surface rebalancing batch
- any directly forced follow-through
- Day 9 artifact and implementation record

### Completion Criteria
- one real reviewed runtime concentration seam is reduced
- proof ownership remains explicit and truthful after cleanup
- the batch stays inside the Day 8 fence and passes the required validation

---

## Day 10: Benchmark / Comparison Follow-Through Design

**Title:** Measurement Design  
**Theme:** Define the bounded evidence package Sprint 86 should land for the
touched runtime seam  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 and Day 9 landings against the Sprint 80 performance
   contract.
2. Decide the exact bounded benchmark / comparison evidence needed next:
   - before/after runtime capture
   - touched ND / reorder corpus comparisons
   - reviewed-path runtime interpretation notes
3. Fix which support-only surfaces move only if truly forced.
4. Keep the design bounded so measurement clarifies the touched seam without
   becoming a broad benchmark-governance rewrite.
5. Write the Day 10 design artifact and working-notes summary.

### Deliverables
- benchmark / comparison design artifact
- exact Day 11 touch fence
- preserved measurement-ownership fence

### Completion Criteria
- Sprint 86 has one explicit third implementation contract
- the measurement package is bounded and ownership-preserving
- Day 11 can land one clear evidence package without reopening generic
  benchmark policy questions

---

## Day 11: Benchmark / Comparison Follow-Through Batch

**Title:** Measurement Batch  
**Theme:** Add the bounded measurement and comparison artifacts needed for the
touched reorder/runtime seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded benchmark / comparison follow-through from the Day 10
   contract.
2. Preserve the touched correctness and proof-owner boundaries while improving
   runtime evidence quality.
3. Reconcile only the directly forced support or maintainer follow-through.
4. Run the required implementation-day validation gate.
5. Record the landed batch in working notes and a Day 11 artifact.

### Deliverables
- landed benchmark / comparison follow-through batch
- any directly forced follow-through
- Day 11 artifact and implementation record

### Completion Criteria
- one real measurement/evidence gap is reduced
- runtime evidence remains bounded to the touched seam
- the batch stays inside the Day 10 fence and passes the required validation

---

## Day 12: CI / Reviewed-Path Alignment & Validation Queue Freeze

**Title:** Alignment  
**Theme:** Reconcile the reviewed validation path and runtime expectations
with the landed Sprint 86 changes and freeze the final validation queue  
**Time estimate:** 12 hours

### Tasks
1. Re-read the landed Day 6, Day 9, and Day 11 surfaces.
2. Recheck whether any touched reviewed-path, CI, README, or maintainer
   wording still overstates or understates the new runtime shape.
3. Reconfirm whether install/export, package, or untouched benchmark surfaces
   truly need movement.
4. Fix the exact Day 13 validation queue in writing.
5. Record the final CI / reviewed-path alignment and validation queue in
   working notes and a Day 12 artifact.

### Deliverables
- CI / reviewed-path alignment artifact
- frozen Day 13 validation queue
- final touched-surface runtime truth map

### Completion Criteria
- no support-only drift remains before the full sweep
- the final validation queue is explicit and unambiguous
- Day 13 can execute from a fixed truth map rather than re-deciding scope

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full Sprint 86 validation queue and capture the measured
close baseline  
**Time estimate:** 12 hours

### Tasks
1. Run the implementation-day gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed baseline:
   - `make quality-review-full`
3. Reconfirm reviewed CMake parity.
4. Rerun the focused reviewed proof-owner tests, representative examples, and
   relevant benchmark/reporting commands fixed on Day 12.
5. Capture retained outputs, runtime shifts, and any non-blocking notes in a
   Day 13 artifact and working notes.

### Deliverables
- full validation artifact
- validated close baseline
- explicit retained proof-owner and runtime outputs

### Completion Criteria
- the full Sprint 86 validation queue passes
- the reviewed anchors stay exact
- Day 14 can close from measured evidence instead of implementation state

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 86 from the validated baseline and fix the next Epic 8
handoff queue  
**Time estimate:** 12 hours

### Tasks
1. Re-read the full Sprint 86 branch package:
   - runtime audit
   - algorithm / proof contract
   - ND runtime reduction
   - proof-surface rebalancing
   - benchmark / comparison follow-through
   - CI / reviewed-path alignment
   - validation
2. Recheck the Sprint 86 project-plan section against the landed work.
3. Reconcile the branch outcome against the next Epic 8 sprint ordering.
4. Write the Day 14 closeout and handoff artifact.
5. Record the final Sprint 86 close state in working notes.

### Deliverables
- Sprint 86 closeout/handoff artifact
- final working-notes closeout entry
- explicit post-Sprint-86 carry-forward queue

### Completion Criteria
- Sprint 86 closes from the validated Day 13 baseline
- the landed runtime/scalability package is summarized truthfully and boundedly
- the next Epic 8 contradiction center is fixed explicitly for the following
  sprint
