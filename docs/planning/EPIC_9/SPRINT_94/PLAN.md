# Sprint 94 Plan: Capability Surface Modernization Phase 3

**Sprint Duration:** 14 days  
**Goal:** Move beyond the current real-only and compile-time-bounded
capability surface on the highest-value lanes first. This sprint implements
the Sprint 94 section of `docs/planning/EPIC_9/PROJECT_PLAN.md`.

**Starting Point:** Sprint 94 begins from:
- the Sprint 90 capability-priority and non-goal contract
- the Sprint 91 compressed-first public/product convergence baseline
- the Sprint 92 widened dense/backend seam and stronger index-format
  discipline on touched performance paths
- the Sprint 93 runtime and benchmark baseline, which keeps capability work
  from reopening the same runtime long-pole seam first
- a still-bounded public capability story centered on real-only scalar
  assumptions, compile-time-selected index width, and narrow solver-family
  breadth on widened scalar/index seams

The strongest Sprint 94 pressure is no longer generic modernization. It is one
bounded capability-breadth package centered on:
- capability reranking across scalar, index-width, and solver-family seams
- a bounded scalar/index architecture contract for the next real widening step
- one first-class scalar-family widening landing on the highest-value public
  and implementation surfaces
- stronger 64-bit and ABI-safety maturity on touched paths
- one bounded solver-family breadth follow-through only where the widened
  scalar/index contract truly needs it
- focused proof, docs, and package wording that matches the wider claim

**End State:** Sprint 94 leaves behind:
- one fresh capability-breadth audit from the live Epic 9 tree
- one explicit scalar/index capability contract and non-claim fence
- one bounded scalar-family widening landing
- one bounded index/ABI maturity landing
- one bounded solver-family breadth landing
- one focused proof/docs/package follow-through package
- one validated Sprint 94 close baseline and Sprint 95 handoff queue

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, which stays within the 14-day cap and matches the
Sprint 94 project-plan `~168` hour scope.

---

## Day 1: Sprint 94 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 94 project-plan section and Sprint 93 handoff into
one bounded capability-widening package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 94 section of
   `docs/planning/EPIC_9/PROJECT_PLAN.md`, the Sprint 93 retrospective, and
   the Sprint 93 Day 14 closeout artifact.
2. Reconfirm the preserved starting assumptions:
   - Sprint 90 capability priorities and non-goal fence stay authoritative
   - Sprint 91-93 product/backend/runtime work stays closed unless capability
     widening truly forces support follow-through
   - the strongest next move is one bounded scalar/index-first widening step,
     not broad generic numeric templating
3. Define the Sprint 94 workstreams explicitly:
   - capability rerank
   - scalar/index architecture design
   - scalar-family widening batch
   - index/ABI maturity batch
   - solver-family breadth batch
   - proof/docs/package alignment
4. Record the strongest likely Sprint 94 touch surfaces:
   - `include/sparse_types.h`
   - touched public scalar/index headers
   - strongest implementation owners using `sparse_scalar_t` / `idx_t`
   - focused tests, benchmarks, and support surfaces only if the widened
     contract truly forces them
5. Open Sprint 94 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 94 scope inventory
- baseline/setup working notes
- explicit workstream map

### Completion Criteria
- Sprint 94 starts from the validated Sprint 93 capability handoff
- the capability-widening problem is explicit before deeper validation and
  audit work begins
- the initial non-goal fence is visible before design or implementation widens

---

## Day 2: Validation & Maintained Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, benchmark, install/export, and
public-surface truth split before capability work begins  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest implementation-day and substantial-batch gates:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm the maintained owners Sprint 94 must treat as authoritative:
   - reviewed CMake parity
   - representative scalar/index-sensitive proof owners
   - canonical benchmark reporting owner
   - install/export proof owner
   - public header, support, and workflow surfaces
3. Recheck the current ownership split so Sprint 94 does not blur:
   - scalar/index public contract owners
   - solver-family proof owners
   - benchmark/reporting owners
   - support/package/workflow wording owners
4. Fix the authoritative rerun set most likely to matter throughout Sprint 94.
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
- later Sprint 94 days have no ambiguity about required truth surfaces

---

## Day 3: Capability Re-rank Audit

**Title:** Capability Audit  
**Theme:** Reduce the broad capability problem to one ranked map of the
highest-value scalar, index, and solver-family widening seams  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the live tree against the strongest Sprint 94 contradiction class:
   - real-only scalar contract
   - compile-time-selected wider-index maturity
   - solver-family breadth constrained by those seams
   - public and package wording that still reads narrower than the touched
     implementation reality
2. Re-rank the candidate widening lanes by user value, proof cost, and risk:
   - complex support
   - mixed precision
   - wider index maturity
   - bounded solver-family breadth
3. Separate:
   - widening seams that should drive Sprint 94 implementation
   - widening seams that should remain explicitly deferred
   - seams already materially improved by prior Epic 9 work
4. Identify the highest-value source, header, test, benchmark, and support
   owners tied to those seams.
5. Write the ranked Day 3 capability audit artifact.

### Deliverables
- ranked capability audit artifact
- fix-now vs later split
- strongest owner-surface map

### Completion Criteria
- the broad capability problem is reduced to one ranked live map
- the highest-value widening seam is explicit before the first implementation
  fence is frozen
- lower-value or higher-risk widening ambitions are separated from the main
  Sprint 94 lane

---

## Day 4: Scalar & Index Capability Contract Design

**Title:** Contract Design  
**Theme:** Define the bounded Sprint 94 capability contract before code moves  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 audit against the Sprint 94 project-plan contract.
2. Define the roles Sprint 94 should preserve across:
   - public scalar aliasing
   - implementation scalar ownership
   - wider-index and ABI-safety expectations
   - solver families that depend on the touched scalar/index seam
3. Decide what Sprint 94 will and will not claim:
   - one real widened capability step
   - stronger touched 64-bit maturity
   - no fake broad numeric genericity
   - no fake full-library complex or mixed-precision maturity
4. Decide the minimum compatibility and fallback interpretation the widened
   scalar/index seam must preserve.
5. Write the Day 4 capability contract artifact.

### Deliverables
- scalar/index capability contract artifact
- compatibility and non-claim fence
- touched-owner role split

### Completion Criteria
- the repo has one explicit capability-widening contract before code moves
- scalar, index, and solver-family scope are separated in writing
- later implementation days can stay bounded to the highest-value seam

---

## Day 5: First Implementation Boundary

**Title:** Boundary Freeze  
**Theme:** Fix one bounded first landing so Sprint 94 starts with the
highest-value scalar seam instead of generic capability churn  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 and Day 4 artifacts against the Sprint 94 project-plan
   contract.
2. Decide the required first landing center:
   - one scalar-family widening seam on the strongest public and
     implementation owners
3. Decide which adjacent surfaces are directly forced support-only follow-
   through and which are explicitly later:
   - index/ABI maturity
   - solver-family breadth
   - proof and public docs
   - package/workflow wording
4. Freeze what Sprint 94 will not do in the first batch:
   - broad templated rewrite
   - fake whole-library complex support
   - detached benchmark-only widening
   - broad package or platform reopening
5. Write the Day 5 boundary artifact and update working notes.

### Deliverables
- first-landing boundary artifact
- required-owner vs support-only map
- explicit deferral list

### Completion Criteria
- Sprint 94 has one explicit first implementation fence
- the first landing is small enough to validate cleanly
- later index, solver-family, and support work is clearly sequenced

---

## Day 6: Scalar-Family Widening Design

**Title:** Scalar Design  
**Theme:** Define the bounded implementation contract for the touched scalar
surface widening seam  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 5 fence against the capability rerank.
2. Define the exact scalar-widening center:
   - public alias and strongest shared implementation seam
3. Decide the preserved invariants:
   - unchanged default reviewed build behavior unless the widened contract
     explicitly says otherwise
   - touched public API clarity remains stronger, not looser
   - current proof owners stay deterministic and auditable
4. Decide the strongest directly forced proof and support follow-through:
   - scalar-sensitive tests
   - one strongest solver-family consumer
   - touched public/support wording only if the contract truly moves
5. Write the Day 6 scalar-widening design artifact.

### Deliverables
- scalar widening design artifact
- preserved-invariant list
- directly forced proof/support map

### Completion Criteria
- the Day 7 implementation center is exact before any code moves
- the widened scalar seam is bounded enough to validate cleanly
- later index and solver-family work remains clearly separate

---

## Day 7: Scalar-Family Widening Batch

**Title:** Scalar Batch  
**Theme:** Land the first real scalar-surface widening on the highest-value
public and implementation seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 6 scalar widening contract on the required owners only.
2. Add or update the minimum focused proof directly forced by that widening.
3. Keep adjacent solver-family, package, and broader capability surfaces
   untouched unless the first landing truly requires them.
4. Run the required implementation-day validation queue:
   - `make format`
   - `make lint`
   - `make test`
5. Record landed scope, preserved invariants, and observed validation results
   in working notes and a Day 7 artifact.

### Deliverables
- bounded scalar widening landing
- focused proof follow-through
- validation evidence for the first batch

### Completion Criteria
- one real scalar capability widening lands cleanly
- the first landing preserves the intended public and implementation contract
- required validation passes before Sprint 94 moves on

---

## Day 8: Post-Landing Audit & Re-rank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining Sprint 94 contradictions after the first
scalar widening landing  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 landing against the Day 3 capability rerank.
2. Identify what the first landing materially closed and what remains open
   across:
   - scalar breadth
   - index/ABI maturity
   - solver-family breadth
   - proof and public-surface clarity
3. Decide whether a second scalar landing is still justified or whether the
   strongest remaining seam is now index/ABI maturity.
4. Freeze the exact Day 9 design center and directly forced support surfaces.
5. Write the Day 8 rerank artifact and update working notes.

### Deliverables
- post-landing rerank artifact
- updated contradiction order
- exact Day 9 design center

### Completion Criteria
- the first landing is assessed against the live tree, not the plan in the
  abstract
- the next implementation center is explicit before more code moves
- lower-value widening opportunities remain deferred

---

## Day 9: Index & ABI Maturity Design

**Title:** Index Design  
**Theme:** Define the bounded touched-path 64-bit and ABI-safety follow-through
contract  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 8 rerank against the Sprint 94 project-plan contract.
2. Define the strongest remaining touched-path maturity seams:
   - 64-bit-safe formatting and parsing
   - width-aware allocation and overflow guards
   - public header and consumer-surface ABI interpretation
3. Decide which owners must move in the index/ABI batch and which remain
   explicitly later.
4. Decide the minimum proof and support follow-through the batch must carry.
5. Write the Day 9 index/ABI design artifact.

### Deliverables
- index/ABI maturity design artifact
- required-owner map
- touched proof/support contract

### Completion Criteria
- the Day 10 implementation center is exact before code moves
- stronger touched 64-bit maturity is defined without reopening broad product
  claims
- proof and ABI interpretation stay bounded to touched paths

---

## Day 10: Index & ABI Maturity Batch

**Title:** Index Batch  
**Theme:** Land the strongest remaining touched-path 64-bit and ABI-safety
improvements  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 9 index/ABI contract on the required owners only.
2. Add or update the minimum focused proof directly forced by that maturity
   batch.
3. Keep broader solver-family or package surfaces untouched unless the index
   batch truly requires them.
4. Run the required implementation-day validation queue:
   - `make format`
   - `make lint`
   - `make test`
5. Record landed scope, preserved ABI reading, and observed validation results
   in working notes and a Day 10 artifact.

### Deliverables
- bounded index/ABI maturity landing
- focused proof follow-through
- validation evidence for the second batch

### Completion Criteria
- touched paths have materially stronger wider-index maturity
- the batch preserves the bounded Sprint 94 capability contract
- required validation passes before solver-family follow-through begins

---

## Day 11: Solver-Family Breadth & Alignment Design

**Title:** Breadth Design  
**Theme:** Decide the smallest truthful solver-family expansion and the proof,
docs, and package alignment it requires  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 10 landing against the Sprint 94 rerank and contract.
2. Decide the strongest remaining solver-family breadth seam the widened
   scalar/index contract actually needs.
3. Decide whether the remaining Sprint 94 work is:
   - one small solver-family landing
   - proof/docs/package alignment only
   - or a no-op implementation retirement if evidence says the new breadth is
     already sufficient
4. Freeze the exact Day 12 center and directly forced support surfaces.
5. Write the Day 11 breadth/alignment design artifact.

### Deliverables
- solver-family breadth design artifact
- required-owner vs support-only map
- exact Day 12 center

### Completion Criteria
- the final Sprint 94 implementation or support batch is explicit
- solver-family widening stays bounded to the touched capability contract
- proof/docs/package alignment is sequenced behind the actual breadth need

---

## Day 12: Solver-Family Breadth & Proof/Docs/Package Follow-Through

**Title:** Alignment Batch  
**Theme:** Land the final bounded solver-family and support-surface follow-
through required by the widened capability contract  
**Time estimate:** 12 hours

### Tasks
1. Land the Day 11 breadth or support-only batch on required owners only.
2. Add or update focused proof for the touched solver-family or capability
   seam.
3. Reconcile touched public docs, package wording, or maintainer surfaces only
   where the widened capability contract truly moved.
4. Run the required validation queue for the touched scope:
   - `make format`
   - `make lint`
   - `make test`
5. Record the final Sprint 94 landing set in working notes and a Day 12
   artifact.

### Deliverables
- bounded solver-family or support-surface follow-through landing
- focused proof/docs/package alignment
- validation evidence for the final implementation batch

### Completion Criteria
- the widened capability claim is supported by code, proof, and wording
- no unnecessary broadening moves land beyond the Sprint 94 contract
- required validation passes before the full close sweep

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Re-run the strongest full-stack validation and evidence surfaces
from the final Sprint 94 state  
**Time estimate:** 12 hours

### Tasks
1. Run the full implementation-day validation queue from the final Sprint 94
   branch state:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest substantial-batch validation and evidence queue:
   - `make quality-review-full`
   - reviewed CMake parity checks
   - representative scalar/index/solver-family reruns
   - `make bench-canonical-report`
3. Recheck install/export and support surfaces only if the widened contract
   touched them materially.
4. Record all authoritative pass/fail results, runtime notes, and any
   remaining non-blocking residuals.
5. Write the Day 13 validation artifact and update working notes.

### Deliverables
- full validation artifact
- final authoritative rerun results
- residual non-blocking note list

### Completion Criteria
- all required quality gates pass from the final Sprint 94 branch state
- the widened capability claim is anchored to the strongest maintained proof
  surfaces
- Day 14 can close from one validated baseline

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Freeze the Sprint 94 result, residual queue, and Sprint 95 handoff
from the validated baseline  
**Time estimate:** 12 hours

### Tasks
1. Summarize the final Sprint 94 result against the original project-plan
   contract:
   - capability rerank
   - scalar/index contract
   - scalar widening landing
   - index/ABI maturity landing
   - solver-family and support follow-through
2. Freeze the residual queue:
   - deferred capability lanes
   - still-bounded non-claims
   - any support or package work intentionally left for later sprints
3. Record the strongest validated quality anchors and any remaining runtime or
   proof notes that do not block Sprint 94 closeout.
4. Decide whether `docs/planning/EPIC_9/PROJECT_PLAN.md` needs any correction
   based on the live branch result.
5. Write the Day 14 closeout artifact and Sprint 95 handoff notes.

### Deliverables
- Sprint 94 closeout artifact
- residual queue and non-claim freeze
- explicit Sprint 95 handoff queue

### Completion Criteria
- Sprint 94 closes from one validated final baseline
- the repo has one bounded capability-widening close state with explicit
  residuals
- Sprint 95 receives a precise handoff instead of another broad modernization
  reset
