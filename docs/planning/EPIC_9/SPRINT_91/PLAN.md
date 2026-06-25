# Sprint 91 Plan: Compressed-First Product Convergence Phase 3

**Sprint Duration:** 14 days  
**Goal:** Reduce the largest remaining linked-list-first product costs by
making compressed-first workflows more central and clarifying lifecycle
ownership on the highest-value direct paths. This sprint implements the Sprint
91 section of `docs/planning/EPIC_9/PROJECT_PLAN.md`.

**Starting Point:** Sprint 90 closed with:
- one validated Epic 9 baseline package
- one bounded state-of-the-art target-state contract
- one frozen comparison and non-goal fence
- one ranked contradiction map that puts compressed-first product convergence
  first
- one explicit Sprint 91-first handoff queue

The strongest Sprint 91 pressure is no longer generic product cleanup. It is
one bounded compressed-first convergence package centered on:
- remaining linked-list-first conversion and publication costs
- compressed-first construction/import/publication seams
- one-shot vs repeated-run direct-workflow lifecycle clarity
- proof and public-surface follow-through only where the product contract
  truly moves

**End State:** Sprint 91 leaves behind:
- one fresh linked-list-first cost audit from the live Epic 9 starting tree
- one explicit compressed-first product and lifecycle contract
- one bounded construction/import landing
- one bounded publication/lifecycle landing
- focused proof, benchmark, and public-surface follow-through
- one validated Sprint 91 close baseline and Sprint 92 handoff queue

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, which stays within the 14-day cap and matches the
Sprint 91 project-plan `~168` hour scope.

---

## Day 1: Sprint 91 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 91 project-plan section and Sprint 90 handoff into
one bounded compressed-first implementation package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 91 section of
   `docs/planning/EPIC_9/PROJECT_PLAN.md`, the Sprint 90 retrospective, and
   the Sprint 90 Day 14 closeout artifact.
2. Reconfirm the preserved starting assumptions:
   - Sprint 90 target-state and non-goal fence stay authoritative
   - compressed-first product convergence is the first implementation lane in
     Epic 9
   - reviewed, install/export, and reporting truth surfaces remain unchanged
3. Define the Sprint 91 workstreams explicitly:
   - shell-cost audit
   - compressed-first architecture design
   - construction/import batch
   - publication/lifecycle batch
   - regression/proof follow-through
   - public-surface alignment
4. Record the strongest likely Sprint 91 touch surfaces:
   - `include/sparse_matrix.h`
   - `src/sparse_matrix.c`
   - strongest direct-family public surfaces
   - touched proof-owner tests, examples, and maintainer/public docs
5. Open Sprint 91 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 91 scope inventory
- baseline/setup working notes
- explicit workstream map

### Completion Criteria
- Sprint 91 starts from the validated Sprint 90 planning contract
- the compressed-first convergence problem is explicit before deeper
  validation and audit work begins
- the initial non-goal fence is visible before design or implementation widens

---

## Day 2: Validation & Maintained Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, install/export, benchmark, example,
and public-surface truth split before compressed-first implementation begins  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest implementation-day and substantial-batch gates:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm the maintained proof and support owners Sprint 91 must treat as
   authoritative:
   - reviewed CMake parity
   - representative direct-workflow tests
   - install/export proof
   - canonical reporting owner
   - public support and maintainer surfaces
3. Recheck the current ownership split so Sprint 91 does not blur:
   - direct-workflow correctness owners
   - lifecycle/integration owners
   - install/export proof owners
   - support-surface and workflow owners
4. Fix the authoritative rerun set most likely to matter throughout Sprint 91.
5. Record the validation and maintained-surface split in working notes and a
   Day 2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- maintained surface ownership map
- authoritative rerun list

### Completion Criteria
- the strongest local validation contract is explicit before audit or design
  findings are written
- proof ownership across reviewed tests, examples, install/export checks, and
  support surfaces is fixed in writing
- later Sprint 91 days have no ambiguity about the required truth surfaces

---

## Day 3: Remaining Linked-List-First Cost Audit

**Title:** Shell-Cost Audit  
**Theme:** Reduce the live product-model problem to one ranked map of the
highest-value linked-list-first costs  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the live tree against the strongest Sprint 91 contradiction class:
   - linked-list-first construction
   - linked-list-first import/export and publication
   - shell-centric direct-workflow entry paths
   - lifecycle ambiguity on mutated vs solve-ready states
2. Capture where compressed CSC/CSR-backed workflows still pay unnecessary
   shell conversion or publication cost.
3. Separate:
   - costs that should drive Sprint 91 implementation
   - costs that remain compatibility-only and should stay deferred
   - costs already materially reduced by Epic 8
4. Identify the highest-value source, header, test, and public-surface owners
   tied to those costs.
5. Write the ranked Day 3 audit artifact.

### Deliverables
- ranked linked-list-first cost artifact
- fix-now vs compatibility-only split
- strongest owner-surface map

### Completion Criteria
- the broad compressed-first problem is reduced to one ranked live map
- the highest-value construction/import/publication costs are explicit before
  the first implementation fence is frozen
- lower-value shell concerns are separated from the main Sprint 91 lane

---

## Day 4: First Implementation Boundary

**Title:** Boundary Freeze  
**Theme:** Fix one bounded first landing so Sprint 91 starts with the highest-
value compressed-first seam instead of generic product churn  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 audit against the Sprint 91 project-plan contract.
2. Decide the required first landing center:
   - construction/import seam inside the strongest product owner
3. Decide which adjacent surfaces are directly forced support-only follow-
   through and which are explicitly later:
   - publication/lifecycle
   - proof owners
   - headers/examples/docs
4. Freeze what Sprint 91 will not do in the first batch:
   - broad shell removal
   - family-wide lifecycle rewriting
   - package or capability widening
5. Write the Day 4 boundary artifact and update working notes.

### Deliverables
- first-landing boundary artifact
- required-owner vs support-only map
- explicit deferral list

### Completion Criteria
- Sprint 91 has one explicit first implementation fence
- the first landing is small enough to validate cleanly
- later product, lifecycle, and support work is clearly sequenced

---

## Day 5: Compressed-First Architecture Design

**Title:** Architecture Design  
**Theme:** Define the bounded Sprint 91 contract for compressed-first
construction/import/publication and shell containment  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 fence against the linked-list-first cost audit.
2. Define the future role of:
   - linked-list shell
   - CSC/CSR-backed construction/import
   - public publication/export seams
3. Decide how one-shot and repeated-run direct workflows should read publicly
   after the first two Sprint 91 landings.
4. Decide which compatibility shims remain acceptable and which should stop
   being conceptual center stage.
5. Write the Day 5 architecture artifact.

### Deliverables
- compressed-first architecture artifact
- public workflow role split
- compatibility-shim policy

### Completion Criteria
- the repo has one explicit compressed-first contract before code moves
- the shell is bounded conceptually even if compatibility remains
- Day 6 implementation can land without reopening product intent

---

## Day 6: Construction/Import Batch

**Title:** Construction Batch  
**Theme:** Land the highest-value compressed-first construction or import seam
without breaking compatibility callers  
**Time estimate:** 12 hours

### Tasks
1. Implement the required first construction/import landing from the Day 5
   design.
2. Keep the batch bounded to the highest-value public/direct workflow seam.
3. Add directly forced follow-through only where the landing requires it:
   - touched headers
   - touched tests
   - touched internal helpers
4. Run the required implementation-day validation gates.
5. Record the landed batch in working notes and a Day 6 artifact.

### Deliverables
- bounded construction/import implementation batch
- directly forced proof follow-through
- validated Day 6 baseline

### Completion Criteria
- at least one compressed-first construction/import workflow is materially
  better than on Sprint 90
- compatibility callers remain intact
- the required validation gates pass cleanly

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining compressed-first work after the first code
landing  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 landing against the Day 3 audit and Day 5 design.
2. Decide whether the strongest next seam is now:
   - publication/publication-owner cleanup
   - lifecycle clarification
   - direct proof follow-through
3. Fix the exact Day 8 design center from the live post-Day-6 tree.
4. Separate what Sprint 91 no longer needs from what still carries the
   strongest product-model payoff.
5. Record the rerank in working notes and a Day 7 artifact.

### Deliverables
- post-landing rerank artifact
- next design-center choice
- updated support-only map

### Completion Criteria
- the strongest remaining Sprint 91 seam is explicit after the first landing
- the second implementation center is chosen from live evidence
- unnecessary widened follow-through is avoided

---

## Day 8: Publication & Lifecycle Design

**Title:** Lifecycle Design  
**Theme:** Define the bounded second Sprint 91 implementation contract around
publication and direct-workflow lifecycle clarity  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank and confirm the strongest next product seam.
2. Define the bounded second implementation center:
   - publication/public-surface clarification
   - one-shot vs repeated-run lifecycle contract tightening
3. Decide the strongest support-only follow-through if the batch truly forces
   movement:
   - direct-workflow proof owners
   - integration owners
   - touched support/public surfaces
4. Freeze which larger lifecycle or family-wide rewrites remain explicitly out
   of scope.
5. Write the Day 8 design artifact.

### Deliverables
- second implementation contract
- support-only follow-through map
- explicit non-touch list

### Completion Criteria
- Day 9 has one exact bounded publication/lifecycle contract
- the second batch is still small enough to validate cleanly
- broader lifecycle churn remains fenced off

---

## Day 9: Publication/Lifecycle Batch

**Title:** Lifecycle Batch  
**Theme:** Land the bounded publication or lifecycle tightening that makes the
compressed-first product story easier to use correctly  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded Day 8 publication/lifecycle landing.
2. Apply directly forced proof or support follow-through only where needed.
3. Keep the batch centered on clearer one-shot vs repeated-run and
   shell-to-compute/publication semantics.
4. Run the required implementation-day validation gates.
5. Record the landed batch in working notes and a Day 9 artifact.

### Deliverables
- bounded publication/lifecycle implementation batch
- directly forced proof/support follow-through
- validated Day 9 baseline

### Completion Criteria
- the direct-workflow product story is materially smaller and clearer
- solve-ready/publication ambiguity is reduced on the touched seam
- the required validation gates pass cleanly

---

## Day 10: Regression & Proof Follow-Through Design

**Title:** Proof Design  
**Theme:** Define the strongest focused proof and benchmark follow-through
still needed after the two product-model landings  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 and Day 9 landings against the Sprint 91 goals.
2. Identify the highest-value remaining proof gaps:
   - direct workflow regression coverage
   - lifecycle-preservation proof
   - benchmark/measurement confirmation on touched seams
3. Fix the exact Day 11 center and the strongest support-only follow-through.
4. Keep the proof batch bounded to Sprint 91 claims rather than widening into
   broader Epic 9 comparison work.
5. Write the Day 10 design artifact.

### Deliverables
- focused proof-follow-through design artifact
- exact Day 11 center
- bounded rerun/proof queue

### Completion Criteria
- the remaining Sprint 91 proof need is explicit and bounded
- Day 11 has one exact implementation or support center
- broader comparison work remains deferred to later epic lanes

---

## Day 11: Regression, Proof & Public-Surface Follow-Through

**Title:** Follow-Through Batch  
**Theme:** Land the strongest remaining regression/proof/public-surface
follow-through required by the Sprint 91 product contract  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded Day 10 follow-through batch.
2. Add the strongest needed regression/lifecycle proof on touched direct
   workflows.
3. Reconcile touched public or maintainer wording only where the landed
   product contract truly changed interpretation.
4. Run the required validation gates:
   - implementation-day gates if only bounded code moved
   - stronger reviewed batch if the support/proof surface widened materially
5. Record the batch in working notes and a Day 11 artifact.

### Deliverables
- focused proof/public-surface follow-through batch
- validated Day 11 baseline
- updated product-contract wording where required

### Completion Criteria
- the landed Sprint 91 product claims now have focused proof support
- no unnecessary broad support-surface churn is introduced
- the required validation gates pass cleanly

---

## Day 12: Final Alignment & Validation Queue Freeze

**Title:** Alignment Pass  
**Theme:** Freeze the final Sprint 91 proof-owner map and the Day 13
validation queue before the full sweep  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6, Day 9, and Day 11 landings together.
2. Decide whether any support-only edit is still needed before the full
   validation sweep.
3. Freeze the final proof-owner split for:
   - touched direct-workflow owners
   - lifecycle/integration owners
   - touched public support surfaces
4. Freeze the exact Day 13 queue:
   - implementation-day gates
   - reviewed-path gates if needed
   - representative examples and reporting follow-ons
5. Record the alignment pass in working notes and a Day 12 artifact.

### Deliverables
- final proof-owner map
- frozen validation queue
- Day 12 alignment artifact

### Completion Criteria
- no major support-only ambiguity remains before the sweep
- the final Sprint 91 validation queue is fixed in writing
- Day 13 can execute from one stable package

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the required final validation queue for the Sprint 91 product-
model package  
**Time estimate:** 12 hours

### Tasks
1. Run the required implementation-day validation gates:
   - `make format`
   - `make lint`
   - `make test`
2. Run the stronger reviewed-path baseline if Sprint 91’s landed surface
   warrants it:
   - `make quality-review-full`
3. Run the focused representative proof owners and examples fixed on Day 12.
4. Record the measured validation results, parity anchors, and any residual
   non-blocking notes.
5. Write the Day 13 validation artifact and update working notes.

### Deliverables
- full Sprint 91 validation artifact
- measured close baseline
- explicit residual note set, if any

### Completion Criteria
- all required validation gates pass
- the final Sprint 91 baseline is explicit in writing
- the branch is ready for closeout

---

## Day 14: Sprint 91 Closeout & Sprint 92 Handoff

**Title:** Closeout  
**Theme:** Close Sprint 91 from the validated product-model package and leave
one explicit Sprint 92-first backend-maturity handoff queue  
**Time estimate:** 12 hours

### Tasks
1. Summarize what Sprint 91 established:
   - live shell-cost rerank
   - compressed-first architecture contract
   - construction/import landing
   - publication/lifecycle landing
   - proof and public-surface follow-through
2. Decide whether `docs/planning/EPIC_9/PROJECT_PLAN.md` needs any correction
   from the actual Sprint 91 result.
3. Write the Sprint 91 closeout and handoff artifact.
4. Fix the Sprint 92-first queue explicitly:
   - portable backend and dense-kernel maturity second
   - runtime/threading next
   - capability, coherence, maintainability, packaging, comparison, and
     final closeout later in the already-ranked order
5. Record the final close state in working notes.

### Deliverables
- Sprint 91 closeout artifact
- explicit Sprint 92 handoff queue
- completed Sprint 91 working-notes close state

### Completion Criteria
- Sprint 91 closes from one explicit validated product-model package
- the Sprint 92-first execution order is fixed in writing
- Epic 9 implementation remains bounded and evidence-backed rather than
  reopening generic ambition
