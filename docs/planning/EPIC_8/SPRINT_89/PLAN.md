# Sprint 89 Plan: Final Integration, External Comparison & Epic 8 Closeout

**Sprint Duration:** 14 days  
**Goal:** Reassess the library against the Epic 8 opening review, run the full
cross-surface validation and bounded external comparison sweep, and close Epic
8 from a truthful validated end state. This sprint implements the Sprint 89
section of `docs/planning/EPIC_8/PROJECT_PLAN.md`.

**Starting Point:** Sprint 88 closed with a cleaner front door, better example
and install guidance layering, and a preserved static-first package contract.
The strongest remaining Epic 8 work is no longer one isolated capability,
assurance, or support contradiction. It is final integration:
- re-auditing the live post-Sprint-88 tree against the original Epic 8 review
  categories
- comparing the shipped library against the chosen external reference class on
  correctness, package shape, and bounded performance signals
- landing only the highest-value last-mile reconciliations that the re-audit
  proves are still worth fixing
- freezing a final validated baseline strong enough to support Epic 8 closeout
- writing an explicit residual queue that distinguishes real carry-forward work
  from deliberate non-claims
- producing the final sprint, epic, and project-close surfaces from evidence,
  not aspiration

The highest-value Sprint 89 work is therefore not generic “final polish.” It
is one bounded integration and closeout package that:
- repeats the architecture, capability, performance, usability, packaging, and
  workflow review against the live tree
- runs a bounded external comparison sweep against the retained reference class
- lands one final cross-surface fix batch only where the re-audit shows real
  contradictions
- runs the strongest reviewed, install/export, benchmark-reporting, and
  follow-on proof surfaces
- calibrates the post-Epic-8 residual queue explicitly
- produces the Sprint 89 retrospective, Epic 8 closeout notes, and final
  project-summary surfaces from the validated end state

**End State:** Sprint 89 leaves behind:
- a final post-Epic-8 contradiction map
- one bounded external comparison package
- one final last-mile cross-surface fix batch
- one fully validated Epic 8 close baseline
- one explicit residual queue for the next planning cycle
- one Sprint 89 retrospective and handoff package
- one Epic 8 retrospective and final project-summary package

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, which stays within the practical 14-day cap while
preserving the Sprint 89 ordering and intent from the project-plan `~180` hour
scope.

---

## Day 1: Sprint 89 Scope Audit & End-State Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 89 project-plan section and Sprint 88 closeout into
one bounded final-integration execution package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 89 section of
   `docs/planning/EPIC_8/PROJECT_PLAN.md`, the Sprint 88 closeout artifact,
   and the Sprint 88 retrospective.
2. Reconfirm the preserved Sprint 89 starting assumptions:
   - Sprints 80-88 are complete
   - the residual queue is stable enough for final comparison and calibration
   - Sprint 89 should close Epic 8 from evidence, not reopen broad new scope
   - the strongest local reviewed baseline remains `make quality-review-full`
3. Define the Sprint 89 workstreams explicitly:
   - end-state re-audit
   - external comparison sweep
   - final cross-surface fix batch
   - full validation and reporting sweep
   - residual queue finalization
   - retrospective, handoff, and project-summary closeout
4. Record the strongest likely Sprint 89 touch surfaces:
   - project-planning closeout docs
   - maintained proof scripts
   - representative benchmark/reporting surfaces
   - the highest-value live contradictions discovered by the re-audit
5. Open Sprint 89 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 89 scope inventory
- final-integration workstream map
- starting working-notes baseline

### Completion Criteria
- Sprint 89 starts from the validated Sprint 88 end state
- the final-integration problem is explicit before deeper audit begins
- the non-goal fence is visible before design or implementation work

---

## Day 2: Validation & Maintained Cross-Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, install/export, example, and
benchmark ownership split before final closeout work begins  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline and implementation-day gate:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm the maintained cross-surface proof owners that Sprint 89 must
   treat as authoritative:
   - reviewed CMake parity
   - representative reviewed examples
   - `tests/test_install.sh`
   - `tests/test_cmake_install.sh`
   - `make bench-canonical-report`
3. Recheck the current proof-owner split so final integration does not blur:
   - reviewed correctness owners
   - install/export proof owners
   - benchmark/reporting owners
   - support-surface and workflow owners
4. Fix the authoritative rerun list most likely to matter throughout Sprint
   89.
5. Record the validation and maintained-surface split in working notes and a
   Day 2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- maintained cross-surface ownership map
- authoritative Sprint 89 rerun list

### Completion Criteria
- the strongest local validation contract is explicit before implementation
  work lands
- proof ownership across reviewed tests, install/export checks, examples,
  reports, and docs is fixed in writing
- later code or docs days have no ambiguity about the required validation gate

---

## Day 3: End-State Re-audit

**Title:** End-State Audit  
**Theme:** Reduce the full post-Sprint-88 state to one ranked final
contradiction map  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the live tree against the Epic 8 opening review categories:
   - architecture and maintainability
   - capability surface
   - numerical assurance and comparison
   - runtime and scalability
   - packaging and consumer shape
   - front-door usability and workflow layering
2. Capture where the current post-Sprint-88 tree still falls short of the
   desired Epic 8 end state.
3. Separate:
   - contradictions that require one final fix batch
   - contradictions that only need explicit non-claims or calibration
   - contradictions already closed by earlier sprints
   - contradictions that should move to the next planning cycle
4. Identify which proof surfaces and comparison surfaces matter most for the
   end-state call.
5. Write the ranked end-state re-audit artifact.

### Deliverables
- ranked end-state contradiction artifact
- fix-now vs calibrate-only split
- proof-surface priority map

### Completion Criteria
- Sprint 89’s broad closeout problem is reduced to one ranked live map
- the strongest remaining contradictions are explicit before boundary freeze
- lower-value spillover work is separated from the final lane

---

## Day 4: Final Integration Boundary Freeze

**Title:** Boundary Freeze  
**Theme:** Fix the first bounded Sprint 89 implementation fence and the
allowed spillover  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 end-state audit against the Sprint 89 project-plan scope.
2. Decide the required first implementation center:
   - the bounded external comparison and end-state evidence lane first
   - the final fix batch only after the evidence package is explicit
3. Decide which surfaces move only if forced:
   - proof scripts
   - benchmarks and reports
   - support-surface closeout docs
   - representative implementation files implicated by the re-audit
4. Fix the preserved non-goal fence for the first landing:
   - no broad reopening of earlier sprint scope
   - no speculative optimization or capability widening
   - no benchmark-policy rewrite detached from maintained proof owners
   - no support-surface churn without an end-state contradiction
5. Record the first implementation fence in working notes and a Day 4
   artifact.

### Deliverables
- final-integration boundary artifact
- required vs support-only touch set
- preserved non-goal fence

### Completion Criteria
- Sprint 89 has one explicit first landing boundary
- support-only surfaces are clearly separated from the first batch center
- Day 5 can design one integration contract instead of a broad final sweep

---

## Day 5: Comparison & Fix Architecture Design

**Title:** Integration Design  
**Theme:** Define the exact external-comparison contract and the downstream
conditions for the last fix batch  
**Time estimate:** 12 hours

### Tasks
1. Design the bounded external comparison sweep:
   - retained comparison class
   - correctness signal
   - package-shape signal
   - bounded performance signal
2. Design how Sprint 89 will interpret comparison outcomes:
   - immediate final fix candidate
   - calibrated non-claim
   - future residual item
3. Decide the exact ownership split across:
   - comparison artifact owner
   - final fix owner
   - validation/reporting owner
   - closeout-writing owner
4. Define the entry criteria for the final fix batch so it only lands if
   evidence justifies it.
5. Record the design in working notes and a Day 5 artifact.

### Deliverables
- external-comparison design artifact
- fix-batch entry contract
- ownership split for comparison, fixes, validation, and closeout

### Completion Criteria
- the comparison lane is explicit before implementation work begins
- the final fix batch has objective entry criteria
- Sprint 89 has a defensible evidence path for closeout

---

## Day 6: End-State Re-audit Batch

**Title:** Re-audit Package  
**Theme:** Materialize the final architecture, capability, usability,
performance, and packaging review against the live tree  
**Time estimate:** 12 hours

### Tasks
1. Execute the bounded end-state re-audit from the Day 3 and Day 5 design.
2. Write the re-audit package with category-by-category outcomes:
   - closed
   - partially closed but calibrated
   - still contradictory and fix-worthy
3. Identify the smallest truthful set of final-fix candidates.
4. Reconcile the re-audit against the original Epic 8 opening concerns.
5. Record the result in working notes and a Day 6 artifact.

### Deliverables
- final re-audit artifact
- smallest truthful final-fix candidate set
- opening-review reconciliation notes

### Completion Criteria
- the post-Sprint-88 tree has one explicit end-state review package
- the last-mile contradiction list is bounded and evidence-backed
- Day 7 can rerank from a live re-audit instead of assumptions

---

## Day 7: Post-Re-audit Rerank

**Title:** Rerank  
**Theme:** Convert the re-audit into one exact landing order for the
comparison sweep and the final fix batch  
**Time estimate:** 12 hours

### Tasks
1. Re-rank the Day 6 outcomes by value, urgency, and proof strength.
2. Decide whether the strongest next move is:
   - external comparison first
   - one bounded fix first
   - or comparison plus fix in a tightly coupled sequence
3. Separate:
   - must-land-before-close items
   - comparison-only evidence items
   - residual-queue items
4. Fix the exact Day 8 design center.
5. Record the rerank in working notes and a Day 7 artifact.

### Deliverables
- post-re-audit rerank artifact
- must-land vs residual split
- exact next-batch center

### Completion Criteria
- the strongest remaining contradiction is explicit after the re-audit
- the comparison lane and fix lane are ordered clearly
- no ambiguity remains about the next implementation center

---

## Day 8: External Comparison Sweep Design

**Title:** Comparison Design  
**Theme:** Freeze the exact bounded comparison protocol and reporting shape  
**Time estimate:** 12 hours

### Tasks
1. Fix the exact comparison inputs, fixtures, and retained reference class.
2. Freeze the comparison outputs Sprint 89 will report:
   - correctness agreement
   - package/consumer shape alignment
   - bounded performance observations
3. Decide the touched local surfaces, scripts, fixtures, or docs needed to run
   and record the sweep.
4. Decide the acceptance criteria for a “good enough to close” comparison
   result.
5. Record the design in working notes and a Day 8 artifact.

### Deliverables
- comparison protocol artifact
- accepted output/reporting shape
- touched-surface map for the comparison lane

### Completion Criteria
- the external sweep is explicit before it runs
- comparison results will be interpretable without ad hoc framing
- any forced implementation spillover is known up front

---

## Day 9: External Comparison Sweep

**Title:** Comparison Batch  
**Theme:** Run the bounded external comparison package and capture the result  
**Time estimate:** 12 hours

### Tasks
1. Execute the retained comparison sweep from the Day 8 protocol.
2. Capture correctness, package-shape, and bounded performance observations.
3. Write the comparison artifact and summarize where the library:
   - agrees strongly
   - differs acceptably
   - still needs one last reconciliation
4. Decide whether the Day 11 fix batch is still necessary and, if so, how
   small it can stay.
5. Record the result in working notes and a Day 9 artifact.

### Deliverables
- external comparison artifact
- agreement/difference interpretation notes
- bounded final-fix decision input

### Completion Criteria
- Sprint 89 has a real comparison package, not just internal validation
- the comparison result is explicit enough to justify or retire the final fix
  batch
- Day 10 can design the last landing from evidence

---

## Day 10: Final Cross-Surface Fix Design

**Title:** Final Fix Design  
**Theme:** Freeze the smallest truthful last-mile reconciliation batch  
**Time estimate:** 12 hours

### Tasks
1. Review the Day 6 re-audit and Day 9 comparison results together.
2. Decide whether a final cross-surface fix batch is required.
3. If required, fix:
   - the exact implementation center
   - the exact support-only follow-through set
   - the exact validation gate
4. If not required, document that the final fix batch is intentionally empty
   and move directly to closeout calibration.
5. Record the design in working notes and a Day 10 artifact.

### Deliverables
- final-fix design artifact
- exact touch set or explicit empty-batch decision
- validation expectations for the final landing

### Completion Criteria
- the last fix batch is bounded or explicitly retired
- support-only churn is prevented before the final validation sweep
- Day 11 can execute one exact landing

---

## Day 11: Final Cross-Surface Fix Batch

**Title:** Final Fix Batch  
**Theme:** Land the highest-value last-mile reconciliations found by the
re-audit and comparison sweep  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded final-fix batch from Day 10, if one is required.
2. Apply only the directly forced support-surface follow-through.
3. Run the required validation gate for the touched surfaces.
4. Record exactly what was fixed, what stayed intentionally deferred, and why.
5. Write the Day 11 artifact and working-notes entry.

### Deliverables
- landed final fix batch or explicit no-op confirmation
- touched-surface validation results
- final contradiction-resolution notes

### Completion Criteria
- the last real contradictions are either fixed or explicitly calibrated
- no unsupported “close enough” claims remain
- the branch is ready for residual-queue and full-close validation work

---

## Day 12: Residual Queue Finalization & Closeout Design

**Title:** Residual Freeze  
**Theme:** Calibrate the post-Epic-8 carry-forward queue and freeze the final
validation/reporting close path  
**Time estimate:** 12 hours

### Tasks
1. Write the truthful post-Epic-8 residual queue:
   - real remaining work
   - deliberate non-claims
   - lower-value deferred ideas
2. Freeze the exact Day 13 validation and reporting queue:
   - strongest reviewed baseline
   - install/export proof
   - canonical benchmark reporting
   - touched follow-on proofs
3. Freeze the Day 14 closeout-writing scope:
   - Sprint 89 retrospective
   - Epic 8 closeout notes
   - final project summary
4. Record the calibration in working notes and a Day 12 artifact.
5. Reconfirm that the closeout package will be evidence-based and bounded.

### Deliverables
- residual-queue finalization artifact
- frozen Day 13 validation/reporting queue
- frozen Day 14 closeout-writing queue

### Completion Criteria
- the post-Epic-8 carry-forward queue is explicit before final validation
- the final validation run list is fixed in writing
- the closeout-writing package has a bounded truthful scope

---

## Day 13: Full Validation & Reporting Sweep

**Title:** Validation Sweep  
**Theme:** Run the strongest reviewed baseline, maintained consumer proof, and
reporting surfaces for the Epic 8 close baseline  
**Time estimate:** 12 hours

### Tasks
1. Run the strongest reviewed baseline:
   - `make quality-review-full`
2. Run and record the maintained install/export proof:
   - `bash tests/test_install.sh`
   - `bash tests/test_cmake_install.sh`
3. Run and record the maintained reporting surface:
   - `make bench-canonical-report`
4. Run any touched follow-on proofs required by the final fix batch.
5. Write the Day 13 validation artifact with exact outcomes, parity anchors,
   and any non-blocking residual runtime notes.

### Deliverables
- full validation-and-reporting artifact
- final Epic 8 close baseline metrics
- exact touched-surface proof results

### Completion Criteria
- the strongest maintained validation surfaces all pass
- the close baseline is explicit and sourceable from one artifact
- Day 14 can close Sprint 89 and Epic 8 from a validated state

---

## Day 14: Retrospective, Handoff & Epic 8 Closeout

**Title:** Closeout  
**Theme:** Finish Sprint 89, close Epic 8, and publish the final project
summary surfaces from the validated end state  
**Time estimate:** 12 hours

### Tasks
1. Write the Sprint 89 closeout and handoff artifact.
2. Write the Sprint 89 retrospective from the sprint artifacts and working
   notes.
3. Write the Epic 8 closeout notes from the validated end state.
4. Write the final project-summary surface for the post-Epic-8 tree.
5. Reconfirm the final residual queue and next-cycle planning handoff.

### Deliverables
- Sprint 89 closeout artifact
- Sprint 89 retrospective
- Epic 8 closeout notes
- final project-summary surface

### Completion Criteria
- Sprint 89 closes from the validated Day 13 baseline
- Epic 8 has one explicit closeout package
- the next planning cycle inherits a truthful residual queue and final summary

