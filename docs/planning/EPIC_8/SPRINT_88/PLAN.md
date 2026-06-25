# Sprint 88 Plan: Front-Door Usability & Workflow Simplification

**Sprint Duration:** 14 days  
**Goal:** Make the library easier to adopt without weakening the truthfulness
and proof-ownership discipline built in earlier epics. This sprint implements
the Sprint 88 section of `docs/planning/EPIC_8/PROJECT_PLAN.md`.

**Starting Point:** Sprint 87 closed with a sharper static-first package and
consumer contract, stronger local install/export proof, and better-bounded
workflow/platform language. The strongest remaining first-tier Epic 8
contradiction is now front-door usability:
- the repo has a stronger install/export and package story than earlier Epic 8
  phases, but the first user path across README, tutorial, examples, install,
  and benchmark references still carries too much policy density
- audience boundaries across README, examples, benchmark references, install
  guidance, and maintainer-only material are still blurrier than they should be
- the highest-signal public header narratives still expose more internal policy
  than an adoption-focused front door needs
- Sprint 87 stabilized the package/platform contract, so Sprint 88 can now
  simplify the adoption path without re-deciding install/export semantics
- the strongest local reviewed baseline remains `make quality-review-full`

The highest-value Sprint 88 work is therefore not generic “improve docs.” It
is one bounded front-door usability and workflow-simplification package that:
- re-ranks the strongest remaining adoption-friction contradictions
- defines one explicit usability contract for support-surface layering and
  advanced-doc separation
- tightens README and tutorial/front-door guidance around user decisions
- simplifies examples and workflow references without losing correctness
  guidance
- consolidates benchmark/install/support references around clearer audience
  boundaries
- reduces remaining internal-policy leakage from the highest-signal public
  header narratives

**End State:** Sprint 88 leaves behind:
- a refreshed front-door usability contradiction map
- one explicit usability and support-layering design contract
- one landed README / tutorial simplification batch
- one landed examples / workflow simplification batch
- one clearer support-surface audience split
- one bounded public narrative cleanup on the highest-signal headers
- one aligned close baseline and handoff package

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the practical 14-day cap while preserving
the Sprint 88 scope and ordering from the project plan.

---

## Day 1: Sprint 88 Scope Audit & Front-Door Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 88 project-plan section and Sprint 87 closeout into
one bounded front-door usability execution package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 88 section of
   `docs/planning/EPIC_8/PROJECT_PLAN.md`, the Sprint 87 closeout artifact,
   and the Sprint 87 retrospective.
2. Reconfirm the preserved Sprint 88 starting assumptions:
   - Sprint 81 already clarified earlier product/workflow shape
   - Sprint 87 fixed the bounded package/platform contract
   - Sprint 88 should simplify adoption without widening claims
   - advanced guidance should stay available without dominating the front door
3. Define the Sprint 88 workstreams explicitly:
   - user-journey audit
   - workflow-simplification design
   - README / tutorial batch
   - examples / workflow batch
   - support-surface consolidation
   - header / API narrative cleanup
   - validation and closeout
4. Record the strongest likely Sprint 88 touch surfaces:
   - `README.md`
   - install and support docs
   - examples and example references
   - benchmark reference surfaces
   - highest-signal public headers
5. Open Sprint 88 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 88 scope inventory
- front-door usability workstream map
- starting working-notes baseline

### Completion Criteria
- Sprint 88 starts from the validated Sprint 87 end state
- the first usability contradiction is explicit before deeper audit begins
- the non-goal fence is visible before design or implementation work

---

## Day 2: Validation & Maintained Support-Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, install/export, example, and
support-surface ownership split before usability changes begin  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline and implementation-day gate:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm the maintained package, example, and install/export proof
   surfaces that Sprint 88 must not blur:
   - `tests/test_install.sh`
   - `tests/test_cmake_install.sh`
   - representative reviewed examples
   - canonical benchmark/reporting surfaces
3. Recheck reviewed CMake parity, install/export proof ownership, benchmark
   ownership, and workflow ownership so Sprint 88 keeps correctness evidence
   separate from usability cleanup.
4. Fix the authoritative rerun list most likely to matter throughout Sprint
   88.
5. Record the validation / maintained-surface split in working notes and a Day
   2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- preserved support-surface ownership map
- authoritative Sprint 88 rerun list

### Completion Criteria
- the strongest local validation contract is explicit before implementation
  work lands
- proof ownership across reviewed tests, install/export checks, examples, and
  support docs is fixed in writing
- later code or docs days have no ambiguity about the required validation gate

---

## Day 3: User-Journey Audit

**Title:** Journey Audit  
**Theme:** Reduce the front-door usability problem to one ranked live
contradiction map  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the highest-signal adoption surfaces:
   - `README.md`
   - install guidance
   - example references
   - benchmark references
   - maintainer guide entry points
   - highest-signal public headers
2. Capture where the current front door is still hardest to adopt:
   - decision overload
   - unclear audience splits
   - example discoverability friction
   - benchmark/install reference clutter
   - policy density in public narratives
3. Separate:
   - strongest first-batch implementation center
   - second-tier example and support-surface follow-through seams
   - support-only wording surfaces
   - deliberate non-goals
4. Reconcile the rerank against Sprint 87’s package/platform closeout.
5. Write the ranked usability-gap artifact.

### Deliverables
- ranked front-door usability artifact
- first-tier vs deferred contradiction map
- Sprint 87 carry-forward reconciliation notes

### Completion Criteria
- Sprint 88’s broad usability problem is reduced to one ranked live map
- the strongest implementation center is explicit before design
- lower-value spillover work is clearly separated from the first lane

---

## Day 4: First Usability Boundary Freeze

**Title:** Boundary Freeze  
**Theme:** Fix the first bounded Sprint 88 implementation fence and the
allowed support-surface movement  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 usability ranking against the Sprint 88 project-plan
   scope.
2. Decide the required first implementation center:
   - README / tutorial simplification first
   - examples / workflow simplification second
   - support-surface consolidation and public narrative cleanup later unless
     truly forced
3. Decide which support surfaces move only if forced:
   - install docs
   - example references
   - benchmark docs
   - maintainer guide wording
   - high-signal public headers
4. Fix the preserved non-goal fence for the first landing:
   - no package/platform contract reopening
   - no correctness ownership redistribution
   - no benchmark-policy rewrite detached from adoption guidance
   - no internal architectural rewrite disguised as docs cleanup
5. Record the first implementation fence in working notes and a Day 4
   artifact.

### Deliverables
- first usability-boundary artifact
- required vs support-only touch set
- preserved first-batch non-goal fence

### Completion Criteria
- Sprint 88 has one explicit first landing boundary
- support-only surfaces are clearly separated from the batch center
- Day 5 can design one front-door contract instead of a broad doc rewrite

---

## Day 5: Workflow-Simplification Design

**Title:** Usability Design  
**Theme:** Define the bounded adoption-guidance and support-layering contract
Sprint 88 will actually support  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 boundary and the strongest user-journey contradictions.
2. Define the ownership split for:
   - front-door guidance surfaces
   - example/workflow adoption surfaces
   - support-only advanced references
   - maintainer-only detail surfaces
3. Decide the layering rules for:
   - what belongs in the README front door
   - what belongs in examples
   - what belongs in install/support docs
   - what belongs only in maintainer-facing material
4. Fix the touch fence for docs, examples, and public-header narrative
   surfaces.
5. Write the Day 5 usability-design artifact and working-notes summary.

### Deliverables
- explicit workflow-simplification design contract
- ownership split for touched front-door seams
- preserved bounded-scope and non-goal fence

### Completion Criteria
- Sprint 88 has one explicit implementation contract
- ownership between front-door, example, support, and maintainer surfaces is
  clear
- Day 6 can implement one bounded landing without reopening design questions

---

## Day 6: README / Tutorial Batch

**Title:** Front-Door Batch  
**Theme:** Tighten the README and top-level adoption guidance around user
decisions instead of policy density  
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-value README / tutorial seam from the Day 5 contract.
2. Keep the landing bounded to the required first implementation center.
3. Preserve correctness, package, and reviewed proof ownership outside the
   touched front-door guidance seam.
4. Update only directly forced support surfaces.
5. Run the required implementation-day validation gate or targeted docs-day
   sanity checks, depending on the landed surface.

### Deliverables
- first landed README / tutorial simplification batch
- bounded support-surface follow-through if truly forced
- recorded validation or sanity-check result

### Completion Criteria
- one real front-door contradiction is closed in the repo
- the landed change matches the Day 5 fence
- required validation passes before the day closes

---

## Day 7: Post-Landing Audit & Re-Rank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining adoption-friction contradictions after the
first front-door batch lands  
**Time estimate:** 12 hours

### Tasks
1. Re-audit the touched usability surfaces after the Day 6 landing.
2. Decide whether the strongest remaining contradiction has shifted to:
   - examples / workflow simplification
   - support-surface consolidation
   - public header / API narrative cleanup
3. Confirm which next seam is highest-value and still bounded.
4. Reconfirm which lower-value front-door or support surfaces remain deferred.
5. Record the rerank in working notes and a Day 7 artifact.

### Deliverables
- post-landing rerank artifact
- updated next-step priority map
- refreshed deferred-work list

### Completion Criteria
- the next implementation center is explicit after the first landing
- Sprint 88 does not drift into generic follow-up churn
- Day 8 begins from a refreshed contradiction map rather than assumptions

---

## Day 8: Examples / Workflow Simplification Design

**Title:** Examples Design  
**Theme:** Define the bounded example and workflow-adoption package Sprint 88
will add next  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank and preserved non-goal fence.
2. Decide the exact second implementation center:
   - example discovery and ordering
   - workflow adoption references
   - example/support cross-link cleanup
3. Identify directly forced support-only surfaces:
   - example references
   - install/support docs
   - benchmark references only if the adoption contract truly changes
4. Freeze the boundaries for local examples vs advanced support references vs
   maintainer material.
5. Write the Day 8 design artifact and working-notes summary.

### Deliverables
- examples/workflow simplification design artifact
- exact second implementation contract
- support-only follow-through list

### Completion Criteria
- Day 9 has one exact example/workflow center
- support-only surfaces are bounded before implementation begins
- support-surface consolidation remains explicitly separate unless forced

---

## Day 9: Examples / Workflow Simplification Batch

**Title:** Examples Batch  
**Theme:** Simplify the example path and workflow adoption story without
losing correctness guidance  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 8 example/workflow contract.
2. Keep the landing bounded to adoption-flow improvement rather than broad
   product or correctness changes.
3. Avoid widening package, ABI, or platform claims beyond what the touched
   surfaces already maintain.
4. Update only directly forced support surfaces.
5. Run the required validation gate or targeted docs/example sanity checks for
   the touched surfaces.

### Deliverables
- landed examples/workflow simplification batch
- updated support surfaces if truly forced
- recorded validation or sanity-check result

### Completion Criteria
- the example and workflow story is easier to navigate than at sprint start
- the landed proof and guidance align with the Day 8 contract
- required validation passes before the day closes

---

## Day 10: Support-Surface Consolidation Design

**Title:** Support Design  
**Theme:** Define the bounded benchmark/install/support audience cleanup Sprint
88 can truthfully maintain  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 9 adoption path against current benchmark, install, and
   support wording.
2. Identify the exact audience-boundary gap still worth closing:
   - benchmark references in adoption surfaces
   - install references in README/example flow
   - support vs maintainer boundary blur
3. Separate:
   - one bounded support-surface consolidation seam
   - support-only advanced wording
   - deliberate non-goals that remain too broad for Sprint 88
4. Freeze the exact implementation center and support-only follow-through
   surfaces.
5. Write the Day 10 design artifact and working-notes summary.

### Deliverables
- support-surface consolidation design artifact
- exact Day 11 implementation center
- preserved audience-boundary fence

### Completion Criteria
- Day 11 has one exact bounded support-surface target
- audience claims remain tied to maintained evidence and purpose
- Sprint 88 avoids drifting into generic support-doc expansion

---

## Day 11: Support-Surface Consolidation Batch

**Title:** Support Batch  
**Theme:** Land one bounded audience-boundary improvement across install,
benchmark, and support references  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 10 support-surface consolidation seam.
2. Keep the landing inside the preserved audience-boundary fence.
3. Avoid introducing broader product, performance, or platform claims than the
   touched surfaces can sustain.
4. Update directly forced public-header or maintainer wording only if truly
   required by the landed consolidation.
5. Run the required validation gate or targeted docs-day sanity checks.

### Deliverables
- landed support-surface consolidation batch
- bounded docs/header follow-through if forced
- recorded validation or sanity-check result

### Completion Criteria
- one real support-surface audience split is improved
- the landed change matches the Day 10 contract
- required validation passes before the day closes

---

## Day 12: Header / API Narrative Cleanup & Validation Queue Freeze

**Title:** Narrative Freeze  
**Theme:** Reconcile high-signal public narrative surfaces and freeze the
exact final validation queue before the full sweep  
**Time estimate:** 12 hours

### Tasks
1. Re-audit the touched front-door, example, install, benchmark, and support
   surfaces after the implementation days.
2. Decide whether any final bounded public narrative reconciliation is still
   required in the highest-signal public headers.
3. Freeze the final Sprint 88 support and proof-owner map:
   - reviewed baseline owners
   - install/export proof owners
   - example/workflow adoption owners
   - support-only advanced references
4. Freeze the exact Day 13 validation queue.
5. Record the alignment pass in working notes and a Day 12 artifact.

### Deliverables
- final narrative/alignment artifact
- final Sprint 88 support and proof-owner map
- exact Day 13 validation queue

### Completion Criteria
- no ambiguity remains about front-door, example, support, and maintainer
  ownership
- final validation is fully specified before it starts
- any remaining support-only edits are either landed or explicitly unnecessary

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the complete Sprint 88 validation and usability-close baseline  
**Time estimate:** 12 hours

### Tasks
1. Run the full required validation gate if any code or header surfaces moved:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Run the exact docs/example/package proof reruns frozen on Day 12.
3. Reconfirm reviewed CMake parity, maintained install/export proof, and any
   touched example/workflow surfaces.
4. Record key outputs, usability outcomes, and any residual bounded debt.
5. Write the Day 13 validation artifact and working-notes summary.

### Deliverables
- full validation-sweep artifact
- refreshed reviewed/example/package close baseline
- explicit residual-risk notes

### Completion Criteria
- the full Sprint 88 validation queue passes cleanly
- reviewed, example, and package anchors are explicit in writing
- only non-blocking residual debt remains going into closeout

---

## Day 14: Sprint 88 Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 88 from the validated baseline and hand off the final
Epic 8 queue cleanly  
**Time estimate:** 12 hours

### Tasks
1. Reconcile Sprint 88 outcomes against the original project-plan section.
2. Record what actually landed across:
   - user-journey rerank
   - workflow-simplification design contract
   - README / tutorial batch
   - examples / workflow batch
   - support-surface consolidation
   - public narrative cleanup
   - validation
3. Decide whether `docs/planning/EPIC_8/PROJECT_PLAN.md` needs any bounded
   correction.
4. Write the Sprint 88 closeout and handoff artifact with the next recommended
   Epic 8 order.
5. Update working notes with the final sprint-close state.

### Deliverables
- Sprint 88 closeout and handoff artifact
- final working-notes close summary
- explicit carry-forward queue into Sprint 89

### Completion Criteria
- Sprint 88 closes from a validated baseline rather than implementation intent
- the next Epic 8 queue is explicit and bounded
- no Sprint 88 ambiguity remains in the handoff package
