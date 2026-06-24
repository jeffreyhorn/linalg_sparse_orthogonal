# Sprint 87 Plan: Packaging, ABI & Cross-Platform Quality Convergence Phase 3

**Sprint Duration:** 14 days  
**Goal:** Strengthen the install/export/consumer story, reduce package and
workflow asymmetry across maintained platforms, and decide whether the library
remains permanently static-first or earns one bounded shared-library product
lane. This sprint implements the Sprint 87 section of
`docs/planning/EPIC_8/PROJECT_PLAN.md`.

**Starting Point:** Sprint 86 closed from a validated reviewed-runtime baseline
with a materially smaller reviewed long pole and cleaner runtime evidence on
the ND / reorder lane. The strongest remaining first-tier Epic 8 contradiction
is now package / ABI / consumer truthfulness:
- the install/export story is stronger than earlier Epic 8 phases, but it is
  still narrower and more asymmetric than the reviewed quality story
- the static/shared contract and downstream-consumer promise are not yet fixed
  with enough precision for a durable long-term package claim
- cross-platform workflow wording still needs to be constrained to what the
  repo can realistically maintain
- Sprint 80 already fixed the first packaging/platform direction, so Sprint 87
  should widen only where bounded evidence justifies it
- the strongest local reviewed baseline remains `make quality-review-full`

The highest-value Sprint 87 work is therefore not generic “improve packaging.”
It is one bounded package / ABI / install-export modernization package that:
- re-ranks the strongest remaining package and consumer contradictions
- defines one explicit static/shared, ABI, and downstream-consumer contract
- lands one bounded packaging batch for the chosen contract
- strengthens local install/export proof and maintained consumer evidence
- adds only realistic workflow / platform follow-through
- reconciles support surfaces with the widened package/platform contract

**End State:** Sprint 87 leaves behind:
- a refreshed package / ABI / consumer gap map
- one explicit product-matrix design contract
- one landed packaging / export modernization batch
- one stronger local install/export and downstream-consumer proof package
- one bounded workflow / platform follow-through package
- one aligned support-surface closeout and validated baseline

**Time budget:** Each day is capped at 12 hours as requested. Because that cap
allows at most `168` hours across 14 days, this day-by-day plan totals `168`
hours rather than the slightly higher project-plan estimate of `~170` hours,
while preserving the Sprint 87 scope and ordering.

---

## Day 1: Sprint 87 Scope Audit & Packaging Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 87 project-plan section and Sprint 86 closeout into
one bounded packaging / ABI / consumer execution package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 87 section of
   `docs/planning/EPIC_8/PROJECT_PLAN.md`, the Sprint 86 closeout artifact, and
   the Sprint 86 retrospective.
2. Reconfirm the preserved Sprint 87 starting assumptions:
   - Sprint 80 already fixed the earlier packaging/platform direction
   - Sprint 84 proof-expansion work is available for any truthful widened
     platform claim
   - Sprint 87 should not promise broader product support than the repo can
     maintain
   - Sprint 87 should not weaken the static-first path unless a bounded shared
     lane earns explicit proof
3. Define the Sprint 87 workstreams explicitly:
   - release / package gap audit
   - product-matrix design
   - packaging batch
   - consumer-proof expansion
   - workflow / platform follow-through
   - support-surface alignment
   - validation and closeout
4. Record the strongest likely Sprint 87 touch surfaces:
   - install/export scripts
   - `CMakeLists.txt`
   - `Makefile`
   - package / install docs
   - workflow surfaces
   - downstream consumer proof owners
5. Open Sprint 87 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 87 scope inventory
- package / ABI / consumer workstream map
- starting working-notes baseline

### Completion Criteria
- Sprint 87 starts from the validated Sprint 86 end state
- the first packaging contradiction is explicit before deeper audit begins
- the non-goal fence is visible before design or implementation work

---

## Day 2: Validation & Maintained Consumer-Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest install/export, consumer-proof, workflow, and
reviewed validation split before packaging changes begin  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline and implementation-day gate:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm the maintained package / consumer proof surfaces:
   - `tests/test_install.sh`
   - `tests/test_cmake_install.sh`
   - representative local consumer or export verification surfaces
3. Recheck reviewed CMake parity, install/export proof ownership, benchmark
   ownership, and workflow ownership so Sprint 87 does not blur correctness vs
   packaging evidence.
4. Fix the authoritative rerun list most likely to matter throughout Sprint
   87.
5. Record the validation / maintained-surface split in working notes and a Day
   2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- preserved install/export and consumer-proof map
- authoritative Sprint 87 rerun list

### Completion Criteria
- the strongest local validation contract is explicit before implementation
  work lands
- proof ownership across reviewed tests, install/export checks, and workflow
  surfaces is fixed in writing
- later code days have no ambiguity about the required validation gate

---

## Day 3: Release / Package Gap Audit

**Title:** Gap Audit  
**Theme:** Reduce the packaging and ABI problem to one ranked live
contradiction map  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the highest-signal package and consumer surfaces:
   - `CMakeLists.txt`
   - `Makefile`
   - install/export scripts
   - package / install docs
   - workflow surfaces
2. Capture where the current package/platform ceiling is strongest:
   - static/shared ambiguity
   - ABI-language ambiguity
   - export-surface incompleteness
   - downstream consumer friction
   - cross-platform workflow asymmetry
3. Separate:
   - strongest first-batch implementation center
   - second-tier consumer and workflow follow-through seams
   - support-only docs / wording surfaces
   - deliberate non-goals
4. Reconcile the rerank against Sprint 80’s earlier package direction and
   Sprint 86’s close handoff.
5. Write the ranked package-gap artifact.

### Deliverables
- ranked package / ABI / consumer artifact
- first-tier vs deferred contradiction map
- Sprint 80/86 carry-forward reconciliation notes

### Completion Criteria
- Sprint 87’s broad packaging problem is reduced to one ranked live map
- the strongest implementation center is explicit before design
- lower-value spillover work is clearly separated from the first lane

---

## Day 4: First Packaging / ABI Boundary Freeze

**Title:** Boundary Freeze  
**Theme:** Fix the first bounded Sprint 87 implementation fence and the
allowed package / workflow movement  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 package-gap ranking against the Sprint 87 project-plan
   scope.
2. Decide the required first implementation center:
   - packaging/export batch first
   - consumer-proof expansion second
   - workflow/platform follow-through only where the first batch truly forces
     it
3. Decide which support surfaces move only if forced:
   - install/export proof scripts
   - workflow files
   - `README.md`
   - `INSTALL.md`
   - maintainer guide wording
4. Fix the preserved non-goal fence for the first landing:
   - no platform claims without maintained proof
   - no generic build-system churn detached from the chosen product contract
   - no broad ABI promises without bounded evidence
   - no support-surface churn detached from a real landed packaging seam
5. Record the first implementation fence in working notes and a Day 4
   artifact.

### Deliverables
- first packaging/ABI-boundary artifact
- required vs support-only touch set
- preserved first-batch non-goal fence

### Completion Criteria
- Sprint 87 has one explicit first landing boundary
- support-only surfaces are clearly separated from the batch center
- Day 5 can design one package contract instead of a broad release rewrite

---

## Day 5: Product-Matrix Design

**Title:** Product Design  
**Theme:** Define the bounded static/shared, ABI, and downstream-consumer
contract Sprint 87 will actually support  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 boundary and the strongest package / consumer
   contradictions.
2. Define the ownership split for:
   - product-matrix contract surfaces
   - package/export implementation owners
   - consumer-proof owners
   - workflow/platform evidence owners
3. Decide whether Sprint 87 remains:
   - static-first only, with stronger export and consumer truth
   - or one bounded shared-library lane, with explicit ABI and platform limits
4. Fix the touch fence for code, build, scripts, workflows, and docs.
5. Write the Day 5 product-matrix artifact and working-notes design summary.

### Deliverables
- explicit product-matrix design contract
- ownership split for touched package / ABI seams
- preserved bounded-scope and non-goal fence

### Completion Criteria
- Sprint 87 has one explicit implementation contract
- ownership between build surfaces, consumer proof, and workflow evidence is
  clear
- Day 6 can implement one bounded landing without reopening design questions

---

## Day 6: Packaging Batch 1

**Title:** Packaging Batch  
**Theme:** Land the highest-value packaging/export modernization required by
the chosen product contract  
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-value packaging/export seam from the Day 5 contract.
2. Keep the landing bounded to the required first implementation center.
3. Preserve the existing correctness and reviewed proof-owner split outside the
   touched package seam.
4. Update only directly forced package or build support surfaces.
5. Run the required implementation-day validation gate.

### Deliverables
- first landed packaging/export batch
- bounded support-surface follow-through if truly forced
- recorded validation result

### Completion Criteria
- one real packaging/export contradiction is closed in the repo
- the landed change matches the Day 5 fence
- required validation passes before the day closes

---

## Day 7: Post-Landing Audit & Re-Rank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining package / consumer contradictions after the
first packaging batch lands  
**Time estimate:** 12 hours

### Tasks
1. Re-audit the touched package surfaces after the Day 6 landing.
2. Decide whether the strongest remaining contradiction has shifted to:
   - consumer-proof expansion
   - workflow/platform follow-through
   - support-surface alignment
3. Confirm which next seam is highest-value and still bounded.
4. Reconfirm which lower-value package or platform surfaces remain deferred.
5. Record the rerank in working notes and a Day 7 artifact.

### Deliverables
- post-landing rerank artifact
- updated next-step priority map
- refreshed deferred-work list

### Completion Criteria
- the next implementation center is explicit after the first landing
- Sprint 87 does not drift into generic follow-up churn
- Day 8 begins from a refreshed contradiction map rather than assumptions

---

## Day 8: Consumer-Proof Expansion Design

**Title:** Consumer Design  
**Theme:** Define the bounded install/export and downstream-consumer proof
package Sprint 87 will add next  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank and preserved non-goal fence.
2. Decide the exact second implementation center:
   - local install/export proof strengthening
   - downstream consumer proof strengthening
   - both only where one bounded package can own them cleanly
3. Identify directly forced support-only surfaces:
   - install/export scripts
   - example consumer surfaces
   - docs only where the proof contract truly changes
4. Freeze the boundaries for local proof vs workflow claims vs docs wording.
5. Write the Day 8 design artifact and working-notes summary.

### Deliverables
- consumer-proof expansion design artifact
- exact second implementation contract
- support-only follow-through list

### Completion Criteria
- Day 9 has one exact proof-expansion center
- support-only surfaces are bounded before implementation begins
- workflow/platform widening remains explicitly separate unless forced

---

## Day 9: Consumer-Proof Expansion Batch

**Title:** Consumer Batch  
**Theme:** Strengthen the maintained install/export and downstream-consumer
story for the chosen product contract  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 8 consumer-proof contract.
2. Keep the landing bounded to maintained local install/export and consumer
   evidence.
3. Avoid widening platform or ABI claims beyond what the touched proof truly
   supports.
4. Update only directly forced support surfaces.
5. Run the required implementation-day validation gate and any package-proof
   reruns owned by the touched surfaces.

### Deliverables
- landed consumer-proof expansion batch
- updated proof-owner surfaces if truly forced
- recorded validation and consumer-proof result

### Completion Criteria
- the maintained consumer story is stronger and more explicit than at sprint
  start
- the landed proof aligns with the chosen product contract
- required validation passes before the day closes

---

## Day 10: Workflow / Platform Follow-Through Design

**Title:** Workflow Design  
**Theme:** Define the bounded cross-platform quality convergence package Sprint
87 can truthfully maintain  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 9 proof expansion against current workflow and platform
   wording.
2. Identify the exact workflow/platform gap still worth closing:
   - Windows/macOS/Linux asymmetry
   - shared vs static workflow truth
   - install/export proof gaps in CI or documented local flows
3. Separate:
   - one bounded workflow/platform follow-through seam
   - support-only docs wording
   - deliberate non-goals that remain too broad for Sprint 87
4. Freeze the exact implementation center and support-only follow-through
   surfaces.
5. Write the Day 10 design artifact and working-notes summary.

### Deliverables
- workflow/platform follow-through design artifact
- exact Day 11 implementation center
- preserved realism fence for platform claims

### Completion Criteria
- Day 11 has one exact bounded workflow/platform target
- platform claims remain tied to maintained evidence
- Sprint 87 avoids drifting into generic CI expansion

---

## Day 11: Workflow / Platform Follow-Through Batch

**Title:** Workflow Batch  
**Theme:** Land one bounded workflow/platform convergence improvement for the
chosen package contract  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 10 workflow/platform follow-through seam.
2. Keep the landing inside the realism fence for maintained evidence.
3. Avoid introducing broader product or ABI claims than the touched workflow
   can sustain.
4. Update only directly forced docs or maintainer wording if truly required by
   the landed workflow change.
5. Run the required validation gate and any touched workflow/package proof
   reruns.

### Deliverables
- landed workflow/platform follow-through batch
- bounded docs/workflow follow-through if forced
- recorded validation result

### Completion Criteria
- one real workflow/platform asymmetry is reduced
- the landed change matches the Day 10 contract
- required validation passes before the day closes

---

## Day 12: Support-Surface Alignment & Validation Queue Freeze

**Title:** Alignment Freeze  
**Theme:** Reconcile support wording and freeze the exact final validation
queue before the full sweep  
**Time estimate:** 12 hours

### Tasks
1. Re-audit the touched package, consumer, and workflow surfaces after the
   implementation days.
2. Decide whether any final support-only reconciliation is still required in:
   - `README.md`
   - `INSTALL.md`
   - maintainer guide
   - workflow notes
3. Freeze the final Sprint 87 proof-owner map:
   - reviewed baseline owners
   - install/export proof owners
   - downstream-consumer proof owners
   - bounded workflow/platform evidence owners
4. Freeze the exact Day 13 validation queue.
5. Record the alignment pass in working notes and a Day 12 artifact.

### Deliverables
- support-surface alignment artifact
- final Sprint 87 proof-owner map
- exact Day 13 validation queue

### Completion Criteria
- no ambiguity remains about package, consumer, and workflow ownership
- final validation is fully specified before it starts
- any remaining support-only edits are either landed or explicitly unnecessary

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the complete Sprint 87 validation and package-proof close
baseline  
**Time estimate:** 12 hours

### Tasks
1. Run the full required validation gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Run the exact package/consumer/workflow proof reruns frozen on Day 12.
3. Reconfirm reviewed CMake parity and the maintained install/export proof.
4. Record key outputs, runtime notes, package outcomes, and any residual
   bounded debt.
5. Write the Day 13 validation artifact and working-notes summary.

### Deliverables
- full validation-sweep artifact
- refreshed reviewed and package-proof baseline
- explicit residual-risk notes

### Completion Criteria
- the full Sprint 87 validation queue passes cleanly
- reviewed and package-proof anchors are explicit in writing
- only non-blocking residual debt remains going into closeout

---

## Day 14: Sprint 87 Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 87 from the validated baseline and hand off the next
Epic 8 queue cleanly  
**Time estimate:** 12 hours

### Tasks
1. Reconcile Sprint 87 outcomes against the original project-plan section.
2. Record what actually landed across:
   - package-gap rerank
   - product-matrix contract
   - packaging batch
   - consumer-proof expansion
   - workflow/platform follow-through
   - support-surface alignment
   - validation
3. Decide whether `docs/planning/EPIC_8/PROJECT_PLAN.md` needs any bounded
   correction.
4. Write the Sprint 87 closeout and handoff artifact with the next recommended
   Epic 8 order.
5. Update working notes with the final sprint-close state.

### Deliverables
- Sprint 87 closeout artifact
- next-step Epic 8 handoff queue
- final working-notes close state

### Completion Criteria
- Sprint 87 closes from a validated package / consumer baseline
- the next Epic 8 queue is explicit and evidence-based
- the sprint can transition cleanly into retrospective and PR packaging
