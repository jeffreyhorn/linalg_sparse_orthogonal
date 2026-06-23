# Sprint 84 Plan: Numerical Assurance & Differential Testing Phase 2

**Sprint Duration:** 14 days  
**Goal:** Strengthen assurance with maintained external differential checks,
broader property testing, and better failure-path numerical proof. This sprint
implements the Sprint 84 section of
`docs/planning/EPIC_8/PROJECT_PLAN.md`.

**Starting Point:** Sprint 83 closed from a validated capability-surface
baseline with the highest-value shared scalar-owner seams widened on the
matrix shell, the shared scalar/index vocabulary reconciled, and one bounded
QR public-header widening landed without overstating numeric breadth. The
strongest remaining first-tier Epic 8 contradiction is now assurance depth:
- the touched shared/direct lanes need stronger maintained external
  differential proof
- deterministic property coverage is still narrower than the current public
  lifecycle surface
- the most fragile failure-path numerical guarantees still rely too heavily on
  bounded local tests rather than a stronger cross-check architecture
- Sprint 83’s capability widening should not be reopened as another generic
  public-surface rewrite
- the strongest local reviewed baseline remains `make quality-review-full`

The highest-value Sprint 84 work is therefore not generic “add more tests.”
It is one bounded assurance-modernization package that:
- reranks the highest-value direct, iterative, and eigensolver lanes for
  maintained external differential proof
- fixes one explicit oracle/property/failure-path architecture contract
- lands the first external comparison harnesses on the strongest core direct
  lanes
- extends deterministic property coverage beyond the current bounded lifecycle
  seams
- adds focused cancellation, error-path, and stress-fixture proof where public
  lifecycle guarantees are most fragile
- reconciles proof ownership, CI reading, and support-surface wording without
  overstating repo-wide oracle maturity

**End State:** Sprint 84 leaves behind:
- a refreshed differential-proof and assurance-priority map
- one explicit oracle/property/failure-path architecture contract
- one landed bounded direct-family external differential batch
- one focused seeded-property expansion package
- one focused failure-path numerical proof package
- one focused policy/CI/support-surface alignment package
- one validated closeout baseline and Sprint 85-ready handoff

**Time budget:** Each day is capped at 12 hours as requested. Because that cap
allows at most `168` hours across 14 days, this day-by-day plan totals `168`
hours rather than the higher project-plan estimate of `~172` hours, while
preserving the Sprint 84 scope and ordering.

---

## Day 1: Sprint 84 Scope Audit & Assurance Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 84 project-plan section and Sprint 83 closeout into
one bounded numerical-assurance execution package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 84 section of
   `docs/planning/EPIC_8/PROJECT_PLAN.md`, the Sprint 83 closeout artifact, and
   the Sprint 83 retrospective.
2. Reconfirm the preserved Sprint 84 starting assumptions:
   - Sprint 80 already fixed the external-oracle contract fence
   - Sprint 83 already fixed the strongest touched capability-surface seams
   - Sprint 84 should not reopen generic capability widening
   - Sprint 84 should not inflate CI or support-surface claims beyond what one
     maintained assurance lane can truly support
3. Define the Sprint 84 workstreams explicitly:
   - differential-proof audit
   - oracle/property architecture design
   - direct-family differential batch
   - seeded property expansion
   - failure-path numerical proof
   - policy/CI/support-surface alignment
   - validation and closeout
4. Record the strongest likely Sprint 84 touch surfaces:
   - direct-family proof owners
   - iterative/eigs property owners
   - oracle harness and fixture owners
   - CI/support-surface wording owners
5. Open Sprint 84 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 84 scope inventory
- assurance workstream map
- starting working-notes baseline

### Completion Criteria
- Sprint 84 starts from the validated Sprint 83 end state
- the first assurance contradiction is explicit before deeper audit begins
- the non-goal fence is visible before design or implementation work

---

## Day 2: Validation & Proof-Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, proof-owner, benchmark, and
install/export validation split before assurance changes begin  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline and implementation-day gate:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm reviewed CMake parity, canonical benchmark-report ownership, and
   install/export proof ownership.
3. Recheck the representative proof-owner surfaces most likely to move during
   Sprint 84:
   - direct-family tests
   - iterative/eigs tests with retained scalar-owner seams
   - representative examples
   - canonical benchmark/reporting command surfaces
   - install/export proof scripts
4. Fix the authoritative rerun list most likely to matter throughout Sprint
   84.
5. Record the validation/proof split in working notes and a Day 2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- preserved proof-owner map
- authoritative Sprint 84 rerun list

### Completion Criteria
- the strongest local validation contract is explicit before implementation
  work lands
- proof ownership across reviewed tests, benchmarks, and install/export
  surfaces is fixed in writing
- later code days have no ambiguity about the required validation gate

---

## Day 3: Differential-Proof Audit

**Title:** Differential Audit  
**Theme:** Re-rank the highest-value direct, iterative, and eigensolver lanes
for maintained external differential proof  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the highest-signal assurance surfaces:
   - direct-family solve and factor proof owners
   - iterative and eigensolver proof owners
   - existing oracle-touching or cross-check-friendly fixtures
   - failure-path lifecycle surfaces
2. Identify where the current assurance ceiling is strongest:
   - missing maintained external comparisons
   - weak deterministic property coverage
   - fragile error-path or cancellation guarantees
   - benchmark or example surfaces that should not become correctness owners
3. Separate:
   - strongest first-batch implementation center
   - second-tier property and failure-path follow-through seams
   - support-only CI/docs/package surfaces
   - deliberate non-goals
4. Reconcile the audit against the Sprint 80 oracle fence and Sprint 83
   capability closeout.
5. Write the ranked assurance artifact.

### Deliverables
- ranked differential-proof artifact
- first-tier vs deferred seam map
- Sprint 80/83 carry-forward reconciliation notes

### Completion Criteria
- Sprint 84’s broad assurance problem is reduced to one ranked live map
- the strongest implementation center is explicit before boundary design
- lower-value spillover work is clearly separated from the first lane

---

## Day 4: First Assurance Boundary Freeze

**Title:** Boundary Freeze  
**Theme:** Fix the first bounded oracle/property/failure-path implementation
fence and the allowed external-proof reading  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 ranking against the Sprint 84 project-plan scope.
2. Decide the required first implementation center:
   - direct-family differential harnesses first
   - deterministic property expansion second
   - failure-path proof only where the first batch truly forces it
3. Decide which support surfaces move only if forced:
   - proof-owner tests
   - CI wording
   - support docs
   - package/runtime surfaces
4. Fix the preserved non-goal fence for the first landing:
   - no broad oracle dependency story for untouched families
   - no benchmark-governance drift into correctness ownership
   - no repo-wide claim that every solver now has external maintained proof
   - no support-surface churn detached from a real landed proof seam
5. Record the first implementation fence in working notes and a Day 4 artifact.

### Deliverables
- first assurance-boundary artifact
- required vs support-only touch set
- preserved first-batch non-goal fence

### Completion Criteria
- Sprint 84 has one explicit first landing boundary
- support-only surfaces are clearly separated from the batch center
- Day 5 can design one assurance contract instead of a broad proof rewrite

---

## Day 5: Oracle / Property / Failure-Path Architecture Design

**Title:** Assurance Design  
**Theme:** Define the bounded external-differential, seeded-property, and
failure-path contract Sprint 84 will actually land  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 boundary and the strongest assurance contradictions.
2. Define the ownership split for:
   - maintained external-oracle harnesses
   - deterministic seeded property generators
   - failure-path invariant and cancellation checks
   - direct-family vs iterative/eigs adoption boundaries
3. Decide how the first landing preserves the Sprint 80 oracle fence while
   widening assurance depth on touched lanes.
4. Fix the touch fence for tests, CI wording, docs, and package/runtime
   surfaces.
5. Write the Day 5 architecture artifact and working-notes design summary.

### Deliverables
- oracle/property/failure-path architecture contract
- ownership split for touched seams
- preserved bounded-assurance and non-goal fence

### Completion Criteria
- Sprint 84 has one explicit implementation contract
- ownership between oracle harnesses, property coverage, and failure-path
  proof is clear
- Day 6 can implement one bounded landing without reopening design questions

---

## Day 6: Direct-Family Differential Batch 1

**Title:** Differential Batch  
**Theme:** Land the first bounded maintained external differential harnesses on
the highest-value core direct lanes  
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-value direct-family differential seam from the Day 5
   contract.
2. Keep the landing bounded to the required first implementation center.
3. Preserve the existing proof-owner split and avoid turning benchmarks or
   examples into correctness owners.
4. Update any truly forced local proof-owner tests or support surfaces.
5. Record the landing in working notes and a Day 6 artifact.
6. Run the required validation gate for touched code.

### Deliverables
- first maintained direct-family differential landing
- any forced focused regression follow-through
- Day 6 implementation artifact

### Completion Criteria
- the first bounded differential batch lands inside the Day 5 fence
- touched direct-family proof remains deterministic and support-surface-safe
- the required validation gate passes

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the strongest remaining assurance seam after the first
direct-family differential landing  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 implementation, proof, and touched support surfaces.
2. Identify what contradiction actually closed and what contradiction remains
   strongest:
   - more direct-family differential work
   - seeded property expansion
   - failure-path numerical proof
   - CI/support-surface alignment
3. Verify whether any support-only churn can still be deferred safely.
4. Fix the exact Day 8 design center and the strongest support-only follow-on
   surfaces.
5. Record the rerank and handoff artifact for the second half of Sprint 84.

### Deliverables
- post-landing audit artifact
- reranked remaining assurance queue
- exact second-half design center

### Completion Criteria
- the strongest remaining contradiction is reranked from the landed tree
- Day 8 has one explicit design target
- later work is guided by evidence instead of the original estimate buckets

---

## Day 8: Seeded Property Expansion Design

**Title:** Property Design  
**Theme:** Define the bounded deterministic property-expansion contract for the
highest-value retained lifecycle seams  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank and the current property-owner surfaces.
2. Decide which deterministic property lanes give the best Sprint 84 value:
   - repeated-run lifecycle invariants
   - reorder/factor/solve agreement properties
   - residual and invariance properties on touched public flows
3. Separate:
   - required implementation center
   - support-only direct-family follow-through
   - lower-value non-touch property ideas
4. Fix the exact Day 9 implementation contract.
5. Write the Day 8 design artifact and working-notes summary.

### Deliverables
- seeded-property design artifact
- required vs support-only property seam map
- exact Day 9 implementation contract

### Completion Criteria
- Sprint 84 has one explicit second implementation contract
- deterministic property expansion is bounded to the highest-value lifecycle
  seams
- Day 9 can land without reopening the direct-family boundary

---

## Day 9: Seeded Property Expansion Batch

**Title:** Property Batch  
**Theme:** Land the bounded deterministic property-expansion package on the
highest-value retained lifecycle lanes  
**Time estimate:** 12 hours

### Tasks
1. Implement the required property-expansion center from the Day 8 contract.
2. Keep the batch bounded to deterministic seeded properties and retained
   public lifecycle seams.
3. Update only truly forced support surfaces.
4. Record the landing in working notes and a Day 9 artifact.
5. Run the required validation gate for touched code.

### Deliverables
- seeded-property expansion landing
- any forced proof/support follow-through
- Day 9 implementation artifact

### Completion Criteria
- the bounded property batch lands inside the Day 8 fence
- deterministic reproducibility remains explicit
- the required validation gate passes

---

## Day 10: Failure-Path Numerical Proof Design

**Title:** Failure-Path Design  
**Theme:** Fix the bounded cancellation, error-path, and stress-fixture proof
contract for the most fragile public lifecycle guarantees  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 9 landing and remaining fragile lifecycle surfaces.
2. Decide the highest-value failure-path numerical proof center:
   - cancellation and callback short-circuit guarantees
   - error-path factor/solve/refactor preservation
   - stress-fixture invariants under retained public lifecycle flows
3. Separate:
   - required proof-owner surfaces
   - support-only CI/docs wording if truly needed
   - lower-value non-touch stress ideas
4. Fix the exact Day 11 implementation contract.
5. Write the Day 10 design artifact and working-notes summary.

### Deliverables
- failure-path proof design artifact
- required vs support-only seam map
- exact Day 11 implementation contract

### Completion Criteria
- Sprint 84 has one explicit third implementation contract
- the fragile lifecycle proof center is fixed before batch work lands
- Day 11 can implement without reopening the first two assurance lanes

---

## Day 11: Failure-Path Numerical Proof Batch

**Title:** Failure-Path Batch  
**Theme:** Land the bounded cancellation, error-path, and stress-fixture proof
package on the most fragile touched lifecycle lanes  
**Time estimate:** 12 hours

### Tasks
1. Implement the required failure-path proof center from the Day 10 contract.
2. Keep the landing bounded to the touched lifecycle and retained direct-family
   seams.
3. Update only truly forced support surfaces or proof-owner wording.
4. Record the landing in working notes and a Day 11 artifact.
5. Run the required validation gate for touched code.

### Deliverables
- failure-path numerical proof landing
- any forced support-surface follow-through
- Day 11 implementation artifact

### Completion Criteria
- the bounded failure-path batch lands inside the Day 10 fence
- retained lifecycle guarantees are strengthened with measured proof
- the required validation gate passes

---

## Day 12: Policy / CI / Support-Surface Alignment

**Title:** Alignment Pass  
**Theme:** Reconcile proof ownership, CI reading, and support-surface wording
with the widened Sprint 84 assurance model  
**Time estimate:** 12 hours

### Tasks
1. Re-read all landed Sprint 84 assurance surfaces and proof owners.
2. Recheck the authoritative wording owners:
   - maintainer guide
   - README only if truly needed
   - CI/policy reading surfaces
3. Confirm which proof and support surfaces already remain truthful without
   movement.
4. Fix the exact Day 13 validation queue in writing.
5. Record the final alignment map and validation queue artifact.

### Deliverables
- final proof-owner and CI/policy alignment artifact
- explicit Day 13 validation queue
- retained support-surface truth map

### Completion Criteria
- no stale assurance wording remains on touched surfaces
- the Day 13 queue is fixed with no ambiguity
- no extra support churn is carried into validation by accident

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full Sprint 84 validation queue and capture the measured
assurance close baseline  
**Time estimate:** 12 hours

### Tasks
1. Run the code-day validation gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed baseline:
   - `make quality-review-full`
3. Reconfirm reviewed CMake parity.
4. Rerun the focused reviewed proof-owner tests, representative examples, and
   relevant benchmark/reporting commands fixed on Day 12.
5. Capture retained outputs and any non-blocking runtime notes in a Day 13
   artifact and working notes.

### Deliverables
- full validation artifact
- validated close baseline
- explicit retained proof-owner outputs

### Completion Criteria
- the full Sprint 84 validation queue passes
- the reviewed anchors stay exact
- Day 14 can close from measured evidence instead of implementation state

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 84 from the validated baseline and fix the Sprint 85
handoff queue  
**Time estimate:** 12 hours

### Tasks
1. Re-read the full Sprint 84 branch package:
   - rerank
   - architecture contract
   - direct-family differential landing
   - property expansion
   - failure-path proof
   - alignment and validation
2. Recheck the Sprint 84 project-plan section against the landed work.
3. Reconcile the branch outcome against the next Epic 8 sprint ordering.
4. Write the Day 14 closeout and handoff artifact.
5. Record the final Sprint 84 close state in working notes.

### Deliverables
- Sprint 84 closeout/handoff artifact
- final working-notes closeout entry
- explicit Sprint 85-ready carry-forward queue

### Completion Criteria
- Sprint 84 closes from the validated Day 13 baseline
- the landed assurance package is summarized truthfully and boundedly
- the next Epic 8 contradiction center is fixed explicitly for the following
  sprint
