# Sprint 83 Plan: Capability Surface Modernization Phase 2

**Sprint Duration:** 14 days  
**Goal:** Move beyond the current real-only and compile-time-bounded capability
surface on the highest-value lanes first. This sprint implements the Sprint 83
section of `docs/planning/EPIC_8/PROJECT_PLAN.md`.

**Starting Point:** Sprint 82 closed from a validated dense-backend baseline
with one bounded optional backend seam landed on the Cholesky and LDL^T CSC
lanes. Sprint 83 therefore starts from a more coherent performance/runtime
story, but the strongest remaining first-tier Epic 8 contradiction is now the
capability surface itself:
- the public and family-local seams still read primarily as real-scalar,
  `double`-centered surfaces
- wider-index maturity is still uneven across the highest-value public and
  package-sensitive paths
- mixed-precision, complex-scalar, and algorithm-surface widening are not all
  equally valuable as the first next step
- Sprint 82’s backend work should not be reopened as a generic ABI rewrite
- the strongest local reviewed baseline remains `make quality-review-full`

The highest-value Sprint 83 work is therefore not generic “add more math
types.” It is one bounded capability-modernization package that:
- re-ranks complex-scalar support, mixed precision, wider-index maturity, and
  algorithm-surface expansion by value and risk
- fixes one explicit scalar/index architecture contract for the next Epic 8
  breadth step
- widens the highest-value public seams first instead of scattering capability
  churn across the tree
- closes the strongest remaining touched-path 64-bit and ABI safety seams
- expands one bounded solver-family or algorithm surface only where the widened
  scalar/index contract truly requires it
- proves and documents the widened reading without overstating repo-wide
  capability maturity

**End State:** Sprint 83 leaves behind:
- a refreshed capability-priority and contradiction map
- one explicit scalar/index architecture contract
- one landed bounded scalar-surface expansion on the highest-value public seams
- one focused index/ABI follow-through package on touched paths
- one bounded algorithm-surface widening package
- one focused regression/docs/package alignment package
- one validated closeout baseline and Sprint 84-ready handoff

**Time budget:** Each day is capped at 12 hours as requested. Because that cap
allows at most `168` hours across 14 days, this day-by-day plan totals `168`
hours rather than the higher project-plan estimate of `~180` hours, while
preserving the Sprint 83 scope and ordering.

---

## Day 1: Sprint 83 Scope Audit & Capability Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 83 project-plan section and Sprint 82 closeout into
one bounded capability-surface execution package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 83 section of
   `docs/planning/EPIC_8/PROJECT_PLAN.md`, the Sprint 82 closeout artifact, and
   the Sprint 82 retrospective.
2. Reconfirm the preserved Sprint 83 starting assumptions:
   - Sprint 80 already fixed the broader capability-priority framing
   - Sprint 82 already fixed the current dense/backend ABI fence
   - Sprint 83 should not widen into broad package/platform maturity claims
   - Sprint 83 should not promise repo-wide complex or mixed-precision support
     before one bounded seam is truly landed
3. Define the Sprint 83 workstreams explicitly:
   - capability re-rank
   - scalar/index architecture design
   - scalar-surface expansion
   - index/ABI follow-through
   - algorithm-surface widening
   - regression/docs/package alignment
   - validation and closeout
4. Record the strongest likely Sprint 83 touch surfaces:
   - `include/sparse_types.h`
   - `include/sparse_matrix.h`
   - highest-value public solver headers
   - `src/sparse_matrix.c`
   - strongest algorithm-family implementation owners
   - proof and package-sensitive support surfaces
5. Open Sprint 83 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 83 scope inventory
- capability workstream map
- starting working-notes baseline

### Completion Criteria
- Sprint 83 starts from the validated Sprint 82 end state
- the first capability contradiction is explicit before deeper audit begins
- the non-goal fence is visible before design or implementation work

---

## Day 2: Validation & Proof-Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, proof-owner, and package-sensitive
validation split before capability changes begin  
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
   Sprint 83:
   - public matrix and types tests
   - QR / SVD / direct-family tests most likely to reflect capability widening
   - representative examples
   - canonical benchmark/reporting command surfaces
   - install/export proof scripts
4. Fix the authoritative rerun list most likely to matter throughout Sprint
   83.
5. Record the validation/proof split in working notes and a Day 2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- preserved proof-owner map
- authoritative Sprint 83 rerun list

### Completion Criteria
- the strongest local validation contract is explicit before implementation
  work lands
- proof ownership across reviewed tests, benchmarks, and install/export
  surfaces is fixed in writing
- later code days have no ambiguity about the required validation gate

---

## Day 3: Capability Re-rank Audit

**Title:** Capability Audit  
**Theme:** Re-rank complex scalar support, mixed precision, wider-index
maturity, and algorithm-surface expansion by value and risk  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the highest-signal public and family-local capability surfaces:
   - `sparse_types` width and scalar assumptions
   - matrix ownership and construction seams
   - QR / SVD / factorization public API contracts
   - package-visible compile-time or ABI-sensitive seams
2. Identify where the current capability ceiling is strongest:
   - real-only public assumptions
   - hard `double` storage and callback expectations
   - width-sensitive array, count, and export/import paths
   - algorithm surfaces that implicitly assume the narrower scalar/index model
3. Separate:
   - strongest first-batch implementation center
   - second-tier follow-through seams
   - support-only proof and package/runtime surfaces
   - deliberate non-goals
4. Reconcile the audit against the Sprint 80 capability map and Sprint 82
   backend closeout.
5. Write the ranked capability artifact.

### Deliverables
- ranked capability-surface artifact
- first-tier vs deferred seam map
- Sprint 80 / 82 carry-forward reconciliation notes

### Completion Criteria
- Sprint 83’s broad capability problem is reduced to one ranked live map
- the strongest implementation center is explicit before boundary design
- lower-value spillover work is clearly separated from the first lane

---

## Day 4: First Capability Boundary Freeze

**Title:** Boundary Freeze  
**Theme:** Fix the first bounded scalar/index implementation fence and the
allowed capability reading  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 ranking against the Sprint 83 project-plan scope.
2. Decide the required first implementation center:
   - scalar-surface widening on highest-value public seams
   - types/index contract widening on shared public seams
   - one bounded compatibility-preserving architecture lane
3. Decide which support surfaces move only if forced:
   - proof-owner tests
   - benchmarks
   - public headers outside the batch center
   - package/runtime docs
4. Fix the preserved non-goal fence for the first landing:
   - no repo-wide complex-number promise
   - no broad mixed-precision framework
   - no ABI churn detached from touched public seams
   - no algorithm-family widening before the shared contract is explicit
5. Record the first implementation fence in working notes and a Day 4 artifact.

### Deliverables
- first capability-boundary artifact
- required vs support-only touch set
- preserved first-batch non-goal fence

### Completion Criteria
- Sprint 83 has one explicit first landing boundary
- support-only surfaces are clearly separated from the batch center
- Day 5 can design one scalar/index contract instead of a broad rewrite

---

## Day 5: Scalar / Index Architecture Design

**Title:** Architecture Design  
**Theme:** Define the bounded scalar/index contract Sprint 83 will actually
land  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 boundary and the strongest capability contradictions.
2. Define the ownership split for:
   - shared scalar and width vocabulary
   - public matrix and types exposure
   - compatibility-preserving internal representation seams
   - family-level adoption boundaries
3. Decide how the first landing will preserve current callers while widening
   the next capability step.
4. Fix the touch fence for tests, benchmarks, package/runtime docs, and
   headers.
5. Write the Day 5 architecture artifact and working-notes design summary.

### Deliverables
- scalar/index architecture contract
- ownership split for touched seams
- preserved compatibility and non-goal fence

### Completion Criteria
- Sprint 83 has one explicit implementation contract
- ownership between shared types, public seams, and family-local follow-through
  is clear
- Day 6 can implement one bounded landing without reopening design questions

---

## Day 6: Scalar-Surface Expansion Batch

**Title:** Surface Expansion  
**Theme:** Land the first bounded widening of the scalar contract on the
highest-value public seams  
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-value scalar/index seam from the Day 5 contract.
2. Keep the landing bounded to the required first implementation center.
3. Preserve existing callers and untouched families where the widened contract
   does not yet apply.
4. Update any truly forced local proof-owner tests or package/runtime
   plumbing.
5. Record the landing in working notes and a Day 6 artifact.
6. Run the required validation gate for touched code.

### Deliverables
- first scalar-surface expansion landing
- any forced focused regression follow-through
- Day 6 implementation artifact

### Completion Criteria
- the first bounded capability batch lands inside the Day 5 fence
- compatibility behavior remains preserved on untouched seams
- the required validation gate passes

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the strongest remaining capability seam after the first
public-surface widening  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 landing against the Day 5 architecture contract.
2. Decide whether the strongest remaining seam is now:
   - index / ABI follow-through
   - algorithm-surface widening
   - proof / docs / package alignment
3. Separate:
   - required next landing center
   - support-only proof and benchmark surfaces
   - support-only package/runtime wording surfaces
   - preserved non-goals
4. Record the rerank in working notes and a Day 7 artifact.
5. Fix the exact Day 8 design center.

### Deliverables
- post-landing rerank artifact
- next-step implementation center
- updated support-only seam map

### Completion Criteria
- Sprint 83’s next contradiction center is explicit after the first landing
- Day 8 can design one bounded follow-through batch
- support drift is separated from real capability work

---

## Day 8: Index / ABI Follow-Through Design

**Title:** ABI Design  
**Theme:** Fix the exact touched-path width, ABI, and package-safety contract
required after the scalar-surface landing  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank and the landed Day 6 scalar/index contract.
2. Decide the exact strongest follow-through seams to move next:
   - touched public headers
   - width-sensitive import/export or count paths
   - package-visible ABI or compile-time interpretation seams
3. Define how touched-path 64-bit safety, compatibility, and packaging should
   read from the updated surface.
4. Separate required proof/package follow-through from support-only wording.
5. Write the Day 8 design artifact and working-notes summary.

### Deliverables
- index / ABI follow-through design
- required vs support-only touch set
- preserved compatibility and non-goal fence

### Completion Criteria
- Sprint 83 has one exact second implementation contract
- the strongest touched-path width/ABI seams are explicit
- Day 9 can land one bounded follow-through batch without reopening design

---

## Day 9: Index / ABI Follow-Through Batch

**Title:** ABI Batch  
**Theme:** Close the strongest remaining touched-path 64-bit and ABI safety
seams  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded width / ABI follow-through contract from Day 8.
2. Keep the landing inside the required touch set only.
3. Preserve compatibility behavior, unchanged callers, and untouched package
   surfaces outside the batch.
4. Update any truly forced proof-owner tests and package-sensitive wording.
5. Record the landing in working notes and a Day 9 artifact.
6. Run the required validation gate for touched code.

### Deliverables
- index / ABI follow-through landing
- any forced focused proof/package updates
- Day 9 implementation artifact

### Completion Criteria
- the width / ABI follow-through lands inside the Day 8 fence
- touched-path safety and compatibility remain truthful and measurable
- the required validation gate passes

---

## Day 10: Algorithm-Surface Widening Design

**Title:** Algorithm Design  
**Theme:** Fix the exact solver-family or algorithm-surface widening justified
by the widened scalar/index contract  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 and Day 9 landings plus the retained proof-owner
   surfaces.
2. Decide the exact algorithm-family seam that now needs movement:
   - QR
   - SVD
   - one direct-family surface only if truly forced by the new contract
3. Separate:
   - executable algorithm-surface widening
   - proof-owner follow-through
   - support-only docs/package wording
4. Fix the exact Day 11 touch set and non-touch set.
5. Write the Day 10 design artifact.

### Deliverables
- algorithm-surface widening design
- exact Day 11 touch set
- support-only and non-touch map

### Completion Criteria
- no extra family drift is implied beyond the bounded touched seam
- Day 11 can land one focused capability follow-through batch
- algorithm and support roles stay clearly separated

---

## Day 11: Algorithm-Surface Widening Batch

**Title:** Algorithm Batch  
**Theme:** Land the focused solver-family or algorithm-surface widening
actually required by the new scalar/index contract  
**Time estimate:** 12 hours

### Tasks
1. Implement the focused algorithm-surface changes fixed on Day 10.
2. Keep the landing bounded to one solver-family or algorithm lane.
3. Preserve current behavior on untouched families and unchanged public seams.
4. Record the landing in working notes and a Day 11 artifact.
5. Run the required validation gate for touched code.

### Deliverables
- focused algorithm-surface widening landing
- any forced proof-owner follow-through
- Day 11 implementation artifact

### Completion Criteria
- the widened capability seam has the required algorithm follow-through only
- no fake repo-wide capability claim is introduced
- the required validation gate passes

---

## Day 12: Regression / Docs / Package Alignment

**Title:** Alignment  
**Theme:** Land the focused regression, support-surface, and package alignment
actually required by the widened capability reading  
**Time estimate:** 12 hours

### Tasks
1. Re-read the landed implementation, proof, header, and package-sensitive
   surfaces.
2. Confirm whether any further support-only edits are truly required.
3. Fix the final Sprint 83 proof-owner map across:
   - reviewed CMake tests
   - representative examples
   - canonical benchmark/reporting command surfaces
   - install/export proof if package mechanics moved
   - public docs or headers that still misread the widened surface
4. Fix the exact Day 13 validation queue in writing.
5. Record the alignment pass in working notes and a Day 12 artifact.

### Deliverables
- final proof-owner and alignment map
- authoritative Day 13 validation queue
- explicit no-op note for any untouched support surfaces

### Completion Criteria
- no validation ambiguity remains before the full sweep
- support surfaces match the widened capability claim on touched paths
- Day 13 can execute from one stable measured queue

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full Sprint 83 validation queue and capture the retained
closeout baseline  
**Time estimate:** 12 hours

### Tasks
1. Run the standard code-day validation gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed validation baseline:
   - `make quality-review-full`
3. Re-run the authoritative focused proof-owner binaries and representative
   examples fixed on Day 12.
4. Re-run any touched benchmark/reporting follow-ons.
5. Re-run install/export proof if Sprint 83 moved package/runtime mechanics.
6. Record the final measured baseline in working notes and a Day 13 artifact.

### Deliverables
- full validation-sweep artifact
- retained reviewed anchors
- retained focused proof / benchmark / package outputs

### Completion Criteria
- the full Sprint 83 rerun set passes
- retained anchors and representative outputs are fixed in writing
- Day 14 can close from measured evidence rather than partial implementation
  state

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 83 from the validated Day 13 baseline and hand off the
next Epic 8 contradiction center  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 83 project-plan section, landed artifacts, and Day 13
   validated baseline.
2. Summarize exactly what Sprint 83 changed in the capability contract.
3. Fix the ranked carry-forward queue for Sprint 84 and later Epic 8 work.
4. Recheck whether `docs/planning/EPIC_8/PROJECT_PLAN.md` needs any Sprint 83
   correction.
5. Write the Day 14 closeout/handoff artifact and finalize working notes.

### Deliverables
- closeout/handoff artifact
- final working-notes close state
- Sprint 84-ready handoff queue

### Completion Criteria
- Sprint 83 closes from a validated baseline rather than from implied context
- the next Epic 8 contradiction center is explicit
- the branch is ready for retrospective generation and handoff
