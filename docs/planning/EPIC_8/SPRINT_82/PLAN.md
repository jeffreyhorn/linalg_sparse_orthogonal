# Sprint 82 Plan: Dense Backend & Performance Ceiling Phase 3

**Sprint Duration:** 14 days  
**Goal:** Raise the dense-kernel performance ceiling with one bounded optional
backend layer while preserving the builtin self-contained default build. This
sprint implements the Sprint 82 section of
`docs/planning/EPIC_8/PROJECT_PLAN.md`.

**Starting Point:** Sprint 81 closed from a validated product/storage baseline
with the highest-value linked-list-first construction/import costs reduced and
the repeated-run Cholesky and LDL^T path kept on the analysis-backed CSC-aware
route for all problem sizes. The strongest remaining first-tier Epic 8
contradiction is now the builtin scalar dense/backend performance ceiling.

The highest-value Sprint 82 work is therefore not generic performance tuning.
It is one bounded backend modernization package that:
- re-measures the densest helper hotspots on the maintained workloads
- fixes one explicit dense-kernel descriptor and runtime-selection contract
- lands one first optional accelerated dense-kernel integration slice
- wires that seam through the strongest direct-family consumers without
  destabilizing the builtin fallback path
- proves the widened backend surface with focused benchmark and differential
  evidence

**End State:** Sprint 82 leaves behind:
- one refreshed dense-hotspot and backend-priority map
- one explicit dense-kernel ABI and runtime-selection contract
- one bounded optional accelerated backend landing with builtin fallback intact
- one focused solver-adoption follow-through package
- one focused benchmark/differential/runtime-alignment package
- one validated closeout baseline and Sprint 83-ready handoff

**Time budget:** Each day is capped at 12 hours as requested. Because that cap
allows at most `168` hours across 14 days, this day-by-day plan totals `168`
hours rather than the higher project-plan estimate of `~184` hours, while
preserving the Sprint 82 scope and ordering.

---

## Day 1: Sprint 82 Scope Audit & Dense Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 82 project-plan section and Sprint 81 closeout into
one bounded dense-backend execution package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 82 section of
   `docs/planning/EPIC_8/PROJECT_PLAN.md`, the Sprint 81 closeout artifact, and
   the Sprint 81 retrospective.
2. Reconfirm the preserved Sprint 82 starting assumptions:
   - Sprint 80 already fixed the oracle and benchmark-governance fence
   - Sprint 81 already fixed the first product/storage contradiction
   - Sprint 82 should not widen into capability, package/platform, or broad
     state-of-the-art comparison work
3. Define the Sprint 82 workstreams explicitly:
   - dense hotspot profiling
   - backend ABI design
   - first external dense-kernel integration
   - solver adoption follow-through
   - benchmark/differential proof
   - packaging/runtime alignment
   - validation and closeout
4. Record the strongest likely Sprint 82 touch surfaces:
   - dense helper owners
   - highest-value direct-family consumers
   - benchmark/proof owners
   - package/runtime wording surfaces
5. Open Sprint 82 working notes and record the intended landing order and
   validation expectations.

### Deliverables
- Sprint 82 scope inventory
- dense/backend workstream map
- starting working-notes baseline

### Completion Criteria
- Sprint 82 starts from the validated Sprint 81 end state
- the first dense/backend contradiction is explicit before profiling begins
- the non-goal fence is visible before any design or implementation work

---

## Day 2: Validation & Proof-Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, benchmark, and install proof split
before backend changes begin  
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
   Sprint 82:
   - direct-family tests
   - representative examples
   - canonical benchmark/reporting command surfaces
   - install/export proof scripts
4. Fix the authoritative rerun list most likely to matter throughout Sprint 82.
5. Record the validation/proof split in working notes and a Day 2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- preserved proof-owner map
- authoritative Sprint 82 rerun list

### Completion Criteria
- the strongest local validation contract is explicit before implementation
  work lands
- proof ownership across reviewed tests, benchmarks, and install/export
  surfaces is fixed in writing
- later code days have no ambiguity about the required validation gate

---

## Day 3: Dense Hotspot Profiling Audit

**Title:** Hotspot Audit  
**Theme:** Re-rank the strongest dense-helper and backend-ceiling costs against
the maintained solver and benchmark workloads  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the highest-signal dense/helper implementation surfaces:
   - dense kernel helpers
   - Cholesky dense-panel consumers
   - LDL^T dense-helper consumers
   - QR and SVD dense-path surfaces
2. Identify where builtin dense costs are highest:
   - panel factor/solve paths
   - blocked update kernels
   - repeated tiny-kernel dispatch overhead
   - fallback-vs-accelerated adoption boundaries
3. Separate:
   - strongest first-batch implementation center
   - second-tier adoption seams
   - support-only proof and package/runtime surfaces
   - deliberate non-goals
4. Reconcile the audit against the Sprint 80 contradiction map and the Sprint
   81 storage closeout.
5. Write the ranked hotspot artifact.

### Deliverables
- ranked dense-hotspot artifact
- first-tier vs deferred seam map
- Sprint 80/81 carry-forward reconciliation notes

### Completion Criteria
- Sprint 82’s broad backend problem is reduced to one ranked live map
- the strongest implementation center is explicit before boundary design
- lower-value spillover work is clearly separated from the first lane

---

## Day 4: Backend Candidate Audit & First Boundary Freeze

**Title:** Boundary Freeze  
**Theme:** Fix the first bounded backend implementation fence and the allowed
accelerated-backend reading  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 ranking against the Sprint 82 project-plan scope.
2. Decide the required first implementation center:
   - dense-kernel descriptor/runtime-selection seam
   - first accelerated integration seam
   - strongest direct-family consumer seam
3. Decide which support surfaces move only if forced:
   - proof-owner tests
   - benchmarks
   - headers
   - package/runtime docs
4. Fix the preserved non-goal fence for the first landing:
   - no mandatory heavyweight dependency for default builds
   - no fake platform parity
   - no shared-library maturity claim
   - no benchmark-gate inflation
5. Record the first implementation fence in working notes and a Day 4 artifact.

### Deliverables
- first backend-boundary artifact
- required vs support-only touch set
- preserved first-batch non-goal fence

### Completion Criteria
- Sprint 82 has one explicit first landing boundary
- support-only surfaces are clearly separated from the batch center
- Day 5 can design one backend contract instead of a broad rewrite

---

## Day 5: Dense-Kernel ABI & Runtime-Selection Design

**Title:** ABI Design  
**Theme:** Define the bounded backend descriptor and runtime contract Sprint 82
will actually land  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 boundary and the strongest dense/backend contradictions.
2. Define the ownership split for:
   - builtin dense-kernel descriptor
   - optional accelerated backend hook points
   - runtime selection and observability
   - solver-family adoption boundaries
3. Decide how the first landing will preserve the builtin self-contained path
   while widening optional acceleration.
4. Fix the touch fence for tests, benchmarks, package/runtime docs, and
   headers.
5. Write the Day 5 backend contract artifact and working-notes design summary.

### Deliverables
- dense-kernel ABI and runtime-selection contract
- ownership split for touched seams
- preserved builtin-default and non-goal fence

### Completion Criteria
- Sprint 82 has one explicit implementation contract
- ownership between builtin fallback and optional acceleration is clear
- Day 6 can implement one bounded landing without reopening design questions

---

## Day 6: External Dense-Kernel Integration Batch 1

**Title:** Integration Batch  
**Theme:** Land the first bounded optional accelerated dense-kernel seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-value backend descriptor/runtime seam from the Day 5
   contract.
2. Keep the landing bounded to the required first implementation center.
3. Preserve the builtin default path and the no-optional-backend build path.
4. Update any truly forced local proof-owner tests or package/runtime plumbing.
5. Record the landing in working notes and a Day 6 artifact.
6. Run the required validation gate for touched code.

### Deliverables
- first optional accelerated backend landing
- any forced focused regression follow-through
- Day 6 implementation artifact

### Completion Criteria
- the first bounded backend batch lands inside the Day 5 fence
- builtin fallback behavior remains preserved on the touched paths
- the required validation gate passes

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the strongest remaining backend seam after the first
integration landing  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 landing against the Day 5 backend contract.
2. Decide whether the strongest remaining seam is now:
   - solver adoption follow-through
   - benchmark/differential proof
   - package/runtime alignment
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
- Sprint 82’s next contradiction center is explicit after the first landing
- Day 8 can design one bounded follow-through batch
- support drift is separated from real backend work

---

## Day 8: Solver Adoption Follow-Through Design

**Title:** Adoption Design  
**Theme:** Fix the exact direct-family adoption contract for the widened dense
backend seam  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank and the landed Day 6 backend contract.
2. Decide the exact strongest solver-family consumers to move next:
   - Cholesky
   - LDL^T
   - QR and SVD only if truly justified
3. Define how runtime selection, fallback behavior, and observability should
   read from the solver-side surface.
4. Separate required proof/benchmark follow-through from support-only wording.
5. Write the Day 8 design artifact and working-notes summary.

### Deliverables
- solver adoption follow-through design
- required vs support-only touch set
- preserved fallback and non-goal fence

### Completion Criteria
- Sprint 82 has one exact second implementation contract
- the strongest direct-family adoption surfaces are explicit
- Day 9 can land one bounded solver-side batch without reopening design

---

## Day 9: Solver Adoption Follow-Through Batch

**Title:** Adoption Batch  
**Theme:** Wire the widened dense backend seam through the strongest direct
solver consumers  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded solver-side adoption contract from Day 8.
2. Keep the landing inside the required consumer set only.
3. Preserve fallback behavior, cancellation/runtime semantics, and unchanged
   callers on untouched families.
4. Update any truly forced proof-owner tests and benchmark-side measurability.
5. Record the landing in working notes and a Day 9 artifact.
6. Run the required validation gate for touched code.

### Deliverables
- solver adoption follow-through landing
- any forced focused proof/benchmark updates
- Day 9 implementation artifact

### Completion Criteria
- the solver-side follow-through lands inside the Day 8 fence
- fallback/default behavior remains truthful and measurable
- the required validation gate passes

---

## Day 10: Benchmark / Differential Proof Design

**Title:** Proof Design  
**Theme:** Fix the exact benchmark, differential, and runtime-alignment surface
needed after the code landings  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 and Day 9 landings plus their retained proof-owner
   surfaces.
2. Decide the exact proof owners that now need movement:
   - differential correctness owners
   - benchmark measurability owners
   - package/runtime wording owners only if forced
3. Separate:
   - executable regression proof
   - benchmark-side reporting/profiling proof
   - support-only runtime/package wording
4. Fix the exact Day 11 touch set and non-touch set.
5. Write the Day 10 design artifact.

### Deliverables
- proof and runtime-alignment design
- exact Day 11 touch set
- support-only and non-touch map

### Completion Criteria
- no extra proof or package drift is implied beyond the bounded touched seam
- Day 11 can land one focused follow-through batch
- benchmark and differential roles stay clearly separated

---

## Day 11: Benchmark / Differential / Runtime Alignment Batch

**Title:** Proof Batch  
**Theme:** Land the focused benchmark, differential, and runtime/package
follow-through actually required by the backend widening  
**Time estimate:** 12 hours

### Tasks
1. Implement the focused proof and alignment changes fixed on Day 10.
2. Keep benchmark reporting threshold-free and separate from pass/fail logic.
3. Preserve the bounded external-oracle contract and the builtin-default build
   reading.
4. Record the landing in working notes and a Day 11 artifact.
5. Run the required validation gate for touched code.

### Deliverables
- focused proof and runtime/package alignment landing
- any forced benchmark/reporting follow-through
- Day 11 implementation artifact

### Completion Criteria
- the widened backend seam has the required proof and alignment only
- no fake platform, packaging, or benchmark-threshold claim is introduced
- the required validation gate passes

---

## Day 12: Final Proof Alignment & Validation Queue

**Title:** Proof Alignment  
**Theme:** Fix the final Sprint 82 proof-owner map and exact Day 13 rerun set  
**Time estimate:** 12 hours

### Tasks
1. Re-read the landed implementation, proof, benchmark, and package/runtime
   surfaces.
2. Confirm whether any further support-only edits are truly required.
3. Fix the final Sprint 82 proof-owner map across:
   - reviewed CMake tests
   - representative examples
   - canonical benchmark/reporting command surfaces
   - install/export proof if package mechanics moved
4. Fix the exact Day 13 validation queue in writing.
5. Record the alignment pass in working notes and a Day 12 artifact.

### Deliverables
- final proof-owner map
- authoritative Day 13 validation queue
- explicit no-op note for any untouched support surfaces

### Completion Criteria
- no validation ambiguity remains before the full sweep
- proof ownership is explicit across tests, benchmarks, and package/runtime
  surfaces
- Day 13 can execute from one stable measured queue

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full Sprint 82 validation queue and capture the retained
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
5. Re-run install/export proof if Sprint 82 moved package/runtime mechanics.
6. Record the final measured baseline in working notes and a Day 13 artifact.

### Deliverables
- full validation-sweep artifact
- retained reviewed anchors
- retained focused proof/benchmark/package outputs

### Completion Criteria
- the full Sprint 82 rerun set passes
- retained anchors and representative outputs are fixed in writing
- Day 14 can close from measured evidence rather than partial implementation
  state

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 82 from the validated Day 13 baseline and hand off the
next Epic 8 contradiction center  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 82 project-plan section, landed artifacts, and Day 13
   validated baseline.
2. Summarize exactly what Sprint 82 changed in the dense/backend contract.
3. Fix the ranked carry-forward queue for Sprint 83 and later Epic 8 work.
4. Recheck whether `docs/planning/EPIC_8/PROJECT_PLAN.md` needs any Sprint 82
   correction.
5. Write the Day 14 closeout/handoff artifact and finalize working notes.

### Deliverables
- closeout/handoff artifact
- final working-notes close state
- Sprint 83-ready handoff queue

### Completion Criteria
- Sprint 82 closes from a validated baseline rather than from implied context
- the next Epic 8 contradiction center is explicit
- the branch is ready for retrospective generation and handoff
