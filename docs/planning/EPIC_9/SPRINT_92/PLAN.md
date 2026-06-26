# Sprint 92 Plan: Portable Dense Backend & Kernel Maturity Phase 4

**Sprint Duration:** 14 days  
**Goal:** Raise the dense-kernel performance ceiling with one bounded portable
optional backend lane while preserving the builtin self-contained default
build. This sprint implements the Sprint 92 section of
`docs/planning/EPIC_9/PROJECT_PLAN.md`.

**Starting Point:** Sprint 91 closed with:
- one validated compressed-first product-convergence baseline
- first-class CSR/CSC constructor-style public entry paths
- a clearer public one-shot vs repeated-run direct-workflow story
- focused direct-workflow proof follow-through for constructor-built matrices
- a clean Sprint 92 handoff that puts portable dense backend maturity next

The strongest Sprint 92 pressure is no longer generic direct-family cleanup. It
is one bounded backend-maturity package centered on:
- dense-kernel hotspot reranking on the highest-value direct paths
- backend descriptor and runtime-selection design
- first portable accelerated backend integration on the strongest kernel seam
- solver adoption follow-through without destabilizing builtin fallback
- benchmark, correctness, build, and package follow-through only where the
  backend contract truly moves

**End State:** Sprint 92 leaves behind:
- one fresh dense-hotspot and backend-ceiling audit from the live Epic 9 tree
- one explicit builtin-vs-portable backend contract
- one bounded portable backend integration landing
- direct-family adoption on the strongest kernel consumers
- focused observability, proof, and package wording follow-through
- one validated Sprint 92 close baseline and Sprint 93 handoff queue

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, which stays within the 14-day cap and matches the
Sprint 92 project-plan `~168` hour scope.

---

## Day 1: Sprint 92 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 92 project-plan section and Sprint 91 handoff into
one bounded backend-maturity implementation package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 92 section of
   `docs/planning/EPIC_9/PROJECT_PLAN.md`, the Sprint 91 retrospective, and
   the Sprint 91 Day 14 closeout artifact.
2. Reconfirm the preserved starting assumptions:
   - Sprint 90 target-state and non-goal fence stay authoritative
   - Sprint 91 compressed-first work stays closed unless backend work truly
     forces follow-through
   - builtin self-contained default build remains the primary product truth
3. Define the Sprint 92 workstreams explicitly:
   - dense hotspot profiling
   - backend ABI and runtime-selection design
   - portable backend integration
   - solver adoption follow-through
   - benchmark/proof observability
   - build/package alignment
4. Record the strongest likely Sprint 92 touch surfaces:
   - `src/sparse_dense.c`
   - dense-kernel consumers in direct-family owners
   - benchmark and proof-owner tests
   - build/package and support surfaces only if forced
5. Open Sprint 92 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 92 scope inventory
- baseline/setup working notes
- explicit workstream map

### Completion Criteria
- Sprint 92 starts from the validated Sprint 91 implementation contract
- the backend-maturity problem is explicit before deeper validation and audit
  work begins
- the initial non-goal fence is visible before design or implementation widens

---

## Day 2: Validation & Maintained Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, benchmark, install/export, example,
and support-surface truth split before backend implementation begins  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest implementation-day and substantial-batch gates:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm the maintained proof and support owners Sprint 92 must treat as
   authoritative:
   - reviewed CMake parity
   - dense-kernel direct-family proof owners
   - benchmark/reporting owners
   - install/export proof owners
   - package/support/workflow surfaces
3. Recheck the current ownership split so Sprint 92 does not blur:
   - dense-kernel implementation owners
   - direct-family solver correctness owners
   - benchmark observability owners
   - build/package and workflow owners
4. Fix the authoritative rerun set most likely to matter throughout Sprint 92.
5. Record the validation and maintained-surface split in working notes and a
   Day 2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- maintained surface ownership map
- authoritative rerun list

### Completion Criteria
- the strongest local validation contract is explicit before audit or design
  findings are written
- proof ownership across reviewed tests, benchmarks, install/export checks, and
  support surfaces is fixed in writing
- later Sprint 92 days have no ambiguity about the required truth surfaces

---

## Day 3: Dense Hotspot Profiling Audit

**Title:** Hotspot Audit  
**Theme:** Reduce the live dense-backend problem to one ranked map of the
highest-value portable-kernel opportunities  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the live tree against the strongest Sprint 92 contradiction class:
   - builtin dense-kernel hotspots
   - strongest direct-family dense consumers
   - Darwin-only or bounded acceleration seams
   - runtime and allocation costs around dense work
2. Capture where dense kernels dominate the current ceiling on direct-family
   workloads.
3. Separate:
   - hotspots that should drive Sprint 92 implementation
   - hotspots that remain lower-value or later-epic candidates
   - hotspots already materially bounded by Epic 8 and Sprint 91
4. Identify the highest-value source, benchmark, test, and public-surface
   owners tied to those costs.
5. Write the ranked Day 3 audit artifact.

### Deliverables
- ranked dense-hotspot artifact
- fix-now vs later split
- strongest owner-surface map

### Completion Criteria
- the broad backend problem is reduced to one ranked live map
- the highest-value dense-kernel targets are explicit before the first
  implementation fence is frozen
- lower-value backend ambitions are separated from the main Sprint 92 lane

---

## Day 4: First Implementation Boundary

**Title:** Boundary Freeze  
**Theme:** Fix one bounded first landing so Sprint 92 starts with the
highest-value backend seam instead of generic dense rewrites  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 audit against the Sprint 92 project-plan contract.
2. Decide the required first landing center:
   - the portable backend seam inside the strongest dense owner
3. Decide which adjacent surfaces are directly forced support-only follow-
   through and which are explicitly later:
   - direct-family adopters
   - benchmark/proof owners
   - build/package wording
   - workflow or install/export surfaces
4. Freeze what Sprint 92 will not do in the first batch:
   - broad dense rewrite
   - fake platform symmetry
   - broad runtime/threading expansion
   - capability-surface widening
5. Write the Day 4 boundary artifact and update working notes.

### Deliverables
- first-landing boundary artifact
- required-owner vs support-only map
- explicit deferral list

### Completion Criteria
- Sprint 92 has one explicit first implementation fence
- the first landing is small enough to validate cleanly
- later solver-adoption, proof, and package work is clearly sequenced

---

## Day 5: Portable Backend ABI & Runtime Contract Design

**Title:** Backend Design  
**Theme:** Define the bounded builtin-vs-portable backend contract before code
movement  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 fence against the dense-hotspot audit.
2. Define the future role of:
   - builtin dense kernels
   - optional portable accelerated backend
   - runtime or compile-time backend selection
3. Decide the minimum backend observability and fallback semantics Sprint 92
   must uphold.
4. Decide which backend compatibility shims remain acceptable and which should
   stop being conceptual center stage.
5. Write the Day 5 architecture artifact.

### Deliverables
- backend ABI/runtime contract artifact
- fallback and observability policy
- compatibility-shim policy

### Completion Criteria
- the repo has one explicit builtin-vs-portable backend contract before code
  moves
- fallback truth stays stronger than acceleration claims
- Day 6 implementation can land without reopening product intent

---

## Day 6: Portable Backend Integration Batch

**Title:** Backend Batch  
**Theme:** Land the highest-value bounded portable backend seam without
breaking the builtin default path  
**Time estimate:** 12 hours

### Tasks
1. Implement the required first backend landing from the Day 5 design.
2. Keep the batch bounded to the strongest dense-kernel seam.
3. Add directly forced follow-through only where the landing requires it:
   - touched headers
   - touched dense helpers
   - touched tests or benchmarks
4. Run the required implementation-day validation gates.
5. Record the landed batch in working notes and a Day 6 artifact.

### Deliverables
- bounded portable backend implementation batch
- directly forced proof or benchmark follow-through
- validated Day 6 baseline

### Completion Criteria
- at least one dense-kernel path has a real bounded portable accelerated lane
- builtin fallback remains intact and truthfully primary
- the required validation gates pass cleanly

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Rerank the backend problem after the first landing and fix the next
highest-value seam  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 implementation against the Day 3 hotspot audit.
2. Determine what contradiction the first landing actually closed.
3. Identify the strongest remaining seam:
   - direct-family adoption
   - benchmark/proof observability
   - build/package wording
   - runtime-selection cleanup
4. Decide whether Sprint 92’s second implementation center stays code-owned or
   shifts to proof/benchmark owners.
5. Write the Day 7 rerank artifact and update working notes.

### Deliverables
- post-landing rerank artifact
- next-target decision
- updated owner map

### Completion Criteria
- the post-Day-6 contradiction map is explicit
- Sprint 92 has one exact next-center decision
- no generic dense cleanup batch slips in without reranking evidence

---

## Day 8: Solver Adoption Follow-Through Design

**Title:** Adoption Design  
**Theme:** Define the bounded follow-through that wires the widened backend
seam through the strongest direct-family consumers  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank against the Day 5 backend contract.
2. Choose the exact Day 9 implementation center:
   - the strongest direct-family adopter or adopters
3. Decide which support-only surfaces are directly forced if adoption moves:
   - tests
   - benchmarks
   - build/package wording
   - maintainer/public docs
4. Freeze what Day 9 should not become:
   - broad family-wide direct cleanup
   - runtime-threading work
   - capability widening
5. Write the Day 8 adoption-design artifact.

### Deliverables
- solver-adoption design artifact
- exact Day 9 center
- bounded support-only follow-through map

### Completion Criteria
- Sprint 92 has one exact second implementation contract
- adoption work is bounded to the strongest backend consumers
- later observability and package work remains sequenced behind real adoption

---

## Day 9: Solver Adoption Batch

**Title:** Adoption Batch  
**Theme:** Land the strongest direct-family adoption follow-through for the
new backend seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded Day 8 adoption contract.
2. Keep proof and benchmark follow-through local to what the adoption truly
   forces.
3. Preserve builtin fallback behavior and public non-claims.
4. Run the required validation gates for the touched batch.
5. Record the landed batch in working notes and a Day 9 artifact.

### Deliverables
- bounded direct-family backend adoption batch
- directly forced proof/benchmark follow-through
- validated Day 9 baseline

### Completion Criteria
- the widened backend seam is used on the strongest intended direct path
- fallback behavior remains clean and validated
- the required gates pass without reopening unrelated solver families

---

## Day 10: Observability & Proof Design

**Title:** Proof Design  
**Theme:** Define the bounded benchmark, correctness, and fallback proof needed
for the widened backend surface  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 9 landing against the Sprint 92 observability goals.
2. Decide the exact Day 11 center:
   - benchmark owner
   - proof owner
   - or one bounded combined follow-through
3. Freeze the reporting shape Sprint 92 should produce:
   - backend selection visibility
   - fallback behavior visibility
   - bounded performance evidence
4. Decide which build/package or support wording is directly forced if
   observability moves.
5. Write the Day 10 design artifact.

### Deliverables
- observability/proof design artifact
- exact Day 11 center
- frozen reporting shape

### Completion Criteria
- the remaining Sprint 92 evidence gap is explicit before the final
  implementation batch
- Day 11 has one bounded proof/benchmark center
- support-only wording movement is kept behind real evidence changes

---

## Day 11: Observability & Build Alignment Batch

**Title:** Observability Batch  
**Theme:** Land the bounded proof, benchmark, and build/package follow-through
required by the widened backend contract  
**Time estimate:** 12 hours

### Tasks
1. Implement the required Day 10 observability or proof batch.
2. Add directly forced support-only build/package wording only if the new
   backend contract truly changes it.
3. Preserve the builtin-default product story and bounded acceleration claims.
4. Run the required validation gates for the touched batch.
5. Record the landed batch in working notes and a Day 11 artifact.

### Deliverables
- bounded observability/proof/build-alignment batch
- directly forced package or support follow-through
- validated Day 11 baseline

### Completion Criteria
- the widened backend surface has explicit correctness, fallback, and
  observability evidence
- public/build/package wording stays truthful
- the required gates pass cleanly

---

## Day 12: Final Alignment & Validation Queue Freeze

**Title:** Alignment Pass  
**Theme:** Freeze the final Sprint 92 owner map and validation queue before the
full sweep  
**Time estimate:** 12 hours

### Tasks
1. Re-read all landed Sprint 92 batches and artifacts together.
2. Confirm whether any final docs-only support alignment is still needed.
3. Freeze the final owner split across:
   - dense/backend implementation owners
   - direct-family proof owners
   - benchmark/reporting owners
   - build/package/support owners
4. Freeze the exact Day 13 validation and follow-on queue.
5. Record the Day 12 alignment artifact and update working notes.

### Deliverables
- final owner-map artifact
- frozen Day 13 validation queue
- final support follow-through decision

### Completion Criteria
- no ambiguity remains about what Sprint 92 changed and who owns proof
- the Day 13 queue is explicit before execution
- any residual support-only follow-through is either landed or explicitly
  rejected

---

## Day 13: Full Validation & Evidence Sweep

**Title:** Validation Sweep  
**Theme:** Execute the full Sprint 92 validation, benchmark, and proof queue
from the live branch state  
**Time estimate:** 12 hours

### Tasks
1. Run the required full validation queue:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Run the frozen focused follow-ons:
   - reviewed proof owners
   - representative examples
   - backend-aware benchmark or reporting commands
3. Record the final measured baseline:
   - reviewed parity anchors
   - correctness/fallback outputs
   - backend-aware runtime evidence
4. Capture any non-blocking residual notes that remain after validation.
5. Write the Day 13 artifact and update working notes.

### Deliverables
- full validation artifact
- measured Sprint 92 close baseline
- residual runtime/backend notes

### Completion Criteria
- all required validation gates pass
- the widened backend lane has concrete evidence attached
- the Sprint 92 close baseline is explicit before closeout writing begins

---

## Day 14: Sprint Closeout & Sprint 93 Handoff

**Title:** Closeout  
**Theme:** Close Sprint 92 from the validated baseline and freeze the handoff
to runtime/threading convergence  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 92 goal, Day 13 results, and project-plan contract.
2. Summarize what Sprint 92 actually closed:
   - backend hotspot rerank
   - portable backend contract
   - backend integration and adoption
   - observability/proof/build follow-through
3. Record any remaining residual debt truthfully:
   - runtime concentration
   - broader backend/platform symmetry
   - later capability or maintainability work
4. Freeze the Sprint 93 handoff order and validation expectations.
5. Write the Day 14 closeout artifact and update working notes.

### Deliverables
- Sprint 92 closeout artifact
- frozen Sprint 93 handoff queue
- final Sprint 92 status in working notes

### Completion Criteria
- Sprint 92 closes from a validated baseline
- the residual queue is explicit and truthful
- Sprint 93 starts with a clear runtime/threading convergence handoff
