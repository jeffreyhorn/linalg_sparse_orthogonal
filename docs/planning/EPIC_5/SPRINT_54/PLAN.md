# Sprint 54 Plan: Public Repeated-Run Solver Lifecycle Completion

**Sprint Duration:** 14 days  
**Goal:** Decide and land the steady-state public repeated-run story across the
remaining iterative and eigensolver solver families, so Epic 5 leaves behind a
clear and honestly bounded solver-lifecycle surface.
This sprint implements the Sprint 54 section of
`docs/planning/EPIC_5/PROJECT_PLAN.md`.

**Starting Point:** Sprint 53 closed with the direct-solver lifecycle and CSC
follow-through package in a stronger validated state, while Epic 4 already left
public repeated-run handles for the main CG, GMRES, and symmetric eigensolver
paths. The remaining high-value gap is not “add handles everywhere,” but
decide and prove the final steady-state support boundary for the remaining
iterative and eigensolver families:
- MINRES / BiCGSTAB lifecycle symmetry
- selected block or advanced iterative workflow exposure vs explicit exclusion
- any remaining eigensolver lifecycle asymmetry or documentation drift
- benchmark, regression, and example alignment for the final public support set

**End State:** Sprint 54 leaves behind a bounded and validated public
repeated-run solver contract: the supported handle-based solver families are
explicit, intentionally excluded families are documented honestly, and the
benchmark/test/docs surfaces all match the final public lifecycle story.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
152 hours, matching the Sprint 54 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 54 Scope Audit & Solver-Lifecycle Baseline

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 54 project-plan items plus the Epic 4 and Sprint 53
close states into a bounded repeated-run solver-lifecycle map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 54 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`,
   the Sprint 49 public-handle closeout, the Sprint 53 closeout, and the most
   relevant deferred repeated-run notes from Epic 4 and early Epic 5.
2. Reconfirm the preserved Sprint 54 constraints:
   - keep the Sprint 49 public-handle compatibility fence intact
   - keep the Sprint 50 direct-lifecycle non-goal fence intact
   - decide and clarify the support boundary before broad implementation
   - preserve the strongest local reviewed baseline and truthfulness anchors
3. Define the Sprint 54 workstreams explicitly:
   - public solver lifecycle audit
   - solver-surface decision batch
   - iterative handle expansion where justified
   - eigensolver lifecycle tightening
   - benchmark/test/example/docs alignment
   - validation and closeout
4. Record the highest-risk seams for the sprint:
   - silently inconsistent lifecycle support across solver families
   - public docs implying broader support than the code really offers
   - benchmark or test surfaces proving internal reuse seams instead of the
     public handle path
   - accidental scope growth into broad solver-API redesign
5. Open Sprint 54 working notes and record the initial landing order and
   touched-surface expectations.

### Deliverables
- Sprint 54 scope inventory
- Repeated-run solver-lifecycle baseline notes
- Working-notes starting assumptions

### Completion Criteria
- Sprint 54 starts from the existing public handle and validated direct-solver
  state rather than reopening old architecture work
- Preserved compatibility and scope fences are explicit before new solver
  audits or code patches land
- The repeated-run solver workstreams are named before implementation begins

---

## Day 2: Validation Baseline & Touched-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed local baseline and the exact iterative /
eigensolver rerun set Sprint 54 code days must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the maintained reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity
   - current truthfulness-anchor counts
2. Reconfirm the mandatory gate for later `*.c` / `*.h` solver-lifecycle
   batches:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial public repeated-run API or
   solver-family integration batches:
   - `make quality-review-full`
4. Refresh the targeted Sprint 54 follow-on binaries most likely to be needed:
   - `./build/test_iterative`
   - `./build/test_eigs`
   - `./build/test_eigs_lobpcg`
   - `./build/example_iterative`
   - `./build/example_eigs`
   - repeated-run iterative and eigensolver benchmark binaries
5. Record the authoritative validation boundary for docs-only decision days
   versus code-touch lifecycle landing days.

### Deliverables
- Refreshed validation/truthfulness notes
- Sprint 54 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 54 uses the same baseline wording and parity anchors as the live repo
- The authoritative iterative/eigensolver rerun set is explicit before
  implementation work begins
- No validation ambiguity remains around lifecycle decision or expansion days

---

## Day 3: Public Solver Lifecycle Audit

**Title:** Lifecycle Audit  
**Theme:** Audit the remaining public repeated-run asymmetries before deciding
what actually belongs on the supported steady-state surface  
**Time estimate:** 10 hours

### Tasks
1. Audit the live repeated-run public surfaces across:
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
   - corresponding `src/` implementations
   - repeated-run tests, examples, and benchmark drivers
2. Separate already-supported public handle paths from solver families that
   still rely on one-shot-only exposure or internal reuse seams.
3. Identify where public documentation or examples currently imply more solver
   lifecycle symmetry than the code really provides.
4. Rank the highest-value remaining solver-family gaps and explicitly reject
   larger redesign surfaces.
5. Write the audit artifact and the ranked target list.

### Deliverables
- Public solver lifecycle audit
- Support-gap inventory
- Ranked implementation and exclusion targets

### Completion Criteria
- The repeated-run solver problem is reduced to named seam classes instead of
  a generic “finish the remaining families” instruction
- Existing supported handle paths and unsupported families are clearly
  separated
- Sprint 54 can start decision work from a concrete audit

---

## Day 4: Solver-Surface Decision Batch

**Title:** Decision Batch  
**Theme:** Decide the steady-state public support boundary before broadening
the handle surface  
**Time estimate:** 12 hours

### Tasks
1. Evaluate the remaining iterative and advanced solver families against the
   Day 3 criteria:
   - user value
   - implementation cost
   - regression proof cost
   - lifecycle clarity
2. Decide, explicitly and in writing, whether the following remain in-scope or
   are intentionally excluded for Sprint 54:
   - MINRES
   - BiCGSTAB
   - selected block iterative workflows
   - any remaining eigensolver reuse subpaths
3. Define the justification for each chosen exclusion so the repo does not
   imply “not implemented yet” when the real answer is “intentionally bounded.”
4. Fix the implementation order from the chosen support boundary.
5. Record the final Sprint 54 decision artifact and working-notes handoff.

### Deliverables
- Explicit repeated-run support boundary
- Chosen inclusion/exclusion decisions
- Updated landing order

### Completion Criteria
- The public steady-state support set is fixed before code expansion begins
- Chosen exclusions are documented as conscious boundaries, not accidental gaps
- Sprint 54 implementation scope is materially smaller and clearer than the raw
  project-plan placeholder

---

## Day 5: Iterative Handle Expansion Batch I

**Title:** Iterative Batch I  
**Theme:** Land the first bounded public repeated-run extension for the
remaining iterative families, if justified by the Day 4 support boundary  
**Time estimate:** 12 hours

### Tasks
1. Land the first bounded iterative public lifecycle patch on the
   highest-value in-scope family or seam.
2. Preserve the Sprint 49 handle contract:
   - zero/init
   - prepare
   - run
   - reuse preserves capacity/setup, not stale numerical state
   - free
3. Keep existing one-shot solver entries first-class and intact.
4. Add focused internal comments only where the new path would otherwise be
   hard to reason about.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- First iterative handle-expansion patch
- Preserved one-shot compatibility behavior
- Validation output for batch I

### Completion Criteria
- A high-value iterative repeated-run seam is made public or tightened
- The public handle contract remains bounded and consistent with Sprint 49
- All required validation passes before the next solver batch

---

## Day 6: Iterative Handle Expansion Batch II

**Title:** Iterative Batch II  
**Theme:** Extend or tighten the next highest-value iterative lifecycle seam
without reopening broader solver-API questions  
**Time estimate:** 12 hours

### Tasks
1. Land the second bounded iterative lifecycle patch on the next ranked seam.
2. Reconfirm that the patch strengthens support symmetry only where the public
   lifecycle story stays clear and supportable.
3. Preserve one-shot-only behavior for any families intentionally left outside
   the public repeated-run support boundary.
4. Add focused regression coverage if the touched seam would otherwise be
   under-proved.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched iterative follow-ons justified by the batch

### Deliverables
- Second iterative lifecycle patch
- Updated proof for the touched seam
- Validation output for batch II

### Completion Criteria
- A second high-value iterative seam is removed, reduced, or explicitly closed
- Excluded families remain honestly bounded rather than half-exposed
- Required validation passes before eigensolver tightening begins

---

## Day 7: Eigensolver Lifecycle Tightening Batch

**Title:** Eigensolver Batch  
**Theme:** Reconcile any remaining eigensolver lifecycle asymmetry or drift
against the final Sprint 54 support boundary  
**Time estimate:** 12 hours

### Tasks
1. Audit the live public eigensolver repeated-run surface against the Day 4
   decisions.
2. Land the bounded eigensolver lifecycle tightening patch:
   - contract cleanup
   - support-boundary clarification
   - implementation or regression tightening if needed
3. Preserve the existing supported symmetric eigensolver handle story and keep
   excluded subpaths explicit rather than implied.
4. Refresh any direct eigensolver regression proof required by the patch.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched eigensolver follow-ons justified by the batch

### Deliverables
- Eigensolver lifecycle tightening patch
- Updated eigensolver support-boundary notes
- Validation output for the batch

### Completion Criteria
- Remaining public eigensolver lifecycle drift is removed or explicitly bounded
- Supported eigensolver handles read coherently beside the iterative surface
- Validation passes before benchmark alignment begins

---

## Day 8: Public Reuse Benchmark Alignment Audit

**Title:** Benchmark Audit  
**Theme:** Audit the repeated-run benchmark surfaces against the final public
support set before editing the benchmark drivers  
**Time estimate:** 10 hours

### Tasks
1. Audit repeated-run iterative and eigensolver benchmarks for alignment with
   the final Sprint 54 public lifecycle support boundary.
2. Separate benchmarks that already prove the public handle path from those
   still proving internal-only reuse seams or outdated support assumptions.
3. Rank the smallest benchmark updates that would make the measured story match
   the public support story.
4. Explicitly reject broad benchmark-framework redesign.
5. Record the benchmark-alignment target list and landing order.

### Deliverables
- Public reuse benchmark audit
- Ranked benchmark-alignment target list
- Explicit non-goal boundary for benchmark work

### Completion Criteria
- The benchmark queue is reduced to a small concrete update set
- Public-handle proof and internal-only proof surfaces are clearly separated
- Sprint 54 can land benchmark updates without reopening framework work

---

## Day 9: Public Reuse Benchmark Alignment Batch

**Title:** Benchmark Batch  
**Theme:** Update the repeated-run benchmark surfaces so they match the final
public solver lifecycle support set  
**Time estimate:** 10 hours

### Tasks
1. Land the bounded repeated-run benchmark updates from the Day 8 audit.
2. Keep benchmark reporting truthful about which path is being measured:
   - public handle path
   - one-shot path
   - intentionally excluded family behavior where relevant
3. Preserve any benchmark coverage that still matters for non-public internal
   performance seams, but avoid presenting it as public support proof.
4. Refresh benchmark-facing README or usage text only if the live benchmark
   behavior changes materially.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched benchmark follow-ons justified by the batch

### Deliverables
- Updated repeated-run benchmark drivers
- Truthful benchmark-facing reporting
- Validation output for the batch

### Completion Criteria
- The benchmark story now matches the final public solver-lifecycle support set
- No benchmark overclaims remain about unsupported repeated-run families
- Validation passes before docs/examples/regressions are broadened

---

## Day 10: Regression and Example Adoption Batch I

**Title:** Adoption Batch I  
**Theme:** Add the highest-value direct tests and example/doc adoption for the
final repeated-run solver story  
**Time estimate:** 12 hours

### Tasks
1. Land the first bounded regression/example adoption batch on the highest
   signal public solver surfaces.
2. Add or tighten direct tests for the newly supported repeated-run families or
   explicit rejection behavior for intentionally excluded families.
3. Update the strongest example or README entry points so they describe the
   final solver-lifecycle boundary honestly.
4. Keep example scope bounded; do not rewrite the whole tutorial or example
   corpus.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - relevant example or test follow-ons justified by the batch

### Deliverables
- First regression/example adoption patch
- Direct proof for the highest-value public solver paths
- Validation output for batch I

### Completion Criteria
- The highest-value public solver-lifecycle surfaces now have direct proof
- The strongest user-facing docs/examples reflect the final support boundary
- Validation passes before the final sweep batch

---

## Day 11: Regression and Example Adoption Batch II

**Title:** Adoption Batch II  
**Theme:** Close the remaining high-value proof and docs-example gaps without
starting broad documentation churn  
**Time estimate:** 10 hours

### Tasks
1. Land the second bounded regression/example/docs adoption batch on the next
   ranked surfaces.
2. Add explicit proof for any remaining supported repeated-run family that
   still lacks direct public-handle coverage.
3. Add bounded docs clarifications for intentionally excluded families where a
   reader could otherwise infer planned support.
4. Reconfirm that examples, README text, and regression coverage all describe
   the same final support boundary.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched follow-ons justified by the batch

### Deliverables
- Second regression/example adoption patch
- Closed proof gaps for the final support surface
- Validation output for batch II

### Completion Criteria
- Remaining high-value proof gaps are closed
- Docs/examples no longer drift from the final public lifecycle support story
- Validation passes before compatibility audit and full sweep

---

## Day 12: Post-Landing Compatibility Audit

**Title:** Compatibility Audit  
**Theme:** Audit the landed Sprint 54 branch against the preserved public
handle fence and chosen exclusion boundaries  
**Time estimate:** 10 hours

### Tasks
1. Audit the landed branch across code, docs, examples, benchmarks, and tests.
2. Reconfirm the preserved compatibility rules:
   - supported one-shot APIs remain first-class
   - supported handle paths remain opt-in repeated-run paths
   - reuse semantics remain honestly bounded
   - explicitly excluded families still read as bounded exclusions, not broken
     partial implementations
3. Check whether any benchmark, example, or README text now overclaims solver
   lifecycle symmetry.
4. Fix the Day 13 validation checklist from the landed state.
5. Record any residual non-blocking queue explicitly for later sprints.

### Deliverables
- Post-landing compatibility audit
- Day 13 validation checklist
- Residual queue notes

### Completion Criteria
- The landed Sprint 54 branch still matches the preserved public-handle fence
- No blocker-level drift remains between code, docs, tests, and benchmarks
- The final validation sweep has an explicit checklist

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full reviewed baseline plus the Sprint 54 targeted solver
follow-ons from the landed branch state  
**Time estimate:** 12 hours

### Tasks
1. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm reviewed CMake parity and truthfulness anchors from the live
   validated state.
3. Run the targeted Sprint 54 iterative/eigensolver follow-ons justified by the
   landed changes.
4. Record representative measured repeated-run outputs from the final support
   surfaces.
5. Write the validation artifact and any final non-blocking residual notes.

### Deliverables
- Full validation record
- Final truthfulness-anchor measurements
- Targeted solver follow-on results

### Completion Criteria
- All required validation passes
- Maintained reviewed parity anchors remain exact or are documented truthfully
- No new blocker queue appears during the full sweep

---

## Day 14: Closeout and Handoff

**Title:** Closeout  
**Theme:** Package Sprint 54’s solver-lifecycle decisions, implementation, and
validated boundary for the next sprint  
**Time estimate:** 10 hours

### Tasks
1. Summarize the final Sprint 54 support boundary:
   - supported repeated-run solver families
   - intentionally excluded or deferred families
   - preserved one-shot compatibility rules
2. Summarize the landed implementation, benchmark, regression, and docs
   outcomes from the validated state.
3. Record the final residual queue for later sprints without turning it into a
   hidden blocker list.
4. Write the Day 14 closeout artifact and update working notes with the final
   synthesis.
5. Check whether Sprint 54 revealed any need to update
   `docs/planning/EPIC_5/PROJECT_PLAN.md`.

### Deliverables
- Sprint 54 closeout and handoff artifact
- Final working-notes synthesis
- Explicit next-sprint residual queue

### Completion Criteria
- Sprint 54 leaves behind one coherent repeated-run solver-lifecycle package
- The next sprint inherits a clear supported-vs-excluded boundary
- Sprint closeout is grounded in the Day 13 validated state
