# Sprint 52 Plan: Analysis/Refactor Integration & Direct-Solver Lifecycle Phase 2

**Sprint Duration:** 14 days  
**Goal:** Deepen the direct-solver lifecycle model so the public
analysis/factor/refactor workflow is a real first-class path rather than a
lightly wrapped alternative surface. This sprint implements the Sprint 52
section of `docs/planning/EPIC_5/PROJECT_PLAN.md`.

**Starting Point:** Sprint 51 closed with the first implemented public
direct-solver lifecycle phase in place:
- shared lifecycle contract visible in the direct headers
- bounded LU lifecycle integration
- shared Cholesky and LDL^T lifecycle routing
- preserved one-shot compatibility posture
- focused lifecycle regression coverage
- validated reviewed-baseline closeout at `53` reviewed CMake tests

The remaining work is now deeper integration work rather than phase-1 API
exposure: reduce avoidable fallback to one-shot symbolic work, tighten
refactor semantics and reuse behavior, refresh factor-many benchmark proof,
expand public lifecycle regression coverage, and align the stronger repeated-run
story in the most important caller-facing docs/examples.

**End State:** Sprint 52 leaves behind a stronger public
analysis/factor/refactor direct-solver workflow with tighter numeric reuse and
refactor behavior, measurable factor-many evidence, expanded regression proof,
and aligned high-signal documentation/example surfaces that hand off a stable
Phase 2 lifecycle package to Sprint 53.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
156 hours, matching the Sprint 52 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 52 Scope Audit & Phase-2 Baseline

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 52 project-plan items plus the Sprint 51 closeout
package into a bounded deeper-integration map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 52 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`,
   the Sprint 51 closeout artifact, the Sprint 51 retrospective, and the
   highest-value Sprint 51 implementation artifacts.
2. Reconfirm the preserved Sprint 52 constraints:
   - keep one-shot LU / Cholesky / LDL^T APIs first-class
   - preserve the Sprint 50 non-goal and compatibility fence
   - deepen the existing analysis/factor/refactor path instead of inventing a
     new generic direct abstraction
   - preserve the strongest local reviewed baseline and truthfulness anchors
3. Define the Sprint 52 implementation workstreams explicitly:
   - analysis-contract audit
   - numeric reuse integration
   - refactor path tightening
   - factor-many benchmark proof
   - docs/example adoption
   - regression expansion
   - validation and closeout
4. Record the highest-risk seams for the sprint:
   - silent fallback into one-shot symbolic work
   - overstated reuse/refactor guarantees
   - family drift between LU, Cholesky, and LDL^T
   - benchmark claims becoming stronger than measured evidence
5. Open Sprint 52 working notes and record starting assumptions, landing
   order, and touched-surface expectations.

### Deliverables
- Sprint 52 scope inventory
- Phase-2 baseline notes
- Working-notes starting assumptions

### Completion Criteria
- Sprint 52 starts from the Sprint 51 implementation state rather than
  reopening basic lifecycle design
- Preserved compatibility and validation constraints are explicit before deeper
  code edits begin
- The phase-2 workstreams are named before analysis/refactor patches land

---

## Day 2: Validation Baseline & Touched-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed local baseline and the exact rerun set
Sprint 52 code days must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the maintained reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity
   - current truthfulness-anchor counts
2. Reconfirm the mandatory gate for any later `*.c` / `*.h` lifecycle batch:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial public lifecycle batches:
   - `make quality-review-full`
4. Refresh the targeted follow-on binaries Sprint 52 is most likely to need:
   - `./build/example_analysis`
   - `./build/bench_refactor`
   - `./build/bench_refactor_csc`
   - `./build/test_integration`
   - `./build/test_cholesky`
   - `./build/test_ldlt`
   - `./build/test_etree`
   - `./build/test_chol_csc`
   - `./build/test_ldlt_csc`
5. Record the smallest authoritative validation boundary for docs-only notes
   versus code-touch deeper-integration days.

### Deliverables
- Refreshed validation/truthfulness notes
- Sprint 52 touched-surface rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 52 uses the same baseline wording and parity anchors as the live repo
- The authoritative rerun set is explicit before numeric-reuse work begins
- No validation ambiguity remains around the phase-2 lifecycle landing days

---

## Day 3: Analysis/Factors Contract Audit

**Title:** Contract Audit  
**Theme:** Audit the live public lifecycle ownership and fallback behavior
before tightening deeper integration  
**Time estimate:** 10 hours

### Tasks
1. Audit the live `sparse_analysis_t` / `sparse_factors_t` path across:
   - public headers
   - `src/sparse_analysis.c`
   - LU / Cholesky / LDL^T family integrations
   - direct integration tests
2. Map where the current lifecycle path still falls back into one-shot
   symbolic or family-local setup work.
3. Separate acceptable family-local behavior from avoidable fallback or drift.
4. Identify the highest-value phase-2 integration seams and explicitly reject
   surfaces that would broaden this into redesign work.
5. Write the audit artifact and the ranked implementation target list.

### Deliverables
- Analysis/factors contract audit
- Fallback/drift inventory
- Ranked phase-2 integration targets

### Completion Criteria
- The main phase-2 problem is reduced to named fallback/reuse seams rather than
  a generic “tighten integration” instruction
- The public lifecycle ownership boundary remains explicit
- Sprint 52 can start numeric-reuse patches from a concrete audit

---

## Day 4: Numeric Reuse Integration Batch I

**Title:** Reuse Batch I  
**Theme:** Reduce avoidable fallback to one-shot symbolic work in the highest
value direct lifecycle paths  
**Time estimate:** 12 hours

### Tasks
1. Land the first bounded numeric-reuse integration patch in the strongest
   shared lifecycle seam identified on Day 3.
2. Keep the repeated-run path analysis-centric:
   - analyze once
   - factor / solve
   - refactor / solve many
3. Preserve the one-shot compatibility path and any accepted family-specific
   behavior outside the bounded patch.
4. Add focused comments only where the new reuse path would otherwise be
   unclear to maintainers.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- First numeric-reuse integration patch
- Preserved one-shot compatibility behavior
- Validation output for batch I

### Completion Criteria
- Avoidable fallback is reduced on the strongest shared path
- One-shot compatibility remains intact
- All required validation passes before the next reuse batch

---

## Day 5: Numeric Reuse Integration Batch II

**Title:** Reuse Batch II  
**Theme:** Extend the deeper integration to the next highest-value direct
family path without broadening scope  
**Time estimate:** 12 hours

### Tasks
1. Land the second bounded numeric-reuse patch on the next highest-value seam.
2. Reconfirm that the patch strengthens reuse of:
   - symbolic structure
   - permutation/setup state
   and does not overpromise reuse of old numeric factor contents.
3. Preserve accepted family-local differences where the shared path still
   should not absorb them.
4. Add focused regression coverage if the new path would otherwise be under-
   proved.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched follow-ons justified by the batch

### Deliverables
- Second numeric-reuse integration patch
- Updated proof for the touched reuse seam
- Validation output for batch II

### Completion Criteria
- A second high-value fallback seam is removed or reduced
- Reuse semantics remain honestly bounded
- Required validation passes before refactor tightening begins

---

## Day 6: Refactor Path Tightening Batch I

**Title:** Refactor Batch I  
**Theme:** Tighten the public refactor semantics and result reuse behavior on
the highest-value lifecycle path  
**Time estimate:** 12 hours

### Tasks
1. Audit the live `sparse_refactor_numeric(...)` path against the Day 3
   findings.
2. Land the first bounded refactor-tightening patch.
3. Preserve the first-factorization-on-zeroed-state behavior that Sprint 51
   deliberately supported.
4. Reconfirm that refactor wording and implementation still describe same-
   pattern numeric refresh rather than a broader rebuild guarantee.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- First refactor-tightening patch
- Updated refactor semantics/proof notes
- Validation output for refactor batch I

### Completion Criteria
- The public refactor path is stronger and more explicit than Sprint 51
- Same-pattern numeric refresh remains the governing contract
- Validation passes before later refactor work begins

---

## Day 7: Refactor Path Tightening Batch II

**Title:** Refactor Batch II  
**Theme:** Complete the bounded refactor tightening across the remaining
highest-value touched family path  
**Time estimate:** 10 hours

### Tasks
1. Land the second bounded refactor-tightening patch.
2. Recheck lifecycle behavior on zero-init, first-factorization, and repeat
   refactor/solve sequences.
3. Keep the patch inside the Sprint 50-52 fence:
   - no raw storage exposure
   - no broad factor-container redesign
   - no structural-pattern verifier redesign
4. Update the phase-2 artifact with the final refactor behavior boundary.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched follow-ons justified by the batch

### Deliverables
- Second refactor-tightening patch
- Updated final refactor boundary notes
- Validation output for refactor batch II

### Completion Criteria
- Phase-2 refactor tightening is materially complete on the targeted paths
- The public behavior boundary remains explicit and bounded
- Required validation passes before benchmark proof work begins

---

## Day 8: Factor-Many Benchmark Proof

**Title:** Benchmark Proof  
**Theme:** Refresh or extend factor-many benchmark evidence so the deeper
integration story is measured instead of implied  
**Time estimate:** 12 hours

### Tasks
1. Audit the live factor-many benchmark surfaces:
   - `bench_refactor`
   - `bench_refactor_csc`
   - benchmark docs
2. Land only the smallest high-signal benchmark updates needed to prove the
   stronger lifecycle story.
3. Keep benchmark changes bounded to measurement/proof, not framework redesign.
4. Reconfirm representative cases across:
   - moderate analysis/refactor comparisons
   - heavier CSC repeated-run comparisons
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched benchmark follow-ons

### Deliverables
- Updated factor-many benchmark proof
- Measured representative benchmark outputs
- Validation output for the benchmark batch

### Completion Criteria
- The phase-2 performance story is grounded in current benchmark evidence
- Benchmark scope remains bounded and non-framework-like
- Validation passes before docs/example adoption begins

---

## Day 9: Post-Benchmark Audit & Adoption Boundary

**Title:** Adoption Audit  
**Theme:** Narrow the remaining docs/example work after the integration and
benchmark proof land  
**Time estimate:** 10 hours

### Tasks
1. Audit the strongest caller-facing direct repeated-run surfaces after the
   code and benchmark work:
   - `README.md`
   - `examples/example_analysis.c`
   - `examples/README.md`
   - `benchmarks/README.md`
2. Separate what now needs updating from what is already aligned enough to
   leave alone.
3. Fix the Day 10 docs/example scope boundary:
   - strongest adoption targets
   - secondary surfaces that should not be widened yet
4. Reconfirm that tutorial-scale or sweeping example conversion is still
   out-of-scope.
5. Write the post-benchmark audit artifact.

### Deliverables
- Post-benchmark caller-surface audit
- Day 10 adoption boundary
- Deferred-surface notes

### Completion Criteria
- The remaining adoption queue is concrete instead of generic
- High-signal caller surfaces are identified before docs/example edits begin
- No broad docs rewrite is accidentally pulled into the sprint

---

## Day 10: Example / Doc Adoption Batch

**Title:** Adoption Batch  
**Theme:** Align the highest-value public docs/examples with the stronger
analysis/factor/refactor lifecycle story  
**Time estimate:** 12 hours

### Tasks
1. Update the smallest highest-signal docs/example surfaces identified on
   Day 9.
2. Keep the repeated-run direct story explicit around:
   - analyze once
   - factor / solve
   - refactor / solve many
   - preserved one-shot compatibility path
3. Avoid converting every example; only update the surfaces that actually teach
   or summarize the public repeated-run story.
4. Reconfirm any new links or wording against the live implementation state.
5. Run targeted touched-surface sanity checks and any justified binaries.

### Deliverables
- Updated direct repeated-run docs/examples
- Aligned adoption wording for the touched surfaces
- Targeted sanity-check output

### Completion Criteria
- The highest-value caller-facing repeated-run surfaces are aligned with the
  live phase-2 behavior
- The one-shot story remains visible and honest
- The batch stays bounded to adoption rather than tutorial overhaul

---

## Day 11: Regression Coverage Expansion

**Title:** Regression Batch  
**Theme:** Add the remaining direct public lifecycle and factor-many
regression proof for the deeper phase-2 integration  
**Time estimate:** 10 hours

### Tasks
1. Add the next highest-value regression cases for:
   - public analysis/factor/refactor sequencing
   - first-factorization and repeat-refactor behavior
   - compatibility parity between one-shot and lifecycle paths where relevant
2. Keep the new cases concentrated in the strongest existing regression homes
   rather than scattering them widely.
3. Avoid broad test-suite churn that does not directly strengthen the Sprint 52
   public lifecycle claim.
4. Re-run the strongest touched follow-ons for the added cases.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- Expanded public lifecycle regression coverage
- Focused direct-lifecycle follow-on results
- Validation output for the regression batch

### Completion Criteria
- The deeper phase-2 lifecycle path is directly proved in regression coverage
- The coverage remains focused on public behavior rather than internal trivia
- Full validation passes before the compatibility audit

---

## Day 12: Post-Landing Compatibility Audit

**Title:** Compatibility Audit  
**Theme:** Verify that the landed phase-2 implementation still matches the
Sprint 50-51 compatibility contract  
**Time estimate:** 12 hours

### Tasks
1. Audit the live touched headers, sources, tests, examples, and benchmark docs
   after the Day 4-11 batches.
2. Reconfirm:
   - one-shot APIs remain first-class peer entry points
   - repeated direct runs remain analysis/factors-centric
   - reuse/refactor semantics remain honestly bounded
   - benchmark/doc claims do not overstate the measured evidence
3. Identify any blocker-level residual drift before the validation sweep.
4. Fix the Day 13 pre-validation checklist from the actual landed state.
5. Write the post-landing compatibility audit artifact.

### Deliverables
- Post-landing compatibility audit
- Final pre-validation checklist
- Residual-risk notes

### Completion Criteria
- The live branch still matches the intended compatibility fence
- Any remaining drift is classified before Day 13 validation
- The final validation checklist is explicit and complete

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full validated closeout from the landed Sprint 52 state  
**Time estimate:** 12 hours

### Tasks
1. Run the full required validation gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm truthfulness anchors:
   - reviewed CMake `ctest -N`
   - Makefile/CMake parity
   - full reviewed CMake `ctest`
3. Re-run the Sprint 52 targeted follow-ons identified on Day 2 and refined by
   the later implementation work.
4. Record representative direct results for:
   - public lifecycle correctness
   - factor-many benchmark evidence
   - direct family regression reruns
5. Write the full validation artifact and working-notes entry.

### Deliverables
- Full validation-sweep artifact
- Updated working notes with exact validation record
- Final measured truthfulness anchors

### Completion Criteria
- All required validation passes
- Maintained reviewed-baseline anchors remain exact
- The sprint has a real measured validation close state before Day 14

---

## Day 14: Closeout & Sprint 53 Handoff

**Title:** Closeout  
**Theme:** Summarize the validated phase-2 lifecycle package and hand forward
the next bounded queue  
**Time estimate:** 12 hours

### Tasks
1. Summarize the actual Sprint 52 delivered package:
   - stronger analysis/refactor integration
   - tightened refactor semantics
   - refreshed factor-many benchmark proof
   - updated docs/examples
   - expanded regression proof
2. Reconfirm the preserved non-goals and compatibility boundaries.
3. Record the exact validated baseline carried into Sprint 53.
4. Identify only the real remaining next-step queue implied by the landed
   state.
5. Write the closeout/handoff artifact and final working-notes synthesis.

### Deliverables
- Sprint 52 closeout artifact
- Final working-notes synthesis
- Sprint 53 handoff boundary

### Completion Criteria
- Sprint 52 closes from the Day 13 validated baseline
- The delivered phase-2 lifecycle package is summarized clearly and truthfully
- The next queue is bounded to real post-sprint work rather than generic
  future ideas
