# Sprint 65 Plan: Performance Governance, Benchmark Consolidation & Solver Efficiency Follow-Through

**Sprint Duration:** 14 days  
**Goal:** Turn the benchmark surface into a clearer performance-governance
surface and apply the resulting insight to the highest-value solver-efficiency
follow-through while preserving the reviewed truthfulness contract. This sprint
implements the Sprint 65 section of
`docs/planning/EPIC_6/PROJECT_PLAN.md`.

**Starting Point:** Sprint 64 closed with the first bounded Epic 6
backend/performance architecture package landed and validated:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- the first backend-aware lane is now explicitly bounded to CSC supernodal
  Cholesky
- the default self-contained build remains authoritative
- `bench_chol_csc` now reports the active path-identification fields and the
  active dense-kernel descriptor
- the next Epic 6 priority is no longer the first backend-aware kernel seam
  itself; it is performance governance, benchmark consolidation, and
  efficiency follow-through guided by a smaller canonical proof surface

The next highest-value work is not another generic backend framework sprint.
It is a bounded performance-governance sprint focused on clarifying which
benchmarks are authoritative, normalizing how they report, shrinking the
canonical performance surface, and then applying that information to the
highest-value solver-efficiency seams.

**End State:** Sprint 65 leaves behind one coherent performance-governance
package:
- a ranked benchmark-role audit separating regression-sensitive, proof, and
  exploratory drivers
- normalized benchmark output and documentation around the new taxonomy
- a smaller canonical performance surface the repo can actually maintain
- targeted solver-efficiency follow-through on the highest-value surfaced seam
- bounded local/CI-friendly regression reporting or checks where maintainable
- full validation and closeout from the landed state

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
144 hours, staying within the Sprint 65 estimate and below the 168-hour limit.

---

## Day 1: Sprint 65 Scope Audit & Performance Governance Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 65 project-plan scope plus the Sprint 64 validated
close into a bounded performance-governance implementation map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 65 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 64 retrospective, and
   the strongest Sprint 64 closeout artifacts.
2. Reconfirm the preserved Sprint 65 constraints:
   - no fake performance claims beyond reviewed evidence
   - no benchmark-governance sprawl disconnected from real proof surfaces
   - no broad backend/platform rewrite disguised as efficiency work
   - no widening that weakens the self-contained default build or truthfulness
     contract
3. Define the Sprint 65 workstreams explicitly:
   - benchmark-role audit
   - output/taxonomy normalization
   - canonical baseline selection
   - solver-efficiency follow-through
   - regression/reporting checks
   - docs/example alignment
   - validation and closeout
4. Record the strongest likely Sprint 65 touch surfaces:
   - benchmark binaries and benchmark docs
   - quality/validation truth surfaces
   - solver and hotspot implementation seams likely to be influenced by the
     benchmark audit
5. Open Sprint 65 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 65 scope inventory
- Performance-governance baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 65 starts from the Sprint 64 validated close rather than reopening
  backend-abstraction-first work
- The performance-governance workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or code edits land

---

## Day 2: Validation Baseline & Benchmark/Proof Rerun Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed baseline and rerun set that Sprint 65
benchmark, efficiency, and regression changes must preserve  
**Time estimate:** 8 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity counts
   - current quality/truthfulness wording
2. Reconfirm the mandatory gate for later `*.c` / `*.h` days:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial benchmark-governance,
   performance, or solver-efficiency work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Sprint 65:
   - benchmark binaries under possible taxonomy/output change
   - direct/CSC proof surfaces likely to matter for efficiency follow-through
   - representative examples and adjacent solver sentinels that should not drift
5. Record the authoritative validation split for docs-only, bounded code-day,
   and substantial performance-governance days.

### Deliverables
- Refreshed validation notes
- Sprint 65 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 65 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before implementation work begins
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Benchmark Role Audit

**Title:** Benchmark Audit I  
**Theme:** Re-rank benchmark drivers into regression-sensitive, proof, and
exploratory categories from the live repo state  
**Time estimate:** 11 hours

### Tasks
1. Inventory the current benchmark binaries and their present roles across:
   - throughput proof
   - workflow proof
   - backend/path proof
   - exploratory comparison
2. Classify each benchmark by:
   - user-facing importance
   - repeatability and noise tolerance
   - maintainability in local and CI contexts
   - overlap or duplication with other proof surfaces
3. Identify the strongest current category mismatches:
   - benchmarks acting like authoritative proof without stable outputs
   - exploratory drivers being treated like regression gates
   - redundant benchmark lanes with no distinct product value
4. Rank the strongest candidates for the canonical Sprint 65 surface.
5. Write the audit artifact with the explicit benchmark-role map.

### Deliverables
- Live benchmark-role inventory
- Ranked benchmark-category draft
- Initial canonical-surface candidate list

### Completion Criteria
- The broad “benchmark/performance governance” claim is reduced to a concrete
  role map
- The strongest benchmark-role contradictions are explicit before redesign begins
- Day 4 can proceed from a real current-state taxonomy instead of generic
  benchmark concerns

---

## Day 4: Benchmark Role Follow-Through & Canonical-Surface Rerank

**Title:** Benchmark Audit II  
**Theme:** Separate must-keep canonical benchmark surfaces from later
exploratory or maintenance-only drivers  
**Time estimate:** 10 hours

### Tasks
1. Re-rank the Day 3 categories against the Epic 6 state-of-the-art target.
2. Separate:
   - canonical maintained performance surfaces
   - benchmark-side proof surfaces
   - exploratory drivers that stay out of the authoritative regression lane
3. Confirm which binaries belong in:
   - machine-readable regression-sensitive output normalization
   - documentation-only explanation
   - later consolidation or possible de-emphasis
4. Fix the first Sprint 65 normalization target set in writing.
5. Record the residual benchmark queue that Sprint 65 should not absorb.

### Deliverables
- Refined benchmark taxonomy
- Canonical baseline candidate set
- Deferred benchmark residual map

### Completion Criteria
- The Sprint 65 target set is smaller and sharper than the original epic-level
  review
- The first normalization target set is explicit before output design begins
- Later exploratory or low-value work is clearly separated from the bounded
  Sprint 65 surface

---

## Day 5: Output & Taxonomy Normalization Design

**Title:** Normalization Design  
**Theme:** Define the normalized machine-readable output and category contract
for the selected benchmark surfaces  
**Time estimate:** 11 hours

### Tasks
1. Design the benchmark taxonomy vocabulary and exact maintained meanings for:
   - regression-sensitive
   - proof
   - exploratory
2. Define the normalized output contract for the selected benchmark surfaces:
   - stable column expectations
   - path/backend identifiers where relevant
   - compact enough retained outputs for docs and reviews
3. Define the preserved compatibility rules:
   - no misleading benchmark claims
   - no unstable pseudo-regression gates
   - no output churn without a reason tied to governance clarity
4. Decide which normalization belongs in:
   - benchmark binary output
   - benchmark docs
   - maintainer policy
   - CI/reporting only if maintainable
5. Record the explicit safety contract for the first implementation batch.

### Deliverables
- Output/taxonomy design artifact
- Explicit safety/compatibility contract
- First normalization implementation contract

### Completion Criteria
- The benchmark taxonomy is explicit before output edits start
- Binary output, docs, and maintainer ownership are separated clearly enough
  to prevent drift
- The canonical performance story is defined tightly enough to support later
  efficiency work

---

## Day 6: Canonical Performance Surface Design

**Title:** Canonical Surface  
**Theme:** Convert the benchmark taxonomy into a smaller set of performance
baselines the repo can actually maintain  
**Time estimate:** 9 hours

### Tasks
1. Select the canonical maintained performance surfaces from the Day 5 design.
2. Define what each canonical surface proves:
   - backend/path identity
   - throughput signal
   - repeated-run efficiency signal
   - workflow non-regression
3. Define what is intentionally *not* canonical:
   - highly noisy local probes
   - exploratory corpus sweeps
   - one-off engineering diagnostics
4. Fix the exact Day 7-10 touched-file fence for taxonomy/output and
   efficiency work.
5. Record the first solver-efficiency target candidate exposed by the audit.

### Deliverables
- Canonical performance-surface plan
- Exact touched-surface map
- Initial solver-efficiency target shortlist

### Completion Criteria
- The canonical performance surface is smaller than the current benchmark set
- The implementation fence is explicit before the first code/doc batch lands
- The efficiency follow-through target emerges from the benchmark audit rather
  than assumption

---

## Day 7: Solver-Efficiency Target Selection & Landing Design

**Title:** Efficiency Design  
**Theme:** Fix the first actual solver-efficiency follow-through seam and its
proof plan before implementation starts  
**Time estimate:** 10 hours

### Tasks
1. Re-read the live benchmark/implementation surfaces for the strongest
   efficiency candidate exposed by Days 3-6.
2. Select the first bounded efficiency target by:
   - payoff visible in the maintained benchmark surface
   - bounded touched surface
   - proof burden and fallback risk
3. Define the exact implementation fence for:
   - solver code
   - benchmark output or metadata if needed
   - regression or proof surfaces
4. Separate:
   - required implementation files
   - conditional support files
   - explicit non-goals
5. Record the landing design and Day 8/9 split in writing.

### Deliverables
- Solver-efficiency landing design
- Exact first code-batch fence
- Proof and non-goal split

### Completion Criteria
- The efficiency target is selected from measured benchmark evidence
- The first code batch has an exact touched-file fence
- The implementation plan is bounded enough to land without widening the sprint

---

## Day 8: Benchmark Taxonomy & Output Batch 1

**Title:** Output Batch I  
**Theme:** Land the first bounded taxonomy/output normalization slice on the
selected canonical benchmark surfaces  
**Time estimate:** 12 hours

### Tasks
1. Implement the first normalized output changes on the selected benchmark
   binaries.
2. Land the minimum benchmark-local doc updates needed to explain the new
   fields or categories.
3. Add or tighten proof for:
   - stable output fields
   - category truthfulness
   - unchanged fallback/solver correctness where applicable
4. Keep the batch inside the Day 7 fence:
   - no broad CI lane expansion
   - no benchmark-governance rewrite outside the selected surfaces
   - no unrelated solver changes
5. Run the required validation gate for the touched implementation state.

### Deliverables
- First normalized benchmark output batch
- Benchmark-local proof updates
- Working-notes landing record

### Completion Criteria
- The selected canonical benchmark surfaces now report the intended normalized
  fields
- The batch improves performance-governance clarity without widening beyond
  the selected surfaces
- Validation passes from the landed Day 8 tree

---

## Day 9: Canonical Baseline Consolidation Batch

**Title:** Output Batch II  
**Theme:** Complete the smaller canonical performance surface and its
maintained classification story  
**Time estimate:** 11 hours

### Tasks
1. Finish the benchmark-classification and canonical-baseline follow-through on
   the remaining selected surfaces.
2. Tighten benchmark docs so the canonical/proof/exploratory split reads as
   one coherent model.
3. Add bounded maintainer-facing ownership where necessary.
4. Re-run the most important benchmark proof surfaces and confirm the new
   outputs remain truthful.
5. Record the exact efficiency target carried into Day 10 implementation.

### Deliverables
- Canonical benchmark surface batch
- Benchmark docs consolidation
- Finalized efficiency target handoff

### Completion Criteria
- The canonical performance surface is explicit in both outputs and docs
- The maintained benchmark story is smaller and clearer than at sprint start
- Day 10 starts from a fixed efficiency target instead of a generic optimization backlog

---

## Day 10: Solver-Efficiency Follow-Through Batch

**Title:** Efficiency Batch  
**Theme:** Apply one bounded solver-efficiency improvement exposed by the
benchmark-governance pass  
**Time estimate:** 12 hours

### Tasks
1. Land the first solver-efficiency improvement inside the Day 7 fence.
2. Preserve the reviewed truthfulness and fallback contract while improving the
   selected hot path.
3. Add or tighten the proof home for:
   - correctness
   - non-regression
   - benchmark-visible retained signal
4. Re-run the targeted benchmark and proof surfaces that motivated the batch.
5. Record the implementation and retained output evidence.

### Deliverables
- Solver-efficiency implementation batch
- Targeted proof updates
- Retained benchmark evidence

### Completion Criteria
- A real efficiency improvement lands on the selected seam without widening
  into generic optimization work
- The targeted proof and benchmark surfaces stay truthful
- Validation passes from the landed Day 10 tree

---

## Day 11: Local/CI-Friendly Regression Checks

**Title:** Regression Checks  
**Theme:** Add bounded performance-regression reporting or checks only where
they are genuinely maintainable  
**Time estimate:** 11 hours

### Tasks
1. Re-rank the canonical benchmark surfaces for regression-check suitability.
2. Land the smallest maintainable regression/reporting addition:
   - local report
   - CI-friendly summary
   - threshold-free comparison output
   only if it is stable enough to keep
3. Avoid:
   - noisy pseudo-thresholds
   - fragile timing gates
   - platform-closure claims unsupported by evidence
4. Tighten benchmark/maintainer wording around the new regression story.
5. Validate the landed regression/reporting surface.

### Deliverables
- Bounded regression/reporting addition
- Maintainer/regression wording follow-through
- Working-notes decision record

### Completion Criteria
- Sprint 65 adds regression-sensitive performance reporting only where it is
  actually maintainable
- The canonical benchmark story is stronger without creating fake timing gates
- Validation passes from the landed Day 11 tree

---

## Day 12: Docs & Example Alignment

**Title:** Docs Follow-Through  
**Theme:** Align top-level docs and examples with the new performance-governance
story and the landed efficiency follow-through  
**Time estimate:** 9 hours

### Tasks
1. Update the top-level user-facing docs where the benchmark/performance story
   is actually taught.
2. Align example and benchmark docs so proof ownership is clearer:
   - what benchmarks prove
   - what examples teach
   - what remains maintainer-only interpretation
3. Keep the batch bounded to high-signal touched surfaces only.
4. Recheck terminology consistency across README, benchmark docs, examples, and
   maintainer surfaces.
5. Record the docs-only closeout state for Day 13 validation.

### Deliverables
- Docs/example alignment batch
- Updated maintainer/performance story
- Final docs-only follow-through artifact

### Completion Criteria
- The benchmark-governance story reads coherently across the maintained docs
- Example and benchmark proof roles are clearer than at sprint start
- The batch stays bounded and does not widen into unrelated doc cleanup

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full reviewed validation set plus targeted performance and
benchmark follow-ons and capture the retained evidence  
**Time estimate:** 12 hours

### Tasks
1. Run the required full validation set:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm reviewed CMake parity counts and any authoritative timing notes.
3. Re-run the targeted Sprint 65 proof set:
   - canonical benchmark surfaces
   - touched solver proof binaries
   - representative examples
4. Capture representative retained outputs for the normalized benchmark and
   efficiency surfaces.
5. Record any non-blocking warnings or caveats without diluting the pass/fail
   baseline.

### Deliverables
- Full validation record
- Targeted proof rerun results
- Representative retained output set

### Completion Criteria
- All required validation passes from the landed Sprint 65 state
- Reviewed parity anchors remain exact or any change is explained concretely
- The branch is ready for closeout from a validated baseline

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Convert the validated Sprint 65 branch into a clear handoff package
for Sprint 66 and the remaining Epic 6 performance-governance work  
**Time estimate:** 8 hours

### Tasks
1. Summarize the landed performance-governance and efficiency outcomes.
2. Record the preserved compatibility and truthfulness fence:
   - reviewed baseline remains authoritative
   - canonical benchmark surface remains smaller and explicit
   - default build and fallback-path truth stay visible
3. Record the validated Day 13 baseline and the strongest retained benchmark,
   regression, and solver-efficiency evidence.
4. Define the remaining deferred queue for later backend, packaging, or
   broader performance-governance work.
5. Write the closeout artifact and update working notes for retrospective
   creation.

### Deliverables
- Sprint 65 closeout artifact
- Final working-notes summary
- Explicit Sprint 66 handoff queue

### Completion Criteria
- Sprint 65 ends with one coherent performance-governance package
- The smaller canonical performance surface and preserved truthfulness fence
  are explicit in writing
- The next sprint inherits a ranked queue instead of a generic benchmark or
  efficiency backlog
