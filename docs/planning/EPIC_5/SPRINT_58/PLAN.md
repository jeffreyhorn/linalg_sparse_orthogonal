# Sprint 58 Plan: Documentation, Examples & Benchmark Story Simplification

**Sprint Duration:** 14 days  
**Goal:** Reduce stale sprint-history narrative across the highest-signal
public docs, examples, headers, and benchmark docs so the shipped workflow
story reads as a stable product surface rather than an accumulated sprint log.
This sprint implements the Sprint 58 section of
`docs/planning/EPIC_5/PROJECT_PLAN.md`.

**Starting Point:** Sprint 57 closed with the maintained implementation and
regression baseline still intact:
- direct lifecycle, repeated-run, and factor-many behavior already validated
- giant implementation and test hotspots already reduced across Sprints 55-57
- remaining Epic 5 cleanup pressure now concentrated in public wording,
  example framing, and benchmark taxonomy drift

The next highest-value work is not feature expansion. It is product-surface
simplification across the strongest caller-facing materials:
- `README.md`
- `docs/tutorial.md`
- public headers under `include/`
- `examples/README.md` plus the highest-signal examples
- `benchmarks/README.md`

**End State:** Sprint 58 leaves behind smaller, clearer public docs and
examples, a more stable benchmark taxonomy, less stale sprint chronology in
public headers, and the same reviewed validation baseline and compatibility
fence preserved through Sprint 57.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
136 hours, matching the Sprint 58 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 58 Scope Audit & Product-Surface Baseline

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 58 project-plan items plus the Sprint 57 close
state into a bounded docs/examples/benchmark simplification map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 58 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`,
   the Sprint 57 closeout, and the Epic 5 notes related to caller-facing
   documentation drift.
2. Reconfirm the preserved Sprint 58 constraints:
   - no public API redesign
   - no reopening validated lifecycle semantics
   - wording-first, not feature-first
   - preserve reviewed validation and truthfulness anchors
3. Define the Sprint 58 workstreams explicitly:
   - public docs audit
   - README/tutorial reduction
   - public-header narrative cleanup
   - example modernization
   - benchmark taxonomy cleanup
   - final sanity sweep and closeout
4. Record the highest-signal public surfaces most likely to be touched:
   - `README.md`
   - `docs/tutorial.md`
   - `include/sparse_analysis.h`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
   - `examples/README.md`
   - `benchmarks/README.md`
5. Open Sprint 58 working notes and record the initial landing order plus
   validation expectations.

### Deliverables
- Sprint 58 scope inventory
- Public-surface baseline notes
- Working-notes starting assumptions

### Completion Criteria
- Sprint 58 starts from the Sprint 57 validated state rather than reopening
  design questions
- Preserved compatibility and scope fences are explicit before docs or example
  changes land
- The documentation/examples/benchmark workstreams are named before
  implementation begins

---

## Day 2: Validation Baseline & Touched-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed local baseline and the exact rerun set
Sprint 58 code days must preserve  
**Time estimate:** 9 hours

### Tasks
1. Reconfirm the maintained reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity
   - current truthfulness-anchor counts
2. Reconfirm the mandatory gate for later `*.c` / `*.h` days:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial example or implementation
   surface changes:
   - `make quality-review-full`
4. Refresh the targeted Sprint 58 follow-on binaries and docs/example surfaces
   most likely to be needed:
   - `./build/example_analysis`
   - `./build/example_iterative`
   - `./build/example_ic_minres`
   - `./build/example_eigs`
   - `./build/example_svd_lowrank`
   - `./build/bench_refactor`
   - `./build/bench_refactor_csc`
   - `./build/bench_iterative_reuse`
   - `./build/bench_eigs_reuse`
5. Record the authoritative validation boundary for docs-only days versus days
   that touch shipped code or headers.

### Deliverables
- Refreshed validation/truthfulness notes
- Sprint 58 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 58 uses the same baseline wording and parity anchors as the live repo
- The authoritative rerun set is explicit before public-surface edits begin
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Public Docs Audit

**Title:** Docs Audit  
**Theme:** Reduce the public documentation problem to concrete drift classes
before permanent wording changes land  
**Time estimate:** 10 hours

### Tasks
1. Audit the strongest caller-facing docs:
   - `README.md`
   - `docs/tutorial.md`
   - `examples/README.md`
   - `benchmarks/README.md`
2. Separate each surface into concrete drift classes:
   - stale sprint chronology
   - repeated-run workflow ambiguity
   - one-shot versus advanced-path imbalance
   - example coverage mismatch
   - benchmark taxonomy mismatch
3. Rank the cleanup targets by:
   - caller visibility
   - confusion risk
   - ease of truthful simplification
   - dependency on later header/example changes
4. Reject generic narrative expansion that would make the docs longer without
   making the workflows clearer.
5. Write the public-docs audit artifact and ranked landing order.

### Deliverables
- Public docs drift audit
- Ranked docs cleanup targets
- Proposed first documentation landing boundary

### Completion Criteria
- The documentation problem is reduced to named drift classes
- The first landing target is justified by caller value, not only file size
- Sprint 58 can start docs simplification from a concrete map

---

## Day 4: README/Tutorial Reduction Design

**Title:** Docs Design  
**Theme:** Freeze the first bounded README/tutorial simplification boundary
before editing the highest-signal public docs  
**Time estimate:** 10 hours

### Tasks
1. Select the Day 3 highest-value docs seam for the first landing.
2. Define the exact reduction boundary across:
   - stable workflow guidance
   - one-shot versus repeated-run positioning
   - direct versus iterative/eigensolver caller-story summaries
   - explicit exclusions or non-goals worth preserving
3. Define the invariants the docs reduction must preserve:
   - truthful workflow claims
   - alignment with validated example and benchmark behavior
   - stable top-level navigability
4. Define the cleanup policy for touched prose:
   - remove sprint-history narrative
   - keep product-level guidance
   - avoid expanding into tutorial-scale rewrites outside scope
5. Record the design artifact and landing checklist.

### Deliverables
- README/tutorial reduction design
- Wording boundary map
- Simplification invariants and checklist

### Completion Criteria
- The first docs reduction boundary is explicit before prose edits begin
- Ownership is defined by workflow sections, not vague cleanup intent
- Cleanup expectations are fixed before high-signal docs are rewritten

---

## Day 5: README & Tutorial Reduction Batch I

**Title:** Docs Batch I  
**Theme:** Land the first bounded top-level docs simplification patch  
**Time estimate:** 12 hours

### Tasks
1. Simplify the selected `README.md` and `docs/tutorial.md` sections around
   stable workflow guidance.
2. Remove stale sprint-history and future-sprint wording from the touched
   sections.
3. Tighten the one-shot versus repeated-run story so the default path remains
   clear while advanced reuse workflows stay explicit.
4. Reconcile touched wording against the current example and benchmark
   surfaces.
5. Run targeted docs sanity checks over the touched wording and anchors.

### Deliverables
- Landed README/tutorial reduction patch
- Reduced public sprint-history narrative
- Updated sanity-check record

### Completion Criteria
- The top-level docs are shorter and more product-level than before
- The touched README/tutorial sections stay aligned with the shipped behavior
- No contradiction is introduced across the highest-signal public docs

---

## Day 6: README & Tutorial Reduction Batch II

**Title:** Docs Batch II  
**Theme:** Finish the strongest remaining top-level docs drift without
expanding into a broad rewrite  
**Time estimate:** 10 hours

### Tasks
1. Re-audit the landed top-level docs after Day 5.
2. Simplify the next highest-value residual drift:
   - workflow summary tables or bullets
   - product-structure summaries
   - example and benchmark entry-point framing
3. Normalize terminology so the repo-wide public story prefers workflow
   categories over sprint-local phrasing.
4. Record any intentionally deferred docs density that should remain outside
   Sprint 58.
5. Run targeted sanity checks across the touched top-level docs surfaces.

### Deliverables
- Follow-through README/tutorial cleanup patch
- Normalized public wording
- Updated deferred-docs note

### Completion Criteria
- The remaining high-signal docs drift is smaller and more concrete after Day 6
- Top-level workflow language is more stable and less sprint-local
- Sprint 58 can move to header cleanup from a cleaner caller-facing baseline

---

## Day 7: Header Narrative Cleanup Audit & Design

**Title:** Header Audit  
**Theme:** Reduce the public-header cleanup problem to a bounded offender list
and exact wording goals before touching API-adjacent text  
**Time estimate:** 9 hours

### Tasks
1. Audit the strongest public-header narrative offenders under `include/`,
   especially:
   - `include/sparse_analysis.h`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
   - any direct-solver family headers with visible sprint-history drift
2. Separate the cleanup targets into:
   - stale sprint chronology
   - stale future-work wording
   - overlong lifecycle explanation
   - terminology mismatch with README/tutorial wording
3. Rank the offenders by caller visibility and cleanup risk.
4. Define the exact bounded Day 8 header set and the wording invariants:
   - preserve API semantics
   - preserve ownership truth
   - keep useful concise behavioral comments
5. Record the audit/design artifact and landing checklist.

### Deliverables
- Public-header narrative audit
- Ranked header cleanup targets
- Exact Day 8 touched-header set

### Completion Criteria
- The header cleanup problem is reduced to named offender classes
- The bounded header set is chosen before edits begin
- API-adjacent wording invariants are explicit before touching public headers

---

## Day 8: Header Narrative Cleanup Batch

**Title:** Header Batch  
**Theme:** Land the bounded public-header narrative cleanup patch  
**Time estimate:** 12 hours

### Tasks
1. Remove stale sprint-history and future-sprint narrative from the selected
   public headers.
2. Tighten overlong lifecycle/workspace wording where the shipped behavior is
   already stable and better explained elsewhere.
3. Keep concise comments that still carry real API-usage or ownership value.
4. Reconcile touched header wording against the current README/tutorial story.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - targeted `rg` sanity checks for removed stale phrases

### Deliverables
- Landed public-header cleanup patch
- Reduced stale narrative in the strongest public headers
- Updated validation record

### Completion Criteria
- The selected headers read more like stable product surfaces than sprint logs
- Public semantics are unchanged
- Required validation passes after the header cleanup

---

## Day 9: Example Modernization Audit & Design

**Title:** Example Audit  
**Theme:** Freeze the highest-value example modernization target before
touching shipped example code or docs  
**Time estimate:** 8 hours

### Tasks
1. Audit the highest-signal example surfaces:
   - `examples/example_analysis.c`
   - `examples/example_iterative.c`
   - `examples/example_ic_minres.c`
   - `examples/example_eigs.c`
   - `examples/README.md`
2. Separate likely work into:
   - better caller-story comments
   - clearer output wording
   - missing example README framing
   - bounded example additions if one small gap is real
3. Rank the example targets by caller value, truthfulness impact, and code-risk
   cost.
4. Define the exact Day 10 example boundary and the non-goals:
   - no broad tutorial rewrite
   - no example explosion
   - no behavioral redesign
5. Record the audit/design artifact and landing checklist.

### Deliverables
- Example modernization audit
- Ranked example targets
- Exact Day 10 touched-example boundary

### Completion Criteria
- The example problem is reduced to concrete caller-story gaps
- The Day 10 landing set is explicit before code/docs edits begin
- Non-goals are fixed before example work starts

---

## Day 10: Example Modernization Batch

**Title:** Example Batch  
**Theme:** Land the highest-value bounded example and example-doc alignment
patch  
**Time estimate:** 12 hours

### Tasks
1. Update the selected example source and/or `examples/README.md` surfaces to
   better teach the final stable workflow story.
2. Improve comments or printed output where the current example still reads too
   much like sprint-local framing.
3. Keep the one-shot-first posture where that is the intended example story,
   while making repeated-run/factor-many examples explicit where they already
   exist.
4. Reconcile touched example wording against the README/tutorial/header story.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - targeted example binaries touched by the batch

### Deliverables
- Landed example modernization patch
- Better aligned example docs/source comments
- Updated example sanity-check record

### Completion Criteria
- The touched examples better reflect the final product workflow story
- No new example behavior drift is introduced
- Required validation passes after the example batch

---

## Day 11: Benchmark Taxonomy Cleanup Batch

**Title:** Benchmark Batch  
**Theme:** Reorganize the benchmark story around stable workflow categories
rather than sprint-local terminology  
**Time estimate:** 11 hours

### Tasks
1. Rework `benchmarks/README.md` around clear workflow groupings:
   - one-shot compatibility/comparison
   - analyze-once / factor-many direct paths
   - iterative-handle reuse
   - eigensolver-handle reuse
2. Remove stale sprint-local naming and category drift from benchmark docs.
3. Ensure the benchmark docs match the current shipped drivers and their real
   proof roles.
4. Run targeted benchmark/docs sanity checks:
   - `./build/bench_refactor`
   - `./build/bench_refactor_csc`
   - `./build/bench_iterative_reuse`
   - `./build/bench_eigs_reuse`
   - `rg` checks over the updated wording
5. Record any intentionally deferred benchmark-doc density outside Sprint 58.

### Deliverables
- Landed benchmark taxonomy cleanup patch
- Workflow-based benchmark organization
- Updated benchmark sanity-check record

### Completion Criteria
- Benchmark docs are organized by stable workflow rather than sprint history
- Touched benchmark wording matches the live drivers and proof roles
- Remaining benchmark-doc drift is explicitly bounded

---

## Day 12: Post-Landing Compatibility Audit

**Title:** Compatibility Audit  
**Theme:** Confirm that the landed Sprint 58 public-surface cleanup preserved
the steady-state contract before final validation  
**Time estimate:** 8 hours

### Tasks
1. Re-audit the landed Sprint 58 surfaces:
   - README
   - tutorial
   - touched headers
   - examples docs/source comments
   - benchmark docs
2. Confirm the preserved fence still reads clearly:
   - one-shot APIs remain first-class
   - repeated-run paths remain bounded opt-in workflows
   - supported solver-family boundaries remain honest
3. Identify any residual contradiction or wording overreach that would block
   final validation.
4. Lock the Day 13 validation checklist from the landed state.
5. Record the audit artifact and any consciously deferred follow-ons.

### Deliverables
- Post-landing compatibility audit
- Final validation checklist
- Explicit residual/deferred wording queue

### Completion Criteria
- No blocker-level contract drift remains before Day 13
- The preserved public workflow fence is still explicit after the cleanup
- Final validation scope is fixed from the landed state

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Reconfirm the full reviewed baseline and targeted caller-facing
surfaces from the final Sprint 58 tree  
**Time estimate:** 11 hours

### Tasks
1. Run the full required baseline:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm reviewed CMake parity and test-count truthfulness anchors.
3. Rerun the targeted Sprint 58 public-surface follow-ons:
   - touched example binaries
   - touched benchmark binaries
   - any touched direct lifecycle example paths
4. Record representative stable outputs that support the final doc/example
   story.
5. Write the validation artifact and lock the closeout baseline.

### Deliverables
- Full validation record
- Maintained parity/truthfulness anchors
- Representative public-surface output notes

### Completion Criteria
- All required validation passes from the final Sprint 58 tree
- Reviewed parity remains exact
- The closeout baseline is fixed with no unresolved blocker-level drift

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Package Sprint 58 as one coherent documentation/examples/benchmark
simplification handoff  
**Time estimate:** 4 hours

### Tasks
1. Summarize the final Sprint 58 deliverables across:
   - public docs reduction
   - header cleanup
   - example modernization
   - benchmark taxonomy cleanup
2. Record the preserved compatibility fence and the final validated baseline.
3. Capture the explicit deferred queue, if any, for later product-surface
   cleanup.
4. Check whether `docs/planning/EPIC_5/PROJECT_PLAN.md` needs a correction or
   follow-on note.
5. Write the final closeout/handoff artifact and working-notes summary.

### Deliverables
- Sprint 58 closeout artifact
- Final handoff summary
- Explicit deferred follow-on list

### Completion Criteria
- Sprint 58 closes as one coherent simplified public-surface package
- The maintained validation baseline is explicitly carried forward
- Any remaining cleanup is framed as future work rather than hidden drift
