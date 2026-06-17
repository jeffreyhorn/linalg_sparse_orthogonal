# Sprint 76 Plan: Benchmark Governance, Profiling & Longitudinal Reporting

**Sprint Duration:** 14 days  
**Goal:** Strengthen the canonical benchmark surface so it supports better
longitudinal comparison, narrower thresholding where justified, and clearer
state-of-the-art performance claims. This sprint implements the Sprint 76
section of `docs/planning/EPIC_7/PROJECT_PLAN.md`.

**Starting Point:** Sprint 75 closed with a validated second backend-aware
package and a ranked Sprint 76 carry-forward queue:
- the strongest local reviewed baseline remains `make quality-review-full`
- Sprint 65 established the canonical benchmark surface without turning it
  into a noisy timing gate
- Sprint 70 fixed the Epic 7 performance target and truthfulness fence
- Sprint 75 made one backend-aware panel-solver seam directly measurable
  through `bench_chol_csc`
- the strongest remaining benchmark-governance pressure is now concentrated
  in:
  - canonical vs exploratory benchmark role clarity
  - cross-run and cross-branch report comparability
  - narrow threshold/comparison policy only where stability is defensible
  - workflow alignment across `bench-fast`, canonical reports, and docs
- the preserved Epic 7 non-goal fence is still in force:
  - no benchmark timing thresholds masquerading as portable pass/fail proof
  - no widened product/platform/backend claim detached from maintained evidence
  - no broad backend or packaging rewrite hidden inside reporting work
  - no fake “state of the art” claim unsupported by measured retained surfaces

The highest-value Sprint 76 work is therefore not a broad benchmarking
overhaul. It is a bounded governance and reporting phase on the maintained
canonical benchmark surfaces, with explicit role separation, stronger
machine-readable artifacts, narrow comparison policy where justified, and
workflow/docs alignment that keeps the benchmark story truthful.

**End State:** Sprint 76 leaves behind a smaller and more coherent benchmark
gap:
- a ranked, evidence-based benchmark role map
- one explicit bounded reporting/governance contract for the canonical surface
- one landed longitudinal-reporting batch on the maintained benchmark owners
- narrower, defensible comparison policy where justified
- aligned local/CI/docs reading of the maintained benchmark surfaces
- a validated Sprint 76 closeout package that lets Sprint 77 start from a
  stronger benchmark-governance baseline instead of a mixed reporting backlog

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
153 hours, staying within the day-cap limit while covering the Sprint 76
implementation scope.

---

## Day 1: Sprint 76 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 76 project-plan scope plus the Sprint 65, Sprint
70, and Sprint 75 handoff into a bounded benchmark-governance sprint  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 76 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Sprint 70 performance target, the Sprint 65 benchmark closeout, and
   the Sprint 75 closeout artifact.
2. Reconfirm the preserved Sprint 76 constraints:
   - no benchmark timing thresholds masquerading as portable proof
   - no widened backend/product/platform claims
   - no governance rewrite that breaks the maintained canonical surface
   - no backend/performance implementation reopening unless reporting truly
     forces it
3. Define the Sprint 76 workstreams explicitly:
   - governance re-audit
   - longitudinal report design
   - canonical reporting batch
   - threshold/comparison policy batch
   - workflow alignment
   - docs/maintainer updates
   - validation and closeout
4. Record the strongest likely Sprint 76 touch surfaces:
   - canonical benchmark drivers
   - reporting scripts and artifact schemas
   - benchmark docs and maintainer policy surfaces
   - workflow entry points and CI-facing command surfaces
5. Open Sprint 76 working notes and record the intended landing order,
   required artifacts, and validation expectations.

### Deliverables
- Sprint 76 scope inventory
- Benchmark-governance workstream map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 76 starts from the Sprint 65, Sprint 70, and Sprint 75 contract
  rather than reopening broad Epic 7 planning
- The implementation workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or landing work proceeds

---

## Day 2: Validation Baseline & Truth-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the implementation-day validation contract and the
highest-signal rerun surfaces Sprint 76 must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 76 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial benchmark/workflow/governance batches default to
     `make quality-review-full`
   - docs-only audit/design days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 76 is most likely to stress:
   - maintained canonical benchmarks
   - benchmark report-generation entry points
   - representative examples and proof-owner tests
   - install/package proof scripts
4. Refresh the targeted rerun set most likely to matter in Sprint 76.
5. Record the authoritative validation split in the working notes.

### Deliverables
- Refreshed validation notes
- Sprint 76 rerun checklist
- Preserved proof-surface checklist

### Completion Criteria
- Sprint 76 uses the same reviewed/truthfulness reading fixed in Sprint 70
- The code-day validation contract is explicit before reporting/governance
  work starts
- No rerun ambiguity remains around the likely touched benchmark, workflow,
  and docs seams

---

## Day 3: Governance Re-audit

**Title:** Governance Audit  
**Theme:** Re-rank canonical, regression-sensitive, and exploratory benchmark
roles by current user value, maintenance cost, and proof leverage  
**Time estimate:** 12 hours

### Tasks
1. Audit the current benchmark surface across:
   - canonical proof/report benchmarks
   - fast local/CI benchmark surfaces
   - exploratory or comparative benches
   - docs and maintainer interpretation surfaces
2. Classify the strongest remaining benchmark-governance burdens:
   - role confusion
   - schema/reporting gaps
   - weak comparison policy seams
   - workflow interpretation drift
3. Record where the benchmark surface is already coherent and where it still
   behaves like an artifact pile instead of one governed surface.
4. Rank the strongest contradiction centers by:
   - review value
   - maintenance cost
   - proof clarity
   - bounded Sprint 76 payoff
5. Write the governance re-audit artifact.

### Deliverables
- Governance re-audit
- Ranked contradiction list
- First benchmark-governance hotspot map

### Completion Criteria
- The broad Sprint 76 benchmark problem is reduced to a concrete seam ranking
- The highest-value governance contradictions are explicit
- Day 4 can proceed from a real current-state ranking

---

## Day 4: First Governance Boundary

**Title:** Boundary Freeze  
**Theme:** Refine the ranking and freeze the first reporting/governance fence
for the sprint  
**Time estimate:** 11 hours

### Tasks
1. Re-rank the Day 3 governance hotspots against:
   - reporting value
   - review clarity
   - compatibility risk
   - bounded cleanup payoff
2. Separate:
   - first-batch landing surfaces
   - support surfaces that move only if the first batch forces it
   - later or explicitly deferred benchmark work
3. Identify the strongest first Sprint 76 fence:
   - schema/reporting lane
   - threshold/comparison lane
   - workflow/docs lane
4. Record the strongest non-goals:
   - no broad backend or packaging rewrite
   - no universal timing-threshold policy
   - no widened state-of-the-art claim without retained measured proof
5. Fix the Day 4 modernization boundary in writing.

### Deliverables
- Refined governance ranking
- First landing boundary
- Deferred/support-surface map

### Completion Criteria
- The first Sprint 76 landing fence is explicit before design begins
- Lower-value or higher-risk governance work is clearly separated from the
  first lane
- Support surfaces are bounded rather than assumed

---

## Day 5: Longitudinal Report Design

**Title:** Reporting Design  
**Theme:** Define the bounded implementation contract for longitudinal
benchmark reporting before edits begin  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 65 benchmark contract and Sprint 75 measurable backend
   additions against the first-batch surfaces.
2. Decide the ownership split for the strongest reporting work:
   - schema/manifest owner
   - canonical benchmark field owner
   - artifact-generation owner
   - docs/policy interpretation owner
3. Define the guarantees the batch must preserve:
   - canonical reporting remains threshold-free by default
   - historical comparison data is additive rather than contract-breaking
   - retained fields stay reviewable and machine-readable
4. Fix the exact first-batch non-touch set:
   - unrelated solver implementation work
   - platform/install contract redesign
   - broad docs cleanup spill
   - backend claim widening
5. Record the Day 5 design artifact.

### Deliverables
- Longitudinal-report design
- First-batch non-touch set
- Preserved compatibility checklist

### Completion Criteria
- The first reporting batch is explicitly designed before code edits begin
- The landing is bounded to reporting/governance clarity, not generic
  refactoring
- Compatibility and non-goal fences are fixed in writing

---

## Day 6: Design Freeze & Proof Map

**Title:** Proof Map  
**Theme:** Finalize the exact code/proof ownership split for the first
reporting landing before edits begin  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Day 5 design against the likely proof-owner benchmarks and
   workflow/reporting surfaces.
2. Decide where the first proof should live:
   - benchmark CSV/report fields
   - artifact output files
   - workflow commands
3. Identify whether a benchmark-side regression, report-schema check, or docs
   proof is the primary Day 7 owner.
4. Fix the exact Day 7 touch set and support-only follow-through set.
5. Record the Day 6 proof map.

### Deliverables
- Design freeze
- Day 7 touch map
- Proof-owner map for the first reporting landing

### Completion Criteria
- The first landing has one explicit proof map before edits begin
- Benchmark-side vs docs-side ownership is fixed in writing
- Day 7 can execute from a bounded touch set rather than a generic shortlist

---

## Day 7: Canonical Reporting Batch

**Title:** Reporting Batch  
**Theme:** Land the first bounded longitudinal-reporting slice on the
maintained canonical benchmark surface  
**Time estimate:** 11 hours

### Tasks
1. Implement the first bounded reporting/schema batch on the Day 6 touch set.
2. Preserve current benchmark semantics while adding the new additive
   machine-readable reporting surface.
3. Keep canonical benchmark output reviewable both by humans and tooling.
4. Add the focused benchmark/reporting proof required by the landed batch.
5. Validate the touched code and record the landing artifact.

### Deliverables
- First longitudinal-reporting implementation batch
- Focused reporting proof
- Day 7 landing artifact

### Completion Criteria
- The canonical benchmark surface emits the new reporting artifact cleanly
- Existing benchmark meaning is preserved
- The touched reporting seam is proved in its natural owner

---

## Day 8: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-audit the benchmark-governance surface after the reporting
landing and rerank the strongest remaining seam  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 landing against the Day 3-6 governance design.
2. Confirm which contradiction the landing actually closed.
3. Re-rank the remaining governance seams:
   - threshold/comparison policy
   - workflow alignment
   - docs/maintainer interpretation
4. Fix the strongest Day 9 design center.
5. Record the rerank artifact.

### Deliverables
- Post-landing audit
- Updated governance ranking
- Exact Day 9 design center

### Completion Criteria
- The next highest-value seam is explicit after the first landing
- The sprint does not drift into repetitive reporting-only churn
- Day 9 starts from a fresh ranked surface, not the original backlog

---

## Day 9: Threshold / Comparison Policy Design

**Title:** Policy Design  
**Theme:** Define the narrow comparison or threshold policy that is genuinely
stable enough to support  
**Time estimate:** 12 hours

### Tasks
1. Re-read the reranked benchmark/reporting seams against retained benchmark
   variability and reviewer value.
2. Decide whether the next batch should land:
   - narrow comparison bands
   - historical diff summaries
   - explicit no-threshold policy wording with stronger report context
3. Fix the exact preserved fence:
   - no portable pass/fail timing claims
   - no unstable thresholding on noisy surfaces
   - no silent promotion of exploratory benches into canonical truth
4. Identify the exact Day 10 touch set and support-only surfaces.
5. Record the policy-design artifact.

### Deliverables
- Threshold/comparison policy design
- Day 10 touch set
- Preserved stability checklist

### Completion Criteria
- The next policy batch is explicitly bounded and defensible
- Unstable or misleading threshold stories are excluded in writing
- Day 10 can land from a precise policy contract

---

## Day 10: Threshold / Comparison Batch

**Title:** Policy Batch  
**Theme:** Land the bounded comparison or threshold follow-through justified by
the Day 9 contract  
**Time estimate:** 10 hours

### Tasks
1. Implement the bounded threshold/comparison policy batch on the Day 9 touch
   surfaces.
2. Preserve the threshold-free canonical reading where stronger thresholding
   is not justified.
3. Add only the minimum proof/reporting follow-through required by the batch.
4. Validate the touched code/docs/workflow surfaces.
5. Record the Day 10 landing artifact.

### Deliverables
- Threshold/comparison implementation batch
- Focused proof/reporting follow-through
- Day 10 landing artifact

### Completion Criteria
- The landed policy is narrower and more truthful than a broad threshold gate
- Canonical benchmark reading remains coherent
- The touched surfaces validate cleanly

---

## Day 11: Workflow Alignment Batch

**Title:** Workflow Alignment  
**Theme:** Reconcile `bench-fast`, canonical reports, and the new reporting or
comparison surfaces across maintained local/CI entry points  
**Time estimate:** 11 hours

### Tasks
1. Re-read the landed reporting/policy work against maintained workflow entry
   points.
2. Update command surfaces and scripts only where the landed contract truly
   requires it.
3. Keep local, CI, and artifact-generation interpretations aligned.
4. Add the minimum proof or sanity checks required for the touched workflow
   surfaces.
5. Record the Day 11 workflow artifact.

### Deliverables
- Workflow alignment batch
- Updated command/workflow contract
- Day 11 artifact

### Completion Criteria
- `bench-fast`, canonical reports, and retained reporting/policy surfaces no
  longer drift in meaning
- Workflow changes stay bounded to the landed governance contract
- The touched workflow surfaces validate cleanly

---

## Day 12: Docs / Maintainer Follow-Through

**Title:** Docs Alignment  
**Theme:** Close the remaining benchmark-governance interpretation gap across
README, benchmark docs, and maintainer policy  
**Time estimate:** 10 hours

### Tasks
1. Re-read the landed Sprint 76 package against:
   - `README.md`
   - `benchmarks/README.md`
   - `docs/maintainer_guide.md`
2. Fix only the wording required to align public, benchmark, and maintainer
   interpretations of the stronger governance model.
3. Reconfirm proof ownership and non-goals in writing.
4. Fix the exact Day 13 validation queue from the post-Day-12 state.
5. Record the Day 12 docs/proof artifact.

### Deliverables
- Docs/maintainer alignment batch
- Final proof-owner map
- Exact Day 13 validation queue

### Completion Criteria
- Public, benchmark, and maintainer surfaces agree on the landed benchmark
  governance model
- No redundant reporting or workflow churn is introduced
- Day 13 starts from one explicit validation queue

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Validate the full Sprint 76 branch from the strongest reviewed
baseline plus the exact touched follow-ons  
**Time estimate:** 12 hours

### Tasks
1. Run the standard code-day gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed baseline:
   - `make quality-review-full`
3. Recheck reviewed CMake parity anchors and total reviewed runtime.
4. Run the exact touched Sprint 76 follow-ons:
   - canonical benchmarks
   - report-generation surfaces
   - representative examples
   - install/package proof scripts
5. Record retained outputs and any non-blocking operational notes.

### Deliverables
- Full validation sweep
- Reviewed parity/runtime anchors
- Focused follow-on validation record

### Completion Criteria
- The full Sprint 76 branch validates from the strongest reviewed baseline
- The touched benchmark/reporting/workflow surfaces all pass their focused
  follow-ons
- Day 14 can close from a real validated state

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 76 from the validated baseline with one explicit
benchmark-governance handoff package and a ranked Sprint 77 queue  
**Time estimate:** 10 hours

### Tasks
1. Re-read the validated Sprint 76 package against the sprint goal and
   preserved fence.
2. Summarize the landed benchmark-governance package explicitly:
   - role map
   - longitudinal reporting
   - threshold/comparison policy
   - workflow alignment
   - docs/policy alignment
3. Fix the ranked carry-forward queue for Sprint 77 and beyond.
4. Recheck `docs/planning/EPIC_7/PROJECT_PLAN.md` for any Sprint 76
   correction need.
5. Record the closeout and handoff artifact in working notes and artifacts.

### Deliverables
- Sprint 76 closeout summary
- Ranked Sprint 77 carry-forward queue
- Day 14 handoff artifact

### Completion Criteria
- Sprint 76 closes from a validated benchmark-governance package
- The carry-forward queue is explicit and ranked
- The branch is ready for retrospective creation without missing closeout
  context
