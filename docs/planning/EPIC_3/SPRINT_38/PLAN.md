# Sprint 38 Plan: Coverage, Regression-Proofing & Quality-Gate Expansion

**Sprint Duration:** 14 days  
**Goal:** Expand the cleaned-up warning/dead-code baseline into broader regression-proofing so coverage, compile-only surfaces, dead-code checks, and quality-readiness reporting become routine protection rather than periodic manual cleanup. This sprint implements the Sprint 38 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`.

**Starting Point:** Sprint 37 closed with a validated maintainability baseline, clearer helper ownership, a cleaner quality-target layout, and the Sprint 34 through Sprint 36 reviewed-quality contracts still intact. Sprint 38 starts from that stable base and focuses on the next layer of regression protection: honest coverage expectations, broader compile-only protection for non-routinely-run surfaces, better dead-code evidence classification, clearer release/readiness checks, and CI/reporting output that makes failures easier to diagnose.

**End State:** Sprint 38 leaves behind a more durable quality baseline: coverage language matches actual active/opt-in behavior, compile-only regression protection closes or re-documents the lingering exclusion surfaces, dead-code reporting is closer to actionable routine use, CI/reporting output is easier to interpret, and the repo has a concise readiness checklist that ties the major quality contracts together.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 136 hours, matching the Sprint 38 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 38 Scope Audit & Baseline

**Title:** Regression Baseline  
**Theme:** Convert the Sprint 38 project-plan items into a concrete regression-proofing scope  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 38 section of `docs/planning/EPIC_3/PROJECT_PLAN.md` plus the Sprint 34, Sprint 36, and Sprint 37 handoff/retrospective docs so the sprint stays anchored to the documented prerequisites.
2. Confirm the validated baseline that must remain true through Sprint 38: reviewed Makefile wrappers pass, reviewed CMake parity remains auditable, dead-code staged/excluded expectations remain explicit, and the Sprint 34 compile-db exclusion list is still treated as open work rather than silently closed.
3. Inventory the regression-proofing surfaces named by the plan: coverage honesty, compile-only regression protection, dead-code workflow maturation, release/readiness checks, and CI/reporting polish.
4. Record the current command/report surfaces, open limitations, and likely implementation batches before any edits begin.
5. Open Sprint 38 working notes and capture the baseline assumptions, current gate boundaries, and initial risk map.

### Deliverables
- Sprint 38 regression baseline
- Initial coverage/gate/reporting inventory
- Named first-pass audit targets for coverage, dead-code, compile-only surfaces, and readiness docs

### Completion Criteria
- Sprint 38 starts from a documented Sprint 37 validated baseline
- The regression-proofing scope is separated from resolved warning/dead-code cleanup
- The main Sprint 38 work surfaces are identified before implementation begins

---

## Day 2: Coverage-Honesty Audit

**Title:** Coverage Audit  
**Theme:** Reconcile coverage language and expectations with the actual active/opt-in test contract  
**Time estimate:** 8 hours

### Tasks
1. Audit coverage-related docs, target names, summaries, and artifact wording against the actual test categories now present in the repo.
2. Cross-check Sprint 32 truthfulness work so active, slow, experimental, and intentionally-skipped tests are represented accurately instead of implicitly treated as one flat coverage surface.
3. Identify any mismatch between what the repo says is “covered” and what is actually run by default versus by opt-in environment toggles.
4. Distinguish wording problems from real instrumentation/gating gaps.
5. Write the audit note that defines the coverage-honesty cleanup queue.

### Deliverables
- Coverage-honesty audit note
- Ranked list of wording, reporting, and expectation mismatches
- Initial keep/fix/defer classification for coverage-related surfaces

### Completion Criteria
- Coverage expectations are mapped before edits begin
- Truthfulness issues are separated from actual runtime/test gaps
- The later coverage cleanup batches are bounded clearly enough to implement safely

---

## Day 3: Compile-Only Regression Surface Audit

**Title:** Compile Surface Audit  
**Theme:** Map the non-routinely-run binaries that still need meaningful compile-only protection  
**Time estimate:** 10 hours

### Tasks
1. Audit benchmark/example binaries that are compile-checked but not routinely executed, with special attention to the Sprint 34 dead-code compile-db exclusion list.
2. Confirm the exact current status of `bench_svd`, `example_basic_solve`, `example_condition`, `example_iterative`, `example_least_squares`, `example_matrix_free`, and `example_svd_lowrank`.
3. Distinguish “missing from compile-db/reporting” from “already protected elsewhere but under-documented.”
4. Record which surfaces should be closed by implementation versus re-documented honestly as staged/deferred.
5. Write the audit note that defines the compile-only regression protection batch.

### Deliverables
- Compile-only regression surface audit note
- Explicit status map for the Sprint 34 exclusion list
- Initial close/re-document/defer classification for compile-only protection work

### Completion Criteria
- The lingering compile-only protection gaps are explicit before implementation
- Named excluded binaries are mapped to real current behavior
- The implementation batch is bounded to concrete surfaces rather than generic “more compile coverage”

---

## Day 4: Dead-Code Workflow Maturation Audit

**Title:** Dead-Code Audit  
**Theme:** Reassess the advisory dead-code workflow before making it more actionable  
**Time estimate:** 10 hours

### Tasks
1. Audit the current dead-code workflow artifacts, report buckets, staged/excluded surfaces, and residual `cppcheck` evidence buckets inherited from Sprint 33.
2. Separate “actionable enough to gate harder” from “still too noisy or coverage-limited.”
3. Re-check the shared-path serialization limitation so later stronger enforcement does not overclaim concurrency safety.
4. Identify the highest-value refinements: residual evidence classification, report wording, or workflow topology changes.
5. Write the audit note that defines the dead-code maturation queue.

### Deliverables
- Dead-code workflow maturation audit note
- Ranked residual evidence/noise queue
- Initial actionable/staged/defer classification for stronger dead-code enforcement

### Completion Criteria
- The dead-code maturation queue is explicit before edits begin
- Residual advisory/noise limitations are separated from true implementation gaps
- Later dead-code work is grounded in current evidence, not assumed readiness

---

## Day 5: Coverage-Honesty Design & Narrow Batch

**Title:** Coverage Batch I  
**Theme:** Convert the coverage audit into the safest first truthfulness cleanup slice  
**Time estimate:** 10 hours

### Tasks
1. Choose the highest-value coverage-honesty batch from the Day 2 audit, limited to wording/reporting fixes with low semantic risk.
2. Implement the cleanup in the most authoritative surfaces so coverage expectations are taught consistently.
3. Preserve the Sprint 32 opt-in test contract while simplifying any overstated coverage wording.
4. Validate the touched docs/report surfaces directly.
5. Record the residual coverage-honesty queue that remains for a later batch.

### Deliverables
- First coverage-honesty cleanup batch
- Residual coverage-truthfulness queue
- Updated notes describing the now-authoritative coverage wording

### Completion Criteria
- The highest-value coverage wording drift is reduced
- Coverage truthfulness improves without inventing fake new coverage
- Remaining coverage-honesty work is narrowed explicitly

---

## Day 6: Compile-Only Regression Protection Batch

**Title:** Compile Batch I  
**Theme:** Close or honestly re-document the highest-value compile-only protection gaps  
**Time estimate:** 10 hours

### Tasks
1. Choose the first compile-only regression batch from the Day 3 audit, focused on the clearest exclusion-list follow-through.
2. Implement the strongest safe improvement to compile-only protection, reporting, or documentation for the named benchmark/example surfaces.
3. Keep the batch narrow enough that its effect on maintained local/CI quality paths remains easy to reason about.
4. Validate the touched compile-only surfaces directly.
5. Record the residual exclusion-list queue that remains for later cleanup or deliberate deferral.

### Deliverables
- First compile-only regression protection batch
- Updated status of the Sprint 34 exclusion list
- Residual compile-only queue

### Completion Criteria
- At least the highest-value named compile-only gap is closed or re-documented honestly
- Compile-only protection becomes more explicit and less assumption-driven
- Remaining compile-only debt is smaller and clearly named

---

## Day 7: Dead-Code Workflow Maturation Design

**Title:** Dead-Code Design  
**Theme:** Define the safest next-stage shape for dead-code reporting and enforcement  
**Time estimate:** 10 hours

### Tasks
1. Turn the Day 4 audit into a concrete design for the next dead-code maturity step.
2. Decide which refinements belong in Sprint 38: evidence reclassification, report structure changes, stronger check behavior, or shared-path isolation preparation.
3. Keep the Sprint 34 and Sprint 36 limitations explicit where the workflow is still staged or serialized.
4. Define the smallest useful dead-code implementation batch that improves routine signal quality without overstating readiness.
5. Write the design note that defines the dead-code maturity implementation batch.

### Deliverables
- Dead-code maturation design note
- Defined implementation batch for report/check refinement
- Explicit residual limitations that remain staged after the batch

### Completion Criteria
- The dead-code implementation contract is chosen before edits begin
- Stronger signaling is separated from overclaiming concurrency or coverage readiness
- The dead-code batch is bounded enough to implement deliberately

---

## Day 8: Dead-Code Workflow Maturation Implementation

**Title:** Dead-Code Batch I  
**Theme:** Improve the report/check signal without breaking the staged contract  
**Time estimate:** 10 hours

### Tasks
1. Implement the first dead-code maturation batch from the Day 7 design.
2. Refine report/check behavior, evidence classification, or artifact output as chosen, keeping staged/excluded surfaces explicit.
3. Revalidate the authoritative serial dead-code path directly.
4. Record the residual dead-code queue for later maturation work.
5. Update notes with the new actionable-vs-staged boundary after the batch.

### Deliverables
- First dead-code maturation batch
- Residual dead-code maturation queue
- Updated notes describing the new report/check boundary

### Completion Criteria
- Dead-code output is closer to routine enforcement while staying truthful
- The authoritative serial dead-code path still passes
- Remaining dead-code maturity work is narrower and explicitly staged

---

## Day 9: Quality-Gate Expansion Audit & Design

**Title:** Gate Expansion Design  
**Theme:** Decide the safest next tier of reviewed targets/toolchains to harden  
**Time estimate:** 10 hours

### Tasks
1. Audit the current reviewed gate boundaries again in light of the coverage and compile-only findings from earlier sprint days.
2. Decide where Sprint 38 should expand the quality gates next: local wrappers, CI entry points, report checks, or broader compile-only audited surfaces.
3. Preserve the Sprint 36 enforced/staged/excluded platform contract instead of collapsing it into false uniformity.
4. Choose the smallest meaningful gate-expansion batch that improves regression protection without destabilizing the current baseline.
5. Write the design note that defines the quality-gate expansion batch.

### Deliverables
- Quality-gate expansion design note
- Defined next-tier gate-expansion batch
- Explicit residual staged/excluded surfaces after the planned expansion

### Completion Criteria
- Gate expansion is driven by audited readiness, not by aspirational symmetry
- The implementation batch is concrete and reviewable before edits begin
- The current enforced/staged platform contract remains explicit

---

## Day 10: Quality-Gate Expansion Implementation

**Title:** Gate Batch I  
**Theme:** Implement the chosen next-tier reviewed gate/report improvement  
**Time estimate:** 10 hours

### Tasks
1. Implement the chosen gate-expansion batch from the Day 9 design.
2. Keep the change narrow enough that the resulting local/CI behavior remains attributable and easy to validate.
3. Revalidate the touched reviewed-quality path directly.
4. Record any residual gate-expansion work that still belongs to later sprints.
5. Update notes with the new enforced/staged boundary after the batch.

### Deliverables
- First quality-gate expansion batch
- Residual gate-expansion queue
- Updated notes describing the new reviewed-gate boundary

### Completion Criteria
- Regression protection expands in one concrete, validated area
- The touched gate/report path remains understandable and attributable
- Remaining expansion work is smaller and explicitly staged

---

## Day 11: Release/Readiness Checklist Design

**Title:** Readiness Design  
**Theme:** Define a concise quality-readiness checklist that matches the cleaned-up repo reality  
**Time estimate:** 8 hours

### Tasks
1. Audit the current quality/readiness expectations spread across README, CI docs, handoff notes, and prior sprint artifacts.
2. Decide the canonical checklist scope: warnings, dead code, test truthfulness, docs/examples consistency, compile-only protection, and cross-platform parity.
3. Separate the authoritative checklist from supporting detail docs so it stays concise enough to use.
4. Choose the landing surface and structure for the checklist.
5. Write the design note that defines the readiness-checklist implementation batch.

### Deliverables
- Release/readiness checklist design note
- Defined checklist scope and authoritative landing surface
- Initial keep/link/defer classification for surrounding supporting docs

### Completion Criteria
- The readiness checklist contract is chosen before edits begin
- The checklist is scoped to real maintained quality signals
- Later checklist implementation is bounded and easy to review

---

## Day 12: Release/Readiness Checklist & Reporting Polish

**Title:** Readiness Batch I  
**Theme:** Ship the concise checklist and the highest-value reporting/output clarifications  
**Time estimate:** 10 hours

### Tasks
1. Implement the release/readiness checklist from the Day 11 design.
2. Apply the highest-value CI/reporting polish that improves failure readability around the touched quality paths.
3. Keep the changes compact and operator-facing rather than broad documentation churn.
4. Validate the touched docs/report surfaces directly.
5. Record any residual reporting/readiness polish that remains for later work.

### Deliverables
- First release/readiness checklist
- First CI/reporting polish batch
- Residual readiness/reporting queue

### Completion Criteria
- The repo has a concise quality-readiness checklist grounded in maintained signals
- Failure/report output is clearer in at least one high-value area
- Remaining reporting/readiness work is narrower and explicitly recorded

---

## Day 13: Full Validation Sweep

**Title:** Full Validation  
**Theme:** Re-run the full practical quality/test matrix and record the new baseline  
**Time estimate:** 12 hours

### Tasks
1. Run the full maintained local quality path practical for the sprint.
2. Re-run the reviewed wrappers, reviewed CMake parity path, and authoritative dead-code path as appropriate to the touched surfaces.
3. Confirm coverage/readiness wording still matches the actual validated behavior after the sprint’s gate/report changes.
4. Record durations, counts, and any reconciliations required.
5. Write the validation artifact that becomes the authoritative Sprint 38 close baseline.

### Deliverables
- Full validation sweep artifact
- Updated measured baseline for Sprint 38 close
- Named reconciliation notes if anything required clarification during the sweep

### Completion Criteria
- The maintained quality/test matrix practical for the sprint passes
- The new Sprint 38 baseline is recorded with concrete measured outputs
- No unresolved validation ambiguity remains going into closeout

---

## Day 14: Closeout, Handoff & Future Routing

**Title:** Closeout & Handoff  
**Theme:** Consolidate outcomes, route residual work forward, and make the Sprint 38 end state explicit  
**Time estimate:** 10 hours

### Tasks
1. Write Sprint 38 handoff and retrospective docs grounded in the Day 13 validated baseline.
2. Summarize what was improved in coverage honesty, compile-only protection, dead-code maturity, gate expansion, and readiness reporting.
3. Route any residual staged/deferred work into the correct later sprint sections of `docs/planning/EPIC_3/PROJECT_PLAN.md` if needed.
4. Record the final validated close state and the remaining bounded queue for future sprints.
5. Ensure the sprint notes clearly distinguish shipped protection from still-staged or excluded surfaces.

### Deliverables
- Sprint 38 `HANDOFF.md`
- Sprint 38 `RETROSPECTIVE.md`
- Any required `PROJECT_PLAN.md` routing updates for residual deferred work

### Completion Criteria
- Sprint 38 closes with an explicit validated baseline and a bounded residual queue
- Future sprint planning inherits any remaining work through the project plan rather than scattered sprint notes
- The closeout clearly separates shipped regression protection from still-staged follow-on work
