# Sprint 90 Plan: Epic 9 Baseline, State-of-the-Art Target & Product Contract

**Sprint Duration:** 14 days  
**Goal:** Freeze the post-Epic-8 baseline, define the actual Epic 9
competitive target, and establish the product-model, comparison, and non-goal
contract before structural implementation work begins. This sprint implements
the Sprint 90 section of `docs/planning/EPIC_9/PROJECT_PLAN.md`.

**Starting Point:** Epic 8 closed from a validated baseline with:
- `make quality-review-full`
- reviewed CMake parity at `53`
- maintained install/export proof passing
- one bounded maintained external SPD comparison lane
- a materially cleaner front-door, install/export, and support-surface story

The strongest remaining work is no longer generic cleanup. It is baseline and
target freezing:
- reconfirming the strongest post-Epic-8 measured baseline
- deciding what “state of the art” actually means for this repository
- turning the live post-Epic-8 contradiction set into one ranked Epic 9
  execution order
- fixing the anti-sprawl contract before compressed-first, backend, runtime,
  capability, packaging, and coherence work begins
- producing one authoritative review + todo + project-plan package strong
  enough to drive Sprints 91-99

**End State:** Sprint 90 leaves behind:
- one fresh post-Epic-8 baseline artifact
- one explicit Epic 9 target-state and non-goal fence
- one ranked contradiction map for Epic 9
- one external comparison / measurement contract
- one Epic 9 review, todo, and project-plan package
- one Sprint 90 closeout and Sprint 91 handoff package

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, which stays within the practical 14-day cap while
preserving the Sprint 90 ordering and intent from the project-plan `~166` hour
scope.

---

## Day 1: Sprint 90 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 90 project-plan section and Epic 8 close state into
one bounded Epic 9 baseline package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 90 section of `docs/planning/EPIC_9/PROJECT_PLAN.md`,
   the Epic 8 retrospective, and the Sprint 89 closeout artifact.
2. Reconfirm the preserved starting assumptions:
   - Epic 8 is complete
   - the strongest local reviewed baseline remains
     `make quality-review-full`
   - the maintained install/export proof and bounded external SPD comparison
     lane remain the strongest current truth surfaces
3. Define the Sprint 90 workstreams explicitly:
   - baseline recheck
   - target-state freeze
   - product-model audit
   - comparison/measurement contract
   - non-goal and risk fence
   - review/todo/project-plan package
4. Record the strongest likely Sprint 90 touch surfaces:
   - `docs/planning/EPIC_9/PROJECT_PLAN.md`
   - `docs/planning/EPIC_8/EPIC_8_RETROSPECTIVE.md`
   - Epic 8 review/todo artifacts
   - support, build, workflow, and proof-owner surfaces likely to matter in
     the new review
5. Open Sprint 90 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 90 scope inventory
- baseline/setup working notes
- explicit workstream map

### Completion Criteria
- Sprint 90 starts from the validated Epic 8 end state
- the Epic 9 baseline problem is explicit before deeper validation and audit
  work begins
- the initial non-goal fence is visible before design or planning widens

---

## Day 2: Validation & Maintained Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, install/export, benchmark-reporting,
and workflow truth split before Epic 9 review work begins  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest implementation-day and substantial-batch gates:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm the maintained cross-surface proof owners that Epic 9 planning
   must treat as authoritative:
   - reviewed CMake parity
   - representative reviewed examples
   - `tests/test_install.sh`
   - `tests/test_cmake_install.sh`
   - `make bench-canonical-report`
3. Recheck the current ownership split so the Epic 9 review does not blur:
   - reviewed correctness owners
   - install/export proof owners
   - benchmark/reporting owners
   - support-surface and workflow owners
4. Fix the authoritative rerun list most likely to matter throughout Sprint
   90 and Epic 9.
5. Record the validation and maintained-surface split in working notes and a
   Day 2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- maintained surface ownership map
- authoritative rerun list

### Completion Criteria
- the strongest local validation contract is explicit before review findings
  are written
- proof ownership across reviewed tests, install/export checks, examples,
  reports, and workflows is fixed in writing
- later review and planning days have no ambiguity about the required truth
  surfaces

---

## Day 3: Product, Performance & Capability Gap Audit

**Title:** Core Gap Audit  
**Theme:** Reduce the live post-Epic-8 tree to one ranked contradiction map
across product model, backend ceiling, and capability breadth  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the live tree against the strongest structural Epic 9 concern
   classes:
   - linked-list-first public/product model
   - dense/backend maturity ceiling
   - capability breadth and ABI/index maturity
2. Capture where the current post-Epic-8 tree still falls short of a more
   competitive sparse numerical product.
3. Separate:
   - contradictions that should drive Epic 9 implementation
   - contradictions that remain bounded non-claims
   - contradictions already materially improved by Epic 8
4. Identify the highest-value source, header, and public-surface owners tied
   to those gaps.
5. Write the ranked Day 3 audit artifact.

### Deliverables
- ranked product/performance/capability contradiction artifact
- fix-now vs bounded-non-claim split
- strongest owner-surface map

### Completion Criteria
- the broad Epic 9 structural problem is reduced to one ranked live map
- the highest-value product, backend, and capability contradictions are
  explicit before target-state design begins
- lower-value spillover work is separated from the main Epic 9 lane

---

## Day 4: Maintainability, Coherence & Workflow Audit

**Title:** Surface Audit  
**Theme:** Map the remaining maintainability, documentation-coherence, and
build/workflow duplication costs  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the live tree against the remaining Epic 9 concern classes:
   - large mixed-role implementation hotspots
   - giant-test and proof-owner concentration
   - sprint-era chronology in permanent docs/code/tests
   - duplicated build/package/workflow topology
2. Rank the largest residual code and proof hotspots by future review and
   extension cost.
3. Rank the highest-friction public/support surfaces by user and contributor
   cost.
4. Capture where build/package/workflow duplication still carries long-term
   maintenance drag.
5. Write the Day 4 audit artifact and fold the results into working notes.

### Deliverables
- ranked maintainability/coherence/duplication artifact
- hotspot map
- public-surface friction map

### Completion Criteria
- the strongest maintainability and coherence costs are explicit before
  competitive-target design
- large source/proof owners and duplicated workflow surfaces are named and
  ranked
- Sprint 90 now has one full contradiction inventory across all main Epic 9
  categories

---

## Day 5: Competitive Target & Product Contract Design

**Title:** Target Design  
**Theme:** Define what Epic 9 is actually trying to make the library become  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 and Day 4 audit results against the Epic 9 project-plan
   intent.
2. Decide the target-state reading Epic 9 is trying to earn:
   - high-end research sparse toolkit
   - production-grade sparse numerical library
   - bounded state-of-the-art sparse linear algebra library
3. Decide which public/product claims Epic 9 should and should not try to
   earn.
4. Fix the most important target-state success markers:
   - product-model convergence
   - backend maturity
   - capability breadth
   - runtime and proof maturity
   - package/platform truthfulness
5. Write the Day 5 target-state design artifact.

### Deliverables
- explicit Epic 9 target-state artifact
- success-marker list
- first-pass claim fence

### Completion Criteria
- “state of the art” has a concrete, bounded meaning for this repo
- the intended product and proof outcomes are explicit before the non-goal
  fence and comparison contract are frozen
- later review and plan documents can reference one stable target state

---

## Day 6: Comparison & Measurement Contract Design

**Title:** Comparison Design  
**Theme:** Freeze the external correctness, runtime, and package-shape
comparison model for Epic 9  
**Time estimate:** 12 hours

### Tasks
1. Re-rank which external and internal evidence surfaces should remain or
   become authoritative:
   - maintained SPD external correctness lane
   - install/export proof
   - bounded runtime-reference slices
   - canonical benchmark reporting
2. Decide which comparison classes Epic 9 should try to widen:
   - correctness-only
   - runtime/fill/reference only
   - advisory but not maintained
3. Fix which comparisons are strong enough for product claims and which remain
   bounded support evidence only.
4. Decide how runtime evidence should stay bounded to avoid fake timing-gate
   inflation.
5. Write the Day 6 comparison/measurement contract artifact.

### Deliverables
- external comparison contract
- runtime/benchmark claim fence
- package-shape and workflow comparison contract

### Completion Criteria
- Epic 9 has one explicit comparison and measurement contract
- correctness, runtime, and package-shape evidence classes are clearly
  separated
- later sprint planning can widen evidence intentionally instead of
  opportunistically

---

## Day 7: Non-goal & Risk Fence Freeze

**Title:** Risk Fence  
**Theme:** Fix the anti-sprawl contract before the final review and plan
package is written  
**Time estimate:** 12 hours

### Tasks
1. Re-read the target-state and comparison design artifacts.
2. Decide which tempting but invalid widenings Epic 9 must reject:
   - fake full compressed-first rewrite claims
   - fake platform symmetry
   - fake broad complex/mixed-precision claims
   - fake benchmark supremacy claims
   - generic rewrite-without-proof proposals
3. Record the strongest execution risks:
   - branch-wide sprawl
   - build/package drift
   - proof-owner overload
   - runtime/benchmark overclaim
4. Decide which residual areas should remain explicit carry-forward work even
   after Epic 9.
5. Write the Day 7 non-goal and risk-fence artifact.

### Deliverables
- explicit non-goal fence
- execution-risk map
- carry-forward boundary list

### Completion Criteria
- the anti-sprawl contract is explicit before the review verdict is finalized
- the highest execution risks are documented
- Epic 9 now has one stable “will do / will not do” boundary

---

## Day 8: Review Structure & Evidence Freeze

**Title:** Review Freeze  
**Theme:** Turn the audit, target, comparison, and risk outputs into one final
review structure  
**Time estimate:** 12 hours

### Tasks
1. Re-read the prior Epic review format and the current Epic 9 audits.
2. Decide the exact review structure:
   - scope
   - baseline
   - executive verdict
   - strengths
   - repository snapshot
   - ranked findings
   - category assessment
   - bottom-line gap summary
3. Freeze the evidence that each major finding must cite.
4. Ensure the review distinguishes:
   - actual engineering debt
   - bounded non-claims
   - Epic 8 improvements that already moved the repo
5. Record the Day 8 evidence freeze in working notes and an artifact.

### Deliverables
- review outline
- evidence map for major findings
- review-writing checklist

### Completion Criteria
- the final review can be written from a frozen evidence package
- the review scope is complete without drifting into generic sparse-library
  commentary
- the strongest findings are supported by concrete repo signals

---

## Day 9: Draft the Epic 9 Review

**Title:** Review Draft  
**Theme:** Write the full repo-wide Epic 9 review from the frozen evidence
package  
**Time estimate:** 12 hours

### Tasks
1. Draft the repo-wide review in
   `docs/planning/EPIC_9/reviews/review-codex-YYYY-MM-DD.md`.
2. Cover all requested evaluation dimensions:
   - efficiency
   - maintainability
   - usability
   - documentation
   - coherence
   - test coverage
   - state-of-the-art readiness
3. Make the findings concrete and detailed rather than generic.
4. Ensure the review records both:
   - the strongest post-Epic-8 strengths
   - the ranked gaps Epic 9 must close
5. Re-read the draft for coherence and internal consistency.

### Deliverables
- complete Epic 9 review draft
- ranked finding set
- category assessment table

### Completion Criteria
- the review is detailed enough to drive planning without extra interpretation
- all major findings are evidence-backed
- the review clearly separates strengths, limits, and action-worthy gaps

---

## Day 10: Draft the Gap-Closure Todo

**Title:** Todo Draft  
**Theme:** Convert the review findings into one ordered closure sequence  
**Time estimate:** 12 hours

### Tasks
1. Re-read the final review draft and extract the ranked gap set.
2. Group the gaps into a staged closure sequence.
3. Decide the ordering logic:
   - target and claim fence first
   - structural ceilings before broader widening
   - coherence and duplication after major architecture moves
   - competitive calibration and final closeout later
4. Draft the todo document in
   `docs/planning/EPIC_9/reviews/todo-codex-YYYY-MM-DD.md`.
5. Re-read the todo against the review to ensure no major gap is omitted or
   over-expanded.

### Deliverables
- complete Epic 9 gap-closure todo draft
- staged closure sequence
- done-when criteria for each stage

### Completion Criteria
- the todo document is a concrete execution sequence, not just a summary
- every major review finding maps to a remediation stage
- the sequence is narrow and ordered enough to drive the sprint plan

---

## Day 11: Draft the 10-Sprint Epic 9 Project Plan

**Title:** Plan Draft  
**Theme:** Turn the review and todo into one 10-sprint Epic 9 execution plan  
**Time estimate:** 12 hours

### Tasks
1. Re-read the current project-plan format used in Epics 3-8.
2. Map the todo stages onto Sprints 90-99.
3. Decide each sprint’s:
   - title
   - goal
   - prerequisites
   - item list
   - deliverables
4. Keep each sprint within the requested `168`-hour cap.
5. Draft `docs/planning/EPIC_9/PROJECT_PLAN.md`.

### Deliverables
- complete 10-sprint Epic 9 project-plan draft
- per-sprint goal and item mapping
- bounded estimated hours per sprint

### Completion Criteria
- the sprint plan fully covers the review/todo gap set
- each sprint remains bounded and ordered
- the format matches the repo’s existing epic-plan style

---

## Day 12: Cross-Document Alignment Pass

**Title:** Alignment Pass  
**Theme:** Make the review, todo, and project plan read as one coherent Epic 9
package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the review, todo, and project plan together.
2. Check for mismatches in:
   - ranked priorities
   - terminology
   - claim boundaries
   - sequencing
   - estimates
3. Tighten wording where one document overclaims or under-specifies compared
   with another.
4. Ensure the project-plan sprint order matches the todo stage order.
5. Record the alignment pass in working notes and a Day 12 artifact.

### Deliverables
- aligned review/todo/plan package
- cross-document terminology and sequencing fixups
- Day 12 alignment artifact

### Completion Criteria
- the three Epic 9 planning documents are internally consistent
- no major contradiction remains between findings, remediation, and sprint
  sequencing
- the package is ready for final validation and closeout

---

## Day 13: Final Validation & Readability Sweep

**Title:** Final Sweep  
**Theme:** Run the required final checks for the docs-only Sprint 90 package  
**Time estimate:** 12 hours

### Tasks
1. Recheck the strongest maintained baseline anchors relevant to the planning
   package:
   - reviewed CMake parity count
   - install/export proof-owner references
   - current project-plan references
2. Re-read the final docs for formatting and readability issues.
3. Confirm all file paths, dates, and sprint references are correct.
4. Confirm the sprint-plan hour caps and totals still satisfy the requested
   bounds.
5. Record the final validation sweep in working notes and an artifact.

### Deliverables
- final validation/readability artifact
- verified file/path/date/reference set
- frozen Sprint 90 planning package

### Completion Criteria
- the docs-only package is validated against the live repo state
- all references and estimates are internally correct
- the branch is ready for Sprint 90 closeout

---

## Day 14: Sprint 90 Closeout & Sprint 91 Handoff

**Title:** Closeout  
**Theme:** Close Sprint 90 from the validated planning package and leave one
explicit Sprint 91-first handoff queue  
**Time estimate:** 12 hours

### Tasks
1. Summarize what Sprint 90 established:
   - baseline
   - target state
   - contradiction map
   - comparison contract
   - non-goal fence
   - review/todo/project-plan package
2. Decide whether `docs/planning/EPIC_9/PROJECT_PLAN.md` needs any correction
   from the actual Sprint 90 result.
3. Write the Sprint 90 closeout and handoff artifact.
4. Fix the Sprint 91-first queue explicitly:
   - compressed-first product convergence first
   - backend maturity second
   - runtime, capability, coherence, packaging, and comparison work later in
     the already-ranked order
5. Record the final close state in working notes.

### Deliverables
- Sprint 90 closeout artifact
- explicit Sprint 91 handoff queue
- completed Sprint 90 working-notes close state

### Completion Criteria
- Sprint 90 closes from one explicit validated planning package
- the Sprint 91-first execution order is fixed in writing
- Epic 9 now starts from a clear, bounded, evidence-backed contract instead of
  a generic ambition statement
