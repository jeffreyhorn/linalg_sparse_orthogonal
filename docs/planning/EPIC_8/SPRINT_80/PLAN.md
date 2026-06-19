# Sprint 80 Plan: Epic 8 Baseline, Competitive Target & External Oracle Contract

**Sprint Duration:** 14 days  
**Goal:** Freeze the post-Epic-7 baseline, define the real Epic 8 competitive
target, and establish the external-oracle, performance, and non-goal contract
before implementation work begins. This sprint implements the Sprint 80
section of `docs/planning/EPIC_8/PROJECT_PLAN.md`.

**Starting Point:** Epic 7 closed from a validated baseline with a strong
reviewed test and package story, but with the major state-of-the-art gaps still
explicit:
- the public/core storage model remains linked-list-first
- the dense numeric backend remains builtin and scalar
- the capability surface remains bounded
- packaging and reviewed platform evidence remain intentionally asymmetric
- several large source and giant-test hotspots remain
- the project has strong internal proof but limited maintained external
  differential proof
- Epic 8 already has a top-level review, todo, and project-plan package, but
  Sprint 80 must turn that into a live baseline, a competitive target, and one
  explicit execution fence for the implementation sprints

The highest-value Sprint 80 work is therefore not “start coding improvements.”
It is to make the next epic executable from one fresh, truthful, measurable
starting point.

**End State:** Sprint 80 leaves behind:
- one refreshed post-Epic-7 baseline package
- one ranked live gap inventory grounded in the tree
- one explicit external-oracle and performance-comparison contract
- one fixed Epic 8 non-goal and risk fence
- one complete Sprint 80 artifact and handoff package that later sprints can
  execute without reopening the target state

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
166 hours, matching the Sprint 80 project-plan total while staying within the
requested day-cap limit.

---

## Day 1: Sprint 80 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Epic 8 project-plan section and Epic 7 closeout into one
bounded Sprint 80 execution package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 80 section of
   `docs/planning/EPIC_8/PROJECT_PLAN.md`, the Epic 7 retrospective, and the
   strongest Epic 7 closeout artifacts.
2. Reconfirm the preserved Epic 8 starting assumptions:
   - strong reviewed baseline already exists
   - major structural gaps are known but not yet frozen into a live sprint
     contract
   - no implementation work should start before the baseline and competitive
     target are fixed
3. Define the Sprint 80 workstreams explicitly:
   - baseline recheck
   - competitive gap inventory
   - external oracle contract
   - performance / benchmark contract
   - non-goal and risk fence
   - review-package and closeout documentation
4. Record the strongest likely Sprint 80 touch surfaces:
   - reviewed validation commands and outputs
   - maintainer policy and benchmark governance surfaces
   - package/install/export proof surfaces
   - likely external comparison candidates
5. Open Sprint 80 working notes and record the intended landing order,
   artifacts, and validation expectations.

### Deliverables
- Sprint 80 scope inventory
- Workstream map
- Starting working-notes baseline

### Completion Criteria
- Sprint 80 starts from the Epic 7 validated end state instead of from generic
  planning prose
- The workstreams are explicit before deeper baseline rechecks begin
- The sprint non-goal pressure is visible before design or landing work starts

---

## Day 2: Reviewed Validation & Package Baseline Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest local reviewed baseline, install proof, and
reviewed parity anchors  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline commands and ownership:
   - `make quality-review-full`
   - reviewed CMake parity
   - install/export proof scripts
2. Reconfirm the implementation-day validation split Sprint 80 and later Epic 8
   work must preserve.
3. Recheck the reviewed CMake surface, representative proof-owner tests,
   examples, and the maintained install/package proof surfaces.
4. Record the baseline commands, counts, and proof split in working notes and a
   Day 2 artifact.
5. Fix the authoritative rerun list most likely to matter throughout Epic 8.

### Deliverables
- Refreshed validation-baseline artifact
- Preserved reviewed / install proof map
- Authoritative rerun list

### Completion Criteria
- The strongest local reviewed baseline is freshly re-materialized
- Reviewed parity and install/export proof ownership are explicit
- Later Epic 8 code days have no ambiguity about the required validation gate

---

## Day 3: Live Competitive Gap Inventory

**Title:** Gap Inventory  
**Theme:** Re-rank the live structural, capability, performance, usability, and
product gaps against the current tree  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the highest-signal implementation, giant-test, package, workflow,
   benchmark, and support surfaces.
2. Re-rank the major gap categories against the live tree:
   - storage/product-model ceiling
   - dense/backend ceiling
   - capability breadth ceiling
   - maintainability hotspots
   - package/platform asymmetry
   - usability/documentation overload
   - assurance/external-oracle gaps
3. Separate:
   - strongest first-epic implementation targets
   - second-tier targets
   - support-only contradictions
   - deliberate non-goals
4. Record which previously deferred Epic 7 seams remain real candidates for
   Epic 8 and which should stay deferred.
5. Write the ranked gap-inventory artifact.

### Deliverables
- Ranked live gap inventory
- First-tier vs deferred gap map
- Epic 7 carry-forward reconciliation notes

### Completion Criteria
- The broad Epic 8 problem is reduced to one ranked live contradiction map
- Carry-forward work is grounded in the current tree rather than inherited
  blindly from Epic 7
- The strongest likely execution order is explicit before external-contract
  design begins

---

## Day 4: External Oracle Candidate Audit

**Title:** Oracle Audit  
**Theme:** Identify the strongest realistic external correctness and performance
reference classes for Epic 8  
**Time estimate:** 12 hours

### Tasks
1. Audit the live tree for existing external fixtures, corpus usage, and any
   current comparison assumptions.
2. Identify realistic external comparison candidates for bounded Epic 8 use:
   - SuiteSparse-family direct references where practical
   - BLAS/LAPACK-class dense references
   - advisory comparison classes for performance calibration
3. Classify each candidate by:
   - correctness oracle value
   - performance comparison value
   - packaging/tooling burden
   - portability risk
   - maintenance realism
4. Separate candidates suitable for maintained proof from candidates suitable
   only for local/advisory comparison.
5. Record the candidate audit and preliminary external-oracle ranking.

### Deliverables
- External-oracle candidate audit
- Maintained vs advisory comparison split
- Preliminary external-reference ranking

### Completion Criteria
- External comparison is framed as a bounded contract, not a vague aspiration
- Unrealistic or high-burden candidates are filtered before design
- Day 5 can choose from a ranked candidate set

---

## Day 5: External Oracle Contract Design

**Title:** Oracle Contract  
**Theme:** Define which external references Epic 8 will use and how they will
be interpreted  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 ranking against the strongest Epic 8 gap categories.
2. Decide which external references belong in the maintained Epic 8 contract.
3. Fix the interpretation split for each approved external lane:
   - correctness-only
   - performance-only
   - advisory comparison
   - non-shipping exploratory lane
4. Define the preserved non-goals:
   - no fake portability claims
   - no mandatory heavyweight external stack for normal builds
   - no broad benchmark gate disguised as reference comparison
5. Write the Day 5 contract artifact and working-notes design summary.

### Deliverables
- External-oracle contract artifact
- Approved reference-lane list
- Preserved external-comparison non-goal fence

### Completion Criteria
- The external-oracle surface is explicit, bounded, and credible
- Maintained versus advisory reference lanes are clearly separated
- Later implementation sprints can add comparison proof without reopening
  policy design

---

## Day 6: Performance & Benchmark Contract Recheck

**Title:** Benchmark Contract  
**Theme:** Freeze the performance-governance interpretation Epic 8 backend and
runtime work must preserve  
**Time estimate:** 12 hours

### Tasks
1. Re-read the canonical benchmark-governance surfaces:
   - `README.md`
   - `benchmarks/README.md`
   - `scripts/bench_canonical_report.sh`
   - `Makefile` benchmark/report targets
2. Reconfirm the maintained benchmark split:
   - canonical threshold-free reporting
   - bounded runtime signals
   - narrow thresholded regression gates
   - exploratory/context-only lanes
3. Decide which performance-comparison readings Epic 8 can strengthen without
   inflating claims.
4. Fix how backend and runtime comparison work will integrate with the
   canonical reporting model.
5. Record the benchmark/performance contract artifact.

### Deliverables
- Performance / benchmark contract artifact
- Canonical reporting interpretation notes
- Allowed vs disallowed comparison-claim matrix

### Completion Criteria
- Epic 8 backend and runtime work has one explicit performance-claim fence
- Canonical reporting remains threshold-free unless a later sprint justifies a
  narrower separate gate
- Benchmark proof and product claims are still cleanly separated

---

## Day 7: Non-goal & Risk Fence Freeze

**Title:** Risk Fence  
**Theme:** Fix the strongest Epic 8 non-goals and execution risks before the
rest of the epic starts spending budget  
**Time estimate:** 12 hours

### Tasks
1. Re-rank the main execution risks surfaced by Days 1-6:
   - over-broad architecture churn
   - fake state-of-the-art claims
   - external dependency sprawl
   - platform-claim inflation
   - benchmark-governance drift
2. Separate:
   - approved Epic 8 target claims
   - deferred claims
   - explicitly prohibited “marketing” interpretations
3. Fix the strongest Epic 8 non-goal fence in writing:
   - no generic rewrite of the whole library
   - no fake automatic platform parity
   - no shared-library or backend maturity claim without proof
   - no capability-genericity claim without landed support
4. Record the risk register and mitigation ordering.
5. Fold the fence back into Sprint 80 working notes and artifacts.

### Deliverables
- Epic 8 non-goal artifact
- Risk register
- Mitigation ordering notes

### Completion Criteria
- The biggest Epic 8 failure modes are explicit before implementation sprints
  begin
- Claims the repo will not make are fixed in writing
- Later sprint design work can build against one stable fence

---

## Day 8: Review Package Refinement

**Title:** Review Refinement  
**Theme:** Strengthen the top-level Epic 8 review so it reads from the refreshed
baseline and the newly fixed contract surfaces  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Epic 8 review against the Day 2-7 findings.
2. Tighten any review sections that need stronger grounding in:
   - the refreshed baseline
   - the external-oracle candidate set
   - the benchmark/performance contract
   - the explicit non-goal fence
3. Ensure the review findings are still ordered by severity and support the
   later sprint structure.
4. Update or annotate any review wording that now reads too broad or too weak.
5. Record the refined review-readiness artifact.

### Deliverables
- Refined review notes
- Review-readiness checklist
- Cross-reference map between review findings and Sprint 80 artifacts

### Completion Criteria
- The Epic 8 review remains the authoritative finding set for the epic
- The review is aligned with the refreshed baseline and contract decisions
- No major contradiction remains between review language and Sprint 80 findings

---

## Day 9: Todo / Closure Sequence Refinement

**Title:** Todo Refinement  
**Theme:** Tighten the Epic 8 gap-closure sequence so it matches the live
baseline and contract decisions  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Epic 8 todo document against the Day 3-7 findings.
2. Re-sequence any closure work that now needs to move earlier or later.
3. Tighten the distinction between:
   - must-do architectural work
   - high-value assurance work
   - maintainability cleanup
   - support-surface simplification
   - final comparison and closeout
4. Ensure the todo does not implicitly reopen disallowed non-goals.
5. Record the refined closure-sequence artifact.

### Deliverables
- Refined todo notes
- Closure-sequence cross-check
- Deferred/non-goal reconciliation notes

### Completion Criteria
- The todo still closes the major gaps in a credible order
- The closure sequence matches the refreshed Sprint 80 baseline
- No low-value or policy-contradicting work remains hidden in the todo

---

## Day 10: Project-Plan Contract Recheck

**Title:** Plan Recheck  
**Theme:** Reconcile the 10-sprint Epic 8 plan against the refreshed findings
and contracts before Sprint 80 closes  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Epic 8 project plan against all Sprint 80 findings.
2. Reconfirm the major sprint ordering:
   - baseline and contract
   - storage/workflow modernization
   - dense/backend modernization
   - capability expansion
   - assurance expansion
   - maintainability cleanup
   - reorder/runtime convergence
   - packaging/platform convergence
   - usability simplification
   - final integration and closeout
3. Tighten any sprint descriptions or dependency wording that now needs better
   alignment with the refreshed contract.
4. Reconfirm each sprint remains plausibly bounded and within the intended
   time envelope.
5. Record the project-plan reconciliation artifact.

### Deliverables
- Project-plan reconciliation notes
- Updated dependency cross-check
- Sprint-order validation artifact

### Completion Criteria
- The 10-sprint Epic 8 plan still reads as the correct execution sequence
- Sprint ordering and dependencies match Sprint 80 findings
- No sprint now depends on an unfixed or contradictory assumption

---

## Day 11: Support-Surface Truth Sweep

**Title:** Truth Sweep  
**Theme:** Check whether the main support and policy surfaces already read
truthfully against the refreshed Epic 8 baseline  
**Time estimate:** 12 hours

### Tasks
1. Re-read the highest-signal support surfaces against the Day 2-10 findings:
   - `README.md`
   - `INSTALL.md`
   - `benchmarks/README.md`
   - `docs/maintainer_guide.md`
   - workflow-level CI notes
2. Check whether any Sprint 80 docs-only correction is actually needed.
3. If no change is needed, record the explicit no-op result and why.
4. If a contradiction appears, bound it for Day 12 rather than widening into
   broad support-surface churn.
5. Write the support-surface truth-sweep artifact.

### Deliverables
- Support-surface truth-sweep artifact
- Explicit no-op or bounded follow-through decision
- Final pre-validation support-surface map

### Completion Criteria
- The main support surfaces have been checked against the refreshed baseline
- Any needed follow-through is bounded explicitly
- Sprint 80 does not drift into unnecessary documentation churn

---

## Day 12: Sprint 80 Proof Alignment & Final Validation Queue

**Title:** Proof Alignment  
**Theme:** Fix the exact Sprint 80 validation queue and align the proof-owner
map for the baseline/contract package  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the exact Sprint 80 proof-owner and command-owner surfaces:
   - reviewed baseline commands
   - install/export scripts
   - canonical benchmark/report commands
   - likely external comparison staging points
2. Decide whether any focused follow-through is required before full
   validation.
3. Fix the authoritative Day 13 validation queue in writing.
4. Reconfirm the command expectations and retained counts/anchors Sprint 80
   will use for closeout.
5. Record the final proof-alignment artifact.

### Deliverables
- Proof-owner map
- Final Sprint 80 validation queue
- Retained anchor list for closeout

### Completion Criteria
- The full validation queue is explicit before Day 13 begins
- Proof ownership for the Sprint 80 package is fixed in writing
- No validation ambiguity remains around the refreshed baseline package

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Reconfirm that Sprint 80 closes from a fully validated, freshly
measured baseline  
**Time estimate:** 11 hours

### Tasks
1. Run the required validation commands:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm reviewed CMake parity counts and full reviewed CTest results.
3. Re-run maintained install/export proof:
   - `bash tests/test_install.sh`
   - `bash tests/test_cmake_install.sh`
4. Re-run the canonical benchmark/report command needed to confirm reporting
   ownership still behaves as expected.
5. Record the validation artifact with retained anchors, any issues found, and
   any bounded fixes required before closeout.

### Deliverables
- Day 13 validation artifact
- Fresh reviewed parity anchors
- Install/export and reporting recheck notes

### Completion Criteria
- Sprint 80 has a fresh validated baseline, not just an inherited one
- Reviewed parity, install/export proof, and canonical report surfaces all
  recheck cleanly
- Any real issue discovered is resolved or explicitly escalated before Day 14

---

## Day 14: Sprint 80 Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 80 as the executable baseline and contract sprint for
the rest of Epic 8  
**Time estimate:** 11 hours

### Tasks
1. Summarize what Sprint 80 fixed and what it intentionally left unchanged:
   - baseline recheck
   - competitive gap inventory
   - external-oracle contract
   - performance / benchmark contract
   - non-goal and risk fence
   - refreshed review/todo/project-plan readiness
2. Record the preserved carry-forward queue for Sprint 81 and later work.
3. Confirm whether `docs/planning/EPIC_8/PROJECT_PLAN.md` needs any correction
   after the live Sprint 80 findings.
4. Write the Day 14 closeout/handoff artifact and finalize working notes.
5. Recheck branch cleanliness and final artifact completeness.

### Deliverables
- Sprint 80 closeout artifact
- Handoff queue for Sprint 81
- Final Sprint 80 working-notes package

### Completion Criteria
- Sprint 80 closes from a validated, measured, and explicitly bounded baseline
- The next Epic 8 sprints inherit one stable contract instead of reopening
  Sprint 80’s target state
- The sprint leaves behind a clean handoff and no hidden baseline ambiguity
