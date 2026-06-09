# Sprint 60 Plan: Epic 6 Baseline, Productization Audit & Architecture Contract

**Sprint Duration:** 14 days  
**Goal:** Freeze the post-Epic-5 baseline, define the real Epic 6
productization target, inventory the strongest remaining gaps, and lock the
architecture, validation, and platform fences before implementation work
begins. This sprint implements the Sprint 60 section of
`docs/planning/EPIC_6/PROJECT_PLAN.md`.

**Starting Point:** Epic 5 is closed, the post-Epic-5 branch is validated, and
Epic 6 already has a branch-level review, gap list, and 10-sprint project
plan:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- the current library is strong and coherent, but not yet state of the art as
  a packaged, productized sparse linear algebra library
- the main remaining gaps now center on productization, configuration,
  performance governance, architecture contracts, and assurance depth rather
  than basic feature absence

The next highest-value work is not immediate implementation churn. It is
freezing a truthful Epic 6 baseline and converting the broad Epic 6 review into
one precise, measured architecture and execution contract.

**End State:** Sprint 60 leaves behind one coherent Epic 6 baseline package:
- a validated post-Epic-5 starting point
- a concrete productization gap inventory grounded in the live repo
- an explicit state-of-the-art target and non-goal fence
- a mapped configuration/performance surface for later implementation sprints
- a frozen validation/platform contract for Epic 6 implementation work

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
148 hours, matching the Sprint 60 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 60 Scope Audit & Epic 6 Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Epic 6 review, project plan, and Epic 5 closeout state
into a bounded Sprint 60 working map  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Sprint 60 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Epic 6 review, the Epic 6 todo
   list, and the Epic 5 retrospective.
2. Reconfirm the preserved Sprint 60 constraints:
   - no fake state-of-the-art claims
   - no premature implementation work before the baseline is frozen
   - no reopening Epic 5 solved scope
   - no architecture redesign without a written contract
3. Define the Sprint 60 workstreams explicitly:
   - baseline recheck
   - productization gap inventory
   - target/non-goal definition
   - configuration/performance surface audit
   - validation/platform contract freeze
   - sprint closeout package
4. Record the strongest likely Sprint 60 surfaces:
   - public headers and top-level docs
   - build/quality wrappers and CI
   - benchmark/example entry points
   - major implementation hotspots and internal configuration seams
5. Open Sprint 60 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 60 scope inventory
- Epic 6 baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 60 starts from the merged Epic 5 close state rather than reopening old
  design debates
- The baseline, audit, contract, and closeout workstreams are explicit before
  deeper investigation begins
- The sprint non-goal fence is fixed before any edits land

---

## Day 2: Validation Baseline & Truthfulness Anchor Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the maintained reviewed baseline and exact rerun set that
Epic 6 implementation work must preserve  
**Time estimate:** 9 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity counts
   - current quality/truthfulness wording
2. Reconfirm the mandatory gate for later `*.c` / `*.h` days:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial implementation work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Epic 6:
   - direct lifecycle proofs
   - iterative/eigensolver proofs
   - representative examples
   - representative benchmark drivers
   - parity/count anchors
5. Record the authoritative validation boundary for:
   - docs-only days
   - code-touching days
   - substantial architecture/performance days

### Deliverables
- Refreshed validation/truthfulness notes
- Epic 6 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 60 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before architecture or product audits
  deepen
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Productization Gap Inventory I

**Title:** Productization Audit I  
**Theme:** Reduce the broad Epic 6 review into concrete user-facing gap classes
grounded in the current codebase  
**Time estimate:** 10 hours

### Tasks
1. Audit the strongest usability and product-surface gaps across:
   - public API ergonomics
   - examples and onboarding flows
   - configuration discoverability
   - benchmark/story clarity
2. Separate gaps into concrete classes:
   - usability friction
   - configuration opacity
   - packaging/platform asymmetry
   - documentation overload or ambiguity
   - advanced-user control gaps
3. Rank the findings by:
   - user-facing pain
   - architectural leverage
   - risk of misleading product claims
   - implementation cost
4. Identify which gaps are true Epic 6 product work versus future
   research/performance stretch goals.
5. Write the first productization-inventory artifact with named findings.

### Deliverables
- Productization gap inventory, part I
- Ranked usability/configuration findings
- Candidate product goals for later filtering

### Completion Criteria
- The biggest product gaps are reduced to named classes rather than generic
  “not productized enough” language
- The inventory is grounded in the live repo, not only the earlier review
- Day 4 can continue from concrete findings rather than broad impressions

---

## Day 4: Productization Gap Inventory II

**Title:** Productization Audit II  
**Theme:** Extend the inventory into platform, assurance, and maintainability
residuals and collapse the full Epic 6 queue into a ranked map  
**Time estimate:** 11 hours

### Tasks
1. Audit the strongest remaining non-usability gaps across:
   - performance governance
   - backend flexibility
   - platform/build packaging
   - assurance and oracle depth
   - maintainability hotspots
2. Merge Day 3 and Day 4 findings into one ranked Epic 6 inventory.
3. Identify which findings are:
   - must-fix product gaps
   - important quality/performance gaps
   - bounded maintainability debt
   - explicit non-goals for Epic 6
4. Confirm which deferred Epic 5 seams remain relevant Epic 6 candidates.
5. Record the final ranked productization inventory and transition note to the
   target-definition phase.

### Deliverables
- Productization gap inventory, part II
- Unified ranked Epic 6 gap map
- Preliminary must-fix versus defer split

### Completion Criteria
- The overall Epic 6 gap list is ranked and coherent
- Product, quality, performance, and maintainability findings are separated
  clearly enough to drive target-setting
- The inventory can now support a real state-of-the-art target definition

---

## Day 5: State-of-the-Art Target Definition

**Title:** Target Definition  
**Theme:** Decide which Epic 6 outcomes are real product goals versus explicit
non-goals  
**Time estimate:** 12 hours

### Tasks
1. Convert the ranked inventory into candidate Epic 6 target statements.
2. Define what “state of the art” should and should not mean for this project:
   - local single-node sparse linear algebra product quality
   - user-facing ergonomics
   - performance governance
   - validation rigor
   - platform reach
3. Reject target inflation that would imply:
   - distributed/HPC scope
   - vendor-tuned backend parity
   - impossible broad platform guarantees
   - immediate universal algorithm coverage
4. Write the explicit non-goal fence for Epic 6.
5. Record the candidate scorecard later sprints should satisfy.

### Deliverables
- State-of-the-art target definition
- Explicit Epic 6 non-goal fence
- Candidate success scorecard

### Completion Criteria
- Epic 6 goals are concrete enough to guide implementation prioritization
- Unrealistic or out-of-scope ambitions are explicitly rejected
- Later sprint work can be evaluated against a stable target instead of vague
  aspiration

---

## Day 6: Architecture Contract Audit

**Title:** Architecture Audit  
**Theme:** Map the strongest architectural seams that later Epic 6 work must
respect before code changes begin  
**Time estimate:** 10 hours

### Tasks
1. Audit the strongest architecture-sensitive seams across:
   - direct lifecycle ownership
   - iterative/eigensolver handle ownership
   - configuration/control surfaces
   - benchmark and test governance seams
   - internal hotspot files and helper boundaries
2. Identify the most fragile or high-leverage seams where accidental widening
   would create product confusion.
3. Identify the interfaces that later sprints are likely to touch:
   - public option surfaces
   - internal configuration flow
   - backend dispatch/control
   - packaging/build entry points
4. Record the architecture risks of proceeding without a frozen contract.
5. Write the architecture-seam artifact that leads into Days 7-8.

### Deliverables
- Architecture seam audit
- High-risk interface map
- Candidate architecture-contract topics

### Completion Criteria
- The strongest architectural seams are mapped before implementation begins
- High-risk product/architecture boundaries are explicit
- Later contract writing can proceed from concrete code ownership rather than
  abstract architecture language

---

## Day 7: Configuration & Performance Surface Audit I

**Title:** Config Audit I  
**Theme:** Reduce the strongest env-var-driven and advanced control surfaces to
named productization gaps  
**Time estimate:** 9 hours

### Tasks
1. Audit the current configuration/control surfaces across:
   - environment variables
   - compile-time switches
   - public option structures
   - process-global behavior
2. Identify where advanced tuning is discoverable, hidden, fragmented, or too
   global.
3. Separate findings into:
   - must-be-public later
   - must-be-internal later
   - documentation-only drift
   - architecture-risk seams
4. Cross-check the findings against the Epic 6 review claims so the audit stays
   grounded.
5. Record the first configuration/performance artifact with named seams and
   risks.

### Deliverables
- Configuration/performance audit, part I
- Env-var/control-surface findings
- Candidate public/internal split

### Completion Criteria
- The strongest configuration and control gaps are explicit and ranked
- The audit distinguishes real productization work from mere wording cleanup
- Day 8 can continue from concrete surface maps instead of broad complaints

---

## Day 8: Configuration & Performance Surface Audit II

**Title:** Config Audit II  
**Theme:** Extend the audit into backend sensitivity, benchmark governance, and
performance-story contract  
**Time estimate:** 11 hours

### Tasks
1. Audit the strongest backend/performance-sensitive surfaces across:
   - sparse direct workflows
   - iterative/eigensolver workflows
   - benchmark drivers and README stories
   - performance-sensitive build or runtime choices
2. Identify which later Epic 6 sprints should own:
   - backend/control public options
   - benchmark governance improvements
   - performance-baseline policy
   - packaging/platform follow-through
3. Separate real implementation needs from measurement-only or docs-only needs.
4. Merge Day 7 and Day 8 findings into one configuration/performance map.
5. Record the final audit artifact and the strongest future implementation
   queue it implies.

### Deliverables
- Configuration/performance audit, part II
- Unified config/performance surface map
- Ranked future implementation queue

### Completion Criteria
- The strongest env-var-driven and backend-sensitive seams are concretely
  mapped
- Later Epic 6 sprints can inherit a clear implementation order from this audit
- The performance story is grounded in real benchmark/workflow surfaces

---

## Day 9: Architecture Contract Design

**Title:** Contract Design  
**Theme:** Freeze the architectural rules that later Epic 6 implementation
work must preserve  
**Time estimate:** 11 hours

### Tasks
1. Convert the earlier audits into a written architecture contract covering:
   - product-facing workflow boundaries
   - configuration/public-option rules
   - internal versus public control placement
   - benchmark/performance governance expectations
   - assurance and test-expansion expectations
2. Define what future sprints may widen versus what must stay bounded.
3. Define the compatibility rules for:
   - public one-shot workflows
   - repeated-run direct lifecycle
   - iterative handles
   - eigensolver handles
4. Define the non-goal fence for backend and platform ambition.
5. Write the draft contract artifact and review it against the Epic 6 target
   definition.

### Deliverables
- Draft Epic 6 architecture contract
- Compatibility and control-placement rules
- Bounded widening/non-goal rules

### Completion Criteria
- Later implementation work has a written architecture fence
- Public/internal ownership rules are concrete enough to prevent drift
- The contract aligns with the Day 5 target definition

---

## Day 10: Validation & Platform Contract Freeze

**Title:** Validation Contract  
**Theme:** Freeze the Epic 6 truthfulness, validation, and platform contract
before implementation sprints begin  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the maintained reviewed baseline and platform story.
2. Define the Epic 6 validation contract explicitly:
   - default gates for code-touching work
   - stronger gates for substantial architecture/performance work
   - parity-count anchor expectations
   - docs-only exceptions
3. Define the truthful platform contract for:
   - Linux reviewed path
   - Windows reviewed subset
   - macOS staged limits
   - dead-code and coverage expectations
4. Cross-check the contract against:
   - README
   - maintainer guide
   - Makefile
   - CI/workflow surfaces
5. Write the frozen validation/platform artifact.

### Deliverables
- Frozen Epic 6 validation contract
- Frozen Epic 6 platform/truthfulness contract
- Explicit gate-selection policy

### Completion Criteria
- Implementation sprints inherit a stable validation/platform contract
- No platform or gate ambiguity remains around Epic 6 execution
- The contract matches the live repo instead of aspirational tooling claims

---

## Day 11: Cross-Surface Reconciliation Audit

**Title:** Reconciliation Audit  
**Theme:** Check the new baseline, target, architecture, and validation
artifacts against each other before final closeout writing  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 60 artifacts as one package.
2. Identify any contradiction across:
   - target definition
   - non-goal fence
   - architecture contract
   - validation/platform contract
   - ranked gap inventory
3. Resolve wording-level contradictions inside Sprint 60 notes/artifacts.
4. Record any residual open questions that must be carried intentionally into
   Sprint 61 rather than silently ignored.
5. Write the reconciliation audit artifact.

### Deliverables
- Cross-artifact reconciliation audit
- Explicit carry-forward residual list
- Closeout-writing checklist

### Completion Criteria
- Sprint 60’s artifacts read as one coherent package
- Any remaining ambiguity is recorded as a conscious future input, not a hidden
  contradiction
- Day 12-14 closeout work has a stable final writing base

---

## Day 12: Compatibility & Readiness Audit

**Title:** Readiness Audit  
**Theme:** Confirm Sprint 60 is ready to hand off into implementation-oriented
Epic 6 work without reopening baseline questions  
**Time estimate:** 8 hours

### Tasks
1. Audit the Sprint 60 package against the preserved repo compatibility fence.
2. Confirm the baseline package does not widen public behavior or silently
   rewrite the maintained product story.
3. Confirm that Sprint 61 can begin from:
   - frozen target definition
   - frozen architecture contract
   - frozen validation/platform contract
   - ranked implementation queue
4. Record the exact Day 13 validation checklist for the sprint close.
5. Write the readiness/compatibility artifact.

### Deliverables
- Sprint 60 readiness audit
- Preserved compatibility note
- Day 13 validation checklist

### Completion Criteria
- Sprint 60 leaves no unresolved baseline or contract ambiguity for Sprint 61
- The preserved compatibility fence remains explicit
- The validation checklist is fixed before the final sweep

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Reconfirm the full measured baseline and representative workflow
surfaces from the final Sprint 60 tree  
**Time estimate:** 12 hours

### Tasks
1. Run the maintained validation gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm the parity/count anchors:
   - `ctest -N --test-dir build/quality-review-cmake`
   - Makefile/CMake parity
   - full reviewed CMake `ctest`
3. Run the targeted Sprint 60 follow-ons:
   - representative direct tests
   - representative iterative/eigensolver tests
   - representative examples
   - representative benchmark drivers
4. Record measured outputs and any non-blocking anomalies.
5. Write the full validation artifact from the final landed tree.

### Deliverables
- Full Sprint 60 validation record
- Measured parity anchors
- Representative workflow proof results

### Completion Criteria
- Sprint 60 closes from a fresh validated baseline
- All maintained quality gates pass
- Representative workflow surfaces still match the preserved product story

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Package Sprint 60 into a clean Epic 6 handoff for the first
implementation sprint  
**Time estimate:** 12 hours

### Tasks
1. Summarize the final Sprint 60 outcomes:
   - baseline freeze
   - ranked gap inventory
   - target/non-goal definition
   - configuration/performance map
   - architecture contract
   - validation/platform contract
2. Record the final carry-forward queue for Sprint 61.
3. Confirm whether `docs/planning/EPIC_6/PROJECT_PLAN.md` needs any correction
   from the landed Sprint 60 findings.
4. Write the Day 14 closeout/handoff artifact and final Sprint 60 working-note
   synthesis.
5. Make the final end-state explicit for the next sprint.

### Deliverables
- Sprint 60 closeout and handoff artifact
- Final Sprint 60 synthesis in working notes
- Explicit Sprint 61 starting queue

### Completion Criteria
- Sprint 60 closes from a validated, coherent, and well-bounded baseline
- Sprint 61 can begin implementation work without reopening Sprint 60 contract
  decisions
- The final handoff package states both the achieved baseline and the preserved
  non-goal fence clearly
