# Sprint 142 Plan: Runtime Backend Governance & Sentinel Expansion

**Sprint Duration:** 14 days
**Goal:** Convert runtime/backend behavior into a clearer maintained contract
and expand sentinels where they provide useful local regression evidence. This
sprint implements the Sprint 142 section of
`docs/planning/EPIC_12/PROJECT_PLAN.md`.

**Starting Point:** Sprint 142 begins from:
- Sprint 141 normalized report fields and freshness diagnostics
- the Sprint 141 `runtime_backend` defer row and closeout handoff
- current OpenMP, backend dispatch, dense helper, eigensolver backend,
  direct-solver dispatch, environment-variable, and typed-option surfaces
- existing benchmark, sentinel, guardrail, runtime, backend, and maintainer
  documentation
- the current validation expectations for C/header, script, report, and docs
  changes

The sprint must:
- audit runtime/backend controls before changing product behavior
- define maintained precedence among typed options, environment overrides,
  compile-time flags, backend fallback, and deterministic defaults
- promote the highest-value implicit or environment-only controls into typed
  options, or explicitly classify them as maintainer-only
- expand normalized local sentinel rows only where they provide useful
  regression evidence without creating portable timing claims
- update docs and examples for earned runtime/backend behavior
- run focused runtime/backend tests, sentinels, report freshness checks, and
  full quality gates if C or header files change
- publish earned runtime/backend claims, non-claims, and Sprint 143
  package/ABI prerequisites

**End State:** Sprint 142 leaves behind:
- runtime/backend control audit
- maintained precedence contract
- typed-control or explicit deferral batch
- normalized sentinel rows for selected backend decisions
- updated runtime/backend docs and examples
- validation evidence for changed C/header, script, report, and docs surfaces
- Sprint 143 package/ABI handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 142 project-plan estimate.

---

## Day 1: Runtime Governance Intake

**Title:** Governance Intake
**Theme:** Establish Sprint 142 scope, inherited runtime/backend handoff, and
claim boundaries before auditing controls
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 142 section of
   `docs/planning/EPIC_12/PROJECT_PLAN.md`.
2. Review Sprint 141 artifacts, retrospective, normalized report fields, and
   `runtime_backend` defer row.
3. Create Sprint 142 working notes and artifact directory structure.
4. Identify the initial runtime/backend surfaces: OpenMP, backend dispatch,
   dense helpers, eigensolver backend selection, direct-solver dispatch,
   environment variables, compile-time flags, typed options, sentinels, and
   docs.
5. Map Sprint 142 Items 1-7 to day-level owners.
6. Record initial non-claims, stop conditions, and validation expectations.

### Deliverables
- Sprint 142 working-notes baseline
- artifact directory structure
- inherited Sprint 141 handoff summary
- initial runtime/backend surface map
- item-to-day owner map
- initial claim-boundary and validation register

### Completion Criteria
- every Sprint 142 project-plan item has a day-level owner
- the Sprint 141 runtime/backend defer row is represented as a handoff, not
  unfinished report-index work
- stop conditions are explicit before code or contract changes begin

---

## Day 2: Runtime Control Inventory

**Title:** Control Inventory
**Theme:** Build the canonical inventory of runtime/backend controls, entry
points, defaults, and evidence owners
**Time estimate:** 12 hours

### Tasks
1. Inspect source, headers, tests, examples, benchmarks, scripts, workflows,
   and docs for runtime/backend controls.
2. Inventory environment variables, compile-time flags, typed options,
   implicit defaults, backend routing thresholds, fallback paths, and sentinel
   commands.
3. Record owner, scope, default behavior, current validation, user visibility,
   and documentation status for each control.
4. Identify controls that affect deterministic behavior, performance,
   backend selection, package behavior, or platform support.
5. Separate public controls from maintainer-only or diagnostic controls.
6. Write the runtime control inventory artifact.

### Deliverables
- runtime/backend control inventory
- command/test/doc ownership map
- public versus maintainer-only classification draft
- unknowns and risk list

### Completion Criteria
- OpenMP, direct-solver, eigensolver, dense helper, backend dispatch,
  environment, and typed-option surfaces are accounted for
- each control has an owner, current behavior, and validation status
- ambiguous controls are flagged before precedence design begins

---

## Day 3: Backend Dispatch Audit

**Title:** Dispatch Audit
**Theme:** Audit backend routing and fallback behavior in direct solvers,
eigensolvers, dense helpers, and sentinel paths
**Time estimate:** 12 hours

### Tasks
1. Inspect direct-solver backend dispatch paths for AUTO, forced backend, and
   fallback behavior.
2. Inspect eigensolver backend selection and workspace/runtime control paths.
3. Inspect dense helper selection, panel solver, OpenMP, and build-mode
   interactions where they affect backend behavior.
4. Map test coverage for each dispatch and fallback path.
5. Identify missing tests or report rows needed to make runtime decisions
   observable.
6. Write the backend dispatch audit artifact.

### Deliverables
- backend dispatch audit
- fallback and AUTO-routing map
- current test coverage map
- candidate sentinel expansion list

### Completion Criteria
- backend routing and fallback semantics are documented before changes
- missing coverage is described with proposed proof owners
- no implementation change is made before precedence rules are drafted

---

## Day 4: Precedence Contract Design

**Title:** Precedence Design
**Theme:** Define maintained precedence for typed options, environment
overrides, compile-time flags, fallback, and deterministic behavior
**Time estimate:** 12 hours

### Tasks
1. Define precedence among explicit typed options, environment compatibility
   overrides, compile-time flags, backend AUTO routing, and default behavior.
2. Define how invalid, unsupported, or unavailable backend requests fail or
   fall back.
3. Define deterministic behavior requirements for tests, sentinels, examples,
   and local benchmark/report rows.
4. Define public versus maintainer-only control language.
5. Draft validation scenarios for the precedence contract.
6. Write the precedence contract design artifact.

### Deliverables
- maintained precedence contract draft
- fallback and failure behavior matrix
- public/maintainer-only control classification
- validation scenario list

### Completion Criteria
- typed options and environment overrides have an explicit ordering
- fallback behavior is documented without broad platform or performance claims
- validation scenarios are concrete enough to implement

---

## Day 5: Precedence Contract Implementation

**Title:** Contract Implementation
**Theme:** Land the mechanical contract surface for runtime/backend precedence
without broadening support claims
**Time estimate:** 12 hours

### Tasks
1. Update source, headers, internal helpers, or test fixtures required to make
   the Day 4 precedence contract executable.
2. Add or update focused tests for selected precedence paths.
3. Preserve existing public behavior unless the Day 4 contract explicitly
   changes it.
4. Keep compatibility environment variables or debug controls maintainer-only
   when they are not promoted.
5. Run focused compile/test checks for touched surfaces.
6. Write the precedence implementation artifact.

### Deliverables
- implemented precedence contract batch
- focused precedence tests
- updated internal or public control surface as needed
- implementation artifact

### Completion Criteria
- selected precedence behavior is mechanically testable
- unsupported or maintainer-only controls remain clearly scoped
- any C/header change has focused validation recorded

---

## Day 6: Typed-Control Selection

**Title:** Typed-Control Selection
**Theme:** Select the highest-value environment-only or implicit controls for
typed promotion or explicit maintainer-only deferral
**Time estimate:** 12 hours

### Tasks
1. Review the Day 2 inventory and Day 4 precedence contract for candidate
   typed-control promotion.
2. Score candidates by user value, implementation risk, validation surface,
   documentation burden, and claim risk.
3. Select a bounded implementation batch that can be completed this sprint.
4. Explicitly defer controls that are diagnostic, platform-specific,
   experimental, or not ready for public API exposure.
5. Define tests and docs needed for selected controls.
6. Write the typed-control selection artifact.

### Deliverables
- typed-control candidate matrix
- selected implementation batch
- explicit deferral list with owners and reasons
- test and documentation plan

### Completion Criteria
- the selected batch is small enough to finish and validate
- deferred controls are not silently dropped
- public API and maintainer-only boundaries are explicit

---

## Day 7: Typed-Control Implementation

**Title:** Typed-Control Batch
**Theme:** Promote selected runtime/backend controls or make deferrals
explicit in code and tests
**Time estimate:** 12 hours

### Tasks
1. Implement the selected typed-control batch from Day 6.
2. Add public headers, option structs, internal parsing, or helper updates as
   required by the selected controls.
3. Add focused unit/integration tests for default, explicit, invalid,
   unsupported, and fallback cases.
4. Update source-list or build registration if new test files are added.
5. Preserve ABI/package non-claims unless explicitly earned.
6. Write the typed-control implementation artifact.

### Deliverables
- implemented typed-control or explicit-deferral batch
- focused tests for selected controls
- build registration updates if needed
- implementation artifact

### Completion Criteria
- selected controls are observable through typed APIs or explicitly deferred
- tests cover default and non-default behavior
- C/header changes are ready for full quality gates later in the sprint

---

## Day 8: Runtime Sentinel Design

**Title:** Sentinel Design
**Theme:** Design normalized local sentinel rows for selected hot paths or
backend decisions without portable timing claims
**Time estimate:** 12 hours

### Tasks
1. Review current performance sentinel, large-matrix guardrail, normalized
   report-index, and benchmark artifacts.
2. Select runtime/backend decisions that need local regression visibility.
3. Define row meaning, support tier, artifact path, command, threshold,
   freshness policy, and non-claims for each candidate sentinel row.
4. Separate hard local gates from advisory threshold-free measurements.
5. Define tests or synthetic fixtures for normalized sentinel row parsing.
6. Write the sentinel expansion design artifact.

### Deliverables
- sentinel expansion design
- selected runtime/backend sentinel rows
- hard-gate versus advisory-row classification
- normalized report-index integration plan

### Completion Criteria
- selected sentinel rows are tied to maintained commands or fixtures
- portable performance and platform claims remain explicit non-claims
- report-index integration is deterministic

---

## Day 9: Sentinel Implementation

**Title:** Sentinel Implementation
**Theme:** Add selected runtime/backend sentinel rows and normalized index
integration
**Time estimate:** 12 hours

### Tasks
1. Update sentinel scripts, report manifests, normalized-index ingestion, or
   tests for the selected Day 8 rows.
2. Preserve existing sentinel and guardrail row semantics.
3. Add focused tests or synthetic fixtures for new runtime/backend sentinel
   rows.
4. Run normalized report-index and freshness checks for sentinel-related
   families.
5. Document generated-output policy and ignored artifact paths.
6. Write the sentinel implementation artifact.

### Deliverables
- implemented runtime/backend sentinel rows
- normalized index/freshness integration
- focused sentinel parsing or script tests
- implementation artifact

### Completion Criteria
- new sentinel rows appear deterministically in normalized output
- hard gates and advisory rows are distinguishable
- local measurements are not described as portable performance evidence

---

## Day 10: Runtime Docs And Examples

**Title:** Docs And Examples
**Theme:** Update user and maintainer docs for earned runtime/backend contract
behavior
**Time estimate:** 12 hours

### Tasks
1. Update README, maintainer guide, benchmark docs, and cookbook/runtime
   examples affected by the precedence contract or typed-control batch.
2. Update API comments or examples if public typed options were changed.
3. Document maintainer-only controls and explicit deferrals where relevant.
4. Document sentinel interpretation and normalized report-index commands for
   runtime/backend rows.
5. Preserve package, ABI, platform, and portable performance non-claims.
6. Write the documentation alignment artifact.

### Deliverables
- updated runtime/backend docs
- updated examples or API comments if needed
- sentinel interpretation guidance
- documentation alignment artifact

### Completion Criteria
- docs match implemented runtime/backend behavior
- users can distinguish public typed controls from maintainer-only controls
- sentinel rows are framed as local regression evidence only

---

## Day 11: Focused Runtime Validation

**Title:** Focused Validation
**Theme:** Run focused runtime/backend tests, sentinels, normalized report
checks, and repair any scoped failures
**Time estimate:** 12 hours

### Tasks
1. Run focused tests for precedence, typed controls, backend dispatch,
   fallback, and sentinel row parsing.
2. Run selected runtime/backend sentinel commands when feasible locally.
3. Run normalized report-index checks and freshness checks for affected
   families.
4. Inspect generated artifacts for ignored/untracked hygiene.
5. Repair scoped failures that are clearly within Sprint 142 scope.
6. Write the focused validation artifact.

### Deliverables
- focused runtime/backend validation evidence
- sentinel and report-index validation evidence
- scoped repair notes if needed
- validation artifact

### Completion Criteria
- focused tests pass or failures are repaired before broad validation
- generated outputs remain ignored unless intentionally source-controlled
- any remaining issue has an owner and stop condition

---

## Day 12: Full Quality Gate

**Title:** Quality Gate
**Theme:** Run required full quality gates and repository hygiene checks for
the final implementation surface
**Time estimate:** 12 hours

### Tasks
1. Determine whether C or header files changed during Sprint 142.
2. If C/header files changed, run `make format && make lint && make test`.
3. Run Python compile checks for changed scripts and tests.
4. Run corpus schema validation and normalized report-index/freshness checks.
5. Run docs and whitespace hygiene checks.
6. Write the quality gate artifact with exact commands and results.

### Deliverables
- full quality gate results
- script/report/docs validation evidence
- generated-output hygiene evidence
- quality gate artifact

### Completion Criteria
- all required checks for touched surfaces pass
- full C quality gate passes if C/header files changed
- validation evidence is current and reproducible

---

## Day 13: Claim Closure And Sprint 143 Handoff

**Title:** Claim Closure
**Theme:** Publish earned runtime/backend claims, remaining non-claims, and
package/ABI prerequisites for Sprint 143
**Time estimate:** 12 hours

### Tasks
1. Compare Sprint 142 outcomes against the Day 1 scope and project-plan items.
2. Identify exactly which runtime/backend claims are earned by implemented
   code, tests, sentinels, and docs.
3. Identify remaining non-claims for backend portability, platform support,
   performance, package/ABI, and state-of-the-art status.
4. Prepare Sprint 143 package/ABI prerequisites and handoff requirements.
5. Update working notes with claim closure and residual risk.
6. Write the claim closure and Sprint 143 handoff artifact.

### Deliverables
- earned runtime/backend claim list
- remaining non-claim register
- Sprint 143 package/ABI handoff
- claim closure artifact

### Completion Criteria
- runtime/backend claims are backed by specific evidence
- residual non-claims are explicit
- Sprint 143 receives package/ABI prerequisites rather than vague runtime debt

---

## Day 14: Closeout

**Title:** Closeout
**Theme:** Finalize Sprint 142 artifacts, validation evidence, working notes,
and handoff package
**Time estimate:** 12 hours

### Tasks
1. Re-run final report-index, freshness, docs hygiene, and required quality
   checks after Day 13 updates.
2. Review all Sprint 142 artifacts for consistency with implemented behavior.
3. Confirm runtime/backend rows do not overclaim portable performance,
   platform support, package/ABI support, or broad backend governance closure.
4. Update working notes with final validation, changed files, decisions,
   deferred work, and known risks.
5. Write the closeout validation summary artifact.
6. Prepare the sprint for retrospective and PR creation.

### Deliverables
- final validation evidence
- Sprint 142 closeout summary
- final Sprint 143 handoff
- updated working notes
- complete artifact package

### Completion Criteria
- Sprint 142 deliverables are present and traceable to Items 1-7
- validation evidence is current and reproducible
- remaining runtime/backend or package/ABI work is explicitly routed forward
