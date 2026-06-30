# Sprint 100 Plan: Epic 10 Baseline, State-of-the-Art Target & Evidence Contract

**Sprint Duration:** 14 days
**Goal:** Freeze the post-Epic-9 baseline, define the precise
state-of-the-art comparison target, and create the evidence templates that
will govern implementation work across Epic 10. This sprint implements the
Sprint 100 section of `docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 100 begins from:
- merged Epic 9 closeout and PR #113
- the Epic 9 retrospective, residual queue, and
  `docs/planning/EPIC_9/POST_EPIC_9_HANDOFF.md`
- the Epic 10 Codex review and gap-closure todo
- a repository with strong reviewed quality surfaces but remaining product,
  comparison, packaging, and maintainability gaps

The strongest Sprint 100 pressure is to prevent Epic 10 from becoming a broad
modernization grab bag. The sprint must:
- re-confirm the live post-Epic-9 quality and evidence baseline
- define what state-of-the-art means for this repository
- separate earned claims, candidate claims, residuals, and non-goals
- create reusable evidence templates for later implementation sprints
- capture objective source, test, package, and comparison metrics
- audit public claims before new work widens the surface

**End State:** Sprint 100 leaves behind:
- a post-Epic-9 baseline artifact
- an Epic 10 state-of-the-art target and non-goal fence
- evidence templates for solver comparisons, benchmarks, platform tiers,
  packaging proof, and non-claim tracking
- a claim map for Sprints 101-109
- Sprint 100 working notes and handoff criteria

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 100 project-plan estimate.

---

## Day 1: Sprint 100 Scope & Baseline Setup

**Title:** Scope Baseline
**Theme:** Convert the Sprint 100 project-plan section into one bounded
execution package before evidence collection starts
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 100 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read the Epic 10 review and todo documents.
3. Inventory required Sprint 100 workstreams:
   - baseline quality recheck
   - state-of-the-art definition
   - residual queue conversion
   - evidence template creation
   - baseline metrics capture
   - public claim audit
   - sprint closeout and handoff
4. Create Sprint 100 working notes and an artifacts directory.
5. Record validation expectations for documentation-only days and any
   command-running evidence days.

### Deliverables
- Sprint 100 workstream inventory
- working-notes baseline
- initial artifacts directory structure
- validation expectation list

### Completion Criteria
- Sprint 100 work is bounded before collection begins
- all project-plan items have a day-level owner
- validation expectations are visible in working notes

---

## Day 2: Reviewed Quality Baseline Recheck

**Title:** Quality Baseline
**Theme:** Reconfirm the live reviewed quality surface after Epic 9 and PR
#113
**Time estimate:** 12 hours

### Tasks
1. Run or inspect the strongest documented reviewed quality entry points.
2. Capture Makefile reviewed test counts and command expectations.
3. Capture CMake reviewed test discovery and parity expectations.
4. Capture install/export proof status and consumer-proof expectations.
5. Record any commands that are intentionally deferred, skipped, or
   supplemental.

### Deliverables
- reviewed quality baseline artifact
- Make/CMake parity notes
- install/export proof notes
- skipped or supplemental command register

### Completion Criteria
- Sprint 100 has a current quality baseline
- reviewed and supplemental checks are separated clearly
- any failure or unclear command stops implementation planning

---

## Day 3: Build, Package & CI Evidence Baseline

**Title:** Build Evidence
**Theme:** Map the current build, package, CI, and platform proof surfaces
before Epic 10 changes them
**Time estimate:** 12 hours

### Tasks
1. Review Makefile, CMake, package, pkg-config, and install/export surfaces.
2. Review CI workflows for Linux, macOS, Windows, sanitizer, coverage,
   benchmark, and package lanes.
3. Capture reviewed versus supplemental platform support boundaries.
4. Identify package and platform non-claims inherited from Epic 9.
5. Write the build/package/CI baseline artifact.

### Deliverables
- build and package proof map
- CI lane summary
- platform tier draft
- package/platform non-claim list

### Completion Criteria
- package proof surfaces are named and current
- CI lanes have clear reviewed or supplemental status
- platform asymmetries are explicit rather than inferred

---

## Day 4: Source, Test & Maintainability Metrics

**Title:** Metrics Baseline
**Theme:** Capture objective source, test, and maintainability metrics for
later Epic 10 extraction work
**Time estimate:** 12 hours

### Tasks
1. Count repository files and relevant source, header, test, benchmark, and
   example lines.
2. Identify largest source files and largest test files.
3. Identify history-heavy, fallback-heavy, or temporary-sounding comments in
   permanent code and tests.
4. Capture source-list and CMake source ownership signals.
5. Write the maintainability metrics artifact.

### Deliverables
- source/test size baseline
- large-file hotspot table
- chronology/fallback comment sample
- source-list ownership notes

### Completion Criteria
- maintainability claims use measured data
- hotspot candidates are ranked for later sprints
- no file-splitting work starts during the baseline sprint

---

## Day 5: External Comparison & Benchmark Baseline

**Title:** Comparison Baseline
**Theme:** Inventory the current external comparison, benchmark, coverage, and
reporting surfaces
**Time estimate:** 12 hours

### Tasks
1. Review benchmark scripts, benchmark documentation, and generated report
   expectations.
2. Inventory maintained external comparison lanes by solver family.
3. Capture coverage command status, threshold, and reviewed/supplemental
   boundary.
4. Identify comparison gaps for direct, iterative, eigensolver, SVD, reorder,
   and graph paths.
5. Write the comparison and benchmark baseline artifact.

### Deliverables
- external comparison lane inventory
- benchmark/reporting surface map
- coverage architecture notes
- comparison gap table by solver family

### Completion Criteria
- every current comparison lane has a named owner
- benchmark claims remain local and bounded
- uncovered comparison gaps are candidates, not claims

---

## Day 6: State-of-the-Art Definition Draft

**Title:** Target Draft
**Theme:** Define what state-of-the-art means for this repository without
importing irrelevant product obligations
**Time estimate:** 12 hours

### Tasks
1. Define the comparison set for sparse linear algebra product maturity.
2. Classify expected capabilities into must-have, stretch, and explicit
   non-goal categories.
3. Separate algorithmic quality, API usability, package maturity, platform
   proof, and benchmark evidence dimensions.
4. Draft the Epic 10 state-of-the-art target.
5. List disallowed broad claims that require future evidence.

### Deliverables
- state-of-the-art target draft
- capability category table
- disallowed claim list
- comparison-dimension taxonomy

### Completion Criteria
- the target is specific to this C library
- non-goals are explicit enough to prevent scope sprawl
- later sprints can map work to evidence dimensions

---

## Day 7: Residual Queue Conversion

**Title:** Residual Map
**Theme:** Convert Epic 9 carry-forward work into an Epic 10 claim and risk
map
**Time estimate:** 12 hours

### Tasks
1. Re-read Epic 9 residuals, non-claims, and
   `docs/planning/EPIC_9/POST_EPIC_9_HANDOFF.md`.
2. Map each residual to a Sprint 100-109 candidate owner.
3. Assign risk levels and evidence requirements to each residual.
4. Separate residuals that are in scope from deliberate Epic 10 non-goals.
5. Write the residual-to-claim map artifact.

### Deliverables
- Epic 9 residual conversion table
- Sprint 101-109 owner draft
- risk and evidence requirement map
- updated non-goal register

### Completion Criteria
- every Epic 9 residual has an Epic 10 disposition
- high-risk residuals have evidence requirements
- out-of-scope residuals are written as explicit non-goals

---

## Day 8: Claim Map & Sprint Dependency Model

**Title:** Claim Model
**Theme:** Turn the target definition and residual conversion into a sprint
dependency model
**Time estimate:** 12 hours

### Tasks
1. Build the Epic 10 claim map for Sprints 101-109.
2. Identify prerequisites between compressed-first, solver evidence,
   backend/runtime, reorder/graph, maintainability, API docs, packaging, and
   final closeout work.
3. Mark claims as earned, candidate, blocked, or non-goal.
4. Define minimum evidence required to move each claim to earned status.
5. Write the claim-map artifact.

### Deliverables
- Epic 10 claim map
- sprint dependency table
- earned/candidate/blocked/non-goal classification
- minimum evidence criteria by claim

### Completion Criteria
- every implementation sprint has clear claim dependencies
- broad claims require explicit evidence before closeout
- final closeout criteria are traceable to the claim map

---

## Day 9: Solver Comparison Evidence Template

**Title:** Solver Template
**Theme:** Create reusable evidence templates for direct, iterative,
eigensolver, SVD, reorder, and graph comparisons
**Time estimate:** 12 hours

### Tasks
1. Define required fields for solver-family comparison artifacts.
2. Include matrix fixture identity, solver options, tolerance, residual,
   expected failure, external oracle, and environment fields.
3. Add sections for unsupported cases and non-claims.
4. Pilot the template against one existing comparison lane.
5. Store the solver comparison template in Sprint 100 artifacts.

### Deliverables
- solver comparison evidence template
- pilot-filled example from an existing lane
- unsupported-case and non-claim fields
- template usage notes

### Completion Criteria
- future solver comparison artifacts can use the template without redesign
- template distinguishes correctness, convergence, and timing evidence
- unsupported inputs are captured as first-class results

---

## Day 10: Benchmark, Coverage & Performance Template

**Title:** Benchmark Template
**Theme:** Create reusable templates for benchmark interpretation, coverage
evidence, and bounded performance sentinels
**Time estimate:** 12 hours

### Tasks
1. Define benchmark artifact fields for command, fixture, environment, metric,
   local timing, and interpretation.
2. Define coverage artifact fields for command, threshold, scope, reviewed
   status, and known exclusions.
3. Define performance sentinel fields that avoid portable superiority claims.
4. Pilot the template against one existing benchmark or coverage surface.
5. Store benchmark and coverage templates in Sprint 100 artifacts.

### Deliverables
- benchmark interpretation template
- coverage evidence template
- bounded performance sentinel template
- pilot-filled benchmark or coverage example

### Completion Criteria
- benchmark templates prevent overclaiming local timings
- coverage templates separate reviewed and supplemental gates
- future performance sentinels have a consistent evidence shape

---

## Day 11: Platform & Packaging Evidence Template

**Title:** Package Template
**Theme:** Create reusable evidence templates for package proof, ABI decisions,
consumer validation, and platform tiers
**Time estimate:** 12 hours

### Tasks
1. Define install/export and downstream consumer proof fields.
2. Define package-version, exact-package, pkg-config, and CMake package fields.
3. Define platform-tier fields for Linux, macOS, Windows, exclusions, and
   reviewed versus supplemental status.
4. Define ABI/shared-library decision fields for Sprint 108.
5. Store platform and packaging templates in Sprint 100 artifacts.

### Deliverables
- package proof template
- platform tier template
- ABI decision template
- consumer validation checklist

### Completion Criteria
- Sprint 108 can use the templates directly
- platform exclusions are tracked explicitly
- package claims have a required proof field

---

## Day 12: Public Claim Audit

**Title:** Claim Audit
**Theme:** Recheck public and support surfaces for unsupported
state-of-the-art, package, performance, and platform claims
**Time estimate:** 12 hours

### Tasks
1. Audit README, install docs, benchmark docs, examples, maintainer docs, and
   public headers for broad or stale claims.
2. Compare claims against the Day 8 claim map and Day 9-11 templates.
3. Mark each claim as supported, candidate, needs wording change, or non-claim.
4. Draft a public-claim audit artifact.
5. Identify any immediate documentation fixes that are small enough for Sprint
   100 closeout.

### Deliverables
- public claim audit artifact
- supported/candidate/unsupported claim table
- candidate wording-change queue
- immediate fix recommendation list

### Completion Criteria
- public claims are tied to evidence or marked for later work
- unsupported claims are not allowed to flow into implementation sprints
- any immediate fix has a bounded validation path

---

## Day 13: Sprint 100 Integration & Handoff Package

**Title:** Handoff Package
**Theme:** Integrate all Sprint 100 evidence into a coherent Epic 10 launch
package
**Time estimate:** 11 hours

### Tasks
1. Reconcile the baseline artifacts, target definition, residual map, claim
   map, and evidence templates.
2. Resolve contradictions between artifact wording and the project plan.
3. Create the Sprint 100 handoff summary for Sprints 101-109.
4. Update working notes with validation results and any deferred items.
5. Prepare closeout checks for Day 14.

### Deliverables
- integrated Sprint 100 handoff package
- reconciled claim and non-goal registers
- updated working notes
- Day 14 closeout checklist

### Completion Criteria
- Sprints 101-109 have a usable baseline and evidence contract
- handoff language does not overclaim state-of-the-art status
- remaining contradictions are listed before closeout

---

## Day 14: Final Validation & Sprint Closeout

**Title:** Closeout
**Theme:** Validate Sprint 100 artifacts and close the baseline sprint from a
clean documentation and evidence state
**Time estimate:** 11 hours

### Tasks
1. Run final documentation hygiene checks for Sprint 100 files.
2. Run any required quality checks based on touched files.
3. Confirm all Sprint 100 project-plan items have deliverables.
4. Write the Sprint 100 closeout notes and retrospective input.
5. Record the exact handoff requirements for Sprint 101.

### Deliverables
- final validation notes
- complete Sprint 100 artifact index
- Sprint 100 closeout notes
- Sprint 101 handoff requirements

### Completion Criteria
- all Sprint 100 artifacts are present and internally consistent
- validation requirements are satisfied or explicitly blocked
- Sprint 101 can start from a clear compressed-first product-model baseline
