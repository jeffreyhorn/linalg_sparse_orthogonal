# Sprint 147 Plan: Epic 13 Baseline, Claim Targets & Evidence Gates

**Sprint Duration:** 14 days
**Goal:** Freeze the post-Epic-12 baseline, select Epic 13 closure targets,
and define evidence gates for platform, corpus, report, ABI, and comparison
work. This sprint implements the Sprint 147 section of
`docs/planning/EPIC_13/PROJECT_PLAN.md`.

**Starting Point:** Sprint 147 begins from:
- Epic 12 closeout merged into `master`
- `docs/planning/EPIC_12/EPIC_12_RETROSPECTIVE.md`
- Sprint 146 residual queue and final closeout package
- Epic 13 review and remediation todo in
  `docs/planning/EPIC_13/reviews/`
- current Linux, macOS, and Windows CI support-tier definitions
- current corpus/report schemas, package tests, adoption docs, and
  state-of-the-art non-claim boundaries

The sprint must:
- capture the current post-Epic-12 technical and evidence baseline
- reconcile Epic 12 residuals into selected Epic 13 gaps and explicit
  non-goals
- define narrow claim targets for Epic 13 and preserve rejected
  state-of-the-art claims
- create evidence gates for Windows parity, corpus-family proof, generated
  freshness, ABI decisions, and external comparisons
- map required validation by touched surface
- freeze public claim wording before implementation sprints begin
- publish Sprint 148 Windows portability prerequisites

**End State:** Sprint 147 leaves behind:
- Epic 13 baseline inventory
- selected-gap register and duplicate fences
- claim target and non-goal register
- evidence gate templates
- quality surface map
- public claim freeze audit
- Sprint 148 Windows staged-test handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 147 project-plan estimate.

---

## Day 1: Baseline Intake And Artifact Setup

**Title:** Baseline Intake
**Theme:** Establish Sprint 147 scope, artifact structure, source inputs, and
baseline capture rules
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 147 section of
   `docs/planning/EPIC_13/PROJECT_PLAN.md`.
2. Review Epic 12 retrospective, Sprint 146 residual queue, and Epic 13 review
   inputs.
3. Create Sprint 147 working notes and artifact directory structure.
4. Define baseline categories: source/test size, build/package, CI, corpus,
   report, docs, support tiers, and residuals.
5. Map Sprint 147 Items 1-7 to day-level owners.
6. Record stop conditions for unsupported claim promotion or unclear evidence.

### Deliverables
- Sprint 147 working-notes baseline
- artifact directory structure
- baseline category map
- item-to-day owner map
- stop-condition register

### Completion Criteria
- every Sprint 147 project-plan item has a day-level owner
- source inputs are listed and current
- later days have a clear evidence-capture format

---

## Day 2: Source, Test, And Build Baseline

**Title:** Technical Baseline
**Theme:** Capture current source, test, build, package, and CI structure after
Epic 12
**Time estimate:** 12 hours

### Tasks
1. Count current source, header, test, benchmark, example, script, and docs
   surfaces.
2. Identify largest source and test files that create maintainability risk.
3. Capture Makefile and CMake source/test ownership and drift risks.
4. Record Windows CTest registration count and staged-exclusion policy.
5. Inventory package proof scripts, static-first deferral guard, and install
   validation commands.
6. Write the technical baseline artifact.

### Deliverables
- source/test/build baseline table
- large-file maintainability inventory
- Make/CMake drift notes
- Windows CTest and staged-exclusion snapshot
- package proof command inventory

### Completion Criteria
- baseline numbers are reproducible from repo commands
- Windows and package baselines are tied to current files
- maintainability risks are specific enough for later prioritization

---

## Day 3: Corpus, Report, And Evidence Baseline

**Title:** Evidence Baseline
**Theme:** Capture maintained corpus, report-family, generated-output, and
validation evidence after Epic 12
**Time estimate:** 12 hours

### Tasks
1. Inventory current corpus fixture, generator, expected-result, and
   optional-data rows.
2. Inventory report-family rows, freshness policies, row meanings, and
   generated artifact patterns.
3. Map QR and partial-SVD fixture-local closures to their proof owners and
   report commands.
4. Identify generated rows that are local/advisory rather than pass evidence.
5. Capture current corpus/report validation commands and required freshness
   checks.
6. Write the corpus and report baseline artifact.

### Deliverables
- corpus row baseline
- report-family baseline
- QR and partial-SVD proof-owner map
- generated-local evidence classification
- corpus/report validation command list

### Completion Criteria
- source-controlled metadata is separated from observed generated evidence
- fixture-local claim scopes are preserved
- report freshness boundaries are explicit

---

## Day 4: Residual Queue Reconciliation Part 1

**Title:** Residual Intake
**Theme:** Reconcile Epic 12 residuals R1-R14 into Epic 13 candidate gaps,
owners, and dependencies
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 146 published residual queue.
2. Group residuals by platform, corpus, report, package/ABI, adoption,
   runtime/backend, and competitive positioning.
3. Identify dependencies among residuals and detect duplicate or overlapping
   work.
4. Record owner surfaces and prerequisite evidence for each residual.
5. Classify each residual as candidate, blocked, duplicate, deferred, or
   rejected for Epic 13.
6. Write the residual intake artifact.

### Deliverables
- residual grouping table
- dependency map
- owner and prerequisite evidence map
- duplicate and overlap notes
- initial Epic 13 residual classification

### Completion Criteria
- every Epic 12 residual has an Epic 13 disposition
- duplicates are explicit
- no residual becomes a claim without a gate

---

## Day 5: Selected Gap Decision

**Title:** Gap Selection
**Theme:** Select the Epic 13 gaps that can be fully closed and define
explicit non-goals
**Time estimate:** 12 hours

### Tasks
1. Rank candidate gaps by product value, feasibility, evidence maturity, and
   closure risk.
2. Select the gaps Epic 13 will attempt to close completely.
3. Define non-selected gaps and explain why they remain residual.
4. Create duplicate fences so later sprints do not reopen adjacent work.
5. Map selected gaps to Sprints 148-156.
6. Write the selected-gap register artifact.

### Deliverables
- selected Epic 13 gap register
- non-selected residual list
- duplicate fences
- sprint-to-gap map
- feasibility and risk notes

### Completion Criteria
- selected gaps can plausibly close within Epic 13
- non-goals are explicit and defensible
- Sprint 148-156 sequencing follows dependencies

---

## Day 6: Claim Target Register

**Title:** Claim Targets
**Theme:** Define narrow claims Epic 13 may earn and preserve rejected
state-of-the-art claims
**Time estimate:** 12 hours

### Tasks
1. Convert selected gaps into candidate earned claims.
2. Define required evidence for each claim: implementation, tests, reports,
   docs, CI, package proof, or comparison output.
3. Define claim wording boundaries and non-claim language.
4. Identify state-of-the-art, external parity, performance, ABI, and platform
   claims that remain rejected.
5. Record claim promotion and rollback rules.
6. Write the claim target register artifact.

### Deliverables
- candidate earned-claim table
- required evidence map
- non-claim and rejected-claim register
- state-of-the-art decision boundary
- claim promotion and rollback rules

### Completion Criteria
- every candidate claim has concrete required evidence
- unsupported broad claims remain rejected
- later sprint docs have wording boundaries to reuse

---

## Day 7: Windows Evidence Gate Template

**Title:** Windows Gate
**Theme:** Define the evidence gate for Windows staged test portability and
install-validation parity
**Time estimate:** 12 hours

### Tasks
1. Inventory current Windows reviewed, supplemental, staged, and deferred
   support tiers.
2. Define evidence required to promote `test_threads`,
   `test_sprint4_integration`, and `test_fuzz`.
3. Define evidence required to promote or reject reviewed Windows
   install-validation parity.
4. Define CMake registration, expected-count, and hosted log requirements.
5. Define required documentation and report-row updates for Windows promotion.
6. Write the Windows evidence gate template.

### Deliverables
- Windows staged-test promotion gate
- Windows install-validation parity gate
- expected-count and CTest policy notes
- required workflow/documentation/report updates
- Sprint 148 prerequisite checklist

### Completion Criteria
- Sprint 148 can implement against an explicit gate
- reviewed and supplemental Windows claims stay separate
- unpromoted Windows parity remains a non-claim

---

## Day 8: Corpus-Family Evidence Gate Template

**Title:** Corpus Gates
**Theme:** Define evidence gates for broader QR and partial-SVD maintained
corpus families
**Time estimate:** 12 hours

### Tasks
1. Define corpus-family row requirements for fixtures, generators, expected
   rows, proof owners, and optional data.
2. Define QR comparison semantics for rank, residual, nullspace, minimum-norm,
   and reorder/COLAMD families.
3. Define partial-SVD comparison semantics for values, projectors, residuals,
   convergence, sparse output, and fail-closed behavior.
4. Define report/oracle row requirements and generated-local boundaries.
5. Define validation commands for corpus-family promotion.
6. Write the corpus-family evidence gate template.

### Deliverables
- QR corpus-family gate
- partial-SVD corpus-family gate
- comparison semantics table
- oracle/report row requirements
- corpus validation checklist

### Completion Criteria
- Sprint 150 and Sprint 151 have reusable corpus gates
- raw-basis and raw-vector identity claims are excluded
- generated rows remain correctly classified

---

## Day 9: Generated Report Freshness Gate Template

**Title:** Freshness Gates
**Theme:** Define generated report freshness gates for selected claim-bearing
families
**Time estimate:** 12 hours

### Tasks
1. Inventory generated report families relevant to Epic 13 claims.
2. Decide which families can become required-generated checks and which remain
   advisory.
3. Define command, artifact path, manifest, commit, branch, platform, compiler,
   and configuration requirements.
4. Define failure semantics for missing, stale, advisory, skipped, and deferred
   rows.
5. Define CI artifact policy for generated outputs.
6. Write the generated freshness gate template.

### Deliverables
- selected generated-family list
- required-generated decision table
- freshness metadata requirements
- missing/stale/advisory semantics
- CI artifact policy draft

### Completion Criteria
- Sprint 152 has a concrete freshness implementation target
- advisory rows are not treated as pass evidence
- generated reports have clear source-control and CI boundaries

---

## Day 10: ABI And Package Evidence Gate Template

**Title:** ABI Gate
**Theme:** Define the shared-library ABI product-decision gate and package
proof requirements
**Time estimate:** 12 hours

### Tasks
1. Inventory static-first package proof and shared-library deferral boundaries.
2. Define public symbol, header, version, visibility, loader, and package
   metadata evidence needed for shared-library support.
3. Define rejection criteria for preserving static-first-only support.
4. Define downstream consumer and platform validation requirements.
5. Define documentation and non-claim updates for either decision.
6. Write the ABI/package evidence gate template.

### Deliverables
- shared-library implementation gate
- stronger deferral gate
- package metadata and loader proof checklist
- downstream consumer validation requirements
- Sprint 153 decision handoff

### Completion Criteria
- Sprint 153 can make a product-level decision
- shared-library support cannot be accidentally implied
- static-first support remains guarded if shared support is deferred

---

## Day 11: External Comparison Evidence Gate Template

**Title:** Comparison Gate
**Theme:** Define the evidence gate for the first narrow external comparison
study
**Time estimate:** 12 hours

### Tasks
1. Identify candidate comparison targets from QR and partial-SVD corpus work.
2. Define dependency pinning, optional-data, and skip/defer policy.
3. Define comparison row schema for library, version, fixture, metric,
   tolerance, platform, compiler, status, and caveat.
4. Define acceptable wording for narrow state-of-practice claims.
5. Define rejected wording for broad state-of-the-art and ecosystem parity.
6. Write the external comparison evidence gate template.

### Deliverables
- comparison target candidate list
- dependency and optional-data policy
- comparison row schema
- narrow claim wording rules
- Sprint 154 handoff requirements

### Completion Criteria
- Sprint 154 has a bounded comparison target
- comparison evidence cannot be widened into broad parity
- state-of-the-art remains rejected without direct evidence

---

## Day 12: Quality Surface Map

**Title:** Quality Map
**Theme:** Map required validation for each likely Epic 13 touched surface
**Time estimate:** 12 hours

### Tasks
1. Map validation required for `.c`, `.h`, scripts, Makefile, CMake, CI,
   package, corpus, report, docs, benchmarks, and generated artifacts.
2. Define when `make format && make lint && make test` is required.
3. Define supplemental checks for package, CMake install, report freshness,
   corpus oracle, Windows CMake, and external comparison.
4. Define stop conditions for failing checks or unclear review feedback.
5. Define final sprint validation package expectations.
6. Write the quality surface map artifact.

### Deliverables
- touched-surface validation table
- full C gate trigger rules
- supplemental check map
- stop-condition register
- Sprint 156 validation package seed

### Completion Criteria
- every planned touched surface has a validation owner
- C/header changes require the full quality gate
- generated and hosted evidence requirements stay separate

---

## Day 13: Public Claim Freeze Audit

**Title:** Claim Freeze
**Theme:** Re-audit public and support surfaces for unsupported widened claims
before implementation begins
**Time estimate:** 11 hours

### Tasks
1. Scan README, INSTALL, benchmark docs, maintainer guide, solver-selection,
   cookbook, tutorial, and selected headers.
2. Identify wording related to state-of-the-art, external parity,
   shared-library ABI, package-manager support, Windows parity, performance,
   and generated report freshness.
3. Classify findings as supported, explicit non-claim, needs fix, or residual.
4. Apply documentation fixes only if a clear unsupported claim exists.
5. Record the public claim freeze result and handoff warnings.
6. Write the claim freeze audit artifact.

### Deliverables
- public/support claim scan log
- supported and non-claim classification table
- wording fix list or no-fix rationale
- implementation-sprint claim warnings
- Sprint 148-156 wording baseline

### Completion Criteria
- no unsupported claim is allowed to enter implementation sprints
- any wording fix is evidence-backed
- explicit non-claims remain visible

---

## Day 14: Sprint Closeout And Windows Handoff

**Title:** Closeout Handoff
**Theme:** Publish Sprint 147 closeout artifacts and prepare Sprint 148
Windows portability prerequisites
**Time estimate:** 11 hours

### Tasks
1. Review all Sprint 147 artifacts and working notes for consistency.
2. Confirm all Sprint 147 project-plan items are complete or explicitly
   deferred.
3. Publish the Sprint 148 Windows staged-test prerequisite checklist.
4. Publish the final selected-gap, claim target, evidence gate, and quality map
   indexes.
5. Run lightweight documentation validation for Sprint 147 artifacts.
6. Prepare Sprint 147 retrospective input notes.

### Deliverables
- Sprint 147 closeout artifact
- Sprint 148 Windows prerequisite checklist
- artifact index and handoff map
- retrospective input notes
- validation summary

### Completion Criteria
- Sprint 147 deliverables are complete
- Sprint 148 can begin without reopening baseline decisions
- documentation validation passes
- residuals and non-goals remain explicit
