# Sprint 137 Plan: Epic 12 Baseline, Gap Selection & Evidence Contract

**Sprint Duration:** 14 days
**Goal:** Freeze the post-Epic-11 baseline, select the gaps Epic 12 will close
completely, and create the evidence contracts that govern implementation. This
sprint implements the Sprint 137 section of
`docs/planning/EPIC_12/PROJECT_PLAN.md`.

**Starting Point:** Sprint 137 begins from:
- merged Epic 11 closeout and PR #151
- the Epic 11 retrospective and Sprint 136 residual queue
- the Epic 12 Codex review and gap-closure todo
- a repository with broad solver coverage, static-first packaging, explicit
  platform tiers, and remaining corpus, QR, partial-SVD, report, runtime,
  package/ABI, platform, and adoption gaps

The sprint must:
- capture objective post-Epic-11 source, test, package, CI, report, and
  support-tier metrics
- convert Epic 11 residuals into Epic 12 owners, dependencies, duplicate
  fences, and non-goals
- decide which gaps Epic 12 can close completely
- create reusable evidence contracts for corpus fixtures, oracle rows, report
  indexes, stale-report checks, package/ABI decisions, platform promotion, and
  claim gates
- map validation requirements by touched surface
- freeze public claims before later sprints widen any product wording

**End State:** Sprint 137 leaves behind:
- a post-Epic-11 baseline package
- an Epic 12 residual owner and non-goal map
- an Epic 12 gap-selection decision
- evidence contract templates for Sprints 138-146
- a quality surface map
- a public claim freeze artifact
- Sprint 138 corpus-architecture handoff requirements

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 137 project-plan estimate.

---

## Day 1: Sprint 137 Scope & Artifact Setup

**Title:** Scope Setup
**Theme:** Convert the Sprint 137 project-plan section into a bounded
execution package before baseline collection starts
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 137 section of
   `docs/planning/EPIC_12/PROJECT_PLAN.md`.
2. Re-read the Epic 12 review and gap-closure todo.
3. Re-read the Epic 11 retrospective and Sprint 136 residual queue.
4. Create Sprint 137 working notes and artifact directory structure.
5. Map Sprint 137 Items 1-7 to day-level owners.
6. Record initial validation expectations for documentation-only,
   script/report, build-system, CI, and `.c`/`.h` changes.

### Deliverables
- Sprint 137 working-notes baseline
- artifact directory structure
- item-to-day owner map
- inherited-input inventory
- initial validation expectation register

### Completion Criteria
- every Sprint 137 project-plan item has a day-level owner
- inherited Epic 11 and Epic 12 inputs are visible before decisions begin
- validation expectations are documented before any later implementation work

---

## Day 2: Source, Test & Maintainability Baseline

**Title:** Code Metrics
**Theme:** Capture objective source, test, benchmark, example, and
maintainability metrics for the post-Epic-11 codebase
**Time estimate:** 12 hours

### Tasks
1. Count implementation, public-header, private-header, test, benchmark, and
   example files.
2. Capture line-count totals for C and header surfaces.
3. Identify largest implementation files and largest proof-owner tests.
4. Record source-list and CMake source ownership signals.
5. Identify current giant-test and large-source risks relevant to Epic 12 gap
   closure.
6. Write the source/test baseline artifact.

### Deliverables
- source/test size baseline
- large-file hotspot table
- implementation/test ownership risk notes
- source-list and CMake ownership notes

### Completion Criteria
- baseline metrics are reproducible from named commands
- high-risk source and proof-owner files are ranked
- maintainability risks are tied to Epic 12 candidate gaps

---

## Day 3: Build, Package, CI & Report Baseline

**Title:** Proof Baseline
**Theme:** Capture current build, package, platform, CI, report, benchmark, and
support-tier evidence
**Time estimate:** 12 hours

### Tasks
1. Review Makefile, CMake, install, pkg-config, package, and static-deferral
   surfaces.
2. Review Linux, macOS, Windows, sanitizer, ThreadSanitizer, dead-code,
   coverage, benchmark, package, and install CI workflows.
3. Inventory current benchmark, sentinel, guardrail, coverage, dead-code, and
   package report artifacts.
4. Record reviewed, supplemental, staged, local-only, deferred, and
   unsupported support tiers.
5. Write the build/package/CI/report baseline artifact.

### Deliverables
- build and package proof map
- CI lane summary
- report-family inventory
- platform/support-tier baseline
- package and report non-claim notes

### Completion Criteria
- every current proof lane has an owner and support tier
- package and platform asymmetries are explicit
- generated reports are separated from public correctness or performance
  claims

---

## Day 4: Epic 11 Residual Intake

**Title:** Residual Intake
**Theme:** Convert Epic 11 residuals into a complete Epic 12 intake queue
without duplicating already-closed work
**Time estimate:** 12 hours

### Tasks
1. Extract residuals from the Epic 11 retrospective.
2. Extract residuals from the Sprint 136 residual publication artifact.
3. Review recent Sprint 130-136 retrospectives for carry-forward items.
4. Group residuals by QR, partial-SVD, corpus/report, runtime/backend,
   package/ABI, platform, Windows staged tests, adoption, and maintainability.
5. Mark duplicate, closed, obsolete, and candidate residuals.
6. Write the residual intake artifact.

### Deliverables
- Epic 11 residual intake table
- candidate/duplicate/closed/obsolete classifications
- initial residual grouping by Epic 12 workstream
- unresolved question list

### Completion Criteria
- all Epic 11 closeout residuals have been considered
- duplicates and already-closed items are fenced
- candidate residuals are grouped for owner assignment

---

## Day 5: Residual Owner & Non-Goal Map

**Title:** Owner Map
**Theme:** Assign candidate residuals to Epic 12 owners, dependencies,
promotion gates, and explicit non-goals
**Time estimate:** 12 hours

### Tasks
1. Assign each candidate residual to an Epic 12 workstream owner.
2. Define dependency order across corpus, QR, partial-SVD, report, runtime,
   package, platform, adoption, and closeout work.
3. Identify residuals that should remain non-goals for Epic 12.
4. Define promotion gates for residuals that may become future claims.
5. Record stop conditions for unclear or overbroad residuals.
6. Write the residual owner and non-goal map.

### Deliverables
- residual owner map
- dependency graph or ordered dependency list
- Epic 12 non-goal register
- promotion gate table
- stop-condition notes

### Completion Criteria
- each active residual has one owner workstream
- non-goals are explicit rather than hidden
- promotion gates require implementation, validation, and documentation proof

---

## Day 6: Gap-Selection Criteria

**Title:** Selection Rules
**Theme:** Define the decision criteria for choosing gaps Epic 12 can close
completely
**Time estimate:** 12 hours

### Tasks
1. Define completion criteria for a gap to count as closed.
2. Score candidate gaps by user value, state-of-the-art relevance, dependency
   risk, testability, platform risk, and documentation impact.
3. Define anti-goals for shallow partial progress.
4. Define claim gates for QR, partial-SVD, corpus/report, runtime/backend,
   package/ABI, platform, and adoption work.
5. Review feasibility against the 10-sprint Epic 12 budget.
6. Write the gap-selection criteria artifact.

### Deliverables
- gap-selection scoring rubric
- complete-closure definition
- anti-goal list
- claim gate matrix
- feasibility notes

### Completion Criteria
- gap selection can be justified from written criteria
- the rubric favors complete closure over broad partial work
- unsupported state-of-the-art expansion is blocked by explicit gates

---

## Day 7: Epic 12 Gap-Selection Decision

**Title:** Gap Decision
**Theme:** Select the specific gaps Sprint 138-146 will close, defer, or reject
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 6 rubric to the residual owner map.
2. Select the maintained numerical corpus/oracle gap for Sprint 138.
3. Select the priority QR residual for Sprint 139.
4. Select the priority partial-SVD residual for Sprint 140.
5. Select the report, runtime/backend, package/ABI, platform, and adoption
   gap-closure targets.
6. Write the Epic 12 gap-selection decision artifact.

### Deliverables
- selected gap list for Sprints 138-146
- explicit deferral/rejection list
- dependency-ordered sprint handoff notes
- claim-boundary notes for selected gaps

### Completion Criteria
- every later Epic 12 sprint has a selected gap target
- deferred and rejected gaps have reasons
- no selected gap depends on unearned or unavailable evidence

---

## Day 8: Corpus & Oracle Evidence Templates

**Title:** Corpus Templates
**Theme:** Define reusable evidence contracts for maintained corpus fixtures
and oracle rows
**Time estimate:** 12 hours

### Tasks
1. Design corpus fixture metadata fields.
2. Design deterministic generated-matrix metadata fields.
3. Design optional-data skip/defer fields.
4. Design oracle row fields for expected result, observed result, tolerance,
   support tier, command, fixture key, and source commit.
5. Define failure interpretation for oracle mismatches, unsupported fixtures,
   and unavailable external data.
6. Write the corpus/oracle evidence template artifact.

### Deliverables
- corpus fixture template
- generated-matrix template
- optional-data skip/defer template
- oracle row template
- oracle failure interpretation rules

### Completion Criteria
- Sprint 138 can implement the corpus lane without redefining row semantics
- skipped optional data cannot be mistaken for pass evidence
- oracle rows preserve fixture-local claim boundaries

---

## Day 9: Report Index & Freshness Templates

**Title:** Report Templates
**Theme:** Define report-index and stale-report contracts for maintained
generated evidence
**Time estimate:** 12 hours

### Tasks
1. Inventory row meanings for benchmark, sentinel, guardrail, coverage,
   dead-code, package, corpus, and oracle reports.
2. Define shared report metadata fields.
3. Define freshness fields for commit, command, platform, compiler,
   configuration, and generated time.
4. Define stale-report check expectations and failure semantics.
5. Identify report families that cannot safely be normalized yet.
6. Write the report index and freshness template artifact.

### Deliverables
- shared report metadata template
- stale-report template
- report-family normalization eligibility table
- report non-claim rules

### Completion Criteria
- Sprint 141 can implement report normalization from written templates
- row meanings are preserved rather than flattened
- stale-report checks do not imply release or performance proof

---

## Day 10: Package, ABI, Platform & Claim Templates

**Title:** Product Templates
**Theme:** Define decision templates for package/ABI, platform promotion, and
public claim gates
**Time estimate:** 12 hours

### Tasks
1. Define the package/ABI decision template for shared-library support versus
   static-first-only continuation.
2. Define required proof for CMake, Make install, pkg-config, downstream
   consumers, version constraints, loader behavior, and unsupported artifacts.
3. Define platform promotion criteria for Linux, macOS, Windows, and staged
   Windows test lanes.
4. Define public claim gate fields for evidence, support tier, docs updates,
   and non-claims.
5. Write the package/ABI/platform/claim template artifact.

### Deliverables
- package/ABI decision template
- downstream consumer proof template
- platform promotion template
- public claim gate template
- unsupported-artifact checklist

### Completion Criteria
- Sprint 143 can decide package/ABI direction without missing proof categories
- Sprint 144 can evaluate platform promotion without inference
- claim gates require docs and validation alongside implementation

---

## Day 11: Quality Surface Map

**Title:** Quality Map
**Theme:** Map required checks by touched surface for Epic 12 implementation
sprints
**Time estimate:** 12 hours

### Tasks
1. Map required checks for documentation-only changes.
2. Map required checks for script/report-generator changes.
3. Map required checks for Makefile, CMake, pkg-config, install, and CI
   changes.
4. Map required checks for `.c` and `.h` changes.
5. Map supplemental checks for benchmarks, coverage, dead-code, platform
   lanes, and generated reports.
6. Write the quality surface map artifact.

### Deliverables
- touched-surface quality matrix
- required command map
- supplemental command map
- hosted-CI dependency notes
- stop-condition list

### Completion Criteria
- later sprints can select validation from touched surfaces
- `.c`/`.h` changes clearly require the full C quality chain
- supplemental and hosted-CI-only checks are not treated as local proof

---

## Day 12: Public Claim Freeze

**Title:** Claim Freeze
**Theme:** Reconfirm public wording before Epic 12 implementation widens any
claim surface
**Time estimate:** 11 hours

### Tasks
1. Audit README, INSTALL, solver-selection, cookbook, tutorial, algorithm,
   benchmark, examples, and maintainer docs for current claims.
2. Check wording around state-of-the-art status, external parity, package/ABI,
   platform parity, runtime/backend behavior, performance, corpus evidence,
   and generated reports.
3. Identify wording that must remain frozen until later proof lands.
4. Record any unsupported wording that needs immediate cleanup.
5. Write the public claim freeze artifact.

### Deliverables
- public claim inventory
- frozen-claim register
- unsupported wording cleanup list
- non-claim register

### Completion Criteria
- current public wording is reconciled before implementation sprints begin
- unsupported claim cleanup is separated from future candidate claims
- state-of-the-art wording remains blocked unless proof exists

---

## Day 13: Handoff Synthesis & Sprint 138 Readiness

**Title:** Handoff Synthesis
**Theme:** Convert baseline, residual, selection, template, quality, and claim
artifacts into a Sprint 138-ready corpus handoff
**Time estimate:** 11 hours

### Tasks
1. Reconcile Day 2-12 artifacts for contradictions, duplicates, and missing
   owners.
2. Confirm selected gaps match Sprint 138-146 project-plan sequencing.
3. Convert corpus/oracle decisions into Sprint 138 prerequisites.
4. Convert report, runtime, package, platform, and adoption decisions into
   later-sprint handoff notes.
5. Write the handoff synthesis artifact.

### Deliverables
- reconciled Sprint 137 artifact index
- Sprint 138 corpus handoff
- later-sprint dependency notes
- contradiction and duplicate cleanup notes

### Completion Criteria
- Sprint 138 can begin without redoing baseline or gap selection
- all later sprint handoffs are dependency ordered
- contradictions across Sprint 137 artifacts are resolved or explicitly noted

---

## Day 14: Sprint 137 Closeout

**Title:** Closeout
**Theme:** Publish Sprint 137 completion evidence, residuals, and final
readiness criteria for Epic 12 implementation
**Time estimate:** 12 hours

### Tasks
1. Verify all Sprint 137 deliverables exist.
2. Validate markdown links, whitespace, and touched documentation paths.
3. Confirm whether any `.c` or `.h` files changed and select final validation
   accordingly.
4. Write Sprint 137 closeout notes and residual register.
5. Publish Sprint 138 readiness criteria.
6. Update working notes with final evidence and validation results.

### Deliverables
- final Sprint 137 deliverable checklist
- validation result summary
- residual register
- Sprint 138 readiness criteria
- completed working notes

### Completion Criteria
- all Sprint 137 project-plan deliverables are present or explicitly deferred
- validation matches touched surfaces
- Sprint 138 has clear prerequisites, inputs, and stop conditions
