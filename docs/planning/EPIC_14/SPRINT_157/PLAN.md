# Sprint 157 Plan: Epic 14 Baseline, Evidence Freeze & Claim Targets

**Sprint Duration:** 14 days
**Goal:** Freeze the post-Epic-13 baseline and select only the complete-gap
targets Epic 14 will attempt to close. This sprint implements the Sprint 157
section of `docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Starting Point:** Sprint 157 begins from:
- Epic 13 retrospective and residual queue available;
- Epic 14 review and gap-closure todo available;
- current `master` CI baseline available for Linux, macOS, and Windows;
- generated-report, corpus, comparison, package, and API reference policies
  documented;
- public claims remain bounded by reviewed evidence and explicit non-claims.

The sprint must:
- capture the current source, header, test, script, benchmark, example, docs,
  corpus, generated report, and package baseline;
- convert Epic 13 residuals into selected Epic 14 targets, long-horizon
  deferrals, and explicit non-goals;
- define evidence contracts for API docs, hosted reports, comparison rows,
  Windows package decisions, performance reports, and claim audits;
- publish a quality surface map for docs, scripts, C/header, build-system,
  package, CI, and generated artifacts;
- keep unsupported state-of-the-art, broad external parity, performance,
  package-manager, shared-library, dynamic ABI, and broad Windows claims out of
  scope;
- leave Sprint 158 with a clean generated API reference publication handoff.

**End State:** Sprint 157 leaves behind:
- Epic 14 baseline inventory;
- selected target and non-goal register;
- evidence contract templates;
- quality surface map;
- claim target register;
- risk and handoff artifacts;
- Sprint 158 API-doc handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 157 project-plan estimate.

---

## Day 1: Sprint Intake And Artifact Structure

**Title:** Sprint Intake
**Theme:** Establish Sprint 157 scope, artifact layout, and evidence sources
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 157 section of
   `docs/planning/EPIC_14/PROJECT_PLAN.md`.
2. Review the Epic 14 code review and gap-closure todo.
3. Create Sprint 157 working notes and artifact directory structure.
4. Record the exact branch, commit, and current `master` synchronization
   state.
5. Identify source artifacts needed for baseline, residual selection, evidence
   contracts, quality mapping, and Sprint 158 handoff.
6. Write the Day 1 sprint-intake artifact.

### Deliverables
- Sprint 157 working-notes baseline
- artifact directory structure
- source artifact index
- initial risk and assumption list
- Day 1 sprint-intake artifact

### Completion Criteria
- Sprint 157 scope is tied to the Epic 14 project plan
- all required planning inputs are identified
- unsupported broad claims are blocked before inventory work begins

---

## Day 2: Code And Public Surface Inventory

**Title:** Surface Inventory
**Theme:** Capture source, public header, example, benchmark, and script shape
**Time estimate:** 12 hours

### Tasks
1. Inventory `src/`, `include/`, `tests/`, `benchmarks/`, `examples/`, and
   `scripts/`.
2. Capture file counts, line-count hotspots, and largest owner files.
3. Identify public headers and installed-header surfaces.
4. Identify build-system source lists and consistency checks that protect the
   source inventory.
5. Record maintainability risks from large test owners or duplicated source
   declarations.
6. Write the Day 2 code/public-surface inventory artifact.

### Deliverables
- source and public-header inventory
- largest-file and ownership hotspot list
- build-system source-list risk notes
- Day 2 code/public-surface artifact

### Completion Criteria
- current implementation and public surface are captured with concrete paths
- maintainability risks are documented without proposing unrelated rewrites
- later evidence contracts can reference exact owner surfaces

---

## Day 3: Test And CI Baseline Inventory

**Title:** Test Baseline
**Theme:** Freeze current local, hosted, reviewed, supplemental, and staged
validation surfaces
**Time estimate:** 12 hours

### Tasks
1. Inventory C test targets, script tests, corpus tests, install tests,
   sanitizer paths, dead-code checks, and benchmark checks.
2. Review Linux, macOS, and Windows workflow lane names and support-tier
   wording.
3. Record Windows CTest expected count and Windows CMake-first boundaries.
4. Identify generated-report, coverage, benchmark, and dead-code artifacts that
   are advisory or local-only.
5. Map current validation commands to docs-only, script, C/header,
   build-system, package, and CI changes.
6. Write the Day 3 test/CI baseline artifact.

### Deliverables
- test target inventory
- CI support-tier baseline
- Windows reviewed-surface snapshot
- validation command matrix draft
- Day 3 test/CI baseline artifact

### Completion Criteria
- reviewed and supplemental lanes are not conflated
- Windows package and platform non-claims remain visible
- quality mapping has enough detail for later sprint work

---

## Day 4: Documentation And Claim Baseline

**Title:** Claim Baseline
**Theme:** Inventory public documentation, support-tier wording, and current
claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Review README, INSTALL, API reference, tutorial, cookbook, solver-selection,
   benchmark, maintainer, corpus, and example docs.
2. Capture current positive claims and explicit non-claims.
3. Identify where generated API HTML, hosted reports, comparison breadth,
   Windows package parity, performance, and ABI language is owned.
4. Scan for unsupported state-of-the-art, broad external parity, package,
   platform, performance, shared-library, or dynamic ABI wording.
5. Record documentation coherence risks and duplicate ownership boundaries.
6. Write the Day 4 documentation/claim baseline artifact.

### Deliverables
- public documentation inventory
- positive claim and non-claim register draft
- support-tier ownership map
- Day 4 documentation/claim artifact

### Completion Criteria
- every public claim category has a source document owner
- unsupported broad claims are either absent or recorded as defects
- documentation ownership is clear enough for Sprint 158 handoff work

---

## Day 5: Generated Artifact Baseline

**Title:** Generated Baseline
**Theme:** Freeze current generated docs, corpus, oracle, comparison, report,
coverage, benchmark, and dead-code surfaces
**Time estimate:** 12 hours

### Tasks
1. Inventory generated API docs policy and current `docs/api/html/` tracking
   state.
2. Inventory corpus manifests, expected rows, report families, oracle outputs,
   and freshness commands.
3. Inventory comparison harness outputs and selected comparison report
   semantics.
4. Inventory benchmark, sentinel, large-matrix, coverage, and dead-code report
   surfaces.
5. Classify each generated family as reviewed, hosted, local-only,
   supplemental, advisory, or deferred.
6. Write the Day 5 generated-artifact baseline artifact.

### Deliverables
- generated artifact inventory
- support-tier classification table
- freshness command list
- source-controlled vs ignored-output decision list
- Day 5 generated baseline artifact

### Completion Criteria
- generated output cannot be mistaken for source-controlled pass evidence
- selected local-only promotion candidates are visible
- Sprint 158 generated API docs work has a concrete baseline

---

## Day 6: Package, ABI, And Platform Baseline

**Title:** Package Baseline
**Theme:** Capture static-first install/export, Windows package boundaries,
and ABI non-claims
**Time estimate:** 12 hours

### Tasks
1. Review Make install, CMake install/export, `pkg-config`, and static-first
   deferral proof surfaces.
2. Record Linux and macOS reviewed static-first package proofs.
3. Record Windows CMake install/downstream validation and remaining Windows
   Makefile/`pkg-config` non-claims.
4. Inventory shared-library rejection, dynamic ABI blockers, package-manager
   deferrals, and runtime-loader non-claims.
5. Identify package metadata and docs that must stay synchronized.
6. Write the Day 6 package/ABI/platform baseline artifact.

### Deliverables
- static-first package baseline
- Windows package parity delta list
- shared-library and dynamic ABI blocker list
- package metadata ownership map
- Day 6 package baseline artifact

### Completion Criteria
- static-first support is distinguished from shared-library or ABI support
- Windows CMake validation is not conflated with Makefile or `pkg-config`
  parity
- Sprint 162 package decision prerequisites are recorded

---

## Day 7: Residual Consolidation

**Title:** Residual Consolidation
**Theme:** Convert Epic 13 residuals and Epic 14 review findings into a single
claim-oriented backlog
**Time estimate:** 12 hours

### Tasks
1. Review the Epic 13 residual queue and Epic 14 review gaps.
2. Merge duplicate residuals by claim surface rather than by source sprint.
3. Assign each residual to documentation, generated evidence, comparison,
   platform/package, API, performance, ABI, or long-horizon product category.
4. Record owner role, blocker, prerequisite, and promotion gate for each
   residual.
5. Identify residuals suitable for complete closure during Epic 14.
6. Write the Day 7 residual-consolidation artifact.

### Deliverables
- consolidated residual register
- owner/blocker/prerequisite/promotion-gate table
- complete-closure candidate shortlist
- Day 7 residual consolidation artifact

### Completion Criteria
- duplicate residuals are consolidated
- each residual has a concrete promotion gate or retained non-claim
- complete-closure candidates are separated from long-horizon work

---

## Day 8: Epic 14 Target Selection

**Title:** Target Selection
**Theme:** Select Epic 14 complete-gap targets and explicit non-goals
**Time estimate:** 12 hours

### Tasks
1. Score closure candidates by user value, proof cost, runtime cost, risk, and
   claim impact.
2. Select targets for generated API docs, hosted report promotion, QR
   comparison, partial-SVD comparison, Windows package decision, performance
   publication, header cleanup, and static-first boundary hardening.
3. Explicitly reject or defer broad state-of-the-art, broad ecosystem parity,
   package-manager, shared-library, dynamic ABI, and portable performance
   superiority goals.
4. Map selected targets to Sprints 158 through 166.
5. Write the Day 8 target-selection artifact.

### Deliverables
- selected Epic 14 target register
- explicit non-goal register
- target-to-sprint map
- Day 8 target-selection artifact

### Completion Criteria
- every selected target can end with a binary proof, artifact, or decision
- long-horizon deferrals cannot be mistaken for selected work
- Sprint 158 through Sprint 166 scopes remain coherent

---

## Day 9: Evidence Contract Templates

**Title:** Evidence Contracts
**Theme:** Define reusable proof templates for selected Epic 14 work
**Time estimate:** 12 hours

### Tasks
1. Create an API documentation publication evidence template.
2. Create a hosted generated-report promotion evidence template.
3. Create QR and partial-SVD comparison evidence templates.
4. Create a Windows package parity decision evidence template.
5. Create performance publication and public-header declaration-preservation
   evidence templates.
6. Write the Day 9 evidence-contract artifact.

### Deliverables
- API docs evidence template
- hosted report promotion template
- comparison evidence templates
- Windows package decision template
- performance and header cleanup templates
- Day 9 evidence-contract artifact

### Completion Criteria
- each selected target has a reusable evidence format
- templates distinguish pass evidence from advisory output
- later sprint artifacts can be compared consistently

---

## Day 10: Quality Surface Map

**Title:** Quality Map
**Theme:** Define validation commands by change type and evidence surface
**Time estimate:** 12 hours

### Tasks
1. Map documentation-only changes to whitespace, link, and claim-scan checks.
2. Map script changes to targeted Python or shell tests and generated-report
   freshness checks.
3. Map C/header changes to `make format && make lint && make test`.
4. Map build-system and package changes to CMake, install/export, static
   deferral, and downstream consumer checks.
5. Map CI changes to lane-name, support-tier, expected-count, artifact, and
   hosted-log reconciliation.
6. Write the Day 10 quality-surface artifact.

### Deliverables
- change-type validation matrix
- package/build-system quality map
- CI reconciliation checklist
- Day 10 quality map artifact

### Completion Criteria
- validation expectations are explicit before implementation sprints begin
- C/header quality gates are separated from documentation-only work
- package and CI changes have focused supplemental checks

---

## Day 11: Claim Target Register

**Title:** Claim Register
**Theme:** Publish accepted Epic 14 claims, rejected broad claims, and evidence
owners
**Time estimate:** 12 hours

### Tasks
1. Convert selected targets into precise claim statements.
2. Tie each accepted claim to an evidence owner, command, artifact, or CI lane.
3. Record rejected claims for state-of-the-art status, external parity,
   portable performance, package-manager support, shared-library support,
   dynamic ABI, runtime-loader behavior, and broad Windows parity.
4. Identify docs that must be updated when each claim is earned or rejected.
5. Write the Day 11 claim-target-register artifact.

### Deliverables
- accepted claim register
- explicit non-claim register
- evidence-owner table
- docs update checklist
- Day 11 claim register artifact

### Completion Criteria
- every accepted claim has recurring evidence or a planned proof owner
- rejected claims are stated clearly enough for later audits
- public docs have known ownership for claim wording

---

## Day 12: Risk Register And Sprint 158 Handoff Draft

**Title:** Risk Handoff
**Theme:** Identify implementation risks and prepare the generated API docs
handoff
**Time estimate:** 12 hours

### Tasks
1. Consolidate risks from baseline, residual selection, evidence contracts,
   quality mapping, and claim register work.
2. Prioritize risks by likelihood, impact, and mitigation path.
3. Draft the Sprint 158 generated API HTML publication handoff.
4. Identify Doxygen warning, generated-output tracking, public-header coverage,
   and source-header-first wording risks.
5. Define Day 1 prerequisites for Sprint 158.
6. Write the Day 12 risk/handoff artifact.

### Deliverables
- Epic 14 risk register
- mitigation and stop-condition list
- Sprint 158 handoff draft
- Day 12 risk handoff artifact

### Completion Criteria
- Sprint 158 can start without rediscovering generated API docs scope
- risks have owners, mitigations, or explicit deferrals
- API docs work is bounded by source-header-first policy

---

## Day 13: Baseline Reconciliation

**Title:** Baseline Reconciliation
**Theme:** Reconcile all Sprint 157 artifacts against the Epic 14 project plan
**Time estimate:** 11 hours

### Tasks
1. Review Days 1 through 12 artifacts for contradictions or missing links.
2. Reconcile selected targets with Epic 14 project-plan sprint scopes.
3. Reconcile quality gates with expected touched surfaces.
4. Reconcile claim register with public documentation owners.
5. Update working notes and residuals for any narrowed or deferred work.
6. Write the Day 13 reconciliation artifact.

### Deliverables
- reconciled Sprint 157 artifact index
- target-to-sprint reconciliation table
- residual and deferral updates
- Day 13 reconciliation artifact

### Completion Criteria
- Sprint 157 artifacts agree with each other
- every selected target maps to a later sprint or explicit deferral
- no unsupported claim category was introduced

---

## Day 14: Sprint Closeout And Sprint 158 Handoff

**Title:** Closeout Handoff
**Theme:** Finalize Sprint 157 evidence, closeout notes, and generated API docs
handoff
**Time estimate:** 11 hours

### Tasks
1. Finalize Sprint 157 working notes and artifact index.
2. Confirm baseline inventory, target register, evidence contracts, quality
   map, claim register, and risk register are complete.
3. Finalize the Sprint 158 generated API docs handoff.
4. Run documentation-only validation for the planning artifacts.
5. Record final residuals and open questions.
6. Write the Day 14 closeout artifact.

### Deliverables
- final Sprint 157 artifact index
- completed Sprint 158 handoff
- final residual and open-question list
- Day 14 closeout artifact
- completed Sprint 157 plan status

### Completion Criteria
- all Sprint 157 project-plan items have a completed artifact or explicit
  residual
- Sprint 158 can begin from a concrete generated API docs handoff
- documentation validation passes
