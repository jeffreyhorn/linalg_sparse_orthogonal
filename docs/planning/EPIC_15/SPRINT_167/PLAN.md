# Sprint 167 Plan: Epic 15 Baseline, Evidence Ledger & Claim Gate

**Sprint Duration:** 14 days
**Goal:** Establish the Epic 15 baseline and define exact evidence gates for
performance, ABI, package, API, comparison, and platform claims. This sprint
implements the Sprint 167 section of
`docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md`, but the active merged Sprint 167
project-plan section lives in `docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Starting Point:** Sprint 167 begins from:
- Epic 14 closeout and PR #184 landed on `master`;
- Epic 15 review and roadmap artifacts created under
  `docs/planning/EPIC_15/`;
- existing generated reports, install validation, package metadata, claim
  documentation, and CI evidence surfaces;
- retained non-claims for unqualified state-of-the-art status, broad
  external-library parity, portable performance superiority, shared-library
  support, dynamic ABI stability, package-manager distribution, and broad
  platform parity.

The sprint must:
- extract the unresolved residual queue from Epic 13 and Epic 14;
- inventory source, header, test, script, generated report, package, install,
  and CI evidence surfaces;
- create an Epic 15 evidence ledger that separates supported, partially
  supported, local-only, hosted-only, advisory, and unsupported claims;
- select the exact Epic 15 gaps to close and define completion gates for each;
- establish Sprint 167 working notes and artifacts for later implementation
  sprints;
- run lightweight documentation and repository sanity checks for the planning
  artifacts.

**End State:** Sprint 167 leaves behind:
- an Epic 15 evidence ledger;
- a residual queue audit;
- a CI and report inventory;
- claim gate criteria for the rest of Epic 15;
- Sprint 167 working notes and artifacts;
- a validation record for the new planning artifacts.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 167 project-plan estimate.

---

## Day 1: Sprint Intake And Artifact Setup

**Title:** Sprint Intake
**Theme:** Establish Sprint 167 scope, artifact layout, and source-of-truth
planning references
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 167 section of
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.
2. Review the Epic 15 review and todo artifacts.
3. Create Sprint 167 working notes and artifact directory structure.
4. Record the active source artifact path and the prompt path mismatch.
5. Define evidence categories for claims, non-claims, reports, CI, package,
   API, platform, and performance.
6. Write the Day 1 sprint-intake artifact.

### Deliverables
- Sprint 167 working-notes baseline
- artifact directory structure
- source artifact note
- evidence category list
- Day 1 sprint-intake artifact

### Completion Criteria
- Sprint 167 scope is tied to the active Epic 15 project plan
- artifact locations are created and named consistently
- evidence categories are defined before audit work begins

---

## Day 2: Prior Epic Residual Audit

**Title:** Residual Audit I
**Theme:** Extract unresolved residuals from Epic 13 and Epic 14 closeouts
**Time estimate:** 12 hours

### Tasks
1. Review `docs/planning/EPIC_13/EPIC_13_RETROSPECTIVE.md`.
2. Review `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md`.
3. Extract deferred work, retained non-claims, and recommended next-epic
   candidates.
4. Classify each residual by source epic, affected surface, and evidence type.
5. Identify residuals already addressed by later merged work.
6. Write the Day 2 residual-audit artifact.

### Deliverables
- Epic 13 residual extraction
- Epic 14 residual extraction
- duplicate or resolved residual notes
- residual source map
- Day 2 residual-audit artifact

### Completion Criteria
- prior-epic residuals are listed with source references
- resolved and still-open residuals are separated
- residuals are ready for risk and value classification

---

## Day 3: Residual Risk And Value Classification

**Title:** Residual Audit II
**Theme:** Rank residuals by claim risk, user value, and closure feasibility
**Time estimate:** 12 hours

### Tasks
1. Assign claim-risk levels to each open residual.
2. Assign user-value levels based on adoption, correctness, package, platform,
   or documentation impact.
3. Estimate closure feasibility within one or more Epic 15 sprints.
4. Identify dependencies among performance, ABI, package, API, comparison, and
   platform residuals.
5. Produce a ranked residual queue for Epic 15 selection.
6. Write the Day 3 residual-classification artifact.

### Deliverables
- ranked residual queue
- claim-risk classification
- user-value classification
- closure-feasibility notes
- Day 3 residual-classification artifact

### Completion Criteria
- every open residual has risk, value, and feasibility labels
- dependency relationships are explicit
- the highest-value closeable gaps are visible

---

## Day 4: Source And Header Surface Inventory

**Title:** Code Surface Inventory
**Theme:** Inventory implementation and public-header surfaces relevant to
Epic 15 gaps
**Time estimate:** 12 hours

### Tasks
1. Inventory `src/` implementation files and identify large or high-risk
   solver families.
2. Inventory `include/` public headers and identify remaining coherence
   cleanup candidates.
3. Map implementation families to public headers and examples where practical.
4. Identify allocation-heavy subsystems that could support failure-path
   evidence work.
5. Record source/header surfaces that should trigger `make format`,
   `make lint`, and `make test` in later sprints.
6. Write the Day 4 source-header-inventory artifact.

### Deliverables
- source implementation inventory
- public-header inventory
- high-risk solver family notes
- allocation-failure candidate list
- Day 4 source-header-inventory artifact

### Completion Criteria
- source and header surfaces are mapped to Epic 15 gap candidates
- public-header cleanup candidates are concrete
- allocation-failure candidates are bounded enough for future sprint work

---

## Day 5: Test And Corpus Surface Inventory

**Title:** Test Surface Inventory
**Theme:** Inventory test, corpus, oracle, and comparison evidence surfaces
**Time estimate:** 12 hours

### Tasks
1. Inventory `tests/` coverage by solver family and platform scope.
2. Review maintained corpus manifests and generated report families.
3. Map oracle and external comparison scripts to report outputs and freshness
   checks.
4. Identify one or more candidate comparison families for Epic 15 completion.
5. Record which tests are local, hosted, advisory, platform-specific, or
   excluded.
6. Write the Day 5 test-corpus-inventory artifact.

### Deliverables
- test coverage inventory
- corpus and oracle surface map
- comparison-family candidate list
- platform/test-scope classification
- Day 5 test-corpus-inventory artifact

### Completion Criteria
- test and corpus evidence is grouped by solver family
- hosted and local-only proof surfaces are distinguished
- comparison candidates are ready for selection

---

## Day 6: CI And Workflow Inventory

**Title:** CI Inventory
**Theme:** Inventory hosted workflows, supplemental lanes, and platform tiers
**Time estimate:** 12 hours

### Tasks
1. Review GitHub Actions workflow files and expected platform lanes.
2. Map Linux, macOS, and Windows checks to the claims they support.
3. Identify generated report, package, install, performance, and comparison
   freshness gates.
4. Record local-only checks that do not have hosted CI proof.
5. Identify brittle count, path, shell, or platform assumptions in CI wording.
6. Write the Day 6 CI-inventory artifact.

### Deliverables
- hosted workflow inventory
- platform-tier map
- local-only versus hosted evidence table
- CI brittleness notes
- Day 6 CI-inventory artifact

### Completion Criteria
- hosted CI evidence is mapped to specific claims
- platform tier boundaries are explicit
- CI gaps are ready for Epic 15 selection

---

## Day 7: Package And Install Evidence Inventory

**Title:** Package Inventory
**Theme:** Inventory static-first install, pkg-config, CMake package, ABI, and
package-manager evidence
**Time estimate:** 12 hours

### Tasks
1. Review static-library install and uninstall behavior in Make and CMake.
2. Review CMake package metadata and pkg-config metadata claims.
3. Inventory install validation scripts and downstream consumer examples.
4. Record shared-library, dynamic ABI, runtime-loader, and package-manager
   non-claims.
5. Identify the package-manager readiness decision candidates for Epic 15.
6. Write the Day 7 package-install-inventory artifact.

### Deliverables
- static-first package evidence map
- install validation inventory
- ABI and package-manager non-claim register
- package decision candidate list
- Day 7 package-install-inventory artifact

### Completion Criteria
- supported package behavior is separated from unsupported package claims
- install validation owners are source-backed
- package-manager decision candidates are concrete

---

## Day 8: Documentation And Claim Surface Inventory

**Title:** Claim Surface Inventory
**Theme:** Inventory README, generated docs, report indexes, and claim wording
**Time estimate:** 12 hours

### Tasks
1. Review README claim and non-claim sections.
2. Review install, maintainer, generated report, API, and planning indexes.
3. Identify state-of-the-art, performance, package, ABI, platform, generated
   API, and external-parity wording that needs evidence mapping.
4. Record documentation surfaces that are authoritative versus historical.
5. Identify stale or ambiguous links that could affect Epic 15 claim gates.
6. Write the Day 8 documentation-claim-inventory artifact.

### Deliverables
- documentation surface inventory
- authoritative claim source map
- stale or ambiguous wording notes
- claim wording candidate list
- Day 8 documentation-claim-inventory artifact

### Completion Criteria
- public claim wording has named source files
- historical planning artifacts are separated from current user-facing docs
- stale or ambiguous claim surfaces are ready for cleanup planning

---

## Day 9: Evidence Ledger Draft

**Title:** Ledger Draft
**Theme:** Build the first Epic 15 evidence ledger from inventories
**Time estimate:** 12 hours

### Tasks
1. Create the evidence ledger artifact.
2. Add claim rows for solver correctness, generated reports, package install,
   public API, performance, external comparison, and platform support.
3. Classify each row as supported, partially supported, unsupported, local-only,
   hosted-only, advisory, or deferred.
4. Attach source files, commands, reports, and CI lanes to each row.
5. Identify missing evidence links or unclear owners.
6. Write the Day 9 ledger-draft artifact.

### Deliverables
- draft Epic 15 evidence ledger
- claim support classification table
- missing evidence register
- evidence owner notes
- Day 9 ledger-draft artifact

### Completion Criteria
- all major claim categories have ledger rows
- every row has a status and at least one evidence reference or explicit gap
- missing evidence is visible before gap selection

---

## Day 10: Evidence Ledger Review And Corrections

**Title:** Ledger Review
**Theme:** Reconcile ledger rows against source docs, commands, and current
non-claims
**Time estimate:** 12 hours

### Tasks
1. Review ledger rows against README, install docs, report indexes, and
   project-plan language.
2. Correct unsupported or over-broad claim classifications.
3. Add explicit non-claim rows for unqualified state-of-the-art status,
   broad external-library parity, portable performance superiority,
   shared-library support, dynamic ABI stability, package-manager
   distribution, and broad platform parity.
4. Confirm each row has a clear future sprint owner or retained deferral.
5. Write the Day 10 ledger-review artifact.

### Deliverables
- reviewed Epic 15 evidence ledger
- non-claim ledger rows
- correction notes
- future owner or deferral labels
- Day 10 ledger-review artifact

### Completion Criteria
- ledger language does not overstate evidence
- unsupported claims are explicit non-claims
- every high-risk row has an owner or retained deferral

---

## Day 11: Gap Selection Gate

**Title:** Gap Selection
**Theme:** Select the exact Epic 15 gaps to close based on ledger evidence
**Time estimate:** 12 hours

### Tasks
1. Compare ranked residuals against the evidence ledger.
2. Select the Epic 15 closure targets for performance publication, ABI
   decision, package-manager readiness, public headers, generated API,
   external comparison, cross-platform freshness, and allocation-failure
   evidence.
3. Define why lower-priority residuals are deferred.
4. Map each selected gap to a future Sprint 168-176 owner.
5. Write the Day 11 gap-selection artifact.

### Deliverables
- selected Epic 15 gap list
- deferred residual list
- sprint ownership map
- gap-selection rationale
- Day 11 gap-selection artifact

### Completion Criteria
- Epic 15 closure targets are explicit and finite
- each selected gap has a future sprint owner
- deferred gaps are documented rather than hidden

---

## Day 12: Acceptance Criteria And Stop Conditions

**Title:** Claim Gates
**Theme:** Define completion criteria, validation commands, and stop
conditions for selected gaps
**Time estimate:** 12 hours

### Tasks
1. Define acceptance criteria for each selected Epic 15 gap.
2. Define required validation commands or CI evidence for each gap.
3. Define stop conditions for unclear evidence, failing checks, unsupported
   platform behavior, and over-broad claim wording.
4. Define artifact templates for implementation sprint handoffs.
5. Update the evidence ledger with acceptance criteria links.
6. Write the Day 12 claim-gates artifact.

### Deliverables
- acceptance criteria table
- validation command map
- stop-condition register
- implementation handoff template
- Day 12 claim-gates artifact

### Completion Criteria
- every selected gap has objective completion criteria
- future sprints know which checks must pass
- stop conditions prevent accidental claim drift

---

## Day 13: Sprint 167 Reconciliation

**Title:** Sprint Reconciliation
**Theme:** Reconcile Sprint 167 artifacts and prepare handoff to Sprint 168
**Time estimate:** 11 hours

### Tasks
1. Review all Sprint 167 artifacts for consistency and missing links.
2. Reconcile the evidence ledger, residual queue, CI inventory, package
   inventory, documentation inventory, and gap-selection artifacts.
3. Prepare a Sprint 168 handoff focused on hosted performance publication.
4. Record known residuals and any evidence that needs hosted PR confirmation.
5. Write the Day 13 reconciliation artifact.

### Deliverables
- reconciled Sprint 167 artifact set
- Sprint 168 handoff notes
- open question and residual register
- Day 13 reconciliation artifact

### Completion Criteria
- Sprint 167 artifacts agree with one another
- Sprint 168 has clear prerequisites and evidence inputs
- open residuals are explicit

---

## Day 14: Final Validation And Closeout

**Title:** Sprint Closeout
**Theme:** Validate Sprint 167 planning artifacts and publish the baseline for
Epic 15 execution
**Time estimate:** 11 hours

### Tasks
1. Run documentation consistency checks selected during the sprint.
2. Run lightweight repository sanity checks for changed planning artifacts.
3. Confirm no code files changed unless intentionally required.
4. Record validation output and any skipped checks with reasons.
5. Finalize Sprint 167 working notes and closeout summary.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables
- final Sprint 167 validation record
- finalized evidence ledger
- finalized gap-selection and claim-gate artifacts
- Sprint 167 closeout summary
- Day 14 sprint-closeout artifact

### Completion Criteria
- planning artifacts pass lightweight validation
- Sprint 167 deliverables match the Epic 15 project-plan items
- Sprint 168 can begin with a clear hosted performance evidence target
