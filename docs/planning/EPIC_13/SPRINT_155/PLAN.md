# Sprint 155 Plan: Tutorial, Header Cleanup & API Reference Coherence

**Sprint Duration:** 14 days
**Goal:** Close the remaining adoption and documentation gap by aligning the
tutorial, selected public headers, and API reference surfaces around the
current earned claims. This sprint implements the Sprint 155 section of
`docs/planning/EPIC_13/PROJECT_PLAN.md`.

**Starting Point:** Sprint 155 begins from:
- Sprint 145 first-use ladder available;
- Sprint 146 residuals R8 and R9 available;
- current earned claims from Sprints 148 through 154 known;
- current tutorial, README, cookbook, examples, solver-selection, package, and
  support-tier documentation available for reconciliation;
- public header cleanup must preserve declarations, signatures, install
  surface, static-first packaging, and ABI boundaries.

The sprint must:
- audit `docs/tutorial.md` against the current adoption, package,
  solver-selection, example, and support-tier surfaces;
- realign the tutorial around build, first solve, data input, solver choice,
  diagnostics, install, and advanced controls;
- select a high-impact public-header cleanup batch without signature or ABI
  drift;
- clarify ownership notes, error contracts, and claim boundaries in selected
  public headers;
- add API reference guidance or a generated-reference publication plan;
- run declaration-preservation evidence for public header edits;
- run docs checks, examples/install checks as needed, and full quality gates if
  `.c` or public `.h` files change;
- leave Sprint 156 a clean final-closeout handoff.

**End State:** Sprint 155 leaves behind:
- aligned tutorial;
- cleaned selected public headers;
- API reference publication plan;
- declaration-preservation evidence;
- updated maintainer guidance for future header cleanup;
- validation evidence and residual register;
- Sprint 156 closeout handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 155 project-plan estimate.

---

## Day 1: Sprint Intake And Documentation Baseline

**Title:** Intake Baseline
**Theme:** Establish Sprint 155 scope, artifact structure, earned-claim
boundaries, and documentation inventory
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 155 section of
   `docs/planning/EPIC_13/PROJECT_PLAN.md`.
2. Review Sprint 145, Sprint 146, and Sprints 148 through 154 handoff
   artifacts for adoption, claim, package, and report constraints.
3. Create Sprint 155 working notes and artifact directory structure.
4. Inventory `docs/tutorial.md`, README, cookbook, examples,
   solver-selection, package, support-tier, API, and maintainer documents.
5. Record current earned claims and explicit non-claims that tutorial and
   header cleanup must preserve.
6. Write the Day 1 baseline artifact.

### Deliverables
- Sprint 155 working-notes baseline
- artifact directory structure
- documentation inventory
- earned-claim and non-claim register
- Day 1 baseline artifact

### Completion Criteria
- Sprint 155 scope is tied to current earned evidence
- tutorial and header cleanup cannot widen unsupported claims
- follow-on audit work has a complete source list

---

## Day 2: Tutorial Audit

**Title:** Tutorial Audit
**Theme:** Audit `docs/tutorial.md` against current adoption, examples,
package, solver-selection, diagnostics, and support-tier surfaces
**Time estimate:** 12 hours

### Tasks
1. Compare tutorial build guidance against README, install, CMake, Make, and
   pkg-config documentation.
2. Compare first-solve and data-input guidance against maintained examples and
   cookbook workflows.
3. Compare solver-choice language against current solver-selection and
   support-tier claims.
4. Compare diagnostics and report references against generated report-index
   freshness policy.
5. Identify stale, duplicated, missing, or overbroad tutorial content.
6. Write the tutorial audit artifact with fix categories and priorities.

### Deliverables
- tutorial source-to-source comparison matrix
- stale or missing tutorial section list
- claim-risk findings
- tutorial rewrite backlog
- Day 2 tutorial audit artifact

### Completion Criteria
- every major tutorial section has a current-source comparison
- stale and overbroad tutorial claims are identified before rewriting
- the rewrite backlog is small enough to execute during Sprint 155

---

## Day 3: Tutorial Flow Design

**Title:** Tutorial Flow
**Theme:** Design the target tutorial structure around first-use workflow,
solver selection, diagnostics, install, and advanced controls
**Time estimate:** 12 hours

### Tasks
1. Define the tutorial reader path from build to first solve.
2. Define tutorial sections for data input, matrix formats, solver choice, and
   diagnostics.
3. Define where install/package guidance belongs without duplicating full
   package documentation.
4. Define advanced-control coverage and what belongs in API reference instead.
5. Map tutorial sections to maintained examples and cross-links.
6. Write the tutorial flow design artifact.

### Deliverables
- target tutorial outline
- section-to-source mapping
- example and cookbook cross-link plan
- advanced-control boundary notes
- Day 4 tutorial rewrite checklist

### Completion Criteria
- tutorial flow supports a new user without becoming an API dump
- package and support-tier claims remain delegated to authoritative docs
- rewrite work has concrete section ownership

---

## Day 4: Tutorial Rewrite Batch 1

**Title:** Tutorial Core
**Theme:** Rewrite the core tutorial path for build, first solve, data input,
and solver choice
**Time estimate:** 12 hours

### Tasks
1. Update tutorial build and include guidance against current static-first
   packaging.
2. Rewrite the first-solve path around maintained example behavior.
3. Add or revise data-input guidance for supported matrix workflows.
4. Align solver-choice guidance with current earned solver-selection claims.
5. Remove stale or duplicated content found during Day 2 audit.
6. Capture unresolved tutorial questions for Day 5.

### Deliverables
- updated core tutorial sections
- revised first-solve narrative
- data-input guidance updates
- solver-choice alignment notes
- unresolved issue list

### Completion Criteria
- tutorial core path compiles conceptually against maintained examples
- unsupported solver or package claims are not introduced
- remaining tutorial work is limited to diagnostics, install, and advanced
  controls

---

## Day 5: Tutorial Rewrite Batch 2

**Title:** Tutorial Finish
**Theme:** Finish tutorial diagnostics, install, advanced controls, cross-links,
and claim reconciliation
**Time estimate:** 12 hours

### Tasks
1. Add diagnostics guidance aligned with report and solver evidence.
2. Add install and downstream-consumer references without duplicating package
   contract details.
3. Add advanced-control guidance and links to API/reference surfaces.
4. Normalize tutorial cross-links to README, examples, cookbook, package, and
   support-tier docs.
5. Re-check tutorial language against earned claims and non-claims.
6. Write the tutorial alignment summary artifact.

### Deliverables
- finished tutorial rewrite
- diagnostics and install guidance updates
- advanced-control links
- cross-link reconciliation notes
- tutorial alignment summary

### Completion Criteria
- tutorial has a coherent end-to-end adoption path
- each advanced topic points to a maintained reference surface
- tutorial wording remains consistent with current evidence

---

## Day 6: Header Cleanup Selection

**Title:** Header Selection
**Theme:** Select high-impact public headers for documentation cleanup without
signature, declaration, or ABI drift
**Time estimate:** 12 hours

### Tasks
1. Inventory installed public headers and current declaration-preservation
   tooling.
2. Identify headers with stale comments, unclear ownership, weak error
   contracts, or overbroad claims.
3. Score candidate headers for user impact, cleanup risk, cross-doc value, and
   validation cost.
4. Select a bounded cleanup batch for Sprint 155.
5. Record headers deferred to later work and why.
6. Write the header cleanup selection artifact.

### Deliverables
- public-header inventory
- header cleanup scorecard
- selected cleanup batch
- deferred-header register
- Day 7 cleanup contract handoff

### Completion Criteria
- selected headers are high-impact and feasible within the sprint
- signature and ABI preservation constraints are explicit
- deferred headers have documented rationale

---

## Day 7: Header Cleanup Contract

**Title:** Cleanup Contract
**Theme:** Define comment, ownership, error-contract, and declaration
preservation rules before editing headers
**Time estimate:** 12 hours

### Tasks
1. Define allowed header edits, including comments, grouping, ownership notes,
   and claim-boundary wording.
2. Define disallowed edits, including declaration order drift, signature
   changes, exported symbol changes, and unsupported ABI wording.
3. Define declaration-preservation scan commands and expected evidence.
4. Define header cleanup review checklist for maintainers.
5. Update or draft maintainer guidance for public header cleanup.
6. Write the Day 7 cleanup contract artifact.

### Deliverables
- public-header cleanup rules
- declaration-preservation scan plan
- maintainer checklist draft
- error-contract wording guidance
- Day 8 implementation checklist

### Completion Criteria
- header edits have clear guardrails before implementation
- validation expectations are known before files change
- maintainers can repeat the cleanup pattern after Sprint 155

---

## Day 8: Header Cleanup Batch 1

**Title:** Header Batch 1
**Theme:** Clean the first selected public-header tranche while preserving all
declarations and claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Apply cleanup to the first selected public-header tranche.
2. Clarify ownership, lifetime, input/output, and error-return expectations.
3. Remove stale or unsupported comments.
4. Keep declaration spelling, order, signatures, and exported names unchanged.
5. Run focused declaration-preservation checks for edited headers.
6. Record implementation evidence and any residual cleanup risks.

### Deliverables
- first public-header cleanup batch
- focused declaration-preservation evidence
- error-contract clarification notes
- residual risk list
- Day 9 header handoff

### Completion Criteria
- edited headers preserve public declarations
- comments clarify usage without changing behavior
- any failing preservation check stops the sprint for repair

---

## Day 9: Header Cleanup Batch 2

**Title:** Header Batch 2
**Theme:** Finish the selected header cleanup batch and reconcile it with
tutorial and reference surfaces
**Time estimate:** 12 hours

### Tasks
1. Apply cleanup to the remaining selected public headers.
2. Normalize ownership notes, error contracts, and claim boundaries across the
   selected batch.
3. Reconcile header wording with tutorial and support-tier docs.
4. Run focused declaration-preservation checks for all edited headers.
5. Update the deferred-header register if new issues are found.
6. Write the header cleanup summary artifact.

### Deliverables
- completed selected header cleanup batch
- all-header declaration-preservation evidence
- updated deferred-header register
- header cleanup summary
- API reference planning inputs

### Completion Criteria
- selected header batch is complete
- tutorial and header wording no longer contradict each other
- declaration-preservation evidence covers the full edited batch

---

## Day 10: API Reference Baseline And Publication Plan

**Title:** API Plan
**Theme:** Design Doxygen/API index guidance or a generated reference
publication plan grounded in the cleaned headers
**Time estimate:** 12 hours

### Tasks
1. Inventory current API, Doxygen, generated reference, and docs-index
   surfaces.
2. Identify gaps between cleaned headers and user-facing API reference needs.
3. Decide whether Sprint 155 adds direct API index guidance, a generated
   reference publication plan, or both.
4. Define publication commands, output ownership, freshness semantics, and
   source-control policy.
5. Define how API reference content should avoid unsupported completeness or
   ABI claims.
6. Write the API reference plan artifact.

### Deliverables
- API reference surface inventory
- Doxygen or generated-reference decision
- publication command and ownership policy
- reference freshness semantics
- Day 11 implementation handoff

### Completion Criteria
- API reference plan is compatible with static-first packaging
- generated outputs have explicit ownership and freshness rules
- reference guidance cannot imply unearned ABI or ecosystem claims

---

## Day 11: API Reference Guidance Implementation

**Title:** Reference Guidance
**Theme:** Implement API reference guidance, index links, and maintainer
publication instructions
**Time estimate:** 12 hours

### Tasks
1. Add or update API reference documentation according to the Day 10 plan.
2. Add Doxygen or generated-reference publication guidance as selected.
3. Update docs indexes and cross-links from tutorial, README, cookbook, and
   maintainer docs.
4. Clarify generated-output freshness and source-control expectations.
5. Check API reference wording against public-header comments.
6. Write the API reference implementation artifact.

### Deliverables
- API reference guidance updates
- reference publication instructions
- docs-index and cross-link updates
- freshness and source-control notes
- implementation summary artifact

### Completion Criteria
- users can find API reference guidance from adoption docs
- maintainers can regenerate or publish reference material predictably
- reference docs align with cleaned public headers

---

## Day 12: Declaration Preservation And Cross-Doc Reconciliation

**Title:** Preservation
**Theme:** Run declaration-preservation scans and reconcile tutorial, headers,
API reference, and maintainer guidance
**Time estimate:** 12 hours

### Tasks
1. Run declaration-preservation scans for edited public headers.
2. Compare installed-header expectations against selected header edits.
3. Reconcile tutorial, API reference, README, cookbook, package, and
   support-tier cross-links.
4. Update maintainer guidance with the final header cleanup process.
5. Repair stale paths, stale titles, and unsupported claim language.
6. Write the Day 12 preservation and reconciliation artifact.

### Deliverables
- final declaration-preservation evidence
- cross-document reconciliation notes
- updated maintainer header-cleanup guidance
- stale-link or stale-claim repair list
- validation checklist for Day 13

### Completion Criteria
- public declarations are demonstrably preserved
- docs point to the correct authoritative surfaces
- maintainer guidance matches the process used in the sprint

---

## Day 13: Integrated Validation And Repair

**Title:** Validation Repair
**Theme:** Run required docs, examples, install, and code quality gates and
repair any failures
**Time estimate:** 12 hours

### Tasks
1. Run documentation checks available in the repository.
2. Run example and install checks as needed for tutorial or package guidance
   changes.
3. If `.c` or public `.h` files changed, run
   `make format && make lint && make test`.
4. Re-run declaration-preservation checks after any repair.
5. Fix validation failures or stop and document blockers.
6. Write the integrated validation artifact.

### Deliverables
- docs validation evidence
- examples/install validation evidence as needed
- full quality-gate evidence if code or public headers changed
- repaired stale links or comments
- integrated validation artifact

### Completion Criteria
- all required checks pass before closeout
- header edits remain declaration-preserving after repair
- no validation failure is deferred without explicit blocker status

---

## Day 14: Closeout And Sprint 156 Handoff

**Title:** Closeout Handoff
**Theme:** Finalize Sprint 155 artifacts, residuals, and Sprint 156
claim-recalibration handoff
**Time estimate:** 10 hours

### Tasks
1. Review all Sprint 155 artifacts for completeness and consistency.
2. Summarize tutorial alignment, header cleanup, API reference, and validation
   outcomes.
3. Record residuals, deferred headers, deferred API reference work, and any
   unearned claims still blocked.
4. Prepare Sprint 156 handoff focused on final validation and claim
   recalibration.
5. Update working notes with final commands, artifacts, and decisions.
6. Write the closeout artifact.

### Deliverables
- Sprint 155 closeout artifact
- residual and deferred-work register
- Sprint 156 handoff
- final working-notes updates
- validation summary

### Completion Criteria
- Sprint 155 deliverables are traceable to the project-plan items
- Sprint 156 has a clear closeout and claim-recalibration handoff
- final artifacts distinguish earned documentation improvements from deferred
  reference or header work
