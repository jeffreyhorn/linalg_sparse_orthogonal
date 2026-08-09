# Sprint 145 Plan: Adoption Surface Simplification & High-Level Workflow Front Door

**Sprint Duration:** 14 days
**Goal:** Simplify first-use adoption after the corpus, QR, partial-SVD,
report, runtime, package, and platform decisions have landed. This sprint
implements the Sprint 145 section of
`docs/planning/EPIC_12/PROJECT_PLAN.md`.

**Starting Point:** Sprint 145 begins from:
- Sprint 139 QR claim settlement and bounded corpus evidence
- Sprint 140 partial-SVD claim settlement and convergence/residual evidence
- Sprint 141 report semantics, freshness gates, and normalized report index
  contract
- Sprint 142 runtime/backend governance and sentinel boundaries
- Sprint 143 static-first package/ABI decision and install/export proof
- Sprint 144 platform support-tier settlement, including macOS reviewed
  static-first install/export proof and Windows CMake-first boundaries
- existing README, INSTALL, cookbook/tutorial/example, public header,
  benchmark, solver-selection, and maintainer guidance surfaces

The sprint must:
- audit first-use friction across public documentation, examples, headers, and
  workflow entry points
- design a concise high-level workflow front door for build/install, solver
  selection, solve execution, diagnostics, and advanced-control escalation
- add or revise maintained examples and cookbook entries for QR, partial-SVD,
  corpus/report, runtime/backend, package, and platform behavior
- simplify README and INSTALL while preserving exact support-tier boundaries
  and deeper maintainer references
- clean selected public headers where maintainer-only historical detail
  obscures first-use contracts
- run documentation, example, install/downstream, and full C quality gates as
  required by touched files
- publish an adoption claim map, residual documentation debt, and Sprint 146
  closeout handoff

**End State:** Sprint 145 leaves behind:
- simplified first-use adoption path
- updated examples and cookbook entries
- README/INSTALL support-tier alignment
- public-header cleanup for selected surfaces
- adoption claim map and residual doc-debt ledger
- Sprint 146 closeout handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 145 project-plan estimate.

---

## Day 1: Adoption Surface Intake

**Title:** Adoption Intake
**Theme:** Establish Sprint 145 scope, inherited evidence, adoption surfaces,
and first-use success criteria
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 145 section of
   `docs/planning/EPIC_12/PROJECT_PLAN.md`.
2. Review Sprint 139-144 retrospectives, closeout artifacts, and handoffs for
   earned claims, non-claims, and deferred adoption work.
3. Create Sprint 145 working notes and artifact directory structure.
4. Inventory first-use adoption surfaces: README, INSTALL, cookbook, tutorial,
   examples, solver-selection guidance, benchmark docs, public headers, and
   maintainer guide entry points.
5. Map Sprint 145 Items 1-7 to day-level owners.
6. Record initial adoption-success criteria, claim boundaries, validation
   requirements, and stop conditions.

### Deliverables
- Sprint 145 working-notes baseline
- artifact directory structure
- first-use adoption surface inventory
- item-to-day owner map
- initial adoption criteria and non-claim register

### Completion Criteria
- every Sprint 145 project-plan item has a day-level owner
- inherited QR, partial-SVD, report, runtime, package, and platform claims are
  treated as evidence boundaries, not rewritten as broader claims
- adoption simplification has explicit validation and stop conditions

---

## Day 2: Adoption Friction Audit

**Title:** Friction Audit
**Theme:** Audit first-use friction, stale wording, duplicated entry points,
and support-tier ambiguity
**Time estimate:** 12 hours

### Tasks
1. Audit README, INSTALL, cookbook/tutorial docs, examples, benchmark docs, and
   solver-selection guidance for first-use density and stale support wording.
2. Audit selected public headers for maintainer-only historical detail,
   duplicated caveats, unclear ownership, or hidden prerequisites.
3. Identify where QR, partial-SVD, corpus/report, runtime/backend, package, and
   platform claims are hard to discover or easy to overread.
4. Classify friction by user path: build/install, choose solver, run solve,
   inspect diagnostics, and escalate to advanced controls.
5. Rank fixes by adoption value, implementation risk, validation cost, and
   claim-boundary risk.
6. Write the adoption friction audit artifact.

### Deliverables
- adoption friction inventory
- stale wording and duplication register
- public-header cleanup candidate list
- first-use path risk ranking
- prioritized fix shortlist

### Completion Criteria
- first-use blockers are backed by concrete file references
- stale or ambiguous support-tier wording is separated from historical planning
  evidence
- candidate fixes are ranked before workflow design begins

---

## Day 3: Workflow Front-Door Design

**Title:** Workflow Design
**Theme:** Design the concise high-level adoption path before editing docs or
examples
**Time estimate:** 12 hours

### Tasks
1. Define the canonical first-use path: build/install, choose solver, run
   solve, inspect diagnostics, and move to advanced controls.
2. Decide which existing docs remain front-door content and which move behind
   deeper links.
3. Design examples/cookbook entry points that demonstrate earned QR,
   partial-SVD, runtime/backend, package, and platform behavior without
   overclaiming.
4. Define public-header cleanup rules for selected surfaces.
5. Define validation commands for docs, examples, install/downstream checks,
   and C/header changes if any.
6. Write the high-level workflow design artifact.

### Deliverables
- high-level workflow front-door design
- content routing map
- examples/cookbook design checklist
- public-header cleanup rules
- validation plan

### Completion Criteria
- workflow design covers build/install, solver selection, solve execution,
  diagnostics, and advanced controls
- design preserves all Sprint 139-144 claim boundaries
- implementation can proceed without inventing new solver/package/platform
  claims

---

## Day 4: Example And Cookbook Design

**Title:** Example Design
**Theme:** Specify maintained examples and cookbook updates for the new
front-door workflow
**Time estimate:** 12 hours

### Tasks
1. Inventory maintained examples and cookbook/tutorial entries that overlap
   first-use workflows.
2. Select the examples or cookbook entries that should be revised or added for
   QR, partial-SVD, solver selection, diagnostics, runtime/backend, and
   install/export behavior.
3. Define command lines, expected outputs, and validation owners for each
   selected example.
4. Identify examples that should stay advanced-only rather than front-door
   content.
5. Define documentation links from README and INSTALL into the selected
   examples.
6. Write the example and cookbook implementation plan.

### Deliverables
- maintained example/cookbook selection matrix
- expected command/output plan
- advanced-only example list
- README/INSTALL link plan
- example validation checklist

### Completion Criteria
- selected example changes are scoped to first-use adoption
- every new or revised example has a validation owner
- advanced examples do not make the front door dense again

---

## Day 5: Example And Cookbook Batch

**Title:** Example Batch
**Theme:** Implement the maintained example and cookbook front-door updates
**Time estimate:** 12 hours

### Tasks
1. Add or revise selected maintained examples and cookbook/tutorial entries.
2. Keep examples short, runnable, and aligned with the designed first-use path.
3. Add expected output notes only where they are stable and useful.
4. Link examples to the appropriate QR, partial-SVD, runtime/backend, package,
   and platform evidence boundaries.
5. Run focused example build or syntax checks available locally.
6. Record implementation notes and changed proof owners.

### Deliverables
- revised maintained examples
- revised cookbook/tutorial entries
- example-output notes where appropriate
- local example validation summary
- changed proof-owner list

### Completion Criteria
- examples are runnable or explicitly documented as docs-only snippets
- example wording does not widen numerical, package, or platform claims
- focused validation for touched examples passes

---

## Day 6: README Front-Door Restructure

**Title:** README Simplification
**Theme:** Simplify README first-use flow while preserving exact support
boundaries
**Time estimate:** 12 hours

### Tasks
1. Rework the README opening adoption path around build/install, solver choice,
   first solve, diagnostics, and deeper references.
2. Reduce repeated support-tier detail in the README by routing deep details to
   INSTALL, maintainer guidance, reports, or sprint evidence.
3. Preserve QR, partial-SVD, runtime/backend, package, and platform non-claims.
4. Link maintained examples and cookbook entries from the front-door path.
5. Run docs scans for stale support-tier wording and unsupported claims.
6. Record README simplification evidence.

### Deliverables
- simplified README adoption path
- updated README links to examples and deeper references
- README claim-boundary scan summary
- README implementation notes

### Completion Criteria
- README has a clear first-use path without burying the user in maintainer
  detail
- README still preserves all earned support tiers and non-claims
- docs scans pass for touched README sections

---

## Day 7: INSTALL Front-Door Restructure

**Title:** INSTALL Simplification
**Theme:** Simplify install and downstream-consumer guidance around the
static-first platform contract
**Time estimate:** 12 hours

### Tasks
1. Rework INSTALL around the most common first-use install path and downstream
   consumer paths.
2. Keep static-first package support, CMake install/export, `pkg-config`, and
   platform support tiers precise.
3. Move or link maintainer-only details that interrupt first-use install
   guidance.
4. Preserve Linux source-of-truth, macOS reviewed static-first proof, and
   Windows CMake-first boundaries.
5. Run install-doc claim scans and link/reference checks available locally.
6. Record INSTALL simplification evidence.

### Deliverables
- simplified INSTALL first-use path
- updated downstream-consumer guidance
- support-tier boundary preservation notes
- install-doc validation summary

### Completion Criteria
- INSTALL is easier to follow for first-use static install and downstream
  consumption
- platform support-tier wording remains consistent with Sprint 144
- install-doc validation and claim scans pass

---

## Day 8: Solver-Selection And Diagnostics Front Door

**Title:** Solver Front Door
**Theme:** Clarify solver-selection, diagnostics, and advanced-control
escalation
**Time estimate:** 12 hours

### Tasks
1. Audit current solver-selection and diagnostics guidance for first-use gaps.
2. Add or revise concise guidance for direct solvers, QR, partial-SVD,
   iterative solvers, runtime/backend decisions, and diagnostics.
3. Route detailed numerical corpus/report evidence to deeper docs rather than
   front-door sections.
4. Preserve bounded QR and partial-SVD claim language from Sprint 139-140.
5. Run docs scans for unsupported numerical or state-of-the-art claims.
6. Record solver front-door evidence.

### Deliverables
- solver-selection front-door update
- diagnostics and advanced-control escalation path
- bounded numerical claim-preservation notes
- validation summary

### Completion Criteria
- first-use users can identify the likely solver path and diagnostic path
- QR and partial-SVD wording stays bounded to earned evidence
- unsupported numerical claim scans pass

---

## Day 9: Public Header Cleanup Design

**Title:** Header Design
**Theme:** Select public-header cleanup targets and rules before modifying
contracts
**Time estimate:** 12 hours

### Tasks
1. Review public headers for adoption-facing friction, historical detail, and
   overly dense comments.
2. Select the smallest set of high-impact public headers for cleanup.
3. Define which details can move to docs and which must remain API contract
   comments.
4. Identify any C/header quality gates required by planned edits.
5. Define before/after review criteria for public API contract preservation.
6. Write the public header cleanup design artifact.

### Deliverables
- selected public-header cleanup target list
- comment-routing rules
- API contract preservation checklist
- C/header quality-gate plan

### Completion Criteria
- public-header changes are scoped and justified
- no API behavior or ABI promise changes are planned accidentally
- required quality gates are known before header edits begin

---

## Day 10: Public Header Cleanup Batch

**Title:** Header Cleanup
**Theme:** Clean selected public headers without weakening contracts
**Time estimate:** 12 hours

### Tasks
1. Edit selected public headers to reduce maintainer-only or historical detail
   from high-impact first-use surfaces.
2. Preserve API contracts, parameter semantics, error semantics, ownership
   rules, and non-claims.
3. Move or link deeper historical detail to maintainer docs where appropriate.
4. Run formatting and focused header/API scans.
5. If `.c` or `.h` files changed, run the required full C quality gate before
   proceeding.
6. Record header cleanup evidence.

### Deliverables
- cleaned selected public headers
- moved/linked maintainer detail where appropriate
- API contract preservation notes
- C/header quality-gate results if required

### Completion Criteria
- selected headers are clearer for first-use readers
- public API contracts are not weakened or broadened
- required C/header quality gates pass if headers changed

---

## Day 11: Cross-Surface Coherence Pass

**Title:** Coherence Pass
**Theme:** Ensure README, INSTALL, examples, solver guidance, headers, reports,
and maintainer docs agree
**Time estimate:** 12 hours

### Tasks
1. Compare front-door README/INSTALL wording against examples, cookbook,
   headers, maintainer guide, report rows, and Sprint 139-144 evidence.
2. Scan for stale support-tier wording, duplicated outdated instructions, and
   unsupported claims.
3. Confirm links route advanced users to deeper docs without requiring first-use
   readers to parse maintainer-only detail.
4. Confirm package/platform boundaries remain static-first and support-tier
   accurate.
5. Fix small coherence issues found during the pass.
6. Write the cross-surface coherence artifact.

### Deliverables
- cross-surface coherence report
- stale wording fix list
- support-tier consistency summary
- advanced-link routing summary

### Completion Criteria
- public adoption surfaces tell one consistent story
- stale wording and unsupported-claim scans pass
- any coherence fixes are documented and validated

---

## Day 12: Validation Gate

**Title:** Validation Gate
**Theme:** Run the required documentation, example, install/downstream, report,
and C/header quality gates
**Time estimate:** 12 hours

### Tasks
1. Run documentation link/reference checks available in the repository.
2. Run maintained example builds or smoke checks affected by the sprint.
3. Run install/downstream checks if README/INSTALL or package examples changed.
4. Run report normalization/freshness checks if report rows or report docs were
   touched.
5. If `.c` or `.h` files changed, run `make format && make lint && make test`.
6. Record exact commands, results, skipped checks, and environment constraints.

### Deliverables
- validation command log
- pass/fail summary
- skipped-check rationale
- environment constraint register
- fix list for any failures

### Completion Criteria
- all required checks for changed surfaces pass
- any skipped checks have explicit rationale
- no validation failure remains unresolved at the end of the day

---

## Day 13: Adoption Claim Map And Residual Debt

**Title:** Claim Map
**Theme:** Publish the final adoption claim map, residual documentation debt,
and Sprint 146 handoff draft
**Time estimate:** 12 hours

### Tasks
1. Map each simplified adoption claim to its source evidence and validation
   owner.
2. Document residual documentation debt that remains outside Sprint 145 scope.
3. Confirm public-header cleanup status and any deferred header surfaces.
4. Confirm README/INSTALL/examples/cookbook/report/support-tier agreement.
5. Draft the Sprint 146 closeout handoff with adoption implications.
6. Write the adoption claim map and residual debt artifact.

### Deliverables
- adoption claim map
- residual documentation debt ledger
- public-header cleanup status
- Sprint 146 closeout handoff draft
- final support-tier consistency check

### Completion Criteria
- every adoption-facing claim has source evidence
- residual doc debt is explicit and source-owned
- Sprint 146 handoff identifies closeout implications

---

## Day 14: Closeout And Handoff

**Title:** Closeout
**Theme:** Complete Sprint 145 documentation, validation summary, and Sprint
146 handoff
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 145 artifacts and working notes for consistency.
2. Update the final validation summary with commands, touched surfaces,
   adoption claims, and support-tier boundaries.
3. Confirm deliverables from the Sprint 145 project-plan section are satisfied
   or explicitly deferred with proof.
4. Finalize Sprint 146 closeout handoff.
5. Run final lightweight repository checks such as status, diff review, and
   whitespace checks.
6. Prepare closeout notes for the retrospective.

### Deliverables
- Sprint 145 closeout validation summary
- completed working notes
- Sprint 146 closeout handoff
- final deliverable checklist
- retrospective input notes

### Completion Criteria
- simplified first-use adoption path is complete or explicitly scoped with
  residual debt
- README, INSTALL, examples/cookbook, selected headers, and support-tier docs
  agree
- Sprint 146 can start from a clear adoption and closeout handoff
