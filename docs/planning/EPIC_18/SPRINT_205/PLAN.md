# Sprint 205 Plan: Support Matrix and Adoption Quick-Reference Consolidation

**Sprint Duration:** 14 days
**Goal:** Reduce public documentation friction by centralizing support truth
and adding a compact problem-shape quick reference without weakening claim
boundaries.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `164` hours, matching the Sprint 205 estimate in the Epic 18
project plan.

**Primary scope:** Audit public and maintainer documentation for duplicate
support caveats, design and publish a compact adoption quick reference,
centralize support/readiness truth, normalize diagnostics vocabulary across
solver and report docs, update claim guards, and validate all changed
documentation and guard surfaces.

**Non-goals:** New solver behavior, public API or ABI expansion, package
manager support, broad Windows or platform parity claims, portable performance
claims, release claims, hosted generated API publication, state-of-the-art
claims, or broad support promotion not backed by existing Epic 18 evidence.

---

## Day 1: Sprint Intake And Surface Map

**Title:** Support Intake
**Theme:** Establish Sprint 205 scope, inherited support decisions, and the
documentation surfaces that need consolidation.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 205 Epic 18 project-plan section and map items 205.1
   through 205.6 to expected artifacts, docs, guard files, and validation
   commands.
2. Inventory README, INSTALL, tutorial, cookbook, solver-selection,
   examples, benchmark, API reference, and maintainer guide support wording.
3. Record the package, Windows, benchmark, comparison, and generated API
   decisions inherited from Sprints 198 through 204.
4. Create `WORKING_NOTES.md` with an item checklist, evidence map, risk
   register, validation matrix, and open questions.
5. Record non-goals and claim boundaries before editing any user-facing docs.

### Deliverables

- Sprint 205 working-notes scaffold.
- Item-to-surface traceability map.
- Inherited support-decision inventory.
- Initial risk register and validation matrix.

### Completion Criteria

- Every Sprint 205 item has an initial evidence path or artifact category.
- All public and maintainer support surfaces are identified before edits.
- Unsupported package, ABI, platform, performance, release, and
  state-of-the-art claims remain explicitly out of scope.

---

## Day 2: Public Documentation Audit

**Title:** Public Audit
**Theme:** Audit user-facing documentation for duplicate caveats, stale
support truth, and adoption friction.
**Time estimate:** 12 hours

### Tasks

1. Read README, INSTALL, `docs/tutorial.md`, `docs/cookbook.md`,
   `docs/solver_selection.md`, `docs/api_reference.md`, and examples for
   repeated support/readiness wording.
2. Identify places where users must read too many files to answer common
   adoption questions.
3. Mark wording that is accurate, stale, duplicated, over-specific, too
   broad, or missing links to authoritative support truth.
4. Separate user-facing friction from maintainer-only claim interpretation.
5. Write the Day 2 public documentation audit artifact.

### Deliverables

- Public documentation duplication and friction inventory.
- Support/readiness wording status table.
- User workflow question list.
- Day 2 audit artifact.

### Completion Criteria

- Item 205.1 has public-doc evidence for every listed user-facing surface.
- Duplicated caveats are classified before consolidation starts.
- No claim is edited without knowing its current evidence source.

---

## Day 3: Maintainer And Report Surface Audit

**Title:** Maintainer Audit
**Theme:** Audit maintainer, benchmark, report, and planning-adjacent docs for
support truth drift and diagnostics vocabulary inconsistency.
**Time estimate:** 12 hours

### Tasks

1. Review `docs/maintainer_guide.md`, benchmark docs, report manifests, claim
   boundary notes, and Epic 18 residual/status docs for support wording.
2. Identify diagnostics terms used for direct solvers, iterative solvers,
   QR/SVD, eigensolvers, generated reports, and validation guards.
3. Classify repeated maintainer caveats that should stay local versus move to
   a centralized support/readiness reference.
4. Record any planning evidence that should be linked from maintainer docs but
   not become primary user workflow.
5. Write the Day 3 maintainer/report audit artifact.

### Deliverables

- Maintainer and report documentation audit.
- Diagnostics vocabulary inventory.
- Planning-evidence routing notes.
- Day 3 audit artifact.

### Completion Criteria

- Item 205.1 includes maintainer and report surfaces, not only public docs.
- Diagnostics vocabulary conflicts are identified before wording changes.
- Planning artifacts remain evidence, not replacement user documentation.

---

## Day 4: Quick Reference Design

**Title:** Quick Reference Design
**Theme:** Design the compact problem-shape to workflow table for common local
and installed use cases.
**Time estimate:** 12 hours

### Tasks

1. Define the problem-shape categories users naturally ask about: SPD,
   general square, least-squares, rank-deficient, eigensolver, SVD,
   matrix-free, IO, threading, and packaging.
2. Map each problem shape to source-controlled docs, examples, supported
   readiness, validation evidence, and retained non-claims.
3. Decide whether the quick reference belongs in README, INSTALL,
   `docs/solver_selection.md`, or a linked combination of surfaces.
4. Draft the table structure, row vocabulary, and link targets.
5. Define acceptance criteria for keeping the table compact and
   claim-calibrated.

### Deliverables

- Problem-shape category list.
- Quick reference table design.
- Link target and ownership map.
- Day 4 design artifact.

### Completion Criteria

- Item 205.2 has a concrete table design before implementation.
- Every row has an intended support/readiness interpretation.
- Compact wording cannot imply unearned package, ABI, platform, or
  performance support.

---

## Day 5: Support Truth Architecture

**Title:** Support Truth Design
**Theme:** Design the centralized support/readiness routing model that will
replace repeated caveats with safe links.
**Time estimate:** 12 hours

### Tasks

1. Select the authoritative support/readiness surface or surfaces for package,
   platform, generated API, benchmark, comparison, ABI, release, and
   state-of-the-art status.
2. Define which repeated caveats can be replaced by links and which must stay
   inline for safety.
3. Specify link text and anchor names for support/readiness routing.
4. Map public-doc and maintainer-doc updates needed for item 205.3.
5. Write the Day 5 support truth architecture artifact.

### Deliverables

- Central support/readiness ownership model.
- Duplicate caveat conversion plan.
- Support truth link and anchor list.
- Day 5 architecture artifact.

### Completion Criteria

- Item 205.3 has an implementation-ready routing model.
- Replaced caveats have authoritative destinations.
- Critical non-claims remain visible where users could otherwise overinfer
  support.

---

## Day 6: Quick Reference Implementation

**Title:** Quick Reference Implementation
**Theme:** Add the compact adoption quick reference to the selected
user-facing surface and route users to detailed docs.
**Time estimate:** 12 hours

### Tasks

1. Implement the problem-shape quick reference table in the selected
   documentation surface.
2. Add links to solver selection, tutorial, cookbook, API reference, examples,
   and INSTALL readiness rows as appropriate.
3. Keep each row compact while preserving support status, local/build
   context, and non-claim boundaries.
4. Update adjacent README or solver-selection text so the quick reference is
   discoverable.
5. Record changed files and claim boundary notes in `WORKING_NOTES.md`.

### Deliverables

- Implemented adoption quick reference.
- Updated discovery links.
- Claim-boundary change log.
- Day 6 implementation notes.

### Completion Criteria

- Item 205.2 is implemented in source-controlled docs.
- Users can find the right workflow from common problem shapes.
- The quick reference does not broaden support, package, ABI, platform,
  performance, or state-of-the-art claims.

---

## Day 7: Support Truth Consolidation Batch

**Title:** Support Consolidation
**Theme:** Centralize support/readiness truth and replace repeated caveats
with safe links where the Day 5 architecture allows.
**Time estimate:** 12 hours

### Tasks

1. Update INSTALL and README support/readiness wording to point at the
   selected support truth surfaces.
2. Replace duplicated package, platform, generated API, comparison, and
   benchmark caveats with links where safe.
3. Preserve inline warnings where users may otherwise infer installation,
   ABI, platform, or release support.
4. Update maintainer guidance to explain the new support truth routing.
5. Record before/after caveat consolidation decisions.

### Deliverables

- Consolidated support/readiness documentation.
- Duplicate-caveat replacement ledger.
- Maintainer interpretation update.
- Day 7 consolidation notes.

### Completion Criteria

- Item 205.3 has landed in public and maintainer docs.
- Repeated caveats are reduced without losing necessary local warning text.
- Support truth routing is clear enough for future sprint updates.

---

## Day 8: Examples And Workflow Link Pass

**Title:** Example Routing
**Theme:** Align examples, tutorial, cookbook, and solver-selection routing
with the new quick reference and support truth model.
**Time estimate:** 12 hours

### Tasks

1. Review example references from README, tutorial, cookbook, and
   solver-selection docs.
2. Add or adjust links from examples and workflow docs back to the quick
   reference only where it reduces user friction.
3. Ensure example wording distinguishes local source builds, installed use,
   optional dependencies, and unsupported package-manager claims.
4. Remove stale duplicate caveats that are now safely covered by the support
   truth routing.
5. Write the Day 8 example and workflow routing artifact.

### Deliverables

- Updated example/workflow routing.
- Example support interpretation notes.
- Reduced stale duplicate caveats.
- Day 8 routing artifact.

### Completion Criteria

- Users can move between quick reference, detailed docs, and examples without
  circular or stale routes.
- Example docs do not imply unsupported install or package-manager support.
- Item 205.3 remains consistent after workflow-doc edits.

---

## Day 9: Diagnostics Vocabulary Design

**Title:** Diagnostics Design
**Theme:** Define consistent result, residual, convergence, and status
vocabulary across solver and report documentation.
**Time estimate:** 12 hours

### Tasks

1. Group existing diagnostics terms by direct solvers, iterative solvers,
   QR/SVD, eigensolver, matrix/report generation, and validation guards.
2. Define preferred terms for success, warning, residual, convergence,
   unsupported, unavailable, deferred, generated, hosted, local-only, and
   selected evidence states.
3. Identify public-facing terms that should differ from maintainer-only
   diagnostics.
4. Create a vocabulary decision table with examples and replacement targets.
5. Record risks where changing wording could alter expected user
   interpretation.

### Deliverables

- Diagnostics vocabulary decision table.
- Replacement target list.
- Public versus maintainer wording notes.
- Day 9 diagnostics design artifact.

### Completion Criteria

- Item 205.4 has a concrete vocabulary before edit passes begin.
- Preferred terms are consistent with existing error/status APIs.
- Wording choices do not imply new behavior or stronger validation evidence.

---

## Day 10: Diagnostics Vocabulary Implementation

**Title:** Diagnostics Implementation
**Theme:** Apply the selected diagnostics vocabulary across user-facing and
maintainer documentation.
**Time estimate:** 12 hours

### Tasks

1. Update direct solver, iterative solver, QR/SVD, eigensolver, and report
   documentation wording according to the Day 9 decision table.
2. Normalize residual, convergence, unavailable, unsupported, generated, and
   selected-evidence vocabulary.
3. Preserve exact API names, enum names, return codes, and command output
   where docs quote implementation surfaces.
4. Add maintainer notes for future diagnostics wording changes.
5. Record changed files and any retained exceptions.

### Deliverables

- Diagnostics vocabulary updates across selected docs.
- Retained-exception ledger.
- Maintainer wording guidance.
- Day 10 implementation notes.

### Completion Criteria

- Item 205.4 is implemented across the selected documentation surfaces.
- User-facing and maintainer-facing terms are coherent.
- No implementation behavior is implied to have changed.

---

## Day 11: Claim Guard Design

**Title:** Guard Design
**Theme:** Design documentation guard updates that preserve simplified wording
and support truth consolidation.
**Time estimate:** 11 hours

### Tasks

1. Inventory existing documentation, support, routing, and claim-boundary
   guard scripts and tests.
2. Identify which new quick-reference, support-truth, and diagnostics wording
   must become executable guard coverage.
3. Define failure messages for missing support truth anchors, broadened
   package/platform/ABI/performance claims, and stale quick-reference routes.
4. Decide whether to update existing tests or add new focused guard tests.
5. Write the Day 11 guard design artifact.

### Deliverables

- Claim guard update plan.
- Guard-to-wording traceability table.
- Failure message and regression fixture design.
- Day 11 guard design artifact.

### Completion Criteria

- Item 205.5 has an implementation-ready guard design.
- Guard coverage targets the simplified wording, not incidental formatting.
- Unsupported support, package, ABI, platform, performance, release, and
  state-of-the-art claims remain guarded.

---

## Day 12: Claim Guard Implementation

**Title:** Guard Implementation
**Theme:** Implement or update docs guards for quick-reference, support truth,
diagnostics vocabulary, and retained non-claims.
**Time estimate:** 11 hours

### Tasks

1. Implement the selected guard or regression updates from Day 11.
2. Add fixtures for missing quick-reference routes, missing support truth
   anchors, and broadened support/non-claim wording.
3. Ensure guard checks distinguish user-facing wording from maintainer-only
   interpretation.
4. Run focused guard tests and record outputs.
5. Update working notes with changed files and validation evidence.

### Deliverables

- Updated documentation guard scripts or tests.
- Quick-reference and support-truth regression fixtures.
- Focused guard validation record.
- Day 12 implementation artifact.

### Completion Criteria

- Item 205.5 is implemented with executable coverage.
- Simplified wording cannot silently broaden support or readiness claims.
- Focused guard tests pass or any blocker is documented with exact output.

---

## Day 13: Integrated Documentation Validation

**Title:** Integrated Validation
**Theme:** Run integrated docs, claim, install, support, generated API, and
format checks for the changed surfaces.
**Time estimate:** 11 hours

### Tasks

1. Run documentation checks, support/claim guards, install docs checks,
   generated API freshness checks, and any focused quick-reference guards.
2. Run `make format`, `make lint`, and `make test` only if C or header files
   changed; otherwise record why the full C gate is not required.
3. Review `git diff --check`, changed-file inventory, and generated-output
   tracking state.
4. Fix validation failures within the sprint scope or record exact blockers.
5. Write the Day 13 integrated validation artifact.

### Deliverables

- Integrated validation command log.
- Changed-file and generated-output inventory.
- Full C gate decision record.
- Day 13 validation artifact.

### Completion Criteria

- Item 205.6 has current validation evidence.
- Required checks pass before closeout, or blockers are explicit and scoped.
- No generated or temporary output is accidentally tracked.

---

## Day 14: Closeout And Review Hardening

**Title:** Closeout Review
**Theme:** Finalize Sprint 205 evidence, review claim boundaries, and prepare
the branch for retrospective and PR review.
**Time estimate:** 11 hours

### Tasks

1. Review all Sprint 205 artifacts, working notes, docs edits, and guard
   changes against items 205.1 through 205.6.
2. Audit final quick-reference and support-truth wording for claim creep,
   stale links, duplicate caveats, and missing evidence routes.
3. Run any final focused validation needed after closeout edits.
4. Create a Day 14 closeout review artifact with completed, narrowed,
   deferred, and residual outcomes.
5. Update `WORKING_NOTES.md` with final status and retrospective inputs.

### Deliverables

- Day 14 closeout review artifact.
- Final item status table.
- Retained residual and non-claim list.
- Retrospective input notes.

### Completion Criteria

- Every Sprint 205 item has completion evidence or an explicit residual.
- Quick-reference, support matrix, diagnostics wording, and guards are
  internally consistent.
- The branch is ready for Sprint 205 retrospective creation and PR review.
