# Sprint 201 Plan: Additional Review-Surface Reduction

**Sprint Duration:** 14 days
**Goal:** Reduce one large QR, LDLT, SVD, etree, integration, or direct-solver
review surface without changing behavior.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 201 estimate in the Epic 18
project plan.

**Primary scope:** Rank current large source and test surfaces, select exactly
one cluster with clear behavior-preservation invariants, extract helper or
module boundaries only where reviewability improves, add or update ownership
guards, run focused regression evidence, and validate source-list, CMake, docs,
format, lint, and full tests as required by the changed surface.

**Non-goals:** Behavior changes, new public API or ABI, solver algorithm
rewrites, broad review-surface reduction across multiple clusters, performance
claims, package-manager work, platform support promotion, or state-of-the-art
claims.

---

## Day 1: Large Surface Intake

**Title:** Surface Intake
**Theme:** Establish the Sprint 201 review-surface scope and collect current
large source/test candidates.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 201 Epic 18 project-plan section and map items 201.1
   through 201.6 to expected artifacts.
2. Inventory current source, test, helper, benchmark, and documentation files
   that exceed the review-size threshold or have dense ownership concerns.
3. Review Sprint 193 and Epic 18 residual evidence for prior review-surface
   reduction patterns, guard expectations, and deferred candidates.
4. Create `WORKING_NOTES.md` with an item checklist, candidate ledger,
   validation matrix, risk register, and open questions.
5. Record non-goals so intake cannot be mistaken for behavior or API work.

### Deliverables

- Sprint 201 working-notes scaffold.
- Large-surface candidate inventory.
- Item-to-artifact traceability map.
- Initial risk register and validation matrix.

### Completion Criteria

- Every Sprint 201 item has an initial artifact path or evidence category.
- Candidate surfaces are identified before selecting a cluster.
- No code extraction begins before review-risk ranking is recorded.

---

## Day 2: Candidate Ranking

**Title:** Candidate Ranking
**Theme:** Rank large review surfaces by review risk, ownership clarity, and
extraction feasibility.
**Time estimate:** 12 hours

### Tasks

1. Score candidate surfaces by line count, reviewer burden, ownership
   ambiguity, helper cohesion, behavior-risk exposure, and test coverage.
2. Separate source-module candidates from test/helper candidates so extraction
   risk is comparable.
3. Identify clusters that are already guarded by focused tests, CTest labels,
   source-list checks, or helper registration scripts.
4. Mark candidates that should be deferred because extraction would change
   behavior, public API, or algorithm boundaries.
5. Record the ranking table and preferred cluster shortlist.

### Deliverables

- Ranked review-surface candidate table.
- Deferred-candidate ledger.
- Preferred cluster shortlist.
- Item 201.1 evidence record.

### Completion Criteria

- Item 201.1 has a documented ranking.
- Ranking prefers complete closure of one cluster over shallow work across
  many files.
- Deferred candidates have explicit reasons and future handoff notes.

---

## Day 3: Cluster Selection and Boundaries

**Title:** Cluster Selection
**Theme:** Select exactly one review-surface cluster and freeze the no-behavior
change boundary.
**Time estimate:** 12 hours

### Tasks

1. Choose one cluster from the Day 2 shortlist and document why it is the
   Sprint 201 target.
2. Define the files, functions, helpers, tests, and build registrations inside
   the selected cluster.
3. Define out-of-scope neighboring code that must not be changed during the
   sprint.
4. Record the no-public-API-change, no-ABI-change, no-algorithm-change, and
   no-output-change boundaries.
5. Identify focused behavior-preservation checks that must pass after
   extraction.

### Deliverables

- Selected-cluster decision record.
- In-scope and out-of-scope file/function map.
- Behavior-preservation boundary statement.
- Focused validation checklist.

### Completion Criteria

- Item 201.2 has a selected cluster and frozen boundary.
- The cluster can be reviewed independently from broader solver work.
- Public API, ABI, algorithm, and output behavior remain explicitly unchanged.

---

## Day 4: Behavior-Preservation Invariants

**Title:** Preservation Invariants
**Theme:** Convert the selected cluster boundary into pre-edit invariants and
test expectations.
**Time estimate:** 12 hours

### Tasks

1. Write invariants for function signatures, data ownership, error propagation,
   numerical output, deterministic ordering, and cleanup behavior.
2. Map each invariant to existing focused tests or identify a narrow regression
   that must be added later.
3. Identify source-list, CMake, or test registration surfaces that may need
   updates if files are extracted.
4. Record reviewability goals such as reduced file size, cohesive helper
   naming, and clearer ownership.
5. Add a pre-edit checklist to the working notes and Day 4 artifact.

### Deliverables

- Behavior-preservation invariant record.
- Invariant-to-test traceability draft.
- Registration-impact checklist.
- Reviewability success criteria.

### Completion Criteria

- Item 201.2 has concrete behavior-preservation invariants before code edits.
- Every planned extraction step maps to a preservation invariant.
- Missing focused tests are identified before implementation begins.

---

## Day 5: Extraction Design

**Title:** Extraction Design
**Theme:** Design the smallest helper or module extraction that improves
reviewability without changing behavior.
**Time estimate:** 12 hours

### Tasks

1. Inspect the selected cluster and group candidate helpers by ownership,
   state dependencies, and call direction.
2. Choose the extraction shape: local static helpers, new private helper
   header, new private source module, or test-helper split.
3. Define include dependencies, static visibility, build-list updates, and
   source-list impacts.
4. Plan how to preserve existing formatting, naming, diagnostics, and error
   flow.
5. Record risks for circular dependencies, public leakage, and review churn.

### Deliverables

- Minimal extraction design.
- Dependency and build-registration plan.
- Visibility and naming decision record.
- Implementation checklist for Days 6 through 8.

### Completion Criteria

- Item 201.3 has an extraction design that can be reviewed locally.
- The design does not create public headers or public API.
- Registration impacts are explicit before files move or helpers split.

---

## Day 6: First Extraction Pass

**Title:** Initial Extraction
**Theme:** Perform the first narrowly scoped extraction while preserving
behavior and local buildability.
**Time estimate:** 12 hours

### Tasks

1. Move or isolate the first cohesive helper group selected on Day 5.
2. Keep function visibility private unless the existing codebase requires a
   private internal declaration.
3. Update include paths, source lists, CMake registrations, or test-helper
   includes only where required.
4. Run a focused compile or test command that proves the extraction is locally
   reachable.
5. Record the changed files and any deviations from the Day 5 design.

### Deliverables

- First extraction implementation pass.
- Required registration updates.
- Focused local sanity-check result.
- Updated working-notes implementation log.

### Completion Criteria

- Extracted code builds or the blocking error is fixed before proceeding.
- Behavior-relevant code is moved without semantic edits.
- Review-surface reduction remains limited to the selected cluster.

---

## Day 7: Second Extraction Pass

**Title:** Cohesion Pass
**Theme:** Complete the core extraction and tighten helper ownership around
the selected cluster.
**Time estimate:** 12 hours

### Tasks

1. Move any remaining helper group needed to complete the selected extraction
   boundary.
2. Remove duplicated declarations or local helper copies introduced during the
   first pass.
3. Confirm private helper names, comments, and file layout match local project
   conventions.
4. Run the selected focused compile or test command again.
5. Update the invariant-to-test traceability record for the final extracted
   shape.

### Deliverables

- Completed core extraction.
- Duplicate/dead local helper cleanup.
- Updated traceability record.
- Focused sanity-check result.

### Completion Criteria

- Item 201.3 core extraction is implemented.
- Helper ownership is clearer than the pre-sprint surface.
- Focused checks pass after the second extraction pass.

---

## Day 8: Build and Registration Alignment

**Title:** Registration Alignment
**Theme:** Align Make, CMake, source-list, test-list, and helper registration
surfaces with the extracted cluster.
**Time estimate:** 12 hours

### Tasks

1. Update Makefile source lists, CMake source lists, or helper guard inputs if
   the extraction created, removed, or renamed files.
2. Run source-list and CMake parity checks that apply to the changed
   registrations.
3. Add or update registration tests so future drift fails clearly.
4. Confirm generated build artifacts remain ignored and unstaged.
5. Record registration evidence and any residual registration risks.

### Deliverables

- Build/source/test registration updates.
- Source-list and CMake parity evidence.
- Registration drift guard updates.
- Generated-artifact hygiene check.

### Completion Criteria

- Item 201.4 registration ownership is implemented where needed.
- Extracted files cannot silently drop from build or test surfaces.
- Registration checks pass or blockers are fixed before proceeding.

---

## Day 9: Ownership Guard

**Title:** Ownership Guard
**Theme:** Add or update focused guard scripts/tests for the selected
review-surface boundary.
**Time estimate:** 12 hours

### Tasks

1. Define guard rules for selected-cluster file membership, helper placement,
   registration, and no-public-API leakage.
2. Implement or update the smallest guard script/test that enforces those
   rules.
3. Include negative or drift-sensitive checks when the existing guard pattern
   supports them.
4. Wire the guard into an existing Make target if local conventions support
   it.
5. Run the guard and record exact output.

### Deliverables

- Focused ownership guard.
- Guard wiring or invocation record.
- Drift-sensitive check list.
- Item 201.4 evidence artifact.

### Completion Criteria

- Item 201.4 has a guard that fails clearly on registration or ownership drift.
- Guard wording stays selected-cluster scoped.
- No broad review-surface claim is introduced.

---

## Day 10: Focused Regression Review

**Title:** Focused Regression
**Theme:** Run focused behavior-preservation checks and add narrow regression
coverage only where extraction safety needs it.
**Time estimate:** 12 hours

### Tasks

1. Run the selected focused tests, labels, or binaries mapped on Days 3 and 4.
2. Add narrow regression tests only if existing coverage does not prove an
   extraction-sensitive invariant.
3. Confirm existing expected outputs, diagnostics, and error codes are
   preserved.
4. Update the invariant-to-test traceability table with final focused
   regression evidence.
5. Record any intentionally untested breadth as residual risk.

### Deliverables

- Focused regression evidence.
- Any narrowly required regression tests.
- Final invariant-to-test table.
- Residual untested-breadth record.

### Completion Criteria

- Item 201.5 has focused behavior-preservation evidence.
- New tests, if any, are extraction-safety tests rather than feature tests.
- Behavior preservation is evidenced before docs promotion.

---

## Day 11: Documentation and Maintainer Alignment

**Title:** Maintainer Alignment
**Theme:** Update maintainer and planning documentation so review-surface
claims match the selected extraction.
**Time estimate:** 12 hours

### Tasks

1. Update maintainer documentation with the selected cluster, extracted helper
   ownership, guard command, and validation expectations.
2. Update Epic 18 project-plan status and residual queue entries for Sprint
   201.
3. Record that Sprint 201 reduces one selected review surface, not every large
   review surface.
4. Link behavior-preservation artifacts, focused tests, and guard commands.
5. Confirm public docs need no change unless the extraction affects a
   user-facing explanation.

### Deliverables

- Claim-safe maintainer documentation updates.
- Updated project-plan and residual status.
- Selected-cluster proof references.
- Documentation non-claim record.

### Completion Criteria

- Item 201.6 documentation work is complete for changed ownership surfaces.
- Documentation does not imply behavior, API, or broad review-surface changes.
- Proof claims are linked to runnable evidence.

---

## Day 12: Integrated Local Validation

**Title:** Validation Pass
**Theme:** Run focused and required local validation before closeout.
**Time estimate:** 12 hours

### Tasks

1. Run the focused selected-cluster regression checks.
2. Run the ownership guard and source-list/CMake parity checks that apply.
3. Run `make format`.
4. Run `make lint`.
5. Run `make test` and record results, failures, or environmental residuals.

### Deliverables

- Integrated validation transcript summary.
- Focused regression and ownership guard results.
- Format, lint, and test result record.
- Updated risk register.

### Completion Criteria

- Item 201.6 validation evidence is recorded.
- Required quality checks have run or have explicit environment residuals.
- Any failures are fixed before closeout proceeds.

---

## Day 13: Review-Surface Hardening

**Title:** Review Hardening
**Theme:** Audit the selected extraction for clarity, narrowness, behavior
preservation, and evidence consistency.
**Time estimate:** 11 hours

### Tasks

1. Review all code, tests, docs, build registrations, and planning diffs for
   unnecessary breadth.
2. Confirm each extraction-sensitive invariant maps to focused regression or
   guard evidence.
3. Confirm the selected cluster did not introduce public API, ABI, behavior,
   performance, or support claims.
4. Re-run focused checks affected by review-hardening changes.
5. Prepare closeout notes with completed items and residual review surfaces.

### Deliverables

- Review-hardening notes.
- Final invariant-to-regression traceability table.
- Cleaned-up selected extraction surface.
- Closeout checklist draft.

### Completion Criteria

- The review surface is limited to the selected cluster and required guards.
- Evidence, docs, and tests use consistent no-behavior-change vocabulary.
- Remaining residuals are explicit and not hidden in completion claims.

---

## Day 14: Closeout and Retrospective Inputs

**Title:** Closeout Evidence
**Theme:** Finalize Sprint 201 evidence, item status, and retrospective inputs.
**Time estimate:** 11 hours

### Tasks

1. Reconcile items 201.1 through 201.6 against deliverables and validation
   evidence.
2. Update `WORKING_NOTES.md` with final item status, command results, known
   residuals, and follow-up candidates.
3. Create a Day 14 closeout artifact summarizing selected review-surface
   reduction completion and behavior-preservation boundaries.
4. Prepare retrospective inputs: completed work, validation, deviations,
   deferred breadth, and recommendations.
5. Confirm the working tree contains only Sprint 201 intended changes.

### Deliverables

- Final Sprint 201 working-notes status.
- Day 14 closeout artifact.
- Retrospective input summary.
- Final validation and residual-risk ledger.

### Completion Criteria

- Sprint 201 has enough evidence for a later retrospective.
- All six Sprint 201 items have clear completion or residual status.
- The selected review-surface reduction is documented without unsupported
  behavior, API, ABI, performance, or broad reviewability claims.
