# Sprint 164 Plan: Public Header And API Coherence Batch

**Sprint Duration:** 14 days
**Goal:** Complete a declaration-preserving public-header cleanup batch and
keep the API reference, tutorial, cookbook, and solver-selection docs coherent.
This sprint implements the Sprint 164 section of
`docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references the older Epic 12 project-plan
path, but the current Sprint 164 project-plan section lives in
`docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Starting Point:** Sprint 164 begins from:
- Sprint 158 generated API publication policy and reference freshness work;
- Sprint 157 quality surface mapping;
- Sprint 163 methodology-bound performance publication and API-header handoff;
- public package, ABI, runtime-loader, backend-superiority, and
  state-of-the-art non-claim boundaries;
- current public headers under `include/sparse/` and their matching README,
  tutorial, cookbook, solver-selection, and API-reference surfaces.

The sprint must:
- select a bounded high-impact public-header batch before editing;
- capture normalized declarations before and after cleanup;
- preserve public signatures unless an explicit reviewed exception is made;
- improve comments around ownership, lifetimes, errors, output buffers,
  options/results, backend behavior, and non-claims;
- align user-facing docs and generated-reference policy with the selected
  headers;
- avoid converting documentation cleanup into package, ABI, performance,
  backend-superiority, or broad platform claims;
- run the required quality gates if any `.c` or `.h` files change.

**End State:** Sprint 164 leaves behind:
- cleaned public header batch;
- declaration-preservation evidence;
- updated API reference and documentation cross-links;
- validation record for the changed code/documentation surface;
- Sprint 165 static-first package/API handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 164 project-plan estimate.

---

## Day 1: Sprint Intake And API Surface Inventory

**Title:** Sprint Intake
**Theme:** Establish Sprint 164 scope, artifact layout, and current public API
surfaces
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 164 section of
   `docs/planning/EPIC_14/PROJECT_PLAN.md`.
2. Review Sprint 158 generated API policy, Sprint 157 quality surface map, and
   Sprint 163 API-header handoff.
3. Create Sprint 164 working notes and artifact directory structure.
4. Inventory public headers, API reference inputs, tutorial, cookbook,
   solver-selection docs, README API sections, and maintainer guidance.
5. Record explicit non-goals for signature changes, ABI guarantees,
   package-manager support, shared-library support, runtime-loader behavior,
   backend superiority, and performance claims.
6. Write the Day 1 sprint-intake artifact.

### Deliverables
- Sprint 164 working-notes baseline
- artifact directory structure
- public API surface inventory
- non-goal and stop-condition register
- Day 1 sprint-intake artifact

### Completion Criteria
- Sprint 164 scope is tied to the Epic 14 project plan
- public header and API documentation owners are identified
- cleanup work is separated from package, ABI, and performance proof

---

## Day 2: Header Candidate Selection

**Title:** Header Selection
**Theme:** Select the bounded public-header batch for cleanup
**Time estimate:** 12 hours

### Tasks
1. Rank public headers by user impact, documentation ambiguity, claim risk,
   option/result complexity, and downstream visibility.
2. Identify headers with ownership, lifetime, output-buffer, error-status,
   backend, or solver-selection comment gaps.
3. Exclude headers whose cleanup would require signature changes or broad
   algorithm rewrites.
4. Select the final Sprint 164 header batch with rationale and deferred
   candidates.
5. Map each selected header to related README, tutorial, cookbook,
   solver-selection, generated-reference, and maintainer docs.
6. Write the Day 2 header-selection artifact.

### Deliverables
- selected header batch
- deferred header queue
- header-to-doc cross-link map
- cleanup-risk register
- Day 2 header-selection artifact

### Completion Criteria
- selected batch is small enough for declaration-preserving cleanup
- every selected header has a source-backed rationale
- deferred headers have explicit reasons

---

## Day 3: Declaration Baseline Design

**Title:** Baseline Design
**Theme:** Define how public declarations will be captured and compared
**Time estimate:** 12 hours

### Tasks
1. Inspect existing declaration, API reference, and header validation scripts
   or targets.
2. Decide the normalized declaration capture method for the selected header
   batch.
3. Define declaration-drift failure modes: removed declarations, renamed
   symbols, reordered function parameters, type changes, macro changes, enum
   changes, struct layout wording, and visibility changes.
4. Define acceptable non-signature edits: comments, section organization,
   examples, cross-links, and non-claim wording.
5. Record exact commands for baseline capture and later re-capture.
6. Write the Day 3 declaration-baseline-design artifact.

### Deliverables
- declaration capture method
- declaration drift taxonomy
- acceptable edit policy
- baseline command list
- Day 3 declaration-baseline-design artifact

### Completion Criteria
- declaration-preservation proof can be repeated
- signature drift rules are explicit before editing
- comment cleanup boundaries are clear

---

## Day 4: Declaration Baseline Capture

**Title:** Baseline Capture
**Theme:** Capture current declarations for the selected header batch
**Time estimate:** 12 hours

### Tasks
1. Run the selected declaration-capture commands.
2. Store or document normalized before-state declarations for the selected
   header batch.
3. Record current generated API-reference state for the selected headers.
4. Record current tutorial, cookbook, solver-selection, and README references
   to selected APIs.
5. Identify any pre-existing declaration or documentation inconsistencies.
6. Write the Day 4 declaration-baseline artifact.

### Deliverables
- before-state declaration evidence
- current API-reference state notes
- current documentation reference map
- pre-existing inconsistency list
- Day 4 declaration-baseline artifact

### Completion Criteria
- selected declarations are captured before edits
- existing inconsistencies are separated from introduced drift
- cleanup can proceed against a concrete baseline

---

## Day 5: Ownership And Lifetime Comment Cleanup

**Title:** Ownership Cleanup
**Theme:** Improve ownership, lifetime, allocation, and destruction wording
**Time estimate:** 12 hours

### Tasks
1. Edit selected headers to clarify caller-owned, library-owned, borrowed,
   transferred, and destroyed resources.
2. Clarify allocation and free responsibilities for matrices, vectors,
   factors, workspaces, options, and result structs in the selected batch.
3. Align nullability and lifetime wording with existing implementation
   behavior and tests.
4. Avoid adding ABI, allocator, package, or thread-safety claims not backed by
   existing evidence.
5. Update matching documentation cross-links when public header comments point
   users to docs.
6. Write the Day 5 ownership-cleanup artifact.

### Deliverables
- ownership/lifetime header cleanup
- matching documentation link updates where needed
- non-claim wording notes
- Day 5 ownership-cleanup artifact

### Completion Criteria
- ownership comments are clearer without declaration changes
- memory responsibility is consistent across selected headers
- unsupported ABI/package claims are not introduced

---

## Day 6: Error And Output-Buffer Comment Cleanup

**Title:** Error Cleanup
**Theme:** Clarify status codes, failure behavior, and output-buffer contracts
**Time estimate:** 12 hours

### Tasks
1. Clarify return-status meanings for selected APIs.
2. Clarify output-buffer size, initialization, overwrite, and partial-output
   behavior where supported by implementation behavior.
3. Clarify failure-path ownership for output pointers, result structs, and
   caller buffers.
4. Align examples or documentation references with the clarified contracts.
5. Avoid promising fail-closed behavior, diagnostics, or error classes beyond
   the current API contract.
6. Write the Day 6 error-output-cleanup artifact.

### Deliverables
- error/status comment cleanup
- output-buffer contract cleanup
- linked docs or example notes
- Day 6 error-output-cleanup artifact

### Completion Criteria
- users can distinguish inputs, outputs, and failure behavior
- no public declarations drift
- unsupported diagnostic guarantees are avoided

---

## Day 7: Options, Results, And Backend Wording Cleanup

**Title:** Options Cleanup
**Theme:** Clarify option/result structs and backend-selection boundaries
**Time estimate:** 12 hours

### Tasks
1. Clarify selected option structs, defaults, optional fields, and result
   interpretation.
2. Clarify backend-request, backend-selected, fallback, and built-in behavior
   only where the current implementation supports it.
3. Preserve non-superiority wording from Sprint 163 for backend and
   performance references.
4. Align solver-selection docs when headers reference solver or backend
   selection behavior.
5. Record any option/result ambiguity deferred out of the selected batch.
6. Write the Day 7 options-backend-cleanup artifact.

### Deliverables
- option/result comment cleanup
- backend-boundary wording cleanup
- solver-selection doc alignment notes
- deferred ambiguity queue
- Day 7 options-backend-cleanup artifact

### Completion Criteria
- option/result behavior is clearer for the selected batch
- backend wording does not imply superiority or external-library parity
- declaration preservation remains intact

---

## Day 8: Header Organization And Cross-Link Cleanup

**Title:** Cross-Link Cleanup
**Theme:** Align header sections, references, and user-facing navigation
**Time estimate:** 12 hours

### Tasks
1. Review selected headers for section ordering, duplicate comment blocks,
   stale references, and inconsistent terminology.
2. Update header cross-links to README, tutorial, cookbook, generated API
   docs, or solver-selection docs where useful.
3. Update README, tutorial, cookbook, and solver-selection references that no
   longer match selected header wording.
4. Keep visible user docs concise and avoid maintainer-only historical detail
   in public API surfaces.
5. Record remaining cross-link gaps deferred to later sprints.
6. Write the Day 8 cross-link-cleanup artifact.

### Deliverables
- header organization cleanup
- public docs cross-link updates
- stale-reference cleanup notes
- Day 8 cross-link-cleanup artifact

### Completion Criteria
- selected headers and public docs use consistent terminology
- users can navigate from API comments to maintained docs
- maintainer-only detail is kept out of public headers

---

## Day 9: Generated Reference Policy Check

**Title:** API Reference Check
**Theme:** Apply Sprint 158 generated-reference policy to the changed header
batch
**Time estimate:** 12 hours

### Tasks
1. Re-run or inspect generated API-reference tooling relevant to the selected
   headers.
2. Check generated reference output for stale names, broken links, unsupported
   claims, or missing selected-header coverage.
3. Verify that generated-reference publication wording matches the header
   cleanup and Sprint 158 policy.
4. Record generated-reference gaps that are policy or tooling issues rather
   than header-comment issues.
5. Update documentation or artifacts where generated-reference expectations
   need clarification.
6. Write the Day 9 generated-reference-check artifact.

### Deliverables
- generated API-reference check record
- stale or broken link notes
- Sprint 158 policy alignment notes
- Day 9 generated-reference-check artifact

### Completion Criteria
- changed headers are compatible with generated-reference policy
- generated docs do not introduce unsupported claims
- tooling gaps are separated from Sprint 164 header cleanup

---

## Day 10: Declaration Preservation Re-Capture

**Title:** Declaration Re-Capture
**Theme:** Prove selected cleanup preserved public declarations
**Time estimate:** 12 hours

### Tasks
1. Re-run the Day 4 declaration-capture commands after header cleanup.
2. Compare before and after normalized declarations.
3. Investigate any detected signature, macro, enum, struct, or visibility
   drift.
4. Revert or explicitly document any drift that was not intended and reviewed.
5. Record declaration-preservation evidence and command output.
6. Write the Day 10 declaration-preservation artifact.

### Deliverables
- after-state declaration evidence
- before/after comparison result
- drift investigation notes
- Day 10 declaration-preservation artifact

### Completion Criteria
- zero unreviewed public declaration drift
- any intentional drift is explicitly documented
- comment cleanup is proven declaration-preserving

---

## Day 11: Documentation Coherence Pass

**Title:** Docs Coherence
**Theme:** Align README, tutorial, cookbook, solver-selection, maintainer, and
API-reference docs
**Time estimate:** 12 hours

### Tasks
1. Review all docs touched or referenced by the selected header batch.
2. Align function names, option/result terms, ownership wording, status
   wording, backend boundaries, and solver-selection references.
3. Remove stale examples, stale cross-links, or inconsistent claims where
   found.
4. Ensure package, ABI, runtime-loader, backend-superiority, performance, and
   state-of-the-art non-claims remain intact.
5. Record doc surfaces changed and intentionally deferred.
6. Write the Day 11 documentation-coherence artifact.

### Deliverables
- coherent API documentation updates
- stale-reference cleanup notes
- retained non-claim trace
- Day 11 documentation-coherence artifact

### Completion Criteria
- public docs and selected headers tell the same API story
- user-facing docs do not contradict maintainer guidance
- unsupported claims remain excluded

---

## Day 12: Focused Validation And Quality Gates

**Title:** Focused Validation
**Theme:** Run required checks for the changed header, docs, and tooling
surface
**Time estimate:** 12 hours

### Tasks
1. Run header/API declaration preservation checks from Day 10.
2. Run generated-reference, documentation, schema, or report-index checks
   affected by Sprint 164 changes.
3. Run `make format`, `make lint`, and `make test` if any `.c` or `.h` files
   changed.
4. Run focused examples, compile checks, or install/package guards if selected
   docs or headers touch those surfaces.
5. Fix validation failures or stop with a clear blocker if a failure cannot be
   resolved safely.
6. Write the Day 12 focused-validation artifact.

### Deliverables
- validation command record
- pass/fail evidence
- fixes or blocker notes
- Day 12 focused-validation artifact

### Completion Criteria
- required gates pass before closeout work begins
- validation scope matches changed files
- no generated or cache artifacts are accidentally committed

---

## Day 13: Evidence Review And Sprint 165 Handoff

**Title:** Evidence Review
**Theme:** Review declaration, documentation, non-claim, and package/API
handoff evidence
**Time estimate:** 12 hours

### Tasks
1. Trace selected header cleanup to before/after declaration evidence and
   changed documentation surfaces.
2. Trace retained non-claims for ABI, package, runtime-loader, backend,
   performance, platform, and state-of-the-art wording.
3. Review diffs for stale generated outputs, unsupported API claims, or
   ambiguous ownership/error/lifetime wording.
4. Finalize Sprint 165 static-first package/API handoff.
5. Record residual header, docs, generated-reference, and package/API work
   that remains out of scope.
6. Write the Day 13 evidence-review artifact.

### Deliverables
- declaration-to-cleanup trace
- retained non-claim trace
- Sprint 165 package/API handoff
- residual queue
- Day 13 evidence-review artifact

### Completion Criteria
- header cleanup is reviewable end to end
- positive API wording is bounded by declaration evidence
- Sprint 165 handoff is ready

---

## Day 14: Closeout And Retrospective Prep

**Title:** Closeout
**Theme:** Finalize Sprint 164 artifacts, validation record, residuals, and
retrospective inputs
**Time estimate:** 10 hours

### Tasks
1. Re-run final targeted checks required by the changed-file surface.
2. Update Sprint 164 working notes with final decisions, commands, outputs,
   and non-claim boundaries.
3. Finalize closeout artifacts for selected headers, declaration preservation,
   documentation coherence, generated reference checks, validation, residuals,
   and Sprint 165 handoff.
4. Review changed files for stale generated outputs, signature drift, and
   unsupported API/package/performance claims.
5. Prepare retrospective inputs from artifacts and working notes.
6. Record the Day 14 closeout artifact.

### Deliverables
- final validation record
- declaration-preservation closeout notes
- residual queue
- complete working notes
- retrospective input set
- Day 14 closeout artifact

### Completion Criteria
- Sprint 164 deliverables are complete and traceable
- validation status is recorded with exact commands
- Sprint 165 static-first package/API handoff is ready
