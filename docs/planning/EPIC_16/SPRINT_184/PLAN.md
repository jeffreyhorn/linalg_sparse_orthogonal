# Sprint 184 Plan: Public Header Coherence Batch 3

**Sprint Duration:** 14 days
**Goal:** Normalize one more high-impact public header family without
declaration drift, improving API usability and generated documentation input.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 184 estimate in the Epic 16
project plan.

**Primary scope:** Confirm the selected public header family, likely QR/SVD or
LDLT based on prior adoption-risk work, then normalize contract wording,
declaration organization, examples, docs, mechanical guards, and validation for
that family.

**Non-goals:** Changing public API declarations, broadening the sprint to
unselected header families, rewriting implementation behavior, or adding
unsupported claims in documentation.

---

## Day 1: Sprint Intake and Prior-Art Review

**Title:** Header Family Intake
**Theme:** Establish the Sprint 184 context and reuse prior guardrails.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 184 section of the Epic 16 project plan and capture the
   acceptance boundaries for items 184.1 through 184.6.
2. Review previous declaration-preserving header cleanup batches for structure,
   wording style, guard patterns, and validation expectations.
3. Review Sprint 177 selection notes or related artifacts to understand why
   QR/SVD or LDLT is the likely next high-impact family.
4. Inventory candidate public headers, related examples, generated docs inputs,
   and existing guard coverage.
5. Start `WORKING_NOTES.md` with candidate families, assumptions, risks, and
   open questions.

### Deliverables

- Sprint intake notes with candidate family context and inherited guardrails.
- Initial inventory of candidate headers and documentation surfaces.
- Open-question list for family selection and declaration preservation.

### Completion Criteria

- The sprint scope is traceable to project-plan items 184.1 through 184.6.
- Prior header-cleanup patterns are identified and ready to reuse.
- Candidate family evidence is sufficient to begin baseline capture.

---

## Day 2: Declaration Baseline Capture

**Title:** Baseline and Preservation Rules
**Theme:** Capture the current public declarations before cleanup begins.
**Time estimate:** 12 hours

### Tasks

1. Capture current declarations for each candidate family, including functions,
   types, enums, macros, option/result structures, and lifecycle helpers.
2. Record declaration ordering, grouping, comments that affect generated docs,
   and any existing mismatch between header comments and examples.
3. Identify available declaration checksum, ABI, docs coverage, or unsupported
   claim guards that can protect the selected family.
4. Define declaration-preservation rules for the sprint, including what may be
   reordered and what must remain byte-for-byte stable.
5. Narrow the candidate list to the strongest family or document why a tie
   remains.

### Deliverables

- Declaration baseline for candidate public headers.
- Preservation rule set for Sprint 184 edits.
- Candidate-family comparison notes.

### Completion Criteria

- Baseline data can detect unintended declaration drift.
- The selected guard strategy is clear enough to use before edits.
- Remaining family-selection uncertainty is explicit and bounded.

---

## Day 3: Family Selection Decision and Contract Map

**Title:** Selected Family Contract Map
**Theme:** Confirm the header family and map its cleanup surface.
**Time estimate:** 12 hours

### Tasks

1. Confirm the Sprint 184 header family using adoption risk, public API
   visibility, documentation impact, and guard feasibility.
2. Record the selection decision, rejected alternatives, and non-goals in
   `WORKING_NOTES.md`.
3. Build a contract map for lifecycle, ownership, error-code, tolerance,
   workspace, option/result, and cancellation wording.
4. Identify inconsistent terms, duplicate concepts, unsupported claims, and
   stale examples for the selected family.
5. Draft the first cleanup checklist grouped by declaration section.

### Deliverables

- Selected header-family decision record.
- Contract wording map for the selected family.
- Cleanup checklist tied to declarations and docs surfaces.

### Completion Criteria

- One public header family is selected for the sprint.
- Cleanup areas are mapped without changing declarations.
- The checklist covers every contract wording category from item 184.2.

---

## Day 4: Lifecycle, Ownership, and Error Contracts

**Title:** Core Contract Cleanup
**Theme:** Normalize the highest-risk API contract wording first.
**Time estimate:** 12 hours

### Tasks

1. Normalize lifecycle wording for create/init, configure, solve/factorize,
   query, reset, and destroy flows in the selected header family.
2. Clarify ownership and borrowing rules for inputs, outputs, workspaces,
   options, result objects, and caller-managed buffers.
3. Align error-code language with established project terminology and public
   status semantics.
4. Remove ambiguous or unsupported behavior claims while preserving useful API
   guidance.
5. Run focused declaration-drift checks after the first documentation cleanup
   pass.

### Deliverables

- Updated lifecycle, ownership, and error-code comments for the selected
  family.
- Notes for any ambiguous behavior that requires follow-up.
- Focused declaration-preservation check results.

### Completion Criteria

- Core contract wording is consistent with existing public API conventions.
- No public declarations drift as a result of comment cleanup.
- Unsupported or ambiguous behavior claims are removed or bounded.

---

## Day 5: Tolerance, Workspace, Options, and Cancellation Contracts

**Title:** Advanced Contract Cleanup
**Theme:** Normalize operational contract wording across the selected family.
**Time estimate:** 12 hours

### Tasks

1. Normalize tolerance wording, including defaults, valid ranges, convergence
   interpretation, and result reporting where applicable.
2. Clarify workspace requirements, sizing expectations, reuse rules, and
   failure behavior.
3. Align option/result structure comments with current API patterns and avoid
   promises not enforced by implementation.
4. Normalize cancellation wording, including callback ownership, polling
   expectations, and status-code behavior where the family supports it.
5. Update the cleanup checklist and rerun focused declaration-drift checks.

### Deliverables

- Updated tolerance, workspace, option/result, and cancellation comments.
- Checklist entries showing completed contract categories.
- Focused declaration-preservation check results.

### Completion Criteria

- Every item 184.2 contract category has a documented cleanup pass.
- Contract wording is consistent across related declarations.
- Declaration preservation remains intact after comment updates.

---

## Day 6: Declaration Organization Design

**Title:** Organization Guardrail Design
**Theme:** Plan any declaration reordering before touching order-sensitive text.
**Time estimate:** 12 hours

### Tasks

1. Review current declaration grouping and identify confusing order, duplicate
   sections, or misplaced comments in the selected family.
2. Decide whether declaration organization changes are justified under the
   sprint guardrails.
3. Design coherent sections for lifecycle, options, workspace, operations,
   results, diagnostics, and cancellation where applicable.
4. Extend or configure declaration checks so proposed reordering is intentional
   and reviewable.
5. Document allowed organization changes and fallback behavior if guard checks
   expose risk.

### Deliverables

- Declaration organization proposal for the selected family.
- Guardrail notes for allowed reordering or grouping updates.
- Updated mechanical-check plan for organization changes.

### Completion Criteria

- Any declaration organization work has explicit guard coverage.
- The section structure improves readability without changing API meaning.
- Risky or unnecessary reorderings are deferred rather than forced.

---

## Day 7: Declaration Organization Pass

**Title:** Coherent Header Sections
**Theme:** Apply declaration organization changes under guard coverage.
**Time estimate:** 12 hours

### Tasks

1. Reorder declarations only where the Day 6 guardrails allow it.
2. Normalize section headings, comment placement, and nearby explanatory text
   for the selected family.
3. Keep lifecycle, options, workspace, operations, result, diagnostic, and
   cancellation declarations discoverable.
4. Refresh declaration baselines or checksum fixtures when the change is
   intentionally organization-only and mechanically reviewed.
5. Run focused declaration and docs-generation checks after the organization
   pass.

### Deliverables

- Organized selected-family public header sections.
- Updated declaration guard fixtures or notes, if intentional reordering is
  accepted.
- Focused check output captured in working notes.

### Completion Criteria

- Declaration organization changes are intentional, bounded, and checked.
- The header is easier to scan without public API drift.
- Guard outputs clearly distinguish comment/order changes from declaration
  changes.

---

## Day 8: Examples and Documentation Surface Audit

**Title:** Documentation Alignment Map
**Theme:** Find every downstream artifact that describes the selected family.
**Time estimate:** 12 hours

### Tasks

1. Search examples, tutorials, solver-selection guidance, cookbook material,
   and API reference inputs for the selected family.
2. Compare docs wording against the cleaned header contracts.
3. Identify stale snippets, unsupported claims, missing ownership notes, and
   outdated option/result examples.
4. Prioritize documentation edits by user impact and risk of contradiction.
5. Update `WORKING_NOTES.md` with the docs alignment checklist.

### Deliverables

- Complete docs and examples surface map for the selected family.
- Prioritized alignment checklist.
- Notes linking each docs issue to the cleaned header contract.

### Completion Criteria

- Every known selected-family docs surface has been reviewed or explicitly
  ruled out.
- Contradictions between headers and docs are captured before edits.
- The docs-edit plan is small enough to complete without widening sprint scope.

---

## Day 9: Examples and Tutorial Alignment

**Title:** Example Contract Alignment
**Theme:** Update executable and tutorial-facing usage guidance.
**Time estimate:** 12 hours

### Tasks

1. Update selected-family examples to match lifecycle, ownership, error, and
   option/result wording from the cleaned header.
2. Refresh tutorial snippets or narrative that describe selected-family setup,
   execution, diagnostics, or teardown.
3. Remove examples that imply unsupported defaults, hidden allocation behavior,
   or unstated tolerance semantics.
4. Keep example code buildable and consistent with existing style.
5. Run focused checks for touched examples or docs snippets where available.

### Deliverables

- Updated examples for the selected family.
- Updated tutorial material or documented confirmation that no tutorial edits
  are required.
- Focused check results for touched examples.

### Completion Criteria

- Examples no longer contradict the cleaned public header comments.
- Tutorial content uses the same lifecycle and ownership language as the
  selected family.
- Touched example code remains buildable or has a documented check path.

---

## Day 10: Solver-Selection, Cookbook, and API Reference Alignment

**Title:** Reference Documentation Alignment
**Theme:** Bring higher-level docs into agreement with the cleaned family.
**Time estimate:** 12 hours

### Tasks

1. Update solver-selection guidance for the selected family, including when to
   use it and when another family is more appropriate.
2. Align cookbook entries with cleaned contract wording and supported behavior.
3. Refresh API reference input text if generated documentation consumes header
   comments or supplemental markdown.
4. Remove unsupported performance, freshness, cancellation, or workspace claims.
5. Run docs-focused checks or generation smoke tests available in the repo.

### Deliverables

- Updated solver-selection, cookbook, or API reference material as needed.
- Unsupported-claim cleanup notes.
- Docs check or generation smoke-test results.

### Completion Criteria

- Higher-level docs agree with the selected-family public header contract.
- User-facing guidance remains specific without promising unsupported behavior.
- Docs checks pass or any unavailable checks are documented.

---

## Day 11: Mechanical Guard Implementation

**Title:** Declaration and Claim Guards
**Theme:** Add or extend guard coverage for the cleaned header family.
**Time estimate:** 12 hours

### Tasks

1. Add or extend declaration checksum coverage for the selected family if
   existing coverage is incomplete.
2. Add or extend docs coverage checks for required selected-family references.
3. Add or extend unsupported-claim guards for terms cleaned during the sprint.
4. Ensure guard failures provide actionable output for future reviewers.
5. Run the new or extended guards in isolation and capture results.

### Deliverables

- Mechanical guard updates covering declaration drift, docs coverage, or
  unsupported claims.
- Focused guard test output.
- Notes describing what each guard protects.

### Completion Criteria

- The selected family has mechanical protection aligned with item 184.5.
- Guard output is clear enough to diagnose failures.
- New or changed guards pass in focused execution.

---

## Day 12: Guard Hardening and Focused Validation

**Title:** Focused Validation Pass
**Theme:** Exercise changed headers, docs, examples, and guards before full CI.
**Time estimate:** 12 hours

### Tasks

1. Run all focused checks relevant to changed headers, examples, docs, and
   mechanical guards.
2. Fix any declaration-preservation, docs-coverage, unsupported-claim, or
   example failures.
3. Review diffs for accidental implementation changes or public API drift.
4. Update `WORKING_NOTES.md` with validation results and remaining risks.
5. Prepare the full validation command list for Day 13.

### Deliverables

- Focused validation results for all touched surfaces.
- Fixed guard, docs, or example regressions found by focused checks.
- Updated risk and validation notes.

### Completion Criteria

- Focused checks pass before full validation starts.
- Diffs are limited to the selected-family cleanup and required guard/docs
  changes.
- Remaining risks are known and suitable for full-suite validation.

---

## Day 13: Full Validation and Final Cleanup

**Title:** Full Quality Gate
**Theme:** Run repository-level checks and close validation gaps.
**Time estimate:** 12 hours

### Tasks

1. Run `make format`.
2. Run `make lint`.
3. Run `make test`.
4. Run docs checks and declaration guard checks required by the selected-family
   changes.
5. Fix any failures and rerun the failing command until all required checks
   pass.

### Deliverables

- Passing full quality-check results.
- Final cleanup commits or staged changes ready for review.
- Updated `WORKING_NOTES.md` validation section.

### Completion Criteria

- `make format`, `make lint`, and `make test` pass when headers or code are
  touched.
- Required docs and declaration guard checks pass.
- No unresolved validation failures remain.

---

## Day 14: Sprint Closeout and Handoff

**Title:** Retrospective-Ready Handoff
**Theme:** Package the selected-family cleanup for review and future sprints.
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 184 diffs against project-plan items 184.1 through 184.6.
2. Confirm the selected header family, changed docs surfaces, and guard updates
   are documented in `WORKING_NOTES.md`.
3. Prepare summary notes for the Sprint 184 retrospective, including decisions,
   validation, risks, and follow-up candidates.
4. Check for stale TODOs, unresolved open questions, or accidental scope
   expansion.
5. Produce final review-ready notes describing the cleanup and how to verify it.

### Deliverables

- Retrospective-ready Sprint 184 working notes.
- Review-ready change summary and validation summary.
- Follow-up list for any deferred header families or docs cleanup.

### Completion Criteria

- All Sprint 184 project-plan items have a documented outcome.
- The final state is ready for retrospective creation and PR preparation.
- Any deferred work is explicitly separated from the completed sprint scope.
