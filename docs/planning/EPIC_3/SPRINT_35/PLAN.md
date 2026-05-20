# Sprint 35 Plan: Public Docs, Header Examples & API-Usage Consistency

**Sprint Duration:** 14 days  
**Goal:** Make the public-facing guidance consistent with the current codebase by updating headers, README, examples, and usage notes to teach stable patterns instead of stale or brittle ones. This sprint implements the Sprint 35 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`.

**Starting Point:** Sprint 34 closed with reviewed Makefile/CMake quality wrappers, a Linux-first CI enforcement pass, and a documented dead-code workflow. Sprint 35 starts from that enforced baseline and shifts focus from internal quality gating to public-surface truthfulness: header examples, README/tutorial guidance, maintainer standards, API precondition language, and example-facing documentation.

**End State:** Sprint 35 leaves behind public headers, README/tutorial content, example snippets, and installation/quality docs that teach the current stable API patterns, especially designated initializer usage and accurate reorder/precondition guidance, while preserving clean example build validation under the existing quality flow.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 128 hours, matching the Sprint 35 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 35 Scope Audit & Baseline

**Title:** Docs Baseline  
**Theme:** Convert the Sprint 35 project-plan items into a precise public-surface audit scope  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 35 section of `docs/planning/EPIC_3/PROJECT_PLAN.md` plus the Sprint 34 handoff and retrospective so the sprint stays anchored to the documented public-doc consistency scope.
2. Confirm the Sprint 34 baseline invariants that must remain true during Sprint 35: reviewed quality wrappers remain green, current public examples compile, and the dead-code/operator command map is already in force.
3. Inventory the current public-surface documentation targets: installed headers, `README.md`, algorithm/tutorial docs, shipped examples, INSTALL guidance, and benchmark-facing docs that describe public usage.
4. Identify the most likely stale-pattern surfaces, especially positional options-struct examples, reorder-mode wording, and outdated precondition language.
5. Open Sprint 35 working notes and record the initial file map, validation commands, and likely cleanup batches.

### Deliverables
- Sprint 35 public-doc baseline
- Initial public-surface file inventory
- Named first-pass implementation surfaces for headers, README/tutorial docs, and examples

### Completion Criteria
- Sprint 35 starts from a documented Sprint 34 validated baseline
- The public-facing cleanup scope is separated from Sprint 34 enforcement work
- Likely stale-pattern targets are identified before edits begin

---

## Day 2: Header Example Audit

**Title:** Header Audit  
**Theme:** Identify the public header examples that still teach brittle or outdated patterns  
**Time estimate:** 8 hours

### Tasks
1. Audit installed headers for example code that still uses positional options-struct initialization or stale reorder/behavior wording.
2. Separate true public-example debt from private/internal comments that do not belong in Sprint 35.
3. Record which headers already follow the designated-initializer style and which still need conversion.
4. Note where header examples also carry implied precondition guidance that may need follow-on wording cleanup.
5. Write the audit note that defines the Day 4 and Day 5 header-edit batches.

### Deliverables
- Header example audit note
- Named header cleanup queue
- Initial keep/update/defer classification for public header example surfaces

### Completion Criteria
- The header cleanup queue is explicit before edits begin
- Positional-initializer debt is identified at the installed-header layer
- Precondition-language hotspots are noted for later passes

---

## Day 3: Public Initialization Standard Design

**Title:** Style Contract  
**Theme:** Define the maintainer-facing public example standard before rewriting docs  
**Time estimate:** 10 hours

### Tasks
1. Define the maintainer rule for how public options structs should be shown in headers, README snippets, examples, and tests.
2. Decide what counts as an acceptable exception to designated initializer usage in public-facing material.
3. Align the standard with the Sprint 31 and Sprint 34 patterns so public examples do not drift from reviewed implementation practice.
4. Decide where the maintainer rule should live and how Sprint 35 docs should reference it.
5. Write the design note that Day 4 through Day 8 will follow when rewriting examples and snippets.

### Deliverables
- Maintainer-facing initialization standard
- Exception policy for public examples
- Cross-surface style contract for headers, README, examples, and tests

### Completion Criteria
- The public example style rule is defined before broad doc edits begin
- The rule is concrete enough to guide both code-snippet and prose cleanup
- Exceptions are documented instead of handled ad hoc

---

## Day 4: Header Example Cleanup — Batch I

**Title:** Header Batch I  
**Theme:** Convert the first set of installed-header examples to stable designated-initializer patterns  
**Time estimate:** 10 hours

### Tasks
1. Update the first batch of public header examples from positional options-struct usage to designated initializers.
2. Keep the examples minimal and readable so the public docs improve, not just the warning/story consistency.
3. Preserve the current API behavior and avoid changing declarations or semantics while rewriting example code.
4. Cross-check touched examples against the actual public types and field names.
5. Record the completed batch and any follow-on consistency issues that spill into Day 5.

### Deliverables
- First installed-header example cleanup batch
- Notes on cross-header consistency issues
- Updated working notes for the public example standard in practice

### Completion Criteria
- The first batch of public header examples teaches the stable initialization pattern
- Example changes remain documentation-facing and semantically neutral
- Touched headers stay internally consistent with the current API

---

## Day 5: Header Example Cleanup — Batch II

**Title:** Header Batch II  
**Theme:** Finish the installed-header example cleanup and reconcile remaining public API wording drift  
**Time estimate:** 10 hours

### Tasks
1. Complete the remaining installed-header example conversions identified in the Day 2 audit.
2. Reconcile example wording across headers so the same concepts are described consistently.
3. Tighten any adjacent inline notes that still imply outdated option layouts or reorder capabilities.
4. Re-review the full touched header set as one public-surface pass.
5. Record the end state and any residual README/tutorial dependencies for later days.

### Deliverables
- Completed public-header example cleanup
- Header-level wording consistency pass
- Residual README/tutorial dependency list

### Completion Criteria
- Installed headers no longer teach the stale initialization pattern
- Public header wording is internally consistent across the touched API surface
- Remaining Sprint 35 work is clearly shifted to README/tutorial/example docs

---

## Day 6: README & Tutorial Consistency Audit

**Title:** README Audit  
**Theme:** Map all README/tutorial/example snippet drift against the current codebase  
**Time estimate:** 10 hours

### Tasks
1. Audit `README.md`, major tutorial/algorithm docs, and any user-facing usage snippets for stale reorder, initialization, or workflow guidance.
2. Compare the public documentation wording against the current supported reorder modes, option struct layouts, and reviewed Sprint 34 quality-flow names.
3. Identify duplicated or conflicting guidance across docs so later edits can converge instead of patching one file at a time.
4. Separate genuinely public guidance from maintainer-only workflow notes that belong elsewhere.
5. Write the audit note that turns the README/tutorial cleanup into named Day 7 and Day 8 batches.

### Deliverables
- README/tutorial audit note
- Named public-doc cleanup queue
- Conflict map for duplicated or inconsistent guidance

### Completion Criteria
- Public doc drift is identified before broad prose edits begin
- README/tutorial inconsistencies are mapped to concrete file batches
- Maintainer-only guidance is separated from user-facing guidance

---

## Day 7: README/Tutorial Rewrite Design

**Title:** Rewrite Design  
**Theme:** Define the rewrite shape for user-facing docs before implementation  
**Time estimate:** 8 hours

### Tasks
1. Decide how README, tutorial, and example-facing docs should divide responsibilities after the cleanup.
2. Choose the canonical wording for reorder-mode descriptions, initialization examples, and quality-command references.
3. Define where public precondition explanations should live so they are not duplicated or contradicted across files.
4. Plan the implementation order for Day 8 through Day 11 so prose, snippets, and example validation stay synchronized.
5. Write the design note for the user-facing doc rewrite.

### Deliverables
- README/tutorial rewrite plan
- Canonical wording decisions for public API usage
- File-by-file implementation order for the remaining documentation batches

### Completion Criteria
- The rewrite plan is concrete before public doc edits begin
- Canonical wording is chosen once instead of re-litigated per file
- Public precondition guidance has a planned home and structure

---

## Day 8: README & Tutorial Implementation

**Title:** README Rewrite  
**Theme:** Update the main user-facing docs to match current stable API usage  
**Time estimate:** 8 hours

### Tasks
1. Rewrite the primary README and tutorial snippets using the agreed designated-initializer and reorder wording.
2. Update quality-command references so user-facing docs name the current maintained workflow accurately.
3. Remove or rewrite stale examples that no longer match the public API or the current supported workflows.
4. Keep the docs readable for first-time users rather than turning them into maintainer changelogs.
5. Record the completed public-facing rewrite and any residual precondition wording issues for Day 9 and Day 10.

### Deliverables
- Updated README/tutorial docs
- Current stable public usage snippets
- Residual precondition-language queue for follow-on cleanup

### Completion Criteria
- Main user-facing docs match the current API behavior and workflow names
- Public snippets teach the stable initialization pattern
- Remaining public wording debt is narrowed to precondition-focused cleanup

---

## Day 9: API Precondition Language Audit

**Title:** Preconditions Audit  
**Theme:** Identify where public docs still underspecify or misstate safe usage assumptions  
**Time estimate:** 10 hours

### Tasks
1. Audit public headers and major docs for precondition statements that are stale, underspecified, or inconsistent with current behavior.
2. Focus on features whose safe usage story evolved during earlier sprints, especially where examples may still imply older assumptions.
3. Distinguish true public precondition guidance from low-value implementation detail that should not leak into user docs.
4. Map each finding to the right public surface: header note, README/tutorial prose, algorithm doc, or example comment.
5. Write the audit note that defines the Day 10 language-tightening batch.

### Deliverables
- API precondition-language audit note
- Named precondition cleanup queue
- Surface-by-surface mapping for the follow-on wording edits

### Completion Criteria
- Precondition debt is identified concretely before wording edits begin
- Public guidance is scoped to the right documents
- The queue distinguishes safety-critical wording from optional prose polish

---

## Day 10: API Precondition Language Implementation

**Title:** Preconditions Pass  
**Theme:** Tighten the public safety/usage language without changing API semantics  
**Time estimate:** 10 hours

### Tasks
1. Update the public docs and headers identified in the Day 9 audit with clearer, more accurate precondition wording.
2. Remove legacy phrasing that implies outdated constraints or hides current expectations.
3. Keep the wording concise and user-facing rather than overloading docs with internal rationale.
4. Reconcile adjacent examples and notes so the tightened precondition language matches the surrounding snippets.
5. Record the completed pass and any final installation/docs-polish dependencies for Day 11.

### Deliverables
- Updated public precondition language
- Reconciled examples and adjacent usage notes
- Final installation/docs-polish queue

### Completion Criteria
- Public docs now state safe usage expectations accurately
- Precondition language is consistent with nearby examples and current API behavior
- Remaining Sprint 35 work is limited to install/quality docs and validation

---

## Day 11: Installation & Quality Docs Polish

**Title:** Docs Polish  
**Theme:** Bring INSTALL, benchmark, and quality-facing docs into line with the Sprint 35 public-surface rewrite  
**Time estimate:** 8 hours

### Tasks
1. Update INSTALL and related project docs where the public workflow names or example patterns changed during Sprint 35.
2. Refresh benchmark/quality docs where they mention public usage or quality commands in ways that no longer match the current repo state.
3. Ensure the maintainer initialization standard and the user-facing public examples do not contradict each other.
4. Remove low-value duplicated wording that survived the earlier README/tutorial rewrite.
5. Record the final documentation-polish pass and the exact validation scope for Days 12 and 13.

### Deliverables
- Updated INSTALL/benchmark/quality docs
- Final cross-doc consistency pass
- Validation-scope note for the last sprint days

### Completion Criteria
- Supporting docs reflect the Sprint 35 public-surface rewrite
- Maintainer and user-facing guidance no longer conflict
- The remaining work is validation and closeout, not more open-ended rewrite drift

---

## Day 12: Example Build & Snippet Validation

**Title:** Example Validation  
**Theme:** Prove that the rewritten public examples and shipped programs still build cleanly  
**Time estimate:** 10 hours

### Tasks
1. Build the shipped example programs touched or affected by the Sprint 35 cleanup.
2. Validate that rewritten public snippets still match the real public types, supported options, and command names.
3. Re-check benchmark/example compile-only coverage where public docs now depend on those tools as usage references.
4. Capture any last doc/example mismatches and resolve them while the validation context is fresh.
5. Record the example-validation results and the exact end-state commands needed for Day 13.

### Deliverables
- Example build-validation record
- Snippet-to-code consistency check
- Final cleanup of any validation-surfaced doc drift

### Completion Criteria
- Shipped examples compile cleanly after the public-doc rewrite
- Public snippets remain truthful to the current codebase
- Day 13 can focus on the final full validation sweep instead of unresolved drift

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Re-run the reviewed quality and example-facing flows against the Sprint 35 documentation changes  
**Time estimate:** 10 hours

### Tasks
1. Run the Sprint 35 validation set, including the maintained quality commands and the example/benchmark compile checks touched by the public-doc rewrite.
2. Reconfirm that the Sprint 34 reviewed-quality wrappers still pass after the documentation and example updates.
3. Reconfirm that the public examples and any documentation-dependent build surfaces remain green.
4. Capture timings, success state, and any final observations needed for closeout.
5. Update Sprint 35 notes with the final validated end state before handoff docs begin.

### Deliverables
- Full Sprint 35 validation record
- Reconfirmed reviewed-quality and example-build state
- Final validated baseline for closeout

### Completion Criteria
- All intended Sprint 35 validation flows pass
- Public-doc changes do not regress the reviewed quality baseline
- The end state is fully measured before closeout

---

## Day 14: Closeout, Handoff & Forward Queue

**Title:** Sprint Closeout  
**Theme:** Package Sprint 35’s public-surface cleanup for Sprint 36 and later Epic 3 phases  
**Time estimate:** 8 hours

### Tasks
1. Write the Sprint 35 handoff summarizing the shipped public-doc, header-example, maintainer-standard, and validation outcomes.
2. Write the Sprint 35 retrospective covering what worked, what remained intentionally deferred, and which later sprints inherit the remaining public-surface work.
3. Route any concrete deferred items into Sprint 36 or later sections of `docs/planning/EPIC_3/PROJECT_PLAN.md`.
4. Preserve the Sprint 34 reviewed-quality baseline explicitly in the closeout so later doc or portability work does not regress it casually.
5. Ensure the closeout documents any remaining public-surface exclusions or deferred doc questions that still matter after Sprint 35.

### Deliverables
- `HANDOFF.md`
- `RETROSPECTIVE.md`
- Forward-plan updates for deferred Sprint 36+ work if needed

### Completion Criteria
- Sprint 35 artifacts explain both the shipped public-surface cleanup and any remaining phased limitations
- Later sprints can recover the public-doc consistency contract without rereading the full sprint history
- Sprint 35 closes with a clear validated baseline for the next Epic 3 phase
