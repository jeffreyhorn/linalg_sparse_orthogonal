# Sprint 48 Plan: Quality-Contract Simplification, README Reduction & Maintainer Guide

**Sprint Duration:** 14 days  
**Goal:** Reduce duplication across the Makefile, scripts, CI-facing quality surfaces, README, headers, and tutorial prose by moving maintainer policy into a clearer home while keeping README more user-facing. This sprint implements the Sprint 48 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`.

**Starting Point:** Sprint 47 closed with the benchmark/example/tooling auxiliary surface tightened up, but Epic 4 still carries duplication across README, quality-command prose, maintainer expectations, and doc/runtime cross-references. The current quality contract is effective but spread across too many places. Sprint 48 begins from that validated baseline and turns documentation ownership and quality-policy simplification into a bounded maintainability target without reopening core solver behavior or CI architecture.

**End State:** Sprint 48 leaves behind a new maintainer-facing policy home, a smaller and more user-facing README, clearer ownership between quality commands and their documentation, reduced duplication across README/tutorial/headers/maintainer guidance, and a validation record showing the simplified quality contract still matches the live reviewed/dead-code command surfaces.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 124 hours, matching the Sprint 48 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 48 Scope Audit & Baseline Refresh

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 48 project-plan items into a bounded documentation-ownership and quality-contract execution map  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 48 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`, the Sprint 40 validation anchor, the Sprint 42 lifecycle/cancellation closeout, and the Sprint 47 benchmark/example/tooling closeout.
2. Reconfirm the preserved constraints Sprint 48 must not reopen:
   - preserve Sprint 40 validation truthfulness
   - preserve the current reviewed/dead-code command semantics unless explicitly simplified
   - keep README user-facing rather than turning it into a maintainer policy dump again
   - avoid broad CI or workflow redesign
3. Define the Sprint 48 workstreams explicitly:
   - maintainer-policy home design
   - README reduction
   - maintainer-guide implementation
   - tutorial/header cross-reference reconciliation
   - quality-contract ownership simplification
   - docs sanity sweep
   - validation closeout
4. Record the highest-risk duplication seams:
   - reviewed baseline semantics described in multiple places
   - dead-code meaning and expectations repeated inconsistently
   - lifecycle/cancellation caveats split across headers, README, and tutorial
   - maintainer norms embedded in user-facing docs
5. Open Sprint 48 working notes and record scope, assumptions, and initial landing order.

### Deliverables
- Sprint 48 scope inventory
- Documentation-ownership / quality-contract workstream map
- Working-notes baseline assumptions

### Completion Criteria
- Sprint 48 starts from the documented Epic 4 baseline rather than ad hoc doc trimming
- Preserved validation and ownership constraints are explicit before implementation begins
- The documentation and quality-contract targets are named before edits start

---

## Day 2: Documentation and Quality-Contract Surface Inventory

**Title:** Surface Inventory  
**Theme:** Re-map the README, maintainer-policy, header, tutorial, and quality-command seams before choosing the landing order  
**Time estimate:** 8 hours

### Tasks
1. Refresh the live seam inventory for:
   - top-level `README.md`
   - quality-command help text and nearby docs
   - maintainer-facing notes already embedded in sprint artifacts or code comments
   - touched headers and tutorial prose carrying behavioral caveats
2. Classify the current duplication into bounded buckets:
   - quality-command ownership drift
   - README user-vs-maintainer scope drift
   - tutorial/header duplication
   - lifecycle/behavior caveat duplication
   - maintainer norms with no stable policy home
3. Identify the strongest “move to maintainer guide” candidates versus content that should stay:
   - in README
   - in headers
   - in tutorial
4. Confirm the first implementation order:
   - maintainer-guide design first
   - README reduction second
   - policy-home implementation next
   - cross-reference reconciliation after the policy home is stable
5. Write the surface-inventory artifact.

### Deliverables
- Refreshed documentation / quality-contract seam inventory
- Shared-policy-home vs local-doc classification
- First landing-order notes

### Completion Criteria
- The current duplication is reduced to named ownership seams
- README/user-facing content is separated from maintainer-policy candidates
- Later implementation order is grounded in the live docs and command surfaces

---

## Day 3: Maintainer-Guide Design

**Title:** Guide Design  
**Theme:** Define the new maintainer-facing policy home for quality-contract and documentation-ownership guidance  
**Time estimate:** 8 hours

### Tasks
1. Design the target maintainer-policy document set:
   - one main maintainer guide or a small bounded guide cluster
   - location and naming
   - intended audience and scope
2. Define what policy belongs there:
   - reviewed baseline use
   - warning authority
   - dead-code meaning
   - lifecycle/cancellation expectations
   - documentation ownership rules
   - designated-initializer / style norms where still relevant
3. Decide what should explicitly remain outside the guide:
   - end-user quick-start material
   - API reference content that belongs in headers
   - benchmark/example usage details that belong in local READMEs
4. Record cross-reference rules between README, tutorial, headers, and the new guide.
5. Write the maintainer-guide design artifact.

### Deliverables
- Maintainer-guide design
- Policy-home scope definition
- Cross-reference ownership rules

### Completion Criteria
- Sprint 48 has a concrete maintainer-policy target before doc movement begins
- Maintainer-facing and user-facing content boundaries are explicit
- The design stays bounded away from broad docs-site or CI redesign

---

## Day 4: Landing and Validation Design

**Title:** Landing Design  
**Theme:** Bound the documentation redistribution batches and the focused validation shape before edits begin  
**Time estimate:** 8 hours

### Tasks
1. Define the validation shape for Sprint 48 implementation days:
   - docs-only days should use targeted sanity checks
   - script/command-surface days should use focused command validation
   - any `*.c` / `*.h` changes still require the full required gate
2. Decide when the stronger reviewed baseline should be rerun:
   - quality-contract simplification days
   - final validation sweep
3. Bound the intended landing order:
   - README reduction
   - maintainer-guide implementation
   - tutorial/header cross-reference pass
   - quality-contract simplification
   - docs sanity sweep
4. Record explicit out-of-scope items:
   - broad CI redesign
   - dead-code workflow redesign
   - broad tutorial rewrite
   - large benchmark/example content expansion
5. Write the landing/validation design artifact.

### Deliverables
- Validation-plan artifact
- Mid-sprint landing order
- Explicit out-of-scope notes

### Completion Criteria
- The sprint has a clear validation contract before documentation movement begins
- The implementation batches are sequenced from the live ownership map
- Scope boundaries are explicit before edits start

---

## Day 5: README Reduction Design and First Pass

**Title:** README Pass I  
**Theme:** Trim README toward a stronger user/operator entry point while preserving essential runtime guidance  
**Time estimate:** 12 hours

### Tasks
1. Identify which README sections should stay user-facing:
   - quick-start
   - build/run essentials
   - high-level feature map
   - direct links to deeper docs
2. Identify which README sections should move to the new maintainer-policy home or to local docs.
3. Land the first bounded README reduction pass.
4. Preserve important user/operator clarity while removing maintainer-only duplication.
5. Run targeted doc sanity checks on the touched README links and references.

### Deliverables
- First README reduction pass
- README user-facing ownership cleanup
- Targeted README sanity-check notes

### Completion Criteria
- README is smaller and more user-facing after the first pass
- Maintainer-policy duplication is materially reduced
- The touched references remain valid

---

## Day 6: Maintainer-Guide Implementation Batch

**Title:** Guide Batch  
**Theme:** Add the maintainer-facing policy home and move duplicated maintainer guidance into it  
**Time estimate:** 12 hours

### Tasks
1. Create the new maintainer guide using the Day 3 design.
2. Move bounded policy content out of README and other inappropriate homes into the guide:
   - reviewed baseline semantics
   - dead-code meaning
   - documentation ownership expectations
   - relevant maintainer norms
3. Add the minimal necessary cross-references from README or other touched docs to the new guide.
4. Keep the batch bounded:
   - no broad prose rewrite everywhere yet
   - no CI/workflow redesign
5. Run targeted doc sanity checks on the new guide and the moved-reference paths.

### Deliverables
- New maintainer-facing policy home
- Relocated maintainer guidance
- Cross-reference notes for the new guide

### Completion Criteria
- A real maintainer-policy home exists in the repo
- Duplicated maintainer guidance has moved out of README where appropriate
- The touched references remain valid

---

## Day 7: Post-Guide Audit

**Title:** Guide Audit  
**Theme:** Audit the post-Day-6 documentation state to confirm what remains for tutorial/header reconciliation and quality-contract simplification  
**Time estimate:** 8 hours

### Tasks
1. Review the post-Day-6 state and identify remaining duplication in:
   - headers
   - tutorial prose
   - README
   - command-surface documentation
2. Separate:
   - cross-reference fixes that should land next
   - content that should stay local and not move again
   - lower-priority cleanup that should stay outside Sprint 48
3. Confirm the bounded Day 8 target set for tutorial/header cross-reference cleanup.
4. Record any maintainer-guide scope adjustments still needed before the quality-contract batch.
5. Write the post-guide audit artifact.

### Deliverables
- Post-guide audit
- Bounded tutorial/header target list
- Quality-contract follow-on notes

### Completion Criteria
- The remaining documentation queue is concrete rather than generic
- Cross-reference targets are explicit before the next batch
- The sprint remains bounded around ownership simplification

---

## Day 8: Tutorial and Header Cross-Reference Batch

**Title:** Cross-Reference Batch  
**Theme:** Reconcile headers, tutorial prose, README, and maintainer guidance so behavioral caveats are linked instead of repeated  
**Time estimate:** 8 hours

### Tasks
1. Land the bounded cross-reference fixes identified on Day 7.
2. Reduce inconsistent duplication across:
   - touched public headers
   - tutorial prose
   - README
   - maintainer guide
3. Preserve local behavioral caveats where they are API-relevant, but replace duplicated long explanations with stable references where appropriate.
4. Keep the batch bounded away from broad tutorial rewriting.
5. Run targeted doc-reference sanity checks on the touched files.

### Deliverables
- Tutorial/header cross-reference cleanup
- Reduced duplication across touched docs
- Targeted reference-check notes

### Completion Criteria
- Touched behavioral caveats are linked more consistently than before
- Header/tutorial/README duplication is materially reduced
- The touched references remain valid

---

## Day 9: Quality-Contract Simplification Audit

**Title:** Contract Audit  
**Theme:** Audit the live Makefile/script/doc ownership around quality commands before simplifying the remaining contract  
**Time estimate:** 8 hours

### Tasks
1. Re-read the live quality-command ownership surfaces:
   - Makefile quality targets
   - dead-code script behavior
   - touched guide/README/tutorial text
2. Identify remaining duplication or drift in:
   - reviewed baseline wording
   - dead-code meaning and expectations
   - command-to-doc ownership
   - “what to run when” guidance
3. Separate:
   - direct simplification candidates
   - wording that should stay local to command surfaces
   - later support-tooling work that is outside Sprint 48
4. Confirm the bounded Day 10 target set for quality-contract simplification.
5. Write the quality-contract audit artifact.

### Deliverables
- Quality-contract ownership audit
- Bounded simplification target list
- Command-surface follow-on notes

### Completion Criteria
- The remaining quality-contract queue is concrete rather than generic
- Simplification targets are explicit before the next batch
- The sprint stays focused on duplication reduction rather than command redesign

---

## Day 10: Quality-Contract Simplification Batch

**Title:** Contract Batch  
**Theme:** Tighten ownership between Makefile commands, script behavior, guide text, and docs so fewer coordinated edits are needed later  
**Time estimate:** 8 hours

### Tasks
1. Land the bounded quality-contract simplifications identified on Day 9.
2. Clarify where the authoritative explanation now lives for:
   - reviewed baseline use
   - dead-code expectations
   - maintainer-facing quality-command guidance
3. Remove or tighten duplicated wording across the touched command-surface docs.
4. Keep the batch bounded:
   - no workflow redesign
   - no change to substantive quality-command meaning unless required for simplification clarity
5. Run focused sanity checks on the touched command/documentation surfaces.

### Deliverables
- Simplified quality-contract documentation ownership
- Reduced duplication across touched command-surface docs
- Focused command-surface sanity-check notes

### Completion Criteria
- Quality-command ownership is clearer than before
- Future command/documentation edits should require fewer coordinated changes
- The touched command/documentation references remain valid

---

## Day 11: Documentation Sanity Sweep Design and First Pass

**Title:** Docs Sweep I  
**Theme:** Re-read the redistributed documentation set and tighten the highest-signal consistency gaps  
**Time estimate:** 10 hours

### Tasks
1. Re-read the touched Sprint 48 documentation set end-to-end:
   - README
   - maintainer guide
   - touched tutorial sections
   - touched headers
   - local benchmark/example docs if referenced
2. Identify the highest-signal remaining consistency issues:
   - stale wording
   - mismatched links
   - duplicated caveat text
   - missing handoff guidance between user-facing and maintainer-facing docs
3. Land the first bounded sanity-sweep cleanup pass.
4. Confirm whether any remaining issues should stay for Day 12 only.
5. Record the sanity-sweep findings.

### Deliverables
- First documentation sanity-sweep pass
- Remaining consistency-gap notes
- Working-notes summary of the re-read

### Completion Criteria
- The touched docs read more coherently as a set
- Remaining issues are small and explicit
- No broad rewrite is required to reach Sprint 48 closeout

---

## Day 12: Documentation Sanity Sweep Final Pass

**Title:** Docs Sweep II  
**Theme:** Finish the bounded consistency cleanup across the redistributed Sprint 48 documentation set  
**Time estimate:** 6 hours

### Tasks
1. Land the remaining small consistency fixes from Day 11.
2. Reconfirm that README, maintainer guide, tutorial, headers, and local docs point to each other cleanly.
3. Tighten wording where duplication has merely shifted rather than disappeared.
4. Keep the batch bounded to the already-touched surfaces.
5. Record the final documentation-sanity outcome.

### Deliverables
- Final bounded documentation sanity-sweep pass
- Cleaned cross-reference wording
- Final sanity-sweep notes

### Completion Criteria
- The redistributed docs are internally consistent enough for validation closeout
- No obvious duplicate policy block remains in the touched surfaces
- The sprint is ready for the final validation sweep

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the focused Sprint 48 quality gate and the reviewed/dead-code/doc-reference sanity checks for the touched surfaces  
**Time estimate:** 10 hours

### Tasks
1. Run the full required validation gate for any touched `*.c` / `*.h` files if applicable; otherwise run the focused Sprint 48 doc/tool sanity set.
2. Run the stronger reviewed baseline:
   - `make quality-review-full`
3. Run the targeted sanity checks justified by the touched Sprint 48 surfaces:
   - reviewed/dead-code command surfaces
   - relevant doc-reference checks
   - touched script or helper validation
4. Reconcile any small validation issues that surface and rerun the authoritative checks if needed.
5. Record measured results, parity checks, and remaining residual risks in the validation artifact and working notes.

### Deliverables
- Full Sprint 48 validation record
- Targeted reviewed/dead-code/doc-reference follow-on results
- Reconciled measured baseline notes

### Completion Criteria
- The appropriate required gate passes for the touched file types
- The stronger reviewed baseline passes
- The targeted Sprint 48 follow-ons are recorded and green

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Synthesize the Sprint 48 results, residual queue, and handoff constraints for the next Epic 4 phase  
**Time estimate:** 10 hours

### Tasks
1. Summarize what Sprint 48 actually landed across:
   - maintainer-policy home
   - README reduction
   - tutorial/header cross-reference cleanup
   - quality-contract simplification
   - documentation sanity sweep
2. Summarize the final validated baseline and the remaining inherited queue.
3. Check whether Sprint 48 surfaced any roadmap change large enough to require a `PROJECT_PLAN.md` update.
4. Write the closeout-and-handoff artifact.
5. Record the final Sprint 48 synthesis in working notes.

### Deliverables
- Sprint 48 closeout artifact
- Final working-notes handoff
- Explicit residual queue for later Epic 4 work

### Completion Criteria
- Sprint 48 closes with an explicit documentation-ownership and quality-contract handoff
- The final validated baseline is recorded
- Any remaining work is bounded and clearly deferred
