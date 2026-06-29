# Sprint 96 Plan: Large-Source & Giant-Test Maintainability Phase 6

**Sprint Duration:** 14 days
**Goal:** Reduce the largest remaining mixed-role implementation and proof
hotspots after Epic 8 cleanup and the Epic 9 narrative/package changes. This
sprint implements the Sprint 96 section of
`docs/planning/EPIC_9/PROJECT_PLAN.md`.

**Starting Point:** Sprint 96 begins from:
- the Sprint 90 hotspot map
- the Epic 8 maintainability and runtime cleanup baseline
- the Sprint 94 capability-surface modernization work
- the Sprint 95 public narrative, support-surface, and selected proof-owner
  naming cleanup
- a codebase where the remaining maintainability pressure is concentrated in
  large mixed-role implementation owners and dense proof-owner tests

The strongest Sprint 96 pressure is not broad refactoring. It is to reduce
review and reasoning cost on the biggest remaining owners by:
- re-ranking the live implementation and proof hotspots after Sprint 95
- defining one bounded extraction plan before moving code
- landing one direct-family source cleanup batch
- landing one solver/algorithm source cleanup batch
- reducing one or two giant proof-owner concentrations without weakening
  coverage
- removing stale implementation chronology from touched files while preserving
  durable rationale
- closing from full validation and a bounded Sprint 97 handoff queue

**End State:** Sprint 96 leaves behind:
- a refreshed hotspot ranking for implementation and proof-owner surfaces
- a source extraction design for the selected large owners
- smaller or clearer touched direct-family implementation ownership
- smaller or clearer touched solver/algorithm implementation ownership
- reduced proof-owner concentration in selected giant tests
- cleaned internal comments and rationale on touched files
- a validated Sprint 96 closeout package and Sprint 97 handoff queue

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 96 project-plan estimate.

---

## Day 1: Sprint 96 Scope & Hotspot Baseline

**Title:** Hotspot Baseline
**Theme:** Turn the Sprint 96 project-plan section and prior closeouts into one
bounded maintainability package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 96 section of
   `docs/planning/EPIC_9/PROJECT_PLAN.md`.
2. Re-read the Sprint 90 hotspot map and Sprint 95 closeout/handoff queue.
3. Inventory the current large-source and giant-test candidates:
   - `src/sparse_ldlt_csc.c`
   - `src/sparse_iterative.c`
   - adjacent direct-family implementation owners
   - QR, eigensolver, SVD, and reorder/algorithm owners
   - the largest retained `tests/test_*.c` proof owners
4. Separate implementation hotspots from proof-owner, benchmark, support-doc,
   and historical-planning surfaces.
5. Open Sprint 96 working notes and record validation expectations for code,
   header, build, and docs-only days.

### Deliverables
- Sprint 96 scope inventory
- starting large-owner candidate list
- Sprint 96 working-notes baseline

### Completion Criteria
- Sprint 96 starts from the merged Sprint 95 end state
- the likely implementation and proof hotspots are visible before ranking
- validation requirements are explicit before code movement begins

---

## Day 2: Hotspot Rerank

**Title:** Hotspot Rerank
**Theme:** Rank the largest remaining implementation and proof hotspots by
review cost, ownership ambiguity, and extraction risk
**Time estimate:** 12 hours

### Tasks
1. Measure and inspect the candidate hotspots identified on Day 1.
2. Rank candidates by:
   - file size and local complexity
   - number of mixed responsibilities
   - coupling to public API or build surfaces
   - proof coverage density
   - validation blast radius
3. Identify which candidates are source extraction work, test architecture
   work, comment/rationale cleanup, or intentionally deferred.
4. Define the fix-now queue for Sprint 96.
5. Write a Day 2 rerank artifact with the selected implementation and proof
   centers.

### Deliverables
- ranked hotspot rerank artifact
- fix-now vs residual maintainability queue
- extraction-risk and validation-risk notes

### Completion Criteria
- Sprint 96 has one ranked live hotspot map
- the strongest source and proof-owner cleanup candidates are explicit
- broad or speculative refactors are separated from bounded Sprint 96 work

---

## Day 3: Source Extraction Design

**Title:** Extraction Design
**Theme:** Define the next bounded source extraction plan before moving code
**Time estimate:** 12 hours

### Tasks
1. Review the top source hotspots from Day 2, especially:
   - `src/sparse_ldlt_csc.c`
   - `src/sparse_iterative.c`
   - selected adjacent large owners
2. Map each candidate's responsibilities, internal helper clusters, static
   dependencies, and test owners.
3. Select one direct-family extraction target and one solver/algorithm cleanup
   target.
4. Define boundaries for new files, internal headers, or helper ownership only
   where they reduce real complexity.
5. Write the extraction design artifact and validation plan.

### Deliverables
- source extraction design artifact
- selected direct-family cleanup boundary
- selected solver/algorithm cleanup boundary
- validation plan for implementation days

### Completion Criteria
- no code is moved before ownership boundaries are documented
- each selected cleanup has a bounded file and test impact
- deferred hotspots have explicit reasons for not moving in Sprint 96

---

## Day 4: Direct-Family Cleanup Boundary Freeze

**Title:** Direct Boundary
**Theme:** Freeze the first direct-family implementation cleanup batch and its
proof-owner expectations
**Time estimate:** 12 hours

### Tasks
1. Re-read the selected direct-family hotspot against the Day 3 design.
2. Identify the exact helper cluster, comment block, or internal interface that
   should move or be simplified first.
3. Check current proof owners for that cluster and decide which targeted tests
   must be rerun after the batch.
4. Decide whether the cleanup needs new source files, internal headers, or only
   local decomposition.
5. Record the Day 4 direct-family landing plan.

### Deliverables
- direct-family cleanup boundary artifact
- targeted proof-owner rerun list
- exact touched-file plan

### Completion Criteria
- the direct-family batch is ready to implement without scope drift
- validation expectations are clear before source edits
- unrelated direct-family behavior remains explicitly out of scope

---

## Day 5: Direct-Family Source Cleanup Batch 1

**Title:** Direct Cleanup 1
**Theme:** Land the first bounded direct-family source ownership cleanup
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 4 direct-family cleanup.
2. Keep public behavior and API signatures unchanged unless the Day 4 plan
   explicitly allows otherwise.
3. Update internal comments only where ownership or durable rationale changes.
4. Run focused compile or test checks during development.
5. Record the implementation batch and any follow-up discovered during landing.

### Deliverables
- first direct-family source cleanup batch
- updated internal comments or helper ownership notes
- implementation notes for touched files

### Completion Criteria
- the selected direct-family hotspot is smaller or clearer
- behavior-preserving intent is reflected in tests and comments
- no unrelated direct-family refactor is included

---

## Day 6: Direct-Family Source Cleanup Batch 2

**Title:** Direct Cleanup 2
**Theme:** Complete the direct-family cleanup and reconcile proof ownership
**Time estimate:** 12 hours

### Tasks
1. Finish any remaining direct-family cleanup from Day 5.
2. Re-run the targeted direct-family proof owners.
3. Update adjacent maintainer comments or docs only if ownership changed.
4. Check for stale helper names, outdated comments, or duplicated rationale in
   touched direct-family files.
5. Write the direct-family cleanup artifact.

### Deliverables
- completed direct-family cleanup batch
- direct-family proof-owner validation notes
- residual direct-family queue

### Completion Criteria
- direct-family cleanup is coherent across source, comments, and proof owners
- direct-family residuals are explicit
- required code-day validation expectations are met or queued for Day 13

---

## Day 7: Solver/Algorithm Cleanup Boundary Freeze

**Title:** Solver Boundary
**Theme:** Freeze the second implementation cleanup batch in one major
solver/algorithm hotspot
**Time estimate:** 12 hours

### Tasks
1. Re-read the selected solver/algorithm hotspot from Day 3.
2. Decide the bounded cleanup target in one family:
   - iterative solvers
   - QR
   - eigensolvers
   - SVD
   - another ranked algorithm owner from Day 2
3. Map the helper dependencies and proof owners for that target.
4. Define what should remain untouched to avoid cross-family churn.
5. Write the Day 7 solver/algorithm landing plan.

### Deliverables
- solver/algorithm cleanup boundary artifact
- selected implementation target
- proof-owner and benchmark sanity-check list

### Completion Criteria
- the second implementation batch has one clear owner
- public behavior and proof expectations are understood
- unrelated solver families are explicitly out of scope

---

## Day 8: Solver/Algorithm Source Cleanup Batch 1

**Title:** Solver Cleanup 1
**Theme:** Land the first bounded cleanup in the selected solver/algorithm
hotspot
**Time estimate:** 12 hours

### Tasks
1. Implement the first part of the Day 7 solver/algorithm cleanup.
2. Keep API behavior and numerical contracts unchanged.
3. Preserve durable algorithm rationale while removing stale chronology in
   touched comments.
4. Run focused build/test checks on the touched family.
5. Record any second-pass cleanup or proof-owner adjustment needed for Day 9.

### Deliverables
- first solver/algorithm source cleanup batch
- focused validation notes
- follow-up list for the second pass

### Completion Criteria
- the selected solver/algorithm owner is smaller or easier to reason about
- tests or focused checks support the behavior-preserving cleanup
- no broad algorithm rewrite is introduced

---

## Day 9: Solver/Algorithm Source Cleanup Batch 2

**Title:** Solver Cleanup 2
**Theme:** Finish the selected solver/algorithm cleanup and reconcile adjacent
rationale
**Time estimate:** 12 hours

### Tasks
1. Complete the second-pass solver/algorithm cleanup from Day 8.
2. Re-run the selected family proof owners and any adjacent sanity checks.
3. Update comments or maintainer references only where they now misstate
   ownership or rationale.
4. Check for stale helper names, duplicated comments, and confusing internal
   chronology on touched files.
5. Write the solver/algorithm cleanup artifact.

### Deliverables
- completed solver/algorithm cleanup batch
- family proof-owner validation notes
- solver/algorithm residual queue

### Completion Criteria
- the selected solver/algorithm cleanup is complete for Sprint 96 scope
- touched rationale explains current behavior, not sprint chronology
- validation coverage for the touched family is recorded

---

## Day 10: Giant-Test Architecture Design

**Title:** Test Design
**Theme:** Design one bounded reduction in giant proof-owner concentration
**Time estimate:** 12 hours

### Tasks
1. Review the largest retained proof-owner tests after Days 5-9.
2. Identify one or two giant tests where grouping, helper extraction, or file
   splitting would reduce review cost without weakening proof ownership.
3. Map Makefile, CMake, platform, and suite-label consequences for any test
   movement.
4. Define the validation contract for test architecture changes.
5. Write the giant-test architecture design artifact.

### Deliverables
- giant-test architecture design artifact
- selected proof-owner cleanup batch
- build/CMake/test validation plan

### Completion Criteria
- test cleanup is designed before test files move
- proof ownership remains clear
- platform and build registration impacts are identified

---

## Day 11: Giant-Test Architecture Batch

**Title:** Test Cleanup
**Theme:** Land the selected giant-test architecture cleanup without weakening
coverage
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 10 giant-test cleanup.
2. Update Makefile, CMake, suite labels, helper includes, and comments as
   needed.
3. Preserve test semantics and proof-owner intent.
4. Run targeted tests and build registration checks during development.
5. Record the cleanup batch and any deferred proof-owner splits.

### Deliverables
- giant-test architecture cleanup batch
- updated build/test registrations if needed
- proof-owner cleanup notes

### Completion Criteria
- selected giant-test concentration is reduced or better structured
- Makefile and CMake registration remain coherent
- proof-owner naming and suite labels match the new structure

---

## Day 12: Internal Comment & Rationale Cleanup

**Title:** Rationale Cleanup
**Theme:** Remove stale chronology on touched files while preserving durable
algorithm rationale
**Time estimate:** 12 hours

### Tasks
1. Re-scan files touched in Days 5-11 for stale sprint/day chronology,
   duplicated implementation notes, or obsolete helper explanations.
2. Preserve comments that explain durable algorithm choices, invariants, or
   compatibility constraints.
3. Remove or rewrite comments that only explain development sequence.
4. Update any maintainer-facing notes required by ownership changes.
5. Write the rationale cleanup artifact.

### Deliverables
- internal comment/rationale cleanup batch
- durable-rationale vs historical-residue notes
- touched-file support notes

### Completion Criteria
- touched source/test files explain current ownership and invariants
- historical implementation chronology is removed where it no longer helps
- durable algorithm rationale remains available to reviewers

---

## Day 13: Full Validation & Residual Queue

**Title:** Validation Sweep
**Theme:** Validate the implementation and proof-owner cleanup, then freeze the
residual maintainability queue
**Time estimate:** 12 hours

### Tasks
1. Run the strongest appropriate validation for all changed files.
2. If any `.c` or `.h` files changed, run:
   - `make format`
   - `make lint`
   - `make test`
3. Re-check Makefile/CMake registrations and renamed or moved proof owners.
4. Review the Day 2 hotspot queue and mark completed, deferred, and
   intentionally retained large-owner surfaces.
5. Write the validation and residual queue artifact.

### Deliverables
- validation results
- residual maintainability queue
- closeout preparation notes

### Completion Criteria
- all required checks pass before closeout
- no hidden stale source/test/build reference remains
- residual debt is separated from intentional non-goals

---

## Day 14: Sprint 96 Closeout

**Title:** Closeout
**Theme:** Close Sprint 96 with evidence, artifacts, and a bounded Sprint 97
handoff
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 96 project-plan section against completed artifacts.
2. Confirm each project-plan item is done or explicitly deferred:
   - Hotspot Rerank
   - Source Extraction Design
   - Direct-Family Source Cleanup Batch
   - Solver/Algorithm Source Cleanup Batch
   - Giant-Test Architecture Batch
   - Internal Comment/Rationale Cleanup
   - Validation and Closeout
3. Write the Sprint 96 retrospective and handoff notes.
4. Confirm touched implementation and proof hotspots are smaller or clearer
   than the starting state.
5. Record the Sprint 97 handoff queue and close working notes.

### Deliverables
- Sprint 96 retrospective
- Sprint 96 handoff queue
- final validation and artifact index

### Completion Criteria
- Sprint 96 closes from validated evidence
- all project-plan items have a clear done/deferred status
- Sprint 97 receives a bounded maintainability queue instead of a broad
  refactor backlog
