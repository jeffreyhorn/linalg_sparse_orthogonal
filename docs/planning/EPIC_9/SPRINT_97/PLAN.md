# Sprint 97 Plan: Build, Packaging & Cross-Platform Product Convergence Phase 4

**Sprint Duration:** 14 days
**Goal:** Reduce build and workflow duplication, sharpen the long-term package
surface, and improve cross-platform product maturity without claiming fake
parity. This sprint implements the Sprint 97 section of
`docs/planning/EPIC_9/PROJECT_PLAN.md`.

**Starting Point:** Sprint 97 begins from:
- the Sprint 90 package/platform claim fence
- the Sprint 92 backend/package work
- the Sprint 95 public narrative and workflow coherence cleanup
- the Sprint 96 large-source and giant-test maintainability baseline
- a repo with useful Make, CMake, CI, install/export, and platform proof
  surfaces that still carry avoidable duplication and uneven product-story
  pressure

The strongest Sprint 97 pressure is not to promise broad package parity. It is
to make the build and package story more maintainable and more truthful by:
- auditing duplicated source lists, workflow lanes, CMake surfaces, and
  install/export proof contracts
- defining a bounded convergence architecture before changing build topology
- reducing the highest-value duplicated build or workflow surfaces
- deciding whether the durable product contract remains static-first or adds
  one explicitly bounded shared-library lane
- updating consumer proof and workflow documentation to match the refined
  contract
- calibrating macOS and Windows evidence without overstating support
- closing from full validation and a clear residual queue

**End State:** Sprint 97 leaves behind:
- a build-topology duplication map
- a bounded convergence architecture for Make, CMake, CI, and install/export
  surfaces
- one source-list or workflow reduction batch
- a documented package-surface decision
- updated consumer/workflow proof surfaces aligned with that decision
- calibrated macOS and Windows product claims
- a validated Sprint 97 closeout package and Sprint 98 handoff queue

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 97 project-plan estimate.

---

## Day 1: Sprint 97 Scope & Build Topology Baseline

**Title:** Build Baseline
**Theme:** Turn the Sprint 97 project-plan section and prior package/platform
work into one bounded build-convergence package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 97 section of
   `docs/planning/EPIC_9/PROJECT_PLAN.md`.
2. Re-read the Sprint 90 package/platform claim fence, Sprint 92 package work,
   Sprint 95 narrative cleanup, and Sprint 96 closeout queue.
3. Inventory current build and workflow topology:
   - `Makefile`
   - `CMakeLists.txt`
   - `.github/workflows/`
   - install/export proof scripts
   - reviewed CMake consumer lanes
   - Windows and macOS CI proof surfaces
4. Identify where source lists, test lists, example lists, package claims, and
   validation lanes are duplicated.
5. Open Sprint 97 working notes and record validation expectations for docs,
   build, workflow, CMake, and code-touch days.

### Deliverables
- Sprint 97 scope inventory
- build/workflow topology baseline
- starting duplication candidate list
- Sprint 97 working-notes baseline

### Completion Criteria
- Sprint 97 starts from the merged Sprint 96 end state
- Make, CMake, CI, and install/export proof surfaces are visible before design
- validation expectations are explicit before build topology changes begin

---

## Day 2: Build-Topology Duplication Audit

**Title:** Duplication Audit
**Theme:** Rank the highest-cost duplication between Make, CMake, CI, and
consumer proof surfaces
**Time estimate:** 12 hours

### Tasks
1. Compare source, header, test, benchmark, example, and install/export lists
   across Make and CMake.
2. Compare CI workflow lanes against the local reviewed validation gates.
3. Identify duplicated logic that is:
   - high-cost and frequently touched
   - low-risk to centralize or generate
   - necessary to preserve as independent proof
   - intentionally platform-specific
4. Separate product-story duplication from real build-topology duplication.
5. Write the Day 2 build-topology audit artifact with a ranked fix-now queue.

### Deliverables
- ranked build/workflow duplication map
- fix-now vs preserve-independent-proof split
- platform-specific duplication notes

### Completion Criteria
- Sprint 97 has one authoritative duplication ranking
- candidate reductions are tied to real maintenance cost
- independent proof surfaces are not mistaken for redundant code

---

## Day 3: Convergence Architecture Design

**Title:** Convergence Design
**Theme:** Define the bounded architecture for reducing duplication without
weakening proof strength
**Time estimate:** 12 hours

### Tasks
1. Use the Day 2 audit to select the highest-value convergence targets.
2. Decide which surfaces should remain manually independent as proof owners.
3. Evaluate practical convergence mechanisms:
   - generated source lists
   - shared include fragments
   - CMake target properties
   - Make variables
   - workflow-local assertions
   - documentation-only alignment
4. Define the boundary between build implementation, workflow proof, package
   contract, and public narrative.
5. Write the convergence architecture artifact and validation plan.

### Deliverables
- convergence architecture artifact
- selected source-list/workflow reduction target
- independent-proof preservation list
- validation plan for build/workflow changes

### Completion Criteria
- no build topology change is made before the convergence boundary is written
- selected reductions are bounded to Sprint 97 scope
- proof strength and product claims remain explicit

---

## Day 4: Source-List Reduction Boundary Freeze

**Title:** Source-List Boundary
**Theme:** Freeze the first source-list or workflow reduction batch before
editing build surfaces
**Time estimate:** 12 hours

### Tasks
1. Re-read the selected Day 3 source-list or workflow reduction target.
2. Identify the exact files, generated surfaces, or shared variables to touch.
3. Define what must remain manually checked in Make, CMake, or CI after the
   change.
4. Identify local and CI-equivalent validation commands for the batch.
5. Write the Day 4 boundary artifact with the landing sequence and rollback
   notes.

### Deliverables
- source-list/workflow boundary artifact
- exact touched-surface plan
- validation and rollback checklist

### Completion Criteria
- the first reduction batch is ready to implement without scope drift
- retained proof surfaces are listed before edits begin
- validation is specific enough for a build-topology change

---

## Day 5: Source-List/Workflow Reduction Batch 1

**Title:** Reduction Batch 1
**Theme:** Land the first bounded reduction in duplicated source-list or
workflow topology
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 4 reduction batch.
2. Keep public package behavior unchanged unless explicitly allowed by the Day
   4 boundary.
3. Preserve reviewed proof coverage even if lists or variables move.
4. Run focused build, CMake, install/export, or workflow-equivalent checks
   during development.
5. Record changed topology and any residual duplication discovered while
   landing.

### Deliverables
- first source-list/workflow reduction batch
- updated build or workflow surfaces
- implementation notes and residual duplication queue

### Completion Criteria
- at least one high-cost duplication point is reduced or centralized
- local build behavior remains equivalent
- proof-owner independence is preserved where required

---

## Day 6: Source-List/Workflow Reduction Batch 2

**Title:** Reduction Batch 2
**Theme:** Complete the selected build/workflow reduction and reconcile
validation ownership
**Time estimate:** 12 hours

### Tasks
1. Finish any remaining work from Day 5.
2. Update adjacent docs, comments, or workflow labels only where ownership
   changed.
3. Re-run the targeted proof commands from Day 4.
4. Check that Make, CMake, and CI surfaces no longer contradict each other.
5. Write the Day 6 reduction closeout artifact.

### Deliverables
- completed reduction batch
- targeted validation notes
- residual build-topology queue

### Completion Criteria
- the selected duplicated surface is measurably lower-cost to maintain
- validation ownership is clear after the reduction
- residual duplication is explicit and not hidden

---

## Day 7: Package-Surface Decision Audit

**Title:** Package Decision Audit
**Theme:** Decide what evidence is needed before preserving static-first or
earning one bounded shared-library lane
**Time estimate:** 12 hours

### Tasks
1. Re-read current install, export, package, README, and CMake consumer claims.
2. Identify every place that implies static-only, shared-library, or broader
   packaging support.
3. Audit current Make and CMake capabilities against those claims.
4. Compare the cost and proof burden of:
   - preserving a durable static-first contract
   - adding one bounded shared-library lane
   - documenting shared-library work as deferred
5. Write the Day 7 package decision audit artifact.

### Deliverables
- package-surface claim inventory
- static-first vs bounded-shared decision evidence
- package proof-burden notes

### Completion Criteria
- package claims are grounded in live build evidence
- the shared-library decision is framed as an evidence decision
- unsupported package parity remains outside the sprint unless explicitly
  earned

---

## Day 8: Package-Surface Decision Batch

**Title:** Package Decision
**Theme:** Land the selected package contract and remove contradictory product
signals
**Time estimate:** 12 hours

### Tasks
1. Make the Sprint 97 package decision from the Day 7 audit.
2. Update build, install, CMake, or documentation surfaces required by that
   decision.
3. Keep package claims narrow where proof is incomplete.
4. Record any deliberately deferred shared-library, install, or export work.
5. Run focused package and consumer checks appropriate to the touched surfaces.

### Deliverables
- documented package-surface decision
- updated package/build/docs surfaces
- focused package validation notes
- deferred package queue

### Completion Criteria
- the repo has one coherent package contract after the day
- static-first or bounded-shared claims match actual proof
- future readers can distinguish support from non-claims

---

## Day 9: Consumer Proof Follow-Through

**Title:** Consumer Proof
**Theme:** Align install/export and CMake consumer proof with the refined
package/build contract
**Time estimate:** 12 hours

### Tasks
1. Review install/export proof scripts and CMake consumer tests against the Day
   8 package contract.
2. Update consumer proof only where the package contract changed or became
   clearer.
3. Confirm that examples and public docs point consumers to supported build
   paths.
4. Preserve staged exclusions or limitations where they remain truthful.
5. Write the Day 9 consumer proof artifact.

### Deliverables
- updated consumer proof surfaces, if needed
- install/export proof alignment notes
- consumer-facing build-path clarification

### Completion Criteria
- install/export proof matches the package contract
- unsupported consumer workflows are not implied by docs or CI
- consumer guidance is consistent across front-door and build surfaces

---

## Day 10: Workflow Coherence Follow-Through

**Title:** Workflow Coherence
**Theme:** Align CI and local workflow names, counts, and claims with the
refined build/package story
**Time estimate:** 12 hours

### Tasks
1. Compare local Make targets, CMake lanes, and GitHub workflow names.
2. Identify stale workflow wording, expected test counts, staged exclusions,
   or unsupported parity claims.
3. Update workflow surfaces only where the Day 5-9 work changed the contract.
4. Record any platform-specific differences that remain intentional.
5. Run focused workflow-equivalent checks where possible.

### Deliverables
- workflow coherence update batch
- platform/staged-exclusion notes
- local verification notes

### Completion Criteria
- workflow names and messages match the actual supported surfaces
- expected counts and exclusions are documented where maintained
- local and CI proof language no longer contradicts the package story

---

## Day 11: Cross-Platform Truth Calibration

**Title:** Platform Calibration
**Theme:** Recalibrate macOS and Windows proof claims to match current evidence
without fake parity
**Time estimate:** 12 hours

### Tasks
1. Review macOS, Windows, and generic POSIX build/proof surfaces.
2. Identify which claims are backed by:
   - local validation
   - GitHub Actions evidence
   - CMake consumer proof
   - install/export proof
   - documentation-only limits
3. Update claim language or workflow assertions where evidence has changed.
4. Preserve staged exclusions for unsupported or intentionally deferred lanes.
5. Write the Day 11 cross-platform calibration artifact.

### Deliverables
- macOS/Windows proof-evidence map
- calibrated platform claim updates, if needed
- staged-exclusion and non-parity notes

### Completion Criteria
- platform claims are evidence-backed and current
- Windows and macOS proof surfaces are not overstated
- staged exclusions remain visible where they matter

---

## Day 12: Cross-Platform Product Follow-Through

**Title:** Platform Follow-Through
**Theme:** Apply the calibrated platform story to build, package, workflow, and
front-door surfaces
**Time estimate:** 11 hours

### Tasks
1. Apply any remaining platform-claim updates from Day 11.
2. Check README, install docs, package docs, workflow messages, and planning
   notes for stale platform wording.
3. Update proof expectations where supported platform evidence changed.
4. Record platform residuals that should move to Sprint 98 or later.
5. Run focused docs/build checks appropriate to touched surfaces.

### Deliverables
- platform product-story follow-through batch
- updated platform residual queue
- focused verification notes

### Completion Criteria
- public and internal platform language tells the same story
- platform proof gaps are explicitly queued instead of implied away
- no unsupported parity claim remains in touched surfaces

---

## Day 13: Full Validation & Residual Queue

**Title:** Validation Sweep
**Theme:** Validate the build/package convergence work and freeze the residual
queue
**Time estimate:** 12 hours

### Tasks
1. Run the strongest required validation for all touched surfaces.
2. Include code, header, build, CMake, install/export, workflow, or docs
   validation according to the Sprint 97 working notes.
3. Re-check the Day 2 duplication map against the final tree.
4. Separate residuals into:
   - Sprint 98 candidates
   - package/product non-claims
   - platform proof gaps
   - intentionally preserved independent proof
5. Write the Day 13 validation and residual queue artifact.

### Deliverables
- full validation notes
- final duplication delta
- Sprint 98 residual queue
- preserved-proof and non-claim list

### Completion Criteria
- all required validation for touched surfaces passes or is explicitly blocked
- residual work is ranked and evidence-backed
- Sprint 97 has no hidden build/package/platform contradictions

---

## Day 14: Sprint 97 Closeout

**Title:** Sprint Closeout
**Theme:** Close Sprint 97 with a validated build/package/product convergence
record and handoff queue
**Time estimate:** 11 hours

### Tasks
1. Review all Sprint 97 artifacts and working notes.
2. Write the Sprint 97 closeout artifact summarizing:
   - duplication reduced
   - package-surface decision
   - consumer/workflow follow-through
   - cross-platform calibration
   - validation evidence
3. Update the Sprint 98 handoff queue with unresolved build, package, workflow,
   and platform items.
4. Check the plan, artifacts, and working notes for stale claims or missing
   links.
5. Prepare the branch for retrospective and PR creation.

### Deliverables
- Sprint 97 closeout artifact
- Sprint 98 handoff queue
- final Sprint 97 artifact index
- PR-ready planning package

### Completion Criteria
- Sprint 97 closes from evidence rather than aspiration
- build/package/platform residuals are explicit
- the branch is ready for retrospective and PR review
