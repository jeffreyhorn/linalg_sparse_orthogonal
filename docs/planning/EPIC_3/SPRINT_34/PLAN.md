# Sprint 34 Plan: Build-Quality Enforcement Phase 1

**Sprint Duration:** 14 days  
**Goal:** Turn the cleanup work from Sprints 30-33 into enforceable build-quality rules, starting with warning cleanliness and dead-code checks on the most important targets and toolchains. This sprint implements the Sprint 34 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`.

**Starting Point:** Sprint 33 closed with green `make format`, `make lint`, `make test`, `ctest -N`, full `ctest`, and a working dead-code flow (`make deadcode`, `make deadcode-report`, `make deadcode-check`). Sprint 34 starts from that validated baseline and shifts focus from cleanup into repeatable enforcement, compiler-path parity, and first-phase CI hardening.

**End State:** Sprint 34 leaves behind enforceable compile-quality targets in the Makefile, validated CMake parity for the reviewed target set, a first CI integration pass for warning/dead-code drift, clearer failure messages, and explicit handling of the remaining Sprint 33 dead-code coverage and execution-model limitations.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 136 hours, matching the Sprint 34 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 34 Scope Audit & Baseline

**Title:** Enforcement Baseline  
**Theme:** Convert the Sprint 33 handoff into a precise Sprint 34 enforcement inventory  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 34 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`, the Sprint 33 handoff, and the Sprint 33 retrospective so the sprint stays attached to the explicit deferred scope.
2. Confirm the inherited invariants that must remain true during Sprint 34: warning-clean reviewed flows, `53` registered CTest tests, live opt-in truthfulness, and a zero definitely-unused internal queue.
3. Inventory the current quality targets in the Makefile, CMake build tree, and CI workflows.
4. Record the current Sprint 33 dead-code limitations that Sprint 34 must either preserve or harden: compile-db coverage gaps and serial-only dead-code execution.
5. Open Sprint 34 working notes and capture the baseline commands, target files, and first likely implementation surfaces.

### Deliverables
- Sprint 34 enforcement baseline
- Current quality-target inventory
- Named first-pass implementation surfaces for Makefile, CMake, and CI work

### Completion Criteria
- Sprint 34 starts from a documented Sprint 33 validated baseline
- The build-quality enforcement scope is separated from earlier cleanup work
- The known dead-code limitations are captured before implementation begins

---

## Day 2: Warning-Gate Audit By Toolchain

**Title:** Toolchain Audit  
**Theme:** Identify which reviewed targets and compilers should be enforced first  
**Time estimate:** 8 hours

### Tasks
1. Audit the current strict-warning behavior on the Apple Clang CMake path, the normal Makefile path, and the benchmark/example compile-only path.
2. Distinguish the target groups Sprint 34 should gate first: `src/`, key tests, compile-only benchmarks/examples, and dead-code support targets.
3. Note where compiler differences are real portability issues versus acceptable phase-1 exclusions.
4. Review the existing `make lint`, `make tooling-build`, and dead-code commands to understand where enforcement can piggyback and where dedicated targets are needed.
5. Write the audit note that turns “build quality” into an explicit first-phase enforcement target matrix.

### Deliverables
- Toolchain/target audit note
- First-phase enforcement matrix
- Initial include/exclude guidance for Sprint 34 warning gates

### Completion Criteria
- The first enforced target set is defined before target wiring begins
- Compiler differences are documented rather than hand-waved
- Sprint 34 has a defensible basis for phase-1 enforcement scope

---

## Day 3: Warning-Gate Design

**Title:** Gate Design  
**Theme:** Define the warning-clean enforcement contract for reviewed targets  
**Time estimate:** 10 hours

### Tasks
1. Design the warning-clean contract for the first reviewed target set, including which commands are authoritative and which are supporting cross-checks.
2. Decide how compile-only benchmark/example enforcement should relate to existing runtime tests.
3. Specify how Sprint 34 will preserve the Sprint 32 expectation that `make lint` and `make test` remain part of the normal local path.
4. Decide how dead-code checks will remain separate from warning enforcement early in the sprint while still moving toward later integration.
5. Write the design note that Days 4-6 will implement.

### Deliverables
- Warning-gate design note
- Command-level enforcement contract
- Phase-1 target inclusion/exclusion list

### Completion Criteria
- The enforcement contract is concrete enough to implement without scope drift
- Warning enforcement and runtime execution responsibilities are separated clearly
- The design preserves the Sprint 32 and Sprint 33 baseline commands explicitly

---

## Day 4: Makefile Enforcement Target Design

**Title:** Makefile Design  
**Theme:** Turn the warning-gate contract into concrete Makefile target behavior  
**Time estimate:** 10 hours

### Tasks
1. Audit the current Makefile target graph around `lint`, `test`, `tooling-build`, and `deadcode*`.
2. Design the new or refined compile-quality targets for Sprint 34, including names, dependencies, and operator-facing usage.
3. Decide where reviewed-target compile checks should be layered so failure output remains attributable and easy to read.
4. Define how the dead-code flow should be integrated into the quality path without reintroducing the Sprint 33 shared-build-tree race.
5. Write the Makefile implementation plan and output contract.

### Deliverables
- Makefile target design for phase-1 compile-quality enforcement
- Dead-code integration plan for the quality flow
- Failure-output design note for later implementation

### Completion Criteria
- The Makefile wiring plan is explicit before edits begin
- The target graph is understandable and extension-friendly
- Dead-code integration is designed with Sprint 33’s serial constraint in mind

---

## Day 5: Makefile Enforcement Implementation — Batch I

**Title:** Makefile Batch I  
**Theme:** Implement the first reviewed-target compile-quality gates  
**Time estimate:** 10 hours

### Tasks
1. Add or refine the first phase of Makefile compile-quality targets based on the Day 4 design.
2. Wire the chosen reviewed target set into the Makefile without conflating compile-only checks with runtime tests.
3. Keep the target behavior readable and auditable rather than relying on opaque shell chains.
4. Run focused validation on the new targets and capture operator-facing output.
5. Record implementation notes and any friction that affects Day 6 or Day 12.

### Deliverables
- First implemented Makefile compile-quality targets
- Focused validation evidence for Batch I
- Notes on target behavior and output quality

### Completion Criteria
- The first Makefile enforcement targets run end to end
- Failure/success behavior is attributable to named reviewed target groups
- The new targets do not regress the normal local quality path

---

## Day 6: Makefile Enforcement Implementation — Batch II

**Title:** Makefile Batch II  
**Theme:** Complete the phase-1 Makefile enforcement path and dead-code layering  
**Time estimate:** 10 hours

### Tasks
1. Finish the remaining Makefile target wiring from the Day 4 design.
2. Integrate dead-code checks into the reviewed quality flow at the agreed phase-1 level.
3. Tighten target dependencies so redundant reruns and ambiguous behavior are minimized.
4. Validate the full reviewed Makefile quality path and record any remaining limitations.
5. Update Sprint 34 notes with the completed local enforcement workflow.

### Deliverables
- Completed phase-1 Makefile enforcement path
- Dead-code quality-flow integration at Sprint 34 scope
- Validation notes for the full local reviewed-target path

### Completion Criteria
- The reviewed Makefile quality path is complete and usable
- Dead-code checks are integrated intentionally, not bolted on ad hoc
- Remaining limitations are documented rather than hidden

---

## Day 7: CMake Parity Audit & Design

**Title:** CMake Parity Design  
**Theme:** Define how reviewed compile-quality checks stay visible from the CMake path  
**Time estimate:** 8 hours

### Tasks
1. Audit what the current `build/sprint33-day1-cmake` path already proves and what it does not prove for compile-quality enforcement.
2. Identify which reviewed Sprint 34 targets need direct CMake parity versus documentation-backed cross-check coverage.
3. Design the CMake-path commands or target groupings that will keep `ctest -N` and full `ctest` auditable while adding compile-quality confidence.
4. Include the Sprint 33 dead-code compile-db coverage-gap issue in the parity design.
5. Write the CMake parity design note for Day 8 implementation.

### Deliverables
- CMake parity audit note
- CMake-path compile-quality design
- Coverage-gap handling plan for dead-code parity

### Completion Criteria
- Sprint 34 has an explicit CMake parity plan before edits begin
- `ctest -N` and full `ctest` remain part of the parity story
- Dead-code compile-db coverage is treated as a first-class parity issue

---

## Day 8: CMake Parity Implementation

**Title:** CMake Parity  
**Theme:** Implement and validate the first reviewed compile-quality parity checks from the CMake side  
**Time estimate:** 10 hours

### Tasks
1. Implement the chosen CMake-path compile-quality or validation support from Day 7.
2. Ensure the reviewed target set can be validated from the generated build tree without relying on undocumented Make-only behavior.
3. Re-test `ctest -N` and full `ctest` after the parity changes.
4. Record which quality guarantees now have true Make/CMake parity and which still remain phase-1 documented limitations.
5. Update Sprint 34 notes with the implemented parity workflow.

### Deliverables
- Implemented CMake parity support for Sprint 34 reviewed targets
- Validation record for CMake parity
- Documented list of parity guarantees versus preserved limitations

### Completion Criteria
- The CMake path contributes concrete Sprint 34 quality guarantees
- The active suite remains auditable via `ctest -N` and full `ctest`
- Any remaining Make-only behavior is explicit and justified

---

## Day 9: CI Enforcement Design

**Title:** CI Design  
**Theme:** Choose a safe first-phase CI integration strategy for compile-quality and dead-code checks  
**Time estimate:** 12 hours

### Tasks
1. Audit the current GitHub Actions workflows and identify the primary jobs best suited for Sprint 34 phase-1 enforcement.
2. Decide which warning/dead-code checks should run in CI first and which should remain local-only for now.
3. Design how Sprint 33’s dead-code serial-execution constraint will be preserved in CI.
4. Define what the CI jobs should emit on failure so future contributors can act without reverse-engineering the workflow.
5. Write the CI integration design note for Days 10 and 12.

### Deliverables
- Phase-1 CI enforcement design
- Job-selection matrix for reviewed targets/toolchains
- Failure-output expectations for CI

### Completion Criteria
- Sprint 34 has a non-flaky CI integration plan before workflow edits begin
- The dead-code execution model is accounted for explicitly
- The selected jobs are narrow enough for phase 1 and useful enough to matter

---

## Day 10: CI Enforcement Implementation

**Title:** CI Wiring  
**Theme:** Implement the first non-flaky CI enforcement pass for Sprint 34  
**Time estimate:** 10 hours

### Tasks
1. Update the CI workflows to add the reviewed phase-1 warning/dead-code checks.
2. Preserve dead-code serialization or isolate its artifacts/build tree so CI does not recreate Sprint 33’s race condition.
3. Make the CI steps readable and maintainable rather than embedding fragile shell logic.
4. Capture the resulting workflow behavior and any remaining intentionally deferred enforcement.
5. Update Sprint 34 notes with the implemented CI contract.

### Deliverables
- First CI enforcement pass implemented
- Documented CI behavior and constraints
- Notes on deferred CI scope for later sprints

### Completion Criteria
- The chosen CI jobs enforce the intended Sprint 34 checks
- The implementation preserves non-flaky execution assumptions
- Deferred enforcement scope is explicit rather than accidental

---

## Day 11: Initializer Regression Audit & Cleanup

**Title:** Regression Audit  
**Theme:** Close any new positional-initializer drift exposed by the enforcement work  
**Time estimate:** 10 hours

### Tasks
1. Audit the reviewed target set for any positional options-struct initializers that surface as Sprint 34 compile-quality regressions.
2. Distinguish real new regression cleanup from already-closed inherited Sprint 32 backlog.
3. Convert any newly surfaced high-noise reviewed sites to designated initializers.
4. Validate the touched reviewed targets after each small cleanup batch.
5. Record the regression-prevention outcome in Sprint 34 notes.

### Deliverables
- Initializer-regression audit note
- Any required designated-initializer cleanup for reviewed Sprint 34 targets
- Validation evidence for touched reviewed files

### Completion Criteria
- Sprint 34 handles newly surfaced initializer regressions without reopening old backlog
- Cleanups remain mechanical and low-risk
- The reviewed-target warning gates stay green after the fixes

---

## Day 12: Failure-Message Quality & Operator Docs

**Title:** Failure UX  
**Theme:** Improve the clarity of local and CI quality failures  
**Time estimate:** 12 hours

### Tasks
1. Review the operator-facing output from the new Makefile, CMake, and CI enforcement paths.
2. Improve target or workflow messaging where failures are correct but not actionable enough.
3. Document the intended local operator workflow for the new Sprint 34 quality gates.
4. Ensure the docs preserve the distinction between compile-quality enforcement, runtime testing, and dead-code reporting.
5. Update Sprint 34 notes with the final operator-facing command map.

### Deliverables
- Improved failure-message quality for Sprint 34 enforcement paths
- Maintainer/operator doc refresh for the new quality flow
- Final Sprint 34 command map

### Completion Criteria
- Failures explain what broke and where to look next
- Local operators can run the reviewed quality flow without guesswork
- The docs reflect the actual enforcement behavior rather than the design intent alone

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Prove the Sprint 34 enforcement work stays green across local and CI-equivalent paths  
**Time estimate:** 10 hours

### Tasks
1. Run the full reviewed local quality path, including `make format`, `make lint`, `make test`, `ctest -N`, full `ctest`, and the Sprint 34 dead-code/compile-quality targets.
2. Re-run the CMake parity path and the CI-equivalent local commands for the newly enforced jobs.
3. Confirm the Sprint 32 truthfulness baseline still holds and the Sprint 33 dead-code flow still produces the documented artifacts.
4. Record timings, test counts, and any enforcement-path observations needed for closeout.
5. Update Sprint 34 notes with the final validated end state before handoff docs begin.

### Deliverables
- Full Sprint 34 validation record
- CI-equivalent local validation record
- Confirmed post-enforcement active/opt-in/dead-code state

### Completion Criteria
- All reviewed local and CI-equivalent validation flows pass
- Sprint 34’s new quality gates are proven on the intended target set
- The end state is fully measured before closeout

---

## Day 14: Closeout, Handoff & Forward Queue

**Title:** Sprint Closeout  
**Theme:** Package Sprint 34’s enforcement work for Sprint 35 and later Epic 3 phases  
**Time estimate:** 8 hours

### Tasks
1. Write the Sprint 34 handoff summarizing delivered quality targets, CMake parity, CI integration, regression-prevention fixes, and validated commands.
2. Write the Sprint 34 retrospective covering what worked, what remained phase-1 conservative, and where later quality enforcement still needs to expand.
3. Route any concrete deferred work into Sprint 35 or later sections of `docs/planning/EPIC_3/PROJECT_PLAN.md`.
4. Preserve the Sprint 32 and Sprint 33 invariants explicitly in the closeout so later sprints do not regress them while expanding enforcement.
5. Ensure the closeout documents any remaining reviewed-target exclusions or dead-code limitations that still matter after Sprint 34.

### Deliverables
- `HANDOFF.md`
- `RETROSPECTIVE.md`
- Forward-plan updates for deferred Sprint 35+ work if needed

### Completion Criteria
- Sprint 34 artifacts explain both the shipped enforcement work and the remaining phased limitations
- Later sprints can recover the reviewed-target contract without rereading the full sprint history
- Sprint 34 closes with a clear validated baseline for the next enforcement phase
