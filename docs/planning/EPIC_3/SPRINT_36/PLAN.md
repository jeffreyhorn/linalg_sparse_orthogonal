# Sprint 36 Plan: Cross-Platform Quality Parity

**Sprint Duration:** 14 days  
**Goal:** Make the hardening work portable across the supported platforms by closing quality gaps between Apple Clang, Linux/GCC-or-Clang paths, and Windows/MSVC where practical. This sprint implements the Sprint 36 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`.

**Starting Point:** Sprint 35 closed with public docs and examples reconciled to the current API, while Sprint 34 left behind the reviewed Makefile/CMake wrapper contract and Linux-first CI enforcement. Sprint 36 starts from that validated baseline and shifts focus to parity: warning behavior across compilers, reviewed-path expectations across platforms, portability limits in scripts and targets, and explicit cross-platform reporting.

**End State:** Sprint 36 leaves behind a clearer and more portable quality contract across Linux, macOS, and Windows, including named parity findings, concrete targeted fixes, better CI expectation alignment, and a validated cross-platform quality report grounded in the maintained reviewed wrappers.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 144 hours, matching the Sprint 36 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 36 Scope Audit & Baseline

**Title:** Parity Baseline  
**Theme:** Convert the Sprint 36 project-plan items into a concrete cross-platform audit scope  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 36 section of `docs/planning/EPIC_3/PROJECT_PLAN.md` plus the Sprint 34 and Sprint 35 handoff/retrospective docs so the sprint stays anchored to the documented parity scope.
2. Confirm the current validated baseline that must remain true through Sprint 36: reviewed Makefile wrappers pass, reviewed CMake parity paths pass, the active CTest count remains auditable, and the public-doc contract from Sprint 35 stays intact.
3. Inventory the current platform-facing quality surfaces: Linux CI, macOS CI, Windows CI, local Makefile reviewed wrappers, local CMake parity wrappers, and dead-code tooling expectations.
4. Record the most likely parity hotspots before edits begin: warning-flag differences, shell assumptions, tool discovery assumptions, MSVC/Apple Clang behavioral differences, and stale CI expectation gaps.
5. Open Sprint 36 working notes and record the initial file map, reproduction commands, and likely implementation batches.

### Deliverables
- Sprint 36 cross-platform baseline
- Initial parity-surface inventory
- Named first-pass audit targets for macOS, Windows, CI, and portability tooling

### Completion Criteria
- Sprint 36 starts from a documented Sprint 34/Sprint 35 validated baseline
- The parity scope is separated from public-doc cleanup and warning-debt cleanup
- Likely cross-platform drift surfaces are identified before edits begin

---

## Day 2: macOS Warning-Parity Audit

**Title:** macOS Audit  
**Theme:** Identify the reviewed-path quality differences still present on Apple Clang paths  
**Time estimate:** 8 hours

### Tasks
1. Audit the maintained Apple Clang build and validation paths used locally and in CI, including the reviewed wrappers and representative clean rebuilds.
2. Compare Apple Clang warning behavior against the Linux reviewed baseline, focusing on categories the repo has already treated as important in prior sprints.
3. Separate real avoidable parity debt from acceptable platform/compiler behavior differences that should be documented rather than “fixed.”
4. Record the files, targets, or scripts most likely to require Day 5 targeted fixes on macOS.
5. Write the audit note that defines the concrete macOS parity queue.

### Deliverables
- macOS warning-parity audit note
- Named Apple Clang parity queue
- Keep/fix/document classification for macOS-specific issues

### Completion Criteria
- The macOS parity queue is explicit before edits begin
- Real parity debt is separated from expected platform variance
- The likely macOS fix batch is bounded clearly enough for implementation

---

## Day 3: Windows/MSVC Quality Audit

**Title:** MSVC Audit  
**Theme:** Map the Windows/MSVC reviewed-quality gap against the existing Linux-first contract  
**Time estimate:** 10 hours

### Tasks
1. Audit the existing Windows workflow, build flags, and observed output surfaces to identify where MSVC quality expectations differ from the Unix compilers.
2. Focus on warning classes and language/ABI issues that commonly diverge in this codebase: enum exhaustiveness, initializer behavior, missing includes, implicit assumptions around standard headers, and warning-level mismatches.
3. Separate true repo-side parity debt from CI/workflow configuration gaps and from expected compiler-policy differences.
4. Record which findings belong to code changes, which belong to CI alignment, and which belong to docs/reporting.
5. Write the audit note that defines the Windows/MSVC cleanup and alignment queue.

### Deliverables
- Windows/MSVC quality audit note
- Named MSVC parity queue
- Initial split across code, workflow, and documentation/reporting fixes

### Completion Criteria
- The Windows parity surface is mapped concretely before edits begin
- MSVC-specific issues are classified by the right fix surface
- The next implementation batches are bounded enough to execute deliberately

---

## Day 4: CI & Tooling Parity Design

**Title:** Parity Design  
**Theme:** Define the reviewed cross-platform contract before changing CI or helper scripts  
**Time estimate:** 10 hours

### Tasks
1. Decide how Linux, macOS, and Windows should each express the reviewed-quality contract without pretending that all platforms share identical tool availability.
2. Define what “parity” means for Sprint 36 phase 1: reviewed Makefile path, reviewed CMake path, dead-code portability expectation, and CI naming/reporting expectations per platform.
3. Decide which differences should be enforced now versus documented for later phases.
4. Map the implementation order for Days 5 through 10 so code fixes, script fixes, and CI alignment stay synchronized.
5. Write the design note for cross-platform quality parity and reviewed-path interpretation.

### Deliverables
- Cross-platform parity design note
- Defined reviewed-path contract by platform
- File-by-file implementation order for code, scripts, and CI

### Completion Criteria
- The platform parity contract is chosen before implementation begins
- Enforcement-vs-documentation boundaries are explicit
- Later implementation days have a concrete sequence instead of ad hoc edits

---

## Day 5: macOS & Compiler-Behavior Fixes

**Title:** macOS Fixes  
**Theme:** Close the first bounded Apple Clang parity issues without broadening scope  
**Time estimate:** 10 hours

### Tasks
1. Implement the concrete repo-side fixes surfaced by the macOS audit where the Linux reviewed baseline and Apple Clang behavior should converge.
2. Keep the changes narrow, reviewed-path-focused, and semantically neutral wherever possible.
3. Recheck touched warnings, scripts, or helper flows against the actual Apple Clang behavior rather than assuming Linux behavior carries over.
4. Record any residual Apple Clang differences that remain intentional or deferred.
5. Update working notes with the before/after parity state.

### Deliverables
- First macOS/compiler parity fix batch
- Residual Apple Clang keep/defer list
- Updated parity notes with measured deltas

### Completion Criteria
- The first bounded macOS parity issues are closed
- Touched changes stay aligned with the reviewed wrapper contract
- Remaining Apple Clang differences are explicit rather than implicit

---

## Day 6: Windows/MSVC Fix Design & Narrow Batch

**Title:** MSVC Batch I  
**Theme:** Convert the Windows audit into the first practical fix/reporting batch  
**Time estimate:** 10 hours

### Tasks
1. Implement the safest code, script, or workflow fixes from the MSVC audit that do not depend on larger CI restructuring.
2. Tighten platform-conditional behavior only where the reviewed contract genuinely requires it.
3. Keep the batch limited to changes that can be reasoned about from existing Windows workflow evidence.
4. Record any findings that still need the later CI/reporting alignment pass.
5. Update the Sprint 36 notes with the first measured Windows parity improvements.

### Deliverables
- First Windows/MSVC parity fix batch
- Residual Windows queue for CI/reporting follow-on
- Updated parity notes with the narrowed MSVC debt surface

### Completion Criteria
- The first practical Windows/MSVC issues are closed
- Remaining Windows work is reduced to clearly named later batches
- The batch does not overreach beyond the evidence gathered in the audit

---

## Day 7: Script & Target Portability Audit

**Title:** Portability Audit  
**Theme:** Identify Makefile and script assumptions that still bind quality flows to Unix-specific behavior  
**Time estimate:** 10 hours

### Tasks
1. Audit quality and dead-code Makefile/scripts for shell, path, quoting, environment, and tool-discovery assumptions that are Linux/macOS-specific.
2. Focus especially on Sprint 34/Sprint 33 carry-forward constraints: `xunused` setup, dead-code job assumptions, shared-path serialized execution, and compiler/tool lookup logic.
3. Separate unavoidable platform limitations from avoidable portability debt.
4. Map each finding to the right fix surface: shell logic, Makefile target shape, CI setup, or documentation/reporting.
5. Write the portability audit note that defines Days 8 and 9.

### Deliverables
- Script/target portability audit note
- Named portability-fix queue
- Surface-by-surface mapping for Makefile, shell, CI, and docs changes

### Completion Criteria
- Portability debt is identified concretely before edits begin
- Real portability issues are separated from acceptable platform exclusions
- The next implementation batches have a bounded fix surface

---

## Day 8: Script & Target Portability Fixes — Batch I

**Title:** Portability Batch I  
**Theme:** Fix the highest-value Makefile and script portability issues in the reviewed paths  
**Time estimate:** 10 hours

### Tasks
1. Implement the first portability batch in reviewed-quality and dead-code targets/scripts based on the Day 7 audit.
2. Preserve current Linux behavior while making the logic more explicit and portable for macOS/Windows-adjacent paths.
3. Re-run the touched reviewed paths locally to confirm that portability cleanup did not regress the validated baseline.
4. Record residual portability items that still depend on CI alignment or later gating work.
5. Update the working notes with the measured before/after state.

### Deliverables
- First portability fix batch
- Residual portability queue
- Updated parity notes grounded in rerun reviewed paths

### Completion Criteria
- The highest-value reviewed-path portability issues are closed
- Linux baseline behavior remains intact
- Remaining portability work is narrower and explicitly documented

---

## Day 9: CI Job Alignment Implementation

**Title:** CI Alignment  
**Theme:** Make the platform expectations in CI explicit instead of Linux-implied  
**Time estimate:** 10 hours

### Tasks
1. Update CI workflow names, steps, or structure so the reviewed-quality expectations are clear on Linux, macOS, and Windows.
2. Keep Linux’s existing enforced contract intact while making the macOS/Windows intent explicit, whether as parity targets, reporting, or staged reviewed-path checks.
3. Ensure dead-code portability limits remain truthful in CI rather than implied closed.
4. Keep artifact/reporting surfaces understandable so later parity work can compare platforms directly.
5. Record the resulting CI expectation model in the Sprint 36 notes.

### Deliverables
- CI alignment batch
- Clearer platform-specific reviewed-quality expectations
- Updated notes on enforcement vs documentation per workflow

### Completion Criteria
- CI no longer leaves macOS/Windows expectations implicit
- Linux reviewed enforcement remains intact
- Dead-code/tooling limits are represented truthfully in CI

---

## Day 10: CMake/Makefile/CI Parity Report

**Title:** Parity Report  
**Theme:** Turn the audit and implementation work into a compact cross-platform quality map  
**Time estimate:** 10 hours

### Tasks
1. Produce a small parity report that shows which reviewed-quality checks are available from Make, from CMake, and in CI on each platform.
2. Capture which paths are enforced, which are advisory, and which still depend on staged portability work.
3. Reconcile the report wording against the actual Sprint 36 implementation state so it is operationally truthful.
4. Identify any last small mismatches between the report and the actual workflows/targets.
5. Record the parity report as the Day 10 output and feed any final fixes into Day 11.

### Deliverables
- Cross-platform CMake/Makefile/CI parity report
- Enforced/advisory/staged classification by platform
- Final small-fix queue for Day 11

### Completion Criteria
- The repo now has a compact parity map instead of scattered assumptions
- The report matches real targets and workflows
- Remaining fixes are narrowed to a bounded final implementation batch

---

## Day 11: Targeted Fixes — Final Batch

**Title:** Final Batch  
**Theme:** Close the remaining bounded parity issues surfaced by the report and prior audits  
**Time estimate:** 12 hours

### Tasks
1. Implement the final small code, script, CI, or documentation fixes surfaced by the parity report and prior audit batches.
2. Reconcile naming, workflow references, or target comments so the reviewed cross-platform contract is described consistently.
3. Avoid reopening broad public-doc or dead-code maturity work outside Sprint 36’s parity scope.
4. Re-run the directly touched validation surfaces immediately after changes.
5. Update the notes with the final implementation-state delta before the validation sweep begins.

### Deliverables
- Final cross-platform parity fix batch
- Consistent naming/reporting across touched targets and workflows
- Final pre-validation implementation-state summary

### Completion Criteria
- The bounded Sprint 36 parity queue is materially closed
- Remaining differences are deliberate and documented
- Day 12 can focus on validation instead of unresolved implementation drift

---

## Day 12: Platform-Focused Validation

**Title:** Parity Validation  
**Theme:** Re-run the practical platform-facing quality flows touched by Sprint 36 before the full sweep  
**Time estimate:** 12 hours

### Tasks
1. Run the practical local validation paths touched by Sprint 36, including reviewed Makefile paths, reviewed CMake paths, and any portability-sensitive script/target commands.
2. Re-check the parity-report claims against the actual command behavior after the fix batches.
3. Validate representative CI-facing commands or helper sequences locally where feasible.
4. Capture any last mismatch while the implementation context is still fresh and resolve it if the fix is small and clearly in scope.
5. Record the platform-focused validation results and the exact Day 13 full-sweep command set.

### Deliverables
- Platform-focused validation record
- Parity-report truthfulness check
- Final cleanup of any validation-surfaced small drift

### Completion Criteria
- The practical parity flows touched by Sprint 36 pass
- The parity report remains truthful to the real command behavior
- Day 13 can be a pure full validation sweep

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Re-run the maintained quality and parity flows against the Sprint 36 cross-platform changes  
**Time estimate:** 12 hours

### Tasks
1. Run the Sprint 36 full validation set, including the maintained reviewed-quality commands and the reviewed CMake parity paths.
2. Reconfirm that Sprint 34/Sprint 35 baseline invariants still hold after the parity and portability work.
3. Reconfirm that the parity-report surfaces and CI-alignment changes did not regress the local reviewed commands.
4. Capture timings, success state, and any final observations needed for closeout.
5. Update the Sprint 36 notes with the final validated end state before handoff docs begin.

### Deliverables
- Full Sprint 36 validation record
- Reconfirmed reviewed-quality and parity state
- Final validated baseline for closeout

### Completion Criteria
- All intended Sprint 36 validation flows pass
- Cross-platform parity work does not regress the reviewed local/CMake baseline
- The end state is fully measured before closeout

---

## Day 14: Closeout, Handoff & Forward Queue

**Title:** Sprint Closeout  
**Theme:** Package Sprint 36’s parity work for Sprint 37, Sprint 38, and later Epic 3 hardening work  
**Time estimate:** 12 hours

### Tasks
1. Write the Sprint 36 handoff summarizing the shipped cross-platform parity findings, fixes, report outputs, and validation outcomes.
2. Write the Sprint 36 retrospective covering what worked, what remained intentionally deferred, and which later sprints inherit the remaining parity and portability work.
3. Route any concrete deferred items into Sprint 37, Sprint 38, or later sections of `docs/planning/EPIC_3/PROJECT_PLAN.md`.
4. Preserve the Sprint 34/Sprint 35 reviewed-quality and public-doc baselines explicitly in the closeout so later work does not regress them casually.
5. Ensure the closeout documents any remaining platform exclusions or staged enforcement limits that still matter after Sprint 36.

### Deliverables
- `HANDOFF.md`
- `RETROSPECTIVE.md`
- Forward-plan updates for deferred Sprint 37+ work if needed

### Completion Criteria
- Sprint 36 artifacts explain both the shipped parity gains and any remaining staged limitations
- Later sprints can recover the cross-platform quality contract without rereading the full sprint history
- Sprint 36 closes with a clear validated baseline for the next Epic 3 phase
