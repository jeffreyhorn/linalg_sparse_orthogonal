# Sprint 30 Plan: Warning Baseline & Core Compile-Hygiene Triage

**Sprint Duration:** 14 days  
**Goal:** Establish a measured warning baseline for the full tree, fix the highest-signal core-library warnings first, and create a repeatable compile-quality workflow that later sprints can enforce. This sprint implements the Sprint 30 section of `docs/planning/EPIC_3/PROJECT_PLAN.md` (lines 9-39).

**Starting Point:** Epic 3 opens with a functionally strong but not warning-clean repository. The current codebase builds and tests successfully, but the local CMake build emits warning classes across `src/`, `tests/`, `benchmarks/`, and `examples`, including `-Wdouble-promotion`, `-Wmissing-field-initializers`, implicit-declaration warnings in benchmark code, unused-function warnings in dormant test scaffolding, and enum-drift warnings in auxiliary tools. The project already has a broad validation stack: Makefile and CMake build paths, `ctest`, sanitizers, coverage tooling, and Linux/macOS/Windows CI.

**End State:** Sprint 30 leaves behind a measured warning baseline, a documented compile-hygiene playbook, a repeatable rebuild/validation workflow, a first round of core-library warning fixes in the highest-signal files, an audited first-fix queue for non-library warnings, and Sprint 30 notes capturing the before/after warning count and validation results. The repository remains functionally unchanged apart from compile-hygiene improvements.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 120 hours, matching the Sprint 30 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Baseline Build Capture

**Title:** Clean Build Baseline  
**Theme:** Capture the first reproducible warning baseline from a clean tree  
**Time estimate:** 8 hours

### Tasks
1. Run a clean CMake configure/build from scratch and capture the full warning output.
2. Run a clean Makefile build path and capture its warning output separately.
3. Normalize the collected warnings into a single baseline artifact grouped by compiler, target, and source file.
4. Distinguish real compile warnings from informational output so the baseline is machine-comparable later in the sprint.
5. Open the Sprint 30 working notes file and record the raw baseline counts.

### Deliverables
- Clean CMake warning log
- Clean Makefile warning log
- Initial baseline artifact with raw warning counts
- Sprint 30 notes initialized

### Completion Criteria
- Both primary local build paths have reproducible warning captures
- Warning counts are recorded in a form that can be compared later
- No source changes made yet

---

## Day 2: Warning Taxonomy & Ownership Map

**Title:** Warning Classification  
**Theme:** Turn the raw warning stream into a usable work inventory  
**Time estimate:** 8 hours

### Tasks
1. Classify every baseline warning by area: `src/`, `tests/`, `benchmarks/`, `examples`, build scripts, or generated code.
2. Classify every baseline warning by type: `double-promotion`, missing-field-initializers, implicit declaration, enum drift, unused-function, unused-variable, or other.
3. Identify which warnings come from the library proper versus auxiliary code.
4. Rank warning classes by severity and remediation urgency for Sprint 30 versus later sprints.
5. Update Sprint 30 notes with the ownership map and first-pass severity ordering.

### Deliverables
- Classified warning inventory by file area
- Classified warning inventory by warning type
- Ranked cleanup queue for Sprint 30

### Completion Criteria
- Every warning in the Day 1 baseline is categorized
- Core-library warning sites are clearly separated from non-library warning sites
- Sprint 30 priority order is explicit and documented

---

## Day 3: Core-Library Hotspot Audit

**Title:** Core Warning Audit  
**Theme:** Inspect the highest-signal warning sites in the main library before editing  
**Time estimate:** 10 hours

### Tasks
1. Review the current warning-producing sites in `src/sparse_lu.c`, `src/sparse_ldlt.c`, `src/sparse_qr.c`, and `src/sparse_svd.c`.
2. Confirm whether each core warning is a pure compile-hygiene issue or a sign of a deeper numerical or API inconsistency.
3. Decide on a standard idiom for infinity-returning helpers and infinity-assignment sites in the core library.
4. Check whether any similar patterns exist in adjacent source files that are currently warning-free but stylistically inconsistent.
5. Write a short design note in Sprint 30 notes capturing the chosen cleanup pattern before code changes begin.

### Deliverables
- Audited list of core-library warning sites
- Chosen cleanup idiom for infinity-related warnings
- Design note for the first code-change batch

### Completion Criteria
- Every Sprint 30 core warning site has a planned fix strategy
- Chosen idiom is documented before implementation
- No unresolved ambiguity remains about the first code batch

---

## Day 4: Core-Library Warning Fixes — Batch 1

**Title:** Core Fixes I  
**Theme:** Remove the first batch of core-library compile warnings  
**Time estimate:** 10 hours

### Tasks
1. Implement the cleanup for the highest-priority warning sites in the first half of the audited core-library files.
2. Preserve runtime behavior exactly while eliminating compile-hygiene issues.
3. Rebuild after each small batch to verify the warning count drops instead of shifting or growing.
4. Update comments only where needed to explain non-obvious warning-avoidance idioms.
5. Record before/after counts for the edited files in Sprint 30 notes.

### Deliverables
- First batch of core-library warning fixes landed
- Incremental rebuild evidence showing reduced warning count
- Updated Sprint 30 notes with per-file deltas

### Completion Criteria
- Edited core files rebuild cleanly for the targeted warning class
- No functional regressions introduced
- Warning count for the edited batch is lower than the Day 1 baseline

---

## Day 5: Core-Library Warning Fixes — Batch 2

**Title:** Core Fixes II  
**Theme:** Complete the Sprint 30 core-library warning cleanup batch  
**Time estimate:** 8 hours

### Tasks
1. Implement the remaining planned fixes in the audited core-library warning files.
2. Sweep nearby helper sites for the same pattern so the style is consistent within the touched files.
3. Re-run both primary build paths and confirm that the targeted core warnings are gone.
4. Update Sprint 30 notes with the final core-library warning delta.

### Deliverables
- Remaining Sprint 30 core-library warning fixes landed
- Clean rebuild evidence for the targeted core warning class
- Updated notes with consolidated core-library reduction data

### Completion Criteria
- Sprint 30’s targeted core warning sites are resolved
- CMake and Makefile rebuilds confirm the reduction
- No new warnings were introduced by the cleanup batch

---

## Day 6: Compile-Hygiene Playbook

**Title:** Hygiene Rules  
**Theme:** Define what gets fixed now, what can wait, and how later sprints should decide  
**Time estimate:** 8 hours

### Tasks
1. Write the Sprint 30 compile-hygiene playbook covering must-fix warning classes versus backlog-acceptable warning classes.
2. Define expectations for core library, tests, benchmarks, examples, and generated artifacts separately.
3. Document how to treat compiler-specific warnings versus portable warnings.
4. Document the expected evidence for claiming that a warning class is fully addressed.
5. Link the playbook to the Sprint 30 baseline notes and future Epic 3 hardening work.

### Deliverables
- Compile-hygiene playbook draft
- Area-specific warning policy for core code and auxiliary code
- Rules for portable versus compiler-specific warning handling

### Completion Criteria
- The playbook is specific enough to support future gating work
- Must-fix and later-fix classes are clearly separated
- The playbook aligns with the Sprint 30 scope rather than expanding it

---

## Day 7: Rebuild Workflow Automation

**Title:** Rebuild Workflow  
**Theme:** Make warning comparison and validation easy to repeat locally  
**Time estimate:** 10 hours

### Tasks
1. Design a repeatable local workflow for clean configure/build/test runs used during Epic 3.
2. Add or refine helper targets/scripts so warning-baseline reproduction does not require manual command sequences every time.
3. Ensure the workflow captures warning output in stable artifact locations.
4. Verify the workflow works from a clean tree and after code edits.
5. Record the intended workflow in Sprint 30 notes and any supporting docs touched by this sprint.

### Deliverables
- Repeatable local rebuild/validation workflow
- Stable warning-capture locations
- Documentation for the local workflow

### Completion Criteria
- A future maintainer can reproduce the baseline and post-fix counts with the documented workflow
- The workflow supports both pre-change and post-change comparison
- The workflow does not bypass existing validation paths

---

## Day 8: Cross-Compiler Reproduction Pass — Apple Clang & Makefile

**Title:** Compiler Reproduction I  
**Theme:** Separate portable warning debt from path-specific noise  
**Time estimate:** 8 hours

### Tasks
1. Re-run the warning baseline on the primary Apple Clang CMake path after the core cleanup batch.
2. Re-run the warning baseline on the Makefile path after the same cleanup batch.
3. Compare the two result sets and identify which warning classes are shared versus path-specific.
4. Update the baseline artifact to show which warnings remain portable and which remain toolchain-specific.
5. Record any surprises that should inform later Epic 3 sprints.

### Deliverables
- Updated cross-path warning comparison
- Shared-versus-path-specific warning analysis
- Revised baseline artifact

### Completion Criteria
- The team can tell which warnings are genuinely portable problems
- Remaining warning classes are mapped to specific build paths
- Baseline artifacts reflect the post-core-fix state

---

## Day 9: Cross-Compiler Reproduction Pass — Stricter Compile Sweep

**Title:** Compiler Reproduction II  
**Theme:** Stress the current cleanup under a stricter compile-only pass  
**Time estimate:** 10 hours

### Tasks
1. Run at least one stricter compile-only pass or warning-focused quality invocation beyond the default build path.
2. Identify warnings that do not appear in the default build but do appear under stricter settings.
3. Decide which of those findings are actionable in Sprint 30 and which are inventory only for later sprints.
4. Update the warning taxonomy and notes with the stricter-pass findings.
5. Reconfirm that the Sprint 30 core fixes still hold under the stricter pass.

### Deliverables
- Stricter compile-sweep findings
- Updated warning taxonomy with stricter-pass annotations
- Actionable versus deferred split for stricter warnings

### Completion Criteria
- A stricter compile sweep has been run and documented
- New findings are triaged rather than mixed into the baseline blindly
- Sprint 30 scope remains controlled

---

## Day 10: Non-Library Triage — Tests

**Title:** Test Warning Triage  
**Theme:** Build the first realistic cleanup queue for warnings outside the core library  
**Time estimate:** 8 hours

### Tasks
1. Audit the warning-producing test files identified in the baseline.
2. Identify which test warnings are caused by dormant scaffolding, which are caused by initializer drift, and which are incidental.
3. Mark the first-fix subset that should land in Sprint 32 or later.
4. Capture any test-structure issues that affect coverage honesty, not just compile noise.
5. Record the triage results in Sprint 30 notes.

### Deliverables
- Test-warning triage inventory
- First-fix subset for future test cleanup
- Notes tying warnings to broader test-structure concerns

### Completion Criteria
- Test warnings are no longer a single undifferentiated backlog
- A specific future queue exists for dormant scaffolding and initializer cleanup
- The triage aligns with the EPIC 3 review findings

---

## Day 11: Non-Library Triage — Benchmarks & Examples

**Title:** Tooling Warning Triage  
**Theme:** Build the first realistic cleanup queue for benchmark and example warnings  
**Time estimate:** 8 hours

### Tasks
1. Audit the warning-producing benchmark and example files identified in the baseline.
2. Identify portability issues, stale CLI/API drift, and initializer-pattern drift separately.
3. Mark the benchmark/example first-fix subset that should land in Sprint 31.
4. Capture any documentation mismatches discovered during the triage.
5. Update Sprint 30 notes with the benchmark/example cleanup queue.

### Deliverables
- Benchmark/example warning triage inventory
- Sprint 31 first-fix subset identified
- Documentation-drift notes for future cleanup

### Completion Criteria
- Benchmark/example warnings are organized into actionable categories
- Sprint 31 has a specific input queue rather than a generic cleanup goal
- Documentation drift is recorded where it affects usability

---

## Day 12: Baseline Reconciliation & Playbook Finalization

**Title:** Baseline Reconciliation  
**Theme:** Bring together the warning data, cleanup deltas, and policy into one coherent sprint artifact  
**Time estimate:** 8 hours

### Tasks
1. Reconcile Day 1 baseline counts with the post-core-fix warning counts.
2. Confirm that the compile-hygiene playbook matches what was actually observed during the sprint.
3. Finalize the first-fix queues for tests, benchmarks, examples, and stricter-pass findings.
4. Clean up Sprint 30 notes into a reviewable sprint artifact rather than ad hoc scratch notes.
5. Prepare the final validation checklist for the last two days.

### Deliverables
- Reconciled before/after warning counts
- Finalized compile-hygiene playbook for Sprint 30
- Reviewable Sprint 30 notes artifact

### Completion Criteria
- Before/after numbers are internally consistent
- Playbook and triage notes are ready for handoff to later sprints
- Final validation checklist is complete

---

## Day 13: Full Validation Run

**Title:** Validation Pass  
**Theme:** Prove the cleanup work did not disturb project behavior  
**Time estimate:** 8 hours

### Tasks
1. Run the full clean configure/build/test flow using the repeatable workflow created earlier in the sprint.
2. Confirm that the post-cleanup build still passes all existing validation gates practical for Sprint 30.
3. Confirm that the warning-count artifacts are generated correctly from the final code state.
4. Investigate and resolve any regressions uncovered by the final validation pass.
5. Record final validation output in Sprint 30 notes.

### Deliverables
- Final clean validation run for Sprint 30
- Final warning-count artifacts from the post-cleanup state
- Validation results recorded in sprint notes

### Completion Criteria
- Validation passes with the Sprint 30 changes in place
- Warning artifacts reflect the final state, not an intermediate state
- No unresolved regression remains open at end of day

---

## Day 14: Sprint Closeout & Handoff

**Title:** Sprint 30 Closeout  
**Theme:** Lock the results, capture the delta, and prepare the next sprint inputs  
**Time estimate:** 8 hours

### Tasks
1. Summarize Sprint 30 outcomes: baseline established, core warnings reduced, workflow documented, and future queues identified.
2. Record the exact before/after warning counts and any residual warning classes intentionally deferred.
3. Convert Sprint 30 notes into final sprint-plan follow-up inputs for Sprint 31 and later.
4. Verify that the branch contents and documentation reflect the final sprint state cleanly.
5. Perform a final sanity pass on the plan and notes for consistency and completeness.

### Deliverables
- Final Sprint 30 notes with before/after warning counts
- Handoff inputs for Sprint 31 and later EPIC 3 sprints
- Clean Sprint 30 closeout state

### Completion Criteria
- Sprint 30 ends with a documented baseline and documented reduction
- Later sprints have explicit, prepared inputs
- The sprint artifact is complete, internally consistent, and ready for review
