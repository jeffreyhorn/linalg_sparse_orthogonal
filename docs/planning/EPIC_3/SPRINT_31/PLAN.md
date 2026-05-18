# Sprint 31 Plan: Benchmark Tooling Sync & Portability Cleanup

**Sprint Duration:** 14 days  
**Goal:** Bring benchmark utilities back in sync with the current API, remove the known benchmark/example portability debt identified by Sprint 30, and define a lightweight compile-only tooling gate so benchmark drift is caught earlier. This sprint implements the Sprint 31 section of `docs/planning/EPIC_3/PROJECT_PLAN.md` (lines 43-75).

**Starting Point:** Sprint 30 closed the core-library warning cleanup and left the authoritative Apple Clang CMake full-tree inventory at `112` warnings: `src` `0`, `tests` `98`, `benchmarks` `13`, `examples` `1`. The Sprint 30 handoff identifies the first benchmark/tooling cleanup queue in `benchmarks/bench_main.c`, `benchmarks/bench_convergence.c`, `benchmarks/bench_colamd.c`, `benchmarks/bench_chol_csc.c`, `benchmarks/bench_ldlt_csc.c`, and `examples/example_colamd.c`. The main known issues are stale reorder CLI/help coverage, benchmark portability warnings around `_POSIX_C_SOURCE` and `snprintf`, and brittle positional option-struct initialization in public-facing tooling.

**End State:** Sprint 31 leaves behind benchmark tooling that matches the current reorder API surface, removes the known benchmark/example portability and initializer-drift issues in the named files, aligns benchmark help/flag behavior across the main tooling entry points, adds a documented compile-only benchmark/example quality gate, and updates the relevant documentation so the developer-facing guidance matches the actual supported behavior.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 120 hours, matching the Sprint 31 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 31 Scope Audit & Baseline

**Title:** Tooling Baseline  
**Theme:** Convert the Sprint 30 handoff into a precise Sprint 31 work inventory  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 31 section of `docs/planning/EPIC_3/PROJECT_PLAN.md` and the Sprint 30 handoff so the cleanup scope stays anchored to the intended benchmark/example queue.
2. Inspect the current state of `benchmarks/bench_main.c`, `benchmarks/bench_convergence.c`, `benchmarks/bench_colamd.c`, `benchmarks/bench_chol_csc.c`, `benchmarks/bench_ldlt_csc.c`, and `examples/example_colamd.c`.
3. Reproduce the current benchmark/example warning state on the authoritative Apple Clang CMake path and isolate which warnings are still live in the Sprint 31 target files.
4. Record the current CLI/help behavior for `bench_main`, `bench_reorder`, and the specialized benchmark entry points so later consistency work has a baseline.
5. Open Sprint 31 working notes and record the file-by-file cleanup queue, grouped by API drift, portability warning, initializer drift, and documentation drift.

### Deliverables
- Sprint 31 file-by-file work inventory
- Current benchmark/example warning snapshot for the Sprint 31 target files
- Baseline notes for benchmark CLI/help behavior

### Completion Criteria
- Every named Sprint 31 target file has a documented problem statement
- Benchmark/example warning sites are separated by category before edits begin
- The sprint scope is explicitly limited to Sprint 31 tooling cleanup, not broader test cleanup

---

## Day 2: `bench_main` CLI & Reorder Audit

**Title:** CLI Gap Audit  
**Theme:** Define exactly how `bench_main` is out of sync with the current reorder API  
**Time estimate:** 8 hours

### Tasks
1. Audit `bench_main` usage text, reorder parsing, and `reorder_name()` coverage against the reorder modes the library currently supports.
2. Compare `bench_main` flag names and accepted reorder values with `bench_reorder` and any public reorder documentation.
3. Identify stale help text, missing enum coverage, rejected-but-supported reorder names, and any output labels that no longer match the underlying mode.
4. Decide on the canonical accepted spellings and output labels for `none`, `rcm`, `amd`, `colamd`, and `nd`.
5. Write a short Sprint 31 design note capturing the intended `bench_main` CLI contract before implementation starts.

### Deliverables
- Audited list of `bench_main` CLI/help/API drift issues
- Canonical reorder-mode contract for benchmark tooling
- Design note for the `bench_main` cleanup

### Completion Criteria
- The current `bench_main` drift is fully enumerated
- Canonical reorder names and labels are documented before code changes
- No ambiguity remains about the intended CLI behavior for supported reorder modes

---

## Day 3: `bench_main` Sync Fixes — Parsing & Help

**Title:** `bench_main` Sync I  
**Theme:** Implement the first half of the `bench_main` API-alignment cleanup  
**Time estimate:** 10 hours

### Tasks
1. Update `bench_main` parsing so it accepts the reorder modes the library already supports, including `colamd` and `nd`.
2. Update `reorder_name()` and any related label-emission helpers so printed output matches the supported reorder set.
3. Refresh usage/help text so it reflects the real supported reorder modes and any relevant defaults.
4. Rebuild incrementally after each small batch to verify the new code paths compile and route correctly.
5. Record before/after behavior in Sprint 31 notes for each touched `bench_main` CLI path.

### Deliverables
- `bench_main` reorder parsing updated for current API coverage
- `bench_main` usage/help text aligned to supported reorder modes
- Incremental validation notes for the edited CLI paths

### Completion Criteria
- `bench_main` no longer rejects reorder values that the library supports
- Help output matches the actual accepted reorder set
- No new drift is introduced between parsing, labels, and usage text

---

## Day 4: Benchmark Behavior Consistency — Cross-Tool Alignment

**Title:** Tooling Consistency  
**Theme:** Align benchmark help text, labels, and reorder coverage across entry points  
**Time estimate:** 8 hours

### Tasks
1. Compare `bench_main`, `bench_reorder`, and any specialized benchmark programs that expose reorder-related behavior.
2. Fix stale labels, help text, or reorder-mode presentation differences that would confuse a maintainer moving between tools.
3. Normalize any obvious inconsistencies in emitted mode names, help examples, or flag descriptions.
4. Confirm that the Sprint 30 stale CLI/help drift is fully closed rather than partially fixed in only one program.
5. Update Sprint 31 notes with a simple consistency matrix showing which tools expose which modes and flags.

### Deliverables
- Cross-tool reorder/help consistency cleanup
- Consistency matrix for benchmark entry points
- Updated notes confirming the stale CLI/help drift is closed

### Completion Criteria
- Benchmark entry points present reorder modes consistently
- No tool advertises stale or unsupported reorder values
- Sprint 31’s behavior-consistency item has a documented before/after state

---

## Day 5: Portability Warning Audit

**Title:** Portability Audit  
**Theme:** Turn benchmark/example portability debt into a precise implementation plan  
**Time estimate:** 8 hours

### Tasks
1. Audit the `_POSIX_C_SOURCE`, `snprintf`, and any remaining implicit-declaration issues in `benchmarks/bench_main.c`, `benchmarks/bench_convergence.c`, and related benchmark entry points.
2. Confirm whether the known problems are pure include/feature-macro issues, API-selection issues, or a mix of both.
3. Check nearby benchmark/example files for the same portability pattern so fixes can be applied consistently instead of one file at a time.
4. Decide on the standard portability pattern Sprint 31 will use for benchmark entry points.
5. Record the chosen pattern and file list in Sprint 31 notes before edits begin.

### Deliverables
- Audited list of portability-warning sites in Sprint 31 target files
- Chosen portability pattern for benchmark entry points
- File list for the portability-fix batch

### Completion Criteria
- Every known Sprint 31 portability issue has a documented fix strategy
- The benchmark portability pattern is chosen before implementation
- The portability batch is scoped tightly enough to finish within Sprint 31

---

## Day 6: Portability Fixes — Benchmark Entry Points

**Title:** Portability Fixes  
**Theme:** Remove the known benchmark portability debt in the priority files  
**Time estimate:** 10 hours

### Tasks
1. Implement the chosen portability pattern in `benchmarks/bench_main.c` and `benchmarks/bench_convergence.c`.
2. Apply the same fix style to any adjacent benchmark/example entry points in Sprint 31 scope that share the same problem.
3. Verify that the fixes remove the known implicit-declaration and feature-macro debt without changing intended runtime behavior.
4. Rebuild after each small batch so any portability regressions are caught immediately.
5. Record the warning delta for each edited file in Sprint 31 notes.

### Deliverables
- Portability fixes landed in the Sprint 31 priority benchmark files
- Incremental rebuild evidence for the warning reduction
- Updated notes with per-file portability cleanup results

### Completion Criteria
- The known benchmark portability warnings in the named files are resolved
- Runtime behavior is preserved while compile portability improves
- No new portability warnings are introduced in nearby touched code

---

## Day 7: Designated Initializers — Benchmark Batch

**Title:** Initializer Cleanup I  
**Theme:** Replace brittle positional options initialization in benchmark programs  
**Time estimate:** 8 hours

### Tasks
1. Replace positional options-struct initialization in `benchmarks/bench_colamd.c` and `benchmarks/bench_chol_csc.c` with designated initializers.
2. Confirm that each edited initializer documents the intended non-default fields clearly and safely against future struct growth.
3. Check for any companion comments or example snippets in those files that still imply positional initialization.
4. Rebuild incrementally to confirm warning reduction and unchanged behavior.
5. Record the initializer cleanup pattern in Sprint 31 notes so the same style is used consistently in later files.

### Deliverables
- Designated-initializer cleanup in the first benchmark batch
- Notes documenting the chosen public-facing initialization style
- Incremental rebuild evidence for the edited files

### Completion Criteria
- The targeted benchmark files no longer rely on brittle positional initialization
- The designated-initializer style is explicit and consistent
- The cleanup reduces initializer-drift risk for future option-struct growth

---

## Day 8: Designated Initializers — Remaining Benchmark/Example Batch

**Title:** Initializer Cleanup II  
**Theme:** Finish the public-facing initializer cleanup in Sprint 31 scope  
**Time estimate:** 8 hours

### Tasks
1. Replace positional options-struct initialization in `benchmarks/bench_ldlt_csc.c` and `examples/example_colamd.c` with designated initializers.
2. Update any example-facing comments or nearby code patterns that still teach brittle positional initialization.
3. Sweep the four Sprint 31 initializer-cleanup files for consistency of field ordering and readability.
4. Rebuild after each small batch and record the warning delta for the final initializer queue.
5. Update Sprint 31 notes with a consolidated initializer-cleanup summary.

### Deliverables
- Designated-initializer cleanup completed in the remaining Sprint 31 files
- Public-facing example style aligned with the safer initialization pattern
- Consolidated notes for the Sprint 31 initializer batch

### Completion Criteria
- All Sprint 31 initializer-cleanup target files use designated initialization
- Public-facing examples no longer teach the brittle style
- The initializer batch is complete and documented

---

## Day 9: Benchmark Behavior Audit — Labels, Flags, and Coverage

**Title:** Behavior Audit  
**Theme:** Verify that benchmark tools behave consistently after the main code fixes land  
**Time estimate:** 8 hours

### Tasks
1. Re-run the benchmark entry points touched in Sprint 31 and verify their help text, accepted flags, and emitted labels match the documented contract.
2. Compare reorder-mode coverage across `bench_main`, `bench_reorder`, and the specialized programs after the cleanup.
3. Close any remaining small drift in output labels, flag documentation, or accepted values that survived the earlier implementation days.
4. Record a post-cleanup behavior matrix in Sprint 31 notes.
5. Identify any residual benchmark behavior drift that should be deferred rather than expanded into Sprint 31 scope.

### Deliverables
- Post-cleanup benchmark behavior matrix
- Any remaining small consistency fixes needed after the main cleanup
- Deferred residual-drift notes if anything remains out of Sprint 31 scope

### Completion Criteria
- The main benchmark tools behave consistently after cleanup
- Remaining drift, if any, is explicitly deferred rather than left implicit
- Sprint 31’s behavior-consistency work is validated against the actual programs

---

## Day 10: Compile-Only Tooling Gate Design

**Title:** Tooling Gate Design  
**Theme:** Define how benchmarks and examples should be compile-checked without forcing expensive full execution  
**Time estimate:** 8 hours

### Tasks
1. Review the current Makefile, lint flow, and Sprint 30 rebuild workflow to identify the right insertion point for a benchmark/example compile-only quality gate.
2. Decide which benchmark and example targets should be compile-checked in the default quality flow and which should remain outside that gate.
3. Define the contract for the compile-only gate: what it builds, what it proves, and what it intentionally does not prove.
4. Choose whether the gate belongs in `make lint`, a new explicit target, or both.
5. Write a short design note documenting the chosen approach and how it catches Sprint 31’s benchmark drift class earlier.

### Deliverables
- Compile-only benchmark/example gate design note
- Chosen target scope for benchmark/example compile checking
- Clear contract for what the gate validates

### Completion Criteria
- The compile-only gate design is specific enough to implement cleanly
- The chosen approach fits the existing quality workflow instead of bypassing it
- The gate scope is large enough to catch tooling drift but small enough to stay practical

---

## Day 11: Compile-Only Tooling Gate Implementation

**Title:** Tooling Gate Implementation  
**Theme:** Land the compile-only benchmark/example quality check in the local workflow  
**Time estimate:** 10 hours

### Tasks
1. Implement the compile-only benchmark/example gate in the chosen Makefile or helper-script location.
2. Ensure the gate builds the intended benchmark/example targets with warning visibility while avoiding unnecessary full benchmark execution.
3. Integrate the gate cleanly with the existing local quality workflow and any related Sprint 30 rebuild commands.
4. Verify that the gate catches the Sprint 31 target files and fails meaningfully if one of the known drift classes is reintroduced.
5. Document the command and intended usage in Sprint 31 notes and any workflow docs touched by the implementation.

### Deliverables
- Working compile-only benchmark/example quality gate
- Integration of the gate into the local workflow
- Documentation of how to invoke and interpret the gate

### Completion Criteria
- The gate compiles the intended benchmark/example targets without running full benchmark workloads
- The gate is usable by maintainers as part of normal local validation
- The gate would have caught the core Sprint 31 benchmark drift issues earlier

---

## Day 12: Documentation Refresh

**Title:** Documentation Sync  
**Theme:** Bring benchmark/example documentation back in line with the cleaned-up tooling behavior  
**Time estimate:** 8 hours

### Tasks
1. Update README and any benchmark-facing docs where reorder-mode coverage or usage text no longer matched the actual supported behavior before Sprint 31.
2. Update any public-facing examples or doc snippets that still teach positional option initialization.
3. Document the new compile-only benchmark/example gate and when maintainers should use it.
4. Cross-check that docs now match the actual accepted reorder values and benchmark invocation patterns.
5. Record the documentation delta in Sprint 31 notes.

### Deliverables
- Documentation updates for benchmark behavior and usage
- Public-facing examples/docs aligned with designated-initializer guidance
- Documentation for the new compile-only tooling gate

### Completion Criteria
- Benchmark docs match actual supported behavior after cleanup
- Public-facing guidance no longer teaches the brittle initializer style
- The compile-only gate is documented where maintainers will find it

---

## Day 13: Validation Sweep & Residual Mechanical Cleanup

**Title:** Validation Pass  
**Theme:** Rebuild the benchmark/example targets cleanly and close the last Sprint 31 mechanical follow-through  
**Time estimate:** 10 hours

### Tasks
1. Re-run the benchmark/example compile-only gate and verify that all Sprint 31 target files now pass cleanly.
2. Re-run the authoritative warning reproduction flow and compare the benchmark/example warning counts to the Sprint 30 closing baseline.
3. Close the residual mechanical `-Wdouble-promotion` follow-through in `benchmarks/bench_convergence.c` if it is still present after the earlier portability cleanup.
4. Run the standard local validation flow: `make format`, `make lint`, and `make test`.
5. Record final before/after counts and any remaining deferred benchmark/example warning debt in Sprint 31 notes.

### Deliverables
- Final benchmark/example compile-only validation evidence
- Residual `bench_convergence.c` mechanical cleanup, if still needed
- Final warning-count comparison against the Sprint 30 close state

### Completion Criteria
- Sprint 31 target files rebuild cleanly under the new tooling gate
- Standard validation passes cleanly after the final cleanup batch
- Remaining deferred benchmark/example debt, if any, is explicitly documented

---

## Day 14: Closeout, Handoff & Deferred Queue Update

**Title:** Sprint Closeout  
**Theme:** Turn Sprint 31 cleanup results into durable handoff inputs for later Epic 3 work  
**Time estimate:** 8 hours

### Tasks
1. Write the Sprint 31 closeout notes summarizing which benchmark/example issues were fully resolved and which were deferred.
2. Update the deferred queue for Sprint 32 and later so any remaining tooling/test cleanup stays attached to named files and warning classes.
3. Record the final benchmark/example warning delta relative to the Sprint 30 handoff.
4. Confirm that the new compile-only tooling gate, documentation updates, and benchmark behavior cleanup are all reflected in the sprint notes.
5. Prepare the inputs needed for the Sprint 31 retrospective and any later Epic 3 planning updates.

### Deliverables
- Sprint 31 closeout notes and deferred-queue update
- Final benchmark/example warning delta summary
- Handoff inputs for the next Epic 3 sprint

### Completion Criteria
- Sprint 31 results are documented in a form the next sprint can use directly
- Any remaining cleanup is explicitly routed rather than left implicit
- The sprint ends with a clear before/after benchmark tooling state
