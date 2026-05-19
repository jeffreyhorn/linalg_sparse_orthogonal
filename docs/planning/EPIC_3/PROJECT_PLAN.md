# Project Plan: linalg_sparse_orthogonal -- Sprints 30-39 (Epic 3)

Based on findings from `review-codex-2026-05-15.md` and the remediation plan in `todo-codex-2026-05-15.md`.

Epic 3 is intentionally a quality-hardening epic, not a feature-addition epic. The scope is limited to cleaning up, validating, documenting, and enforcing the quality of the code that already exists.

---

## Sprint 30: Warning Baseline & Core Compile-Hygiene Triage

**Duration:** 14 days (~120 hours)

**Goal:** Establish a measured warning baseline for the full tree, fix the highest-signal core-library warnings first, and create a repeatable compile-quality workflow that later sprints can enforce.

### Prerequisites from previous Sprints

- Epic 2 baseline: current CMake/Makefile build scripts, CI matrix, direct solvers, iterative solvers, eigensolvers, benchmarks, and tests.
- EPIC 3 review and remediation plan: the warning-cleanup, tooling-sync, and test-honesty findings are the direct input to this sprint.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Warning baseline inventory | Capture a clean CMake build warning baseline and classify every warning by source area (`src/`, `tests/`, `benchmarks/`, `examples/`) and by type (`double-promotion`, missing-field-initializers, implicit declaration, unused-function, enum drift). | 16 hrs |
| 2 | Core-library `INFINITY` / `NAN` cleanup | Fix the current compile warnings in `src/sparse_lu.c`, `src/sparse_ldlt.c`, `src/sparse_qr.c`, and `src/sparse_svd.c` without changing semantics. Standardize the idiom used for infinity-returning helpers. | 24 hrs |
| 3 | Compile-hygiene playbook | Document which warning classes are considered must-fix immediately versus backlog-acceptable, so later sprints can gate the build consistently instead of debating each warning ad hoc. | 8 hrs |
| 4 | Rebuild loop automation | Add a repeatable local quality invocation pattern for clean configure/build/test runs so warning-count comparisons are easy to reproduce during the epic. | 16 hrs |
| 5 | Cross-compiler reproduction pass | Reproduce the baseline on the primary local/toolchain paths that are practical now (Apple Clang via CMake, Makefile path, and at least one stricter compile-only pass) to distinguish portable warnings from compiler-specific noise. | 20 hrs |
| 6 | Initial non-library triage | Identify the first-fix subset in tests, benchmarks, and examples so the remaining sprints can be sequenced with real effort estimates rather than broad assumptions. | 16 hrs |
| 7 | Validation & sprint notes | Re-run full validation after the first cleanup batch and record before/after warning counts in Sprint 30 notes. | 20 hrs |

### Deliverables

- Measured warning baseline for the full repository
- Core-library warning sites triaged and highest-signal issues fixed
- Repeatable local compile-quality workflow
- Sprint notes recording baseline and first reduction in warning count

**Total estimate:** ~120 hours

---

## Sprint 31: Benchmark Tooling Sync & Portability Cleanup

**Duration:** 14 days (~120 hours)

**Goal:** Bring benchmark utilities back in sync with the current API and remove known portability warnings in benchmark code so the developer tooling is as trustworthy as the main library.

### Prerequisites from previous Sprints

- Sprint 30 warning inventory and compiler-reproduction notes
- Sprint 30 handoff and retrospectives, which identify `benchmarks/bench_main.c`, `benchmarks/bench_convergence.c`, `benchmarks/bench_colamd.c`, `benchmarks/bench_chol_csc.c`, `benchmarks/bench_ldlt_csc.c`, and `examples/example_colamd.c` as the first deferred tooling-cleanup queue
- Existing benchmark suite (`bench_main`, `bench_reorder`, `bench_colamd`, `bench_convergence`, and related scripts)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | `bench_main` reorder support | Update `bench_main` usage text, `reorder_name()` coverage, and CLI parsing so it supports the reorder modes the library already supports (`COLAMD`, `ND`, and existing modes). Explicitly close the Sprint 30 handoff items where `bench_main.c` still rejects `colamd` / `nd` or prints stale help text. | 20 hrs |
| 2 | Benchmark portability warnings | Fix the `_POSIX_C_SOURCE` / `snprintf` issues in `bench_main`, `bench_convergence`, and any similar benchmark sources. Remove the known implicit-declaration portability debt called out by Sprint 30. | 20 hrs |
| 3 | Benchmark/example option initializers | Replace brittle positional options-struct initialization in the specific Sprint 30 handoff files `benchmarks/bench_colamd.c`, `benchmarks/bench_chol_csc.c`, `benchmarks/bench_ldlt_csc.c`, and `examples/example_colamd.c` with designated initializers. | 20 hrs |
| 4 | Benchmark behavior consistency audit | Verify that benchmark help text, supported flags, emitted labels, and reorder-mode coverage are consistent across `bench_main`, `bench_reorder`, and specialized benchmark programs. Include the stale CLI/help drift identified in Sprint 30. | 16 hrs |
| 5 | Compile-only benchmark gate design | Define how benchmarks and examples should be compile-checked in the quality workflow so tooling drift is caught even when full benchmark execution is too expensive for every run. | 16 hrs |
| 6 | Documentation updates | Refresh README / benchmark documentation where the current text no longer matches the supported reorder modes or usage patterns. Update examples/docs to teach designated-initializer usage instead of brittle positional initialization. | 12 hrs |
| 7 | Validation | Rebuild benchmark and example targets cleanly, including the residual `bench_convergence.c` mechanical `-Wdouble-promotion` follow-through from the Sprint 30 handoff, and re-run the normal library validation flow. | 16 hrs |

### Deliverables

- `bench_main` supports the current reorder API surface
- Benchmark portability warnings removed from the known failing files
- Benchmark/example option structs use designated initialization
- Benchmark documentation matches actual supported behavior
- The Sprint 30 benchmark/example deferred queue is explicitly closed or reduced in the named files

**Total estimate:** ~120 hours

---

## Sprint 32: Test-Suite Truthfulness & Dormant Scaffold Resolution

**Duration:** 14 days (~128 hours)

**Goal:** Make the test suite accurately represent what is executed and protected today by resolving dormant scaffolding, formalizing slow/experimental checks, and reducing warning noise from test code that does not actually run.

### Prerequisites from previous Sprints

- Sprint 30 warning taxonomy for unused-function and missing-initializer warnings
- Sprint 30 handoff and retrospectives, which identify `tests/test_reorder_nd.c` as the first dormant-scaffolding target and name the follow-on high-volume initializer and mechanical double-promotion files in the test tree
- Sprint 31 handoff and retrospective, which reduce the benchmark/example queue to zero and convert the remaining deferred warning debt into an explicitly test-only named-file queue
- Sprint 31 benchmark/example initializer cleanup patterns that can be reused in tests

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Dormant-test inventory | Audit the full `tests/` tree for compiled-but-unexecuted test bodies, commented-out `RUN_TEST` sites, and historical “failing as expected” scaffolds. | 16 hrs |
| 2 | `test_reorder_nd.c` refactor | Resolve the current dormant scaffolding in `tests/test_reorder_nd.c`: either delete historical-only code, move evidence into docs, or formalize the checks behind explicit skip/experimental mechanisms. Explicitly close the Sprint 30 deferred issues around non-executed scaffolding, `-Wunused-function`, and designated-initializer drift in this file. | 24 hrs |
| 3 | Slow/experimental test policy | Define a project-level distinction between normal CI tests, slow opt-in tests, and experimental/historical checks so future work does not repopulate the main suite with dormant code. | 16 hrs |
| 4 | Test framework extensions | Add minimal framework support or naming conventions for skipped/experimental tests where that is cleaner than deletion. | 20 hrs |
| 5 | Test warning cleanup | Remove or fix the test-side compile warnings that remain after dormant scaffolding is resolved, especially those caused by stale positional initializers and unused static functions. Include the high-volume initializer files named in the Sprint 30 and Sprint 31 handoffs (`tests/test_ldlt.c`, `tests/test_chol_csc.c`, `tests/test_colamd.c`, `tests/test_cholesky.c`, `tests/test_sprint12_integration.c`, `tests/test_sprint18_integration.c`, `tests/test_sprint19_integration.c`, `tests/test_sprint20_integration.c`, `tests/test_reorder.c`, `tests/test_etree.c`) and the residual mechanical `-Wdouble-promotion` files (`tests/test_sprint20_integration.c`, `tests/test_svd.c`, `tests/test_sprint6_integration.c`, `tests/test_sprint18_integration.c`, `tests/test_bidiag.c`, `tests/test_sprint19_integration.c`, `tests/test_qr.c`, `tests/test_sprint10_integration.c`, `tests/test_block_solvers.c`, `tests/test_ilu.c`, `tests/test_lu_csr.c`, `tests/test_sprint5_integration.c`). | 24 hrs |
| 6 | Coverage-honesty documentation | Update planning/docs notes to explain which checks are active, which are opt-in, and which historical experiments are retained only as evidence. | 12 hrs |
| 7 | Validation | Re-run the full `ctest` / Makefile test suites and verify that the active suite remains green after the structural cleanup. | 16 hrs |

### Deliverables

- Dormant ND test scaffolding removed or formalized
- Clear split between normal, slow, and experimental test categories
- Reduced test-tree warning noise
- Documentation that matches the executed protection surface
- The Sprint 30 and Sprint 31 deferred test-warning queues are explicitly tracked in named files rather than left as a generic backlog

**Total estimate:** ~128 hours

---

## Sprint 33: Dead-Code Detection Infrastructure & First Cleanup Pass

**Duration:** 14 days (~124 hours)

**Goal:** Add explicit dead-code-detection support to the engineering workflow, including Makefile targets and reporting, then use that tooling to remove the first batch of definitely-unused code.

### Prerequisites from previous Sprints

- Sprint 30 warning baseline and warning-class taxonomy
- Sprint 32 dormant-test cleanup, which removes a major source of false-positive “dead code” in the test tree
- Sprint 32 closeout/handoff, which confirms zero residual test-warning debt and a live opt-in test policy (`SPARSE_TEST_SLOW`, `SPARSE_TEST_EXPERIMENTAL`)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Dead-code policy | Define what “dead code” means for this C repository: definitely-unused static functions/variables, unreachable helper paths, stale benchmark/test scaffolds, and separately “candidate unused public API” that needs manual review. Explicitly preserve Sprint 32’s truthfulness rules: opt-in wrappers (`RUN_TEST_SLOW`, `RUN_TEST_EXPERIMENTAL`) are not dormant scaffold, `tests/test_framework_optin.c` remains live coverage for that policy, and historical evidence belongs in docs rather than commented-out registrations. | 12 hrs |
| 2 | Makefile `deadcode` target | Add a Makefile target in the shape requested for Epic 3, starting with `cppcheck --enable=all --quiet src/` and `xunused build/compile_commands.json`. Wire the target so `compile_commands.json` is generated reliably before analysis, and document any local tool prerequisites or portability limits. | 24 hrs |
| 3 | Makefile reporting target | Add a reporting-oriented companion target such as `make deadcode-report` or `make deadcode-check` that wraps the raw `cppcheck` + `xunused` output into a readable artifact suitable for local review and later CI use. | 20 hrs |
| 4 | Candidate-public-API audit | Design a lightweight workflow for reviewing possibly-unused public entry points separately from definitely-dead static code, so the project does not delete public API only because local call sites are absent. | 16 hrs |
| 5 | First cleanup pass | Remove or refactor the first batch of definitely-unused internal code discovered by the new tooling, prioritizing tests, benchmarks, examples, and private helpers before touching any public API. | 24 hrs |
| 6 | Documentation | Document how to run the dead-code targets, what their limitations are, and which finding classes are actionable versus advisory. Preserve the Sprint 32 distinction between active tests, opt-in tests, and historical evidence so dead-code reporting does not collapse those categories back together. | 12 hrs |
| 7 | Validation | Re-run build/test/quality flows after the first dead-code cleanup pass. Preserve the Sprint 32 closeout baseline by checking `make lint`, `make test`, `ctest -N`, and full `ctest` in addition to the dead-code targets so the active suite size remains auditable and the warning-clean state is not lost. | 16 hrs |

### Deliverables

- Makefile dead-code-detection target(s), including `deadcode` built around `cppcheck --enable=all --quiet src/` and `xunused build/compile_commands.json`
- Documented dead-code policy and limitations
- First batch of definitely-unused internal code removed
- Dead-code report flow ready for later CI integration

**Total estimate:** ~124 hours

---

## Sprint 34: Build-Quality Enforcement Phase 1

**Duration:** 14 days (~136 hours)

**Goal:** Turn the cleanup work from Sprints 30-33 into enforceable build-quality rules, starting with warning cleanliness and dead-code checks on the most important targets and toolchains.

### Prerequisites from previous Sprints

- Sprint 30 core-warning cleanup and warning baseline
- Sprint 31 benchmark/example compile cleanup
- Sprint 33 `make deadcode` / reporting targets
- Sprint 33 closeout/handoff, which leaves no residual definitely-unused
  internal cleanup queue and records the remaining compile-db coverage gaps plus
  the current serial-only dead-code workflow constraint
- Sprint 32 closeout/handoff, which leaves no inherited test-warning or initializer backlog; later warning work should be treated as regression prevention

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Warning gate design | Decide which targets must be warning-clean first (`src/`, key tests, benchmarks/examples compile-only) and how that requirement differs across primary compilers. Start from the Sprint 32 closeout invariant that the authoritative Apple Clang CMake full-tree path is at `0` warnings and should stay there. | 16 hrs |
| 2 | Makefile compile-quality targets | Add or refine Makefile targets that perform warning-clean compile checks on the agreed target sets without conflating them with normal runtime test execution. Keep the Sprint 32 expectation that `make lint` and `make test` remain part of the normal local quality path. | 24 hrs |
| 3 | CMake parity | Ensure equivalent compile-quality checks can be run from the CMake path or at least validated from the CMake-generated build tree, so the project does not regress into Make-only quality guarantees. Keep `ctest -N` and full `ctest` usable as auditable views of the active executed suite. As part of dead-code parity, either broaden the dead-code compilation database to cover the current Sprint 33 gap (`bench_svd` plus the six missing examples) or preserve that exclusion list explicitly in generated reporting and enforcement. | 20 hrs |
| 4 | CI integration phase 1 | Add non-flaky warning/dead-code checks to the primary CI jobs, initially for the strictest reliable compiler/target combinations. Preserve Sprint 33's current shared-build-tree limitation by enforcing serialized dead-code execution or by isolating dead-code build/artifact paths before CI starts running those targets concurrently. | 28 hrs |
| 5 | Initializer-regression cleanup | If new high-noise positional options-struct initializers appear in reviewed targets, migrate them to designated initializers as part of warning-gate enforcement. Sprint 32 closed the inherited initializer backlog, so this item is for regression prevention rather than carried debt. | 20 hrs |
| 6 | Failure-message quality | Make the new quality targets fail with clear, actionable output so future contributors can fix issues without reverse-engineering the gate. | 12 hrs |
| 7 | Validation | Re-run local and CI-equivalent flows with the new gates enabled. | 16 hrs |

### Deliverables

- Warning-clean quality targets in Makefile/CMake workflow
- Dead-code checks integrated into quality flow
- First CI enforcement pass for compile-quality drift
- Reduced reliance on manual warning review

**Total estimate:** ~136 hours

---

## Sprint 35: Public Docs, Header Examples & API-Usage Consistency

**Duration:** 14 days (~128 hours)

**Goal:** Make the public-facing guidance consistent with the current codebase by updating headers, README, examples, and usage notes to teach stable patterns instead of stale or brittle ones.

### Prerequisites from previous Sprints

- Sprint 31 benchmark/example cleanup patterns
- Sprint 34 compile-quality gates that will keep example/doc code honest going forward

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Header example cleanup | Replace brittle positional options-struct initialization in public header examples with designated initializers across the API surface. | 20 hrs |
| 2 | README / tutorial consistency pass | Audit README, tutorial docs, and example snippets so they match the actual supported reorder modes, option struct layouts, and current best practices. | 24 hrs |
| 3 | Maintainer initialization standard | Add a maintainer-facing rule that public options structs should be shown with designated initializers in docs, examples, and tests unless there is a specific reason not to. | 12 hrs |
| 4 | API precondition language audit | Tighten public documentation around preconditions, especially where a feature has evolved and older examples no longer describe the safest usage pattern. | 20 hrs |
| 5 | Example build validation | Ensure documented examples and shipped example programs still compile cleanly after the documentation/API-usage cleanup. | 20 hrs |
| 6 | Installation/docs polish | Update INSTALL / benchmark docs / quality docs with the new quality and dead-code targets where appropriate. | 16 hrs |
| 7 | Validation | Re-run quality targets and example/benchmark compile checks. | 16 hrs |

### Deliverables

- Public docs teach current stable initialization patterns
- Header and README examples aligned with current API behavior
- Maintainer guidance for future example/doc consistency
- Example programs compile cleanly under the quality flow

**Total estimate:** ~128 hours

---

## Sprint 36: Cross-Platform Quality Parity

**Duration:** 14 days (~144 hours)

**Goal:** Make the hardening work portable across the supported platforms by closing quality gaps between Apple Clang, Linux/GCC-or-Clang paths, and Windows/MSVC where practical.

### Prerequisites from previous Sprints

- Sprint 34 build-quality enforcement phase 1
- Sprint 35 public-doc and example cleanup
- Existing Linux/macOS/Windows CI workflows from Epic 2

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | macOS warning parity | Audit remaining warning differences on the macOS build paths and eliminate avoidable platform-specific noise in the reviewed targets. | 24 hrs |
| 2 | Windows/MSVC quality audit | Review MSVC build output for equivalent quality issues, especially enum handling, initializer drift, missing includes, and warning-level differences that the Unix compilers may not expose. | 28 hrs |
| 3 | CI job alignment | Update CI so compile-quality expectations are explicitly documented per platform rather than implicitly assumed from the Linux job only. | 20 hrs |
| 4 | Script/target portability | Audit quality and dead-code Makefile/scripts for shell or tool assumptions that make them Linux/macOS-only when the project documents broader platform support. | 20 hrs |
| 5 | CMake/Makefile parity report | Produce a small parity report on which quality checks are available from Make, from CMake, and in CI on each platform. | 16 hrs |
| 6 | Targeted fixes | Implement the concrete fixes surfaced by the parity audit across scripts, CI, and auxiliary code. | 20 hrs |
| 7 | Validation | Re-run the practical cross-platform build/test/quality flows and capture results in sprint notes. | 16 hrs |

### Deliverables

- Reduced platform-specific warning and quality drift
- Explicit cross-platform quality expectations
- More portable dead-code / quality tooling
- Sprint notes capturing quality parity status by platform

**Total estimate:** ~144 hours

---

## Sprint 37: Auxiliary-Code Cleanup & Maintainability Refactor

**Duration:** 14 days (~148 hours)

**Goal:** Improve maintainability in the non-core parts of the repository by reducing duplication, clarifying ownership of quality scripts/targets, and simplifying large auxiliary files that are currently expensive to keep clean.

### Prerequisites from previous Sprints

- Sprint 32 test-structure cleanup
- Sprint 36 cross-platform quality parity findings

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Test helper consolidation | Identify duplicated helper patterns in large test files and consolidate them where that reduces maintenance burden without obscuring test intent. | 24 hrs |
| 2 | Benchmark helper consolidation | Reduce repeated benchmark CLI/help/utility logic where a shared helper would make behavior more consistent and easier to keep warning-clean. | 20 hrs |
| 3 | Quality-target normalization | Clean up Makefile target naming and organization for warning, dead-code, formatting, lint, and validation tasks so the workflow is easier to understand and extend. | 20 hrs |
| 4 | Large-file maintainability pass | Tackle one or two oversized quality-problem files (for example `tests/test_reorder_nd.c` or other auxiliary hot spots) to improve readability, locality, and future cleanup cost. | 28 hrs |
| 5 | Comment/documentation cleanup | Remove stale comments or misleading implementation notes in auxiliary code that no longer reflect current behavior. | 16 hrs |
| 6 | Maintainer workflow docs | Document the intended workflow for running warning/dead-code/compile-quality checks locally before PRs. | 20 hrs |
| 7 | Validation | Re-run the main quality/test flows after the maintainability refactor. | 20 hrs |

### Deliverables

- Lower-maintenance test and benchmark scaffolding
- Clearer quality-target layout in the Makefile
- Reduced duplication in auxiliary code
- Updated maintainer workflow documentation

**Total estimate:** ~148 hours

---

## Sprint 38: Coverage, Regression-Proofing & Quality-Gate Expansion

**Duration:** 14 days (~136 hours)

**Goal:** Expand the hardening work from warning/dead-code hygiene into broader regression-proofing so the cleaned-up state is preserved by routine development rather than periodic manual audits.

### Prerequisites from previous Sprints

- Sprint 34 compile-quality gates
- Sprint 37 maintainability cleanup that makes the target set easier to gate

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Coverage-honesty audit | Reconcile coverage expectations and reporting with the cleaned-up test categories so active, experimental, and opt-in checks are represented accurately. | 20 hrs |
| 2 | Quality-gate expansion | Expand the warning/dead-code gates to the next tier of reviewed targets/toolchains once the initial gates have proven stable. | 24 hrs |
| 3 | Compile-only regression coverage | Ensure examples and benchmarks that are not run routinely are still compile-checked in a way that meaningfully protects them from drift. | 20 hrs |
| 4 | Dead-code workflow maturation | Promote the dead-code tooling from advisory reporting toward an actionable regression signal where the false-positive rate is acceptable. Explicitly burn down the Sprint 33 residual `cppcheck` evidence buckets by reviewing or better classifying the supporting-signal set and static-analysis noise so later enforcement is not built on ambiguous findings. | 20 hrs |
| 5 | Release/readiness checklist | Add a concise quality-readiness checklist covering warnings, dead code, test truthfulness, docs/examples consistency, and cross-platform parity. | 16 hrs |
| 6 | CI/reporting polish | Improve artifact/report output for the new quality gates so failures are easy to understand in CI. | 16 hrs |
| 7 | Validation | Re-run the full quality/test matrix practical for the sprint and record the resulting baseline. | 20 hrs |

### Deliverables

- More accurate coverage/readiness signaling
- Broader compile-quality protection
- Dead-code workflow closer to routine enforcement
- Release/readiness checklist for the cleaned-up repository

**Total estimate:** ~136 hours

---

## Sprint 39: Epic 3 Stabilization, Final Audit & Closeout

**Duration:** 14 days (~124 hours)

**Goal:** Finish Epic 3 with a final repository-wide audit, close remaining quality gaps from the sprint notes, and leave behind stable standards and artifacts that keep the cleaned-up state from regressing.

### Prerequisites from previous Sprints

- Sprints 30-38, especially the enforced quality targets, dead-code workflow, test cleanup, docs consistency, and cross-platform parity work
- Sprint 30 compile-hygiene playbook and warning-workflow baseline, which remain the source of truth for full-tree warning claims (`Apple Clang CMake` authoritative; Makefile `all` library-only cross-check)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Final warning audit | Run a repository-wide compile-quality audit and close any remaining warning regressions in the reviewed target set. Use the Sprint 30 workflow and Apple Clang CMake full-tree inventory as the authoritative reference rather than the narrower Makefile `all` path alone. | 24 hrs |
| 2 | Final dead-code audit | Run the dead-code tooling and resolve or explicitly disposition remaining findings so the project exits Epic 3 with a known-clean or known-justified state. This includes any residual Sprint 33-style coverage-gap exclusions, audited public keeps, and `cppcheck` supporting/noise buckets that earlier enforcement work chose to preserve rather than remove. | 20 hrs |
| 3 | Final cross-platform audit | Re-run the supported build/test/quality paths and record any residual platform-specific limitations that remain intentionally out of scope. | 20 hrs |
| 4 | Standards/documentation closeout | Finalize maintainer docs for warning cleanliness, designated initializers, dormant-test policy, and dead-code workflow. | 20 hrs |
| 5 | Epic summary report | Produce a concise Epic 3 summary of what was fixed, what is now enforced, and what residual risks remain for future work. | 12 hrs |
| 6 | Cleanup of temporary scaffolding | Remove temporary allowlists, transitional notes, or sprint-only helpers that should not remain after closeout. | 12 hrs |
| 7 | Final validation | Perform the final clean configure/build/test/quality run and record the end-state baseline. | 16 hrs |

### Deliverables

- Final Epic 3 warning/dead-code/quality audit
- Stable maintainer standards for the cleaned-up codebase
- Documented end-state baseline and residual limitations
- Epic summary suitable for handing off to later feature work

**Total estimate:** ~124 hours
