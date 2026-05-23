# Project Plan: linalg_sparse_orthogonal -- Sprints 40-49 (Epic 4)

Based on findings from `reviews/review-codex-2026-0521.md` and the remediation plan in `reviews/todo-codex-2026-05-21.md`.

Epic 4 is intentionally a structural-refinement epic. The focus is not adding new numerical capability first; it is making the existing codebase easier to reason about, safer to evolve, cheaper to operate, and clearer to use and maintain.

All sprint estimates below are bounded for a 14-day sprint at no more than 12 hours/day (`<= 168 hours` per sprint).

---

## Sprint 40: Baseline, Lifecycle Inventory & Epic 4 Architecture Contract

**Duration:** 14 days (~124 hours)

**Goal:** Establish the measured Epic 4 baseline, convert the review/todo findings into a concrete architectural contract, and define the future ownership model for matrix state, factor state, and quality-policy surfaces before implementation-heavy refactors begin.

### Prerequisites from previous Sprints

- Epic 3 closeout baseline: `make quality-review-full`, serial `make deadcode-report`, serial `make deadcode-check`, and the documented cross-platform reviewed contract.
- Epic 4 review and remediation plan:
  - `reviews/review-codex-2026-0521.md`
  - `reviews/todo-codex-2026-05-21.md`

### Items

| # | Item Name | Item Description | Estimate |
|---|-----------|------------------|----------|
| 1 | Epic 4 baseline capture | Record the current reviewed-quality, reviewed CMake parity, dead-code, allocation-density, and hotspot-file baseline that later sprints will compare against. | 16 hrs |
| 2 | Matrix/factor lifecycle inventory | Audit every API that mutates `SparseMatrix`, requires identity permutations, depends on `factored`, or assumes caller-managed copying before use. | 24 hrs |
| 3 | State-model taxonomy | Classify APIs into original-matrix consumers, analysis/factor builders, factor consumers, read-only queries, and mutation-heavy paths. | 16 hrs |
| 4 | Future handle-model design | Write the target design for explicit analysis/factor/solve handles and the migration strategy from the current hidden-state model. | 24 hrs |
| 5 | Quality-contract ownership map | Define what belongs in Makefile, scripts, CI, README, headers, and future maintainer docs so later sprints can reduce duplication safely. | 16 hrs |
| 6 | Public migration-risk audit | Identify the API surfaces most likely to need compatibility shims or deprecation layers once lifecycle refactoring begins. | 12 hrs |
| 7 | Sprint notes and validation anchor | Reconfirm the reviewed baseline and document exact validation commands to use after each refactor sprint. | 16 hrs |

### Deliverables

- Measured Epic 4 baseline
- Explicit matrix/factor lifecycle inventory
- Written future handle-model design and migration contract
- Ownership map for quality-policy and documentation surfaces
- Validation anchor for the rest of the epic

**Total estimate:** ~124 hours

---

## Sprint 41: Shared Allocation, Overflow & Internal Utility Consolidation

**Duration:** 14 days (~136 hours)

**Goal:** Centralize overflow checks, allocation sizing, and related internal safety helpers so the repository stops carrying multiple local safety idioms that drift over time.

### Prerequisites from previous Sprints

- Sprint 40 lifecycle inventory and architecture contract
- Existing local helper patterns in:
  - `src/sparse_etree.c`
  - `src/sparse_dense.c`
  - `src/sparse_svd.c`
  - `src/sparse_eigs.c`

### Items

| # | Item Name | Item Description | Estimate |
|---|-----------|------------------|----------|
| 1 | Internal utility design | Design a shared internal utility layer for size multiplication, `idx_t`/`size_t` bounds, and common safe-allocation helpers. | 16 hrs |
| 2 | Core helper implementation | Add the internal utility header/source and wire its initial tests or assertions. | 20 hrs |
| 3 | Migrate first core modules | Replace local helper copies in `sparse_dense.c`, `sparse_svd.c`, `sparse_eigs.c`, and `sparse_etree.c`. | 28 hrs |
| 4 | Broader `src/` migration pass | Audit and migrate the rest of `src/` where equivalent local size/overflow logic exists. | 28 hrs |
| 5 | Auxiliary-surface alignment | Audit examples, benchmarks, and scripts for the same safety patterns and align the highest-signal surfaces. | 18 hrs |
| 6 | Safety-style documentation | Add short internal guidance for when new code must use the shared helper layer rather than ad hoc local arithmetic. | 10 hrs |
| 7 | Validation | Run the full required code-quality gate and verify no behavioral regressions after helper consolidation. | 16 hrs |

### Deliverables

- Shared internal allocation/overflow utility layer
- Local helper duplication removed from the first core modules
- Broader `src/` safety logic standardized
- Initial examples/benchmarks/tooling alignment with the same pattern

**Total estimate:** ~136 hours

---

## Sprint 42: Matrix Lifecycle Refactor Phase 1 — Internal Handles & Compatibility Scaffolding

**Duration:** 14 days (~144 hours)

**Goal:** Start reducing the hidden mutable `SparseMatrix` lifecycle by introducing explicit internal handle boundaries and compatibility scaffolding without breaking the current public API.

### Prerequisites from previous Sprints

- Sprint 40 lifecycle taxonomy and handle-model design
- Sprint 41 shared internal utility layer

### Items

| # | Item Name | Item Description | Estimate |
|---|-----------|------------------|----------|
| 1 | Internal handle scaffolding | Introduce internal analysis/factor/workspace handle types and ownership rules, without yet exposing all of them publicly. | 24 hrs |
| 2 | Matrix-state guard helpers | Centralize “identity permutations”, “original state required”, and “factored-state required” validation logic behind shared helpers. | 20 hrs |
| 3 | Factor-path normalization | Refactor LU/Cholesky/LDL^T/QR/SVD entry paths to use the shared state-validation helpers instead of scattered bespoke checks. | 28 hrs |
| 4 | Cancellation-contract normalization | Reduce drift between headers, implementation, and README around cancellation semantics and in-place mutation boundaries. | 16 hrs |
| 5 | Compatibility wrapper plan | Design the shim layer that will let current one-shot APIs continue to work while later sprints add more explicit lifecycle objects. | 16 hrs |
| 6 | Focused tests | Add or tighten tests around state misuse, cancellation, and copy-before-use requirements before public lifecycle changes land. | 24 hrs |
| 7 | Validation | Run full code validation and verify the existing public APIs remain behavior-compatible. | 16 hrs |

### Deliverables

- Internal handle scaffolding for future explicit lifecycle APIs
- Shared state-validation helpers
- Normalized factor-path precondition checks
- Improved test coverage around lifecycle misuse and cancellation semantics

**Total estimate:** ~144 hours

---

## Sprint 43: Graph / ND Subsystem Decomposition Phase 1

**Duration:** 14 days (~152 hours)

**Goal:** Break `src/sparse_graph.c` into explicit subsystem slices so graph partitioning logic can evolve without one monolithic implementation unit carrying every heuristic and mode switch.

### Prerequisites from previous Sprints

- Sprint 40 architecture contract
- Sprint 41 shared safety helpers
- Sprint 42 compatibility scaffolding and state-guard refactor patterns

### Items

| # | Item Name | Item Description | Estimate |
|---|-----------|------------------|----------|
| 1 | Graph-module boundary design | Define the target split for graph construction, hierarchy building, coarsening, bisection, FM refinement, separator lifting, and runtime strategy parsing. | 20 hrs |
| 2 | Graph ownership / construction extraction | Move graph construction and ownership helpers into a dedicated module with stable interfaces. | 20 hrs |
| 3 | Hierarchy / coarsening extraction | Extract multilevel hierarchy building and heavy-edge matching into their own implementation units. | 28 hrs |
| 4 | Coarsest-bisection extraction | Separate brute-force / spectral / coarse-level bisection logic from the rest of the pipeline. | 24 hrs |
| 5 | Build-system and include cleanup | Update internal headers, file layout, and build scripts for the new graph subsystem shape. | 12 hrs |
| 6 | Focused graph tests | Add or adapt tests around extracted module seams so the structural refactor is pinned before FM and separator passes move. | 32 hrs |
| 7 | Validation | Run full code-quality validation plus graph-focused tests/benchmarks. | 16 hrs |

### Deliverables

- First decomposition of `src/sparse_graph.c`
- Stable boundaries for graph ownership, coarsening, and coarse bisection
- Tests covering the new subsystem seams

**Total estimate:** ~152 hours

---

## Sprint 44: Graph / ND Subsystem Decomposition Phase 2 & Large-Test Maintainability Batch

**Duration:** 14 days (~148 hours)

**Goal:** Finish the graph decomposition by separating FM refinement and separator lifting, and begin the largest test-file maintainability cleanup in the same structural style.

### Prerequisites from previous Sprints

- Sprint 43 graph subsystem phase 1
- Sprint 40/42 lifecycle and validation contracts

### Items

| # | Item Name | Item Description | Estimate |
|---|-----------|------------------|----------|
| 1 | FM refinement extraction | Move `graph_refine_fm(...)` and its direct support logic into a dedicated refinement module. | 32 hrs |
| 2 | Separator lifting extraction | Move edge-to-vertex separator logic and strategy selection into a dedicated module. | 24 hrs |
| 3 | Runtime strategy parsing cleanup | Isolate environment/config parsing from algorithm hot paths to reduce hidden behavior coupling. | 16 hrs |
| 4 | Final graph orchestration cleanup | Simplify the top-level partition orchestration path after the extractions land. | 16 hrs |
| 5 | Large-test helper audit | Audit the highest-volume test files (`test_chol_csc.c`, `test_svd.c`, `test_ldlt_csc.c`, `test_qr.c`) for true helper-extraction opportunities. | 16 hrs |
| 6 | First test-helper consolidation batch | Extract shared helpers/fixtures from the largest tests where the one-binary-per-test model still stays clear. | 28 hrs |
| 7 | Validation | Run the full code-quality gate plus graph and large-test focused reruns. | 16 hrs |

### Deliverables

- Graph FM and separator logic extracted from the monolithic file
- Cleaner orchestration path for ND partitioning
- First large-test maintainability batch landed

**Total estimate:** ~148 hours

---

## Sprint 45: Iterative Solver Workspace Reuse & Repeated-Solve Efficiency

**Duration:** 14 days (~140 hours)

**Goal:** Add reusable workspace support for iterative solvers so repeated solve workloads can avoid repeated heap allocation while preserving the current simple public APIs.

### Prerequisites from previous Sprints

- Sprint 41 shared allocation helpers
- Sprint 42 lifecycle scaffolding
- Existing iterative solver surface in `src/sparse_iterative.c`

### Items

| # | Item Name | Item Description | Estimate |
|---|-----------|------------------|----------|
| 1 | Iterative-workspace API design | Define reusable workspace objects for CG, GMRES, and shared iterative buffers, including ownership, sizing, and reuse rules. | 20 hrs |
| 2 | Shared iterative buffer layer | Introduce internal shared workspace-backed routines for the common iterative allocation patterns. | 24 hrs |
| 3 | CG/GMRES migration | Convert the primary iterative paths to use the reusable workspace-backed internals. | 28 hrs |
| 4 | Block iterative migration | Extend the workspace model to multi-RHS/block iterative paths. | 20 hrs |
| 5 | Compatibility wrappers | Keep the current one-shot public APIs as convenience wrappers over the new workspace model. | 12 hrs |
| 6 | Repeated-solve benchmarks | Add or update microbenchmarks showing allocator-churn reduction and repeated-solve improvements. | 20 hrs |
| 7 | Validation | Run full code-quality validation plus repeated-solve benchmark/test checks. | 16 hrs |

### Deliverables

- Reusable workspace support for iterative solvers
- One-shot APIs preserved through compatibility wrappers
- Measured repeated-solve efficiency improvements

**Total estimate:** ~140 hours

---

## Sprint 46: Eigensolver Workspace Reuse & Advanced Repeated-Run Efficiency

**Duration:** 14 days (~148 hours)

**Goal:** Add reusable workspace/state support for eigensolvers so repeated Lanczos / thick-restart / LOBPCG workloads do not repeatedly allocate the same large buffers.

### Prerequisites from previous Sprints

- Sprint 41 shared allocation helpers
- Sprint 45 iterative-workspace patterns
- Existing eigensolver internals in `src/sparse_eigs.c`

### Items

| # | Item Name | Item Description | Estimate |
|---|-----------|------------------|----------|
| 1 | Eigensolver-workspace design | Define reusable workspace/state objects for grow-m Lanczos, thick-restart Lanczos, and LOBPCG. | 20 hrs |
| 2 | Grow-m / thick-restart buffer reuse | Convert the Lanczos and thick-restart paths to use reusable buffer ownership instead of one-shot allocation bundles. | 32 hrs |
| 3 | LOBPCG buffer reuse | Introduce reusable state for the LOBPCG block workspaces and dependent dense intermediates. | 28 hrs |
| 4 | Compatibility wrappers | Preserve the current simple public eigensolver APIs as wrappers around workspace-capable internals. | 12 hrs |
| 5 | Repeated-run benchmark suite | Add repeated-run benchmarks for eigensolver workloads with stable `(n, k, restart/block)` profiles. | 24 hrs |
| 6 | Memory-behavior documentation | Document the new workspace reuse model and the expected memory/runtime tradeoffs. | 16 hrs |
| 7 | Validation | Run full code-quality validation and eigensolver-focused checks/benchmarks. | 16 hrs |

### Deliverables

- Reusable workspace/state support for eigensolvers
- Preserved public one-shot APIs
- Measured repeated-run memory/allocation improvements

**Total estimate:** ~148 hours

---

## Sprint 47: Benchmark CLI Modernization, Auxiliary Surface Safety & Example/Tooling Cleanup

**Duration:** 14 days (~136 hours)

**Goal:** Bring benchmark CLIs, examples, and auxiliary tooling up to the same usability and safety standard as the core library.

### Prerequisites from previous Sprints

- Sprint 41 safety helper consolidation
- Sprint 45/46 workspace refactors for repeated-run measurement
- Existing `bench_main` / `bench_eigs` comparison points

### Items

| # | Item Name | Item Description | Estimate |
|---|-----------|------------------|----------|
| 1 | Shared CLI parsing helpers | Create reusable parsing helpers for positive integers, bounded integers, finite doubles, and enum-like modes. | 16 hrs |
| 2 | `bench_main` parser modernization | Replace `atoi`-style parsing, tighten error reporting, and align the supported flags with current library capabilities. | 24 hrs |
| 3 | Reorder-mode parity cleanup | Bring `bench_main` and other benchmark tool surfaces into full parity with the intended reorder modes and emitted labels. | 16 hrs |
| 4 | Example safety audit | Audit examples for unchecked size arithmetic, weak parsing, and outdated helper patterns. | 16 hrs |
| 5 | Auxiliary tooling safety cleanup | Align scripts and benchmark/example support code with the shared safety conventions from Sprint 41. | 24 hrs |
| 6 | Benchmark/example docs refresh | Update benchmark and example docs to reflect the modernized CLI and safer patterns. | 12 hrs |
| 7 | Validation | Run full code-quality validation plus benchmark/example compile and CLI sanity checks. | 28 hrs |

### Deliverables

- Shared CLI parsing helpers
- `bench_main` and peer tools modernized
- Examples and auxiliary tooling aligned to the current safety standard
- Updated benchmark/example documentation

**Total estimate:** ~136 hours

---

## Sprint 48: Quality-Contract Simplification, README Reduction & Maintainer Guide

**Duration:** 14 days (~124 hours)

**Goal:** Reduce duplication across Makefile, scripts, CI, README, headers, and tutorial prose by moving maintainer policy into a clearer home and leaving README more user-facing.

### Prerequisites from previous Sprints

- Sprint 40 ownership map
- Sprint 42 lifecycle/cancellation normalization
- Sprint 47 benchmark/example/tooling cleanup

### Items

| # | Item Name | Item Description | Estimate |
|---|-----------|------------------|----------|
| 1 | Maintainer-guide design | Define the target maintainer-policy document(s) for warning authority, reviewed baseline use, dead-code meaning, lifecycle expectations, and designated-initializer norms. | 16 hrs |
| 2 | README reduction pass | Trim README so it remains a strong user/operator entry point without also carrying all maintainer policy. | 24 hrs |
| 3 | Maintainer-guide implementation | Add the new maintainer-facing guidance and move duplicated policy out of README where appropriate. | 24 hrs |
| 4 | Tutorial/header cross-reference pass | Reconcile headers, tutorial prose, README, and maintainer guidance so behavioral caveats are linked rather than repeated inconsistently. | 20 hrs |
| 5 | Quality-contract simplification | Tighten ownership between Makefile commands, script behavior, CI enforcement, and docs so future changes require fewer coordinated edits. | 24 hrs |
| 6 | Documentation sanity sweep | Re-read and tighten all affected docs for consistency after the redistribution. | 8 hrs |
| 7 | Validation | Run focused sanity checks on reviewed/dead-code command surfaces and doc references. | 8 hrs |

### Deliverables

- New maintainer guide / policy home
- Smaller, more user-facing README
- Clearer documentation ownership across README, tutorial, headers, and policy docs
- Reduced duplication in the quality contract

**Total estimate:** ~124 hours

---

## Sprint 49: Lifecycle API Exposure, Final Integration, Validation & Epic 4 Closeout

**Duration:** 14 days (~132 hours)

**Goal:** Land the public-facing lifecycle refinements after the structural groundwork is complete, then run the final Epic 4 integration audit and closeout.

### Prerequisites from previous Sprints

- Sprint 42 lifecycle scaffolding
- Sprint 43-44 graph decomposition
- Sprint 45-46 workspace reuse
- Sprint 47 auxiliary-surface cleanup
- Sprint 48 documentation/policy redistribution

### Items

| # | Item Name | Item Description | Estimate |
|---|-----------|------------------|----------|
| 1 | Public lifecycle API landing | Introduce the explicit lifecycle/handle improvements that are now safe to expose after the groundwork sprints, with compatibility wrappers where needed. | 28 hrs |
| 2 | Migration-path documentation | Document how existing callers can keep using the old one-shot style and when the new explicit lifecycle/workspace style is preferable. | 12 hrs |
| 3 | Cross-surface compatibility sweep | Verify that examples, benchmarks, tests, and docs all still agree on the final lifecycle/workspace model. | 20 hrs |
| 4 | Final residual review | Revisit every finding from `review-codex-2026-0521.md` and classify it as fixed, intentionally deferred, or accepted tradeoff. | 20 hrs |
| 5 | Full integration validation | Run the complete reviewed baseline and any additional targeted checks needed by the lifecycle/API landing. | 24 hrs |
| 6 | Epic 4 summary artifacts | Write the final sprint and epic summary artifacts, including the final post-remediation contract and residual risk set. | 16 hrs |
| 7 | Closeout and handoff | Prepare the final handoff/retrospective material and route any true residuals into future planning. | 12 hrs |

### Deliverables

- Public lifecycle refinements landed with compatibility path
- Final integrated examples/docs/tests around the new model
- Complete Epic 4 residual review
- Final validation and closeout package

**Total estimate:** ~132 hours

