# Remediation Plan

**Date:** 2026-05-22  
**Derived from:** `review-codex-2026-0521.md`

## Goal

Address the major structural issues identified in the Epic 4 review without destabilizing the validated quality baseline inherited from Epic 3.

## Step-by-Step Plan

1. Establish a tracked remediation baseline.
   - Record the current reviewed-quality, CMake parity, and dead-code baseline before changing architecture.
   - Capture the current hot files and allocation-heavy surfaces as reference metrics.
   - Treat this as the “before” snapshot for later regressions.

2. Define the target lifecycle model for `SparseMatrix` and factor objects.
   - Inventory every API that mutates matrix state, requires identity permutations, or depends on the internal `factored` flag.
   - Classify them into:
     - original-matrix consumers,
     - analysis/factor builders,
     - factor consumers,
     - and read-only query paths.
   - Write the desired handle model before changing code:
     - explicit analysis handle,
     - explicit factor handle,
     - and cleaner cancellation semantics.

3. Introduce shared internal allocation/overflow helpers.
   - Add one internal utility header/source for:
     - size-multiplication overflow checks,
     - `idx_t`/`size_t` bound checks,
     - and common safe-allocation helpers.
   - Migrate current local helper copies from:
     - `sparse_dense.c`
     - `sparse_svd.c`
     - `sparse_eigs.c`
     - `sparse_etree.c`
   - After the core migration, audit benchmarks/examples/tooling for the same pattern.

4. Refactor `src/sparse_graph.c` into subsystem slices.
   - Split the file into modules with explicit responsibilities:
     - graph construction / ownership
     - hierarchy coarsening
     - coarsest-level bisection
     - FM refinement
     - separator lifting
     - configuration / strategy parsing
   - Preserve behavior first; do not mix this step with algorithm changes.
   - Add focused tests around each extracted seam before further heuristic work.

5. Reduce the largest test-file ownership blobs.
   - Audit the biggest files first:
     - `tests/test_chol_csc.c`
     - `tests/test_svd.c`
     - `tests/test_ldlt_csc.c`
     - `tests/test_qr.c`
   - Extract shared helpers/fixtures where they are truly shared.
   - Split by feature slice only where the current one-binary-per-test model can still stay understandable.
   - Preserve test-surface truthfulness; do not reintroduce dormant scaffold.

6. Add reusable workspace APIs for iterative solvers.
   - Design a workspace object for:
     - CG
     - GMRES / generic iterative solve
     - block iterative paths
   - Keep the current public one-shot APIs as convenience wrappers around workspace-backed internals.
   - Measure repeated-solve workloads before and after to confirm the refactor improves allocator churn.

7. Add reusable workspace/state APIs for eigensolvers.
   - Identify the dominant reusable buffers in:
     - Lanczos grow-m
     - thick-restart Lanczos
     - LOBPCG
   - Introduce an opt-in workspace/context API for repeated runs with the same `(n, k, restart/block)` profile.
   - Preserve the current simpler API entry points as wrappers.

8. Modernize benchmark and developer CLI parsing.
   - Create shared parsing helpers for:
     - positive integers
     - bounded integers
     - finite doubles
     - enums / named modes
   - Migrate `bench_main` off `atoi`.
   - Bring `bench_main` into full reorder parity with the library surface, including `COLAMD` where applicable.
   - Align benchmark usage text, parser behavior, and enum coverage.

9. Simplify quality-contract ownership.
   - Decide which file owns which truth:
     - Makefile owns commands
     - scripts own behavior
     - CI files own matrix enforcement
     - maintainer guide owns policy
     - README links to those surfaces
   - Reduce duplicated command semantics in README where possible.
   - Keep operator-facing output concise and policy-facing wording centralized.

10. Split user-facing documentation from maintainer policy.
   - Keep `README.md` as a concise user/operator entry point.
   - Add a short Epic 4 maintainer-quality guide covering:
     - warning authority
     - reviewed baseline usage
     - dead-code workflow meaning
     - matrix-state / designated-initializer expectations
   - Remove or link repeated policy prose from README/tutorial where it becomes redundant.

11. Tighten examples and auxiliary tooling to the same safety standard as core library code.
   - Audit examples, benchmarks, and scripts for:
     - unchecked size arithmetic
     - weak parsing
     - duplicated helper logic
   - Standardize them on the same safe helpers and shell-safe conventions used in the maintained reviewed path.

12. Re-run the full validation matrix after each major batch.
   - For `*.c` / `*.h` changes, run:
     - `make format`
     - `make lint`
     - `make test`
   - Re-run:
     - `make quality-review-full`
     - serial `make deadcode-report`
     - serial `make deadcode-check`
   - Reconfirm:
     - reviewed CMake parity count
     - benchmark/example build surfaces
     - cross-platform contract wording

13. Land the lifecycle/API changes only after the structural groundwork is in place.
   - Do not start with public API churn.
   - First reduce file-size complexity, centralize helpers, and harden tests.
   - Then introduce explicit analysis/factor/workspace objects with a cleaner migration path.

14. End with a final Epic 4 closeout audit.
   - Verify that each review finding has either:
     - been fixed,
     - been intentionally deferred with a reason,
     - or been reclassified as acceptable design tradeoff.
   - Record the final post-remediation contract and residual risks.

## Exit Criteria

- The matrix/factor lifecycle is easier to reason about than the current hidden-state model.
- Overflow/allocation safety logic is centralized instead of duplicated across core modules.
- `src/sparse_graph.c` is no longer a single monolithic algorithm host.
- Repeated iterative/eigensolver workloads can reuse workspaces.
- Benchmark CLIs use robust parsing and expose the intended modes consistently.
- README is smaller and more user-facing, with maintainer policy moved to a clearer home.
- The full reviewed baseline still passes after the refactors.
