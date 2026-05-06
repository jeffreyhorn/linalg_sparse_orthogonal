# Project Plan: linalg_sparse_orthogonal -- Sprints 11-24 (Epic 2)

Based on findings from the Codex review (`reviews/review-codex-2026-04-06.md`) and Claude review (`reviews/review-claude-2026-04-06.md`).

---

## Sprint 11: Build System, Tolerance Standardization & Thread Safety Fixes

**Duration:** 14 days (~120 hours)

**Goal:** Fix the most urgent quality and correctness issues identified in both reviews before adding new features. Synchronize build systems, standardize numerical tolerances, and resolve thread safety issues.

### Prerequisites from previous Sprints

- Epic 1 baseline: existing CMake + Makefile build scripts, Cholesky / ILU factorizations, CSR / linked-list formats, README — items 1-7 audit and harden these in place rather than adding new numeric features.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | CMake test target sync | Add the 6 missing test targets to CMakeLists.txt (test_cholesky, test_csr, test_matmul, test_reorder, test_sprint4_integration, test_threads). Add a CI job that builds and tests with CMake to prevent future drift. | 8 hrs |
| 2 | Tolerance standardization | Replace all hardcoded absolute tolerances (1e-30 in Cholesky/ILU, 1e-300 in CSR solve) with norm-relative tolerances. Add internal `compute_rel_tol(matrix_norm, user_tol)` helper. Audit every factorization and solve path for consistency. Update documentation. | 32 hrs |
| 3 | Thread safety fixes | Fix `cached_norm` data race: use `_Atomic` on cached fields or document the limitation. Audit all shared mutable state. Update README thread safety table to accurately reflect guarantees. Add TSan CI job. | 20 hrs |
| 4 | Factored-state flag | Add `factored` field to SparseMatrix struct. Set in `sparse_lu_factor()` / `sparse_cholesky_factor()`, check in solve functions. Return `SPARSE_ERR_BADARG` for solve-before-factor. | 16 hrs |
| 5 | ILU/ILUT documentation | Document identity-permutation precondition in public headers. Document all undocumented preconditions across the API (QR requires unfactored matrix, etc.). | 8 hrs |
| 6 | Version macro generation | Generate `sparse_version_generated.h` from VERSION file at build time (CMake `configure_file`, Makefile `sed`). Remove hardcoded version macros from `sparse_types.h`. | 10 hrs |
| 7 | 32-bit index documentation | Document `idx_t` as int32_t in README known limitations. Add SIZE_MAX/INT32_MAX rationale to sparse_types.h. | 4 hrs |

### Deliverables

- CMake and Makefile build systems produce identical test coverage
- All solvers use consistent norm-relative tolerance strategy
- Thread safety claims in README match actual guarantees
- Solve-before-factor detected and rejected at runtime
- All public API preconditions documented in headers
- Version managed from single source with no manual sync
- 32-bit index limitation documented

**Total estimate:** ~98 hours

---

## Sprint 12: Sparse LDL^T Factorization (Symmetric Indefinite)

**Duration:** 14 days (~160 hours)

**Goal:** Add the highest-priority missing factorization: sparse LDL^T with Bunch-Kaufman symmetric pivoting for symmetric indefinite systems (KKT, saddle-point, constrained optimization).

### Prerequisites from previous Sprints

- Sprint 11 item 4: factored-state flag — `sparse_ldlt_factor` participates in the same precondition-checked solve gate.
- Sprint 11 item 2: norm-relative tolerance helper — Bunch-Kaufman growth control reuses the standardised tolerance machinery.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | LDL^T data structures | Define `sparse_ldlt_t` result struct (permutation, D diagonal/2x2 blocks, L unit lower triangular). Design public API in `sparse_ldlt.h`. | 12 hrs |
| 2 | Bunch-Kaufman pivoting | Implement symmetric pivoting strategy: 1x1 and 2x2 pivots with growth-factor control. Handle diagonal block bookkeeping. | 32 hrs |
| 3 | Sparse LDL^T elimination | Implement the elimination loop operating on linked-list format. Store L below diagonal, D as diagonal/2x2 blocks. Support fill-reducing reordering (AMD/RCM) as preprocessing. | 40 hrs |
| 4 | LDL^T solve | Forward substitution (L), diagonal solve (D, handling 2x2 blocks), backward substitution (L^T), and permutation application. | 20 hrs |
| 5 | LDL^T tests | Test on symmetric indefinite matrices: KKT systems, saddle-point matrices, matrices with negative eigenvalues. Compare with LU on same systems. SuiteSparse symmetric indefinite matrices. | 24 hrs |
| 6 | Integration and documentation | Add to README, update API overview table, add example usage. Integration tests with iterative refinement. | 16 hrs |

### Deliverables

- `sparse_ldlt_factor()` and `sparse_ldlt_solve()` in public API
- Bunch-Kaufman 1x1/2x2 symmetric pivoting
- AMD/RCM reordering support
- Tests on KKT and saddle-point systems
- Documented in README and headers

**Total estimate:** ~144 hours

---

## Sprint 13: Incomplete Cholesky Preconditioner & MINRES Solver

**Duration:** 14 days (~136 hours)

**Goal:** Add the two highest-value iterative solver extensions: incomplete Cholesky (IC(0)) preconditioning for SPD systems and MINRES for symmetric indefinite systems.

### Prerequisites from previous Sprints

- Sprint 11 item 2: tolerance standardization — IC(0) drop tolerance and MINRES residual checks both consume the relative-tolerance helper.
- Sprint 12: `sparse_ldlt_factor` — used as a preconditioner option in MINRES integration tests.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Incomplete Cholesky IC(0) | Implement no-fill incomplete Cholesky: L*L^T approximation preserving the sparsity pattern of A. Store as sparse_ilu_t (reuse L/U struct with L^T = U). | 28 hrs |
| 2 | IC(0) preconditioner callback | Implement `sparse_ic_precond()` compatible with `sparse_precond_fn`. Forward-backward substitution with L and L^T. | 8 hrs |
| 3 | MINRES solver | Implement Lanczos-based MINRES for symmetric (possibly indefinite) systems. Support preconditioning via callback. Return iterations, residual, convergence flag. | 40 hrs |
| 4 | Block MINRES | Multi-RHS MINRES variant with per-column convergence tracking, consistent with block CG/GMRES API pattern. | 20 hrs |
| 5 | IC(0) + MINRES integration | Test IC(0)-preconditioned CG on SPD systems (compare with ILU(0)). Test MINRES on symmetric indefinite systems with IC and LDL^T preconditioners. | 24 hrs |
| 6 | Documentation and benchmarks | Document IC(0) and MINRES in README, headers, INSTALL.md. Benchmark IC(0) vs ILU(0) on SPD SuiteSparse matrices. | 16 hrs |

### Deliverables

- `sparse_ic_factor()`, `sparse_ic_solve()`, `sparse_ic_precond()` in public API
- `sparse_solve_minres()` and `sparse_minres_solve_block()` in public API
- IC(0) tested as CG preconditioner on SPD systems
- MINRES tested on symmetric indefinite systems
- Benchmark comparison: IC(0) vs ILU(0) on SPD matrices

**Total estimate:** ~136 hours

---

## Sprint 14: Symbolic Analysis / Numeric Factorization Split

**Duration:** 14 days (~152 hours)

**Goal:** Separate symbolic analysis from numeric factorization for LU, Cholesky, and LDL^T. This enables repeated numeric refactorization on the same sparsity pattern without redoing ordering and symbolic work.

### Prerequisites from previous Sprints

- Sprint 11 item 4: factored-state flag — analysis objects track their own initialised state with the same convention.
- Sprint 12: `sparse_ldlt_factor` API surface — items 1-5 split it (and Cholesky / LU) into analyze + numeric phases.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Elimination tree computation | Compute the elimination tree (etree) from the sparsity pattern. Store parent pointers and postordering. This is the foundation for symbolic factorization. | 24 hrs |
| 2 | Symbolic factorization for Cholesky | Compute the nonzero structure of L from the etree without performing numeric work. Return column counts, row indices, and total nnz. | 24 hrs |
| 3 | Symbolic factorization for LU | Compute upper bounds on L and U sparsity structure using the etree and column dependency graph. Pre-allocate storage. | 28 hrs |
| 4 | Analysis object API | Design `sparse_analysis_t` struct holding etree, column counts, permutation, and symbolic factor structure. API: `sparse_analyze(A, opts, &analysis)` -> `sparse_factor_numeric(A, analysis, &factors)`. | 20 hrs |
| 5 | Numeric refactorization | Implement `sparse_refactor_numeric()` that reuses the symbolic structure from a prior analysis to factor a new matrix with the same sparsity pattern. Validate pattern match. | 24 hrs |
| 6 | Backward compatibility | Keep existing one-shot `sparse_lu_factor()` and `sparse_cholesky_factor()` APIs working by internally calling analyze + factor. No breaking changes. | 16 hrs |
| 7 | Tests and documentation | Test repeated refactorization (same pattern, different values). Verify memory usage is predictable. Benchmark symbolic-once vs. repeated full factorization. | 16 hrs |

### Deliverables

- `sparse_analyze()` computes etree and symbolic structure
- `sparse_factor_numeric()` performs numeric-only factorization
- `sparse_refactor_numeric()` reuses symbolic structure
- Existing one-shot APIs still work (backward compatible)
- Repeated factorization benchmark showing speedup

**Total estimate:** ~152 hours

---

## Sprint 15: COLAMD Ordering & QR Minimum-Norm Least Squares

**Duration:** 14 days (~140 hours)

**Goal:** Upgrade the ordering stack with COLAMD for unsymmetric/QR problems and add minimum-norm least-squares solves for underdetermined systems.

### Prerequisites from previous Sprints

- Sprint 11 item 2: tolerance standardization — rank-revealing thresholds in item 5 reuse the relative-tolerance helper.
- Sprint 14: `sparse_analyze` / `sparse_analysis_t` — COLAMD permutation feeds the analysis object the same way AMD / RCM already do.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | COLAMD ordering | Implement column approximate minimum degree ordering for unsymmetric matrices and QR factorization. Operate on the column adjacency graph (A^T*A pattern) without forming it explicitly. | 40 hrs |
| 2 | Integrate COLAMD with QR | Add `SPARSE_REORDER_COLAMD` to the reorder enum. Wire into `sparse_qr_factor_opts()` as the default column ordering for QR. Benchmark fill reduction vs current AMD proxy. | 16 hrs |
| 3 | QR minimum-norm solve | Implement minimum-norm least-squares for underdetermined systems (m < n): compute Q*R factorization of A^T, then solve via R^{-T} * Q^T * b. Return the minimum 2-norm solution. | 32 hrs |
| 4 | QR solve API update | Add `sparse_qr_solve_minnorm()` or a flag in the QR solve options to request minimum-norm. Update `sparse_qr_solve()` documentation to clearly distinguish basic vs minimum-norm solutions. | 12 hrs |
| 5 | Rank-revealing improvements | Improve rank detection diagnostics: expose the R diagonal, add rank-deficiency warnings, document threshold selection guidance. | 16 hrs |
| 6 | Tests and documentation | Test COLAMD on SuiteSparse unsymmetric matrices. Test minimum-norm on underdetermined systems with known solutions. Update README. | 24 hrs |

### Deliverables

- `sparse_reorder_colamd()` in public API
- COLAMD integrated as default QR column ordering
- `sparse_qr_solve_minnorm()` for underdetermined systems
- Rank-revealing diagnostics improved
- Fill reduction benchmarks: COLAMD vs AMD on unsymmetric matrices

**Total estimate:** ~140 hours

---

## Sprint 16: BiCGSTAB Solver & Iterative Solver Hardening

**Duration:** 14 days (~128 hours)

**Goal:** Add BiCGSTAB for nonsymmetric systems where restarted GMRES is a poor fit, and harden the iterative solver framework with better convergence diagnostics, stagnation detection, and breakdown handling.

### Prerequisites from previous Sprints

- Sprint 11 item 2: tolerance standardization — convergence checks across CG / GMRES / MINRES / BiCGSTAB share the relative-tolerance helper.
- Sprint 13 item 3: MINRES Lanczos kernel and `sparse_precond_fn` callback shape — BiCGSTAB inherits the same preconditioning contract.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | BiCGSTAB solver | Implement BiCG Stabilized method for general nonsymmetric systems. Support left preconditioning via callback. Return iterations, residual, convergence flag. | 32 hrs |
| 2 | Block BiCGSTAB | Multi-RHS variant with per-column convergence, consistent with block CG/GMRES pattern. | 16 hrs |
| 3 | Matrix-free BiCGSTAB | User-supplied matvec callback variant, following the CG/GMRES matrix-free pattern. | 8 hrs |
| 4 | Stagnation detection | Add stagnation detection to CG, GMRES, MINRES, and BiCGSTAB: if residual fails to decrease for N consecutive iterations, report `SPARSE_ERR_NOT_CONVERGED` early with a stagnation flag in the result struct. | 20 hrs |
| 5 | Convergence diagnostics | Add optional residual history recording (caller-provided array). Add verbose callback option for custom progress reporting. | 16 hrs |
| 6 | Breakdown handling | Audit all iterative solvers for graceful handling of breakdown (zero denominators in CG, lucky breakdown in GMRES, BiCGSTAB rho=0). Document breakdown behavior. | 20 hrs |
| 7 | Tests and documentation | Compare BiCGSTAB vs GMRES on nonsymmetric SuiteSparse matrices. Test stagnation detection. Update README iterative solver section. | 16 hrs |

### Deliverables

- `sparse_solve_bicgstab()`, block, and matrix-free variants
- Stagnation detection in all iterative solvers
- Optional residual history recording
- Breakdown handling audited and documented
- BiCGSTAB vs GMRES comparison benchmarks

**Total estimate:** ~128 hours

---

## Sprint 17: CSR/CSC Numeric Backend for Cholesky and LDL^T — **Complete**

**Duration:** 14 days (~152 hours)

**Goal:** Extend the CSR working-format strategy (proven in Sprint 10 for LU) to Cholesky and LDL^T, making compressed formats the primary numeric backend for all direct solvers while keeping the linked list as the mutable front end.

### Prerequisites from previous Sprints

- Sprint 10 (pre-Epic 2): CSR LU working-format strategy — items 1-2 generalise the same idea to Cholesky / LDL^T.
- Sprint 12: `sparse_ldlt_factor` linked-list kernel — item 3 mirrors it as a CSC variant.
- Sprint 14: `sparse_analyze` / `sparse_analysis_t` — items 1, 2, and 4 consume the elimination tree and column counts to drive the CSC layout.

### Items

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | CSC format for Cholesky | ✅ Complete | `CholCsc` struct + `chol_csc_from_sparse` / `chol_csc_from_sparse_with_analysis` / `chol_csc_to_sparse`. |
| 2 | CSC Cholesky elimination | ✅ Complete | Scatter-gather scalar kernel (`chol_csc_eliminate`) with fill-in handling + drop tolerance; solve via `chol_csc_solve` / `chol_csc_solve_perm`. |
| 3 | CSC LDL^T elimination | ⚠️ Wrapper only | `LdltCsc` storage + CSC solve are native; the Bunch-Kaufman factorization currently delegates to the linked-list kernel (Day 8 design decision — a native CSC BK kernel with symmetric swaps in packed storage is tracked as post-Sprint 17 follow-up). Day 9 solve is fully CSC. |
| 4 | Supernodal detection for Cholesky | ⚠️ Detection + dense primitives ship; batched integration deferred | `chol_csc_detect_supernodes` + `chol_dense_factor` / `chol_dense_solve_lower` land; `chol_csc_eliminate_supernodal` detects then delegates to scalar. The batched dense-kernel path over each supernode is follow-up work. |
| 5 | Benchmarks and validation | ✅ Complete | `bench_chol_csc.c` + `bench_ldlt_csc.c`; measured 1.65× (nos4 scalar) / 2.01× (nos4 supernodal) / 1.13× (bcsstk04 scalar) / 1.22× (bcsstk04 supernodal) one-shot CSC Cholesky speedup with AMD inside the timed region on both paths. Residuals match linked-list to 1e-15. |
| 6 | Documentation | ✅ Complete | README + `docs/algorithm.md` + `PERF_NOTES.md`; file-level design comments in both .c files; cross-links from `sparse_lu_csr.h`. |

### Deliverables (status)

- ⚠️ CSC Cholesky factorization — one-shot speedup **1.13–2.01×** (target was ≥ 2×: met on nos4 supernodal, below target on bcsstk04 once AMD is included on both paths).  Analyze-once / factor-many workflow (Sprint 14 split) is expected to exceed 2× because AMD cost amortizes away, but that benchmark has not yet been run — follow-up work.
- ⚠️ CSC LDL^T factorization — storage + solve native; factor delegated.
- ⚠️ Supernodal detection + dense primitives ship; batched dense-kernel integration deferred.
- ✅ Benchmark results on SPD + symmetric indefinite matrices.

### Follow-up items feeding the next sprint

- **Native CSC LDL^T Bunch-Kaufman kernel** with in-place symmetric swaps and element-growth tracking.
- **Batched supernodal Cholesky factor** that uses `chol_dense_factor` + `chol_dense_solve_lower` per supernode's diagonal block + panel.
- **Transparent size-based dispatch** through `sparse_cholesky_factor_opts` (requires CSC→linked-list writeback).
- **Larger SuiteSparse corpus** (n ≥ 1000) to exercise the supernodal batched path and measure scaling.

**Total estimate:** ~152 hours; actual scope realised: items 1, 2, 5, 6 fully; items 3, 4 partially (storage/API/tests complete, batched kernels deferred).

---

## Sprint 18: CSC Kernel Performance Follow-Ups — **Complete**

**Duration:** 14 days (~124 hours)

**Goal:** Complete the Sprint 17 CSC numeric-backend work by replacing the linked-list delegations with native CSC kernels, exposing a transparent size-based dispatch through the public API, and validating scaling behavior on larger SuiteSparse problems.

### Prerequisites from previous Sprints

- Sprint 17: CSC Cholesky / LDL^T scaffolding (`CholCsc` / `LdltCsc`, scalar kernels, supernodal detection + dense primitives) — items 1-2 replace the linked-list delegations with native CSC kernels.
- Sprint 14: `sparse_analyze` / `sparse_analysis_t` — items 2-3 use the symbolic phase to pre-allocate the full `sym_L` pattern.

### Items

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Native CSC LDL^T Bunch-Kaufman kernel | ✅ Complete | `ldlt_csc_eliminate_native` runs Bunch-Kaufman directly on packed CSC storage: in-place `ldlt_csc_symmetric_swap`, four-criteria 1×1/2×2 pivot selection, element-growth tracking. Matches linked-list output bit-for-bit on the existing `test_ldlt_csc` matrices (20-matrix random cross-check). Factor speedup 2.2–2.5× on bcsstk14 / s3rmt3m3 (`bench_ldlt_csc`). |
| 2 | Batched supernodal Cholesky factor | ✅ Complete | `chol_csc_supernode_extract` + `_eliminate_diag` + `_eliminate_panel` + `_writeback` pipeline wired into `chol_csc_eliminate_supernodal`. Day 12 uncovered and fixed a latent bug where `chol_csc_from_sparse*` only pre-populated A's lower-triangle pattern; the fix materialises the full `analysis->sym_L` pattern up front. Residuals match scalar CSC to round-off on the enlarged corpus. |
| 3 | Transparent `sparse_cholesky_factor_opts` dispatch | ✅ Complete | `sparse_cholesky_factor_opts` routes to the CSC supernodal path when `n >= SPARSE_CSC_THRESHOLD` via `sparse_analyze` → `chol_csc_from_sparse_with_analysis` → `chol_csc_eliminate_supernodal` → `chol_csc_writeback_to_sparse`. `sparse_cholesky_opts_t::backend` (AUTO / LINKED_LIST / CSC) forces a path; `used_csc_path` reports the chosen branch. |
| 4 | Larger SuiteSparse corpus & scaling benchmarks | ✅ Complete | Default corpora now include bcsstk14, s3rmt3m3, Kuu, Pres_Poisson (Cholesky) and bcsstk14, s3rmt3m3 (LDL^T). bloweybq is singular on every backend and tuma1 is too slow for the LDL^T default; both available via single-matrix CLI. Captured in `docs/planning/EPIC_2/SPRINT_18/bench_day12.txt` and extended tables in `PERF_NOTES.md`. |
| 5 | Integration tests & documentation | ✅ Complete | `tests/test_sprint18_integration.c` (10 tests, 234 assertions) covers cross-threshold dispatch + forced-path agreement + native-vs-wrapper LDL^T. File headers in `src/sparse_chol_csc.c`, `src/sparse_ldlt_csc.c`, and `include/sparse_cholesky.h` describe native kernels + transparent dispatch (no "wrapper / delegation / deferred" language remains). `docs/algorithm.md` and `README.md` refreshed with the Day 12 numbers. |

### Deliverables (status)

- ✅ Native CSC Bunch-Kaufman LDL^T kernel replacing the Sprint 17 Day 8 wrapper (bit-identical output on existing tests)
- ✅ Batched supernodal Cholesky using `chol_dense_factor` / `chol_dense_solve_lower` per supernode
- ✅ Transparent threshold-based dispatch through `sparse_cholesky_factor_opts` with lossless CSC→linked-list writeback
- ✅ Expanded SuiteSparse benchmark corpus (n ≥ 1000) with updated `PERF_NOTES.md` scaling tables
- ✅ Documentation updates removing wrapper/delegation language

### Deferred to Sprint 19

- **Analyze-once, factor-many benchmark.** The Sprint 17 PERF_NOTES
  hypothesis that the CSC speedup is larger in the repeated-refactor
  workflow (AMD amortises to zero) is still unmeasured — Day 12
  focused on enlarging the one-shot corpus and finding/fixing the
  sym_L pre-population bug.  Filed for Sprint 19 as a dedicated
  `bench_refactor_csc` add-on.
- **Characterising n ∈ [20, 100] for `SPARSE_CSC_THRESHOLD`.** The
  Day 12 plan forbids moving the threshold without measurement, and
  the enlarged corpus has no matrix below n = 100.  The default
  stays at 100 pending a small-matrix study in Sprint 19.
- **Kuu's scalar-CSC regression (0.77×).** The scalar kernel's
  `shift_columns_right_of` packing cost during drop-tolerance pruning
  is a real loss on this matrix family; the supernodal path side-
  steps it via full `sym_L` pre-allocation.  Whether the scalar
  kernel should also pre-allocate is a Sprint 19 design call.

**Total estimate:** ~124 hours; actual scope realised: all five items complete, three follow-ups explicitly deferred to Sprint 19.

---

## Sprint 19: CSC Kernel Tuning & Native Supernodal LDL^T — **Complete**

**Duration:** 14 days (~168 hours estimated; ~140 hours actual)

**Goal:** Close out the Sprint 18 CSC follow-ups surfaced in `SPRINT_18/RETROSPECTIVE.md`: quantify the analyze-once / factor-many speedup the Sprint 17 + Sprint 18 PERF_NOTES hypothesise, characterise `SPARSE_CSC_THRESHOLD`'s crossover with sub-100 fixtures, fix the scalar-CSC regression on Kuu-like fill patterns, extend the Sprint 18 batched supernodal path from Cholesky to symmetric indefinite LDL^T, and restore the LDL^T scalar kernel's sparse-row scaling by adding a row-adjacency index.

### Prerequisites from previous Sprints

- Sprint 17 / 18: CSC Cholesky and LDL^T (storage, scalar kernels, supernodal Cholesky, transparent dispatch) — item 1 measures the analyze-once / factor-many path through the Sprint 18 dispatch; items 4-5 extend the supernodal path to LDL^T.
- Sprint 14: `sparse_analyze` — item 1's `bench_refactor_csc` driver reuses the analysis across many numeric refactors.
- Sprint 18: `SPARSE_CSC_THRESHOLD` (default 100) — item 2 characterises crossover at n < 100 and confirms the threshold.

### Items

| # | Item | Status | Outcome |
|---|------|--------|---------|
| 1 | `bench_refactor_csc` — analyze-once / factor-many workflow | ✅ Complete | `benchmarks/bench_refactor_csc.c` ships; corpus speedups range from 0.93× (nos4, small) to 16.77× (Pres_Poisson) — confirms the Sprint 17/18 hypothesis that AMD amortisation widens the CSC win on the analyze-once path. Day 1-2 captures live in `bench_day2_refactor.txt`; PERF_NOTES.md extended with the new "Analyze-once / factor-many" section. |
| 2 | Small-matrix corpus + `SPARSE_CSC_THRESHOLD` retrospective | ✅ Complete | 10 synthetic fixtures (tridiag/banded/dense at n ∈ {20, 40, 60, 80}) added to `bench_chol_csc --small-corpus`. Crossover analysis confirmed `SPARSE_CSC_THRESHOLD = 100` is conservative across families (data in PERF_NOTES.md "Threshold guidance" section). Doc comment in `include/sparse_matrix.h` refreshed with measured data. |
| 3 | Scalar-CSC `shift_columns_right_of` regression on Kuu | ✅ Complete | Day 5 `sample` profile confirmed 60% of factor time was `_platform_memmove` from `shift_columns_right_of`. Day 6 fix: `chol_csc_from_sparse_with_analysis` pre-allocates full `sym_L`; `chol_csc_gather` gains a `sym_L_preallocated`-gated fast path that writes in place into existing slots and zeroes drops. Kuu scalar CSC went from 0.77× to 2.43× over linked-list. Day 7 follow-up added the `sym_L_preallocated` flag to skip the merge-walk overhead on small matrices, restoring nos4 to 1.29×. |
| 4 | Native supernodal LDL^T batched kernel | ✅ Complete (SPD path) / ⚠️ Indefinite path scoped down | Days 10-13 shipped `ldlt_csc_detect_supernodes` (2×2-aware), `ldlt_dense_factor` (BK on dense column-major), `ldlt_csc_supernode_extract` / `_writeback` / `_eliminate_diag` / `_eliminate_panel`, and `ldlt_csc_eliminate_supernodal` (interleaved batched + scalar). Batched supernodal LDL^T speedups vs linked-list: bcsstk14 6.83×, bcsstk04 3.05×, nos4 2.62×. Indefinite matrices (KKT-style saddle points) require fallback to scalar `ldlt_csc_eliminate` because heuristic CSC fill from `ldlt_csc_from_sparse` doesn't always cover supernodal cmod fill — same root cause as the Cholesky pre-Sprint-19-Day-6 Kuu regression, deferred to Sprint 20 as `ldlt_csc_from_sparse_with_analysis`. |
| 5 | LDL^T scalar-kernel row-adjacency index | ✅ Complete | Days 8-9 added `row_adj` / `row_adj_count` / `row_adj_cap` to `LdltCsc` (per-row dynamic arrays with geometric 2× growth). `ldlt_csc_cmod_unified` Phase A and Phase B iterate `F->row_adj[col]` instead of `[0, step_k)`. `ldlt_csc_symmetric_swap` propagates swaps into `row_adj` (slot swap). Bit-identical factor output on every existing test; bcsstk14 jumped from ~2.5× to 3.51× on the native scalar bench. |

### Deliverables

- `benchmarks/bench_refactor_csc.c` with analyze-once / factor-many numbers captured in `PERF_NOTES.md` and `bench_day14.txt`
- Small-matrix (n ∈ [20, 100]) corpus measurements; `SPARSE_CSC_THRESHOLD` confirmed at 100 with supporting data documented in `PERF_NOTES.md`
- Scalar-CSC regression on Kuu resolved; scalar kernel restored to a monotonic n-vs-speedup trend across the full Sprint 18 corpus (Kuu 0.77× → 2.43×)
- Native supernodal LDL^T path (detection + dense LDL^T primitive + extract / eliminate_diag / eliminate_panel / writeback) shipping on the SPD path with bcsstk14 6.83× speedup; indefinite path requires `_with_analysis` follow-up (deferred to Sprint 20)
- `LdltCsc` row-adjacency index with `ldlt_csc_cmod_unified` traversing only the contributing prior columns (sparse-row scaling equivalent to the linked-list reference); bit-identical factor output and 3.51× speedup on bcsstk14

### Deferred to Sprint 20

- **`ldlt_csc_from_sparse_with_analysis` mirror.** The batched supernodal LDL^T path needs full `sym_L` pre-allocation (matching the Cholesky `_with_analysis` shim) so its writeback can preserve indefinite cmod fill rows.  Without it, KKT-style saddle points and other matrices with non-trivial off-block structure fall back to the scalar kernel.

**Total estimate:** ~168 hours; actual scope realised: all five items shipped, with one deferred follow-up explicitly scoped for Sprint 20.

---

## Sprint 20: LDL^T Completion & Symmetric Lanczos Eigensolver — **Complete**

**Duration:** 14 days (~136 hours estimated; ~125 hours actual, ~8% under)

**Goal:** Close out the final Sprint 19 follow-ups for the batched supernodal LDL^T path — the `_with_analysis` shim that enables indefinite matrices and a transparent size-based dispatch through `sparse_ldlt_factor_opts` — then land the symmetric Lanczos eigensolver with shift-invert mode.

### Prerequisites from previous Sprints

- Sprint 14: `sparse_analyze()` / `sparse_analysis_t` symbolic analysis API — consumed by the new `_with_analysis` shim.
- Sprint 18: transparent `sparse_cholesky_factor_opts` dispatch (AUTO / LINKED_LIST / CSC backend selector, CSC→linked-list writeback) — the template for the LDL^T dispatch in item 2.
- Sprint 19: batched supernodal LDL^T path (`ldlt_csc_detect_supernodes`, `ldlt_dense_factor`, `ldlt_csc_supernode_extract` / `_writeback` / `_eliminate_diag` / `_eliminate_panel`, `ldlt_csc_eliminate_supernodal`) — item 1 unblocks its indefinite coverage, item 2 exposes it through the public API.
- Sprint 12: `sparse_ldlt_factor_opts` public API — item 2 extends it with a backend selector.

### Items

| # | Item | Description | Estimate | Status |
|---|------|-------------|----------|--------|
| 1 | `ldlt_csc_from_sparse_with_analysis` | Mirror the Cholesky `_with_analysis` shim for LDL^T: pre-allocate the full `sym_L` pattern from `sparse_analysis_t` so the batched supernodal LDL^T writeback can preserve indefinite `cmod` fill rows that the heuristic `ldlt_csc_from_sparse` pattern does not cover. Unblocks the batched supernodal LDL^T path on KKT-style saddle points and other matrices with non-trivial off-block structure (currently forced to the scalar `ldlt_csc_eliminate` fallback). Same lesson as the Cholesky Sprint 19 Day 6 `sym_L` pre-allocation fix. | 24 hrs | ✅ Days 1-3 |
| 2 | Transparent LDL^T dispatch through `sparse_ldlt_factor_opts` | Extend `sparse_ldlt_factor_opts` with a `backend` selector (AUTO / LINKED_LIST / CSC) and a `used_csc_path` result field, mirroring the Cholesky dispatch added in Sprint 18. AUTO routes to the CSC supernodal path when `n >= SPARSE_CSC_THRESHOLD` via `sparse_analyze` → `ldlt_csc_from_sparse_with_analysis` → `ldlt_csc_eliminate_supernodal` → CSC→linked-list writeback, with a structure-based fallback to `ldlt_csc_eliminate` when the indefinite pattern defeats batching. Depends on item 1. | 20 hrs | ✅ Days 4-6 |
| 3 | Eigenvalue API design | Define `sparse_eigs_t` result struct (eigenvalues, eigenvectors, convergence info). API: `sparse_eigs_sym(A, k, which, opts, &result)` with `which` = largest/smallest/nearest_sigma. | 12 hrs | ✅ Day 7 |
| 4 | Symmetric Lanczos eigensolver | Implement thick-restart Lanczos for computing k largest/smallest eigenvalues and eigenvectors of symmetric matrices. Reorthogonalization for numerical stability. | 40 hrs | ✅ Days 8-11 |
| 5 | Shift-invert Lanczos | Add shift-invert mode for interior eigenvalues: solve (A - sigma*I)^{-1} * x using LU or LDL^T factorization inside the Lanczos iteration. Benefits directly from item 2's transparent LDL^T dispatch for symmetric indefinite shifts. | 20 hrs | ✅ Day 12 |
| 6 | Lanczos tests and validation | Test on diagonal matrices (exact eigenvalues known), tridiagonal matrices, SuiteSparse SPD matrices. Validate against existing SVD (eigenvalues of A^T*A = singular values squared). Include indefinite matrix coverage for the new LDL^T `_with_analysis` batched path. | 12 hrs | ✅ Day 13 |
| 7 | Lanczos documentation | Document Lanczos + shift-invert eigenvalue API in README, add example program, update API overview table. LOBPCG documentation follows in Sprint 21 when that item lands. | 8 hrs | ✅ Day 14 |

### Deliverables

- ✅ `ldlt_csc_from_sparse_with_analysis` shim with full `sym_L` pre-allocation; batched supernodal LDL^T working end-to-end on indefinite SuiteSparse matrices
- ✅ Transparent size-based dispatch through `sparse_ldlt_factor_opts` with `used_csc_path` reporting and a structure-aware fallback
- ✅ `sparse_eigs_sym()` with thick-restart Lanczos + full MGS reorth backend for k largest/smallest eigenvalues, Ritz pair extraction, Wu/Simon per-pair residual convergence
- ✅ Shift-invert mode for interior eigenvalues, composing with the Day 4-6 LDL^T AUTO dispatch; `used_csc_path_ldlt` observability on the result struct
- ✅ Lanczos documented in README, algorithm.md, public header doxygen; `examples/example_eigs.c` demonstrates both LARGEST and NEAREST_SIGMA modes with eigenvector residual checks
- ✅ SuiteSparse / SVD cross-check / indefinite / stability test coverage in `tests/test_eigs.c` (19 tests / 154 assertions)

**Total estimate:** ~136 hours (actual ~125 hours, ~8% under — see Sprint 20 retrospective)

---

## Sprint 21: Eigensolver Completion — Thick-Restart, OpenMP & LOBPCG — **Complete**

**Duration:** 14 days (~124 hours estimated; actual ~133 hours per the SPRINT_21/PLAN.md day budgets)

**Goal:** Close out the symmetric eigensolver family started in Sprint 20: land true Wu/Simon thick-restart (replacing the provisional growing-m outer loop so memory is bounded on large-n problems), parallelise the Lanczos reorthogonalization inner loop under OpenMP, add LOBPCG for preconditioned block eigenvalue computation, and ship a permanent `bench_eigs` executable with CSV output.

### Prerequisites from previous Sprints

- Sprint 13: IC(0) factorization and `sparse_precond_fn` callback — LOBPCG's preconditioning path reuses this infrastructure.
- Sprint 17 / 18: `SPARSE_OPENMP` build option already driving `sparse_matvec` — item 2 extends OpenMP coverage into the Lanczos reorthogonalization loop.
- Sprint 20: `sparse_eigs_t`, `sparse_eigs_opts_t`, `sparse_eigs_sym()` API surface; growing-m Lanczos outer loop; full MGS reorthogonalization; Wu/Simon residual gate; shift-invert path through `sparse_ldlt_factor_opts` — items 1 and 2 rework the Sprint 20 Lanczos core; item 3 plugs LOBPCG into the same result struct via the reserved `SPARSE_EIGS_BACKEND_LOBPCG` enum value.

### Items

| # | Item | Status | Description |
|---|------|--------|-------------|
| 1 | True thick-restart Lanczos (Wu/Simon arrowhead) | ✅ Days 1-4, 12 | `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART` ships with the arrowhead state (`lanczos_restart_state_t`), Givens-style reduction (`s21_arrowhead_to_tridiag`), and the outer loop (`s21_thick_restart_outer_loop`). Memory bound verified: bcsstk14 (n = 1806, k = 5) drops peak `V` from ~7 MB (grow-m) to ~565 KB (thick-restart). Wu/Simon monotonicity verified at the public-API level via the Day 12 `test_thick_restart_locked_progress_monotone` two-budget end-to-end test. |
| 2 | Lanczos OpenMP parallelism | ✅ Days 5-6 | The shared `s21_mgs_reorth` kernel parallelises the inner-product / daxpy bodies via `#pragma omp parallel for reduction(+:dot)` and `#pragma omp parallel for`, gated on `n ≥ SPARSE_EIGS_OMP_REORTH_MIN_N` (default 500). TSan clean via `make sanitize-thread`. ~2× speedup at 4 threads on bcsstk14 m=70 — matches the PROJECT_PLAN target. |
| 3 | LOBPCG solver | ✅ Days 7-10, 13 | `SPARSE_EIGS_BACKEND_LOBPCG` ships with block Rayleigh-Ritz over `[X | W | P]`, BLOPEX-style conditioning guard, per-column soft-locking, and preconditioning via `opts->precond` (IC(0) and LDL^T validated). bcsstk04 SMALLEST k=3 cond ≈ 5e6: vanilla 800-iter NOT_CONVERGED → IC(0) 62 iters → LDL^T 8 iters. AUTO routes here on `n ≥ SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD` (1000) when a preconditioner is supplied. |
| 4 | Permanent `benchmarks/bench_eigs.c` | ✅ Day 11 | Driver lives at `benchmarks/bench_eigs.c` with `--sweep default`, `--compare`, `--matrix <path>` modes; CSV via `--csv`; CSV schema documented in `benchmarks/README.md`. `make bench-eigs` runs the default sweep at `--repeats 3` in ~20 sec on the developer machine. |
| 5 | Eigensolver tests, documentation & benchmark captures | ✅ Days 12-14 | `tests/test_eigs_thick_restart.c` 20 tests / 267 assertions; `tests/test_eigs_lobpcg.c` 26 tests / 219 assertions. README + `docs/algorithm.md` updated with thick-restart arrowhead, LOBPCG Rayleigh-Ritz, AUTO decision tree, and OpenMP MGS strategy subsections. Captures committed at `docs/planning/EPIC_2/SPRINT_21/bench_day14.txt` (full sweep, 33 rows × 12 cols) and `bench_day14_compare.txt` (3-backend × 3-precond pivot). |

### Deliverables (status)

- ✅ True Wu/Simon thick-restart backend driving `sparse_eigs_sym` with bounded O((k + m_restart) · n) memory
- ✅ OpenMP-parallel Lanczos reorthogonalization validated under TSan
- ✅ `sparse_eigs_sym` with `SPARSE_EIGS_BACKEND_LOBPCG` supporting preconditioned block eigenvalue computation
- ✅ Permanent `benchmarks/bench_eigs.c` with CSV + `--sweep` + `--compare` modes
- ✅ Tests for thick-restart and LOBPCG on the SuiteSparse corpus; updated README and `docs/algorithm.md`; committed benchmark captures comparing growing-m, thick-restart, and LOBPCG

### Deferred to a future sprint

- LARGEST-via-op-negation adapter for LOBPCG. The PLAN's Day 10 Task 1 mentioned wrapping `op` into `neg_op` to compose preconditioning with LARGEST modes (preconditioning naturally targets SMALLEST). The shipped LOBPCG selects LARGEST directly from the Jacobi eigendecomposition without negation — works but doesn't compose with `M^{-1} ≈ A^{-1}` preconditioning on the LARGEST end. Visible in `bench_day14_compare.txt`'s nos4 LARGEST + IC0/LDLT rows (NOT_CONVERGED). Candidate for a follow-up sprint when a workload demands LARGEST + precond.
- Block-structured BLOPEX P-update with pre-orthogonalised W/P partitions. Day 9's first attempt at the block-preserving form introduced a numerical drift that broke convergence on small fixtures; reverted to the Day 8 orthogonal-projection formula (mathematically equivalent in exact arithmetic when X is orthonormal). The PLAN's "more robust formulation" is implemented as a conditioning guard (`P_new = 0` on near-singular Gram) rather than the full block-structured form.

**Total estimate:** ~124 hours; actual ~133 hours per the day budgets in `SPRINT_21/PLAN.md` (within the 14×12 = 168-hour ceiling). See `docs/planning/EPIC_2/SPRINT_21/RETROSPECTIVE.md` for metrics, lessons, and DoD verification.

---

## Sprint 22: Ordering Upgrades — Nested Dissection & Quotient-Graph AMD — **Complete**

**Duration:** 14 days (~124 hours estimated; actual ~134 hours per the SPRINT_22/PLAN.md day budgets)

**Goal:** Upgrade the ordering stack with nested dissection for large 2D/3D PDE meshes and replace the bitset-based AMD with a quotient-graph implementation for O(nnz) memory, removing the current scaling bottleneck on large matrices.

### Prerequisites from previous Sprints

- Sprint 11 and earlier: existing bitset AMD, RCM, and reorder enum `sparse_reorder_t` — item 4 replaces the bitset AMD wholesale through the same public enum.
- Sprint 14: `sparse_analysis_t` symbolic analysis — nested dissection hands its permutation off to `sparse_analyze` the same way AMD / COLAMD / RCM do today.
- Sprint 15: `SPARSE_REORDER_COLAMD` enum wiring — item 3 follows the same pattern to add `SPARSE_REORDER_ND`.

### Items

| # | Item | Status | Description |
|---|------|--------|-------------|
| 1 | Graph partitioning | ✅ Days 1-5 | `src/sparse_graph.c` ships `sparse_graph_t`, heavy-edge-matching coarsening hierarchy, brute-force / GGGP coarsest bisection, single-pass FM refinement with rollback-on-regress, uncoarsening + smaller-side vertex-separator extraction.  `sparse_graph_partition` produces a 3-way `{0, 1, 2}` partition; deterministic on seed.  Validated on the SuiteSparse smoke (bcsstk14 28 ms, Pres_Poisson 1.3 s — Day 5 capture). |
| 2 | Nested dissection ordering | ✅ Days 6-7 | `src/sparse_reorder_nd.c` recursive `nd_recurse` + public `sparse_reorder_nd`.  Day 6 `ND_BASE_THRESHOLD = 4` (provisional); Day 9 retuned to 32 from the bench sweep.  10×10 grid lands at 1.22× AMD's nnz(L); Pres_Poisson 1.06× of AMD — short of the plan's 0.5× target.  See `### Deferred` below. |
| 3 | Add SPARSE_REORDER_ND to enum | ✅ Day 8 | `SPARSE_REORDER_ND = 4` added to `include/sparse_types.h`.  Dispatch wired through `sparse_analyze` + per-factorization `*_factor_opts` in `cholesky` / `ldlt` / `lu` / `qr`.  Public-header doxygen + README updated.  Cholesky / LU / LDL^T residuals on bcsstk14 / nos4 / bcsstk04 verified ≤ 1e-8 via the enum dispatch (`tests/test_reorder_nd.c`). |
| 4 | Quotient-graph AMD | ✅ Days 10-12 | `src/sparse_reorder_amd_qg.c` ships the simplified Davis-style quotient-graph minimum-degree (single workspace, sorted-merge, on-demand compaction + realloc).  Day 12 deleted the bitset implementation entirely (Option A).  Bit-identical fill across the corpus.  10 000×10 000 banded stress-test factor in 0.24 s (plan target ≤ 5 s). |
| 5 | Tests and benchmarks | ✅ Days 9, 13, 14 | `benchmarks/bench_reorder.c` Day 9 cross-ordering + threshold sweep capture; `benchmarks/bench_amd_qg.c` Day 13 AMD bitset-vs-qg bench (≥ 17× memory reduction at n = 20 000, analytic ~26× at n = 50 000); Day 14 final cross-ordering re-run.  All captures in `docs/planning/EPIC_2/SPRINT_22/`. |

### Deferred to a future sprint

- ND fill ratio on Pres_Poisson currently 1.06× of AMD; plan target was 0.5× (≥ 2× reduction over AMD).  Two contributing axes both deferred to Sprint 23: (a) ND's recursion leaves emit subgraph-local order rather than calling `sparse_reorder_amd_qg` per leaf — should drop 5-15 % on regular grids; (b) the simplified quotient-graph AMD doesn't have supervariable detection / element absorption / approximate-degree updates from the SuiteSparse reference, which is most of what closes the AMD-vs-METIS-AMD gap on PDE meshes.
- ND wall time on Pres_Poisson is ~5× AMD's (~38 s before Day 9 retune; ~24 s after), driven by the naïve O(n) max-gain scan in `graph_refine_fm`.  METIS uses an O(1) gain-bucket structure; porting that is a Sprint 23 perf item.
- The Sprint 22 quotient-graph AMD wins on n ≥ 5 000 banded fixtures (4-7× wall-time speedup over the deleted bitset) but loses ~30 % on n ≤ 1 800 SPD corpus matrices — the bitset's bit-twiddling beats sorted-merge intermediates on small inputs.  Closing this would require porting Davis's full algorithm; see Day 13's "what's left for Sprint 23" note in `PERF_NOTES.md`.

### Deliverables (status)

- ✅ `sparse_reorder_nd()` nested dissection ordering exposed through `SPARSE_REORDER_ND` (Days 6, 8)
- ✅ Quotient-graph AMD replacing bitset AMD; production swap with the bitset deleted (Day 12)
- ✅ Ordering benchmarks on large matrices in `bench_day9_nd.{csv,txt}`, `bench_day13_amd_qg.{csv,txt}`, `bench_day14.{csv,txt}`; fill-in comparison across AMD / RCM / COLAMD / ND in all three captures + `PERF_NOTES.md`

**Total estimate:** ~124 hours; actual ~134 hours per the day budgets in `SPRINT_22/PLAN.md` (within the 14×12 = 168-hour ceiling).  See `docs/planning/EPIC_2/SPRINT_22/RETROSPECTIVE.md` for metrics, lessons, and DoD verification.

---

## Sprint 23: Ordering Quality Follow-Ups (Sprint 22 deferrals)

**Duration:** 14 days (~88 hours estimated)

**Goal:** Close the two quality gaps Sprint 22 deferred — ND's fill ratio on Pres_Poisson (1.06× of AMD's nnz(L); the Sprint 22 plan's "≥ 2× reduction over AMD" target is unmet) and the simplified quotient-graph AMD's wall-time tail on small SPD corpus matrices (currently ~30 % bitset-favoured at n ≤ 1 800).  Two algorithmic fronts — bring `sparse_reorder_amd_qg` up to the full Davis 2006 reference algorithm (supervariable detection + element absorption + approximate-degree updates), and port METIS's O(1) gain-bucket structure into `graph_refine_fm` to lift FM from O(n²) to O(|E|) per pass.  Adds the per-leaf AMD call inside `nd_recurse` that the Sprint 22 ND driver doesn't yet make, and swaps the `Cholesky-via-ND` residual test fixture for one whose conditioning lets the Sprint 22 plan's 1e-12 residual target become assertable.

### Prerequisites from previous Sprints

- Sprint 22 item 4: existing simplified quotient-graph AMD in `src/sparse_reorder_amd_qg.c` — items 2 and 3 below extend this in place rather than rewriting.
- Sprint 22 item 1: existing single-pass FM in `src/sparse_graph.c` (`graph_refine_fm`) — item 5 swaps the gain-pick data structure.
- Sprint 22 item 2: existing `nd_recurse` in `src/sparse_reorder_nd.c` — item 4 lands the per-leaf AMD call alongside the existing natural-ordering fallback.
- Sprint 22 Day 13's `benchmarks/bench_amd_qg.c` — item 6 re-runs the same driver post-items-2-5 to quantify the closure.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Cholesky-via-ND residual test SPD fixture swap | Replace bcsstk14 in `test_cholesky_via_nd_residual_bcsstk14` (in `tests/test_reorder_nd.c`) with a strictly diagonally-dominant synthetic SPD fixture so the Sprint 22 plan's 1e-12 residual target becomes assertable.  The 1e-8 threshold currently in the test was a Sprint-22 fixture-conditioning workaround; bcsstk14's structural-mechanics provenance amplifies roundoff and the residual ratio gets buried in the conditioning rather than telling us about the ND ordering quality. | 4 hrs |
| 2 | Quotient-graph AMD: element absorption + supervariable detection | Extend `src/sparse_reorder_amd_qg.c` with the two big mechanisms the Sprint 22 simplification skipped.  Element absorption: when a vertex's adjacency reduces to a single element `e`, the vertex is absorbed into `e` and its workspace slot is reclaimed by the next compaction.  Supervariable detection: variables with identical adjacency hash (and full compare on hash collision) are merged into a single supervariable; the minimum-degree pivot then operates on supervariables rather than individual variables, shrinking the active set 5-20× on PDE-like matrices.  Same workspace layout as Sprint 22; the new state lives in two extra slices of `iw[]` (`super[]` / `elen[]` per Davis 2006 §7).  Validate against the existing `tests/test_reorder_amd_qg.c` parity tests on nos4 / bcsstk04 / bcsstk14 — fill should stay within Sprint 22's 1.000× bitset parity. | 24 hrs |
| 3 | Quotient-graph AMD: approximate-degree update | Replace the current exact-degree recompute with Davis's approximate degree formula `d_approx(i) = |adj(i, V)| + Σ_e |adj(e, V) \ {pivot}|` plus the dense-row skip from Davis 2006 §7 (vertices whose post-pivot degree exceeds `10·√n` skip the update).  Cuts per-pivot cost from O(adjacency) to O(adjacency-of-adjacency).  Depends on item 2 (the formula reads element adjacency lists that element absorption populates).  Adds new tests in `tests/test_reorder_amd_qg.c` that pin the approximate-degree formula's output against an exact-degree reference on a small synthetic. | 12 hrs |
| 4 | ND recursion leaves call quotient-graph AMD | Replace `nd_emit_natural`'s leaf-base-case behaviour in `nd_recurse`: at `n ≤ ND_BASE_THRESHOLD`, build a temporary `SparseMatrix` from the leaf subgraph, call `sparse_reorder_amd_qg` on it, splice the per-leaf permutation into the global `perm[]` via the existing `vertex_id_map`.  Sprint 22 Day 12 made the per-leaf AMD cheap; this item just has the ND driver use it.  Re-validate on the 10×10-grid fill test in `tests/test_reorder_nd.c`; the 1.5× looseness Sprint 22 settled on should now tighten. | 12 hrs |
| 5 | FM gain-bucket structure | Replace the O(n) max-gain scan in `graph_refine_fm` with a bucket array indexed by gain value (Davis 2006 §7 / METIS reference).  Each bucket is a doubly-linked list of vertex IDs at that gain; an O(1) find-max scans down from the highest non-empty bucket cursor that only ever moves down.  Gain updates become O(1) bucket-move (remove from old bucket, insert at new).  Lifts FM from O(n²) per pass (Sprint 22's max-gain scan dominates ND wall time on Pres_Poisson) to O(|E|) per pass.  Independent of items 2-4 — the bucket structure is local to `graph_refine_fm`. | 16 hrs |
| 6 | Cross-corpus re-bench post-items-2-5 | Re-run `benchmarks/bench_reorder.c` (cross-ordering capture) and `benchmarks/bench_amd_qg.c` (the bitset comparison foil from Sprint 22 Day 13 — unchanged here) after the algorithmic upgrades land.  Capture to `docs/planning/EPIC_2/SPRINT_23/bench_day{N}.{csv,txt}`.  Verify (a) ND/AMD on Pres_Poisson now hits ≤ 0.7× (relax from the Sprint 22 plan's 0.5× target — full closure to 0.5× may still need multi-pass FM, which is also on this sprint's list if there's budget); (b) qg-AMD wall time on bcsstk14 / n=1806 is now ≤ the Sprint 22 bitset baseline; (c) the bench_day14.txt nnz(L) row stays bit-identical or improves. | 8 hrs |
| 7 | Tests + docs + retrospective stub | New tests in `tests/test_reorder_amd_qg.c` for supervariable detection (synthetic fixture with known supervariables → assert merged-set sizes) and approximate-degree updates (parity check vs an exact-degree reference on a small synthetic).  Update `docs/algorithm.md`'s AMD subsection to describe the now-full Davis algorithm.  Append a "Sprint 23 closures" subsection to `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` with the re-bench numbers from item 6.  Stub `docs/planning/EPIC_2/SPRINT_23/RETROSPECTIVE.md` with the same eight-section structure as Sprint 22's retro. | 12 hrs |

### Deliverables

- Quotient-graph AMD upgraded to the full Davis 2006 reference algorithm (element absorption + supervariable detection + approximate-degree updates)
- ND recursion leaves consume the new AMD per-leaf instead of falling back to natural-order
- `graph_refine_fm` runs at O(|E|) per pass via the METIS gain-bucket structure
- Cross-corpus re-bench captures (`SPRINT_23/bench_*.{csv,txt}`) quantifying the closures vs Sprint 22's `bench_day14.txt` baseline
- `docs/algorithm.md` AMD subsection rewritten to describe the full Davis algorithm
- `SPRINT_22/PERF_NOTES.md` extended with a Sprint-23-closures subsection
- `SPRINT_23/RETROSPECTIVE.md` stubbed for the post-sprint write-up

**Total estimate:** ~88 hours.

**Status: Complete.**  Actual ~80 hours per the Day 1-13 budgets in `docs/planning/EPIC_2/SPRINT_23/PLAN.md` (within the 14×12 = 168-hour ceiling).  Headline outcome: ND now beats AMD on Pres_Poisson (1.06× → 0.952×) — first time in this codebase.  Headline gate (a) literal ≤ 0.7× target not met; (b) qg-AMD wall on bcsstk14 ≤ Sprint-22 bitset baseline failed by ~108×; (c) bench_day14 nnz_L bit-identical-or-better passes.  See `docs/planning/EPIC_2/SPRINT_23/RETROSPECTIVE.md` for metrics, lessons, and the Sprint-24 routing of (a) and (b).

---

## Sprint 24: Ordering Follow-Ups (Sprint 23 deferrals)

**Duration:** 14 days (~126 hours estimated)

**Goal:** Close the qg-AMD wall-time regression Sprint 23 introduced (62-199× vs Sprint 22 quotient-graph baseline on irregular SuiteSparse SPD; gate (b) hard fail in `docs/planning/EPIC_2/SPRINT_23/bench_summary_day12.md`), tighten the Pres_Poisson ND/AMD ratio toward the literal Sprint 22 plan target (Sprint 23 landed at 0.952×; this sprint targets ≤ 0.85×), close the Sprint 23 Day 13 deferral that left the AMD parity test at bcsstk14 only (Pres_Poisson skipped because USE_APPROX would push the suite past 30 minutes), and add per-day wall-time regression-check infrastructure to prevent similar regressions in future sprints.  Also lands the Davis 2006 §7.5.1 external-degree refinement (deferred from Sprint 23 Day 5) if Sprint 23's approximate-degree path is retained.

### Prerequisites from previous Sprints

- Sprint 22 item 4: existing simplified quotient-graph AMD baseline at `src/sparse_reorder_amd_qg.c` — items 2 and 4 reference this baseline as the wall-time target.
- Sprint 22 Day 13's `benchmarks/bench_amd_qg.c` — item 1 extends it with a per-fixture regression-threshold check.
- Sprint 23 Days 2-5: element absorption + supervariable detection + approximate-degree formula + dense-row skip in `src/sparse_reorder_amd_qg.c` — item 2 root-causes and fixes the wall-time regression these introduced; item 4 builds on top of the approximate-degree path if item 2 retains it.
- Sprint 23 Day 7: leaf-AMD splice in `src/sparse_reorder_nd.c` — item 5's ND fill-quality work runs against the same recursive-ND driver.
- Sprint 23 Days 9-11: gain-bucket FM + multi-pass FM at the finest level — item 5 tunes these without breaking the determinism contracts they established.
- Sprint 23 Day 13: `tests/test_reorder_amd_qg.c::test_qg_approx_degree_parity_corpus` (bcsstk14 only) — item 3 extends this to Pres_Poisson once item 2 closes the wall-time regression.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Day-by-day wall-time regression-check instrumentation | Add a `make wall-check` target that runs `build/bench_amd_qg --only bcsstk14` and `build/bench_reorder --only Pres_Poisson --skip-factor`, parses the `reorder_ms` field, and exits non-zero if either AMD wall time exceeds the prior committed baseline by > 2× on the same machine class.  Sprint 23's day-by-day commits would have caught the qg-AMD wall-time regression at Day 3 had this gate existed (regression accumulated across Days 2-5 with no intermediate signal).  Document the contract in `Makefile` + a new `### Performance regression gates` subsection in `docs/algorithm.md`.  Lessons-section item from Sprint 23's retrospective (`SPRINT_23/RETROSPECTIVE.md` "Sprint 24 inputs" #4).  Land first so items 2-5 run with the gate active. | 6 hrs |
| 2 | qg-AMD wall-time root-cause + fix | Profile `sparse_reorder_amd_qg` on bcsstk14 / Pres_Poisson under the Sprint 23 default path; identify which of Days 2-5's additions dominate the wall-time regression.  Three candidate fixes documented in `SPRINT_23/bench_summary_day12.md "(b)"`: (a) replace Day 4's hash + O(k²) full-list compare with a sorted-list compare (O(k log k) on collision) — keeps the supervariable-detection win on regular fixtures while bounding worst-case cost on irregular ones; (b) gate supervariable detection by a regularity heuristic that predicts payoff before paying the per-pivot cost; (c) revert Days 2-5 entirely (Day 11's multi-pass FM was the actual headline driver, not Days 2-5; reverting restores Sprint 22 baseline without losing the Pres_Poisson 0.95× outcome).  Pick one based on the profile, implement, validate that nnz(L) stays bit-identical across the full corpus + synthetic banded, and confirm wall time on bcsstk14 ≤ 1.5× of Sprint 22's quotient-graph baseline (~140 ms target — Sprint 22's qg ran at ~30 % above bitset, restoring that band is the realistic close).  This item is the priority-1 sprint deliverable. | 32 hrs |
| 3 | AMD parity test on Pres_Poisson | Extend `tests/test_reorder_amd_qg.c::test_qg_approx_degree_parity_corpus` from bcsstk14 to Pres_Poisson — the conservative-bound contract (`d_approx ≥ d_exact` per pivot under `SPARSE_QG_VERIFY_DEG`) on the canonical 2D-PDE benchmark.  Sprint 23 Day 13 deferred this because Pres_Poisson under USE_APPROX would push the test suite past 30 minutes on the pre-fix wall-time profile; item 2's fix should bring this back into the affordable range (target: full corpus parity test under 5 minutes total).  Skip cleanly if item 2 chooses the "revert Days 2-5 entirely" path (test isn't applicable since the approximate-degree code path is gone). | 8 hrs |
| 4 | Davis 2006 §7.5.1 external-degree refinement | Conditional on item 2 retaining the Sprint 23 approximate-degree code path (i.e. fix candidates (a) or (b), not (c)).  Davis 2006 §7.5.1 describes an "external degree" refinement that tightens the approximate-degree formula's bound by tracking which neighbours are external to the pivot's element-set vs internal.  Currently the Sprint 23 formula counts all element-side adjacency as external; the refinement walks the element's variable-set once per pivot and corrects for the overlap.  Implement, validate against the existing 50-vertex + 200-vertex parity tests (Sprint 23 Days 5 / 6) — `d_approx ≥ d_exact` must still hold — and measure pivot-order divergence vs the Sprint 23 default path on bcsstk14.  If pivot-order is meaningfully tighter (≤ 5 % nnz(L) difference vs exact-degree), promote external-degree to default for the approximate path; otherwise document the refinement as available behind `SPARSE_QG_USE_EXTERNAL_DEG` and move on.  Mentioned in Sprint 23 `davis_notes.md` "Day-1 reading" but not implemented (`SPRINT_23/RETROSPECTIVE.md` "Items deferred"). | 20 hrs |
| 5 | ND fill-quality follow-up — Pres_Poisson ≤ 0.85× | Sprint 23 Day 11's multi-pass FM landed Pres_Poisson at 0.952× of AMD; closing 0.95 → 0.7 (let alone Sprint 22's plan-target 0.5×) needs deeper algorithmic work outside Sprint 23's scope.  Two candidate axes for this sprint: (a) deepen coarsening — current bottoms out at MAX(20, n/100), try MAX(20, n/200) or a fixed coarsening floor of 50 to give the brute-force / GGGP bisection more graph to work with; (b) smarter separator extraction beyond Sprint 22's smaller-side lift — try a balanced-cost variant that lifts the side with smaller boundary regardless of side weight.  Stretch target ≤ 0.85× (a 7-pp tightening from 0.95×); the literal Sprint 22 plan-target 0.7× is explicitly out of scope unless one of (a) or (b) overshoots.  Add a Pres_Poisson nnz_nd fixture-pin to `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` once the new ratio holds (replacing the current `≤ nnz_amd` bound with a tighter `≤ 0.85× nnz_amd`).  `SPRINT_23/RETROSPECTIVE.md` "Items deferred" routes this here. | 32 hrs |
| 6 | Cross-corpus re-bench post-items-2-5 | Re-run `benchmarks/bench_reorder.c` and `benchmarks/bench_amd_qg.c` after items 2-5 land.  Capture to `docs/planning/EPIC_2/SPRINT_24/bench_*.{csv,txt}`.  Verify (a) qg-AMD wall on bcsstk14 ≤ 1.5× Sprint 22 quotient-graph baseline (~210 ms ceiling); (b) qg-AMD nnz(L) bit-identical to Sprint 22 + Sprint 23 captures; (c) Pres_Poisson ND/AMD ≤ 0.85× (item 5 stretch target); (d) all `SPRINT_23/bench_day14.txt` nnz_L rows stay bit-identical or improve.  Build a `bench_summary_day14.md` (Sprint 24 closing-day capture, following the Sprint 23 Day-12 pattern). | 12 hrs |
| 7 | Tests + docs + retrospective | New tests if items 2 / 4 / 5 added user-visible behavior (item 2's regression test in `bench_amd_qg.c` if (a) / (b); item 4's external-degree behind-flag test; item 5's Pres_Poisson nnz fixture-pin).  Update `docs/algorithm.md` AMD subsection if item 2 chose "revert Days 2-5" (drops the §"Four mechanisms" prose to whichever survived).  Append a "Sprint 24 closures" subsection to `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` (or open a new `SPRINT_24/PERF_NOTES.md` if the Sprint 22 file gets too long).  Stub `docs/planning/EPIC_2/SPRINT_24/RETROSPECTIVE.md` with the same eight-section structure as Sprint 23's retro.  Final sweep: `make format && make lint && make test`. | 16 hrs |

### Deliverables

- `make wall-check` target catching > 2× per-day wall-time regressions
- qg-AMD wall on bcsstk14 ≤ 1.5× Sprint 22 quotient-graph baseline (closing the Sprint 23 Day-12 gate (b) hard-fail)
- AMD parity test extended to Pres_Poisson (Sprint 23 Day-13 deferral closed)
- External-degree refinement in approximate-degree path (conditional on item 2 retaining that code path)
- Pres_Poisson ND/AMD ≤ 0.85× (Sprint 23 plan-target tightening; literal Sprint 22 0.5× still out of scope)
- Cross-corpus re-bench captures + summary md (`SPRINT_24/bench_*.{csv,txt}` + `bench_summary_day14.md`)
- `docs/algorithm.md` AMD subsection + `SPRINT_22/PERF_NOTES.md` Sprint-24-closures subsection
- `SPRINT_24/RETROSPECTIVE.md` stubbed for the post-sprint write-up

**Total estimate:** ~126 hours.

**Status: Complete.**  Actual ~80 hours per the Day 1-10 budgets in `docs/planning/EPIC_2/SPRINT_24/PLAN.md` (well under the 126-hour estimate; saved ~46 hours because Day 1's (c) revert closed item 2 in 1 day instead of 4 and made items 3 + 4 N/A — see `SPRINT_24/fix_decision_day1.md`).  Headline outcome: qg-AMD wall-time regression closed via revert of Sprint 23 Days 2-5 (bcsstk14 4 715 ms → 125.8 ms = 39× speedup; Pres_Poisson 759 s → 8.1 s = 93× speedup); Sprint 22 wall-time + memory profile restored bit-identically.  Headline gates: (a) qg-AMD wall on bcsstk14 ≤ 1.5× Sprint 22 baseline — PASS at 125.8 ms; (b) qg-AMD nnz(L) bit-identical — PASS 9/9 fixtures; (c) Pres_Poisson ND/AMD ≤ 0.85× — MISS at 0.952× default / 0.942× best opt-in, routed to Sprint 25 (new ND-fill-quality sprint); (d) Sprint 23 nnz_L bit-identical-or-better — PASS with 1 row in tie-break noise.  Two new ND env-var-gated alternatives shipped as documented advisory (`SPARSE_ND_COARSEN_FLOOR_RATIO=200` for Pres_Poisson; `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary` for non-Pres_Poisson workloads with 8-38pp wins on Kuu/bcsstk14/nos4).  `make wall-check` shipped Day 1; caught no regressions through the sprint.  Days 12-13 routed two pre-existing CI gate failures (coverage at 80.8 % vs 95 % threshold + build-and-test `make bench` 6-hour timeout, both inherited from PRs #28-31) to Sprint 26's CI Hardening item 8.  See `docs/planning/EPIC_2/SPRINT_24/RETROSPECTIVE.md` for metrics, lessons, the Sprint-25 routing of (c), and the Sprint-26 routing of the CI gate calibrations.

---

## Sprint 25: ND Fill-Quality Follow-Up (Sprint 24 deferrals)

**Duration:** 14 days (~128 hours)

**Goal:** Close the Pres_Poisson ND/AMD ≤ 0.85× literal target Sprints 22-24 collectively missed (Sprint 22 1.063× → Sprint 23 0.952× → Sprint 24 0.942× best opt-in) via algorithmic work outside Sprints 22-24's scope: Heavy Connectivity Coarsening (Karypis-Kumar 1998 §5), multi-pass FM at intermediate uncoarsening levels, and spectral bisection at the coarsest level.  Also profile + tighten Sprint 24's ND default-path wall-time drift (1.06-1.10× of Sprint 23 baseline) and add a Pres_Poisson ND wall line to `wall_check_baseline.txt`.  All items routed from `docs/planning/EPIC_2/SPRINT_24/RETROSPECTIVE.md` "Items deferred" + "Sprint 25 inputs" #1-3.

### Prerequisites from previous Sprints

- Sprint 22: multilevel partition pipeline (`src/sparse_graph.c::sparse_graph_hierarchy_build` for HCC; `graph_refine_fm` for multi-pass FM extension; `graph_coarsest_bisection` for spectral bisection swap) — items 1-3 modify these entry points behind env-var gates that preserve the Sprint 22 default path bit-identically.
- Sprint 22 Day 4: edge-to-vertex separator extraction (`graph_edge_separator_to_vertex_separator`) — item 4's tightening of the test fixture pin builds on this.
- Sprint 21 Day 6: Lanczos eigensolver (`sparse_eigs_sym`) — item 3's spectral bisection reuses the eigensolver for Fiedler-vector computation on the coarsest-level Laplacian.
- Sprint 23 Days 9-11: gain-bucket FM + 3-pass FM at the finest uncoarsening level — item 2 extends multi-pass to intermediate levels, building on the bucket-FM cost-per-pass reduction Sprint 23 landed.
- Sprint 24 Day 1: `make wall-check` infrastructure (`scripts/wall_check.sh` + `wall_check_baseline.txt`) — items 5 + 6 add a Pres_Poisson ND baseline line and tighten the threshold.
- Sprint 24 Day 5: `SPARSE_ND_COARSEN_FLOOR_RATIO` env var — item 1's HCC implementation interacts with the coarsening floor; item 4's re-bench evaluates the cross-product.
- Sprint 24 Day 6: `SPARSE_ND_SEP_LIFT_STRATEGY` env var — item 4's re-bench evaluates HCC + sep-strategy + multi-pass FM combinations.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Heavy Connectivity Coarsening (Karypis-Kumar 1998 §5) | Replace `src/sparse_graph.c`'s heavy-edge matching with HCC: a coarsening step that weights candidate edges by a function of (edge weight × neighbour count) rather than just edge weight, producing tighter coarse graphs on 2D PDE meshes.  Gate via a new `SPARSE_ND_COARSENING={hcc,heavy_edge}` env var (default `heavy_edge` to keep Sprint 22 behavior); flip default if HCC produces a clear corpus-wide win on Pres_Poisson + small fixtures.  Add unit tests in `tests/test_graph.c` pinning HCC's match selection on synthetic 2D-grid + irregular-SPD fixtures.  Sprint 24 Day 5's `nd_coarsen_floor_decision.md` documents the existing matching's limitations; this item adds HCC as the primary remediation path. | 28 hrs |
| 2 | Multi-pass FM at intermediate uncoarsening levels | Extend Sprint 23 Day 11's `SPARSE_FM_FINEST_PASSES` (3 passes at finest only) to the second-finest and third-finest uncoarsening levels, configurable via a new `SPARSE_FM_INTERMEDIATE_PASSES` env var defaulting to 1.  Per Sprint 24 RETROSPECTIVE.md "Performance highlights" lesson "multi-pass FM's payoff scales with the cost of a single pass"; the bucket-FM infrastructure (Sprint 23 Days 9-10) made multi-pass affordable, and Sprint 23 Day 11 only exercised it at the finest level.  Cross-corpus benchmark to identify whether 1-2 passes at the 2nd-finest level produces measurable cut tightening on Pres_Poisson without regressing smaller fixtures.  Skipped-vertex re-insertion contract (Sprint 23 Day 10's bcsstk04 LDL^T residual hazard fix) must hold across the new pass placements. | 20 hrs |
| 3 | Spectral bisection at the coarsest level | Replace `graph_coarsest_bisection`'s brute-force-or-GGGP path (currently brute-force for n ≤ 20, GGGP for n > 20) with Fiedler-vector-based spectral bisection: compute the second-smallest eigenvalue's eigenvector of the graph Laplacian via the Sprint 20-21 Lanczos eigensolver (`sparse_eigs_sym` with shift-invert at σ ≈ 0+ε), partition by the median value with a balance-tolerance fallback to GGGP if the Fiedler cut is more imbalanced than 60/40.  Gate via a new `SPARSE_ND_COARSEST_BISECTION={spectral,gggp,brute}` env var; default `gggp` (Sprint 22 behavior).  Spectral bisection produces canonical "geometric" cuts that typically beat brute-force / GGGP on regular meshes; Pres_Poisson (a 2D Poisson grid) is the canonical fit. | 32 hrs |
| 4 | Cross-corpus re-bench + production-default decisions + test-bound tightening | After items 1-3 land, run `bench_reorder.c` + `bench_amd_qg.c` across the full corpus under all combinations of the three new env vars × Sprint 24's two existing ones (`SPARSE_ND_COARSEN_FLOOR_RATIO`, `SPARSE_ND_SEP_LIFT_STRATEGY`).  Pick the corpus-wide-best combination and flip defaults if a clear winner emerges; tighten `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` from Sprint 24's `≤ 0.96× nnz_amd` to whatever the achieved ratio + 2-percentage-point noise margin allows.  Save bench captures to `docs/planning/EPIC_2/SPRINT_25/bench_*.{csv,txt}` and decision rationale to `docs/planning/EPIC_2/SPRINT_25/{coarsening_decision.md, intermediate_fm_decision.md, spectral_bisection_decision.md, headline_summary.md}`. | 16 hrs |
| 5 | ND wall-time profile + tightening | Sprint 24 Day 8 measured 42.86 s default-path Pres_Poisson ND vs Sprint 23's 36.4 s baseline (21 % drift); profile with `clock_gettime` instrumentation (similar to Sprint 24 Day 1's qg-AMD profile pattern, behind a new `SPARSE_ND_PROFILE` env var) to identify whether the drift is real algorithmic cost or run-to-run measurement variance.  If real cost, root-cause and tighten so that Sprint 25's default Pres_Poisson ND wall ≤ Sprint 23 baseline + 5 % (~38 s); if variance, document the 21 % run-to-run band as a measured property and proceed to item 6 with the relaxed threshold. | 12 hrs |
| 6 | `make wall-check` Pres_Poisson ND baseline line | Add a `pres_poisson_nd` baseline entry to `docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt` (or open `docs/planning/EPIC_2/SPRINT_25/wall_check_baseline.txt` if the Sprint 24 file gets unwieldy).  Threshold defaults to 50 % above the baseline (vs the 2× ceiling that the existing AMD baselines use) since item 5 measured run-to-run variance on this fixture at 21 %; if item 5's profile + tightening landed a tighter measurement band, threshold can drop to 25 % accordingly.  Update `scripts/wall_check.sh` to parse the new baseline key and run `bench_reorder --only Pres_Poisson --skip-factor` matching the AMD probe pattern. | 4 hrs |
| 7 | Tests + docs + retrospective | New tests for items 1-3 (HCC parity test on synthetic + corpus fixtures, multi-pass-intermediate FM bucket cleanup test, spectral bisection eigenvalue ordering test).  Update `docs/algorithm.md` ND subsection to describe the three new env vars + their per-fixture deltas; supersede Sprint 24's "Pres_Poisson 0.7× literal target route to Sprint 25" caveat with the actual achievement.  Append a "Sprint 25 closures" subsection to `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` (or open `SPRINT_25/PERF_NOTES.md` if the Sprint 22 file is too long).  Single-pass `docs/planning/EPIC_2/SPRINT_25/RETROSPECTIVE.md` (per Sprint 24 retrospective lesson "skip the stub-vs-body distinction; one Day-14 retro that absorbs the Day-13 work matches the actual time spent"). | 16 hrs |

### Deliverables

- Heavy Connectivity Coarsening implementation gated by `SPARSE_ND_COARSENING`
- Multi-pass FM at intermediate uncoarsening levels gated by `SPARSE_FM_INTERMEDIATE_PASSES`
- Spectral bisection at the coarsest level gated by `SPARSE_ND_COARSEST_BISECTION`, reusing the Sprint 20-21 Lanczos eigensolver for Fiedler-vector computation
- Pres_Poisson ND/AMD ≤ 0.85× literal target met (or partial close documented + Sprint 26 routed if a fourth algorithmic axis is needed)
- ND wall-time drift profiled + tightened, OR documented as run-to-run variance with `wall_check_baseline.txt` threshold matched
- `make wall-check` Pres_Poisson ND baseline line + 50 % threshold
- Sprint 25 cross-corpus re-bench captures + decision docs (`coarsening_decision.md`, `intermediate_fm_decision.md`, `spectral_bisection_decision.md`, `headline_summary.md`)
- Tightened `test_nd_pres_poisson_fill_with_leaf_amd` bound (target ≤ 0.85× pinned with 2pp noise margin)
- `docs/algorithm.md` ND subsection updated; `SPRINT_22/PERF_NOTES.md` Sprint-25-closures subsection appended; Sprint 25 retrospective filled in

**Davis 2006 §7.5.1 external-degree refinement (parking lot):** Sprint 24 deferred item 2 — "N/A under (c) revert; resurrect if a future sprint reintroduces approximate-degree."  No active item in Sprint 25 because the (c) revert took out the approximate-degree code path; if a future sprint reintroduces it (e.g. an alternative supervariable-detection candidate that doesn't pay the `qg_recompute_deg` element-side walk cost), the §7.5.1 refinement re-emerges as a follow-up.  Documented here for traceability; estimate budget 0 hrs unless trigger conditions are met.

**Total estimate:** ~128 hours.

---

## Sprint 26: SVD Improvements, Eigenpair Refinement, Progress Callbacks, CI Hardening & Epic 2 Wrap-Up

**Duration:** 14 days (~168 hours)

**Goal:** Address remaining review findings and the final Sprint 20 deferred follow-up: fix the dense-in-disguise SVD paths, add an opt-in inverse-iteration refinement post-pass for `sparse_eigs_sym` eigenpairs (deferred from Sprint 20), add progress/cancel callbacks for long-running routines, add Windows/macOS CI, improve the sparse low-rank approximation, calibrate the long-running coverage gate + fix the build-and-test bench-step CI hang (both inherited as pre-existing failures from PRs #28-32), and close out Epic 2 with final documentation and validation.

### Prerequisites from previous Sprints

- Sprint 11: CMake/Makefile parity and generated version header — the Windows/macOS CI jobs rely on this.
- Sprints 11–25: all Epic 2 numeric, eigensolver, and ordering features complete (Sprint 22 ordering stack + Sprint 23 ordering-quality follow-ups + Sprint 24 ordering deferrals + Sprint 25 ND fill-quality follow-ups) — needed for the final regression pass, cross-feature integration tests, and README/retrospective sweep.
- Sprint 17 / 18: existing SVD paths and low-rank accumulator whose dense intermediate item 1 replaces.
- Sprint 20: `sparse_eigs_sym` Lanczos output + shift-invert path through `sparse_ldlt_factor_opts` AUTO dispatch — item 3 layers an inverse-iteration refinement post-pass on the returned eigenpairs and reuses the same factored shift to drive the inner solves.
- Sprint 21: LOBPCG backend and the shared `sparse_eigs_t` result struct — item 3's refinement post-pass operates on the same `(λ_i, v_i)` array regardless of which backend produced it.
- Sprint 24 Days 12-13: documented the pre-existing coverage CI gate failure (PRs #28-32 all failed at 80.8 % aggregate vs 95 % `COV_THRESHOLD`) + the pre-existing `build-and-test` CI bench-step 6h timeout (PR #31's job 74315325437 hit the GitHub Actions runner ceiling); item 8 absorbs both.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Sparse low-rank without dense accumulator | Rewrite `sparse_svd_lowrank_sparse()` to build the rank-k approximation using outer product accumulation directly into sparse output, eliminating the current m*n dense intermediate. | 24 hrs |
| 2 | Full SVD U/V output beyond economy mode | Extend full SVD to optionally output complete (non-economy) U and V^T when requested. Currently only economy mode is supported for U/V. | 20 hrs |
| 3 | Iterative refinement for eigenpairs | Add an optional inverse-iteration refinement post-pass for the eigenpairs returned by `sparse_eigs_sym` (both Lanczos and LOBPCG backends). Exposed via a new `opts.refine` flag (default OFF for backward compatibility) and a small `opts.refine_max_iters` budget per pair. When enabled, runs inverse iteration on each `(λ_i, v_i)` reusing the Sprint 20 shift-invert path (`sparse_ldlt_factor_opts` AUTO dispatch) at the converged Ritz value as the shift, refining to either machine epsilon or the pair budget. The Wu/Simon residual gate remains the production accuracy contract; this is opt-in for downstream callers (deflation pipelines, response-function evaluation) who need tighter than `opts.tol` accuracy without bumping the outer iteration budget. Deferred from Sprint 20 retrospective. | 16 hrs |
| 4 | Progress/cancel callbacks | Add optional progress callback to long-running factorization and iterative solve routines (LU, Cholesky, LDL^T, QR, CG, GMRES, MINRES, BiCGSTAB, Lanczos, LOBPCG, nested dissection). Allow cancellation via callback return value. | 16 hrs |
| 5 | Windows CI with CMake | Add GitHub Actions job for Windows/MSVC using CMake. Fix any remaining portability issues (conditional test_fuzz exclusion is already done). | 16 hrs |
| 6 | macOS CI job | Add GitHub Actions job for macOS. Test both Apple Clang and Homebrew GCC. Verify coverage and packaging scripts work. | 12 hrs |
| 7 | API accessor error reporting | Add `sparse_get_err()` variant that returns error codes alongside values, or document the silent-zero-on-error contract explicitly in all accessor headers. | 12 hrs |
| 8 | Final integration testing + coverage-gate calibration + CI bench-step fix | Full regression under all sanitizers, all platforms. Cross-feature tests for new Sprint 11–22 features (including cancel-callback behavior from item 4 and the eigenpair refinement opt-in from item 3). Benchmark suite on representative matrix collection.  Also calibrate the `make coverage` gate: the 95 % `COV_THRESHOLD` has been failing on every PR since #28 (Sprint 20) at ~75-87 % per-file / ~80 % aggregate; this sprint should either tighten the test suite to push aggregate ≥ 95 % or lower `COV_THRESHOLD` to a defensible target (~85 %) with documentation of which file groups carry which coverage levels.  Also fix the `build-and-test` CI job's `make bench` step: PR #31's build-and-test ran for 6 hours before hitting GitHub Actions' max execution timeout, and PR #32's same step repeated — `make bench` runs `bench_reorder.c` with the numeric-factor pass on Pres_Poisson and the cumulative time exceeds the runner ceiling.  Either replace `make bench` in CI with a `bench-fast` / `bench-ci` target that uses `--skip-factor`, or move `make bench` to a separate scheduled (nightly) workflow that doesn't gate PR merges.  Both the coverage and bench-step gates have been red since well before Sprint 24; this item resets them to the operating reality. | 28 hrs |
| 9 | Epic 2 retrospective and documentation | Update README with all new APIs (LDL^T, IC, MINRES, BiCGSTAB, eigensolvers, eigenpair refinement, COLAMD, ND, progress callbacks). Write Epic 2 retrospective. Update INSTALL.md for new platforms. | 24 hrs |

### Deliverables

- Sparse low-rank approximation without dense intermediate
- Full U/V SVD output option
- Optional inverse-iteration refinement for `sparse_eigs_sym` eigenpairs (gated by `opts.refine`), composing with the Sprint 20 shift-invert dispatch
- Progress/cancel callbacks for long-running operations
- CI on Windows (MSVC) and macOS (Clang + GCC)
- Coverage gate calibrated against operating reality (per-file 75-87 %, aggregate ~80 %); either lifted to ≥ 95 % via test additions or `COV_THRESHOLD` lowered to a defensible value with documentation
- `build-and-test` CI bench-step fix: `make bench`'s 6-hour-runner-timeout failure on PRs #31 + #32 closed by either a `bench-fast` / `bench-ci` target or a move-to-nightly-schedule
- All new APIs documented in README
- Epic 2 retrospective with metrics and assessment

**Total estimate:** ~168 hours

---

## Summary

| Sprint | Title | Key Deliverables | Estimate |
|--------|-------|-----------------|----------|
| 11 | Build System, Tolerances & Thread Safety | CMake sync, tolerance standardization, thread safety fixes, factored-state flag | 98 hrs |
| 12 | Sparse LDL^T Factorization | Symmetric indefinite direct solver with Bunch-Kaufman pivoting | 144 hrs |
| 13 | Incomplete Cholesky & MINRES | IC(0) preconditioner, MINRES solver for symmetric indefinite systems | 136 hrs |
| 14 | Symbolic/Numeric Factorization Split | Elimination tree, symbolic analysis, numeric refactorization | 152 hrs |
| 15 | COLAMD Ordering & QR Min-Norm | COLAMD for QR, minimum-norm least squares for underdetermined systems | 140 hrs |
| 16 | BiCGSTAB & Iterative Hardening | BiCGSTAB solver, stagnation detection, convergence diagnostics | 128 hrs |
| 17 | CSR/CSC Numeric Backend | CSC Cholesky and LDL^T with supernodal optimization | 152 hrs |
| 18 | CSC Kernel Performance Follow-Ups | Native CSC BK LDL^T, batched supernodal Cholesky, transparent dispatch, larger corpus | 124 hrs |
| 19 | CSC Kernel Tuning & Native Supernodal LDL^T | Analyze-once bench, small-matrix threshold study, scalar-CSC Kuu regression fix, native supernodal LDL^T, LDL^T row-adjacency index | 168 hrs |
| 20 | LDL^T Completion & Symmetric Lanczos — **Complete** | `ldlt_csc_from_sparse_with_analysis`, transparent `sparse_ldlt_factor_opts` dispatch, Lanczos + shift-invert eigensolver | 136 hrs (~125 actual) |
| 21 | Eigensolver Completion — Thick-Restart, OpenMP & LOBPCG — **Complete** | Wu/Simon thick-restart, OpenMP reorth, LOBPCG, permanent `bench_eigs` | 124 hrs (~133 actual) |
| 22 | Ordering Upgrades — Nested Dissection & Quotient-Graph AMD — **Complete** | Graph partitioning + nested dissection, quotient-graph AMD | 124 hrs (~134 actual) |
| 23 | Ordering Quality Follow-Ups (Sprint 22 deferrals) — **Complete** | Full Davis-style quotient-graph AMD (supervariables / element absorption / approximate-degree), O(1) FM gain buckets, ND leaves call AMD, Cholesky-via-ND test SPD fixture swap | 88 hrs (~80 actual) |
| 24 | Ordering Follow-Ups (Sprint 23 deferrals) — **Complete** | Wall-time regression-check infrastructure, qg-AMD wall-time root-cause + fix (Sprint 23 Days 2-5 reverted), `SPARSE_ND_COARSEN_FLOOR_RATIO` + `SPARSE_ND_SEP_LIFT_STRATEGY` env vars, tightened test bounds | 126 hrs (~80 actual) |
| 25 | ND Fill-Quality Follow-Up (Sprint 24 deferrals) | Heavy Connectivity Coarsening, multi-pass FM at intermediate levels, spectral bisection at coarsest level, ND wall-time profile + tightening, Pres_Poisson ND `wall-check` baseline | 128 hrs |
| 26 | SVD, Progress Callbacks, Eigenpair Refinement, CI & Wrap-Up | Sparse low-rank fix, full SVD, eigenpair iterative refinement, progress/cancel callbacks, Windows/macOS CI, coverage-gate calibration, CI bench-step fix, Epic 2 retrospective | 168 hrs |

**Total Epic 2 estimate:** ~2,136 hours across 16 sprints (~214 days)
