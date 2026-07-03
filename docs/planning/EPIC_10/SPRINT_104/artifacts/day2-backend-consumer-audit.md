# Sprint 104 Day 2 Backend Consumer Audit

## Purpose

Day 2 inventories backend consumers, dense-kernel seams, optional acceleration
points, runtime controls, and benchmark/example observability before Sprint
104 changes descriptor or threading behavior. The goal is to separate builtin
fallback truth from optional acceleration and to expose ambiguous runtime
controls before Day 3 contract design.

## Source Surfaces Reviewed

| area | files reviewed | reason |
|---|---|---|
| Cholesky backend selector | `include/sparse_matrix.h`, `src/sparse_cholesky.c`, `tests/test_chol_csc.c` | public/backend selector and AUTO threshold behavior |
| Cholesky CSC dense-kernel descriptor | `src/sparse_chol_csc_internal.h`, `src/sparse_dense.c`, `src/sparse_chol_csc_supernodal.c`, `tests/test_chol_csc_supernodal.c` | builtin/optional dense-kernel seam |
| LDLT backend selector and dense backend | `include/sparse_ldlt.h`, `src/sparse_ldlt_dense.c`, `src/sparse_ldlt_csc_supernodal.c`, `tests/test_ldlt.c`, `benchmarks/bench_refactor_csc.c` | AUTO CSC dispatch and optional dense-factor backend |
| Eigensolver backend selector | `include/sparse_eigs.h`, `src/sparse_eigs.c`, `src/sparse_eigs_internal.h`, `tests/test_eigs*.c`, `benchmarks/bench_eigs*.c`, `examples/example_eigs.c` | AUTO/grow-m/thick-restart/LOBPCG dispatch and `backend_used` observability |
| OpenMP runtime controls | `src/sparse_matrix.c`, `src/sparse_eigs.c`, `tests/test_omp.c`, `Makefile`, `CMakeLists.txt`, `.github/workflows/ci.yml` | SpMV and MGS OpenMP behavior |
| Graph and ND runtime controls | `src/sparse_graph*.c`, `src/sparse_reorder_nd.c`, `src/sparse_reorder_amd_qg.c`, `include/sparse_analysis.h`, `src/sparse_graph_internal.h`, `src/sparse_reorder_nd_internal.h`, `tests/test_graph.c`, `tests/test_reorder_nd.c` | typed analysis controls, env compatibility, thread-local overrides |
| SVD runtime toggle | `src/sparse_svd.c`, `tests/test_svd.c` | low-rank outer-product env toggle |
| Benchmark and docs observability | `benchmarks/README.md`, `benchmarks/bench_chol_csc.c`, `benchmarks/bench_refactor_csc.c`, `benchmarks/bench_eigs.c`, `benchmarks/bench_eigs_reuse.c`, `docs/maintainer_guide.md`, `README.md` | backend/runtime reporting fields and claim wording |

## Backend Consumer Inventory

| consumer family | primary consumers | backend or runtime control | current observability |
|---|---|---|---|
| Direct Cholesky linked-list/CSC | `sparse_cholesky_factor_opts`, `tests/test_chol_csc.c`, `bench_chol_csc` | `sparse_cholesky_opts_t::backend` with AUTO/LINKED_LIST/CSC; AUTO uses `SPARSE_CSC_THRESHOLD` | benchmark path columns; tests force both paths; public docs mention threshold |
| Cholesky CSC supernodal dense kernels | `chol_csc_supernodal_eliminate_diag`, `chol_csc_supernodal_eliminate_panel`, `bench_chol_csc` | internal `chol_dense_kernels_t`; env `SPARSE_CHOL_DENSE_BACKEND` accepts builtin/external/blas/lapack/accelerate | descriptor `name`; benchmark reports `csc_supernodal_dense_kernel` and panel solver |
| LDLT linked-list/CSC | `sparse_ldlt_factor_opts`, shift-invert eigensolver path, tests, `bench_ldlt_csc`, `bench_refactor_csc` | `sparse_ldlt_opts_t::backend` with AUTO/LINKED_LIST/CSC; AUTO uses `SPARSE_CSC_THRESHOLD` | optional `used_csc_path`; eigensolver `used_csc_path_ldlt`; benchmark backend fields |
| LDLT CSC dense factor | `ldlt_csc_supernode_factor_dense_block`, `tests/test_ldlt.c`, `bench_refactor_csc` | `ldlt_dense_factor_selected`; env `SPARSE_LDLT_DENSE_BACKEND` accepts builtin/external/blas/lapack/accelerate | `ldlt_dense_factor_backend_name()`; benchmark request/selected/fallback columns |
| QR dense helper path | `src/sparse_qr.c`, `tests/test_qr.c` | internal dense buffers and Householder kernels; no optional backend selector | correctness tests only; no backend descriptor |
| SVD and low-rank | `src/sparse_svd.c`, `src/sparse_svd_partial.c`, `tests/test_svd.c`, `bench_svd` | builtin dense/SVD helper paths; env `SPARSE_SVD_LOWRANK_OUTER` toggles low-rank sparse output path | tests cover toggle; no public backend descriptor |
| Eigensolver grow-m/thick-restart/LOBPCG | `sparse_eigs_sym`, `bench_eigs`, `bench_eigs_reuse`, `example_eigs` | `sparse_eigs_opts_t::backend` with AUTO/LANCZOS/THICK_RESTART/LOBPCG; thresholds for thick-restart and LOBPCG | `sparse_eigs_t::backend_used`; benchmark output maps AUTO to concrete backend |
| Shift-invert eigensolver | `src/sparse_eigs.c`, `tests/test_eigs.c` | internal LDLT factorization with `SPARSE_LDLT_BACKEND_AUTO` | `result.used_csc_path_ldlt`; no dense-backend name |
| SpMV and block SpMV | `sparse_matvec`, `sparse_matvec_block`, iterative solvers, eigensolvers, benchmarks | compile-time `SPARSE_OPENMP`; row-wise OpenMP pragmas | `test_omp_status`; `bench_main` reports OpenMP max threads when enabled |
| Lanczos MGS reorthogonalization | grow-m and thick-restart eigensolver internals | compile-time `SPARSE_OPENMP`; `SPARSE_EIGS_OMP_REORTH_MIN_N` compile-time gate | README documents gate; CI TSan OpenMP job exercises eigensolver tests |
| Graph/FM/ND runtime controls | graph partition, nested dissection, analysis dispatch, reorder benchmarks/tests | typed analysis fields plus legacy `SPARSE_ND_*`, `SPARSE_FM_*`, profile/debug env vars and thread-local overrides | maintainer docs list compatibility/env ownership; tests cover precedence |
| Benchmarks and examples | `bench_chol_csc`, `bench_refactor_csc`, `bench_eigs`, `bench_eigs_reuse`, `example_eigs` | benchmark flags, env-derived dense backend names, eigensolver backend selectors | CSV/report fields; example prints LOBPCG backend result |

## Builtin Fallback Behavior

| surface | builtin fallback truth | optional behavior |
|---|---|---|
| Cholesky CSC dense kernels | `chol_csc_supernodal_dense_kernels()` returns the `builtin` descriptor unless a test override is enabled or an optional backend is both requested and successfully probed | `SPARSE_CHOL_DENSE_BACKEND=external/blas/lapack` probes BLAS/LAPACK-class libraries; `accelerate` is Apple-only; failed/unknown requests fall back to builtin |
| LDLT dense factor | `ldlt_dense_factor_selected()` preserves `ldlt_dense_factor(...)` as the shipped builtin implementation | `SPARSE_LDLT_DENSE_BACKEND=external/blas/lapack/accelerate` can select an external dense factor when probe and pivot-pattern constraints allow; otherwise fallback is builtin |
| Cholesky backend selector | AUTO routes by `SPARSE_CSC_THRESHOLD`; explicit LINKED_LIST/CSC force path where valid | optional dense-kernel choice affects only CSC supernodal internals, not the public selector enum |
| LDLT backend selector | AUTO routes by `SPARSE_CSC_THRESHOLD`; explicit LINKED_LIST/CSC force path except documented empty-matrix edge case | dense-factor backend choice is separate from linked-list/CSC selector |
| Eigensolver selector | AUTO selects a concrete backend by preconditioner/block/size thresholds and records `backend_used` on successful calls | no optional vendor backend; LOBPCG and thick-restart are builtin concrete algorithms |
| OpenMP SpMV/MGS | serial build is the default; OpenMP pragmas compile away without `SPARSE_OPENMP` | OpenMP builds parallelize SpMV/block SpMV and Lanczos MGS inner loops |
| Graph/ND controls | typed analysis options are primary where available; env vars are compatibility/default fallbacks | thread-local overrides and legacy env vars can alter selected internal policy |

## Optional Acceleration Points

| acceleration point | selector | provider mechanism | current boundary |
|---|---|---|---|
| Cholesky CSC dense block factor and panel solves | `SPARSE_CHOL_DENSE_BACKEND` | dynamic lookup of Accelerate/OpenBLAS/BLAS/LAPACK-class symbols on non-Windows; Apple-only `accelerate` request | internal Cholesky CSC supernodal descriptor only |
| LDLT CSC dense Bunch-Kaufman factor | `SPARSE_LDLT_DENSE_BACKEND` | dynamic lookup of `dsytrf`-class provider on non-Windows | internal LDLT CSC supernodal dense-factor seam only |
| OpenMP SpMV/block SpMV | build with `SPARSE_OPENMP` | compiler/runtime OpenMP support | matrix-vector kernels only |
| OpenMP Lanczos MGS | build with `SPARSE_OPENMP`; compile-time `SPARSE_EIGS_OMP_REORTH_MIN_N` | compiler/runtime OpenMP support | grow-m and thick-restart MGS inner loops |

No GPU, distributed-memory, universal BLAS/LAPACK, or public vendor-backend
selection surface was found.

## Benchmark and Example Observability

| surface | backend/runtime fields | interpretation |
|---|---|---|
| `bench_chol_csc` | `csc_scalar_path`, `csc_supernodal_path`, `csc_supernodal_dense_kernel`, `csc_supernodal_panel_solver` | Cholesky CSC measurement surface; reports active dense-kernel descriptor, not broad backend parity |
| `bench_refactor_csc` | `ldlt_dense_backend_request`, `ldlt_dense_backend_selected`, `ldlt_dense_backend_fallback` | retained repeated-run LDLT lane; discloses env request and selected dense backend |
| `bench_eigs` | requested and concrete eigensolver backend labels, residuals, iterations, memory/basis metrics | comparison/reporting surface for builtin eigensolver algorithms |
| `bench_eigs_reuse` | one-shot/reuse statuses and `backend_used` | repeated-run lifecycle measurement surface |
| `bench_main` | OpenMP max thread banner when compiled with `SPARSE_OPENMP` | SpMV benchmark context, not universal OpenMP performance claim |
| `example_eigs` | requested LOBPCG and printed backend result | adoption-facing demonstration, not benchmark evidence |

## Runtime-Control Risks

| risk | affected surface | why it matters for Sprint 104 |
|---|---|---|
| Env-selected dense backends are process-global | `SPARSE_CHOL_DENSE_BACKEND`, `SPARSE_LDLT_DENSE_BACKEND` | tests and benchmarks must isolate env state; concurrent calls with env mutation are not safe |
| Cholesky and LDLT dense backend selectors are similar but not identical | `chol_dense_kernels_t`, `ldlt_dense_factor_backend_name()` | Day 3 should decide whether descriptor wording/status fields need alignment |
| Optional backend fallback is silent unless caller reads descriptor/report fields | dense backend accessors and benchmarks | public/runtime contract should state whether silent fallback is expected behavior |
| LDLT `accelerate` request can still report builtin | `SPARSE_LDLT_DENSE_BACKEND=accelerate` | fallback semantics need explicit wording so users do not assume acceleration landed |
| OpenMP thread count is external-runtime controlled | SpMV, MGS, `bench_main`, CI env `OMP_NUM_THREADS` | benchmark/sentinel artifacts must disclose thread settings |
| Nested parallelism is not a first-class runtime contract | eigensolver + SpMV + preconditioner composition, future BLAS calls | Day 3 should define non-claims and any expected restraint |
| Graph/ND env compatibility surface is broad | `SPARSE_ND_*`, `SPARSE_FM_*`, profile/debug env vars | not all runtime controls belong in the Sprint 104 backend descriptor; classify separately |
| Test-only descriptor override can return NULL | Cholesky dense-kernel override | good for error-path proof, but Day 3 should keep it explicitly test-only |
| Benchmark residual columns may look like correctness proof | direct/eigs benchmark CSV rows | Sprint 100 rule: tests/oracles own correctness, benchmarks provide context |

## Initial Cleanup Queue

| priority | candidate | reason | likely owner day |
|---:|---|---|---|
| 1 | Write a unified runtime contract for builtin fallback, optional backend request, selected backend, and fallback reporting | reduces ambiguity before source changes | Day 3 |
| 2 | Decide whether Cholesky and LDLT dense-backend descriptors need aligned naming/status wording | current surfaces are parallel but not identical | Day 4 |
| 3 | Document silent fallback as deliberate or replace with explicit status where local patterns support it | important for benchmark interpretation | Days 3-5 |
| 4 | Classify OpenMP controls as compile-time, runtime env, benchmark context, or non-claim | required before sentinels | Days 6-8 |
| 5 | Keep graph/ND env controls out of dense-backend descriptor scope unless Day 6 threading audit needs them | avoids over-widening Sprint 104 | Day 6 |
| 6 | Require sentinel artifacts to record `OMP_NUM_THREADS`, dense backend env, selected backend, and fallback status where relevant | prevents misleading local timing claims | Days 8-9 |
| 7 | Align benchmark docs so residual and timing fields stay separate from correctness and portability claims | supports Day 10-11 reporting work | Days 10-11 |

## Day 2 Completion Check

| criterion | status |
|---|---|
| direct factorization consumers represented | complete |
| LDLT and Cholesky CSC paths represented | complete |
| QR and dense helper paths represented | complete |
| eigensolver and LOBPCG paths represented | complete |
| SVD and low-rank paths represented | complete |
| benchmark and example paths represented | complete |
| builtin fallback behavior separated from optional acceleration | complete |
| ambiguous runtime-control risks listed before design starts | complete |

## Day 3 Handoff

Day 3 should convert this audit into a runtime contract. The contract should
answer:

1. Which backend/runtime controls are user-facing API, internal diagnostics,
   benchmark-only context, or compatibility env hooks?
2. Whether optional dense-backend fallback remains silent by design or must be
   surfaced through stronger status fields.
3. How OpenMP thread context and nested parallelism should be disclosed in
   tests, benchmarks, sentinels, and docs.
4. Which descriptor or status alignment belongs in Day 4/5 and which belongs
   in docs only.
