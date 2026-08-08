# Day 3 Backend Dispatch Audit

## Purpose

Day 3 audits backend routing and fallback behavior before Sprint 142 drafts
or implements precedence rules. The audit focuses on direct solvers,
eigensolvers, dense helper selection, OpenMP/build-mode interactions, and
sentinel/report paths. It documents current behavior and gaps only; it does
not change implementation behavior.

## Dispatch Summary

| Surface | Selector | AUTO/default route | Forced route behavior | Fallback semantics | Telemetry |
| --- | --- | --- | --- | --- | --- |
| Cholesky one-shot factorization | `sparse_cholesky_opts_t::backend` | `SPARSE_CHOL_BACKEND_AUTO` selects CSC when `n >= SPARSE_CSC_THRESHOLD`, otherwise linked-list. | `SPARSE_CHOL_BACKEND_LINKED_LIST` forces linked-list; `SPARSE_CHOL_BACKEND_CSC` forces the CSC path. Invalid enum values return `SPARSE_ERR_BADARG` before factorization. | No top-level dispatch fallback after `use_csc` is selected. CSC lower-level contract failures surface as backend errors or normal factorization errors. | `used_csc_path` is published immediately after dispatch selection, including before later reorder/factor errors. |
| LDLT one-shot factorization | `sparse_ldlt_opts_t::backend` | `SPARSE_LDLT_BACKEND_AUTO` selects CSC when `n >= SPARSE_CSC_THRESHOLD`, otherwise linked-list. | `SPARSE_LDLT_BACKEND_LINKED_LIST` forces linked-list; `SPARSE_LDLT_BACKEND_CSC` forces CSC except `n == 0`, which remains linked-list. Invalid enum values return `SPARSE_ERR_BADARG`. | "CSC selected" includes the batched supernodal completion path and the resolved scalar-prepass fallback inside the CSC pipeline when the batched path rejects the cached pivot pattern. Empty matrices are the only documented top-level forced-CSC exception. | `used_csc_path` is published immediately after dispatch selection and reports the actual top-level selected path. |
| Eigensolver public API | `sparse_eigs_opts_t::backend` | `SPARSE_EIGS_BACKEND_AUTO` picks LOBPCG when a preconditioner is supplied, `n >= SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD`, and effective block size is at least 4; otherwise it picks thick-restart Lanczos at `n >= SPARSE_EIGS_THICK_RESTART_THRESHOLD` and grow-m Lanczos below that threshold. | Explicit `LANCZOS`, `LANCZOS_THICK_RESTART`, or `LOBPCG` bypasses AUTO. Explicit LOBPCG is allowed even without a preconditioner. Invalid enum values return `SPARSE_ERR_BADARG`. | Lanczos-family backends ignore user preconditioners except shift-invert internals; LOBPCG honors valid preconditioners. Progress/cancel coverage differs by backend: grow-m and LOBPCG emit outer progress; thick-restart currently does not. | `backend_used` records selected backend on successful calls and is best-effort on errors; `peak_basis_size` records memory-relevant basis width; `used_csc_path_ldlt` reports the LDLT path selected for shift-invert. |
| Cholesky dense helper | `SPARSE_CHOL_DENSE_BACKEND` | Unset, empty, invalid, or unsupported values use builtin kernels. | `external`, `blas`, and `lapack` request external BLAS/LAPACK; `accelerate` is accepted on Apple builds. | Invalid and unavailable helper requests fall back to builtin; benchmark rows distinguish request from selected dense kernel. Missing dense kernel descriptors are backend contract errors in focused tests. | `bench_chol_csc` emits dense-kernel and panel-solver descriptors; `performance-sentinels` copies request/selected/panel context into S2 rows. |
| LDLT dense helper | `SPARSE_LDLT_DENSE_BACKEND` | Unset or `builtin` uses builtin kernels. Invalid values fall back to builtin. | `external`, `blas`, and `lapack` request external; `accelerate` is accepted on Apple builds. | Invalid and unavailable helper requests fall back to builtin; LDLT CSC still may use scalar-prepass fallback under the selected CSC top-level path. | `bench_refactor_csc` emits `ldlt_dense_backend_request`, `ldlt_dense_backend_selected`, and `ldlt_dense_backend_fallback`; sentinel metadata records the env request context. |
| OpenMP-enabled kernels | `SPARSE_OPENMP`, `OMP_NUM_THREADS`, `SPARSE_EIGS_OMP_REORTH_MIN_N` | Serial build is default. OpenMP code compiles only when `SPARSE_OPENMP` is enabled. | Build flag enables row-parallel SpMV and eigensolver MGS inner-loop parallel regions. `OMP_NUM_THREADS` remains caller/runtime-owned. | No runtime fallback is exposed by the library; binaries are either compiled with OpenMP regions or not. Small eigensolver vectors stay serial under the reorth threshold. | Benchmark and sentinel reports record build mode and `OMP_NUM_THREADS` as comparison context. |
| SVD low-rank reconstruction | `SPARSE_SVD_LOWRANK_OUTER` | Default/off uses dense-intermediate accumulator. | `on` routes low-rank sparse reconstruction through per-cell outer-product accumulator. | Invalid/unrecognized values fall through to default/off. If the outer-product path returns an error, the call propagates it rather than silently falling back to the dense-intermediate path. | No normalized sentinel row currently records the selector; tests compare dense-intermediate and outer-product outputs. |

## Direct-Solver Routing Details

| Solver path | Implementation owner | Validation owner | Current proof |
| --- | --- | --- | --- |
| Cholesky AUTO below threshold | `s63_cholesky_dispatch_select_backend` in `src/sparse_cholesky.c` | `tests/test_chol_csc.c`, `tests/test_direct_csc_dispatch.c` | Below-threshold fixtures assert `used_csc_path == 0` and successful residual behavior. |
| Cholesky AUTO at/above threshold | `s63_cholesky_dispatch_select_backend`, CSC path in `src/sparse_cholesky.c` and `src/sparse_chol_csc*.c` | `tests/test_chol_csc.c`, `tests/test_direct_csc_dispatch.c`, integration tests | `SPARSE_CSC_THRESHOLD` and SuiteSparse-sized fixtures assert `used_csc_path == 1`, writeback, solve, and reuse behavior. |
| Cholesky forced linked-list and forced CSC | `s63_cholesky_dispatch_select_backend` | `tests/test_chol_csc.c`, `tests/test_direct_csc_dispatch.c`, integration retry tests | Large matrices can force linked-list and small matrices can force CSC; retry tests prove invalid backend requests preserve caller-owned matrix state. |
| Cholesky invalid state/enum | `sparse_cholesky_factor_opts` precondition checks | `tests/test_chol_csc.c`, integration invalid-backend tests | Invalid enum values return `SPARSE_ERR_BADARG`; non-original matrix state is rejected before path-specific behavior diverges. |
| LDLT AUTO below threshold | `ldlt_dispatch_select_backend` in `src/sparse_ldlt.c` | `tests/test_ldlt_backend_dispatch.c`, `tests/test_ldlt.c` | Below-threshold fixtures assert `used_csc_path == 0` and solve residual behavior. |
| LDLT AUTO at/above threshold | `ldlt_dispatch_select_backend`, `ldlt_factor_csc_path` | `tests/test_ldlt_backend_dispatch.c`, `tests/test_ldlt_csc.c`, direct CSC regression tests, integration tests | SPD and KKT fixtures assert `used_csc_path == 1` and solve residual behavior. |
| LDLT forced linked-list and forced CSC | `ldlt_dispatch_select_backend` | `tests/test_ldlt_backend_dispatch.c`, integration forced-path coverage | Large matrices can force linked-list and small matrices can force CSC; `n == 0` is documented as the forced-CSC exception. |
| LDLT CSC internal completion fallback | `ldlt_factor_csc_path`, `src/sparse_ldlt_csc*.c` | `tests/test_direct_csc_regression.c`, LDLT CSC tests | CSC-selected path can complete through batched supernodal logic or resolved scalar-prepass fallback; both remain CSC pipeline behavior for telemetry purposes. |

## Eigensolver Routing Details

| Scenario | Selected backend | Validation owner | Notes |
| --- | --- | --- | --- |
| Explicit grow-m Lanczos | `SPARSE_EIGS_BACKEND_LANCZOS` | `tests/test_eigs.c`, `benchmarks/bench_eigs.c` | Used by tests that pin the older grow-m path, peak-basis behavior, and cross-backend comparisons. |
| Explicit thick-restart Lanczos | `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART` | `tests/test_eigs_thick_restart.c`, `tests/test_eigs.c` | Tests assert bounded peak basis and exact diagonal/corpus behavior. |
| Explicit LOBPCG | `SPARSE_EIGS_BACKEND_LOBPCG` | `tests/test_eigs_lobpcg.c`, integration tests | Explicit LOBPCG overrides AUTO's no-preconditioner preference and records `backend_used == LOBPCG`. |
| AUTO, small `n`, no eligible preconditioner | Grow-m Lanczos | `tests/test_eigs_lobpcg.c`, `tests/test_eigs_thick_restart.c` | Tests below `SPARSE_EIGS_THICK_RESTART_THRESHOLD` assert `backend_used == LANCZOS`. |
| AUTO, large `n`, no eligible preconditioner | Thick-restart Lanczos | `tests/test_eigs_lobpcg.c`, `tests/test_eigs_thick_restart.c` | Tests above threshold assert `backend_used == LANCZOS_THICK_RESTART`. |
| AUTO, large `n`, eligible preconditioner, block size at least 4 | LOBPCG | `tests/test_eigs_lobpcg.c`, `benchmarks/bench_eigs.c` | LOBPCG has priority above the Lanczos threshold rule. |
| AUTO, large `n`, preconditioner absent or block too small | Thick-restart Lanczos | `tests/test_eigs_lobpcg.c`, `benchmarks/bench_eigs.c` | Benchmark preconditioner gating mirrors the public API predicate so CSV rows reflect the backend that can actually use the preconditioner. |
| NEAREST_SIGMA | Selected eigensolver plus internal shift-invert LDLT | `tests/test_eigs.c`, `tests/test_eigs_lobpcg.c` | `used_csc_path_ldlt` reports whether the internal LDLT factorization selected CSC for the shifted system. |

## Dense Helper and Panel Solver Details

| Surface | Current behavior | Validation and reporting |
| --- | --- | --- |
| Cholesky dense backend request | `SPARSE_CHOL_DENSE_BACKEND` accepts builtin/default, external/blas/lapack, and Apple-only accelerate. Unsupported requests resolve to builtin. | `tests/test_chol_csc_supernodal.c` covers default, builtin env, external env, accelerate env, invalid fallback, and missing-kernel backend contract errors. `bench_chol_csc` emits dense kernel and panel solver fields. |
| Cholesky panel solver | Current default descriptor reports `batched_panel` when the dense-kernel descriptor has `solve_panel`. | Supernodal panel tests cover batched panel solve, no-panel fast path, null/missing descriptor failures, and writeback/extract edge cases. |
| LDLT dense backend request | `SPARSE_LDLT_DENSE_BACKEND` accepts builtin/default, external/blas/lapack, and Apple-only accelerate. Unsupported requests resolve to builtin. | `bench_refactor_csc` emits request/selected/fallback fields. LDLT CSC and refactor tests cover solver correctness but dense-env request/fallback coverage is less direct than Cholesky's focused env tests. |
| Sentinel dense reporting | S2 Cholesky rows include backend request, selected dense kernel, dense kernel field, and panel solver field. | `scripts/performance_sentinels.sh` records `SPARSE_CHOL_DENSE_BACKEND`, `SPARSE_LDLT_DENSE_BACKEND`, build mode, `OMP_NUM_THREADS`, and threshold-free Cholesky rows; LDLT env request is currently contextual note data rather than a first-class LDLT sentinel row. |

## OpenMP and Build-Mode Interactions

| Surface | Current behavior | Audit result |
| --- | --- | --- |
| Make/CMake OpenMP enablement | `SPARSE_OPENMP` is off by default and adds compiler/link flags only when requested. | Dispatch audit treats OpenMP as a build-time feature flag, not a runtime backend selector. |
| SpMV | `src/sparse_matrix.c` owns row-parallel linked-list SpMV under `SPARSE_OPENMP`. | Solver paths may use OpenMP indirectly through SpMV. No solver-level backend dispatch should claim thread-count ownership. |
| Eigensolver MGS reorthogonalization | `src/sparse_eigs.c` parallelizes inner product/daxpy bodies under `SPARSE_OPENMP` with `SPARSE_EIGS_OMP_REORTH_MIN_N`. | Reorth threshold affects performance path selection inside a backend, not backend identity. |
| Report context | `scripts/performance_sentinels.sh` and `scripts/bench_canonical_report.sh` record build mode and `OMP_NUM_THREADS`. | Report rows should keep this as comparison context and avoid OpenMP speedup or portability claims. |

## Current Coverage Map

| Path family | Covered now | Gap or limitation |
| --- | --- | --- |
| Cholesky AUTO and forced dispatch | Covered by focused Cholesky CSC, direct CSC dispatch, and integration tests. | Sentinel coverage is Cholesky-focused but timing-only/advisory except wall-check; no separate dispatch-only sentinel row. |
| Cholesky invalid backend/state preservation | Covered by focused and integration retry tests. | Day 4 should define whether telemetry-on-error wording is public contract or diagnostic behavior. |
| LDLT AUTO and forced dispatch | Covered by `tests/test_ldlt_backend_dispatch.c`, LDLT CSC tests, direct CSC regression, and integration tests. | Empty forced-CSC exception should be included in the Day 4 fallback matrix; focused public test coverage for that exact exception should be verified before any wording hardens it. |
| LDLT CSC internal scalar-prepass fallback | Covered indirectly by CSC regression/factor tests. | Reports currently do not distinguish batched supernodal completion from scalar-prepass completion. |
| Eigensolver AUTO dispatch | Covered across grow-m, thick-restart, and LOBPCG tests with `backend_used` assertions. | Error-path `backend_used` is best-effort; Day 4 should keep public wording success-focused. |
| Eigensolver preconditioner gating | Covered in LOBPCG tests and mirrored in `bench_eigs`. | Sentinel rows do not currently expose AUTO eigensolver backend decisions. |
| Shift-invert LDLT dispatch telemetry | Covered by `used_csc_path_ldlt` assertions for large and small shifted systems. | No normalized sentinel row currently records this telemetry. |
| Cholesky dense helper env fallback | Covered by focused supernodal tests and benchmark descriptor rows. | Env selector remains maintainer-only; typed promotion risk belongs to Day 6. |
| LDLT dense helper env fallback | Benchmark rows expose request/selected/fallback; solver tests cover correctness. | Focused unit tests for invalid env fallback are weaker than Cholesky's; candidate proof owner if promoted or surfaced in sentinel rows. |
| OpenMP build mode | Covered by `tests/test_omp.c`, `make omp`, and report metadata. | No runtime API exists; Day 4 should avoid treating `OMP_NUM_THREADS` as library-owned. |
| SVD low-rank env dispatch | Covered by `tests/test_svd.c` and Sprint 29 integration tests. | It is not currently part of runtime/backend sentinel rows; classify as maintainer env unless Day 6 selects it. |

## Candidate Sentinel Expansion List

| Candidate | Why it may help | Proposed proof owner | Claim boundary |
| --- | --- | --- | --- |
| Dispatch-only Cholesky/LDLT route snapshot | Makes AUTO/forced dispatch route observable without relying on timing rows. | Focused test binary or small report script reading existing `used_csc_path` paths. | Local regression/context only; no performance or broad correctness claim. |
| Eigensolver AUTO backend snapshot | Records AUTO selected backend for small, threshold-large, and preconditioned-large cases using `backend_used`. | Existing eigensolver tests or a cheap sentinel helper. | Local dispatch regression only; no convergence superiority or performance claim. |
| Shift-invert LDLT route snapshot | Records `used_csc_path_ldlt` for below/above threshold shifted systems. | Existing eigensolver shift-invert tests or sentinel helper. | Local routing evidence only; no broad shift-invert correctness claim. |
| LDLT dense helper request/selected/fallback row | Gives LDLT parity with Cholesky dense backend fields in sentinel output. | `bench_refactor_csc` output or sentinel script extension. | Threshold-free local context only; no BLAS/Accelerate support claim. |
| SVD low-rank env selector classification row | Makes the maintainer-only status of `SPARSE_SVD_LOWRANK_OUTER` explicit if not promoted. | Documentation/report-index defer row rather than runtime sentinel. | Governance classification only; no memory/performance claim. |

## Day 4 Inputs

- Define "selected backend" as the top-level dispatch result unless a row
  explicitly says "internal completion path."
- Keep LDLT CSC scalar-prepass fallback vocabulary separate from top-level
  fallback to linked-list.
- Decide whether immediate `used_csc_path` publication on later errors is
  public contract wording or diagnostic telemetry.
- State precedence for explicit typed backend requests versus compile-time
  AUTO thresholds.
- State that dense-helper env requests are compatibility/maintainer controls
  unless Day 6 promotes them.
- State that OpenMP build mode and `OMP_NUM_THREADS` are report context, not
  a library-owned runtime policy.
- Keep sentinel candidates threshold-free unless they reuse the existing
  `wall-check` hard gate.

## Day 3 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Backend routing and fallback semantics are documented before changes. | Complete | Dispatch summary, direct-solver, eigensolver, dense-helper, and OpenMP sections. |
| Missing coverage is described with proposed proof owners. | Complete | Coverage map and candidate sentinel expansion list. |
| No implementation change is made before precedence rules are drafted. | Complete | This artifact and working-notes update are documentation-only. |
