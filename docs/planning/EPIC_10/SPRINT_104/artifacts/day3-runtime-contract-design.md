# Sprint 104 Day 3 Runtime Contract Design

## Purpose

Day 3 converts the Day 2 backend consumer audit into a runtime contract for
builtin kernels, optional acceleration, OpenMP behavior, nested parallelism,
and observability. The contract is intentionally conservative: it preserves
the builtin implementation as the portable product truth and treats optional
acceleration and local timing as bounded context, not portable superiority.

## Contract Summary

| area | contract |
|---|---|
| portable baseline | builtin kernels are the default and must remain sufficient for supported builds |
| optional acceleration | optional dense backends may be requested through existing env controls, but unavailable or unsuitable providers fall back to builtin unless a test-only override deliberately forces an error path |
| public selectors | Cholesky, LDLT, and eigensolver backend enums select library algorithms, not universal vendor providers |
| OpenMP | OpenMP is compile-time optional; serial behavior remains the default and the reference behavior |
| nested parallelism | no broad nested-parallel performance contract is claimed; benchmark/sentinel artifacts must disclose thread and backend context |
| observability | public result fields, benchmark CSV fields, tests, and docs have different roles and must not be merged into one unsupported product claim |

## Builtin Dense Kernel Baseline

The builtin dense kernels are the portable baseline for Sprint 104.

| surface | builtin baseline | required behavior |
|---|---|---|
| Cholesky CSC dense kernels | `chol_dense_factor`, `chol_dense_solve_lower`, `chol_dense_solve_panel` through `chol_dense_kernels_t{name="builtin"}` | always available in supported builds; selected when no optional provider is requested or usable |
| LDLT CSC dense factor | `ldlt_dense_factor(...)` through `ldlt_dense_factor_selected(...)` | always available in supported builds; selected when no optional provider is requested or usable |
| QR dense helper path | internal Householder/dense buffers in `src/sparse_qr.c` | builtin only; no Sprint 104 optional backend claim |
| SVD dense/helper path | builtin SVD and low-rank helpers in `src/sparse_svd*.c` | builtin only; `SPARSE_SVD_LOWRANK_OUTER` is a path toggle, not a vendor backend |
| Eigensolver kernels | grow-m Lanczos, thick-restart Lanczos, and LOBPCG | builtin concrete algorithms; no vendor eigensolver backend claim |

Design rule: future descriptor or status changes must not make optional
acceleration look required for correctness, installability, or supported use.

## Optional Backend Selection Semantics

| selector | accepted values | unavailable or unsuitable provider behavior | error behavior |
|---|---|---|---|
| `SPARSE_CHOL_DENSE_BACKEND` | empty/unset, `builtin`, `external`, `blas`, `lapack`, Apple-only `accelerate` | falls back to `builtin` when the requested provider is unavailable, unknown, unsupported on the platform, or probe fails | normal calls should continue through builtin; test-only descriptor override can return `NULL` and exercise `SPARSE_ERR_BACKEND_CONTRACT` |
| `SPARSE_LDLT_DENSE_BACKEND` | empty/unset, `builtin`, `external`, `blas`, `lapack`, `accelerate` where supported/probed | falls back to `builtin` when provider probe fails or pivot-pattern constraints make the external path unsuitable | invalid/unavailable provider is not a user-facing hard failure under current behavior; factorization errors still use existing numeric/error contracts |
| `sparse_cholesky_opts_t::backend` | AUTO, LINKED_LIST, CSC | not an optional vendor selector; AUTO chooses library path by threshold | invalid enum returns `SPARSE_ERR_BADARG`; selected CSC internals may still use builtin dense kernels |
| `sparse_ldlt_opts_t::backend` | AUTO, LINKED_LIST, CSC | not an optional vendor selector; AUTO chooses library path by threshold, with documented empty-matrix linked-list edge case | invalid enum returns `SPARSE_ERR_BADARG`; numeric failures use existing LDLT errors |
| `sparse_eigs_opts_t::backend` | AUTO, LANCZOS, LOBPCG, LANCZOS_THICK_RESTART | not an optional vendor selector; AUTO chooses builtin algorithm by preconditioner/block/size thresholds | invalid enum returns `SPARSE_ERR_BADARG`; `backend_used` is authoritative after successful calls |

### Fallback Policy

Silent fallback to builtin is the current product behavior for optional dense
backend requests. Sprint 104 should treat that as deliberate unless Day 4/5
adds stronger status telemetry. The minimum reporting requirement is:

- tests that force optional backend env values must allow builtin fallback when
  the provider is unavailable;
- benchmarks that interpret timing must report requested and selected backend
  where that surface already has fields;
- docs must state that optional requests are best-effort and self-contained
  builtin remains the portable baseline.

## OpenMP and Threading Contract

| build/runtime surface | contract |
|---|---|
| default builds | serial behavior is the default and remains the reference behavior |
| `SPARSE_OPENMP` builds | OpenMP may parallelize SpMV/block SpMV and Lanczos MGS inner loops |
| `OMP_NUM_THREADS` and runtime settings | controlled by the OpenMP runtime; benchmark and sentinel artifacts must disclose thread settings when timing is interpreted |
| small eigensolver MGS workloads | `SPARSE_EIGS_OMP_REORTH_MIN_N` gates OpenMP MGS work to avoid small-problem overhead |
| thread safety | independent matrices/solves have test coverage; same-matrix mutation requires external synchronization or `SPARSE_MUTEX` where documented |
| nested parallelism | no guarantee of speedup or deterministic scheduling; future sentinels must avoid overstating nested OpenMP or optional BLAS interactions |
| TSan/OpenMP | CI has a bounded OpenMP TSan lane for eigensolver paths with suppressions for runtime internals; it is not proof of all OpenMP/runtime combinations |

### Nested Parallelism Non-Contract

Sprint 104 does not claim:

- nested OpenMP regions are beneficial;
- optional BLAS/LAPACK providers use the same thread count as the project
  OpenMP runtime;
- OpenMP plus optional dense backend timing is portable across machines;
- benchmark timing with one thread setting generalizes to all settings.

Day 6 should decide whether any runtime-control cleanup can reduce ambiguity
without changing public semantics.

## Runtime State Observability

| state | primary visibility | role |
|---|---|---|
| Cholesky linked-list vs CSC selector | public opts enum and tests | public algorithm/path choice |
| Cholesky dense-kernel descriptor name | internal accessor; `bench_chol_csc` CSV field | benchmark/test diagnostic for active dense-kernel seam |
| LDLT linked-list vs CSC selector | public opts enum and optional `used_csc_path` | public algorithm/path choice and result telemetry |
| LDLT dense backend selected | `ldlt_dense_factor_backend_name()` and benchmark fields | diagnostic/benchmark context, not public vendor parity |
| Eigensolver concrete backend | `sparse_eigs_t::backend_used` | public result telemetry after successful calls |
| Eigensolver shift-invert LDLT path | `sparse_eigs_t::used_csc_path_ldlt` | result telemetry for internal LDLT path choice |
| OpenMP enabled/max threads | `test_omp_status`, `bench_main` banner, build flags | validation and benchmark context |
| Graph/ND runtime controls | typed analysis fields, compatibility env vars, thread-local test/bench hooks | separate reordering runtime controls; not part of dense backend descriptor |
| Benchmark residuals | benchmark CSV/report fields | context only unless a test/oracle owns the correctness claim |

## Diagnostics by Audience

| audience | appropriate diagnostics | inappropriate interpretation |
|---|---|---|
| public API user | public selector enums, result telemetry fields, documented env compatibility where relevant | vendor backend portability or broad performance claims |
| test owner | focused status fields, error codes, forced selectors, test-only overrides | benchmark superiority claims |
| benchmark owner | selected backend, fallback, thread count, fixture, timing, residual context | correctness ownership or portable speedup |
| maintainer | env controls, probe/fallback behavior, CI lane limits, residual queue | release claim unless validated and documented elsewhere |

## Determinism Requirements for Tests and Benchmarks

| requirement | rationale |
|---|---|
| builtin fallback must be deterministic when env vars are unset | enables stable default tests and CI |
| optional-backend tests must clean up env state | env controls are process-global |
| benchmark reports must include selected backend if optional backend timing is interpreted | prevents mistaking fallback timing for accelerated timing |
| OpenMP timing artifacts must disclose thread settings | thread count is runtime-controlled |
| `backend_used` and `used_csc_path` fields should be treated as telemetry for successful calls | error returns may be partial or best-effort |

## Day 4/5 Descriptor Boundary Inputs

Day 4 should decide whether to implement any of these:

| candidate | reason | risk |
|---|---|---|
| align Cholesky and LDLT dense-backend terminology around request/selected/fallback | benchmark docs already use this model for LDLT; Cholesky has descriptor name but no explicit fallback field | may widen internal API or benchmark output unnecessarily |
| document silent fallback without source changes | low-risk way to make current behavior explicit | may leave users without programmatic selected-backend status outside benchmark/test internals |
| add a small internal helper for normalized dense-backend request names | could reduce duplicated benchmark/env wording | source-touch requires full C quality gate |
| leave graph/ND env controls out of dense backend descriptor work | prevents Sprint 104 from absorbing reordering runtime policy | Day 6 still needs to audit threading/runtime cleanup separately |

## Explicit Non-Claims

Sprint 104 Day 3 does not claim:

- optional acceleration is available on every platform;
- optional acceleration is faster than builtin;
- fallback to builtin is an error;
- OpenMP improves every workload;
- nested parallelism is tuned or recommended;
- benchmark residual columns are correctness or oracle proof;
- Cholesky/LDLT dense backend seams imply broad vendor backend parity;
- QR, SVD, or eigensolver paths have optional vendor backend support.

## Completion Check

| criterion | status |
|---|---|
| builtin dense kernel baseline defined | complete |
| optional backend selection and fallback behavior defined | complete |
| OpenMP and nested parallelism expectations defined | complete |
| observability split across public/test/benchmark/docs defined | complete |
| local timing non-claims preserved | complete |
| Day 4 descriptor-boundary questions identified | complete |
