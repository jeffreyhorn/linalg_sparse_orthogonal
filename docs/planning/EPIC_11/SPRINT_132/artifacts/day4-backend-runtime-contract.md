# Sprint 132 Day 4 - Backend Runtime Contract

## Purpose

Define the report vocabulary and claim boundaries for dense backend selection,
fallback, OpenMP build mode, thread-count context, and nested-runtime behavior.

This is a governance artifact. It does not change backend dispatch,
benchmarking code, sentinel scripts, OpenMP scheduling, or public API shape.

## Runtime Surface Inventory

| Surface | Current control or field | Report visibility | Sprint 132 interpretation |
| --- | --- | --- | --- |
| Cholesky CSC dense-kernel seam | `SPARSE_CHOL_DENSE_BACKEND`, `chol_csc_supernodal_dense_kernels()` | `bench_chol_csc` columns `csc_supernodal_dense_kernel` and `csc_supernodal_panel_solver`; S2 sentinel row notes | Bounded Cholesky CSC supernodal observability only. The default shipped descriptor is `builtin`; optional backend selection remains local to this lane. |
| LDLT CSC dense-factor seam | `SPARSE_LDLT_DENSE_BACKEND`, internal LDLT dense backend selector | `bench_refactor_csc --indefinite-kkt` columns `ldlt_dense_backend_request`, `ldlt_dense_backend_selected`, and `ldlt_dense_backend_fallback` | Bounded direct-family LDLT runtime observability only. It does not widen backend claims beyond the retained KKT/repeated-run measurement surface. |
| OpenMP build mode | `SPARSE_OPENMP`, `make omp`, Makefile OpenMP flags | Sentinel `build_mode`, benchmark-local headers, OpenMP-specific test output when invoked | Compile-time build context, not a runtime policy object. Serial remains the default product path. |
| OpenMP runtime thread context | `OMP_NUM_THREADS` and vendor OpenMP runtime settings | Sentinel `omp_num_threads`; optional benchmark-local notes where explicitly emitted | Runtime-owned context. The library does not expose a public thread pool, per-call thread limit, or `sparse_set_num_threads` API. |
| OpenMP implementation owners | SpMV row-parallel loops in `src/sparse_matrix.c`; eigensolver reorthogonalization paths in `src/sparse_eigs.c` | Indirect through benchmarks that exercise SpMV, eigs, solvers, SVD, or graph paths | Do not treat indirect OpenMP reachability as proof of each caller's scalability or nested-runtime safety. |
| Benchmark and report provenance | git branch/commit, platform, compiler, build mode, command, fixture | Canonical, sentinel, guardrail, and benchmark-local manifests or rows | Required context for local comparison. Missing provenance weakens report interpretability and blocks hard threshold promotion. |
| Large-matrix guardrail supplemental mode | `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL` | Guardrail manifest and supplemental rows | Runtime opt-in context for supplemental reports, not a dense backend seam and not a reviewed recurring gate. |

## Backend State Vocabulary

| State | Meaning | Report rule | Claim boundary |
| --- | --- | --- | --- |
| `builtin` | The self-contained in-repo dense backend or kernel descriptor is selected. | Record as the selected backend or dense-kernel descriptor. | Proves only that the builtin path was selected for the reported row. |
| `optional-requested` | An environment variable requested an optional backend such as `accelerate` or `external`. | Preserve the normalized request separately from the selected backend. | A request does not prove that the backend was available, linked, selected, or faster. |
| `optional-selected` | The optional backend was selected by the applicable bounded selector. | Record the selected backend and keep row scope tied to the owning benchmark or sentinel. | Does not imply parity with builtin or portability to another host. |
| `fallback-to-builtin` | A requested optional, external, or invalid backend resolved to builtin. | Record fallback explicitly when the surface has a fallback field; otherwise include a note. | Proves only that the local row used the safe builtin path after resolution. |
| `unavailable` | The optional backend could not be linked, probed, loaded, or exercised in the current build/runtime. | Emit `unavailable`, `skip`, or fallback context instead of silently omitting the row. | Not a test failure unless the lane explicitly requires that optional backend. |
| `unknown` | Required backend metadata is missing from the row or artifact. | Treat the row as non-comparable for backend-specific interpretation. | Blocks backend-specific hard thresholds and before/after conclusions. |
| `not-applicable` | The benchmark or guardrail path has no dense backend seam. | Use `n/a` or omit backend-specific columns only when the report family documents that choice. | Does not say anything about backend availability elsewhere. |
| `unsupported` | The request is outside the supported selector vocabulary or the lane does not support backend selection. | Prefer explicit skip/error/fallback wording over implicit success. | Does not create a public tuning contract for unsupported values. |

## Fallback Policy

Backend fallback means the applicable bounded runtime selector chose a safe
supported path when a requested optional backend was unavailable, unsupported,
invalid, or declined by probe logic.

Reports that make any backend-sensitive comparison must keep these facts
separate:

- requested backend
- selected backend
- fallback state
- dense-kernel descriptor when available
- panel-solver capability when available
- build mode and OpenMP thread context

Fallback does not claim:

- optional backend correctness was exercised
- builtin and optional backends have equivalent performance
- optional backends are portable across hosts or CI images
- an optional backend failure occurred unless the selector or report says so
- the public API exposes backend portability guarantees
- the benchmark is a cross-platform performance proof

Hard timing thresholds must be disabled, deferred, or stratified when fallback
state can vary unless the accepted baseline is tied to the exact same backend,
build mode, OpenMP thread context, fixture, command, repeat count, and host
class.

## OpenMP and Thread-Count Boundary

`SPARSE_OPENMP` is a compile-time build option. It should be reported as build
context, not as a runtime policy object.

`OMP_NUM_THREADS` is process/runtime context. The library may report it when a
benchmark or sentinel records the environment, but it must not be described as
a library-owned per-call thread limit.

Nested runtime behavior, affinity, team sizing, vendor OpenMP configuration,
and oversubscription remain owned by the OpenMP runtime and caller process.
Sprint 132 must not add or imply public control over those settings.

Current OpenMP ownership remains narrow:

- SpMV row-parallel loops are the primary matrix path.
- Eigensolver reorthogonalization paths have their own guarded OpenMP use.
- Solvers, SVD, and graph workflows may reach OpenMP indirectly through SpMV
  or eigensolver calls.
- New outer OpenMP regions require a separate nested-runtime and
  oversubscription validation plan.

## Observability Field List

Use the smallest field set that makes each report family interpretable. Fields
marked "where applicable" are required only when the lane owns that concept.

| Field | Requirement | Purpose |
| --- | --- | --- |
| `report_family` | Required for indexed reports | Distinguish canonical, sentinel, guardrail, and benchmark-local evidence. |
| `command` | Required | Reconstruct the measured or validated lane. |
| `artifact` | Required for indexed reports | Point to the CSV, TSV, manifest, or raw output file. |
| `generated_at_utc` | Required for generated reports | Support freshness and stale-report interpretation. |
| `git_commit` | Required when locally available | Bind evidence to source state. |
| `git_branch` | Required when locally available | Support branch-local comparison. |
| `platform` | Required | Preserve host context for local timing rows. |
| `compiler` | Required | Preserve toolchain context. |
| `build_mode` | Required for runtime reports | Distinguish serial and OpenMP builds. |
| `omp_num_threads` | Required for runtime reports | Preserve runtime thread context without turning it into API policy. |
| `backend_request` | Where applicable | Preserve normalized dense backend request. |
| `backend_selected` | Where applicable | Preserve selected dense backend or `n/a`. |
| `backend_fallback` | Where applicable | Preserve fallback truthfulness. |
| `dense_kernel` | Where applicable | Preserve active dense-kernel descriptor. |
| `panel_solver` | Where applicable | Preserve Cholesky supernodal panel capability. |
| `support_tier` | Required for promoted indexed rows | Distinguish reviewed, supplemental, experimental, deferred, and generated evidence. |
| `metric_name` | Required for metric rows | Identify the measured value. |
| `metric_value` | Required for metric rows | Preserve the local observed value. |
| `baseline` | Required only for thresholded rows | Identify comparison baseline. |
| `threshold` | Required only for thresholded rows | Identify the local pass/fail threshold. |
| `claim_boundary` | Required for promoted rows | Prevent local timing or backend metadata from becoming broader claims. |
| `freshness` | Required for indexed generated reports | Preserve fresh, stale, missing, or regenerated status. |

## Report-Family Application

| Report family | Required runtime treatment | Hard-threshold eligibility |
| --- | --- | --- |
| `performance-sentinels` | Keep branch, commit, platform, compiler, build mode, `OMP_NUM_THREADS`, `SPARSE_CHOL_DENSE_BACKEND`, and `SPARSE_LDLT_DENSE_BACKEND`; preserve Cholesky dense-kernel and panel-solver notes for S2. | Only existing S5 wall-check remains hard-gated. Backend-sensitive rows stay threshold-free unless a future baseline is tied to exact runtime state. |
| `bench-canonical-report` | Keep command/artifact/index provenance and preserve backend columns emitted by canonical direct benchmarks. | Threshold-free maintained snapshot. Missing backend metadata blocks backend-specific comparison. |
| `large-matrix-guardrails` | Keep platform, compiler, generated time, supplemental mode, and lane support tier. Treat supplemental mode as opt-in. | Reviewed structural lanes may gate through tests; supplemental timing/fill reports remain non-gating unless promoted later. |
| `benchmark-local` | Record the command, fixture, repeat count, backend context, build mode, and thread context in raw output or notes when used for sprint evidence. | Not threshold eligible until promoted into a report family with owner, baseline, variance, and non-claim policy. |

## Backend Non-Claim Register

Backend/runtime observability must not be used to claim:

- backend parity between builtin, Accelerate, BLAS/LAPACK, external, or future
  optional backends
- optional backend availability on hosts where it was not selected and
  reported
- portable speedup, scalability, memory behavior, or state-of-the-art
  performance
- OpenMP speedup from the presence of `SPARSE_OPENMP`
- library-owned thread-count control from `OMP_NUM_THREADS`
- nested-runtime safety for callers that compose their own parallel regions
- correctness solely from benchmark metadata
- QR, SVD, eigensolver, or broader dense-library backend coverage beyond the
  surfaces that explicitly own it

## Day 5 Handoff

Day 5 should turn this contract into a field-by-field metadata design for the
report families most likely to be touched later in Sprint 132:

- sentinel manifest and TSV fields
- canonical report index and manifest fields
- benchmark-local CSV field expectations for direct/backend rows
- guardrail index fields and supplemental-state handling
- promotion blockers for rows that still report `unknown`, `unavailable`, or
  unstated fallback state

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Builtin and optional backend states are distinguishable. | Complete | Backend state vocabulary separates request, selection, fallback, unavailable, unknown, unsupported, and `n/a` states. |
| OpenMP and nested-runtime boundaries are explicit. | Complete | OpenMP is limited to compile-time build mode plus runtime-owned process context; no public thread-control API is implied. |
| Backend observability does not imply portable performance or backend parity. | Complete | Fallback policy and non-claim register preserve local-only interpretation. |
