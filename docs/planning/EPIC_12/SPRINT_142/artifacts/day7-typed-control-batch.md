# Day 7 Typed-Control Batch

## Purpose

Day 7 implements the Day 6 selected typed-control batch for Sprint 142. The
selected batch is intentionally conservative: do not add public ABI/API surface
where runtime/backend controls are still maintainer diagnostics, compatibility
environment variables, build-time switches, or report context.

## Implementation Result

No public headers, public option structs, public enums, package metadata, or ABI
claims changed on Day 7.

The existing public typed-control surface already covers the high-value
runtime/backend decisions that should be caller-owned today:

- Cholesky backend selection through typed solver options.
- LDLT backend selection through typed solver options.
- Eigensolver backend selection through typed solver options.
- Analysis/reorder strategy selection through typed analysis options.

All other Day 6 candidates remain explicitly deferred until they have stronger
user-facing semantics, portability evidence, and documentation/claim support.
The optional LDLT dense-helper invalid-environment validation was also audited:
`tests/test_ldlt.c` already includes
`test_ldlt_dense_backend_invalid_env_falls_back_to_builtin`, so Day 7 does not
add duplicate C coverage.

## Deferral Ledger

| Control | Sprint 142 Classification | Why Not Promoted | Existing Proof or Owner | Next Owner |
| --- | --- | --- | --- | --- |
| `SPARSE_CHOL_DENSE_BACKEND` | Maintainer compatibility environment variable | Dense helper selection is platform-sensitive and below the public solver-backend contract. | `tests/test_chol_csc_supernodal.c` dense-helper env coverage. | Day 10 maintainer/user documentation boundary. |
| `SPARSE_LDLT_DENSE_BACKEND` | Maintainer compatibility environment variable | Dense helper selection is an implementation detail behind LDLT CSC routing. | `tests/test_ldlt.c` dense-helper default, explicit, external, and invalid-env fallback coverage. | Day 10 maintainer/user documentation boundary. |
| `SPARSE_SVD_LOWRANK_OUTER` | Maintainer runtime experiment environment variable | The low-rank outer strategy is not yet a stable user policy. | `tests/test_svd.c` low-rank behavior coverage. | Future SVD productization if the strategy becomes user-facing. |
| FM strategy/debug/profile variables | Maintainer graph/reorder tuning and diagnostics | Promoting these would create a broad graph-tuning API before the claim surface is stable. | `tests/test_graph.c` and `tests/test_reorder_nd.c`. | Keep maintainer-only unless Epic 12 selects graph API productization. |
| Debug/profile report variables | Maintainer diagnostics | These affect observability/reporting, not mathematical solver policy. | Existing report-generation workflows. | Report-index and documentation owners. |
| `SPARSE_OPENMP` | Build-time feature switch | Runtime thread policy is not library-owned when OpenMP is compiled in or out. | `tests/test_omp.c` and build configuration checks. | Build documentation owner. |
| `OMP_NUM_THREADS` | External runtime context | This is owned by the caller/runtime, not the library API. | Captured as report context where applicable. | Report-index owner. |
| `SPARSE_MUTEX` | Build-time safety switch | This is a compile/link behavior choice, not a runtime backend selector. | Build configuration checks. | Sprint 143 package/link follow-through if needed. |
| `SPARSE_CSC_THRESHOLD` and eigensolver AUTO thresholds | Compile-time tuning constants | Thresholds govern AUTO defaults and are not stable user runtime policy. | Backend dispatch tests and AUTO route tests. | Internal tuning owner. |
| `SPARSE_EIGS_OMP_REORTH_MIN_N` | Compile-time OpenMP tuning constant | It depends on compiled OpenMP availability and workload shape. | Eigensolver/OpenMP tests. | Internal tuning owner. |
| `SPARSE_TEST_*` and `RUN_BENCH` | Test/benchmark workflow controls | These are project workflow toggles, not library behavior contracts. | CI and local test scripts. | CI/documentation owner. |
| CMake/pkg-config/install link behavior | Package contract | Package decisions are install/adoption surface work, not runtime backend governance. | Install and CMake package tests. | Sprint 143 package/link planning. |

## Public Typed Controls Retained

| Surface | Public Control Shape | Day 7 Decision | Proof Owner |
| --- | --- | --- | --- |
| Cholesky backend dispatch | Typed backend option with AUTO/default behavior | Retain as public typed API; no expansion needed. | `tests/test_chol_csc.c` and `tests/test_chol_csc_supernodal.c`. |
| LDLT backend dispatch | Typed backend option with AUTO/default behavior | Retain as public typed API; no expansion needed. | `tests/test_ldlt_backend_dispatch.c` and `tests/test_ldlt.c`. |
| Eigensolver backend dispatch | Typed backend option with AUTO/default behavior | Retain as public typed API; no expansion needed. | `tests/test_eigs_thick_restart.c` and `tests/test_eigs_lobpcg.c`. |
| Analysis/reorder strategy | Typed analysis options with compatibility env fallback | Retain as public typed API; no expansion needed. | `tests/test_reorder_nd.c`. |

## Day 7 Test Decision

Day 7 made documentation/artifact changes only. No C or header files changed,
so the sprint keeps the full `make format && make lint && make test` quality
gate for later code-changing days.

Existing focused tests already cover the selected batch:

- Cholesky default, forced, AUTO, unavailable, and invalid behavior:
  `tests/test_chol_csc.c` and `tests/test_chol_csc_supernodal.c`.
- LDLT default, forced, AUTO, and dispatch behavior:
  `tests/test_ldlt_backend_dispatch.c`.
- LDLT dense-helper default, explicit, external, and invalid-env fallback:
  `tests/test_ldlt.c`.
- Eigensolver AUTO and forced backend behavior:
  `tests/test_eigs_thick_restart.c` and `tests/test_eigs_lobpcg.c`.
- Analysis typed-vs-env precedence:
  `tests/test_reorder_nd.c`.
- Sentinel report row boundary behavior:
  `tests/test_normalize_report_index.py`.

## Day 8 and Day 9 Sentinel Inputs

Day 7 preserves the following sentinel candidates for the upcoming design and
implementation days:

- Dispatch-only Cholesky route snapshots.
- Dispatch-only LDLT route snapshots.
- Eigensolver AUTO backend route snapshots.
- Shift-invert LDLT route snapshots.
- Optional LDLT dense-helper requested/selected/fallback report rows if they
  can remain clearly maintainer-only.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected controls are observable through typed APIs or explicitly deferred. | Complete | Public typed controls retained; remaining candidates recorded in the deferral ledger. |
| Tests cover default and non-default behavior. | Complete | Existing focused test owners are mapped above. |
| C/header changes are ready for full quality gates later in sprint. | Not applicable | Day 7 changed documentation/artifacts only. |
| ABI/package non-claims are preserved. | Complete | No public headers, install metadata, package files, or ABI claims changed. |
