# Sprint 164 Day 1 Sprint Intake And API Surface Inventory

## Purpose

Day 1 establishes the Sprint 164 API/header cleanup boundary before any public
header edits occur. The goal is to inventory the current public API surfaces,
record the source-plan path mismatch, and carry forward the generated-reference,
quality-gate, and performance non-claim handoffs from earlier sprints.

## Source Plan

The current authoritative Sprint 164 source is
`docs/planning/EPIC_14/PROJECT_PLAN.md`, section "Sprint 164: Public Header And
API Coherence Batch". The prompt referenced the older Epic 12 project-plan
path, which does not contain the current Sprint 164 section.

## Handoff Inputs Reviewed

| Input | Relevant Sprint 164 Rule |
| --- | --- |
| Sprint 158 generated API docs policy | Generated Doxygen HTML remains local-only and ignored; `make docs-check` owns local freshness/page coverage. |
| Sprint 157 quality surface map | Public header changes, including comment-only edits, require declaration-preservation proof and `make format && make lint && make test`. |
| Sprint 163 API-header handoff | API/header docs must not use local benchmark/sentinel rows as package, ABI, runtime-loader, backend-superiority, hosted, broad platform, portable performance, or state-of-the-art proof. |
| `docs/maintainer_guide.md` public-header policy | Headers should keep concise API-local caveats and avoid maintainer-policy expansion. |
| `docs/api_reference.md` | Checked-in public headers remain source of truth for declarations and call-site contracts. |

## Public Header Inventory

| Header | Lines | Cleanup Signals |
| --- | ---: | --- |
| `include/sparse_analysis.h` | 488 | Analyze/factor/refactor lifecycle, factor ownership, same-pattern behavior, direct-solver reuse. |
| `include/sparse_bidiag.h` | 72 | Small preprocessing surface; likely lower cleanup risk. |
| `include/sparse_cholesky.h` | 227 | Backend, telemetry, SPD failure, local mutation, and cancellation wording. |
| `include/sparse_csr.h` | 161 | Caller-owned compressed arrays, import/export ownership, compressed-first construction. |
| `include/sparse_dense.h` | 197 | Dense helper output-buffer and result-shape expectations. |
| `include/sparse_eigs.h` | 612 | Backend selection, handle lifecycle, result ownership, shift-invert/preconditioner behavior. |
| `include/sparse_ic.h` | 121 | Factor lifecycle and preconditioner callback behavior. |
| `include/sparse_ilu.h` | 200 | ILU/ILUT options, factor lifecycle, callback behavior, failure modes. |
| `include/sparse_iterative.h` | 731 | Largest public header; iterative options/results, handles, matrix-free callbacks, block solves. |
| `include/sparse_ldlt.h` | 315 | Backend selection, factor ownership, telemetry, symmetric-indefinite failure behavior. |
| `include/sparse_lu.h` | 360 | LU options/results, factor lifecycle, solve/refinement output behavior. |
| `include/sparse_lu_csr.h` | 322 | CSR LU working-format ownership and dense-block/scatter-gather boundaries. |
| `include/sparse_matrix.h` | 585 | Matrix lifecycle, mutation, compressed import/export, Matrix Market I/O, errno. |
| `include/sparse_qr.h` | 373 | QR options/results, least-squares output, rank/nullspace/minimum-norm contracts. |
| `include/sparse_reorder.h` | 186 | Reordering options, permutation ownership, bandwidth/fill behavior. |
| `include/sparse_svd.h` | 243 | SVD/partial-SVD result ownership, low-rank output, convergence and non-claim boundaries. |
| `include/sparse_types.h` | 324 | Shared public types, error codes, scalar/index configuration, enums. |
| `include/sparse_vector.h` | 70 | Small vector-helper surface; likely lower cleanup risk. |
| `include/sparse_version.h.in` | 25 | Generated installed version-header template, owned by install/version validation rather than Doxygen input. |

## API Documentation Surface Inventory

| Surface | Role | Day 1 Finding |
| --- | --- | --- |
| `README.md` | Project front door and API overview table. | Contains public routing to API reference, cookbook, tutorial, and ownership docs. |
| `docs/api_reference.md` | Compact declaration index and generated HTML policy. | Explicitly states checked-in headers are source of truth and generated HTML is local-only. |
| `docs/tutorial.md` | Fuller learning path. | Routes exact declarations and ownership contracts to API reference and headers. |
| `docs/cookbook.md` | First-use workflow recipes. | Should be checked for stale function names or ownership guidance after header cleanup. |
| `docs/solver_selection.md` | Solver-family and escalation guide. | Carries backend/performance non-claim language that selected header edits must preserve. |
| `docs/maintainer_guide.md` | Repository-wide policy owner. | Owns public-header cleanup policy and generated API docs policy. |
| `Makefile` docs targets | Local generated API validation. | Provides `make docs`, `make api-docs-coverage`, and `make docs-check`. |
| `Doxyfile` | Doxygen configured input. | Relevant when selected headers are changed or generated API docs are checked. |

## Evidence Boundaries

Sprint 164 header cleanup can support:

- clearer call-site ownership, lifetime, error, output-buffer, option/result,
  backend, and solver-selection wording;
- declaration-preservation evidence for selected public headers;
- generated-reference policy alignment for the changed header batch;
- public documentation coherence across README, tutorial, cookbook,
  solver-selection, API reference, and maintainer guide.

Sprint 164 header cleanup cannot by itself support:

- dynamic ABI compatibility;
- shared-library support;
- runtime-loader behavior;
- package-manager distribution;
- package/install proof;
- broad platform parity;
- backend superiority;
- external-library parity;
- portable performance;
- state-of-the-art coverage;
- hosted generated API HTML publication.

## Initial Risk Register

| Risk | Control |
| --- | --- |
| Comment-only edits accidentally change declarations. | Capture normalized before/after declarations and run full C/header gate for any header edits. |
| Public headers accumulate maintainer history. | Keep broad policy in `docs/maintainer_guide.md`; headers keep API-local caveats only. |
| Header cleanup implies ABI or package support. | Scan public wording for ABI, shared-library, runtime-loader, package-manager, and platform claims. |
| Backend wording implies superiority. | Preserve Sprint 163 non-superiority boundary and keep backend wording behavioral. |
| Generated HTML is mistaken for committed or hosted API docs. | Keep `docs/api/html/` ignored and rely on `make docs-check` for local validation. |
| Sprint 163 performance rows are reused as API proof. | Cite them only as local methodology-bound performance rows when necessary; never as API/package/ABI proof. |

## Day 2 Handoff

Day 2 should select the bounded public-header batch using:

- user impact;
- ambiguity in ownership/lifetime, errors, output buffers, options/results, or
  backend behavior;
- documentation cross-link risk;
- declaration-preservation feasibility;
- avoidance of signature or struct-layout changes.

High-signal candidate headers for Day 2 ranking are:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `include/sparse_qr.h`
- `include/sparse_lu.h`
- `include/sparse_ldlt.h`
- `include/sparse_lu_csr.h`
- `include/sparse_types.h`

This is an intake candidate list only, not the selected cleanup batch.

## Validation Notes

Day 1 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 1.

## Completion Check

- Sprint 164 scope is tied to the Epic 14 project plan.
- Public header and API documentation owners are identified.
- Cleanup work is separated from package, ABI, generated-hosting, and
  performance proof.
