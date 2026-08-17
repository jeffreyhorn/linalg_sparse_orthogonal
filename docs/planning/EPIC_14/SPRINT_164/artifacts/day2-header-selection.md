# Sprint 164 Day 2 Header Candidate Selection

## Purpose

Day 2 selects the bounded public-header cleanup batch for Sprint 164. The
selection uses Day 1 inventory criteria: user impact, documentation ambiguity,
claim risk, option/result complexity, downstream visibility, and feasibility of
declaration-preserving comment cleanup.

## Selection Criteria

| Criterion | Signal Used |
| --- | --- |
| User impact | README/API reference/tutorial/cookbook/solver-selection links and first-use relevance. |
| Documentation ambiguity | Ownership, lifetime, output-buffer, error-status, option/result, callback, handle, and backend wording density. |
| Claim risk | Backend, package, ABI, performance, state-of-the-art, and local-only wording near public API comments. |
| Option/result complexity | Presence of option structs, result structs, handles, callbacks, telemetry, or backend selectors. |
| Downstream visibility | Inclusion in README API overview, `docs/api_reference.md`, tutorial examples, solver-selection docs, or cookbook links. |
| Cleanup feasibility | Expected comment-only cleanup without changing declarations, struct layout, enum values, macros, include guards, or installed header names. |

## Header Signal Inventory

The Day 2 inventory counted lines and matching API-risk terms such as option,
result, backend, handle, callback, owner, free, NULL, error, output, buffer,
package, ABI, runtime-loader, performance, and superiority.

| Header | Lines | Signal Hits | Day 2 Disposition |
| --- | ---: | ---: | --- |
| `include/sparse_iterative.h` | 731 | 273 | Select. Largest user-facing solver surface with options, results, callbacks, handles, matrix-free APIs, and block APIs. |
| `include/sparse_eigs.h` | 612 | 155 | Select. High backend/result/handle complexity and strong README/tutorial/cookbook/API-reference visibility. |
| `include/sparse_matrix.h` | 585 | 97 | Select. Core adoption surface for lifecycle, mutation, compressed construction, I/O, NULL behavior, and ownership. |
| `include/sparse_analysis.h` | 488 | 64 | Defer. Important lifecycle surface, but lower Day 2 priority than matrix plus iterative/eigs and likely deserves a dedicated direct-solver lifecycle batch. |
| `include/sparse_qr.h` | 373 | 69 | Defer. Good candidate, but QR docs recently received bounded corpus/comparison attention and should follow the selected core/iterative/eigs pass. |
| `include/sparse_lu.h` | 360 | 62 | Defer. Direct-solver one-shot wording should be considered with Cholesky/LDLT/analysis rather than mixed into this batch. |
| `include/sparse_types.h` | 324 | 52 | Defer. Shared type/header cleanup has higher ABI/macro risk and should be isolated if selected later. |
| `include/sparse_lu_csr.h` | 322 | 66 | Defer. Specialized CSR LU working-format surface; better suited to a direct/CSR-specific cleanup batch. |
| `include/sparse_ldlt.h` | 315 | 63 | Defer. Backend telemetry and symmetric-indefinite wording are important but should be paired with direct-solver lifecycle docs later. |
| `include/sparse_svd.h` | 243 | 68 | Defer. Partial-SVD evidence wording is sensitive and should follow the current selected batch. |
| `include/sparse_cholesky.h` | 227 | 57 | Defer. Previously received public-header cleanup; revisit with direct-solver batch if needed. |
| `include/sparse_ilu.h` | 200 | 39 | Defer. Preconditioner lifecycle surface, but smaller and less cross-linked than selected iterative/eigs headers. |
| `include/sparse_dense.h` | 197 | 28 | Defer. Helper surface is lower first-use risk. |
| `include/sparse_reorder.h` | 186 | 37 | Defer. Important but narrower; reorder docs should align with direct-solver/analysis cleanup later. |
| `include/sparse_csr.h` | 161 | 38 | Defer. Compressed-storage helper cleanup should be paired with matrix/CSR docs after matrix core pass. |
| `include/sparse_ic.h` | 121 | 23 | Defer. Smaller preconditioner surface. |
| `include/sparse_bidiag.h` | 72 | 8 | Defer. Low current cleanup risk. |
| `include/sparse_vector.h` | 70 | 14 | Defer. Low current cleanup risk. |
| `include/sparse_version.h.in` | 25 | 0 | Defer. Generated installed version-header behavior remains package/install owned. |

## Selected Header Batch

Sprint 164 selects this declaration-preserving cleanup batch:

1. `include/sparse_iterative.h`
2. `include/sparse_eigs.h`
3. `include/sparse_matrix.h`

This batch is large enough to close meaningful API usability gaps but still
bounded around the highest-visibility public caller surfaces:

- first matrix construction and lifecycle;
- iterative solver options/results, callbacks, handles, and matrix-free APIs;
- eigensolver backend, result, handle, shift-invert, and buffer semantics.

## Source-Backed Rationale

### `include/sparse_iterative.h`

- Highest line count and signal density: 731 lines and 273 API-risk term hits.
- Referenced by tutorial examples, README API overview, README repeated-run
  handle section, solver-selection docs, and API reference.
- Contains multiple option/result structs, callbacks, progress cancellation,
  residual history, preconditioner callbacks, repeated-run handles,
  matrix-free APIs, and block-solver APIs.
- Cleanup target: clarify ownership, callback/output-buffer semantics,
  default/null option behavior, result-field interpretation, handle lifecycle,
  and non-claim boundaries without changing declarations.

### `include/sparse_eigs.h`

- Second-highest complexity: 612 lines and 155 API-risk term hits.
- Referenced by README capability overview, README API overview, tutorial,
  cookbook, solver-selection docs, and API reference.
- Contains backend selectors, AUTO routing thresholds, shift-invert behavior,
  result fields, caller-owned eigenvalue/eigenvector buffers, and repeated-run
  handles.
- Cleanup target: clarify caller-owned buffers, backend selection/fallback
  wording, result interpretation, handle lifecycle, and performance/backend
  non-claims without changing declarations.

### `include/sparse_matrix.h`

- Core first-use surface: 585 lines and 97 API-risk term hits.
- Referenced by README first-use examples, tutorial, solver-selection docs,
  API reference, and compressed-first adoption routes.
- Contains matrix lifecycle, `NULL` and silent-zero behavior, mutation,
  compressed import/export ownership, Matrix Market I/O, errno behavior, and
  backend threshold context.
- Cleanup target: clarify matrix ownership, caller-owned compressed arrays,
  `NULL`/silent-zero contracts, output ownership, mutation behavior, and
  compressed construction errors without changing declarations.

## Header-To-Documentation Map

| Selected Header | Related Docs To Check |
| --- | --- |
| `include/sparse_iterative.h` | `README.md` repeated-run and API overview sections; `docs/api_reference.md`; `docs/tutorial.md` iterative examples; `docs/solver_selection.md` iterative solver and diagnostics sections; `docs/cookbook.md` first-use workflow where relevant; `docs/maintainer_guide.md` public-header policy. |
| `include/sparse_eigs.h` | `README.md` eigensolver capability, backend, and API overview sections; `docs/api_reference.md`; `docs/tutorial.md` eigensolver walkthrough; `docs/cookbook.md` eigensolver path; `docs/solver_selection.md` eigensolver section; `docs/maintainer_guide.md` evidence and generated-reference policy. |
| `include/sparse_matrix.h` | `README.md` first-use and API overview sections; `docs/api_reference.md`; `docs/tutorial.md` matrix creation examples; `docs/cookbook.md` CSR/CSC/Matrix Market first-use routes; `docs/solver_selection.md` data-entry and diagnostics sections; `docs/maintainer_guide.md` public-header policy. |

## Deferred Queue

| Deferred Header | Reason |
| --- | --- |
| `include/sparse_analysis.h` | Direct lifecycle surface is important but should be cleaned with LU/Cholesky/LDLT direct-solver docs. |
| `include/sparse_lu.h` | One-shot direct-solver cleanup should be coordinated with analysis, Cholesky, and LDLT. |
| `include/sparse_ldlt.h` | Backend telemetry and symmetric-indefinite semantics deserve a focused direct/backend batch. |
| `include/sparse_cholesky.h` | Already received prior public-header cleanup; revisit only with the direct-solver batch. |
| `include/sparse_qr.h` | QR has sensitive corpus/comparison evidence boundaries; defer until after core/iterative/eigs cleanup. |
| `include/sparse_svd.h` | Partial-SVD evidence wording is sensitive and should not be mixed into the first Sprint 164 batch. |
| `include/sparse_types.h` | Shared type, enum, macro, scalar/index, and error-code wording has higher ABI/macro risk. |
| `include/sparse_lu_csr.h` | Specialized working-format surface; better paired with CSR/direct internals. |
| `include/sparse_csr.h` | Compressed storage helper should be revisited after matrix ownership wording settles. |
| `include/sparse_ilu.h` | Smaller preconditioner lifecycle surface; defer behind iterative/eigs core pass. |
| `include/sparse_ic.h` | Smaller preconditioner lifecycle surface; defer behind iterative/eigs core pass. |
| `include/sparse_reorder.h` | Reordering cleanup should align with analysis/direct solver docs. |
| `include/sparse_dense.h` | Lower first-use risk helper surface. |
| `include/sparse_bidiag.h` | Low current cleanup risk and narrow helper surface. |
| `include/sparse_vector.h` | Low current cleanup risk and narrow helper surface. |
| `include/sparse_version.h.in` | Generated installed version-header behavior remains install/package owned. |

## Cleanup Risk Register

| Risk | Control |
| --- | --- |
| Selected batch is too large for declaration-preserving review. | Limit Day 5-Day 8 edits to comment cleanup in three headers only; defer broad rewrites. |
| Public declarations drift. | Day 3 and Day 4 must define/capture declaration baseline before header edits; Day 10 must recapture. |
| Header prose becomes tutorial or maintainer policy. | Keep headers API-local and link to README/tutorial/cookbook/solver-selection/maintainer docs when broader explanation is needed. |
| Backend wording implies superiority. | Preserve Sprint 163 non-superiority language; describe backend behavior and selection only. |
| Eigensolver wording implies portable performance. | Keep AUTO/backend threshold wording as implementation behavior, not performance proof. |
| Matrix lifecycle wording implies ABI/package guarantees. | Keep ownership/lifetime wording separate from package, ABI, runtime-loader, and installed-header claims. |

## Day 3 Handoff

Day 3 should design the declaration baseline for exactly:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_matrix.h`

The baseline should detect unexpected changes to:

- function declarations;
- typedef names;
- enum names and values;
- struct field order and field types;
- macro definitions;
- include guards;
- installed header names.

## Validation Notes

Day 2 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 2.

## Completion Check

- The selected batch is bounded to three high-impact public headers.
- Every selected header has source-backed rationale.
- Deferred headers have explicit reasons.
- The Day 3 declaration-baseline target set is ready.
