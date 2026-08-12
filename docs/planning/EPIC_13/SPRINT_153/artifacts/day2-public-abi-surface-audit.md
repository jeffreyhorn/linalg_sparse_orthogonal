# Sprint 153 Day 2 Public ABI Surface Audit

## Purpose

Day 2 inventories the installed public API surface that would become
ABI-relevant under shared-library support. The audit separates maintained
static-first package evidence from any future dynamic ABI claim, and makes the
accidental export risks visible before the Day 3 loader audit.

## Installed Public Header Inventory

The install surface remains `18` source headers plus generated
`sparse_version.h`, for `19` installed public headers.

| Header | Classification | ABI Relevance | Owner Candidate | Notes |
| --- | --- | --- | --- | --- |
| `sparse_types.h` | ABI foundation | High | Core API | Defines `idx_t`, scalar width, error codes, backend and reorder enums, progress callback shape, and error/version utility declarations. |
| `sparse_matrix.h` | Core matrix API | High | Matrix core | Uses an opaque `SparseMatrix` handle, but lifecycle, construction, query, IO, arithmetic, and solver-entry declarations are ABI-relevant. |
| `sparse_csr.h` | Public format layout | High | Format layer | Exposes `SparseCsr` and `SparseCsc` data layouts and conversion/free functions. |
| `sparse_lu.h` | Direct solver API | High | Direct solvers | Exposes LU option layout, factorization, solve, determinant, inverse, and free APIs. |
| `sparse_lu_csr.h` | CSR LU layout/API | High | Direct solvers | Exposes `LuCsr` and dense-block layouts. Several functions use `lu_csr_*` names rather than the `sparse_*` namespace. |
| `sparse_cholesky.h` | Direct solver API | High | Direct solvers | Exposes Cholesky type/options and factorization/solve/free APIs. |
| `sparse_ldlt.h` | Direct solver layout/API | High | Direct solvers | Exposes LDLT factor layout, backend selection, diagnostics, and solve/free APIs. |
| `sparse_qr.h` | QR layout/API | High | QR owner | Exposes QR options, rank information, factor layout, least-squares solve, min-norm solve, diagnostics, and free APIs. |
| `sparse_svd.h` | SVD layout/API | High | SVD owner | Exposes SVD options/result layouts, full and partial SVD entry points, low-rank helpers, and free APIs. |
| `sparse_bidiag.h` | Bidiagonalization layout/API | Medium-High | SVD/QR owner | Exposes bidiagonal factor layout and free API. |
| `sparse_iterative.h` | Iterative solver API | High | Iterative solvers | Exposes callback typedefs, option/result/handle layouts, matrix-free solve APIs, and handle lifecycle. |
| `sparse_eigs.h` | Eigen solver API | High | Eigen solvers | Exposes method/selection/status enums, options/result/handle layouts, and solve/free APIs. |
| `sparse_ilu.h` | Preconditioner layout/API | High | Preconditioners | Exposes ILU and ILUT option/factor layouts plus factorize/apply/free APIs. |
| `sparse_ic.h` | Preconditioner API | High | Preconditioners | Exposes IC option/callback-compatible APIs and factor lifecycle. |
| `sparse_analysis.h` | Analysis/refactor API | High | Solver selection | Exposes analysis enums, analysis/factor statistics layouts, factor handle layout, and analyze/refactor/free APIs. |
| `sparse_reorder.h` | Reordering API | Medium-High | Ordering | Exposes reorder function declarations and ordering constants through `sparse_types.h`. |
| `sparse_dense.h` | Dense helper layout/API | Medium | Dense helpers | Exposes dense matrix layout and dense helper allocation/free surface. |
| `sparse_vector.h` | Vector API placeholder | Low-Medium | Vector helpers | Installed as public header; current declaration surface is small, but inclusion keeps the path part of the package contract. |
| `sparse_version.h` | Version metadata | High | Release/package | Generated from `sparse_version.h.in`; defines semantic version macros and integer version metadata. |

## Public Symbol Groups

- Core matrix lifecycle and operations: creation, copy, transpose, validation,
  conversion, IO, multiplication, and free semantics around `SparseMatrix`.
- Direct solvers: LU, CSR LU, Cholesky, LDLT, QR, SVD, and bidiagonalization
  entry points and factor/result/free APIs.
- Iterative solvers: CG, GMRES, MINRES, BiCGSTAB, block variants, matrix-free
  callbacks, preconditioner callbacks, progress callbacks, and result handles.
- Eigen solvers: LOBPCG/thick-restart selection, options, results, handle
  lifecycle, and residual/status reporting.
- Analysis and solver selection: structure analysis, factor handles, factor
  statistics, refactor APIs, and cleanup APIs.
- Reordering and graph-facing APIs: public reorder entry points plus internal
  graph/order implementation symbols in compiled objects.
- Package/version surface: `sparse_version.h`, `sparse_errno`,
  `sparse_idx_bits`, `sparse_scalar_bits`, CMake target metadata, and
  `sparse.pc` metadata.

## ABI-Sensitive Data Layouts

The strongest ABI risk is not the opaque `SparseMatrix` handle; it is the
number of public structs whose field order and size are visible to downstream
consumers.

Public concrete layouts include:

- `SparseCsr`, `SparseCsc`, `LuCsr`, dense block helper layouts, and
  `dense_matrix_t`;
- LU, ILU, IC, LDLT, Cholesky, QR, SVD, and bidiagonalization option, factor,
  rank, result, and diagnostic structs;
- iterative and eigensolver option/result/handle structs;
- analysis and factor-statistics structs;
- callback typedefs for progress, matrix-free matvec, preconditioner, and
  iteration callbacks.

For a shared-library ABI, changing any public concrete struct field order,
field type, field count, enum value, callback signature, or version macro
semantics must be treated as an ABI decision rather than a private refactor.
Opaque handles are the safer model for any future ABI-stable surface.

## Ownership, Lifetime, Allocator, Callback, And Error Contracts

The current public lifetime model is visible but not centralized in a single
ABI contract:

- `SparseMatrix` values are released with `sparse_free`.
- Public factors and handles have type-specific free APIs, including
  `sparse_qr_free`, `sparse_svd_free`, `sparse_ldlt_free`,
  `sparse_ilu_free`, `sparse_ic_free`, `sparse_bidiag_free`,
  `sparse_analysis_free`, `sparse_factor_free`,
  `sparse_iter_handle_free`, `sparse_eigs_result_free`, and
  `sparse_eigs_handle_free`.
- CSR/CSC and dense helper objects have their own free APIs, including
  `sparse_csr_free`, `sparse_csc_free`, and `dense_free`.
- Some helper APIs return caller-owned arrays, which means a future dynamic ABI
  must specify whether callers use library free functions or the platform C
  runtime allocator.
- Callback contracts are ABI-relevant for argument order, constness, user-data
  ownership, reentrancy, and error propagation.
- `sparse_errno` is backed by thread-local state and therefore carries
  thread-safety and runtime-library expectations under a shared build.

## Installed Versus Internal Headers

Installed headers live under `include/` and exclude the internal source
headers under `src/`. The internal headers cover allocator helpers, factor
state, graph internals, workspace internals, dense backend probes, eigensolver
workspace state, QR internals, reorder internals, and backend dispatch helpers.

The installed-header boundary is clear. The compiled-object symbol boundary is
not yet a shared-library boundary because no export map, visibility macro, or
Windows import/export decoration policy exists.

## Static Globals And Process State

Static-first archives avoid dynamic-loader promises, but several implementation
details become governance issues under shared-library support:

- thread-local error state in `sparse_types.c`;
- thread-local graph, nested-dissection, and refinement override/debug state;
- dense runtime backend handles, loaded function pointers, probe status, and
  test override state;
- LDLT dense-provider handles, probe status, and callback dispatch state;
- default option structs used by iterative and preconditioner implementations.

These surfaces require loader lifecycle, thread-safety, teardown, override, and
test-isolation decisions before the library can make dynamic ABI/runtime claims.

## Accidental Export Risks

If the library were switched from static to shared without an explicit export
policy, non-static internal symbols from compiled objects could be exported.
Risk categories include:

- allocator helpers such as internal sparse allocation wrappers;
- factor-state helpers used by direct solvers;
- workspace constructors, reset helpers, and validation helpers for iterative
  and eigensolver implementations;
- QR householder internals and timing helpers;
- graph, separator, coarsening, refinement, and reorder policy helpers;
- dense backend probe and override helpers;
- symbolic analysis and etree helper functions;
- non-`sparse_*` public-ish CSR LU helpers that do not follow one namespace
  convention.

Shared support therefore needs at least one of: a curated export list, a
visibility macro such as `SPARSE_API`, linker version scripts/export maps, and
Windows `__declspec(dllexport/dllimport)` handling. Without that, shared builds
would expose implementation details as accidental ABI.

## Day 3 Handoff

The Day 3 loader audit should focus on:

- Linux `.so` naming, SONAME, exported-symbol filtering, and downstream link
  proof;
- macOS `.dylib` install name, exported-symbol filtering, and downstream link
  proof;
- Windows `.dll` export decoration, import library naming, runtime lookup, and
  MSVC generator behavior;
- whether the Sprint 153 product decision should implement a minimal supported
  shared ABI or keep static-first packaging with stronger test-backed
  diagnostics and exact blockers.
