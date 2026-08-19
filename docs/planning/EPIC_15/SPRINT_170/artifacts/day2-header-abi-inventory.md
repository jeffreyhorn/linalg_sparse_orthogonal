# Sprint 170 Day 2: Header ABI Inventory

## Purpose

Inventory the installed public header surface and identify ABI-relevant public
declarations before making the Sprint 170 shared-library ABI product decision.

## Installed Header Inventory

The maintained install surface is the checked-in `include/*.h` set plus one
generated install header:

| Header | Installed by | Primary ABI relevance |
| --- | --- | --- |
| `sparse_analysis.h` | Make and CMake | Exposes analyze/factor/refactor option/result/factor structs and lifecycle APIs. |
| `sparse_bidiag.h` | Make and CMake | Exposes bidiagonalization factor struct and free API. |
| `sparse_cholesky.h` | Make and CMake | Exposes backend enum, options struct, one-shot factor/solve APIs. |
| `sparse_csr.h` | Make and CMake | Exposes CSR/CSC storage structs and conversion/free APIs. |
| `sparse_dense.h` | Make and CMake | Exposes dense matrix struct and dense helper APIs. |
| `sparse_eigs.h` | Make and CMake | Exposes eigensolver enums, options/result/handle structs, callbacks, and solve APIs. |
| `sparse_ic.h` | Make and CMake | Reuses `sparse_ilu_t` as IC factor storage and exposes IC solve/preconditioner/free APIs. |
| `sparse_ilu.h` | Make and CMake | Exposes ILU factor and ILUT options structs plus preconditioner callbacks. |
| `sparse_iterative.h` | Make and CMake | Exposes iterative option/result/handle structs, callback typedefs, and solver APIs. |
| `sparse_ldlt.h` | Make and CMake | Exposes LDL^T factor/options structs, backend enum, solve/refine/condest APIs. |
| `sparse_lu.h` | Make and CMake | Exposes LU options struct and matrix-shell factor/solve helper APIs. |
| `sparse_lu_csr.h` | Make and CMake | Exposes CSR LU factorization structs and support APIs. |
| `sparse_matrix.h` | Make and CMake | Exposes opaque `SparseMatrix` handle and core matrix lifecycle/mutation APIs. |
| `sparse_qr.h` | Make and CMake | Exposes QR factor/options/rank-info structs and solve/rank/nullspace APIs. |
| `sparse_reorder.h` | Make and CMake | Exposes reorder enums through shared types and permutation APIs. |
| `sparse_svd.h` | Make and CMake | Exposes SVD options/result structs and SVD/pinv/low-rank APIs. |
| `sparse_types.h` | Make and CMake | Exposes `idx_t`, `sparse_scalar_t`, error enum, shared options, callbacks, and runtime metadata APIs. |
| `sparse_vector.h` | Make and CMake | Exposes dense vector helper APIs from the static archive surface. |
| generated `sparse_version.h` | Make and CMake | Exposes version macros derived from `VERSION` and `include/sparse_version.h.in`. |

Install ownership:

- Make uses `HEADERS = $(wildcard include/*.h)` and installs those headers
  plus generated `$(BUILDDIR)/include/sparse_version.h`.
- CMake installs checked-in `include/*.h`, excludes `*.h.in`, and separately
  installs generated `sparse_version.h`.
- `docs/api_reference.md` names checked-in public headers as the declaration
  source of truth and treats generated `sparse_version.h` as install/version
  policy rather than a Doxygen input.

## Declaration Inventory By Header

The declaration scan found these public declaration-bearing surfaces:

| Header | Public declaration count from scan | Main declarations |
| --- | ---: | --- |
| `sparse_matrix.h` | 30 | Opaque matrix handle, create/free, mutation, accessors, arithmetic, Matrix Market I/O, permutation reset. |
| `sparse_iterative.h` | 29 | Progress/result/options structs, preconditioner and matvec callbacks, reusable handles, CG/GMRES/MINRES/BiCGSTAB APIs. |
| `sparse_analysis.h` | 19 | Analysis/factor/refactor enums, option/result/factor structs, analyze/factor/solve/free/refactor APIs. |
| `sparse_types.h` | 20 | Index/scalar width macros, error enum, shared progress/options structs, callback typedef, error/runtime metadata APIs. |
| `sparse_qr.h` | 18 | QR factor/options/rank-info structs, QR factor/apply/form/solve/rank/nullspace/refine APIs. |
| `sparse_lu.h` | 13 | LU options struct, one-shot factor/solve/block/transpose/refine/condition and triangular helper APIs. |
| `sparse_eigs.h` | 13 | Selection/backend enums, options/result/handle structs, reusable handle APIs, eigensolver APIs. |
| `sparse_svd.h` | 12 | SVD options/result structs, full/partial SVD, pseudoinverse, low-rank, condition APIs. |
| `sparse_ldlt.h` | 11 | LDL^T factor/options structs, backend enum, factor/solve/free/inertia/refine/condest APIs. |
| `sparse_csr.h` | 9 | CSR/CSC storage structs, conversion APIs, compressed storage free APIs. |
| `sparse_ilu.h` | 9 | ILU factor struct, ILUT options struct, factor/solve/free/preconditioner APIs. |
| `sparse_reorder.h` | 7 | Reorder and permutation APIs. |
| `sparse_cholesky.h` | 6 | Cholesky backend enum, options struct, factor/factor_opts/solve APIs. |
| `sparse_ic.h` | 5 | IC factor/solve/preconditioner/free APIs through `sparse_ilu_t`. |
| `sparse_bidiag.h` | 4 | Bidiagonalization factor struct and factor/free APIs. |
| `sparse_dense.h` | 3 | Dense matrix struct and dense create/free/GEMM/GEMV helpers. |
| `sparse_lu_csr.h` | 3 | CSR LU factor and dense-block structs plus CSR LU support declarations. |
| `sparse_vector.h` | 1 | Dense vector helper declarations. |
| `sparse_version.h.in` | 7 | Version macros and encode expression used to generate installed `sparse_version.h`. |

The counts are intentionally a planning inventory from direct header scanning,
not a stable ABI symbol list. Day 4 should use object/archive symbol tools to
separate actual link-visible symbols from header declarations.

## Layout-Exposed Objects

`SparseMatrix` is the main positive ABI candidate because
`include/sparse_matrix.h` exposes it as an opaque typedef:

```c
typedef struct SparseMatrix SparseMatrix;
```

Most other public objects are concrete structs whose size, field order, enum
types, callback member positions, and trailing-field extension behavior would
become ABI commitments in a shared-library product:

| Surface | Public layout exposure |
| --- | --- |
| Dense and compressed storage | `dense_matrix_t`, `SparseCsr`, `SparseCsc` expose dimensions, storage pointers, and ownership assumptions. |
| Direct solver factors | `sparse_bidiag_t`, `sparse_ilu_t`, `sparse_ldlt_t`, `sparse_qr_t`, CSR LU structs, and `sparse_factors_t` expose owned pointers and factor internals. |
| Options structs | LU, Cholesky, LDL^T, QR, SVD, eigensolver, iterative, ILUT, and analysis option structs expose field order and zero-initialization conventions. |
| Result structs | Analysis, QR rank info, eigensolver, iterative solver, and shared progress/result structs expose field order and status telemetry. |
| Reusable handles | `sparse_iter_handle_t` and `sparse_eigs_handle_t` expose workspace pointers and capacity fields. |
| Callback typedefs | `sparse_progress_cb_t`, `sparse_iter_callback_fn`, `sparse_precond_fn`, and `sparse_matvec_fn` expose calling conventions and context borrowing rules. |

## Lifecycle-Managed Handles

The header inventory shows several ownership models that matter for ABI
readiness:

- `SparseMatrix *` is caller-owned and released with `sparse_free()`.
- CSR/CSC compressed objects are returned through output pointers and released
  with `sparse_csr_free()` / `sparse_csc_free()`.
- Dense matrices are created with `dense_create()` and released with
  `dense_free()`.
- One-shot LU and Cholesky APIs mutate a `SparseMatrix` matrix shell and use
  `sparse_mark_factored()` / `sparse_reset_perms()` compatibility state.
- Owned direct factors such as `sparse_ldlt_t`, `sparse_qr_t`,
  `sparse_bidiag_t`, and `sparse_ilu_t` are caller-allocated structs whose
  contents are owned by the library after successful factorization and freed
  by matching `*_free()` calls.
- Repeated-run handles for iterative and eigensolver APIs are caller-allocated
  concrete structs initialized/freed by `*_handle_init()` and
  `*_handle_free()`.
- Callback contexts are borrowed and not retained; that convention appears in
  iterative, eigensolver, Cholesky, and LDL^T option surfaces.

Day 3 should audit whether these lifecycle rules are consistently documented
and failure-safe enough for any shared-library ABI posture.

## Versioning Surface

Version metadata is compile-time and generated-install oriented:

- `include/sparse_types.h` includes generated `sparse_version.h`.
- `include/sparse_version.h.in` defines `SPARSE_VERSION_MAJOR`,
  `SPARSE_VERSION_MINOR`, `SPARSE_VERSION_PATCH`,
  `SPARSE_VERSION_ENCODE()`, `SPARSE_VERSION`, and
  `SPARSE_VERSION_STRING`.
- `sparse_idx_bits()` reports the compile-time `SPARSE_IDX_BITS` width.
- `sparse_scalar_bits()` reports the compile-time scalar width.
- `sparse_strerror()` and `sparse_errno()` expose error and errno metadata.

ABI concern: there is version metadata, but no explicit shared-library ABI
version, SONAME policy, symbol-version map, export header, or runtime API that
distinguishes source version from binary compatibility.

## ABI Hazards For Later Decision

| Hazard | Owner surface | Why it matters |
| --- | --- | --- |
| Concrete public structs | Most headers except opaque `SparseMatrix` | Struct size and field order would become binary commitments if passed across shared-library boundaries. |
| Compile-time `SPARSE_IDX_BITS` | `sparse_types.h` | 32-bit versus 64-bit `idx_t` changes function signatures and struct layouts. |
| Compile-time scalar type | `sparse_types.h` | `sparse_scalar_t` is currently `double`; future configurability would affect ABI if changed. |
| Exposed callbacks | Shared progress, iterative, eigs, matrix-free APIs | Callback signatures and cancellation/error semantics become ABI and calling-convention commitments. |
| Exposed allocator ownership | Matrix, dense, CSR/CSC, factors, handles | Cross-library allocation/free rules must be stable for shared-library consumers. |
| Trailing-field extension pattern | Cholesky, LDL^T, eigs, options/results | Source compatibility notes exist, but compiled downstream object layout compatibility is not proven. |
| Internal helper symbol leakage | Public headers cannot answer alone | A shared-library build would need symbol visibility/export analysis from objects and build rules. |
| Version macros are source/package version | `sparse_version.h.in` | They do not define ABI epoch, ABI break policy, or loader compatibility. |

## Stable Candidates

These surfaces look comparatively easier to support in a future shared-library
ABI decision, subject to Day 3 and Day 4 validation:

- opaque `SparseMatrix *` lifecycle through `sparse_create()` and
  `sparse_free()`;
- plain scalar/status APIs such as `sparse_strerror()`, `sparse_errno()`,
  `sparse_idx_bits()`, and `sparse_scalar_bits()`;
- functions that accept caller-owned buffers and return `sparse_err_t`;
- version macros for source/package identity, as long as they are not
  described as ABI compatibility metadata.

## Ambiguous Surfaces

These require later sprint-days before a decision can be made:

- whether concrete factor/result/option structs should be frozen, opaque, or
  explicitly kept source-compatible-only;
- whether reusable handles can remain concrete if a shared-library ABI is ever
  supported;
- whether generated `sparse_version.h` should grow ABI-version or support-tier
  metadata;
- whether installed headers need an export macro such as `SPARSE_API`;
- whether source-level compatibility notes in headers are sufficient for the
  selected product posture.

## Day 2 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Installed header inventory | Complete | Accounted for 18 checked-in headers plus generated `sparse_version.h`. |
| Public symbol/type inventory | Complete | Grouped declaration counts and main declarations by header. |
| Exposed layout and handle lifecycle map | Complete | Identified opaque `SparseMatrix`, concrete structs, callbacks, factors, handles, and allocator ownership. |
| Versioning surface notes | Complete | Mapped generated version macros and runtime bit-width helpers. |
| Day 2 header-ABI inventory artifact | Complete | This file. |

## Validation

Day 2 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| All installed public headers are accounted for. | Complete | The inventory covers every checked-in `include/*.h` header and generated `sparse_version.h`. |
| ABI-relevant public declarations are mapped to owner files. | Complete | Declaration groups and ABI-relevant surfaces are mapped by header. |
| Unresolved ABI hazards are listed for later decision. | Complete | Hazards, stable candidates, and ambiguous surfaces are recorded for Days 3-9. |
