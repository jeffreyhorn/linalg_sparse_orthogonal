# Sprint 170 Day 3: Lifecycle And Ownership Semantics Audit

## Purpose

Review the public allocation, ownership, callback, workspace, and error
semantics that would become product commitments if the project added a
shared-library ABI support claim.

## Lifecycle API Map

| Surface | Public lifecycle | ABI relevance |
| --- | --- | --- |
| Matrix shell | `sparse_create()`, `sparse_copy()`, conversion constructors, and matrix arithmetic return caller-owned `SparseMatrix *` objects released with `sparse_free()`. `sparse_free(NULL)` is documented as a no-op. | Strongest shared-library candidate because `SparseMatrix` is opaque and all storage is released through library-owned teardown. |
| Matrix-shell state reset | LU/Cholesky one-shot paths mutate the matrix shell and record factored/permutation state. `sparse_reset_perms()` restores permutation state, and `sparse_mark_factored()` records compatibility state. | Behavior is public, but binary layout remains hidden behind `SparseMatrix`. The product decision still needs to preserve one-shot mutation semantics. |
| Dense matrices | `dense_create()` returns a caller-owned `dense_matrix_t *`; `dense_free()` releases it and is safe with `NULL`. Dense GEMM/GEMV write into caller-provided outputs. | `dense_matrix_t` is layout-exposed, so dimensions, storage pointers, and field order would be ABI commitments. |
| CSR/CSC compressed storage | `sparse_to_csr()` and `sparse_to_csc()` allocate compressed objects through output pointers and require `sparse_csr_free()` / `sparse_csc_free()`. Output pointers are documented as set to `NULL` on error. | Struct layouts and owned pointer fields cross the boundary. Allocation/free pairing is documented but still requires a stable allocator policy for shared consumers. |
| Source conversion constructors | `sparse_create_from_csr()` / `sparse_create_from_csc()` return new caller-owned matrices or `NULL`; `sparse_from_csr()` / `sparse_from_csc()` provide `sparse_err_t` output-pointer variants. | Mixed constructor styles are source-compatible, but a shared-library product should make the NULL-versus-error-code convention explicit. |
| One-shot LU and Cholesky | `sparse_lu_factor*()` and `sparse_cholesky_factor*()` mutate caller-owned `SparseMatrix` objects in place; solves require the factored matrix. Cancellation can leave in-place factorization inputs in an indeterminate state. | ABI-safe layout-wise because `SparseMatrix` is opaque, but lifecycle behavior is a visible product contract. |
| Owned direct factors | QR, LDLT, bidiag, ILU/ILUT, IC, and repeated direct factors use caller-allocated structs that receive library-owned internals and are released with matching `*_free()` calls. Many free APIs are safe on zeroed structs. | High ABI risk: concrete factor structs expose pointer fields, scalar fields, capacity assumptions, and overwrite/free expectations. |
| Repeated-run direct analysis | `sparse_analyze()` fills `sparse_analysis_t`; `sparse_factor_numeric()` fills `sparse_factors_t`; `sparse_refactor_numeric()` replaces numeric contents on success while preserving prior factors on failure; matching free calls zero the structs. | Useful lifecycle clarity, but both public structs expose symbolic and numeric factor internals. |
| Iterative repeated-run handles | `sparse_iter_handle_t` is caller-owned, zero-initializable, and freed with `sparse_iter_handle_free()`. Internal workspace is library-owned behind a `void *internal_state`. | Better than concrete workspace layouts, but the one-field struct size and initialization convention would still be ABI. |
| Eigensolver repeated-run handles | `sparse_eigs_handle_t` mirrors the iterative handle lifecycle: caller-owned object, library-owned internal workspace, prepare/run/free helpers, zeroed state accepted. | Same ABI shape as iterative handles; comparatively narrow if kept as a stable one-field handle. |
| Caller-owned result buffers | Iterative, eigensolver, SVD, dense, matrix-free, and solve APIs mostly write into caller-owned buffers supplied for the call. | Favorable for ABI because allocation stays with the caller, but buffer-size preconditions and partial-result semantics must remain stable. |
| Callback contexts | Progress, verbose, preconditioner, and matrix-free matvec callbacks borrow context and buffer pointers only during synchronous invocation. The library does not retain or free caller contexts. | Callback function pointer signatures, calling convention, cancellation behavior, and borrowed-lifetime rules become ABI commitments. |

## Ownership And Allocator Assumptions

- Library-created matrices, dense matrices, CSR/CSC objects, direct-factor
  internals, analysis objects, factor objects, and reusable handle workspaces
  must be released by the matching library free routine.
- Several APIs intentionally permit zero-initialized structs before first use
  and make free routines reset the object to a safe empty state.
- Some factor functions overwrite caller-provided output structs without
  freeing prior contents; those headers require callers to call the matching
  free routine before reuse.
- IC explicitly resets the output object on entry so its free routine is safe
  after an error return; repeated direct refactor explicitly preserves old
  factors on failure.
- Conversion and arithmetic output-pointer APIs often clear outputs to `NULL`
  on error, but the convention is not universal across constructor-style APIs.
- The shared-library risk is cross-boundary allocation ownership, especially
  on Windows where callers and a DLL can use different C runtime allocators.
  The current static-first product avoids that runtime boundary.

## Error-Handling ABI Notes

`sparse_err_t` is a public enum with stable integer values in
`include/sparse_types.h`. Adding new values at the end is source-compatible for
most callers, but changing existing values, meanings, or cancellation semantics
would be an ABI and behavior break.

The main public patterns are:

- `SPARSE_ERR_NULL` for missing required pointers.
- `SPARSE_ERR_BADARG` for invalid enum values, invalid dimensions, invalid
  options, already-mutated matrix state, or callback/preconditioner mismatch.
- `SPARSE_ERR_ALLOC` for allocation failure.
- `SPARSE_ERR_SHAPE` for dimension or matrix-shape mismatch.
- `SPARSE_ERR_SINGULAR`, `SPARSE_ERR_NOT_SPD`, `SPARSE_ERR_NOT_CONVERGED`,
  and `SPARSE_ERR_NUMERIC` for numerical outcomes.
- `SPARSE_ERR_CANCELLED` for progress-callback cancellation, with different
  input-state guarantees for in-place factorization versus out-of-place
  routines.
- `SPARSE_ERR_IO` plus `sparse_errno()` for errno-backed I/O context.

Runtime metadata helpers are narrow and ABI-friendly: `sparse_strerror()`,
`sparse_errno()`, `sparse_idx_bits()`, and `sparse_scalar_bits()`. Current
implementation stores `sparse_errno()` in thread-local state, which is a useful
behavioral fact but not currently elevated to a shared-library ABI policy.

## Stable Lifecycle Contracts

These contracts are credible candidates for a future shared-library decision:

- Opaque `SparseMatrix *` creation, mutation, query, and teardown through
  library APIs.
- Caller-owned dense/scalar buffers for solve, matvec, eigensolver, and SVD
  outputs.
- Borrowed callback contexts that are not retained beyond synchronous calls.
- Explicit `sparse_err_t` returns for most fallible APIs.
- Thread-local errno context as an implementation-backed I/O detail.
- Repeated-run handle lifecycles if their public structs remain intentionally
  narrow and the `void *internal_state` contract is frozen.

## Lifecycle ABI Blockers

| Blocker | Severity | Why it blocks a shared-library claim |
| --- | --- | --- |
| Concrete public factor/result/option structs | High | Size, field order, enum width, padding, and trailing-field extension behavior would become binary commitments. |
| Cross-boundary allocation/free policy | High | Any object allocated by the library and freed by a matching library routine needs explicit CRT/runtime-loader guidance before DLL support can be claimed. |
| Source-only trailing field notes | High | Headers document source compatibility for some appended fields, but downstream binaries compiled against older layouts are not protected. |
| Mixed constructor/error styles | Medium | Some APIs return `NULL`; others return `sparse_err_t` with output pointers. This is usable, but it must be frozen deliberately if ABI support is claimed. |
| In-place cancellation state | Medium | LU/Cholesky cancellation can leave caller-visible matrix state indeterminate. That is documented, but it is a visible lifecycle contract that package docs must not overstate. |
| Callback ABI | Medium | Function pointer signatures and synchronous invocation rules need a calling-convention/export policy on Windows before a DLL product is credible. |
| Compile-time width choices | Medium | `SPARSE_IDX_BITS` changes `idx_t`, signatures, and struct layouts; shared-library packages would need width-specific identity and install metadata. |
| No ABI version policy | High | Version macros identify source/package version, not ABI epoch, SONAME, loader compatibility, or binary break rules. |

## Day 4 Handoff

Day 4 should inspect actual archive/object symbols and build-system visibility
behavior. Header lifecycle clarity alone is not enough to decide shared-library
support because internal helper leakage, export lists, symbol prefixing, and
platform loader behavior are build/link properties.

## Day 3 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Lifecycle API map | Complete | Grouped public lifecycle families and ownership rules. |
| Ownership and allocator assumption inventory | Complete | Recorded library-owned, caller-owned, borrowed, and zeroed-object patterns. |
| Error-handling ABI notes | Complete | Mapped public error codes, cancellation semantics, and errno/runtime helpers. |
| Lifecycle ABI blocker list | Complete | Identified blockers that must drive the product decision. |
| Day 3 lifecycle-audit artifact | Complete | This file. |

## Validation

Day 3 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Handle ownership semantics are clear enough to support a product decision. | Complete | The audit separates opaque matrix, concrete factors, repeated-run handles, and caller-owned buffers. |
| Exposed layout and allocator risks are documented. | Complete | Concrete structs, width choices, callbacks, allocation/free pairing, and version-policy gaps are listed as blockers. |
| No undocumented ABI claim is added. | Complete | The artifact preserves the static-first baseline and does not claim shared-library ABI support. |
