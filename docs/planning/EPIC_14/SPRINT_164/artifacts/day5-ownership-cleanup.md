# Sprint 164 Day 5: Ownership Cleanup

## Purpose

Clarify caller-owned, library-owned, borrowed, and temporary ownership wording
in the selected Sprint 164 public-header batch without changing public
declarations or expanding support claims.

## Edited Headers

### `include/sparse_matrix.h`

- Added a top-level rule that `SparseMatrix *` objects returned from public
  APIs are caller-owned unless a function documents otherwise.
- Clarified that caller-owned matrix objects are released with `sparse_free()`.
- Clarified that `sparse_copy()` returns an independent caller-owned matrix.
- Clarified that `sparse_transpose()` returns a new caller-owned matrix and
  borrows the source matrix without retaining it.
- Clarified that `sparse_matvec()` borrows caller-owned `x` and `y` buffers
  only for the call and overwrites `y`.
- Clarified that `sparse_matvec_block()` borrows caller-owned dense input and
  output buffers only for the call.
- Clarified that `sparse_add()` creates a caller-owned result while borrowing
  input matrices without modifying them.
- Clarified that `sparse_matmul()` returns a caller-owned product matrix
  through the output pointer and sets it to `NULL` on error.
- Clarified that `sparse_load_mm()` returns a loaded caller-owned matrix
  through the output pointer and sets it to `NULL` on error.

### `include/sparse_iterative.h`

- Clarified that preconditioners are caller-supplied callbacks and contexts.
- Clarified that solver calls borrow callback/context pointers only for the
  duration of a solve and do not own preconditioner factors.
- Clarified that progress payload pointers are borrowed only during callback
  invocation and must not be stored afterward.
- Clarified that `residual_history` is a caller-owned output buffer that the
  solver writes but does not allocate, retain, or free.
- Clarified that `sparse_iter_result_t` storage is caller-owned and only scalar
  fields are written.
- Clarified that preconditioner callbacks receive temporary borrowed `r` and
  `z` buffers and must not retain them.
- Clarified that iterative handles are caller-owned objects, while internal
  workspace reachable through them is library-owned.
- Clarified that underprepared CG, GMRES, and MINRES handles may grow internal
  workspace on demand.
- Clarified that matrix-free callback contexts are caller-owned and borrowed
  only during callback invocation.

### `include/sparse_eigs.h`

- Clarified that eigensolver calls borrow `A`, options, callback contexts, and
  result buffers only for the duration of the call.
- Clarified that eigensolver result buffers are caller-owned and are not
  retained or freed by the library.
- Clarified that preconditioner contexts are caller-owned and borrowed only for
  callback invocation.
- Clarified that progress payloads are borrowed only for callback invocation.
- Clarified that `sparse_eigs_t` receives scalar output fields and caller-owned
  result buffers.
- Clarified that eigensolver handles are caller-owned objects, while internal
  workspace reachable through them is library-owned.
- Clarified that underprepared eigensolver handles may grow internal workspace
  on demand.

## Declaration Preservation

The Day 5 cleanup was comment-only. The normalized selected-header declaration
checksum stayed unchanged from the Day 4 baseline:

```text
Day 4 baseline: 513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
Day 5 recapture: 513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
```

No function declarations, typedefs, enums, macros, struct layouts, include
guards, installed header names, or public preprocessor contracts changed.

## Documentation Cross-Links

No checked-in documentation cross-link edits were required on Day 5. The
cleanup stayed local to public header comments, and the existing docs already
route API readers to `docs/api_reference.md` and the checked-in public headers.

Generated Doxygen HTML remains ignored local output and was not committed.

## Claim Boundary

The Day 5 wording does not introduce support claims for:

- dynamic ABI compatibility;
- shared-library support;
- runtime-loader behavior;
- package-manager distribution;
- broad platform parity;
- backend superiority;
- external-library parity;
- portable performance;
- hosted generated API HTML publication;
- release proof;
- state-of-the-art coverage.

## Validation

- `make format && make lint && make test` passed.
- `git diff --check` passed.
- Scoped claim scan over selected headers, README, API reference, tutorial,
  cookbook, solver-selection, and maintainer guide showed only existing scoped
  disclaimers and no new unsupported claims.
- Generated build/doc artifacts remained ignored and outside the commit set.

## Completion Criteria

- Ownership comments are clearer in the selected public-header batch.
- Memory responsibility is consistent with implementation behavior and tests.
- Declaration preservation evidence is recorded.
- No unsupported package, ABI, runtime-loader, performance, backend, hosted, or
  state-of-the-art claims were introduced.
