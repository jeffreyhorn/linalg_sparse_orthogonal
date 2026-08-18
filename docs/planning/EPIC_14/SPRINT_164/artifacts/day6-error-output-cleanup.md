# Sprint 164 Day 6: Error And Output-Buffer Cleanup

## Purpose

Day 6 clarified selected public-header status, failure-path, and output-buffer
contracts without changing declarations or implementation behavior.

The cleanup stayed within the Day 2 selected public-header batch:

- `include/sparse_matrix.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

## Matrix API Cleanup

- Clarified `sparse_matvec()` failure behavior: validation failure means the
  caller-provided `y` buffer is not a completed product and should not be
  consumed as one.
- Clarified `sparse_matvec_block()` output behavior:
  - `Y` is caller-owned and overwritten on success;
  - error returns do not publish a completed block product;
  - negative `nrhs` returns `SPARSE_ERR_BADARG`.
- Clarified matrix-producing APIs:
  - `sparse_add()` sets `C_out` to `NULL` before validation/allocation and
    leaves it `NULL` on error;
  - `sparse_matmul()` sets `C` to `NULL` on entry and leaves it `NULL` on
    error;
  - `sparse_load_mm()` sets `mat_out` to `NULL` on entry and leaves it `NULL`
    on error.
- Clarified `sparse_add_inplace()` failure behavior: after argument validation
  succeeds, allocation or insertion errors may leave `A` partially updated.

## Iterative Solver Cleanup

- Clarified residual-history publication:
  - residual-history buffers remain caller-owned;
  - only the first `result.residual_history_count` entries are meaningful on
    non-convergence or cancellation;
  - `residual_history_count` is 0 when no history buffer was supplied or no
    entries were recorded.
- Clarified preconditioner callback output behavior:
  - callbacks receive borrowed `r` and `z` buffers;
  - callbacks should fully write `z` on `SPARSE_OK`;
  - callback errors are propagated without promising additional diagnostics.
- Clarified CG, GMRES, MINRES, BiCGSTAB, block solver, handle-based solver, and
  matrix-free solver contracts:
  - `x`/`X` contains an approximate solution on `SPARSE_OK` or
    `SPARSE_ERR_NOT_CONVERGED`;
  - `result` is populated on `SPARSE_OK` and `SPARSE_ERR_NOT_CONVERGED`;
  - validation, allocation, cancellation, callback, numeric, or hard-error
    paths leave result fields and output buffers best-effort/unspecified unless
    a narrower contract is documented.

## Eigensolver Cleanup

- Clarified `sparse_eigs_t` publication boundaries:
  - scalar outputs are initialized after validation succeeds;
  - validation failures leave result scalar fields at their pre-call values;
  - `SPARSE_ERR_NOT_CONVERGED` may publish bounded partial outputs;
  - cancellation, allocation, callback, preconditioner, and shift-invert
    failures leave caller buffers best-effort/unspecified outside documented
    telemetry.
- Clarified eigenvalue/eigenvector result buffers:
  - entries or columns at indices `>= n_converged` are not outputs of the call;
  - caller-owned buffers are filled in place on `SPARSE_OK` and
    `SPARSE_ERR_NOT_CONVERGED`.
- Added the documented `SPARSE_ERR_CANCELLED` return path for progress-callback
  cancellation.

## Declaration Preservation

The selected header declaration checksum matched the Day 4 baseline after the
Day 6 edits and final formatting:

```text
Day 6 post-format normalized checksum: 513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
Day 4 normalized checksum: 513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
```

## Documentation Cross-Links

No README, tutorial, cookbook, solver-selection, API reference, or maintainer
guide edits were required. The Day 6 changes are API-local comment
clarifications, and the existing documentation already points users to the
checked-in headers and generated API reference as the contract source.

## Claim Boundary

The scoped claim scan showed only pre-existing bounded disclaimers and local
evidence wording. Day 6 did not add claims for:

- fail-closed behavior beyond the already documented narrow cases;
- diagnostic guarantees or error classes beyond current return codes;
- ABI, package-manager, shared-library, runtime-loader, hosted, release-proof,
  backend-superiority, portable-performance, or state-of-the-art support.

## Validation

- `make format && make lint && make test`
- normalized selected-header declaration checksum comparison against Day 4
  baseline
- `git diff --check`
- scoped claim scan over selected headers plus README/API/user-facing docs
- generated-output status check for `build`, `docs/api/html`,
  `scripts/__pycache__`, and `tests/__pycache__`

## Completion Criteria

- Users can distinguish caller-owned inputs, caller-owned outputs, successful
  outputs, partial non-convergence outputs, and hard-error best-effort outputs
  for the selected APIs.
- Selected public-header declarations did not drift.
- Unsupported diagnostic, fail-closed, ABI, package, runtime, performance, and
  state-of-the-art guarantees were not introduced.
