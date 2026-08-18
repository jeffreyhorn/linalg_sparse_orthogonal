# Sprint 164 Day 7: Options, Results, And Backend Wording Cleanup

## Purpose

Clarify selected public-header option/result behavior and backend-selection
boundaries without changing declarations, support scope, or performance claims.

Day 7 stayed inside the selected Sprint 164 public-header batch:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_matrix.h`
- `docs/solver_selection.md`

## Iterative Options And Results Cleanup

`include/sparse_iterative.h` now documents that NULL options and an explicit
zero-initialized options struct are not always equivalent:

- CG NULL defaults remain `max_iter = 1000`, `tol = 1e-10`, `verbose = 0`.
- An explicit zero-initialized CG options struct requests a zero-iteration
  budget and zero relative residual tolerance.
- Optional callback and residual-history fields remain disabled unless the
  caller supplies them.
- GMRES records the same distinction, including the invalid explicit
  `restart = 0` case.

The iterative result contract now tells callers to interpret the shared result
struct with the return code. Convergence, non-convergence, stagnation, and
breakdown all use `sparse_iter_result_t`, but they differ in what the
approximation in `x` or `X` means.

## Eigensolver Options, Results, And Backend Cleanup

`include/sparse_eigs.h` now documents `SPARSE_EIGS_BACKEND_AUTO` as an
implementation routing policy:

- AUTO routes to LOBPCG only when a preconditioner is supplied, the matrix is
  large enough, and the effective block size is at least 4.
- Otherwise AUTO chooses between grow-m Lanczos and thick-restart Lanczos using
  the documented compile-time thresholds.
- `result->backend_used` records the concrete backend on successful calls, but
  this is routing telemetry rather than backend-superiority evidence.

The eigensolver options struct now makes the zero-initialized-vs-NULL distinction
explicit for nonzero defaults and clarifies:

- `which = 0` means `SPARSE_EIGS_LARGEST`.
- `backend = 0` means `SPARSE_EIGS_BACKEND_AUTO`.
- `block_size` is a workload-control knob and not a portable speedup guarantee.
- `precond`, `precond_ctx`, and `lobpcg_soft_lock` are LOBPCG-specific and are
  ignored by Lanczos-family backends.

The eigensolver result struct now ties telemetry interpretation to the return
code and clarifies:

- caller-owned eigenvalue/eigenvector buffers are not retained or freed;
- entries beyond `n_converged` are not call outputs;
- `iterations` means outer iterations for the selected backend;
- `peak_basis_size` describes local memory behavior rather than a broad memory
  claim;
- `backend_used` documents routing, not broad backend superiority.

## Sparse Matrix Backend Boundary Cleanup

`include/sparse_matrix.h` now clarifies that forcing a sparse direct-solver
branch requests that implementation path only. It does not imply package, ABI,
platform, or broad performance support beyond the direct-solver contract owned
by the solver headers.

## Solver-Selection Alignment

`docs/solver_selection.md` now mirrors the eigensolver backend wording:

- AUTO is a routing policy, not a superiority claim.
- Callers should inspect `backend_used`, `peak_basis_size`, convergence count,
  and residual norm together with the return code before deciding whether an
  explicit backend request is useful.

## Deferred Ambiguity Queue

- Broad direct-solver option/result cleanup for `sparse_cholesky.h`,
  `sparse_ldlt.h`, `sparse_qr.h`, and `sparse_svd.h` remains deferred outside
  the Day 7 selected batch.
- Generated API HTML regeneration/publication remains deferred to the planned
  generated-reference follow-through days.
- Backend threshold retuning remains deferred; Day 7 only clarified existing
  local dispatch policy.
- A solver-by-solver result initialization matrix for every hard-error path
  remains deferred because Day 7 was scoped to selected public-header comments.

## Declaration Preservation

The selected public-header normalized declaration checksum stayed unchanged
after formatting and after the full quality gate:

```text
513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
```

This matches the Day 4 baseline checksum.

## Claim Boundary

The cleanup preserves the Sprint 163 non-superiority posture:

- no backend-superiority claim;
- no portable performance or speedup guarantee;
- no package, ABI, platform, runtime-loader, shared-library, hosted, release,
  external-library parity, or state-of-the-art claim.

## Validation

Commands run:

```sh
make format && make lint && make test
git diff --check
```

Additional Day 7 checks:

- normalized selected-header declaration checksum compared against the Day 4
  baseline before and after the full gate;
- scoped claim scan over selected public headers plus README/API/tutorial/
  cookbook/solver-selection/maintainer documentation;
- generated-output status check for `build`, `docs/api/html`,
  `scripts/__pycache__`, and `tests/__pycache__`.

## Completion Criteria

- Option and result behavior is clearer for the selected public-header batch.
- Backend-selection wording is bounded to implemented routing behavior.
- Solver-selection documentation reflects the same backend boundary.
- Public declarations remain unchanged.
- Required C/header quality gate passes.
