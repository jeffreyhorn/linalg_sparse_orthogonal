# Day 3 Lanczos Iterate Behavior Proof

## Purpose

Day 3 implements the Day 2 `lanczos_iterate_op` behavior proof design across
the basic grow-m Lanczos path, thick-restart path, and LOBPCG-adjacent parity
path. The implementation stays inside existing test translation units and
does not change public API, install headers, source lists, helper targets,
Make/CMake metadata, or reviewed CTest membership.

## Implemented Proofs

| Proof | File | Test | Evidence |
|---|---|---|---|
| Basic grow-m Lanczos public behavior | `tests/test_eigs.c` | `test_growm_lanczos_iterate_op_public_behavior` | Forces `SPARSE_EIGS_BACKEND_LANCZOS`, uses shifted tridiagonal `n = 64`, `k = 2`, explicit `max_iterations = 64`, checks backend identity, convergence count, peak basis size, iteration visibility, residual norm, eigenvalue ordering, and Ritz residuals. |
| Thick-restart empty-state recurrence parity | `tests/test_eigs_thick_restart.c` | `test_thick_restart_iterate_tridiag_empty_state_matches_lanczos` | Uses a non-diagonal tridiagonal SPD fixture with fixed `v0`, `n = 8`, `m = 8`, `reorthogonalize = 1`, and compares `m_actual`, `V`, `alpha`, and `beta` against `lanczos_iterate` to `1e-14`. |
| LOBPCG-adjacent public parity | `tests/test_eigs_lobpcg.c` | `test_lobpcg_adjacent_lanczos_public_result_parity` | Uses Laplacian tridiagonal `n = 30`, `k = 4`, `max_iterations = 200`, and compares LOBPCG against grow-m Lanczos on public eigenvalues, backend identities, convergence counts, iteration visibility, and Ritz residuals. |

## Proof Values Kept Visible

- Matrix dimensions and fixture construction.
- Requested eigenpair count `k`.
- Backend selections.
- Tolerances.
- Explicit iteration budgets.
- Expected public result fields.
- Residual assertions.

No cross-file helper was introduced. The only added declaration is a local
forward declaration for an existing static residual helper in `tests/test_eigs.c`.
The grow-m assertion treats `max_iterations` as the per-run cap that is visible
through `peak_basis_size`; cumulative `result.iterations` may exceed that cap
when grow-m retries are required.

## Drift Assessment

| Surface | Day 3 result |
|---|---|
| Public API / install headers | Unchanged. |
| Source files | Unchanged. |
| Source-list metadata | Unchanged. |
| Make/CMake targets | Unchanged. |
| CTest registration | Unchanged. |
| Test `.c` files | Updated with focused proof tests only. |
| Documentation | Working notes and this artifact updated. |

## Validation Plan

Because Day 3 modifies `.c` test files, the required quality gate is:

```sh
make format && make lint && make test
```

Additional focused reruns may be used before the full gate:

```sh
make test_eigs
make test_eigs_thick_restart
make test_eigs_lobpcg
```

## Completion Criteria

- The grow-m Lanczos path has direct public behavior coverage.
- The thick-restart empty-state path has deterministic recurrence parity
  against base Lanczos on a non-diagonal fixture.
- The LOBPCG-adjacent path has public parity and residual evidence against
  grow-m Lanczos without claiming shared implementation ownership.
- No unsupported API, build metadata, helper-target, source-list, or reviewed
  CTest drift is introduced.
