# Day 6 Ritz Vector Lifting and Publication Boundary Proof Design

## Purpose

Day 6 designs the proof needed before any vector-publication helper movement.
The goal is to make the public vector contract testable across the current
eigensolver backends while keeping proof values visible at call sites.

This is a design-only day. No source movement, helper extraction, public API
change, install-header change, source-list change, helper-target change, Make
change, CMake change, or reviewed CTest membership change is introduced.

## Path Inventory

| Path | Publication location | Current behavior to prove |
|---|---|---|
| Grow-m Lanczos converged path | `src/sparse_eigs.c`, converged branch in `s46_run_growm_backend` | Selected Ritz values are written to `result->eigenvalues[0..take)`, `s20_lift_ritz_vectors(V, Y_long, ...)` writes column-major vectors when `compute_vectors = 1`, and `result->n_converged = take`. |
| Grow-m Lanczos partial path | `src/sparse_eigs.c`, `m_cap` exhaustion fallthrough | Best last-phase Ritz values and vectors are emitted on `SPARSE_ERR_NOT_CONVERGED`, with `result->n_converged = take` and unfilled caller slots left outside the consumed range. |
| Shift-invert grow-m path | `src/sparse_eigs.c`, same grow-m branches | Original-space values use `lambda = sigma + 1 / theta`; lifted vectors are still original-space eigenvectors because `(A - sigma I)^{-1}` preserves eigenspaces. |
| Thick-restart converged path | `src/sparse_eigs_thick_restart.c`, converged branch in `s21_thick_restart_outer_loop` | Selected arrowhead Ritz values are written, and `s20_lift_ritz_vectors(V, Y_arrow, ...)` publishes vectors with the same result-shape contract as grow-m. |
| Thick-restart partial path | `src/sparse_eigs_thick_restart.c`, budget/restart-cap fallthrough | Last-phase values and vectors are emitted on non-convergence, mirroring grow-m's partial publication shape. |
| LOBPCG path | `src/sparse_eigs_lobpcg.c`, final emit block in `s21_lobpcg_solve` | Ritz vectors already live in `X`; the backend copies `X[:, j]` into `result->eigenvectors[:, j]` for `emit = min(k, block_size)` and publishes matching values/counts. |

## Public Result Contract

The Day 7 tests should prove only public caller-visible behavior:

| Field or buffer | Required invariant |
|---|---|
| `result->eigenvalues` | Indices `[0, n_converged)` contain values matching `which`; unconsumed indices are outside the caller's valid result range. |
| `result->eigenvectors` | When `compute_vectors = 1`, columns `[0, n_converged)` are column-major length-`n` vectors paired with `eigenvalues[j]`. |
| `result->n_requested` | Equals the caller's `k` after validation succeeds. |
| `result->n_converged` | Satisfies `0 <= n_converged <= n_requested`; Day 7 should assert both full-convergence and bounded partial-result cases. |
| `result->iterations` | Nonzero for successful or partial eigensolver work, with no stricter cross-backend equality claim. |
| `result->residual_norm` | Bounded by tolerance on `SPARSE_OK`; visible but not used to replace direct vector residual checks. |
| `result->backend_used` | Matches the forced backend under test so the proof belongs to the intended publication path. |

## Day 7 Test Checklist

| Test target | File | Fixture | Assertions |
|---|---|---|---|
| Grow-m vector lift on non-diagonal basis | `tests/test_eigs.c` | Laplacian/tridiagonal SPD, forced `SPARSE_EIGS_BACKEND_LANCZOS`, `compute_vectors = 1`, `reorthogonalize = 1` | `n_requested == k`, `n_converged == k`, backend is grow-m, each vector has norm near 1, pair residual `||A v - lambda v|| / max(|lambda|, 1)` is below tolerance, and columns are mutually orthogonal. |
| Shift-invert vector publication | `tests/test_eigs.c` | Existing interior Laplacian style fixture, `which = NEAREST_SIGMA`, `sigma` visible, forced grow-m or AUTO with deterministic size | Values are original-space `lambda`, vectors satisfy original-space residuals, and vector columns match `[0, n_converged)`. |
| Thick-restart vector lift boundary | `tests/test_eigs_thick_restart.c` | Forced `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART` on a non-diagonal SPD fixture | Backend identity, full count, vector normalization, residuals, and orthogonality; no helper movement yet. |
| LOBPCG publication boundary | `tests/test_eigs_lobpcg.c` | Forced `SPARSE_EIGS_BACKEND_LOBPCG`, `block_size > k` | `n_converged == k`, only columns `[0, k)` are consumed, values/vectors pair correctly, and vectors remain normalized and mutually orthogonal enough for the fixture. |
| Partial-result sentinel preservation | `tests/test_eigs.c` or `tests/test_eigs_thick_restart.c` | A deliberately tight tolerance and bounded iteration/restart budget that returns `SPARSE_ERR_NOT_CONVERGED` with `compute_vectors = 1` | `n_converged <= k`, consumed columns satisfy finite norm/residual shape checks, and sentinel values/vectors beyond `n_converged` remain unchanged. |

## Proof Values to Keep Visible

Day 7 should keep these values at the test call sites:

- matrix size and matrix entries or generator parameters;
- `k`;
- `block_size` when proving `emit = min(k, block_size)`;
- backend selection;
- `which`, `sigma`, `tol`, `max_iterations`, and `reorthogonalize`;
- expected eigenvalue ordering;
- vector norm tolerance;
- pair residual tolerance;
- orthogonality tolerance;
- sentinel values used to prove publication boundaries.

## Helper-Movement Blockers

The following movement remains blocked until Day 7 implementation evidence
exists:

- extracting a shared vector-publication helper from grow-m and thick-restart;
- hiding `s20_lift_ritz_vectors` call-site dimensions behind a broad wrapper;
- centralizing partial-result publication for grow-m and thick-restart;
- changing LOBPCG publication to reuse Lanczos lifting helpers;
- changing public `sparse_eigs_t` result semantics;
- moving eigensolver source ownership or changing build/source lists.

Day 7 may only mark a helper candidate safe if the new tests prove both
full-convergence and partial/publication-boundary behavior without weakening
residual, normalization, and result-shape checks.

## Focused Validation Plan

Day 7 will modify C tests, so the focused and full validation sequence should
be:

```sh
make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg
./build/test_eigs
./build/test_eigs_thick_restart
./build/test_eigs_lobpcg
make format && make lint && make test
```

Day 6 itself changes documentation only. The Day 6 validation is:

```sh
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_114
```

## Completion Criteria

- Day 7 has concrete public assertions for vector lifting and publication.
- Full and partial publication boundaries have explicit sentinel checks.
- Helper movement remains blocked until implementation evidence exists.
- No unsupported API, build, packaging, CTest, or source movement claim is made.
