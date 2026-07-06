# Day 7 Iterative Convergence Boundary

## Purpose

Day 7 defines the safe cleanup boundary for `tests/test_iterative.c`. Iterative
solver tests are convergence-sensitive, so the boundary must keep solver
options, restarts, preconditioners, convergence flags, residual checks, and
direct comparisons visible at the call sites.

## Live Inventory

Current `tests/test_iterative.c` state:

| Area | Current State | Day 7 Disposition |
|---|---|---|
| SPD matrix builders | `build_spd_tridiag`, `build_identity`, and `build_laplacian_2d` already cover common CG fixtures. | Do not move further. |
| Unsymmetric tridiagonal builder | `build_unsym_tridiag` already covers constant-diagonal GMRES fixtures. | Reuse as-is; do not duplicate. |
| Matrix-free helpers | `sparse_matvec_cb`, diagonal matrix-free operator helpers, and sequential RHS helper already support matrix-free parity tests. | Exclude Sprint 107 completed work. |
| CG option/result tests | CG tolerance, max-iteration, residual accuracy, initial guess, and direct-solver comparisons keep convergence evidence inline. | Defer. |
| GMRES restart tests | Restart values and convergence behavior are the proof. | Defer. |
| SuiteSparse GMRES/CG corpus setup | Matrix loading, restart/tolerance selection, residual checks, and direct comparisons are corpus-specific. | Defer. |
| Diagonal preconditioner fixtures | Two GMRES tests repeat poorly scaled unsymmetric tridiagonal construction and `diag_inv` setup. | Select bounded setup helper. |
| Right/left preconditioner semantics | `precond_side`, reported residual, true residual, and convergence assertions encode the proof. | Keep inline. |

## Selected Day 8 Candidate

Add one local helper near the existing iterative matrix builders:

```c
static SparseMatrix *build_scaled_unsym_tridiag_with_diag_inv(idx_t n,
                                                              double diag_base,
                                                              double diag_step,
                                                              double upper,
                                                              double lower,
                                                              double **diag_inv_out);
```

Expected construction:

- create an `n x n` sparse matrix;
- for each row, compute `d = diag_base + diag_step * i`;
- insert diagonal value `d`;
- insert lower and upper first-neighbor entries;
- allocate `diag_inv_out` and fill `1.0 / d`;
- return `NULL` and free partial state on allocation failure.

This helper may hide only matrix and diagonal-inverse fixture construction.

## Approved Day 8 Call Sites

Only these call sites are approved for Day 8 updates:

- `test_gmres_right_precond_diag`
- `test_gmres_diagonal_preconditioner`

The following must remain visible at each call site:

- `idx_t n`;
- exact RHS setup;
- `diag_precond_t pc`;
- `sparse_gmres_opts_t` values;
- `restart`;
- `tol`;
- `precond_side`;
- unpreconditioned, left-preconditioned, and right-preconditioned solve calls;
- convergence assertions;
- iteration reporting;
- reported-vs-true residual assertions.

## Explicit Non-Candidates

### CG convergence lanes

Do not move setup in:

- `test_cg_tight_tolerance`
- `test_cg_loose_tolerance`
- `test_cg_residual_accuracy`
- `test_cg_max_iter_exceeded`
- direct CG-vs-Cholesky tests

Rationale: the options, residuals, iteration counts, and direct comparisons are
the proof. Moving them would hide why convergence behavior is expected.

### GMRES restart lanes

Do not move setup in:

- `test_gmres_restart_comparison`
- `test_gmres_unrestarted`
- `test_gmres_restart_1`
- SuiteSparse restart comparison tests

Rationale: restart values and convergence outcomes are the behavioral surface.
Any cleanup here needs a separate restart-specific boundary.

### SuiteSparse corpus lanes

Do not move setup in:

- `test_cg_nos4`
- `test_cg_bcsstk04`
- `test_gmres_west0067`
- `test_gmres_steam1`
- `test_gmres_orsirr_1`
- direct GMRES-vs-LU and GMRES-vs-CG comparisons

Rationale: corpus-specific loading, exact RHS generation, restart/tolerance
choice, residual checks, and direct comparisons should remain colocated.

### Matrix-free comparison lanes

Do not move setup in matrix-free CG or GMRES tests.

Rationale: Sprint 107 already completed matrix-free tridiagonal and sequential
RHS helper reuse. Remaining matrix-free tests rely on visible callback,
operator, and direct matrix-vs-matrix-free comparison evidence.

## Call-Site Readability Rules

Day 8 must preserve:

- solver option literals at call sites;
- restart values at call sites;
- tolerance values at call sites;
- preconditioner side at call sites;
- explicit solve calls and result objects;
- direct residual checks;
- direct convergence assertions;
- direct left/right and preconditioned/unpreconditioned comparisons.

The helper may move only repeated fixture construction and `diag_inv`
initialization.

## Placement and Target Rules

- Place the helper in `tests/test_iterative.c` near existing matrix builders.
- Do not add a helper header.
- Do not create a new compiled test target.
- Do not change Makefile or CMake membership.
- Do not change CTest registration or reviewed test counts.
- Do not touch public headers or implementation sources.

## Focused Validation Plan

If Day 8 changes `tests/test_iterative.c`, run:

```sh
make build/test_iterative && ./build/test_iterative
make format && make lint && make test
git diff --check
```

Because Day 8 would modify a `.c` test file, the full quality gate is required.

## Day 7 Decision

Proceed to Day 8 with exactly one bounded iterative fixture candidate:
`build_scaled_unsym_tridiag_with_diag_inv`. All broader CG convergence,
GMRES restart, SuiteSparse corpus, matrix-free comparison, and residual-proof
movement remains deferred.

## Completion Criteria Status

- Repeated matrix, RHS, option, restart, preconditioner, and result setup were
  inventoried.
- Solver options, restarts, convergence flags, residuals, and direct
  comparisons that must stay visible were marked.
- One bounded cleanup batch was selected.
- Focused iterative validation commands are known before edits begin.
