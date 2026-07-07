# Sprint 110 Day 9 Direct & Iterative Proof-Owner Boundary Selection

## Purpose

Day 9 selects one bounded proof-owner cleanup family for Day 10 while keeping
least-squares residuals, refinement results, dense oracle comparisons,
convergence status, and residual norms visible at test call sites.

## Candidate Inventory

| Candidate | File | Proof Values At Risk | Boundary Decision |
|---|---|---|---|
| QR sequential RHS setup | `tests/test_qr.c` | least-squares residuals, QR refinement before/after residuals, intentionally literal RHS values | Defer. Sprint 109 already completed exact-RHS helper work, and remaining QR RHS literals often explain the proof. |
| LDLT CSC external dense-reference oracle cleanup | `tests/test_ldlt_csc.c` | dense oracle comparison, Windows skip behavior, permutation handling, LDLT CSC solve residuals | Defer. This needs a dedicated oracle-lane review because too many proof surfaces are coupled. |
| Iterative exact-RHS setup, all solver families | `tests/test_iterative.c` | convergence status, residual norms, restart/preconditioner behavior, per-family options | Reject as too broad. A cross-solver helper would hide family-specific proof context. |
| Iterative CG exact-RHS allocation/setup | `tests/test_iterative.c` | CG convergence status, residual norms, iteration comparisons, preconditioner comparisons | Select. It is a single solver family and can hide only allocation/fill/matvec setup. |

## Duplicate-Work Exclusions

The following are explicitly excluded:

- Sprint 109 QR exact-RHS helper `make_qr_exact_rhs`;
- Sprint 109 QR exact-RHS call-site replacements;
- Sprint 108/Sprint 109 QR fixture builders;
- Sprint 108 LDLT CSC solve-residual helper
  `assert_s20_solve_residual_below`;
- existing iterative sequential RHS helper `fill_sequential_rhs`;
- existing solver helper headers.

## Selected Cleanup Family

Selected Day 10 cleanup:

```text
tests/test_iterative.c CG exact-RHS allocation/setup helper
```

Allowed helper responsibility:

- allocate `x_exact`;
- allocate `b`;
- fill `x_exact` from a caller-specified pattern;
- compute `b = A*x_exact`;
- return failure cleanly if allocation fails.

Recommended first implementation shape:

- keep the helper `static` and local to `tests/test_iterative.c`;
- do not add a shared test header;
- do not add a compiled helper target;
- support only the selected CG call-site pattern needed for Day 10.

## Initial Call-Site Candidates

Day 10 may update a small subset of dynamically allocated CG exact-RHS sites,
for example:

- `test_cg_laplacian_2d`;
- `test_cg_initial_guess`;
- `test_cg_large_tridiag`;
- `test_cg_max_iter_exceeded`;
- `test_cg_diagonal_preconditioner`;
- `test_cg_precond_laplacian`;
- `test_cg_nos4`;
- `test_cg_bcsstk04`;
- `test_cg_suitesparse_initial_guess`;
- `test_cg_tight_tolerance`;
- `test_cg_loose_tolerance`;
- `test_cg_residual_accuracy`.

Day 10 should stop after a bounded subset if the helper begins to obscure
assertion meaning or cleanup flow.

## Proof-Visibility Rules

The helper may hide only repeated setup. It must not hide:

- `sparse_solve_cg` calls;
- `sparse_iter_opts_t` values;
- preconditioner construction;
- convergence assertions;
- iteration comparisons;
- residual thresholds;
- independent residual recomputation;
- printed residual/iteration labels;
- exact initial guess setup;
- non-SPD behavior cases;
- cleanup of solver-owned or matrix-owned state.

Literal small-stack exact-solution arrays may remain inline when the values
help explain the proof.

## Focused Validation Plan

If Day 10 modifies `tests/test_iterative.c`, run:

```sh
make build/test_iterative
build/test_iterative
```

Because a `.c` file would be modified, also run:

```sh
make format && make lint && make test
git diff --check
```

No CMake test target, helper target, public header, private production header,
install/export rule, or source-list change is expected.

## Deferred Work

Deferred to later sprints or later Sprint 110 closeout:

- QR sequential RHS helper for non-exact least-squares/refinement smoke;
- LDLT CSC external dense-reference oracle cleanup;
- GMRES exact-RHS setup cleanup;
- BiCGSTAB exact-RHS setup cleanup;
- MINRES exact-RHS setup cleanup;
- broad cross-solver iterative setup helpers.

## Completion Status

- One cleanup family was selected.
- Proof assertions remain visible and localized by contract.
- No new compiled helper target is required.
- Validation gates are known before Day 10 edits begin.
- No test code moved on Day 9.
