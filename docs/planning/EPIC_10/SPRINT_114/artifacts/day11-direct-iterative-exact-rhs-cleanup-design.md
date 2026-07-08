# Sprint 114 Day 11: Direct/Iterative Exact-RHS Cleanup Design

## Purpose

Day 11 designs a bounded cleanup for direct/iterative exact-RHS setup before
Day 12 implementation. The goal is to reduce repeated allocation/fill/matvec
boilerplate while preserving solver-specific proof values at test call sites:
least-squares residuals, preconditioner choices, restart sizes, convergence
status, iteration comparisons, breakdown/nonconvergence expectations, and
residual thresholds.

No code movement, broad oracle abstraction, public API change, install-header
change, source-list change, helper-target change, Make/CMake change, or
reviewed CTest membership change is part of Day 11.

## Exact-RHS Setup Inventory

| Area | Primary file | Current setup pattern | Proof values to preserve |
|---|---|---|---|
| QR sequential RHS | `tests/test_qr.c` | `make_qr_exact_rhs(A, x_len, b_len, &x_exact, &b)` builds `x_exact[i] = i + 1` and `b = A*x_exact`; some tests allocate solve buffers separately. | Matrix fixture names, QR mode/reorder choices, reported residuals, true residual thresholds, refinement before/after residuals, and least-squares dimensions. |
| CG preconditioner-specific exact RHS | `tests/test_iterative.c` | Early tests use local stack arrays or repeated heap allocation; the file already has `make_iterative_exact_rhs` for generated exact vectors but many later CG/preconditioner tests still hand-roll setup. | SPD matrix assumptions, diagonal/IC preconditioner setup, unpreconditioned vs preconditioned iteration counts, exact-vector formulas, and residual thresholds. |
| GMRES exact RHS | `tests/test_iterative.c` | Small tests use stack `x_exact`; larger restart/preconditioner/SuiteSparse tests repeatedly allocate `x_exact`, `b`, and `x`, fill values, and call `compute_rhs`. | Non-symmetric fixtures, restart settings, right/left preconditioner side, convergence/nonconvergence status, lucky-breakdown or max-iteration behavior, and true-vs-reported residual comparisons. |
| BiCGSTAB exact RHS | `tests/test_bicgstab.c` | SuiteSparse and cross-solver tests repeatedly allocate `x_exact`, `b`, and `x`, fill sequential values, and call local `compute_rhs`. | ILU/ILUT choices, convergence status, accepted nonconvergence on difficult systems, breakdown hardening, residual thresholds, and GMRES comparison values. |
| MINRES exact RHS | `tests/test_minres.c` | SPD, KKT, preconditioner, and direct-solver comparison tests repeatedly allocate exact vectors, fill sequential/sine/cosine values, and call `sparse_matvec`. | Symmetry/indefiniteness assumptions, IC/Jacobi/exact preconditioner behavior, MINRES-vs-CG/GMRES/LDLT comparisons, iteration counts, and residual thresholds. |

## Cleanup Boundaries

The cleanup should stay inside existing test files. It may introduce or extend
small local helpers, but helpers must only own repetitive mechanics:

- allocate `x_exact` and `b`;
- fill `x_exact` with a visible named pattern;
- compute `b = A*x_exact`;
- release helper-owned buffers through ordinary `free`.

The cleanup must not hide:

- solver options or tolerances;
- restart sizes;
- preconditioner construction;
- matrix fixture choice;
- expected convergence or nonconvergence status;
- residual thresholds;
- iteration comparisons;
- exact literal vectors for small analytical tests.

## Solver-Specific Cleanup Plan

### QR

Keep `make_qr_exact_rhs` as the QR-specific helper. Day 12 should only use it
where the proof is sequential `x_exact[i] = i + 1` and the matrix dimensions
match the solve. Do not route rank-deficient, overdetermined, noisy
least-squares, or analytical small-vector tests through a generic helper if it
would hide dimensions or expected residuals.

Bounded Day 12 targets:

- normalize cleanup around `test_qr_solve_nos4`, `test_qr_bcsstk04`,
  `test_qr_west0067`, `test_qr_vs_lu`, `test_qr_reorder_nos4_fillin`, and
  `test_qr_refine_nos4`;
- keep residual assertions and printed before/after refinement values at the
  call sites;
- do not change QR helper ownership beyond `tests/test_qr.c`.

### CG

Use the existing `make_iterative_exact_rhs` pattern in `tests/test_iterative.c`
for generated exact vectors. Day 12 should focus on CG tests that currently
repeat heap allocation and sequential/sine exact-vector setup, especially
preconditioner comparisons where the real proof is the iteration/residual
relationship.

Bounded Day 12 targets:

- `test_cg_diagonal_preconditioner`;
- `test_cg_precond_laplacian`;
- one CG-vs-direct or SuiteSparse comparison that already uses generated
  sequential RHS setup.

Keep the diagonal preconditioner arrays, IC factorization, and iteration
comparison assertions visible at the test call sites.

### GMRES

Use local helper setup only for generated exact RHS cases. Keep small
analytical systems with literal `x_exact` arrays inline because those values
are part of the proof. Keep restart and preconditioner settings visible.

Bounded Day 12 targets:

- `test_gmres_large_unsymmetric`;
- `test_gmres_max_iter_exceeded`;
- `test_gmres_arnoldi_correctness` only if the dense 5x5 exact vector remains
  visible;
- `test_gmres_restart_comparison`;
- `test_gmres_diagonal_preconditioner`;
- one right-preconditioned GMRES test with true-vs-reported residual checks.

Do not introduce a shared GMRES oracle or move restart assertions into a
helper.

### BiCGSTAB

Add a local BiCGSTAB exact-RHS setup helper only if it stays in
`tests/test_bicgstab.c` and supports a small set of named patterns. The first
safe pattern is sequential `x_exact[i] = i + 1`, used across SuiteSparse and
comparison tests.

Bounded Day 12 targets:

- `test_bicgstab_west0067`;
- `test_bicgstab_steam1`;
- `test_bicgstab_orsirr_1`;
- `test_s103_bicgstab_steam1_ilu_vs_gmres30_reference`.

Keep ILU/ILUT options, accepted nonconvergence handling, and residual
thresholds visible at each test.

### MINRES

Add a local MINRES exact-RHS helper only for generated vector patterns, with
separate named pattern callbacks for sequential, sine, cosine, and scaled
sequential values if needed. Do not combine SPD and KKT fixture construction
inside the helper.

Bounded Day 12 targets:

- `test_minres_spd_tridiag`;
- `test_minres_precond_ic_spd`;
- `test_minres_precond_ic_vs_cg`;
- `test_minres_precond_jacobi_indefinite`;
- `test_minres_precond_ic_banded`;
- `test_minres_vs_ldlt_spd` and `test_minres_vs_ldlt_indefinite` only if the
  direct-solver comparison remains obvious at the call sites.

Keep SPD/KKT assumptions, preconditioner construction, comparison solver calls,
iteration expectations, and residual gates inline.

## Broad-Abstraction Blockers

Do not create a cross-solver exact-RHS oracle in Day 12. The current proof
owners still differ in important ways:

- QR has rectangular, rank-deficient, sparse/dense mode, and refinement
  residual semantics.
- CG requires SPD assumptions and preconditioner-specific iteration claims.
- GMRES requires restart, side-of-preconditioning, and accepted
  nonconvergence/lucky-breakdown behavior.
- BiCGSTAB has breakdown and difficult-corpus behavior that must remain
  explicit.
- MINRES depends on symmetric SPD/indefinite fixture assumptions and
  preconditioner symmetry expectations.

## Day 12 Implementation Checklist

1. Start with QR because the helper already exists and the target changes are
   low-risk.
2. Clean `tests/test_iterative.c` CG setup using existing local helper
   patterns; keep options and assertions inline.
3. Clean bounded GMRES generated-RHS cases in the same file; leave literal
   small analytical proofs untouched.
4. Add or reuse a local BiCGSTAB generated-RHS helper for sequential
   SuiteSparse/comparison tests.
5. Add or reuse a local MINRES generated-RHS helper for sequential/sine/cosine
   pattern tests.
6. Run focused validation before the full gate:

```text
make build/test_qr build/test_iterative build/test_bicgstab build/test_minres
./build/test_qr
./build/test_iterative
./build/test_bicgstab
./build/test_minres
```

7. Because Day 12 will modify `.c` tests, finish with:

```text
make format && make lint && make test
```

## Day 11 Validation

Day 11 changes documentation only. Required validation:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_114
```

## Completion Criteria

- QR, CG, GMRES, BiCGSTAB, and MINRES each have bounded Day 12 cleanup
  targets.
- Solver-specific proof values remain visible by design.
- No broad cross-solver oracle abstraction is attempted.
- Focused and full validation commands are defined for Day 12.
