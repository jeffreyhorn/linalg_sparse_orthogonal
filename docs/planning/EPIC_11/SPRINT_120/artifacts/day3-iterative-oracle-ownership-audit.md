# Sprint 120 Day 3 Iterative Oracle Ownership Audit

## Purpose

Day 3 audits the iterative solver proof owners before any source movement or
helper extraction. The audit covers generated-RHS helpers, convergence
contracts, residual expectations, progress/callback-like behavior,
preconditioner ownership, block-solver coverage, and giant-test split
candidates for CG, GMRES, BiCGSTAB, and MINRES.

## Scope

| Surface | Current size | Day 3 role |
|---|---:|---|
| `tests/test_iterative.c` | 2,924 lines | CG, GMRES, SuiteSparse comparisons, right-preconditioned GMRES, matrix-free CG/GMRES, public handle helper inclusion. |
| `tests/test_bicgstab.c` | 1,826 lines | BiCGSTAB, block BiCGSTAB, SuiteSparse comparisons, preconditioner behavior, matrix-free callback behavior. |
| `tests/test_minres.c` | 1,649 lines | MINRES, block MINRES, symmetric indefinite/KKT coverage, preconditioner behavior, direct and GMRES comparisons. |
| `tests/test_iterative_handle_helpers.h` | 195 lines | Public handle validation and reuse/growth proofs included by `test_iterative.c`. |
| `tests/test_solver_helpers.h` | 200 lines | Shared sparse residual and external-reference helpers used by solver proof files. |

No source split is selected by this artifact. It records the owners and
constraints needed for Day 4 fixture design and Day 5 candidate ranking.

## Iterative Oracle Owner Table

| Solver / proof family | Current proof owner | Oracle inputs | Convergence and failure ownership | Split or helper risk |
|---|---|---|---|---|
| CG basic SPD behavior | `tests/test_iterative.c` CG tests using `build_spd_tridiag`, `build_identity`, `build_laplacian_2d`, `compute_rhs`, and exact RHS helpers | Generated exact RHS from known solutions, identity/diagonal/tridiagonal/Laplacian fixtures, zero RHS, initial guess fixtures | Solver-local tolerance checks, exact initial guess convergence, zero-RHS behavior, maximum-iteration failure, nonsymmetric and indefinite rejection/failure expectations | Matrix and RHS builders may be shared only if each CG test keeps its own tolerance, expected iteration, and failure interpretation visible. |
| CG SuiteSparse and direct comparisons | `tests/test_iterative.c` CG SuiteSparse and CG-vs-Cholesky tests | `nos4`, `bcsstk04`, generated RHS, Cholesky reference solve | Residual accuracy, initial-guess behavior, and direct-solver agreement remain CG proof claims rather than generic fixture claims | Direct comparison helpers must not erase which matrix, tolerance, or fallback behavior was accepted for each case. |
| CG preconditioner behavior | `tests/test_iterative.c` diagonal-preconditioner tests and preconditioned Laplacian coverage | Diagonal/preconditioner callback data plus generated RHS | Preconditioner application validity, iteration-count expectations, and residual tolerance | Preconditioner helper extraction is possible, but iteration improvement claims must stay test-local. |
| GMRES basic and restart behavior | `tests/test_iterative.c` GMRES tests using unsymmetric builders and exact RHS helpers | Small unsymmetric systems, identity, zero RHS, larger tridiagonal, exact initial guess, restart variants, Arnoldi correctness | Restart, unrestarted, lucky breakdown, small Krylov, max-iteration, default-option, and verbose behavior | Restart helpers are attractive split candidates, but restart-specific expected outcomes must remain in the named GMRES tests. |
| GMRES SuiteSparse and cross-solver comparisons | `tests/test_iterative.c` west0067, steam1, orsirr_1, GMRES-vs-LU, and GMRES-vs-CG tests | SuiteSparse fixtures, generated RHS, LU or CG reference paths | Matrix-specific tolerances, relaxed residuals, accepted convergence/nonconvergence outcomes, and cross-solver agreement | High split value; high claim risk. Any split must preserve matrix-specific tolerance comments and accepted failure paths. |
| GMRES right-preconditioner behavior | `tests/test_iterative.c` right-preconditioned GMRES tests | Diagonal and ILU right-preconditioners, left/right residual comparison fixtures | Reported residual versus true residual distinction, default right-preconditioner path, ILU setup and cleanup | Must remain visibly GMRES-specific because the right/left residual distinction is a solver contract. |
| CG/GMRES public handle behavior | `tests/test_iterative_handle_helpers.h`, included from `tests/test_iterative.c` | Prepared handles, on-demand handles, reuse, growth, validation inputs | Handle validation, workspace reuse, growth semantics, and cleanup ownership | Already partly separated. Future cleanup should reduce dependency on `test_iterative.c` static builders only after Day 4 placement rules exist. |
| CG/GMRES matrix-free callbacks | `tests/test_iterative.c` matrix-free CG/GMRES tests | `sparse_matvec_cb`, diagonal/scalar callbacks, failing callback, callback context | Callback output equivalence to sparse-matrix solve, callback failure propagation, null callback validation, zero-RHS behavior | Callback helper extraction is possible, but each solver's failure propagation and residual semantics must stay explicit. |
| BiCGSTAB basic and exact-RHS behavior | `tests/test_bicgstab.c` basic BiCGSTAB tests and sequential RHS helpers | Identity, diagonal, SPD/unsymmetric tridiagonal, known 3x3/5x5 systems, generated sequential RHS | True residual, already converged, max-iteration, defaults, result fields, nonzero/random/near-solution initial guesses | Exact RHS helpers can inform Day 4 shared fixtures, but BiCGSTAB breakdown and result-field expectations stay solver-local. |
| BiCGSTAB preconditioners and SuiteSparse | `tests/test_bicgstab.c` ILU/ILUT, west0067, steam1, orsirr_1, and Sprint 103 reference tests | ILU/ILUT preconditioners, SuiteSparse fixtures, GMRES/LU reference comparisons | Fewer-iteration expectations, known-solution preconditioned solves, small-budget expected nonconvergence, accepted SuiteSparse outcomes | Strong split candidate. Preserve matrix-specific residual budgets and explicit expected nonconvergence labels. |
| Block BiCGSTAB | `tests/test_bicgstab.c` block BiCGSTAB tests | Multiple RHS blocks, mixed convergence columns, preconditioned blocks, error-propagation cases | Per-column convergence, result aggregation, `nrhs` validation, preconditioner error propagation, single-RHS equivalence | Good focused split candidate because block ownership is distinct, but block result aggregation must not be hidden by generic helpers. |
| Matrix-free BiCGSTAB | `tests/test_bicgstab.c` matrix-free tests | Sparse callback, scaled identity callback, failing callback, preconditioned matrix-free case | Matrix-free equivalence, callback error propagation, null callback, zero-RHS, zero-size validation | Shares concepts with CG/GMRES matrix-free tests but keeps BiCGSTAB-specific status/result semantics. |
| MINRES SPD, indefinite, and KKT behavior | `tests/test_minres.c` MINRES base tests and exact RHS helpers | SPD tridiagonal, symmetric indefinite tridiagonal, KKT fixtures, generated sequential/sine/scaled RHS | SPD-vs-CG agreement, indefinite convergence, KKT behavior, zero-RHS, already converged, one-by-one positive/negative systems | Matrix/RHS builders may be shared after Day 4, but symmetric-indefinite interpretation must remain MINRES-local. |
| MINRES preconditioners and direct comparisons | `tests/test_minres.c` IC, Jacobi, exact preconditioner, LDLT, and GMRES comparison tests | IC/Jacobi callback data, LDLT reference solve, GMRES comparison on large systems | Preconditioner residuals, fewer-iteration expectations where asserted, direct-solver agreement, large-system tolerance | Split candidate if preconditioner fixtures remain separated from solver-local convergence and comparison thresholds. |
| Block MINRES | `tests/test_minres.c` block MINRES tests | Multi-RHS SPD/indefinite/KKT-style blocks, zero columns, many-RHS cases, preconditioned block paths | Per-column behavior, all-zero RHS handling, sequential equivalence, preconditioner behavior, nonsquare validation | Strong split candidate. Keep block-specific failure modes visible at test names and assertions. |

## Progress, Callback, and Lifecycle Proof Map

| Proof type | Current owner | Behavior that must remain visible |
|---|---|---|
| Public iterative handle reuse and growth | `tests/test_iterative_handle_helpers.h` | CG validates prepared/on-demand handle reuse; GMRES and MINRES validate prepare/reuse/growth behavior and cleanup. |
| Matrix-free matvec callbacks | `tests/test_iterative.c` and `tests/test_bicgstab.c` | Sparse-matrix equivalence, scaled/diagonal callback behavior, callback context ownership, null callback validation, and failing callback propagation. |
| Preconditioner callbacks | `tests/test_iterative.c`, `tests/test_bicgstab.c`, and `tests/test_minres.c` | Diagonal/Jacobi/ILU/IC callback semantics, solver-specific residual interpretation, setup/cleanup lifecycle, and expected iteration effects. |
| Verbose/progress-like execution paths | `tests/test_iterative.c` | CG and GMRES verbose tests preserve the observable progress/reporting path without turning it into a generic success-only helper. |
| Block-solver aggregation lifecycle | `tests/test_bicgstab.c` and `tests/test_minres.c` | Per-column status, aggregate result fields, mixed convergence, zero-column behavior, and cleanup on failure. |

Day 3 did not find a single reusable progress-callback abstraction that can be
split safely without Day 4 design. The current callback evidence is primarily
matrix-free, preconditioner, verbose-path, handle-lifecycle, and block
aggregation proof.

## Convergence, Tolerance, and Failure Notes

| Solver | Notes |
|---|---|
| CG | Keep exact initial guess, zero RHS, maximum-iteration failure, residual accuracy, nonsymmetric/indefinite behavior, diagonal preconditioner, Laplacian, SuiteSparse, and Cholesky-comparison thresholds local to CG tests. |
| GMRES | Keep restart/unrestarted distinctions, lucky-breakdown behavior, Arnoldi correctness, SuiteSparse matrix-specific tolerances, accepted relaxed outcomes, right-preconditioned reported-vs-true residual checks, and LU/CG comparison tolerances visible in GMRES-named tests. |
| BiCGSTAB | Keep true-residual checks, known-solution references, ILU/ILUT iteration expectations, small-budget expected nonconvergence, numerical hardening, matrix-free error propagation, and block aggregation behavior local to BiCGSTAB tests. |
| MINRES | Keep SPD/indefinite/KKT distinctions, SPD-vs-CG checks, LDLT/GMRES comparison tolerances, scaled tolerance behavior, ill-conditioned expectations, early Lanczos termination, IC/Jacobi/exact preconditioner semantics, and block behavior local to MINRES tests. |

## Giant-Test Hotspot Inventory

| Candidate | Why it is hot | Initial Day 5 ranking input |
|---|---|---|
| `tests/test_iterative.c` CG and GMRES combined ownership | One file owns CG, GMRES, SuiteSparse references, right-preconditioned GMRES, matrix-free paths, and public handle helper inclusion. | Rank by isolating GMRES SuiteSparse/restart/right-preconditioner proof blocks first; defer any broad CG/GMRES shared fixture until Day 4 boundaries are approved. |
| `tests/test_bicgstab.c` block and matrix-free ownership | Single file owns scalar, preconditioned, SuiteSparse, external-reference-style, block, numerical-hardening, and matrix-free coverage. | Block BiCGSTAB and matrix-free BiCGSTAB are strong focused split candidates if source-list and CTest impact are planned. |
| `tests/test_minres.c` block/direct/preconditioner ownership | Single file owns SPD, indefinite, KKT, direct comparisons, preconditioners, scaled/ill-conditioned behavior, and block coverage. | Block MINRES and preconditioner/direct comparison groups are focused split candidates after Day 4 designs symmetric-fixture ownership. |
| `tests/test_iterative_handle_helpers.h` dependency on local builders | Helper is separate, but it depends on static builders from `test_iterative.c`. | Do not move independently until Day 4 decides whether handle fixtures belong in a shared helper or solver-local test file. |

## Day 4 Shared-Fixture Design Inputs

- Exact-RHS builders should support sequential, sinusoidal, and scaled variants,
  but should not own solver-specific tolerances or convergence interpretation.
- Residual helpers should remain measurement utilities. Reported residual
  versus true residual checks, especially for right-preconditioned GMRES and
  BiCGSTAB true-residual tests, need solver-local assertions.
- Matrix builders can be shared for identity, diagonal, SPD tridiagonal,
  unsymmetric tridiagonal, symmetric indefinite tridiagonal, KKT, and
  Laplacian fixtures if naming keeps matrix semantics visible.
- Callback fixtures should distinguish matvec callbacks, failing matvec
  callbacks, preconditioner callbacks, and handle lifecycle because they prove
  different contracts.
- Block RHS fixtures should preserve column ordering, zero-column behavior,
  mixed convergence, per-column status, and aggregate result semantics.
- SuiteSparse and direct/cross-solver comparison helpers must carry matrix
  names, reference solver names, and matrix-specific tolerance budgets at the
  call site.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 1 iterative audit inputs are complete | Complete: CG, GMRES, BiCGSTAB, MINRES, block, callback, preconditioner, handle, SuiteSparse, and cross-solver proof owners were inventoried. |
| Every iterative candidate has named proof owners and failure modes | Complete: each candidate is mapped to current files, oracle inputs, convergence/failure ownership, and split risk. |
| Shared helper opportunities do not hide solver-specific convergence contracts | Complete: Day 4 inputs explicitly separate reusable fixtures from solver-local tolerances, residual interpretation, convergence outcomes, and failure modes. |

## Non-Claims

This artifact does not claim broader iterative parity, complete external-oracle
coverage, improved performance, package support, public API expansion, or
state-of-the-art validation. It is an ownership audit and design input for
bounded Sprint 120 proof-owner work.
