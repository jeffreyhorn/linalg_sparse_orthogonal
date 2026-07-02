# Sprint 103 Day 2 Solver Family Comparison Audit

## Purpose

Day 2 inventories current comparison evidence for CG, MINRES, BiCGSTAB,
eigensolver, thick-restart, LOBPCG, and SVD paths before new fixtures, helpers,
or tests are added. The audit ranks solver families by comparison weakness,
user impact, numerical risk, and validation cost so later Sprint 103 work can
start from explicit evidence gaps rather than broad parity assumptions.

## Source Files Reviewed

| file | primary coverage | `RUN_TEST` count |
|---|---|---:|
| `tests/test_iterative.c` | CG, GMRES, matrix-free CG/GMRES, restart, preconditioning, SuiteSparse residuals | 80 |
| `tests/test_minres.c` | MINRES, block MINRES, SPD/indefinite systems, preconditioning, direct-solver cross-checks | 43 |
| `tests/test_bicgstab.c` | BiCGSTAB, ILU preconditioning, SuiteSparse, GMRES/direct comparisons, breakdown cases | 58 |
| `tests/test_stagnation.c` | CG, GMRES, MINRES, BiCGSTAB stagnation, residual history, callbacks, breakdown flags | 46 |
| `tests/test_eigs.c` | public eigensolver, shift-invert, SuiteSparse, refinement, SVD cross-check | 31 |
| `tests/test_eigs_thick_restart.c` | thick-restart backend, restart state, cross-backend parity, bounded-memory cases | 20 |
| `tests/test_eigs_lobpcg.c` | LOBPCG, orthonormalization, preconditioning, nearest-sigma, dispatch behavior | 26 |
| `tests/test_svd.c` | SVD, Golub-Kahan, bidiagonal SVD, rank, reconstruction, partial/full modes | 97 |
| `tests/test_sprint13_integration.c` | iterative cross-solver integration, preconditioner comparison, block MINRES | 14 |
| `tests/test_sprint29_integration.c` | eigs refinement, full-mode SVD reconstruction, cancellation integration | 3 |

## Evidence Type Legend

| evidence type | meaning |
|---|---|
| internal consistency | validates API behavior, residuals, invariants, edge cases, or result fields without an independent oracle |
| deterministic reference | compares against exact known values, closed-form spectra, constructed solutions, or direct algebraic identities |
| direct-solver cross-check | compares an iterative or spectral result against an existing direct solver or related solver path |
| fixture corpus | uses Matrix Market or generated fixture families to exercise realistic sparse behavior |
| property or invariant | checks residual, orthogonality, reconstruction, trace, rank, convergence status, or breakdown invariants |
| smoke | verifies bounded behavior without claiming deep numerical comparison |
| external helper | invokes an out-of-process reference implementation; no Day 2 target family currently has this evidence |

## Current Evidence Inventory

| family | current evidence | current comparison weakness |
|---|---|---|
| CG | Known-solution SPD systems, diagonal/tridiagonal/Laplacian fixtures, SuiteSparse `nos4`/`bcsstk04`, Cholesky cross-checks, preconditioner checks, tolerance and initial-guess behavior, stagnation and residual-history coverage. | Strong internal and direct-solver cross-check coverage, but no external reference lane and no unified convergence-profile fixture taxonomy. |
| MINRES | SPD and symmetric-indefinite fixtures, KKT cases, Jacobi/IC/ILU-style preconditioning, LDLT and GMRES cross-checks, block MINRES, scaled tolerance, stagnation and breakdown coverage. | Good breadth, but external reference evidence is absent and convergence expectations are distributed across several tests without one comparison artifact. |
| BiCGSTAB | Known-solution nonsymmetric systems, true residual checks, LU/GMRES comparisons, ILU preconditioner checks, SuiteSparse `west0067`/`steam1`/`orsirr_1`, high-condition and breakdown cases, stagnation coverage. | Highest iterative comparison gap: nonsymmetric convergence behavior is user-visible, but no external oracle and no concise residual/stagnation fixture set. |
| eigensolver grow-m path | Diagonal and tridiagonal closed-form spectra, eigenvector residual checks, shift-invert cases, SuiteSparse smoke/parity, SVD cross-check through `A^T A`, refinement behavior. | Strong deterministic references for small structured spectra, but no external ARPACK/SciPy-style oracle and limited artifact-level explanation of residual and eigenvector acceptance. |
| thick-restart eigensolver | Arrowhead/tridiagonal spectrum checks, restart-state round trips, grow-m parity, eigenvector residual checks, bounded-memory SuiteSparse cases, auto-dispatch checks. | Good internal parity with grow-m, but no independent oracle; restart behavior is validated mostly against the project's own grow-m backend. |
| LOBPCG | Orthonormalization checks, diagonal and Laplacian closed-form spectra, SuiteSparse `nos4`, deterministic starts, block-size stability, IC/LDLT preconditioning, nearest-sigma and dispatch behavior. | Valuable preconditioned spectral path but comparison evidence is mostly deterministic/internal; lacks external LOBPCG reference and consolidated convergence-profile artifact. |
| SVD | Golub-Kahan extraction, bidiagonal SVD, trace invariants, rank-deficient and low-rank matrices, reconstruction and orthogonality checks, SuiteSparse fixtures, partial/full mode integration, low-rank reconstruction. | Broad invariant coverage but no external LAPACK/NumPy/SciPy comparison lane; rank and reconstruction evidence should share a taxonomy with spectral work before expansion. |

## User Impact and Risk Scoring

Scores use 1 as low and 5 as high.

| family | user impact | comparison gap | numerical risk | validation cost | rationale |
|---|---:|---:|---:|---:|---|
| BiCGSTAB | 5 | 5 | 5 | 3 | Nonsymmetric iterative solves are high-value, convergence is fragile, and current evidence lacks external or tightly bounded deterministic comparison artifacts. |
| LOBPCG | 4 | 4 | 5 | 3 | Preconditioned eigen workflows are strategically important and numerically subtle; existing tests are good but still mostly self-contained. |
| thick-restart eigensolver | 4 | 4 | 4 | 3 | Memory-bounded eigensolver behavior matters to large sparse use, but current comparison is mostly grow-m parity. |
| SVD | 4 | 4 | 4 | 4 | SVD has broad invariant coverage and large test ownership; external singular-value/rank comparison would improve trust but needs careful fixture scoping. |
| MINRES | 4 | 3 | 4 | 3 | Indefinite symmetric solves are important and well-covered internally; the main gap is consolidated external/deterministic comparison evidence. |
| CG | 5 | 2 | 3 | 2 | CG is highly visible but already has strong exact, residual, SuiteSparse, preconditioner, and direct-solver cross-check evidence. |
| grow-m eigensolver | 4 | 3 | 4 | 3 | Core eigensolver has useful closed-form and residual evidence; artifact-level acceptance criteria should improve before new external claims. |

## Validation Cost Notes

| candidate area | likely touched files | focused validation | cost note |
|---|---|---|---|
| BiCGSTAB oracle or deterministic batch | `tests/test_bicgstab.c`, optional helper under `tests/` | `make build/test_bicgstab && ./build/test_bicgstab` | Moderate runtime and concentrated file ownership; no new executable preferred. |
| iterative convergence taxonomy and reporting | planning docs, possibly `tests/test_stagnation.c` or helper headers | docs hygiene; if code touched, focused stagnation/iterative tests plus full gate | High claim value; can start docs-only on Day 3. |
| LOBPCG comparison batch | `tests/test_eigs_lobpcg.c`, possible fixture/helper reuse | `make build/test_eigs_lobpcg && ./build/test_eigs_lobpcg` | Moderate runtime; acceptance criteria must separate residual, orthogonality, and iteration behavior. |
| thick-restart comparison batch | `tests/test_eigs_thick_restart.c` | `make build/test_eigs_thick_restart && ./build/test_eigs_thick_restart` | Moderate cost; independent oracle harder than grow-m parity. |
| SVD comparison follow-through | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, possible helper | `make build/test_svd && ./build/test_svd` | Higher cost due to 97-test owner and rank/reconstruction tolerance sensitivity. |
| CG comparison refresh | `tests/test_iterative.c`, `tests/test_stagnation.c` | focused iterative/stagnation tests | Lower comparison gap; best as shared fixture consumer rather than first implementation target. |
| MINRES comparison refresh | `tests/test_minres.c`, `tests/test_sprint13_integration.c` | focused MINRES and Sprint 13 tests | Useful if BiCGSTAB/LOBPCG fixtures reveal reusable residual reporting. |

## Ranked Sprint 103 Expansion Queue

1. **BiCGSTAB deterministic or external-reference comparison lane.**
   Highest gap-to-impact ratio. It should use nonsymmetric fixtures with known
   solutions, explicit residual thresholds, stagnation/breakdown expectations,
   and optional comparison against an external dense solve only if helper scope
   is frozen first.
2. **LOBPCG residual and orthogonality comparison lane.**
   High user value for preconditioned eigen workflows. It should focus on
   closed-form diagonal/Laplacian and one SuiteSparse fixture before any
   external package wording is considered.
3. **Thick-restart independent fixture comparison lane.**
   Current evidence leans on grow-m parity. A bounded closed-form or
   deterministic spectral fixture would strengthen restart-specific claims
   without requiring broad ARPACK parity.
4. **SVD singular-value/rank/reconstruction follow-through.**
   SVD has broad internal evidence but needs a cleaner comparison artifact.
   It should wait for Day 3 fixture taxonomy and Day 10 spectral/SVD overlap
   scoping.
5. **MINRES consolidated comparison artifact.**
   Existing MINRES coverage is strong enough that the next improvement is
   mostly organization and fixture taxonomy reuse rather than first
   implementation priority.
6. **CG convergence-profile consumer lane.**
   CG is visible and important, but its current direct-solver and residual
   coverage is relatively strong. It should consume shared fixture/reporting
   infrastructure after higher-gap iterative lanes are addressed.
7. **Grow-m eigensolver documentation and residual interpretation.**
   Core eigensolver evidence is broad enough that Day 12 documentation may
   deliver more immediate value than a first implementation slot, unless Day 8
   spectral design identifies a low-cost independent fixture.

## Non-Claims Preserved

This audit does not claim:

- broad external parity for iterative, eigensolver, LOBPCG, thick-restart, or
  SVD paths;
- ARPACK, SciPy, LAPACK, NumPy, SuiteSparse, or PETSc equivalence;
- portable performance superiority from iteration counts or local timings;
- external helper coverage for any Sprint 103 target family;
- that current internal cross-checks are independent external oracles;
- that one deterministic fixture proves state-of-the-art sparse solver quality.

## Day 2 Conclusion

All Sprint 103 target families are classified. The strongest implementation
priority is BiCGSTAB comparison evidence, followed by LOBPCG and
thick-restart spectral evidence. SVD should wait for shared fixture and
reporting taxonomy so rank, reconstruction, singular-value, and orthogonality
claims remain bounded. CG and MINRES remain important but currently have lower
comparison gaps than BiCGSTAB and spectral paths.
