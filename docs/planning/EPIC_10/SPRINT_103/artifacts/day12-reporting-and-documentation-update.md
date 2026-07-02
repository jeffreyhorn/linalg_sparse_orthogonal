# Sprint 103 Day 12 Reporting and Documentation Update

## Purpose

Day 12 documents how Sprint 103 comparison evidence should be interpreted
before closeout. The update keeps convergence, residual, orthogonality, rank,
and comparison wording bounded to the fixtures and test owners that actually
carry evidence.

## Documentation Updated

- `docs/maintainer_guide.md`
  - added `Sprint 103 Iterative, Spectral, and SVD Evidence Boundary Snapshot`
  - documented BiCGSTAB, LOBPCG, thick-restart eigensolver, and SVD evidence
    boundaries
  - defined residual and quality interpretation by solver family
  - added wording rules for public and maintainer documentation that reference
    Sprint 103 evidence

## Evidence Boundary Summary

| Family / lane | Maintained evidence owner | Evidence type | Boundary |
|---|---|---|---|
| BiCGSTAB nonsymmetric convergence | `tests/test_bicgstab.c` | deterministic known solution plus internal consistency | LU-backed deterministic fixture, internal GMRES(30)+ILU comparison on `steam1`, and expected non-convergence budget boundary |
| LOBPCG closed-form and preconditioned fixtures | `tests/test_eigs_lobpcg.c` | deterministic closed-form plus internal preconditioner comparison | Laplacian eigenvalues, Ritz residuals, orthogonality, and fixture-local LDLT-versus-IC(0) behavior |
| Thick-restart exact diagonal fixture | `tests/test_eigs_thick_restart.c` | deterministic exact fixture | exact diagonal eigenpairs, Ritz residuals, orthogonality, and bounded peak-basis behavior |
| SVD deterministic rank and full-UV fixture | `tests/test_svd.c` | deterministic exact fixture | singular values, reconstruction residual, U/Vt orthogonality, and explicit rank thresholds |

## Residual and Quality Interpretation

- BiCGSTAB residuals are solve residuals on named nonsymmetric fixtures.
- LOBPCG and thick-restart residuals are Ritz residuals for requested
  eigenpairs and should not be reworded as package-level eigensolver parity.
- SVD reconstruction residuals are test-computed quality checks, separate from
  singular-value and rank assertions.
- Orthogonality checks are fixture-local quality criteria.
- Iteration counts and preconditioner deltas remain diagnostics unless a test
  explicitly gates them.

## Explicit Non-Claims

Sprint 103 does not add external PETSc, SciPy, Trilinos, ARPACK, LAPACK, NumPy,
or broad ecosystem parity claims for iterative solvers, eigensolvers, or SVD.
External dense-reference evidence remains limited to the direct-solver lanes
documented in the Sprint 102 maintainer-guide snapshot.

## Validation

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/maintainer_guide.md docs/planning/EPIC_10/SPRINT_103`:
  passed; no matches.
