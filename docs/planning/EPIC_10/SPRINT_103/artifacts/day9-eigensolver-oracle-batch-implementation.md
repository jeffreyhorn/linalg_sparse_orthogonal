# Sprint 103 Day 9 Eigensolver Oracle Batch Implementation

## Purpose

Day 9 implements the spectral comparison batch selected on Day 8. The batch
adds focused LOBPCG and thick-restart evidence with explicit eigenvalue,
Ritz-residual, orthogonality, convergence-status, and non-claim boundaries.

## Implemented Batch

| test | fixture key | taxonomy class | reference behavior | expected result |
|---|---|---|---|---|
| `test_s103_lobpcg_laplacian30_smallest4_claim` | `lobpcg_laplacian30_smallest4_claim` | `spd-tridiag-laplacian` | closed-form 1D Laplacian eigenvalues, per-pair Ritz residuals, vector orthogonality | converges; eigenvalues match closed form within `1e-7`; residuals `< 1e-7`; max `|V^T V - I| < 1e-8` |
| enhanced `test_lobpcg_ldlt_beats_ic0_on_bcsstk04` | `lobpcg_bcsstk04_ic0_ldlt_claim` | `spd-mm-clustered` | IC(0) and LDLT preconditioned LOBPCG on `bcsstk04`, now with eigenvectors, residuals, and orthogonality | both converge; LDLT remains faster; eigenvalues agree within `1e-6`; residuals `< 1e-6`; max `|V^T V - I| < 1e-7` |
| `test_s103_thick_restart_diag12_largest4_claim` | `thick_restart_diag12_largest4_claim` | `spd-diag-separated` | exact diagonal eigenvalues, per-pair Ritz residuals, vector orthogonality, bounded peak basis | converges; top four eigenvalues match `{12, 11, 10, 9}` within `1e-10`; residuals `< 1e-10`; max `|V^T V - I| < 1e-10`; peak basis `<= 24` |

## Touched Files

| file | change |
|---|---|
| `tests/test_eigs_lobpcg.c` | added file-local orthogonality helper, new Sprint 103 Laplacian claim test, and residual/orthogonality checks to the existing `bcsstk04` IC(0) versus LDLT comparison |
| `tests/test_eigs_thick_restart.c` | added file-local diagonal builder, residual helper, orthogonality helper, and the Sprint 103 exact diagonal thick-restart claim test |
| `docs/planning/EPIC_10/SPRINT_103/WORKING_NOTES.md` | recorded Day 9 actions and validation |
| `docs/planning/EPIC_10/SPRINT_103/artifacts/day9-eigensolver-oracle-batch-implementation.md` | this implementation artifact |

No public headers, library sources, build files, fixture files, external
helpers, or public API behavior were changed.

## Focused Validation Results

| command | result |
|---|---|
| `make build/test_eigs_lobpcg build/test_eigs_thick_restart` | passed |
| `./build/test_eigs_lobpcg` | passed; 27 tests, 0 failures, 0 skips, 247 assertions |
| `./build/test_eigs_thick_restart` | passed; 21 tests, 0 failures, 0 skips, 285 assertions |

New or enhanced Sprint 103 evidence observed in the focused run:

| lane | observed result |
|---|---|
| LOBPCG Laplacian claim | 63 iterations, aggregate residual `6.208e-11` |
| LOBPCG `bcsstk04` IC(0) versus LDLT claim | IC(0): 60 iterations, residual `8.534e-09`; LDLT: 8 iterations, residual `3.020e-09` |
| thick-restart diagonal claim | 12 iterations, residual `1.781e-17`, peak basis `20` |

## Full Validation Results

Because Day 9 changed `.c` test files, the required full quality chain was
run:

| command | result |
|---|---|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed; `All tests passed.` |
| `git diff --check` | passed |
| `rg -n "[ \t]+$" tests/test_eigs_lobpcg.c tests/test_eigs_thick_restart.c tests/test_bicgstab.c docs/planning/EPIC_10/SPRINT_103` | passed; no matches |

The full test run also executed the updated spectral binaries:

| full-run test binary | result |
|---|---|
| `test_eigs_lobpcg` | passed; 27 tests, 0 failures, 0 skips, 247 assertions |
| `test_eigs_thick_restart` | passed; 21 tests, 0 failures, 0 skips, 285 assertions |

## Claim Boundaries

Day 9 earns only bounded spectral comparison evidence:

- LOBPCG satisfies closed-form eigenvalue, Ritz-residual, and orthogonality
  checks on `lobpcg_laplacian30_smallest4_claim`;
- preconditioned LOBPCG satisfies the named `bcsstk04` IC(0) versus LDLT
  comparison, including vector residual and orthogonality checks;
- thick-restart satisfies exact diagonal eigenvalue, residual, orthogonality,
  and bounded-basis checks on `thick_restart_diag12_largest4_claim`.

Day 9 does not claim:

- ARPACK, SciPy, LAPACK, NumPy, PETSc, Trilinos, or package-wide parity;
- external oracle evidence for eigensolver paths;
- portable performance superiority from local iteration counts;
- that grow-m parity is an independent oracle for thick-restart;
- broad state-of-the-art sparse eigensolver quality from these fixtures;
- any SVD rank, reconstruction, or singular-value evidence.

## Day 10 Handoff

Day 10 should close out the spectral evidence and freeze the SVD scope. The SVD
scope can reuse Day 9's separation of:

- value agreement from residual/reconstruction quality;
- basis orthogonality from convergence status;
- descriptive iteration output from performance claims.

Day 10 should not broaden spectral claims unless it adds new fixture-specific
evidence and validation.
