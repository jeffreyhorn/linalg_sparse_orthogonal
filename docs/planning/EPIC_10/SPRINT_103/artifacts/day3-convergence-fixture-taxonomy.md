# Sprint 103 Day 3 Convergence Fixture Taxonomy

## Purpose

Day 3 defines fixture classes and expected outcomes for Sprint 103 comparison
work before any iterative, eigensolver, LOBPCG, thick-restart, or SVD tests are
changed. The taxonomy separates convergence behavior from correctness
regressions and keeps later comparison claims tied to named fixture classes,
acceptance criteria, validation commands, unsupported cases, and non-claims.

## Taxonomy Rules

Every new Sprint 103 comparison artifact or test must identify:

- fixture key and fixture class;
- source or construction path;
- solver family and exact API path;
- matrix shape, symmetry, definiteness, rank, and spectral notes;
- convergence profile;
- oracle or reference behavior;
- residual, orthogonality, singular-value, reconstruction, rank, or iteration
  acceptance criteria;
- expected success, expected failure, expected non-convergence, or skip status;
- validation command;
- non-claims.

Fixture classes should be solver-neutral where possible. Family-local fixture
keys are allowed when the algorithm requires a specific structure, such as SPD
inputs for CG/LOBPCG, symmetric-indefinite KKT inputs for MINRES/shift-invert,
nonsymmetric inputs for BiCGSTAB, or rectangular/rank-deficient inputs for SVD.

## Matrix Family Catalog

| class id | fixture class | shape | symmetry | rank/spectrum expectation | primary use |
|---|---|---|---|---|---|
| `spd-diag-separated` | diagonal SPD with well-separated eigenvalues | square | symmetric | full rank; exact eigenvalues and singular values | CG, eigensolver, LOBPCG, SVD exact controls |
| `spd-tridiag-laplacian` | 1D or 2D Laplacian-style tridiagonal/stencil | square | symmetric | full rank; closed-form or structured spectrum | CG, MINRES-on-SPD, eigensolver, thick-restart, LOBPCG |
| `spd-mm-small` | small SPD Matrix Market fixture | square | symmetric | full rank; realistic sparse SPD | CG, MINRES, eigensolver, LOBPCG, SVD smoke/comparison |
| `spd-mm-clustered` | SPD corpus fixture with clustered or broad spectrum | square | symmetric | full rank; harder convergence | LOBPCG, thick-restart, eigensolver, preconditioning |
| `sym-indef-kkt` | synthetic KKT saddle-point fixture | square | symmetric | indefinite; expected nonsingular unless stated | MINRES, shift-invert eigensolver, thick-restart, LOBPCG nearest-sigma |
| `nonsym-known-solution` | small nonsymmetric square fixture with constructed `x_true` | square | unsymmetric | expected nonsingular | BiCGSTAB deterministic reference, GMRES cross-check |
| `nonsym-mm-medium` | realistic nonsymmetric Matrix Market fixture | square | unsymmetric | fixture-specific conditioning and convergence | BiCGSTAB and GMRES residual/convergence comparison |
| `ill-conditioned-scale` | scaled SPD, indefinite, or nonsymmetric fixture | square | family-specific | full rank or borderline by construction | tolerance sensitivity, stagnation, residual interpretation |
| `rank-def-square` | square rank-deficient or singular fixture | square | any | deficient by construction | expected failure, breakdown, SVD rank behavior |
| `rank-def-rect` | rectangular rank-deficient fixture | rectangular | n/a | deficient by construction | SVD rank and reconstruction evidence |
| `low-rank-rect` | rectangular low-rank fixture | rectangular | n/a | known rank and singular-value decay | SVD singular-value/rank/reconstruction follow-through |
| `malformed-or-unavailable` | invalid file/helper/platform state | n/a | n/a | n/a | skip/error contract, not numerical evidence |

## Existing Fixture Sources

| source | taxonomy class | current or candidate role |
|---|---|---|
| generated diagonal fixtures in `tests/test_eigs.c`, `tests/test_eigs_lobpcg.c`, and `tests/test_svd.c` | `spd-diag-separated` | exact eigenvalue and singular-value controls |
| generated Laplacian/tridiagonal fixtures in `tests/test_iterative.c`, `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, and `tests/test_eigs_lobpcg.c` | `spd-tridiag-laplacian` | closed-form or dense-reference spectral checks and convergence-profile controls |
| generated KKT fixtures in `tests/test_minres.c`, `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, and `tests/test_eigs_lobpcg.c` | `sym-indef-kkt` | MINRES and shift/nearest-sigma spectral coverage |
| `tests/data/suitesparse/nos4.mtx` | `spd-mm-small` | CG, eigensolver, thick-restart, LOBPCG, and SVD corpus fixture |
| `tests/data/suitesparse/bcsstk04.mtx` | `spd-mm-clustered` | SPD structural mechanics fixture; useful for preconditioned LOBPCG and spectral convergence |
| `tests/data/suitesparse/west0067.mtx` | `nonsym-mm-medium` | nonsymmetric BiCGSTAB/GMRES convergence and SVD corpus candidate |
| `tests/data/suitesparse/steam1.mtx` | `nonsym-mm-medium` or `ill-conditioned-scale` | harder nonsymmetric iterative fixture with relaxed residual expectations |
| `tests/data/suitesparse/orsirr_1.mtx` | `nonsym-mm-medium` | nonsymmetric iterative fixture where convergence may require preconditioning |
| rank-deficient builders in `tests/test_svd.c` | `rank-def-square`, `rank-def-rect`, `low-rank-rect` | SVD rank, reconstruction, and singular-value controls |
| singular and breakdown builders in `tests/test_bicgstab.c` and `tests/test_stagnation.c` | `rank-def-square` | expected failure, non-convergence, or breakdown semantics |

## Convergence Profile Classes

| profile id | profile | expected interpretation |
|---|---|---|
| `fast-exact` | converges in a small, predictable number of iterations or has exact closed-form values | correctness and reference-control fixture, not performance superiority |
| `fast-preconditioned` | preconditioned path should reduce iterations or residual versus unpreconditioned path | preconditioner effectiveness evidence, local to named fixture and tolerance |
| `slow-convergent` | reaches tolerance only with larger iteration or basis budgets | convergence evidence; iteration count is descriptive unless thresholded in advance |
| `expected-nonconvergent` | does not converge within a deliberately small budget | expected status, not a solver correctness regression |
| `stagnation-sensitive` | tight tolerance or restart/preconditioner choice can trigger stagnation | stagnation behavior evidence; must assert deterministic status or accepted alternatives |
| `restart-sensitive` | small restart or memory budget changes convergence quality or basis size | restart behavior evidence, not portable timing |
| `tolerance-sensitive` | outcome changes under tight, loose, scaled, or ill-conditioned tolerances | tolerance-policy evidence; acceptance must be recorded before implementation |
| `orthogonality-sensitive` | eigenvector or singular-vector basis quality is the main proof target | spectral/SVD evidence requiring explicit orthogonality threshold |
| `rank-sensitive` | numerical rank or near-zero singular values are the main proof target | SVD/rank evidence requiring explicit rank threshold |

## Solver-Family Fixture Mapping

| family | preferred fixture classes | preferred profiles | candidate reference behavior |
|---|---|---|---|
| CG | `spd-diag-separated`, `spd-tridiag-laplacian`, `spd-mm-small`, `ill-conditioned-scale` | `fast-exact`, `fast-preconditioned`, `slow-convergent`, `tolerance-sensitive` | constructed solution, Cholesky cross-check, true residual |
| MINRES | `spd-tridiag-laplacian`, `sym-indef-kkt`, `ill-conditioned-scale` | `fast-exact`, `slow-convergent`, `fast-preconditioned`, `tolerance-sensitive` | LDLT/GMRES cross-check, constructed solution, true residual |
| BiCGSTAB | `nonsym-known-solution`, `nonsym-mm-medium`, `ill-conditioned-scale`, `rank-def-square` | `fast-exact`, `slow-convergent`, `stagnation-sensitive`, `expected-nonconvergent` | constructed solution, LU/GMRES cross-check, optional external dense solve after helper boundary |
| grow-m eigensolver | `spd-diag-separated`, `spd-tridiag-laplacian`, `sym-indef-kkt`, `spd-mm-small` | `fast-exact`, `slow-convergent`, `orthogonality-sensitive`, `tolerance-sensitive` | closed-form eigenvalues, dense tridiagonal reference, eigenpair residual |
| thick-restart eigensolver | `spd-diag-separated`, `spd-tridiag-laplacian`, `sym-indef-kkt`, `spd-mm-clustered` | `restart-sensitive`, `slow-convergent`, `orthogonality-sensitive` | closed-form or dense tridiagonal reference, grow-m parity only when declared non-independent |
| LOBPCG | `spd-diag-separated`, `spd-tridiag-laplacian`, `spd-mm-small`, `spd-mm-clustered`, `sym-indef-kkt` for nearest-sigma | `fast-exact`, `fast-preconditioned`, `orthogonality-sensitive`, `tolerance-sensitive` | closed-form eigenvalues, Lanczos comparison, eigenpair residual and orthogonality |
| SVD | `spd-diag-separated`, `rank-def-square`, `rank-def-rect`, `low-rank-rect`, `nonsym-mm-medium` | `rank-sensitive`, `orthogonality-sensitive`, `tolerance-sensitive` | exact singular values, reconstruction error, trace invariant, optional dense SVD after scope freeze |

## Acceptance Criteria by Evidence Type

Thresholds below are defaults for fixture design. A later implementation
artifact may tighten or relax them only if it records the reason before code is
changed.

| evidence type | default criterion | notes |
|---|---|---|
| iterative true residual | relative residual below fixture tolerance, usually `1e-8` to `1e-10` for well-conditioned generated fixtures and relaxed to `1e-4` to `1e-6` for hard corpus fixtures | use `tf_relative_residual_l2` or existing family-local residual helper |
| constructed solution difference | max absolute solution difference within `1e-8` to `1e-10` for small deterministic fixtures | not appropriate for ill-conditioned fixtures unless tolerance is scaled |
| convergence status | `converged`, `not converged`, `stagnated`, or `breakdown` status must match fixture expectation | expected non-convergence is not a failure when declared in taxonomy artifact |
| iteration count | threshold only when the fixture is deterministic and the algorithm contract requires it | otherwise record as descriptive convergence evidence |
| eigenvalue agreement | absolute or relative eigenvalue error within `1e-7` to `1e-10`, scaled by fixture magnitude | diagonal and tridiagonal references can usually use tighter thresholds |
| eigenpair residual | `||Av - lambda v|| / max(1, |lambda|)` below declared tolerance, commonly `1e-8` to `1e-10` | must be separate from eigenvalue ordering checks |
| vector orthogonality | `max |Q^T Q - I|` below declared tolerance, commonly `1e-8` to `1e-12` | applies to eigenvectors, LOBPCG blocks, and SVD U/V |
| SVD singular values | sorted singular values match exact/dense reference within declared absolute or relative tolerance | near-zero singular values need rank threshold |
| SVD reconstruction | `||A - U Sigma V^T|| / max(1, ||A||)` below declared tolerance | full/economy/partial modes must name reconstruction scope |
| rank behavior | numerical rank matches declared threshold policy | rank threshold must be visible before implementation |
| skip or helper unavailable | skip with reason string; never counted as oracle pass | use Sprint 102 status/reason conventions when helpers are reused |

## Expected Status and Failure Rules

| case | expected status | claim impact |
|---|---|---|
| SPD input to CG/LOBPCG/eigensolver | success if tolerance and iteration budget are fixture-appropriate | supports bounded fixture claim only |
| symmetric indefinite input to MINRES | success or declared tolerance-bound non-convergence | supports MINRES fixture claim only |
| nonsymmetric input to BiCGSTAB | success, non-convergence, stagnation, or breakdown depending on fixture class | must not be generalized beyond named matrix |
| singular or rank-deficient square input to iterative solver | expected failure, breakdown, or non-convergence if declared | not a correctness regression |
| small restart or memory budget in GMRES/thick-restart | expected slower convergence or bounded residual behavior | restart evidence, not timing superiority |
| unavailable external helper | skip | no oracle claim earned |
| malformed input | parser or constructor error | not numerical solver evidence |
| SVD rank-deficient input | success with rank/reconstruction semantics | rank claim only if threshold is declared |

## Day 4 Handoff

Day 4 should evaluate helper and reporting boundaries against this taxonomy:

- BiCGSTAB should be the first iterative implementation candidate only if its
  helper/status behavior is frozen before code changes.
- LOBPCG and thick-restart should keep residual and orthogonality criteria
  separate from iteration-count descriptions.
- SVD should not start an external dense comparison lane until rank,
  reconstruction, and singular-value thresholds are explicitly scoped.
- Any helper reuse from Sprint 102 must preserve explicit skip/error/success
  states and must not turn missing external tools into passing evidence.

## Non-Claims Preserved

This taxonomy does not claim:

- any new Sprint 103 comparison implementation has landed;
- external helper parity exists for BiCGSTAB, eigensolver, LOBPCG,
  thick-restart, or SVD paths;
- grow-m parity is an independent external oracle for thick-restart;
- local iteration counts are portable performance evidence;
- one fixture class proves broad state-of-the-art sparse solver quality;
- rank or residual thresholds are universal outside their named fixture
  classes.

## Day 3 Conclusion

Sprint 103 now has a fixture taxonomy that later tests and artifacts can cite
before implementation. The taxonomy supports the Day 2 ranked queue while
keeping convergence, correctness, orthogonality, restart, preconditioning,
rank, and expected-failure behavior separate.
