# Sprint 103 Day 13 Validation and Evidence Reconciliation

## Purpose

Day 13 reconciles the Sprint 103 implementation against the Day 2 ranking,
confirms that each implemented comparison has an artifact and claim boundary,
records complete validation, and freezes the remaining gap queue before
closeout handoff.

## Validation Results

| command | result |
|---|---|
| `make format && make lint && make test` | passed; final output reported `All tests passed.` |
| `git diff --check` | passed |
| `rg -n "[ \t]+$" tests/test_bicgstab.c tests/test_eigs_lobpcg.c tests/test_eigs_thick_restart.c tests/test_svd.c docs/maintainer_guide.md docs/planning/EPIC_10/SPRINT_103` | passed; no matches |

The full gate was required because Sprint 103 touched C test files. The
branch-level validation includes the Sprint 103 BiCGSTAB, LOBPCG,
thick-restart, and SVD comparisons inside the normal test suite.

## Day 2 Ranking Reconciliation

| Day 2 rank | family or lane | Sprint 103 disposition | Evidence owner | Artifact trail |
|---:|---|---|---|---|
| 1 | BiCGSTAB deterministic or external-reference comparison lane | implemented as bounded deterministic and internal-consistency evidence | `tests/test_bicgstab.c` | Day 5 design, Day 6 implementation, Day 7 closeout, Day 12 boundary |
| 2 | LOBPCG residual and orthogonality comparison lane | implemented as closed-form Laplacian and preconditioned `bcsstk04` evidence | `tests/test_eigs_lobpcg.c` | Day 8 design, Day 9 implementation, Day 10 closeout, Day 12 boundary |
| 3 | Thick-restart independent fixture comparison lane | implemented as exact diagonal residual, orthogonality, and bounded-basis evidence | `tests/test_eigs_thick_restart.c` | Day 8 design, Day 9 implementation, Day 10 closeout, Day 12 boundary |
| 4 | SVD singular-value/rank/reconstruction follow-through | implemented as one deterministic full-UV, rank-threshold, and reconstruction fixture | `tests/test_svd.c` | Day 10 scope, Day 11 implementation, Day 12 boundary |
| 5 | MINRES consolidated comparison artifact | deferred | existing `tests/test_minres.c` and `tests/test_sprint13_integration.c` | Day 2 audit and Day 7 rerank |
| 6 | CG convergence-profile consumer lane | deferred | existing `tests/test_iterative.c` and `tests/test_stagnation.c` | Day 2 audit and Day 7 rerank |
| 7 | Grow-m eigensolver documentation and residual interpretation | partially covered by Day 12 documentation rules; no new implementation | existing `tests/test_eigs.c` | Day 2 audit and Day 12 boundary |

## Implemented Evidence Map

| lane | implemented comparison | evidence type | documented claim boundary |
|---|---|---|---|
| BiCGSTAB known nonsymmetric solve | constructed solution plus LU cross-check and true residual | deterministic fixture plus direct-solver cross-check | named-fixture iterative evidence only; no external PETSc, SciPy, Trilinos, or broad nonsymmetric parity |
| BiCGSTAB `steam1` preconditioned comparison | BiCGSTAB+ILU and GMRES(30)+ILU residual comparison | internal consistency | GMRES comparison is not an independent oracle; iteration counts are descriptive |
| BiCGSTAB small-budget non-convergence | expected `SPARSE_ERR_NOT_CONVERGED` with finite residual | boundary/failure behavior | documents one expected non-convergence lane, not broad stagnation proof |
| LOBPCG Laplacian smallest eigenpairs | closed-form eigenvalues, Ritz residuals, and vector orthogonality | deterministic fixture | fixture-local eigensolver evidence; no ARPACK or package parity |
| LOBPCG `bcsstk04` preconditioner comparison | IC(0) and LDLT residual/orthogonality gates with LDLT iteration improvement on the named fixture | internal preconditioner comparison | no portable preconditioner superiority or performance claim |
| Thick-restart diagonal largest eigenpairs | exact diagonal eigenvalues, Ritz residuals, orthogonality, and bounded peak basis | deterministic fixture | restart-specific fixture evidence; no broad memory or ARPACK parity claim |
| SVD diagonal rank/full-UV claim | exact singular values, reconstruction residual, U/Vt orthogonality, and explicit rank thresholds | deterministic fixture | no LAPACK, NumPy, SciPy, or broad SVD parity; no external dense SVD helper lane |

## Artifact and Boundary Traceability

| deliverable | artifact or documentation owner | status |
|---|---|---|
| authoritative inputs and scope baseline | `day1-authoritative-inputs.txt`, `day1-scope-baseline.md` | complete |
| comparison ranking | `day2-solver-family-comparison-audit.md` | complete |
| fixture taxonomy | `day3-convergence-fixture-taxonomy.md` | complete |
| helper and reporting boundary | `day4-helper-reporting-boundary.md` | complete |
| iterative design, implementation, and closeout | `day5-*`, `day6-*`, `day7-*` artifacts | complete |
| eigensolver design and implementation | `day8-*`, `day9-*`, `day10-*` artifacts | complete |
| SVD follow-through | `day10-spectral-closeout-and-svd-scope.md`, `day11-svd-comparison-follow-through.md` | complete |
| residual and claim-boundary documentation | `docs/maintainer_guide.md`, `day12-reporting-and-documentation-update.md` | complete |
| validation and evidence reconciliation | this artifact | complete |

No implemented comparison lacks a matching artifact and documented claim
boundary.

## Remaining Gaps

| gap | reason deferred | recommended owner window |
|---|---|---|
| external PETSc/SciPy/Trilinos iterative helper | helper availability, versioning, skip semantics, and oracle independence were not scoped | Sprint 104 or later external-helper sprint |
| external ARPACK/SciPy eigensolver helper | Sprint 103 used deterministic and internal evidence only | Sprint 104 or later spectral oracle sprint |
| external LAPACK/NumPy/SciPy SVD helper | SVD work deliberately stayed to one deterministic fixture | Sprint 104 or later SVD oracle sprint |
| MINRES consolidated comparison artifact | lower gap than BiCGSTAB after Day 2 audit; existing coverage is broad | future iterative evidence consolidation |
| CG convergence-profile consumer lane | existing exact, residual, SuiteSparse, preconditioner, and direct-solver checks are stronger than higher-gap lanes | future taxonomy consumer work |
| grow-m eigensolver residual documentation | Day 12 added general wording rules but no grow-m-specific new test | future spectral documentation or oracle work |
| broad performance or iteration-count claims | Sprint 103 validation is fixture-local and not a portable benchmark suite | future benchmarking/competitive calibration sprint |

## Sprint 104 Candidate Queue

1. Define external-helper policy for iterative, eigensolver, and SVD oracle
   lanes, including availability, version reporting, skip reasons, and
   subprocess failure semantics.
2. Add one helper-backed iterative comparison lane only after helper policy is
   frozen; BiCGSTAB external comparison remains the highest-value candidate.
3. Add one helper-backed spectral comparison lane for LOBPCG or thick-restart
   after ARPACK/SciPy-style oracle scope is bounded.
4. Add one helper-backed SVD comparison lane for singular values and rank only
   after LAPACK/NumPy/SciPy dependency and tolerance policy is explicit.
5. Consolidate MINRES and CG residual-profile artifacts as consumers of the
   Sprint 103 taxonomy without widening public parity claims.
6. Split public narrative from maintainer evidence maps only after each public
   statement can cite a maintained test owner and artifact.

## Day 13 Conclusion

Sprint 103 implementation, documentation, and validation are reconciled. The
implemented comparisons trace back to the Day 2 ranking, every new evidence
lane has an artifact and claim boundary, and the remaining gaps are explicit
Sprint 104-or-later candidates rather than implicit claims.
