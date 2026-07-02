# Sprint 103 Day 10 Spectral Closeout and SVD Scope

## Purpose

Day 10 closes out the Day 9 spectral comparison implementation and freezes the
SVD follow-through scope for Day 11. The goal is to reuse the residual,
orthogonality, and fixture-claim discipline from LOBPCG and thick-restart work
without broadening Sprint 103 into package-wide SVD parity.

## Spectral Evidence Closeout

Day 9 added or strengthened three bounded spectral lanes.

| lane | fixture key | closeout status |
|---|---|---|
| LOBPCG closed-form residual and orthogonality | `lobpcg_laplacian30_smallest4_claim` | validated; closed-form eigenvalues, per-pair Ritz residuals, and basis orthogonality are asserted |
| LOBPCG preconditioned corpus residual and orthogonality | `lobpcg_bcsstk04_ic0_ldlt_claim` | validated; IC(0) and LDLT both converge, LDLT remains faster on the named fixture, and both vector bases satisfy residual and orthogonality gates |
| thick-restart exact diagonal residual and orthogonality | `thick_restart_diag12_largest4_claim` | validated; exact eigenvalue, residual, orthogonality, and bounded peak-basis checks do not depend on grow-m parity |

The spectral work is complete for the Sprint 103 implementation batch. It
earns fixture-specific LOBPCG and thick-restart evidence, not external ARPACK
or SciPy parity.

## Day 10 Validation Results

| command | result |
|---|---|
| `make build/test_eigs_lobpcg build/test_eigs_thick_restart build/test_svd build/test_sprint29_integration` | passed; all targets already up to date |
| `./build/test_eigs_lobpcg` | passed; 27 tests, 0 failures, 0 skips, 247 assertions |
| `./build/test_eigs_thick_restart` | passed; 21 tests, 0 failures, 0 skips, 285 assertions |
| `./build/test_svd` | passed; 97 tests, 0 failures, 0 skips, 1073 assertions |
| `./build/test_sprint29_integration` | passed; 3 tests, 0 failures, 0 skips, 25 assertions |

The focused validation confirms the Day 9 spectral evidence still passes and
that the adjacent SVD surface is healthy before Day 11 implementation starts.

## Residual, Orthogonality, and Convergence Review

| evidence dimension | Day 9 spectral state | Day 11 SVD implication |
|---|---|---|
| value agreement | LOBPCG and thick-restart compare against exact closed-form values on generated fixtures | SVD should compare singular values against exact diagonal or analytically constructed values |
| residual quality | spectral tests separate residual gates from eigenvalue agreement | SVD should separate reconstruction residual from singular-value agreement |
| basis quality | spectral tests assert `V^T V` quality explicitly | SVD should assert both `U^T U` and `Vt Vt^T` quality explicitly |
| convergence status | spectral tests assert `SPARSE_OK` and `n_converged == k` separately | SVD should assert API success and rank/shape fields separately from numerical quality |
| iteration output | spectral tests print iteration counts as descriptive data only | SVD should avoid performance claims from local runtime or iteration output |

## SVD Coverage Reviewed

| current SVD area | existing evidence | Day 10 decision |
|---|---|---|
| full SVD singular values | exact diagonal, trace invariant, descending order, SuiteSparse smoke | reuse exact diagonal value checks in a claim-owned test |
| SVD U/V reconstruction | square, tall, wide, full-mode, and low-rank reconstruction checks | reuse reconstruction residual language, but keep Day 11 to one fixture |
| SVD orthogonality | Golub-Kahan U/V checks, wide U/V checks, full-mode U/V checks, partial-vector orthogonality | reuse explicit `U^T U` and `Vt Vt^T` threshold language |
| rank-sensitive behavior | rank-1, rank-2, rank-deficient, near-singular rank threshold checks | add one combined rank-threshold assertion only if it fits the selected fixture |
| partial SVD | partial singular-value comparisons, vector residuals, reconstruction error, SuiteSparse fixtures | defer; already broad and not necessary for Day 11's limited scope |
| SuiteSparse SVD | `nos4`, `west0067`, low-rank corpus safety | defer broader corpus expansion to avoid overclaiming |

## Selected Day 11 SVD Scope

| item | target file | fixture key | taxonomy class | profile | expected result |
|---|---|---|---|---|---|
| SVD diagonal claim with rank threshold and full UV checks | `tests/test_svd.c` | `svd_diag6_rank_threshold_claim` | `spd-diag-separated` / `rank-sensitive` | `orthogonality-sensitive` / `rank-sensitive` | singular values match `{9, 5, 2, 1e-9, 0, 0}`; full-mode reconstruction residual `< 1e-10`; max `U^T U` and `Vt Vt^T` errors `< 1e-10`; rank is `4` at tolerance `1e-10` and `3` at tolerance `1e-8` |

This is deliberately one test in one existing binary. It consolidates the
Day 9 evidence style for SVD without adding external helpers, new fixtures,
or broad SuiteSparse claims.

## SVD Criteria Matrix

| criterion | selected Day 11 threshold | notes |
|---|---|---|
| singular-value agreement | absolute `< 1e-10` for exact diagonal singular values, except the `1e-9` value may use `< 1e-12` absolute | fixture is diagonal, so exact values are available |
| reconstruction residual | relative Frobenius `< 1e-10` | separates reconstruction quality from singular-value agreement |
| U orthogonality | max `|U^T U - I| < 1e-10` | full-mode U should be square on this fixture |
| Vt orthogonality | max `|Vt Vt^T - I| < 1e-10` | full-mode Vt should be square on this fixture |
| rank threshold | rank `4` at `tol=1e-10`; rank `3` at `tol=1e-8` | makes rank sensitivity explicit without relying on default tolerance |
| API status | `SPARSE_OK`; expected dimensions and non-null buffers | checked separately from numerical thresholds |

## Focused Validation Commands for Day 11

If Day 11 modifies `tests/test_svd.c`, required validation is:

```sh
make build/test_svd
./build/test_svd
make format && make lint && make test
git diff --check
rg -n "[ \t]+$" tests/test_svd.c tests/test_eigs_lobpcg.c tests/test_eigs_thick_restart.c tests/test_bicgstab.c docs/planning/EPIC_10/SPRINT_103
```

If implementation unexpectedly touches other `.c` or `.h` files, add their
focused test binaries before the full gate.

## Deferred Follow-Ups

| follow-up | disposition |
|---|---|
| external LAPACK, NumPy, or SciPy SVD helper | deferred; helper availability and versioning are not scoped |
| SuiteSparse SVD corpus expansion | deferred; existing `nos4`, `west0067`, and low-rank corpus safety already provide smoke/corpus evidence |
| partial SVD implementation changes | deferred; current partial SVD vector, residual, reconstruction, and corpus checks are broad enough for Sprint 103 |
| low-rank sparse output expansion | deferred; not needed for the selected residual/orthogonality claim |
| public SVD documentation rewrite | Day 12 documentation work should describe claim boundaries rather than add new code |

## Non-Claims Preserved

Day 10 does not claim:

- LAPACK, NumPy, SciPy, ARPACK, PETSc, Trilinos, or package-wide parity;
- that Day 9 spectral comparisons prove broad sparse eigensolver quality;
- that Day 11 SVD work will prove broad SVD correctness beyond one named
  fixture;
- runtime or iteration-count performance superiority;
- external helper-backed SVD evidence.

## Day 10 Conclusion

The spectral implementation is validated and closed. Day 11 should implement a
single SVD diagonal/rank/full-UV claim in `tests/test_svd.c`, then run focused
SVD validation and the full quality gate because a C test file will be touched.
