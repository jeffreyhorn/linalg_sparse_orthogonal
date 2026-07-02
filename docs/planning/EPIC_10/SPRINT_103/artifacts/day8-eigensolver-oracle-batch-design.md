# Sprint 103 Day 8 Eigensolver Oracle Batch Design

## Purpose

Day 8 selects the focused spectral comparison batch for Day 9. The design uses
the Day 7 rerank, Day 3 fixture taxonomy, and current eigensolver test surface
to freeze bounded LOBPCG, thick-restart, and grow-m comparison cases before
any implementation changes are made.

## Source Surface Reviewed

| file | current spectral evidence | Day 8 implication |
|---|---|---|
| `tests/test_eigs_lobpcg.c` | diagonal, Laplacian, `nos4`, preconditioned shifted-Laplacian and `bcsstk04`, soft-locking, nearest-sigma, cross-backend parity, AUTO dispatch | LOBPCG already has breadth; Sprint 103 should add claim-owned residual and orthogonality evidence rather than duplicate broad parity lanes |
| `tests/test_eigs_thick_restart.c` | arrowhead helpers, restart state, small diagonal eigenvectors, SuiteSparse parity, bounded-memory `bcsstk14`, nearest-sigma KKT parity, monotone progress proxy | thick-restart has strong grow-m parity; Sprint 103 should add or tighten an independent fixture reference with explicit residual and orthogonality criteria |
| `tests/test_eigs.c` | grow-m diagonal, tridiagonal dense-reference, shift-invert, handle reuse, public dispatch coverage | grow-m is useful as context but should not be the primary Day 9 implementation target unless shared helper code becomes necessary |

## Selected Day 9 Batch

| item | target file | fixture key | taxonomy class | profile | reference behavior | expected result |
|---|---|---|---|---|---|---|
| LOBPCG closed-form residual and orthogonality claim | `tests/test_eigs_lobpcg.c` | `lobpcg_laplacian30_smallest4_claim` | `spd-tridiag-laplacian` | `orthogonality-sensitive` / `fast-exact` | closed-form Laplacian eigenvalues plus per-pair Ritz residual and vector orthogonality | converges; eigenvalues match closed form within `1e-7`; per-pair residual `< 1e-7`; max `|V^T V - I| < 1e-8` |
| LOBPCG preconditioned corpus comparison claim | `tests/test_eigs_lobpcg.c` | `lobpcg_bcsstk04_ic0_ldlt_claim` | `spd-mm-clustered` | `fast-preconditioned` / `orthogonality-sensitive` | IC(0) and LDLT preconditioned LOBPCG on `bcsstk04`; compare residual, convergence status, eigenvalue agreement, and iteration descriptions | both converge; LDLT iterations less than IC(0); eigenvalues agree within `1e-6`; if vectors are requested, residuals `< 1e-6` |
| Thick-restart independent diagonal/vector claim | `tests/test_eigs_thick_restart.c` | `thick_restart_diag12_largest4_claim` | `spd-diag-separated` | `orthogonality-sensitive` / `restart-sensitive` | exact diagonal eigenvalues, per-pair residuals, vector orthogonality, and bounded peak basis | converges; top four eigenvalues match `{12, 11, 10, 9}` within `1e-10`; residuals `< 1e-10`; max `|V^T V - I| < 1e-10`; peak basis remains bounded |

The first item is the highest-value Day 9 target because Day 7 ranked LOBPCG
first after the BiCGSTAB closeout. The thick-restart item is second because it
reduces reliance on grow-m parity by using an exact diagonal reference. The
preconditioned corpus item should be implemented only if it remains stable with
`compute_vectors = 1`; otherwise Day 9 should record the instability and keep
the existing iteration-only `bcsstk04` comparison as the current coverage.

## Criteria Matrix

| criterion | LOBPCG closed-form | LOBPCG preconditioned corpus | thick-restart diagonal |
|---|---|---|---|
| eigenvalue error | absolute `< 1e-7` against closed-form Laplacian | IC(0) and LDLT eigenvalues agree within `1e-6` relative/absolute scaled by `max(1, |lambda|)` | absolute `< 1e-10` against exact diagonal values |
| Ritz residual | per-pair relative `< 1e-7` | per-pair relative `< 1e-6` if eigenvectors are requested; otherwise report `residual_norm` only | per-pair relative `< 1e-10` |
| orthogonality | max `|V^T V - I| < 1e-8` | max `|V^T V - I| < 1e-7` if vectors are requested | max `|V^T V - I| < 1e-10` |
| convergence status | `SPARSE_OK`, `n_converged == k` | `SPARSE_OK`, `n_converged == k` for both preconditioners | `SPARSE_OK`, `n_converged == k` |
| iteration interpretation | descriptive only | LDLT faster than IC(0) on this named fixture; no portable performance claim | descriptive only |
| skip behavior | no skip expected | if fixture load or factorization fails, fail existing test convention rather than skip | no skip expected |

## Fixture Ownership Notes

| fixture | ownership | implementation notes |
|---|---|---|
| `lobpcg_laplacian30_smallest4_claim` | generated locally in `tests/test_eigs_lobpcg.c` | reuse `build_laplacian_tridiag_lobpcg`; request eigenvectors; add local orthogonality helper only if existing helper is too narrow |
| `lobpcg_bcsstk04_ic0_ldlt_claim` | SuiteSparse fixture `tests/data/suitesparse/bcsstk04.mtx` | reuse existing IC(0), LDLT, and adapter code; prefer adding vector residual checks around the existing comparison rather than introducing a new fixture |
| `thick_restart_diag12_largest4_claim` | generated locally in `tests/test_eigs_thick_restart.c` | diagonal exact spectrum keeps this independent of grow-m parity and external packages |

No public API, build-system, fixture file, or external helper changes are
planned. Any helper added on Day 9 should stay file-local unless duplication
becomes materially worse than local ownership.

## SVD Overlap Opportunities

Day 8 identifies three reusable concepts for Day 10 SVD work:

- basis orthogonality checks can share threshold language with singular-vector
  orthogonality;
- residual checks should stay separate from eigenvalue or singular-value
  agreement;
- generated diagonal or Laplacian fixtures can support both spectral and SVD
  claims, but Day 10 must still name singular-value, rank, and reconstruction
  thresholds explicitly.

Day 9 should not add SVD tests. It should leave the orthogonality and residual
language clean enough for Day 10 to reuse in SVD design.

## Focused Validation Commands for Day 9

If Day 9 modifies `.c` files, required validation is:

```sh
make build/test_eigs_lobpcg build/test_eigs_thick_restart
./build/test_eigs_lobpcg
./build/test_eigs_thick_restart
make format && make lint && make test
git diff --check
rg -n "[ \t]+$" tests/test_eigs_lobpcg.c tests/test_eigs_thick_restart.c docs/planning/EPIC_10/SPRINT_103
```

If Day 9 implements only the highest-value LOBPCG item, it may run
`make build/test_eigs_lobpcg && ./build/test_eigs_lobpcg` before the full gate,
but the full `make format && make lint && make test` chain is still required
because C tests would be modified.

## Deferred Spectral Follow-Ups

| follow-up | disposition |
|---|---|
| external ARPACK or SciPy eigensolver helper | deferred; no helper availability, versioning, or skip contract has been scoped |
| broad thick-restart ARPACK parity wording | rejected for Sprint 103; use fixture-specific exact references and internal parity only |
| grow-m eigensolver implementation expansion | deferred unless Day 9 needs a shared residual or orthogonality helper |
| larger SuiteSparse LOBPCG corpus | deferred; Day 9 should keep runtime and claims bounded |
| SVD singular-value/rank/reconstruction implementation | deferred to Day 10 and later |

## Non-Claims Preserved

The Day 8 design does not claim:

- ARPACK, SciPy, LAPACK, NumPy, PETSc, Trilinos, or package-wide parity;
- external oracle evidence for LOBPCG, thick-restart, grow-m, or SVD paths;
- portable performance superiority from local iteration counts;
- that grow-m parity is an independent oracle for thick-restart;
- that one diagonal, Laplacian, or SuiteSparse fixture proves broad
  state-of-the-art sparse eigensolver behavior;
- that SVD rank or reconstruction evidence has already been implemented.

## Day 8 Conclusion

Day 9 should start with the LOBPCG closed-form residual and orthogonality claim,
then add the thick-restart exact diagonal claim. The preconditioned `bcsstk04`
LOBPCG claim is valuable but should not destabilize the sprint; if vector
residual checks are brittle on that corpus fixture, Day 9 should document the
result and keep the existing iteration/eigenvalue comparison as bounded
preconditioner evidence.
