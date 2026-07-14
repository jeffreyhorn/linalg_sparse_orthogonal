# Sprint 124 Day 13 Validation and Claim Gate

## Purpose

Validate the accepted Sprint 124 implementation work, confirm maintainer
evidence wording, and decide whether any public solver-selection wording can
change without overstating the bounded oracle evidence.

## Accepted Implementation Lanes Rechecked

| Lane | Owners | Day 13 evidence |
| --- | --- | --- |
| QR rank-only rank-deficient fixture `qr_rankdef_duplicate_5x4_rank_only` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Helper emitted `OK 1` and rank `3`; `test_qr_solve` passed with the fixture registered. |
| QR exact minimum-norm fixture `qr_underdetermined_minnorm_2x4` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Helper emitted `OK 6`, solution values `0.5`, `0.5`, `0.5`, `0.5`, residual `0`, and norm `1`; `test_qr_solve` passed with the fixture registered. |
| QR economy projector fixture `qr_economy_projector_5x3` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Helper emitted `OK 29` with shape values `5`, `3`, `3`, `3`; `test_qr` passed with projector and orthogonality checks. |
| Partial-SVD vector-residual fixture `partial_svd_vector_residual_diag6_k2` | `tests/test_svd_partial_helpers.h`, `tests/test_svd.c`, `tests/svd_external_dense_reference.py` | Helper emitted `OK 2` with singular values `9` and `6`; `test_svd` passed with singular-value, residual, and orthogonality checks. |

## Focused Validation

Direct helper validation passed:

1. `python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_rank_only`
2. `python3 tests/qr_external_dense_reference.py qr_underdetermined_minnorm_2x4`
3. `python3 tests/qr_external_dense_reference.py qr_economy_projector_5x3`
4. `python3 tests/svd_external_dense_reference.py partial_svd_diag6_k2`

Focused executable validation passed:

| Command | Result |
| --- | --- |
| `make build/test_qr_solve && ./build/test_qr_solve` | 17 tests, 0 failures, 0 skips, 1069 assertions |
| `make build/test_qr && ./build/test_qr` | 66 tests, 0 failures, 0 skips, 628 assertions |
| `make build/test_svd && ./build/test_svd` | 109 tests, 0 failures, 0 skips, 1803 assertions |

## Required Quality Gate

Because Sprint 124 changed `.c` and `.h` test files, Day 13 reran the full
required quality chain:

1. `make format`
2. `make lint`
3. `make test`

The full chain completed successfully. The final `make test` phase ended with
`All tests passed.`

Additional hygiene checks passed after the full chain:

1. `git diff --check`
2. Focused trailing-whitespace scan over Sprint 124 documentation and touched
   maintainer/test/helper files

## Maintainer Evidence Gate

`docs/maintainer_guide.md` already names the bounded Sprint 124 lanes in the
maintained evidence table:

- QR now lists `qr_rankdef_duplicate_5x4_rank_only`,
  `qr_underdetermined_minnorm_2x4`, and `qr_economy_projector_5x3`.
- SVD now lists `partial_svd_vector_residual_diag6_k2` separately from the
  singular-value-only partial-SVD fixtures.
- The table keeps the public guidance level family-local and ties confidence
  to named test owners.
- The QR and SVD non-claim columns still reject broad external package parity,
  raw Q-basis or sign/orientation claims, broad rank-deficient solve claims,
  broad minimum-norm claims, broad vector/subspace claims, low-rank
  optimality, convergence-budget, platform, performance, and state-of-the-art
  claims.

Day 13 makes no additional maintainer-guide wording change because the table
already reflects the accepted Sprint 124 implementation lanes and still
preserves the required non-claims.

## Solver-Selection Claim Gate

`docs/solver_selection.md` remains unchanged.

The accepted Sprint 124 lanes improve named-fixture evidence for QR and
partial-SVD, but they do not change the public family-selection advice. The
solver-selection guide already says:

- use QR for rectangular, least-squares, minimum-norm, and rank-sensitive
  workflows;
- use SVD APIs for singular values, rank, condition, pseudoinverse, and
  low-rank approximations;
- treat benchmark output and broad comparative behavior as local or
  configuration-sensitive.

No new public wording is justified for broad QR parity, broad SVD parity,
external dense-library parity, raw Q-basis equality, singular-vector parity,
subspace parity, low-rank optimality, convergence guarantees, or performance.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 7 validation obligations are satisfied or clearly blocked. | Complete | Focused helper/executable checks passed; `make format && make lint && make test` passed; hygiene checks passed. |
| Public wording changes are evidence-backed. | Complete | No public solver-selection wording change was made because the evidence remains fixture-scoped. |
| No unsupported QR, partial-SVD, helper, or external-oracle claim is added. | Complete | Maintainer evidence remains named-fixture scoped, helper movement remains deferred, and public docs are unchanged. |

## Residual Risk for Day 14

Day 14 should close Sprint 124 from this validation baseline and consolidate
the deferred owner queue:

- residual-only rank-deficient QR, nullspace, near-threshold, and SuiteSparse
  rank-deficient QR evidence;
- COLAMD/reordered, fallback, refinement, QR-vs-SVD-pseudoinverse, and
  SuiteSparse minimum-norm evidence;
- raw Q-basis, rank-deficient subspace/nullspace, wide economy, sparse-mode,
  and SuiteSparse Q/economy evidence;
- repeated-spectrum, clustered-spectrum, rank-deficient subspace,
  rectangular-vector, SuiteSparse corpus, low-rank optimality, and
  convergence-budget partial-SVD evidence;
- minimum-norm helper movement and Bidiagonal/Golub-Kahan helper extraction.
