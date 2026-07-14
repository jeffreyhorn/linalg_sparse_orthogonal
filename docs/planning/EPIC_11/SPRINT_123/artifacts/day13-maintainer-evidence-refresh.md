# Day 13 Maintainer Evidence Refresh

## Purpose

Refresh maintainer evidence tables and support-claim boundaries so the bounded
Sprint 121-123 SVD, QR, and partial-SVD oracle lanes are visible to maintainers
without widening public product claims.

## Maintainer Guide Update

Updated `docs/maintainer_guide.md` in the Sprint 102 and Sprint 103 evidence
snapshots.

- Replaced stale QR wording that said there was no external dense
  least-squares oracle lane.
- Replaced stale SVD wording that said there was no external dense SVD helper
  lane.
- Added `tests/test_qr_solve.c` and `tests/qr_external_dense_reference.py` as
  QR evidence owners for the bounded external least-squares fixtures.
- Added `tests/test_svd_partial_helpers.h` and
  `tests/svd_external_dense_reference.py` as SVD and partial-SVD evidence
  owners for the bounded external singular-value fixtures.
- Updated the Sprint 103 interpretation so external dense-reference evidence
  is no longer described as limited to Sprint 102 direct-solver lanes.

## Evidence Entries

| Lane | Maintained owner | Trust boundary | Validation command | Non-claims |
| --- | --- | --- | --- | --- |
| `svd_rect_fullrank_6x4` | `tests/test_svd.c`, `tests/svd_external_dense_reference.py` | bounded full-SVD singular-value comparison for one rectangular full-rank fixture | `make build/test_svd && ./build/test_svd` | no LAPACK, NumPy, SciPy, vector, subspace, or broad SVD parity claim |
| `svd_rankdef_duplicate_5x4` | `tests/test_svd.c`, `tests/svd_external_dense_reference.py` | bounded full-SVD singular-value comparison for one duplicate-column rank-deficient fixture | `make build/test_svd && ./build/test_svd` | no broad rank-threshold, pseudoinverse, minimum-norm, or package parity claim |
| `svd_wide_fullrank_4x6` | `tests/test_svd.c`, `tests/svd_external_dense_reference.py` | bounded wide full-row-rank singular-value comparison with `min(m,n)` reference output | `make build/test_svd && ./build/test_svd` | no vector/subspace, low-rank, performance, platform, or broad external parity claim |
| `partial_svd_diag6_k2` | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, `tests/svd_external_dense_reference.py` | bounded top-two singular-value comparison for one diagonal partial-SVD fixture | `make build/test_svd && ./build/test_svd` | no vector, subspace, convergence-budget, repeated-spectrum, or low-rank optimality claim |
| `partial_svd_tall_diag_8x5_k3` | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, `tests/svd_external_dense_reference.py` | bounded top-three singular-value comparison for one tall diagonal partial-SVD fixture | `make build/test_svd && ./build/test_svd` | no vector/subspace, convergence, rank-deficient, low-rank, package, or platform parity claim |
| `qr_overdetermined_incompatible_4x2` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | bounded incompatible least-squares fixture using the external dense normal-equation helper | `make build/test_qr_solve && ./build/test_qr_solve` | no rank-deficient, minimum-norm, Q-basis, economy, sparse-mode, or broad QR parity claim |
| `qr_overdetermined_compatible_5x3` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | bounded compatible full-column-rank least-squares fixture using the external dense normal-equation helper | `make build/test_qr_solve && ./build/test_qr_solve` | no rank-deficient, minimum-norm, Q-basis, economy, sparse-mode, reorder, or performance parity claim |

## Deferred Evidence Entries

| Deferred lane | Future owner | Deferral reason |
| --- | --- | --- |
| Rank-deficient QR external evidence | future QR solve oracle sprint | needs explicit rank-threshold, nullspace, pseudoinverse, and minimum-norm policy before helper-backed comparison |
| QR minimum-norm external evidence | future QR solve / minimum-norm owner | current behavior spans QR solve, COLAMD, SVD pseudoinverse, fallback, refinement, and SuiteSparse paths |
| QR Q/economy external evidence | future QR basis/economy owner | needs sign, orientation, projection, subspace, and economy-shape semantics |
| Partial-SVD vector/subspace evidence | future partial-SVD semantic owner | top-k value evidence does not prove vector orientation or subspace quality |
| Partial-SVD repeated, clustered, rank-deficient, convergence, and low-rank evidence | future partial-SVD oracle owner | each class needs separate tolerance, ambiguity, convergence, and optimality rules |
| Minimum-norm helper migration | future behavior-specific helper owner | generic helper movement would hide scenario-specific assertions and tolerances |
| Bidiagonal/Golub-Kahan helper extraction | future Bidiagonal/GK maintainability owner | specialized transpose, reconstruction, explicit `U`/`V`, and iteration semantics remain local |

## Support-Doc Claim Scan

| Surface | Result |
| --- | --- |
| `README.md` | no fixture-level QR, SVD, or partial-SVD external parity claim found; no update needed |
| `docs/solver_selection.md` | solver guidance remains higher-level and does not claim broad external QR/SVD parity; no update needed |
| `docs/algorithm.md` | algorithm and performance caveats remain unrelated to Sprint 123 helper evidence; no update needed |
| `docs/maintainer_guide.md` | updated as the maintainer proof surface for bounded external fixture ownership and non-claims |

## Sprint 123 Residual Queue Draft

1. Decide on Day 14 whether solver-selection wording needs any user-facing
   update; default is no expansion unless current evidence supports it.
2. Carry rank-deficient QR external evidence as future residual debt.
3. Carry QR minimum-norm and Q/economy external evidence as future residual
   debt.
4. Carry partial-SVD vector, subspace, repeated-spectrum, clustered-spectrum,
   convergence-budget, rank-deficient, and low-rank optimality evidence as
   future residual debt.
5. Carry minimum-norm helper migration as future residual debt.
6. Carry Bidiagonal/Golub-Kahan helper extraction as future residual debt.

## Validation

- `git diff --check`
- `rg -n "[ \t]$" docs/maintainer_guide.md docs/planning/EPIC_11/SPRINT_123`
- `rg -n "external dense-reference evidence remains limited to the Sprint 102|no Sprint 102 external dense|no external dense SVD helper lane" docs/maintainer_guide.md`

## Completion Criteria

- Item 6 is complete.
- Evidence entries match implemented or deferred Sprint 123 outcomes.
- No public or support claim exceeds the current bounded proof.
