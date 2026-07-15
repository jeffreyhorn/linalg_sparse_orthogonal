# Sprint 125 Day 13 Validation and Claim Gate

## Purpose

Validate the accepted Sprint 125 QR and minimum-norm evidence, refresh the
maintainer evidence table, and confirm that public/support wording still stays
inside the bounded rank-deficient QR, nullspace, threshold, SuiteSparse, SVD,
and minimum-norm claims accepted during Days 1-12.

## Changed Surface Inventory

| Surface | Sprint 125 role | Required Day 13 validation |
| --- | --- | --- |
| `tests/qr_external_dense_reference.py` | Standard-library helper for bounded residual-only, nullspace-projector, and threshold-rank QR evidence. | `python3 -m py_compile` plus direct helper invocations. |
| `tests/test_qr.c` | QR nullspace projector and threshold-family product owner. | Focused `test_qr` plus full C-file quality gate. |
| `tests/test_qr_solve.c` | QR residual-only and existing exact minimum-norm external fixture owner. | Focused `test_qr_solve` plus full C-file quality gate. |
| `tests/test_colamd.c` | Owner-local QR minimum-norm COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-SVD, and `west0067` submatrix lanes. | Focused `test_colamd` plus full C-file quality gate. |
| `docs/maintainer_guide.md` | Maintainer evidence table and QR non-claim register. | Claim-boundary audit, `git diff --check`, and focused whitespace scan. |
| `docs/planning/EPIC_11/SPRINT_125/*` | Sprint 125 plan, working notes, and artifacts. | `git diff --check` and focused whitespace scan. |

Because Sprint 125 changed C test files, Day 13 treats the whole sprint state
as requiring the full C-file quality gate.

## Accepted Evidence Rechecked

| Lane | Owners | Day 13 evidence |
| --- | --- | --- |
| `qr_rankdef_duplicate_5x4_residual_only` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Helper emitted `OK 1` and residual `3.7886027630095733`; `test_qr_solve` passed with the residual-only fixture. |
| `qr_rankdef_duplicate_5x4_nullspace_projector` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Helper emitted `OK 20`; `test_qr` passed with projector diff `0.000e+00` and null residual `0.000e+00`. |
| `qr_rank_threshold_diag4_family` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Helper emitted `OK 6` with ranks `3`, `2`, and `1`; `test_qr` passed with product and rank-info agreement. |
| `qr_underdetermined_minnorm_2x4` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `test_qr_solve` passed with solution diff `1.110e-16`, residual diff `0.000e+00`, and norm diff `0.000e+00`. |
| Core QR minimum-norm lanes | `tests/test_colamd.c` | `test_colamd` passed the COLAMD, fallback, rank-deficient, refinement, zero-row, and minimality lanes. |
| QR-vs-SVD minimum-norm cross-check | `tests/test_colamd.c`, `tests/test_svd.c` | `test_colamd` passed `test_minnorm_vs_pinv`; `test_svd` passed `test_pinv_underdetermined_minnorm_solution`. |
| `west0067` minimum-norm submatrix smoke | `tests/test_colamd.c` | `test_colamd` passed the 30 x 67 submatrix lane with residual `1.78e-15` and norm `4.30 <= 8.19`. |

## Focused Validation

Direct helper validation passed:

```text
python3 -m py_compile tests/qr_external_dense_reference.py
python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_residual_only
python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_nullspace_projector
python3 tests/qr_external_dense_reference.py qr_rank_threshold_diag4_family
```

Focused executable validation passed:

| Command | Result |
| --- | --- |
| `make build/test_qr && ./build/test_qr` | 68 tests, 0 failures, 0 skips, 669 assertions |
| `make build/test_qr_solve && ./build/test_qr_solve` | 18 tests, 0 failures, 0 skips, 1089 assertions |
| `make build/test_colamd && ./build/test_colamd` | 70 tests, 0 failures, 0 skips, 299 assertions |
| `make build/test_svd && ./build/test_svd` | 109 tests, 0 failures, 0 skips, 1802 assertions |

## Maintainer Evidence Gate

`docs/maintainer_guide.md` now names the accepted Sprint 125 QR evidence in
the maintained evidence table:

- `qr_rankdef_duplicate_5x4_residual_only`
- `qr_rankdef_duplicate_5x4_nullspace_projector`
- `qr_rank_threshold_diag4_family`
- owner-local minimum-norm lanes for COLAMD, fallback, rank-deficient,
  refinement, zero-row, QR-vs-SVD-pseudoinverse cross-check, and `west0067`
  submatrix behavior

The QR row now includes `tests/test_colamd.c` as an evidence owner because the
minimum-norm lanes are intentionally owner-local and not external-helper lanes.

The non-claim column preserves:

- no broad QR, LAPACK, NumPy, or SciPy parity;
- no global rank-threshold policy;
- no raw Q-basis, Q-sign/orientation, broad rank-deficient solve, nullspace,
  minimum-norm, economy-mode, sparse-mode, reorder, or performance external
  parity claim;
- no SVD-pseudoinverse-as-global-oracle claim;
- no broad SuiteSparse corpus claim.

## Public Claim Gate

`docs/solver_selection.md`, `README.md`, and public headers were audited for
Sprint 125 claim drift. Day 13 does not change those public surfaces.

The accepted evidence improves named-fixture and owner-local confidence, but
does not justify broader public wording for:

- dense-library parity;
- global QR rank-threshold behavior;
- broad rank-deficient QR solve behavior;
- raw nullspace basis or Q-basis orientation;
- broad minimum-norm optimality;
- SuiteSparse-wide corpus behavior;
- platform, backend, performance, package, ABI, or state-of-the-art behavior.

## Required Quality Gate

Because Sprint 125 changed `.c` and Python helper files, Day 13 reran and
passed:

```text
make format && make lint && make test
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_125 docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_qr_solve.c tests/test_colamd.c tests/test_svd.c
find . -path '*/__pycache__' -o -name '*.pyc'
```

The full quality chain completed successfully. The final `make test` phase
ended with `All tests passed.`

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 7 validation evidence is available. | Complete | Focused helper and executable validation passed; `make format && make lint && make test` passed. |
| Public/support wording does not exceed accepted evidence. | Complete | Maintainer evidence was refreshed; public solver-selection, README, and header wording remain unchanged. |
| Any blocker is explicit before closeout. | Complete | No Day 13 validation blocker found. |

## Day 14 Closeout Checklist

- Reconcile all Sprint 125 items as complete or explicitly deferred.
- Carry forward the deferred optional-large SuiteSparse, broader SVD oracle,
  SuiteSparse rank-deficient minimum-norm, multi-dimensional nullspace,
  scaled-threshold, and helper-movement lanes with promotion gates.
- Keep the Sprint 126 handoff bounded to residual work that has named owners
  and does not duplicate Sprint 125 accepted evidence.
