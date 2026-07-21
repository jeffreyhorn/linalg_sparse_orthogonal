# Sprint 129 Day 1 Intake And No-Reopen Boundary

## Purpose

Day 1 establishes Sprint 129 scope, duplicate fences, day-level owners,
validation expectations, and the Sprint 128 residual QR no-reopen rule.

Sprint 129 is a Q-basis, economy, sparse-mode, SuiteSparse Q/economy, and
helper ownership sprint. It is not a continuation sprint for the Sprint 128
residual QR queue unless a candidate directly supports a Sprint 129
Q/economy/helper claim and satisfies the Sprint 128 promotion gate before
implementation.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 129 project-plan section | Defines seven items and the explicit no-reopen goal text. |
| Sprint 129 plan | Defines the 14-day sequence and 166-hour budget. |
| Sprint 124 Q-basis/economy semantics | Defines raw Q sign/orientation policy, projection/subspace metrics, economy shape expectations, and Q/economy owners. |
| Sprint 124 helper ownership follow-through | Defines minimum-norm and Bidiagonal/Golub-Kahan helper naming and movement gates. |
| Sprint 125-128 QR evidence artifacts | Define rank-deficient QR, nullspace/subspace, threshold, SuiteSparse, optional-large, minimum-norm, QR-vs-SVD, and helper claim boundaries. |
| Sprint 128 retrospective residual queue | Defines the end-of-epic deferred residual queue and what Sprint 129 may not silently reopen. |
| `tests/test_qr.c` | Primary Q/economy, sparse-mode, nullspace, rank, and reconstruction owner. |
| `tests/test_qr_solve.c` | QR solve owner; not default Q/economy owner. |
| `tests/test_colamd.c` | Minimum-norm and QR-vs-SVD-pseudoinverse owner. |
| `tests/test_svd.c` | SVD pseudoinverse, Golub-Kahan, Bidiagonal QR iteration, and SVD check owner. |
| `tests/test_bidiag.c` | Bidiagonal reduction and implicit Householder reconstruction owner. |
| QR/SVD helper headers | Candidate helper surfaces that must preserve behavior-specific names and call-site tolerances. |
| `docs/maintainer_guide.md` | Evidence table owner for any accepted bounded evidence. |

## Project-Plan Owner Map

| Item | Sprint 129 owner days | Likely touched files | Required validation |
| --- | --- | --- | --- |
| 1. Q-Basis Evidence Policy Refresh | Days 1-2 | Sprint 129 artifacts, working notes, maybe `docs/maintainer_guide.md` if policy affects evidence wording | Documentation hygiene; maintainer claim scan if public/maintainer wording changes. |
| 2. Raw Q-Column Evidence | Days 2-3 | `tests/test_qr.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md`, Sprint 129 artifacts | Focused helper invocation if Python changes; `make build/test_qr && ./build/test_qr`; full quality gate if `.c` or `.h` changes. |
| 3. Rank-Deficient Q/Nullspace Evidence | Days 4-5 | `tests/test_qr.c`, `tests/qr_external_dense_reference.py`, `tests/test_qr_helpers.h`, `docs/maintainer_guide.md`, Sprint 129 artifacts | Focused QR/helper checks; full quality gate if `.c` or `.h` changes. |
| 4. Wide Economy and Sparse-Mode Evidence | Days 6-7 | `tests/test_qr.c`, QR helpers, Sprint 129 artifacts, maybe maintainer evidence | Focused QR/economy checks; full quality gate if `.c` or `.h` changes. |
| 5. SuiteSparse Q/Economy Evidence | Days 8-9 | `tests/test_qr.c`, SuiteSparse data references, optional-data docs/artifacts, maybe maintainer evidence | Focused SuiteSparse QR/economy diagnostics, skip-path proof, runtime/support-tier note, full quality gate if `.c` or `.h` changes. |
| 6. Minimum-Norm Helper Movement | Days 10-11 | `tests/test_qr_solve.c`, `tests/test_colamd.c`, `tests/test_svd.c`, `tests/test_qr_helpers.h`, `tests/test_svd_helpers.h`, maybe helper headers | Focused QR solve, COLAMD, and SVD checks; full quality gate if `.c` or `.h` changes. |
| 7. Bidiagonal/Golub-Kahan Helper Extraction | Days 12-13 | `tests/test_bidiag.c`, `tests/test_svd.c`, `tests/test_svd_helpers.h`, possible dedicated helper header, Make/CMake source lists if ownership moves | Focused Bidiagonal and SVD checks; source-list/CMake checks if membership changes; full quality gate if `.c` or `.h` changes. |

## Duplicate Fence

The following completed work is not reopened by default:

| Completed scope | Baseline to preserve |
| --- | --- |
| Sprint 124 Q/economy semantics | Raw Q equality remains rejected by default; projection, orthogonality, reconstruction, projector, or principal-angle metrics are preferred. |
| Sprint 124 economy evidence | `qr_economy_projector_5x3` remains the bounded economy projector baseline. |
| Sprint 125-128 nullspace/subspace evidence | Existing projector/subspace fixtures remain baselines; Sprint 129 may use their metric policies but should not duplicate their residual queue. |
| Sprint 125-128 threshold evidence | Existing threshold families remain baselines; Sprint 129 should not reopen threshold debt unless a Q/economy claim requires it. |
| Sprint 125-128 minimum-norm evidence | Existing exact-value, owner-local, QR-vs-SVD, and SuiteSparse smoke lanes remain baselines; generic helper movement must remain behavior-specific. |
| Sprint 128 residual queue | Compatible zero-residual, wide residual-only, near-threshold, SuiteSparse rank-deficient corpus, optional-large, extra exact, and extra QR-vs-SVD debt remain end-of-epic queue items. |

## Sprint 128 Residual No-Reopen Rule

Sprint 129 may touch a Sprint 128 residual queue item only when all of the
following are true before implementation:

1. The candidate directly supports a Sprint 129 Q-basis, economy, sparse-mode,
   SuiteSparse Q/economy, or helper ownership claim.
2. The candidate has a non-duplicate behavior-specific fixture key or helper
   name.
3. The metric, expected shape, expected rank/nullity when relevant, tolerance,
   diagnostics, support tier, skip behavior, runtime expectation, and failure
   interpretation are pinned.
4. The artifact states why the item belongs in Sprint 129 instead of the
   end-of-epic deferred queue.
5. The validation plan includes focused owner tests and the full quality gate
   when `.c` or `.h` files change.

If any condition is missing, the item stays in the end-of-epic deferred QR
residual queue.

## Validation Boundary

| Change class | Day 1 rule |
| --- | --- |
| Documentation-only Sprint 129 artifacts | `git diff --check` and trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_129`. |
| QR Q/economy evidence | Focused QR/helper checks and full quality gate if `.c` or `.h` changes. |
| Python helper protocol | `python3 -m py_compile`, direct helper invocation, affected executable, and diff hygiene. |
| Minimum-norm helper movement | Focused QR solve, COLAMD, SVD checks, and full quality gate if `.c` or `.h` changes. |
| Bidiagonal/Golub-Kahan helper movement | Focused Bidiagonal and SVD checks, source-list/build checks when ownership moves, and full quality gate if `.c` or `.h` changes. |
| Maintainer/public wording | Evidence-to-claim traceability, non-claim scan, path/link hygiene, and docs hygiene. |

## Non-Claims Preserved

Day 1 does not claim:

- raw Q-basis equality, sign, orientation, ordering, or unique-basis parity;
- broad QR factorization, QR solve, Q-basis, economy, sparse-mode, nullspace,
  minimum-norm, SuiteSparse, optional-data, platform, or performance parity;
- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity;
- SVD-pseudoinverse as a global QR oracle;
- generic QR/SVD/minimum-norm, Bidiagonal, or Golub-Kahan helper API;
- package, ABI, public API, install-header, CMake, Makefile, CI, CTest,
  scalability, memory, or state-of-the-art parity.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 129 project-plan item has a day-level owner. | Complete | Owner map ties Items 1-7 to Days 1-14 and likely files. |
| Sprint 128 residual QR debt is not silently reopened. | Complete | No-reopen rule keeps residual debt in the end-of-epic queue unless all promotion criteria are satisfied. |
| Q-basis, economy, and helper ownership dependencies are explicit before new evidence is accepted. | Complete | Input inventory, duplicate fence, and validation boundary identify dependencies and owners. |
