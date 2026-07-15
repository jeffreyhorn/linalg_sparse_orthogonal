# Sprint 125 Day 14 Closeout and Handoff

## Purpose

Close Sprint 125 from the Day 13 validation baseline, consolidate accepted
rank-deficient QR and minimum-norm evidence, preserve explicit deferrals and
non-claims, and hand a stable bounded-evidence package to Sprint 126 and later
QR follow-through work.

## Sprint 125 Closeout Summary

Sprint 125 completed all seven project-plan items through bounded
implementation, explicit policy, or explicit deferral with named promotion
gates.

| Project-plan item | Final state | Evidence |
| --- | --- | --- |
| 1. Deferred QR Dedupe Map | Complete | Day 1 mapped Sprint 124 residual debt to existing Sprint 121-124 evidence and duplicate fences. |
| 2. Rank-Deficient Residual Evidence | Complete with one bounded implementation and explicit deferrals | Day 2 trust gate plus Day 3 `qr_rankdef_duplicate_5x4_residual_only` fixture. |
| 3. Nullspace and Subspace Policy | Complete with one bounded implementation and explicit deferrals | Day 4 policy plus Day 5 `qr_rankdef_duplicate_5x4_nullspace_projector` fixture. |
| 4. Near-Rank-Deficient Threshold Evidence | Complete with one bounded implementation and explicit deferrals | Day 6 threshold-family policy plus Day 7 `qr_rank_threshold_diag4_family` fixture. |
| 5. SuiteSparse Rank-Deficient QR Evidence | Complete by explicit deferral package | Day 8 corpus policy plus Day 9 decision to keep current checked-in SuiteSparse QR matrices as full-rank controls. |
| 6. Minimum-Norm Behavior Evidence | Complete with bounded owner-local implementations and explicit deferrals | Day 10 owner map, Day 11 core minimum-norm evidence, and Day 12 QR-vs-SVD plus `west0067` submatrix decision. |
| 7. Validation and Claim Gate | Complete | Day 13 focused validation, full `make format && make lint && make test`, maintainer evidence refresh, and public claim audit. |

## Accepted Implementation Package

| Accepted lane | Fixture or behavior | Owner surfaces | Claim boundary |
| --- | --- | --- | --- |
| Rank-deficient QR residual-only evidence | `qr_rankdef_duplicate_5x4_residual_only` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | One 5x4 duplicate-column residual-only check against a standard-library column-space projection reference; no rank, nullspace, minimum-norm, solution-vector, or broad QR claim. |
| Rank-deficient QR nullspace projector evidence | `qr_rankdef_duplicate_5x4_nullspace_projector` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | One pinned rank 3/nullity 1 projector check; no raw nullspace vector, Q-basis sign/orientation, broad subspace, or minimum-norm claim. |
| Near-rank-deficient threshold evidence | `qr_rank_threshold_diag4_family` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | One diagonal ladder with ranks 3, 2, and 1 at named relative tolerances; no global rank-threshold, residual, nullspace, corpus, or dense-library parity claim. |
| Exact QR minimum-norm external evidence retained | `qr_underdetermined_minnorm_2x4` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | Sprint 124 exact 2x4 solution/residual/norm lane remains the external QR minimum-norm anchor and was revalidated during Sprint 125. |
| Core owner-local QR minimum-norm evidence | COLAMD, fallback, rank-deficient, refinement, and zero-row lanes | `tests/test_colamd.c`, `docs/maintainer_guide.md` | Behavior-specific assertions for named fixtures only; no COLAMD superiority, fallback generality, broad rank-deficient minimum-norm, refinement convergence-rate, or global optimality claim. |
| QR-vs-SVD minimum-norm cross-check | `test_minnorm_vs_pinv` plus SVD pseudoinverse owner validation | `tests/test_colamd.c`, `tests/test_svd.c`, `docs/maintainer_guide.md` | One bounded cross-check for a 2x4 fixture; SVD pseudoinverse is not a global QR oracle. |
| SuiteSparse minimum-norm submatrix smoke | first 30 rows of checked-in `west0067.mtx` | `tests/test_colamd.c`, `docs/maintainer_guide.md` | One default checked-in 30 x 67 corpus submatrix smoke; no broad SuiteSparse, optional-corpus, platform, or performance claim. |

## Validation Baseline

Day 13 is the authoritative Sprint 125 validation baseline.

Focused helper validation passed:

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

Full required quality validation passed:

```text
make format && make lint && make test
```

The final test phase ended with `All tests passed.`

Day 14 changes planning documentation only. No source, header, helper script,
build metadata, package metadata, public API, or public solver-selection
wording changed on Day 14.

## Consolidated Residual and Future-Owner Queue

| Residual | Future owner | Promotion gate |
| --- | --- | --- |
| Compatible zero-residual rank-deficient QR residual fixture | Future QR solve residual owner | Prove zero-residual evidence adds trust beyond deterministic compatible solve behavior and cannot be misread as minimum-norm evidence. |
| Dependent-row rank-deficient residual fixture | Future QR residual owner | Show an independent structural family adds trust beyond duplicate-column evidence without duplicating deterministic dependent-row tests. |
| Wide rank-deficient residual fixture | Future QR minimum-norm or nullspace/subspace owner | Define underdetermined solve, solution-selection, minimum-norm, and nullspace boundaries before accepting residual evidence. |
| Multi-dimensional nullspace/subspace fixture | Future QR subspace owner | Compare full projectors or two-way projection residuals for nullity greater than 1; forbid raw basis ordering claims. |
| Wide-shape nullspace/subspace fixture | Future QR wide-shape owner | Pin expected rank/nullity and projector tolerance before accepting evidence. |
| Scaled diagonal threshold family | Future numerical-rank owner | Add scale metadata and prove ranks remain unchanged under named scales. |
| Perturbed duplicate-column threshold family | Future numerical-rank owner | Define perturbation sizes separated from thresholds by at least two orders of magnitude. |
| Dependent-row or wide near-threshold families | Future threshold/subspace owner | Define whether rank, residual, or nullspace is the primary claim before implementation. |
| SuiteSparse rank-deficient QR corpus evidence | Future corpus/platform owner | Pin expected-rank metadata, threshold semantics, support tier, diagnostics, skip behavior, and validation before registration. |
| Optional-large SuiteSparse minimum-norm evidence | Future corpus/platform owner | Apply optional-corpus support rules, skip diagnostics, platform expectations, residual/norm metrics, and focused validation. |
| SuiteSparse rank-deficient minimum-norm corpus | Future minimum-norm corpus owner | Pin rank, nullity, residual, norm, and corpus metadata before registration. |
| Additional QR-vs-SVD minimum-norm fixtures | Future QR/SVD cross-check owner | Define fixture keys, SVD tolerance, QR residual/norm metric, and cross-check wording per fixture. |
| Larger underdetermined minimum-norm exact-value lanes | Future QR minimum-norm owner | Decide which existing shape controls deserve expected-value contracts instead of residual-only smoke coverage. |
| Generic QR/SVD minimum-norm helper movement | Future helper owner | Use behavior-specific names and keep tolerances at call sites; run focused QR solve, COLAMD, SVD, and full quality validation. |
| Raw Q-column, wide economy, sparse-mode Q/economy, and SuiteSparse Q/economy follow-through | Future QR basis/economy owner | Reuse Sprint 124 projector policy and Sprint 125 corpus support rules before accepting any broader basis or corpus evidence. |

## Final Non-Claim Register

Sprint 125 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, ecosystem, dense-library, or broad external package parity;
- broad QR factorization, QR solve, rank-deficient solve, nullspace,
  minimum-norm, Q-basis, economy, sparse-mode, reorder, backend, corpus, or
  performance parity;
- global QR rank-threshold policy;
- raw QR Q-basis equality, Q sign/orientation, unique basis parity, raw
  nullspace basis equality, or broad projection/subspace parity;
- broad minimum-norm optimality, SVD-pseudoinverse-as-global-QR-oracle
  behavior, COLAMD superiority, fallback generality, refinement convergence
  guarantees, or SuiteSparse-wide minimum-norm behavior;
- broad SuiteSparse corpus correctness, optional-data behavior, platform
  parity, or performance behavior;
- helper API expansion, generic helper consolidation, CMake/CTest membership
  expansion, package behavior, ABI behavior, public API behavior, platform
  support, scalability, memory behavior, or state-of-the-art behavior.

## Sprint 126 Handoff

Sprint 126 and later work can treat Sprint 125's bounded evidence package as
stable:

| Sprint 126 need | Sprint 125 input |
| --- | --- |
| Rank-deficient QR residual policy | Day 2 trust gate and Day 3 residual-only lane; future residual lanes must not imply nullspace or minimum-norm behavior. |
| Nullspace/subspace policy | Day 4 policy and Day 5 projector lane; future basis work should use projector or two-way projection metrics unless it proves stable raw vector orientation. |
| Threshold-family policy | Day 6 policy and Day 7 diagonal ladder; future threshold work must remain fixture-local. |
| SuiteSparse QR corpus policy | Day 8 policy and Day 9 deferral; current checked-in QR SuiteSparse matrices remain full-rank controls. |
| Minimum-norm owner map | Day 10 owner map plus Day 11-12 accepted lanes; future helper work must keep QR solve, COLAMD, SVD, fallback, refinement, rank-deficient, zero-row, and corpus semantics visible. |
| Validation and claim boundary | Day 13 validation package and maintainer evidence row; public docs remain unchanged because evidence stays fixture-scoped or owner-local. |

## Day 14 Validation

Day 14 is documentation-only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_125 docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_qr_solve.c tests/test_colamd.c tests/test_svd.c
find . -path '*/__pycache__' -o -name '*.pyc'
```

No full C quality gate is required for Day 14 because Day 14 does not change
`.c`, `.h`, helper script, build, package, or public API files. The full code
quality baseline remains the Day 13 `make format && make lint && make test`
pass.

Day 14 documentation hygiene passed:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_125 docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_qr_solve.c tests/test_colamd.c tests/test_svd.c
find . -path '*/__pycache__' -o -name '*.pyc'
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| All seven Sprint 125 project-plan items are complete or explicitly deferred. | Complete | See closeout summary table. |
| Sprint 126 has clear inputs and no hidden dependencies. | Complete | See Sprint 126 handoff and residual queue. |
| Final validation status and residual non-claims are documented. | Complete | Day 13 full quality gate passed; Day 14 documentation hygiene passed. |
