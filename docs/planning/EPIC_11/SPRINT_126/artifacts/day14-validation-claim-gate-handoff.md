# Sprint 126 Day 14 Validation, Claim Gate, and Handoff

## Purpose

Close Sprint 126 from the Day 13 full-gate baseline, consolidate accepted
rank-deficient QR residual, nullspace/subspace, threshold, and minimum-norm
evidence, preserve explicit deferrals and non-claims, and hand clear
Q/economy/helper prerequisites to Sprint 127.

## Sprint 126 Closeout Summary

Sprint 126 completed all seven project-plan items through bounded
implementation, explicit policy, or explicit deferral with named promotion
gates.

| Project-plan item | Final state | Evidence |
| --- | --- | --- |
| 1. Sprint 125 Residual Dedupe and Dependency Map | Complete | Day 1 mapped Sprint 125 residual debt to completed Sprint 121-125 evidence, duplicate fences, and validation boundaries. |
| 2. Compatible and Wide Residual Fixtures | Complete with one bounded implementation and explicit deferrals | Day 2 trust policy plus Day 3 `qr_rankdef_dependent_row_4x3_residual_only` fixture. |
| 3. Nullspace and Subspace Evidence Expansion | Complete with one bounded implementation and explicit deferrals | Day 4 policy plus Day 5 `qr_rank1_4x3_nullspace_projector` fixture. |
| 4. Threshold Family Expansion | Complete with one bounded implementation and explicit deferrals | Day 6 policy plus Day 7 `qr_rank_threshold_diag4_scaled_family` fixture. |
| 5. SuiteSparse Rank-Deficient QR Corpus Follow-Through | Complete by explicit deferral package | Day 8 corpus gate plus Day 9 decision to keep current checked-in SuiteSparse QR matrices as full-rank controls. |
| 6. SuiteSparse and Underdetermined Minimum-Norm Evidence | Complete with one bounded underdetermined implementation and explicit SuiteSparse deferrals | Day 10-11 SuiteSparse minimum-norm gate and deferral plus Day 12-13 `qr_minnorm_5x10_exact_values` fixture. |
| 7. QR-vs-SVD Minimum-Norm Cross-Check Gate | Complete by explicit deferral package | Day 12-13 kept the Sprint 125 2 x 4 QR-vs-SVD cross-check as the bounded baseline and deferred broader cross-checks. |

## Accepted Implementation Package

| Accepted lane | Fixture or behavior | Owner surfaces | Claim boundary |
| --- | --- | --- | --- |
| Dependent-row residual-only QR evidence | `qr_rankdef_dependent_row_4x3_residual_only` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | One 4 x 3 dependent-row residual-only check against a standard-library projection reference; no solution-vector, rank, nullspace, minimum-norm, compatible, wide, or broad QR claim. |
| Multi-dimensional nullspace/subspace projector evidence | `qr_rank1_4x3_nullspace_projector` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | One rank-1/nullity-2 projector check against `I - 11^T / 3`; no raw basis equality, basis ordering, Q sign/orientation, minimum-norm, sparse-mode, or broad subspace claim. |
| Scaled threshold-family evidence | `qr_rank_threshold_diag4_scaled_family` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | Three scale values and three named relative thresholds for a diagonal ladder; no global rank-threshold, perturbation, default-threshold, SuiteSparse, or dense-library parity claim. |
| Exact 5 x 10 minimum-norm evidence | `qr_minnorm_5x10_exact_values` | `tests/test_colamd.c`, `docs/maintainer_guide.md` | One closed-form 5 x 10 fixture with exact solution values and `sqrt(11)` norm; no broad underdetermined, QR-vs-SVD, SVD-pseudoinverse, SuiteSparse, or global optimality claim. |

## Validation Package

Focused helper and executable validation passed during the implementation
days:

| Lane | Focused validation |
| --- | --- |
| `qr_rankdef_dependent_row_4x3_residual_only` | `python3 -m py_compile tests/qr_external_dense_reference.py`; helper emitted `OK 1`; `make build/test_qr_solve && ./build/test_qr_solve` passed 18 tests. |
| `qr_rank1_4x3_nullspace_projector` | helper emitted `OK 13`; `make build/test_qr && ./build/test_qr` passed 69 tests with projector diff `2.220e-16`, null residual `5.088e-16`, and orthogonality error `2.220e-16`. |
| `qr_rank_threshold_diag4_scaled_family` | helper emitted `OK 27`; `make build/test_qr && ./build/test_qr` passed 70 tests and checked all scale/threshold rank pairs. |
| SuiteSparse QR corpus decision | `make build/test_qr_solve && ./build/test_qr_solve` passed 19 tests and confirmed checked-in SuiteSparse controls remain full-rank for this lane. |
| SuiteSparse minimum-norm decision | `make build/test_colamd && ./build/test_colamd` passed 70 tests and preserved the Sprint 125 `west0067` submatrix smoke as the only accepted SuiteSparse minimum-norm corpus evidence. |
| `qr_minnorm_5x10_exact_values` | `make build/test_colamd && ./build/test_colamd` passed 70 tests, 0 failures, 0 skips, and 310 assertions; the fixture printed max residual `8.88e-16` and `||x||=3.3166`. |

Because Sprint 126 changed `.c` and Python helper files, Day 14 reran and
passed the full required quality gate:

```text
$ make format && make lint && make test
All tests passed.
```

Day 14 updates planning and maintainer documentation only. No source, header,
helper script, build metadata, package metadata, public API, or public
solver-selection wording changed on Day 14.

## Maintainer Evidence Gate

`docs/maintainer_guide.md` now names the accepted Sprint 126 QR evidence in
the maintained QR evidence row:

- `qr_rankdef_dependent_row_4x3_residual_only`
- `qr_rank1_4x3_nullspace_projector`
- `qr_rank_threshold_diag4_scaled_family`
- `qr_minnorm_5x10_exact_values`

The same row keeps the existing QR non-claims for broad QR, LAPACK, NumPy,
SciPy, global rank-threshold, raw Q-basis, sign/orientation, broad
rank-deficient solve, nullspace, minimum-norm, economy-mode, sparse-mode,
reorder, SVD-pseudoinverse-as-global-oracle, broad SuiteSparse corpus, and
performance external parity.

## Public Claim Gate

`docs/solver_selection.md`, `README.md`, public headers, and package metadata
do not change for Sprint 126 Day 14.

The accepted evidence improves named-fixture and owner-local confidence, but
does not justify broader public wording for:

- broad QR, dense-library, or external package parity;
- compatible or wide rank-deficient QR solve behavior;
- global QR rank-threshold behavior;
- raw nullspace basis, Q-basis orientation, or broad subspace behavior;
- broad underdetermined or minimum-norm optimality;
- SVD-pseudoinverse-as-global-QR-oracle behavior;
- SuiteSparse-wide corpus behavior;
- sparse-mode, economy-mode, reorder, backend, platform, performance, package,
  ABI, public API, or state-of-the-art behavior.

## Consolidated Residual and Future-Owner Queue

| Residual | Future owner | Promotion gate |
| --- | --- | --- |
| Compatible zero-residual rank-deficient QR residual fixture | Future QR solve residual owner | Prove zero-residual evidence adds trust beyond deterministic compatible solve behavior and cannot be misread as minimum-norm evidence. |
| Wide residual-only QR fixture | Sprint 127 QR Q/economy/sparse-mode or later minimum-norm owner | Define underdetermined output semantics, solution-selection policy, Q/economy boundaries, and residual-only proof value before accepting evidence. |
| Wide-shape nullspace/subspace fixture | Sprint 127 QR Q/economy/sparse-mode owner | Pin expected rank/nullity, projection metric, tolerance, and sparse/economy output semantics before accepting evidence. |
| Dependent-row, near-threshold, or SuiteSparse nullspace/subspace evidence | Future QR subspace owner | Define projector or two-way projection residual metrics, expected rank/nullity, threshold semantics, and support tier. |
| Perturbed duplicate-column threshold family | Future numerical-rank owner | Define perturbation sizes separated from thresholds by at least two orders of magnitude and avoid default-threshold claims. |
| Dependent-row, wide, default-threshold, or SuiteSparse threshold family | Future threshold/subspace owner | Pin the primary claim, expected ranks, threshold semantics, support tier, diagnostics, and failure interpretation. |
| SuiteSparse rank-deficient QR corpus evidence | Future corpus/platform owner | Pin expected-rank metadata, threshold semantics, support tier, diagnostics, skip behavior, runtime budget, and validation before registration. |
| Additional SuiteSparse minimum-norm evidence | Future minimum-norm corpus owner | Pin extraction rule, shape, nnz, RHS, rank/nullity if claimed, residual/norm metrics, skip behavior, and support tier. |
| Optional-large SuiteSparse QR or minimum-norm evidence | Future corpus/platform owner | Use the optional-large gate, prove missing-data skip behavior, and record runtime/platform expectations before default test registration. |
| Additional QR-vs-SVD minimum-norm fixtures | Future QR/SVD cross-check owner | Define fixture keys, QR residual and norm metrics, SVD tolerance, and non-oracle wording per fixture. |
| Larger exact underdetermined minimum-norm lanes | Future QR minimum-norm owner | Choose non-duplicate shapes with closed-form expected values and explicit residual/value/norm tolerances. |
| Generic QR/SVD helper movement | Sprint 127 helper owner | Use behavior-specific helper names, keep tolerances at call sites, and run focused QR solve, COLAMD, SVD, and full quality validation. |
| Raw Q-column, wide economy, sparse-mode Q/economy, and SuiteSparse Q/economy follow-through | Sprint 127 QR basis/economy owner | Reuse Sprint 124 projector policy, Sprint 125/126 corpus support rules, and named output-shape semantics before accepting basis or corpus evidence. |

## Final Non-Claim Register

Sprint 126 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, ecosystem, dense-library, external package, or broad
  library parity;
- broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, nullspace, minimum-norm, Q-basis, economy, sparse-mode,
  reorder, backend, corpus, or performance parity;
- global QR rank-threshold, default-threshold, or numerical-rank policy;
- raw QR Q-basis equality, Q sign/orientation, unique basis parity, raw
  nullspace basis equality, or broad projection/subspace parity;
- broad minimum-norm optimality, SVD-pseudoinverse-as-global-QR-oracle
  behavior, COLAMD superiority, fallback generality, refinement convergence
  guarantees, or SuiteSparse-wide minimum-norm behavior;
- broad SuiteSparse corpus correctness, optional-data behavior, platform
  parity, runtime behavior, or performance behavior;
- helper API expansion, generic helper consolidation, CMake/CTest membership
  expansion, package behavior, ABI behavior, public API behavior, platform
  support, scalability, memory behavior, or state-of-the-art behavior.

## Sprint 127 Handoff

Sprint 127 and later work can treat Sprint 126's bounded evidence package as
stable:

| Sprint 127 need | Sprint 126 input |
| --- | --- |
| Q-basis and economy semantics | Current rank-deficient residual/nullspace work remains projector- or residual-based; no raw Q-basis or economy output claim was added. |
| Sparse-mode Q/economy evidence | Wide and sparse-mode lanes remain deferred until output semantics and projector metrics are pinned. |
| Helper ownership | Helper movement remains deferred; Sprint 126 kept behavior-specific fixtures and call-site tolerances visible. |
| Corpus support tiering | SuiteSparse QR and minimum-norm evidence remains bounded to existing full-rank controls and the Sprint 125 `west0067` minimum-norm smoke. |
| Public wording | Maintainer evidence was refreshed; public solver-selection, README, headers, packages, and API wording remain unchanged. |

## Day 14 Validation

Day 14 is documentation-only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_126 docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_qr_solve.c tests/test_colamd.c
find . -path '*/__pycache__' -o -name '*.pyc'
```

Day 14 documentation hygiene passed:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_126 docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_qr_solve.c tests/test_colamd.c
find . -path '*/__pycache__' -o -name '*.pyc'
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every accepted implementation has validation evidence. | Complete | See validation package and Day 13 full quality gate. |
| Every deferred item has owner, blocker, and promotion-gate notes. | Complete | See consolidated residual and future-owner queue. |
| Public/support claims do not exceed earned Sprint 126 evidence. | Complete | Maintainer evidence row is bounded; public docs remain unchanged. |
| Sprint 127 receives clear Q/economy/helper prerequisites. | Complete | See Sprint 127 handoff. |
