# Sprint 127 Day 14 Validation, Claim Gate, and Handoff

## Purpose

Close Sprint 127 from the Day 13 full-gate baseline, consolidate accepted
rank-deficient QR residual, nullspace/subspace, threshold, and minimum-norm
evidence, preserve explicit deferrals and non-claims, and hand clear
Q/economy/helper prerequisites to Sprint 128.

## Sprint 127 Closeout Summary

Sprint 127 completed all seven project-plan items through bounded
implementation, explicit policy, or explicit deferral with named promotion
gates.

| Project-plan item | Final state | Evidence |
| --- | --- | --- |
| 1. Sprint 126 Deferred Dedupe Map | Complete | Day 1 mapped Sprint 126 residual debt to completed Sprint 121-126 evidence, duplicate fences, validation boundaries, and Sprint 128 handoff needs. |
| 2. Compatible and Wide Residual Semantics | Complete by policy and explicit deferral package | Day 2 semantics policy plus Day 3 evidence decision kept completed residual-only lanes bounded and deferred compatible zero-residual/wide residual expansion. |
| 3. Nullspace and Subspace Expansion | Complete with one bounded implementation and explicit deferrals | Day 4 policy plus Day 5 `qr_rankdef_dependent_row_4x3_nullspace_projector` fixture. |
| 4. Threshold Family Follow-Through | Complete with one bounded implementation and explicit deferrals | Day 6 policy plus Day 7 `qr_rank_threshold_duplicate_5x4_perturbed_family` fixture. |
| 5. SuiteSparse Rank-Deficient QR Corpus Evidence | Complete by explicit deferral package | Day 8 corpus gate plus Day 9 decision to keep checked-in SuiteSparse QR matrices as controls until independent rank/nullity metadata exists. |
| 6. SuiteSparse and Optional-Large Minimum-Norm Gate | Complete by explicit deferral package | Day 10 gate plus Day 11 decision to preserve the Sprint 125 `west0067` 30 x 67 smoke as the only accepted SuiteSparse minimum-norm corpus evidence. |
| 7. Minimum-Norm Cross-Check and Helper Claim Gate | Complete with one bounded implementation and explicit deferrals | Day 12 gate plus Day 13 `qr_minnorm_3x6_exact_values` fixture; additional QR-vs-SVD checks and helper movement remain deferred. |

## Accepted Implementation Package

| Accepted lane | Fixture or behavior | Owner surfaces | Claim boundary |
| --- | --- | --- | --- |
| Dependent-row nullspace/subspace projector evidence | `qr_rankdef_dependent_row_4x3_nullspace_projector` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | One 4 x 3 dependent-row projector check against a rank-2/nullity-1 closed-form projector; no raw basis equality, basis orientation, minimum-norm, sparse-mode, economy, or broad subspace claim. |
| Perturbed duplicate-column threshold evidence | `qr_rank_threshold_duplicate_5x4_perturbed_family` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | One duplicate-column 5 x 4 fixture with perturbation `6e-8`, rank 4 at `1e-10`, and rank 3 at `1e-6`; no global threshold, default-threshold, SuiteSparse, or dense-library parity claim. |
| Exact 3 x 6 minimum-norm evidence | `qr_minnorm_3x6_exact_values` | `tests/test_colamd.c`, `docs/maintainer_guide.md` | One closed-form 3 x 6 fixture with exact solution values and `sqrt(8.4)` norm; no broad underdetermined, QR-vs-SVD, SVD-pseudoinverse, SuiteSparse, optional-large, or global optimality claim. |

## Validation Package

Focused helper and executable validation passed during the implementation
days:

| Lane | Focused validation |
| --- | --- |
| `qr_rankdef_dependent_row_4x3_nullspace_projector` | `make build/test_qr && ./build/test_qr` passed 71 tests after the projector lane landed; helper output was covered through the external QR dense reference path. |
| `qr_rank_threshold_duplicate_5x4_perturbed_family` | Helper emitted `OK 6`; `make build/test_qr && ./build/test_qr` passed 72 tests and printed both accepted rank-threshold diagnostics. |
| SuiteSparse rank-deficient QR corpus decision | `make build/test_qr_solve && ./build/test_qr_solve` passed 19 tests and reconfirmed ranks `100`, `132`, and `67` for `nos4`, `bcsstk04`, and `west0067`. |
| SuiteSparse and optional-large minimum-norm decision | `make build/test_colamd && ./build/test_colamd` passed 70 tests, 0 failures, 0 skips, and 310 assertions, preserving the existing `west0067` submatrix smoke. |
| `qr_minnorm_3x6_exact_values` | `make build/test_colamd && ./build/test_colamd` passed 70 tests, 0 failures, 0 skips, and 317 assertions; the fixture printed max residual `1.78e-15` and `||x||=2.8983`. |

Because Sprint 127 changed `.c` and Python helper files, Day 13 reran and
passed the full required quality gate:

```text
$ make format && make lint && make test
All tests passed.
```

Day 14 updates planning and maintainer documentation only. No source, header,
helper script, build metadata, package metadata, public API, or public
solver-selection wording changed on Day 14.

## Maintainer Evidence Gate

`docs/maintainer_guide.md` now names the accepted Sprint 127 exact
minimum-norm lane in the maintained QR evidence row:

- `qr_minnorm_3x6_exact_values`

The same row keeps existing bounded QR evidence and non-claims for broad QR,
LAPACK, NumPy, SciPy, global rank-threshold policy, raw Q-basis,
Q-sign/orientation, broad rank-deficient solve, nullspace, minimum-norm,
economy-mode, sparse-mode, reorder, SVD-pseudoinverse-as-global-oracle, broad
SuiteSparse corpus, and performance external parity.

## Public Claim Gate

`docs/solver_selection.md`, `README.md`, public headers, package metadata, and
public API documentation do not change for Sprint 127 Day 14.

The accepted evidence improves named-fixture and owner-local confidence, but
does not justify broader public wording for:

- broad QR, dense-library, or external package parity;
- compatible or wide rank-deficient QR solve behavior;
- global QR rank-threshold or default-threshold behavior;
- raw nullspace basis, Q-basis orientation, or broad subspace behavior;
- broad underdetermined or minimum-norm optimality;
- SVD-pseudoinverse-as-global-QR-oracle behavior;
- SuiteSparse-wide corpus behavior;
- optional-large behavior;
- sparse-mode, economy-mode, reorder, backend, platform, performance, package,
  ABI, public API, or state-of-the-art behavior.

## Consolidated Residual and Future-Owner Queue

| Residual | Future owner | Promotion gate |
| --- | --- | --- |
| Compatible zero-residual rank-deficient QR residual fixture | Future QR solve residual owner | Prove zero-residual evidence adds trust beyond deterministic compatible solve behavior and cannot be misread as minimum-norm evidence. |
| Wide residual-only QR fixture | Sprint 128 QR Q/economy/sparse-mode or later minimum-norm owner | Define underdetermined output semantics, solution-selection policy, Q/economy boundaries, and residual-only proof value before accepting evidence. |
| Wide-shape nullspace/subspace fixture | Sprint 128 QR Q/economy/sparse-mode owner | Pin expected rank/nullity, projection metric, tolerance, and sparse/economy output semantics before accepting evidence. |
| Near-threshold or SuiteSparse nullspace/subspace evidence | Future QR subspace owner | Define projector or two-way projection residual metrics, expected rank/nullity, threshold semantics, and support tier. |
| Dependent-row, wide, default-threshold, or SuiteSparse threshold family | Future threshold/subspace owner | Pin the primary claim, expected ranks, threshold semantics, support tier, diagnostics, and failure interpretation. |
| SuiteSparse rank-deficient QR corpus evidence | Future corpus/platform owner | Pin independent expected-rank metadata, threshold semantics, support tier, diagnostics, skip behavior, runtime budget, and validation before registration. |
| Additional SuiteSparse minimum-norm evidence | Future minimum-norm corpus owner | Pin extraction rule, shape, nnz, RHS, rank/nullity if claimed, residual/norm metrics, skip behavior, and support tier. |
| Optional-large SuiteSparse QR or minimum-norm evidence | Future corpus/platform owner | Use the optional-large gate, prove missing-data skip behavior, and record runtime/platform expectations before default test registration. |
| Additional QR-vs-SVD minimum-norm fixtures | Future QR/SVD cross-check owner | Define fixture keys, QR residual and norm metrics, SVD tolerance, and non-oracle wording per fixture. |
| Larger exact underdetermined minimum-norm lanes | Future QR minimum-norm owner | Choose non-duplicate shapes with closed-form expected values and explicit residual/value/norm tolerances. |
| Generic QR/SVD helper movement | Sprint 128 helper owner | Use behavior-specific helper names, keep tolerances at call sites, and run focused QR solve, COLAMD, SVD, and full quality validation. |
| Raw Q-column, wide economy, sparse-mode Q/economy, and SuiteSparse Q/economy follow-through | Sprint 128 QR basis/economy owner | Reuse Sprint 124 projector policy, Sprint 125-127 corpus support rules, named output-shape semantics, projection metrics, support-tier rules, and claim boundaries before accepting basis or corpus evidence. |

## Final Non-Claim Register

Sprint 127 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, ecosystem, dense-library, external package, or broad
  library parity;
- broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, nullspace, minimum-norm, Q-basis, economy, sparse-mode,
  reorder, backend, corpus, optional-data, platform, or performance parity;
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

## Sprint 128 Handoff

Sprint 128 can treat Sprint 127's bounded evidence package as stable:

| Sprint 128 need | Sprint 127 input |
| --- | --- |
| Q-basis and economy semantics | Sprint 127 kept rank-deficient residual/nullspace work projector- or residual-based; no raw Q-basis or economy output claim was added. |
| Wide economy and sparse-mode evidence | Wide and sparse-mode lanes remain deferred until output semantics and projector metrics are pinned. |
| SuiteSparse Q/economy evidence | SuiteSparse rank-deficient QR and minimum-norm corpus work remains blocked on independent metadata, support-tier, skip, and runtime rules. |
| Helper ownership | Generic helper movement remains deferred; Sprint 127 kept behavior-specific fixtures and call-site tolerances visible. |
| Public wording | Maintainer evidence was refreshed only for the new bounded 3 x 6 exact minimum-norm lane; public solver-selection, README, headers, packages, and API wording remain unchanged. |

## Retrospective Input Package

The Sprint 127 retrospective should use:

- `docs/planning/EPIC_11/SPRINT_127/PLAN.md`;
- `docs/planning/EPIC_11/SPRINT_127/WORKING_NOTES.md`;
- all `docs/planning/EPIC_11/SPRINT_127/artifacts/day*.md` artifacts;
- touched code/helper surfaces `tests/test_qr.c`,
  `tests/qr_external_dense_reference.py`, and `tests/test_colamd.c`;
- `docs/maintainer_guide.md` for the bounded maintainer evidence update;
- the Day 13 full quality-gate result.

## Day 14 Validation

Day 14 is documentation-only. Required validation:

```text
git diff --check
find docs/planning/EPIC_11/SPRINT_127 -type f -name '*.md' -print0 | \
  xargs -0 awk '(/[ \t]$/){print FILENAME ":" FNR ": trailing whitespace"; bad=1} END{exit bad}'
rg -n "[[:blank:]]$" docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_colamd.c
find . -path '*/__pycache__' -o -name '*.pyc'
```

Day 14 documentation hygiene passed:

```text
git diff --check
find docs/planning/EPIC_11/SPRINT_127 -type f -name '*.md' -print0 | \
  xargs -0 awk '(/[ \t]$/){print FILENAME ":" FNR ": trailing whitespace"; bad=1} END{exit bad}'
rg -n "[[:blank:]]$" docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_colamd.c
find . -path '*/__pycache__' -o -name '*.pyc'
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 127 project-plan item is implemented or explicitly deferred. | Complete | See Sprint 127 closeout summary. |
| Required checks pass or blockers are documented before closeout. | Complete | Day 13 full quality gate passed; Day 14 documentation hygiene is recorded separately. |
| No unsupported QR, nullspace, minimum-norm, SuiteSparse, optional-data, helper, platform, performance, or parity claim is introduced. | Complete | Maintainer update is bounded and public wording remains unchanged. |
| Sprint 128 receives clear Q/economy/helper prerequisites. | Complete | See Sprint 128 handoff. |
