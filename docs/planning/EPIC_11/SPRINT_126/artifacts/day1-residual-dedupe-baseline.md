# Sprint 126 Day 1 Residual Dedupe Baseline

## Purpose

Day 1 establishes the Sprint 126 working structure and converts Sprint 125's
rank-deficient QR, nullspace/subspace, threshold, SuiteSparse, and
minimum-norm carry-forward debt into dependency-ordered proof owners. The goal
is to make each residual lane visible without duplicating completed Sprint
121-125 evidence.

## Inputs Reviewed

| Input | Relevant Content |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 126 | Defines seven Sprint 126 items and the 166-hour project-plan estimate. |
| `docs/planning/EPIC_11/SPRINT_126/PLAN.md` | Splits Sprint 126 into 14 days with no day above 12 hours. |
| `docs/planning/EPIC_11/SPRINT_125/RETROSPECTIVE.md` | Names the residual deferred debt and the explicit non-claim register. |
| `docs/planning/EPIC_11/SPRINT_125/WORKING_NOTES.md` | Captures completed bounded fixtures, validation rules, duplicate fences, and future-owner handoffs. |
| Sprint 125 Day 1 artifact | Previous deferred QR/minimum-norm dedupe map and duplicate fences. |
| Sprint 125 Day 2-3 artifacts | Residual-only trust gate and completed `qr_rankdef_duplicate_5x4_residual_only` evidence. |
| Sprint 125 Day 4-5 artifacts | Nullspace/subspace policy and completed duplicate-column 5x4 nullspace projector evidence. |
| Sprint 125 Day 6-7 artifacts | Near-rank threshold policy and completed `qr_rank_threshold_diag4_family` evidence. |
| Sprint 125 Day 8-9 artifacts | SuiteSparse rank-deficient QR corpus policy and explicit deferral requirements. |
| Sprint 125 Day 10-12 artifacts | Minimum-norm owner map, COLAMD/fallback/rank-deficient/refinement/zero-row evidence, QR-vs-SVD bounded cross-check decision, and `west0067` minimum-norm submatrix smoke. |
| Sprint 125 Day 13-14 artifacts | Validation gate, maintainer evidence, non-claim register, and Sprint 126 handoff boundaries. |
| Sprint 121-124 artifacts | Earlier QR/SVD/rank taxonomy, bounded external-reference lanes, Q/economy policies, and helper ownership decisions. |

## Day 1 Created Structure

| Path | Purpose |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_126/PLAN.md` | Day-by-day Sprint 126 execution plan. |
| `docs/planning/EPIC_11/SPRINT_126/WORKING_NOTES.md` | Running Sprint 126 notes, validation rules, day ownership, and scope boundaries. |
| `docs/planning/EPIC_11/SPRINT_126/artifacts/day1-residual-dedupe-baseline.md` | Sprint intake, residual dedupe map, duplicate fence, validation boundary, and completion check. |

## Sprint 126 Project-Plan Item Map

| Item | Item Name | Day Owner | Day 1 Interpretation |
| --- | --- | --- | --- |
| 1 | Sprint 125 Residual Dedupe and Dependency Map | Day 1 | Map Sprint 125 residual deferred debt against completed Sprint 121-125 evidence before any new lane is accepted. |
| 2 | Compatible and Wide Residual Fixtures | Days 2-3 | Add or explicitly defer compatible zero-residual, dependent-row, and wide rank-deficient QR residual fixtures only if they add distinct trust without implying nullspace or minimum-norm behavior. |
| 3 | Nullspace/Subspace Evidence Expansion | Days 4-5 | Add or explicitly defer multi-dimensional, wide-shape, near-threshold, dependent-row, and SuiteSparse nullspace/subspace evidence only through projector or two-way projection metrics with pinned rank/nullity metadata. |
| 4 | Threshold Family Expansion | Days 6-7 | Add or explicitly defer scaled diagonal, perturbed duplicate-column, dependent-row, wide, and SuiteSparse threshold families with fixture-local expected ranks and non-global interpretation. |
| 5 | SuiteSparse Rank-Deficient QR Corpus Gate | Days 8-9 | Gate SuiteSparse rank-deficient QR corpus evidence behind expected-rank metadata, support tier, diagnostics, skip behavior, and validation requirements. |
| 6 | SuiteSparse and Underdetermined Minimum-Norm Evidence | Days 10-13 | Add or explicitly defer optional-large SuiteSparse, rank-deficient SuiteSparse, and larger underdetermined minimum-norm evidence with pinned residual, norm, rank, nullity, corpus metadata, and exact-value ownership where justified. |
| 7 | QR-vs-SVD Minimum-Norm Cross-Check Gate | Days 12-14 | Add or explicitly defer additional QR-vs-SVD minimum-norm fixtures only as bounded cross-checks with explicit fixture keys, tolerances, and non-oracle wording. |

## Residual Debt Dedupe Map

| Sprint 125 Deferred Work | Sprint 126 Owner | Completed Evidence to Reuse | Duplicate Fence | Promotion Gate |
| --- | --- | --- | --- | --- |
| Compatible zero-residual rank-deficient QR residual fixtures | Days 2-3 | Deterministic compatible QR solve tests and Sprint 125 residual-only trust gate. | Do not relabel existing compatible solve residuals as new rank-deficient residual corpus evidence. | Prove the fixture adds trust beyond existing deterministic compatible solves and does not imply minimum-norm or nullspace behavior. |
| Dependent-row residual fixtures | Days 2-3 | `tf_qr_make_dependent_row_4x3` and completed rank/nullspace-adjacent deterministic evidence. | Do not duplicate dependent-row rank checks or nullspace policy evidence as residual-only proof. | Define a residual-only behavior, expected residual, diagnostics, and explicit non-claims. |
| Wide rank-deficient residual fixtures | Days 2-3 | Existing wide QR and underdetermined minimum-norm tests. | Do not conflate wide residual evidence with minimum-norm optimality or exact-value contracts. | Show shape-specific residual trust value and fence minimum-norm interpretation. |
| Multi-dimensional nullspace/subspace evidence | Days 4-5 | Sprint 125 nullspace/subspace policy and duplicate-column projector evidence. | Do not compare raw vectors or claim unique basis orientation from one-dimensional projector evidence. | Pin rank/nullity metadata and use projection or two-way projection metrics. |
| Wide-shape nullspace/subspace evidence | Days 4-5 | Existing wide QR tests and underdetermined minimum-norm fixture. | Do not treat wide minimum-norm behavior as nullspace subspace evidence. | Define shape, rank, nullity, projection metric, tolerance, and failure diagnostics. |
| Near-threshold nullspace/subspace evidence | Days 4-5 | `qr_rank_threshold_diag4_family` and threshold policy. | Do not mix threshold-rank evidence with nullspace basis equality claims. | Pin threshold, expected rank/nullity, projection metric, and non-global interpretation. |
| SuiteSparse nullspace/subspace evidence | Days 4-5 and 8-9 | SuiteSparse support-tier policy and `west0067` minimum-norm submatrix smoke. | Do not treat one SuiteSparse submatrix smoke as corpus-wide nullspace evidence. | Define corpus availability, expected rank/nullity metadata, skip behavior, and support tier. |
| Scaled diagonal threshold families | Days 6-7 | Completed diagonal threshold family and expected-rank metadata checks. | Do not duplicate the existing `1e-14`, `1e-10`, `1e-6` fixture without new scale trust value. | Define scale-specific expected ranks, diagnostics, and non-global interpretation. |
| Perturbed duplicate-column threshold families | Days 6-7 | Duplicate-column rank/residual/nullspace fixtures. | Do not reinterpret exact duplicate-column fixtures as near-threshold perturbation evidence. | Define perturbation, threshold values, expected ranks, and stability limits. |
| Dependent-row and wide threshold families | Days 6-7 | Dependent-row and wide deterministic QR fixtures. | Do not create a generic QR numerical-rank policy from fixture-local thresholds. | Define fixture-local ranks, threshold metadata, and failure diagnostics. |
| SuiteSparse threshold families | Days 6-9 | SuiteSparse support-tier and optional-corpus policy. | Do not claim broad SuiteSparse rank-threshold behavior. | Define matrix, expected rank metadata, optional-data behavior, and validation requirements. |
| SuiteSparse rank-deficient QR corpus evidence | Days 8-9 | Sprint 125 SuiteSparse policy and checked-in `west0067` submatrix smoke. | Do not treat optional or skipped corpus data as reviewed support. | Define expected-rank metadata, support tier, diagnostics, skip behavior, and validation. |
| Optional-large SuiteSparse minimum-norm evidence | Days 10-11 | Sprint 125 `west0067` minimum-norm submatrix smoke and SuiteSparse optional-data conventions. | Do not broaden the checked-in smoke into optional-large support without metadata and skip proof. | Pin residual, norm, rank, nullity, shape, corpus metadata, support tier, and skip behavior. |
| Rank-deficient SuiteSparse minimum-norm evidence | Days 10-11 | Sprint 125 owner-local minimum-norm evidence and rank-deficient QR policies. | Do not claim SuiteSparse-wide minimum-norm or pseudoinverse behavior. | Define rank/nullity metadata, norm comparison, residual expectation, diagnostics, and claim boundaries. |
| Larger underdetermined minimum-norm exact-value lanes | Days 12-13 | `qr_underdetermined_minnorm_2x4` and owner-local minimum-norm tests. | Do not relabel the 2x4 exact fixture as larger-shape evidence. | Decide which shapes deserve exact-value contracts and pin fixture keys, values, and tolerances. |
| Additional QR-vs-SVD minimum-norm fixtures | Days 12-13 | Sprint 125 bounded QR-vs-SVD cross-check decision. | Do not treat SVD pseudoinverse as a global QR oracle or dense-library parity claim. | Define fixture keys, tolerances, diagnostic role, and non-oracle wording. |
| Generic QR/SVD minimum-norm helper movement | Day 14 handoff to Sprint 127 | Sprint 125 behavior-specific owner map and Sprint 127 helper ownership plan. | Do not introduce generic helpers during Sprint 126 unless required for accepted evidence and validated locally. | Preserve behavior-specific helper names and defer broad consolidation to Sprint 127. |
| Raw Q-column, wide economy, sparse-mode Q/economy, and SuiteSparse Q/economy follow-through | Day 14 handoff to Sprint 127 | Sprint 124 Q/economy policy and `qr_economy_projector_5x3`. | Do not absorb Q/economy implementation into Sprint 126 residual corpus work. | Hand prerequisite metadata and claim boundaries to Sprint 127. |

## Completed Work Duplicate Fence

| Completed Work | Sprint 126 Handling |
| --- | --- |
| Sprint 121 SVD/QR/rank fixture taxonomy | Use as taxonomy input; do not redesign unless a Sprint 126 decision exposes a concrete gap. |
| Sprint 122 bounded SVD/QR/partial-SVD external oracle lane designs | Use as trust-boundary input; do not reopen earlier decisions. |
| Sprint 123 QR compatible, rank-deficient, underdetermined, and helper decisions | Use as completed evidence and ownership context; do not duplicate as new Sprint 126 evidence. |
| Sprint 124 Q-basis/economy policy and helper decisions | Use as Sprint 127 handoff context; do not pull raw Q/economy work into Sprint 126 unless it is a prerequisite artifact. |
| `qr_rankdef_duplicate_5x4_rank_only` | Use as completed rank-only evidence; do not relabel as residual, nullspace, threshold, SuiteSparse, or minimum-norm proof. |
| `qr_rankdef_duplicate_5x4_residual_only` | Use as completed Sprint 125 residual-only evidence; do not duplicate as compatible, dependent-row, wide, or SuiteSparse residual evidence. |
| `qr_rankdef_duplicate_5x4_nullspace_projector` | Use as completed one-dimensional projector evidence; do not relabel as multi-dimensional, wide, near-threshold, or SuiteSparse subspace evidence. |
| `qr_rank_threshold_diag4_family` | Use as completed diagonal threshold evidence; do not relabel as scaled, perturbed, dependent-row, wide, or SuiteSparse threshold evidence. |
| `qr_underdetermined_minnorm_2x4` | Use as completed exact underdetermined minimum-norm evidence; do not relabel as larger underdetermined, SuiteSparse, rank-deficient SuiteSparse, or QR-vs-SVD proof. |
| Sprint 125 COLAMD/fallback/rank-deficient/refinement/zero-row minimum-norm evidence | Use as owner-local baseline; do not create generic minimum-norm helper proof or broad QR minimum-norm claims. |
| Sprint 125 QR-vs-SVD bounded cross-check decision | Use as non-oracle rule; additional fixtures must preserve bounded diagnostic wording. |
| Sprint 125 `west0067` 30 x 67 minimum-norm submatrix smoke | Use as checked-in SuiteSparse minimum-norm baseline; do not claim optional-large or corpus-wide behavior. |
| `qr_economy_projector_5x3` | Use as Sprint 127 Q/economy input; not Sprint 126 residual corpus or minimum-norm scope. |
| `partial_svd_vector_residual_diag6_k2` | Use as Sprint 128 partial-SVD input; not Sprint 126 QR scope. |

## Validation Boundary

| Scenario | Required Validation |
| --- | --- |
| Documentation-only Day 1 work | `git diff --check` and focused whitespace scan for Sprint 126 markdown files. |
| Future `.c` or `.h` changes | `make format && make lint && make test`. |
| Future Python helper changes | `python3 -m py_compile`, focused helper invocation, affected test executable, and protocol-output proof. |
| Future QR or minimum-norm test changes | Focused test executable proof plus full quality if C or header files changed. |
| Future Makefile/CMake/CTest membership changes | Source-list and CMake/CTest impact proof, including platform count notes where applicable. |
| Future SuiteSparse optional-corpus evidence | Optional-data present/missing behavior, skip-path proof, support-tier diagnostics, and bounded claim note. |
| Future maintainer/public wording changes | Evidence-to-claim traceability, link/path hygiene, claim-boundary scan, and non-claim update. |

## Non-Claim Boundary

Sprint 126 Day 1 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad dense-library parity;
- broad QR factorization, QR solve, rank-deficient solve, nullspace,
  minimum-norm, Q-basis, economy, sparse-mode, reorder, backend, corpus, or
  performance parity;
- compatible, dependent-row, wide, multi-dimensional, near-threshold, or
  SuiteSparse rank-deficient QR evidence beyond completed Sprint 125 fixtures;
- raw nullspace basis equality, unique basis orientation, Q-sign,
  Q-orientation, or subspace external parity;
- global near-rank-deficient threshold policy;
- broad QR minimum-norm, optional-large SuiteSparse, rank-deficient
  SuiteSparse, larger underdetermined, QR-vs-SVD-pseudoinverse, or
  SVD-pseudoinverse oracle parity;
- package, ABI, platform, public API, CMake, Makefile, CI, CTest,
  performance, scalability, memory, or state-of-the-art claims.

## Downstream Handoff Needs

| Downstream Need | Day 1 Owner Boundary |
| --- | --- |
| Day 2 residual fixture trust policy | Must start from completed Sprint 125 residual-only evidence, then prove compatible/dependent-row/wide candidates add distinct trust. |
| Day 4 nullspace/subspace expansion policy | Must not start implementation until rank/nullity metadata and projection metrics are explicit. |
| Day 6 threshold family expansion policy | Must reuse completed diagonal threshold evidence while avoiding global threshold claims. |
| Day 8 SuiteSparse QR corpus gate | Must decide expected-rank metadata, support tier, skip behavior, diagnostics, and validation before implementation. |
| Day 10 SuiteSparse minimum-norm gate | Must decide residual, norm, rank, nullity, corpus metadata, and optional-data behavior before implementation. |
| Day 12 QR-vs-SVD cross-check gate | Must preserve bounded non-oracle wording and explicit fixture keys. |
| Day 14 validation and handoff | Must validate all accepted work and hand Q/economy/helper prerequisites to Sprint 127 without expanding claims. |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 126 project-plan item has a day-level owner. | Complete | See Sprint 126 project-plan item map. |
| Completed Sprint 121-125 evidence is not duplicated or silently reopened. | Complete | See completed work duplicate fence. |
| Dependency order is explicit before any new evidence is accepted. | Complete | See residual debt dedupe map and downstream handoff needs. |
| Sprint 126 working notes exist. | Complete | `WORKING_NOTES.md` created. |
| Sprint 126 artifact directory exists. | Complete | `artifacts/day1-residual-dedupe-baseline.md` created. |
| Validation expectations are explicit. | Complete | See validation boundary. |
