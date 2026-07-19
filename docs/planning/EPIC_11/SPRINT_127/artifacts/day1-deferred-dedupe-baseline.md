# Sprint 127 Day 1 Deferred Dedupe Baseline

## Purpose

Day 1 establishes the Sprint 127 working structure and converts Sprint 126's
rank-deficient QR residual, nullspace/subspace, threshold-family,
SuiteSparse corpus, optional-large, minimum-norm, QR-vs-SVD, and helper
carry-forward debt into dependency-ordered proof owners. The goal is to make
each residual lane visible without duplicating completed Sprint 121-126
evidence.

## Inputs Reviewed

| Input | Relevant Content |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 127 | Defines seven Sprint 127 items and the 166-hour project-plan estimate. |
| `docs/planning/EPIC_11/SPRINT_127/PLAN.md` | Splits Sprint 127 into 14 days with no day above 12 hours. |
| `docs/planning/EPIC_11/SPRINT_126/RETROSPECTIVE.md` | Names the residual deferred debt and explicit non-claim register. |
| `docs/planning/EPIC_11/SPRINT_126/WORKING_NOTES.md` | Captures completed bounded fixtures, validation rules, duplicate fences, and future-owner handoffs. |
| Sprint 126 Day 1 artifact | Previous residual dedupe map and duplicate fences. |
| Sprint 126 Day 2-3 artifacts | Compatible/dependent-row/wide residual trust policy and completed dependent-row residual-only evidence. |
| Sprint 126 Day 4-5 artifacts | Nullspace/subspace expansion policy and completed rank-1/nullity-2 projector evidence. |
| Sprint 126 Day 6-7 artifacts | Threshold-family policy and completed scaled diagonal threshold evidence. |
| Sprint 126 Day 8-9 artifacts | SuiteSparse rank-deficient QR corpus gate and explicit deferral requirements. |
| Sprint 126 Day 10-11 artifacts | SuiteSparse minimum-norm corpus gate and explicit deferral requirements. |
| Sprint 126 Day 12-13 artifacts | Exact underdetermined and QR-vs-SVD gate plus completed 5 x 10 exact minimum-norm evidence. |
| Sprint 126 Day 14 artifact | Final validation, maintainer evidence, non-claim register, and Sprint 127 handoff boundaries. |
| Sprint 121-125 artifacts | Earlier QR/SVD/rank taxonomy, bounded external-reference lanes, nullspace policies, threshold policies, SuiteSparse gates, minimum-norm owner maps, QR-vs-SVD rules, and helper ownership decisions. |

## Day 1 Created Structure

| Path | Purpose |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_127/PLAN.md` | Day-by-day Sprint 127 execution plan. |
| `docs/planning/EPIC_11/SPRINT_127/WORKING_NOTES.md` | Running Sprint 127 notes, validation rules, day ownership, and scope boundaries. |
| `docs/planning/EPIC_11/SPRINT_127/artifacts/day1-deferred-dedupe-baseline.md` | Sprint intake, deferred dedupe map, duplicate fence, validation boundary, and completion check. |

## Sprint 127 Project-Plan Item Map

| Item | Item Name | Day Owner | Day 1 Interpretation |
| --- | --- | --- | --- |
| 1 | Sprint 126 Deferred Dedupe Map | Day 1 | Map Sprint 126 residual deferred debt against completed Sprint 121-126 evidence before any new lane is accepted. |
| 2 | Compatible and Wide Residual Semantics | Days 2-3 | Add or explicitly defer compatible zero-residual and wide residual-only QR evidence only if the lane adds distinct trust, has pinned output semantics, and cannot be misread as minimum-norm or nullspace evidence. |
| 3 | Nullspace and Subspace Expansion | Days 4-5 | Add or explicitly defer wide-shape, dependent-row, near-threshold, and SuiteSparse nullspace/subspace evidence only through projector or two-way projection metrics with pinned rank/nullity metadata. |
| 4 | Threshold Family Follow-Through | Days 6-7 | Add or explicitly defer perturbed duplicate-column, dependent-row, wide, default-threshold, and SuiteSparse threshold families with fixture-local expected ranks and non-global interpretation. |
| 5 | SuiteSparse Rank-Deficient QR Corpus Evidence | Days 8-9 | Gate SuiteSparse rank-deficient QR corpus evidence behind expected-rank metadata, threshold semantics, support tier, diagnostics, skip behavior, runtime budget, and validation requirements. |
| 6 | SuiteSparse and Optional-Large Minimum-Norm Gate | Days 10-11 | Add or explicitly defer SuiteSparse and optional-large QR/minimum-norm evidence only after extraction, shape, nnz, RHS, rank/nullity, residual/norm metrics, skip behavior, runtime, and support tier are pinned. |
| 7 | Minimum-Norm Cross-Check and Helper Claim Gate | Days 12-14 | Add or explicitly defer larger exact underdetermined lanes and QR-vs-SVD minimum-norm fixtures only as bounded cross-checks with explicit fixture keys, tolerances, non-oracle wording, and behavior-specific helper boundaries. |

## Residual Debt Dedupe Map

| Sprint 126 Deferred Work | Sprint 127 Owner | Completed Evidence to Reuse | Duplicate Fence | Promotion Gate |
| --- | --- | --- | --- | --- |
| Compatible zero-residual rank-deficient QR residual evidence | Days 2-3 | Deterministic compatible QR solve behavior, Sprint 125 residual-only trust gate, and Sprint 126 dependent-row residual-only evidence. | Do not relabel existing compatible solve residuals or dependent-row residual-only evidence as compatible zero-residual proof. | Prove the zero-residual lane adds diagnostics beyond deterministic compatible solves and cannot be misread as minimum-norm or nullspace evidence. |
| Wide residual-only QR evidence | Days 2-3 | Existing wide QR, underdetermined minimum-norm, and Q/economy policy artifacts. | Do not conflate wide residual evidence with underdetermined solution selection, minimum-norm optimality, Q/economy shape, or sparse-mode behavior. | Define output semantics, solution-selection policy, Q/economy boundaries, residual-only proof value, diagnostics, and non-claims. |
| Wide-shape nullspace/subspace evidence | Days 4-5 | Sprint 125-126 nullspace policies, duplicate-column projector evidence, and rank-1/nullity-2 projector evidence. | Do not treat minimum-norm or wide residual behavior as nullspace/subspace evidence. | Pin shape, rank, nullity, projection metric, sparse/economy output semantics, tolerance, and failure diagnostics. |
| Dependent-row nullspace/subspace evidence | Days 4-5 | `tf_qr_make_dependent_row_4x3`, dependent-row residual-only evidence, and projector metric policy. | Do not treat dependent-row residual-only evidence as subspace evidence. | Define projector or two-way projection metric, expected rank/nullity, threshold semantics, and diagnostics. |
| Near-threshold nullspace/subspace evidence | Days 4-5 | `qr_rank_threshold_diag4_family`, `qr_rank_threshold_diag4_scaled_family`, and threshold policy artifacts. | Do not mix threshold-rank evidence with raw basis equality or unique nullspace claims. | Pin threshold, expected rank/nullity, projection metric, tolerance, and non-global interpretation. |
| SuiteSparse nullspace/subspace evidence | Days 4-5 and 8-9 | SuiteSparse corpus gates, full-rank controls, and minimum-norm submatrix smoke. | Do not treat full-rank controls or one submatrix smoke as corpus-wide subspace evidence. | Define corpus availability, expected rank/nullity metadata, support tier, skip behavior, runtime, projection metric, and validation. |
| Perturbed duplicate-column threshold evidence | Days 6-7 | Duplicate-column rank/residual/nullspace evidence and scaled threshold-family evidence. | Do not reinterpret exact duplicate-column fixtures or scaled diagonal ladders as perturbation evidence. | Separate perturbation sizes from thresholds by at least two orders of magnitude and preserve default-threshold fences. |
| Dependent-row threshold families | Days 6-7 | Dependent-row residual fixture and threshold policy artifacts. | Do not derive a generic numerical-rank policy from one dependent-row fixture. | Define primary claim, expected ranks, threshold semantics, diagnostics, and failure interpretation. |
| Wide threshold families | Days 6-7 | Existing wide QR tests and underdetermined/minimum-norm policy artifacts. | Do not imply wide solve, minimum-norm, Q/economy, or sparse-mode behavior from threshold evidence. | Define shape, expected ranks, threshold semantics, output boundaries, diagnostics, and non-claims. |
| Default-threshold families | Days 6-7 | Existing explicit-threshold families and default-threshold non-claims. | Do not turn fixture-local default behavior into a global QR rank-threshold policy. | Define primary default-threshold claim, expected ranks, diagnostics, support tier, and failure interpretation. |
| SuiteSparse threshold families | Days 6-9 | SuiteSparse rank-deficient QR corpus gates and support-tier policy. | Do not claim broad SuiteSparse threshold or corpus behavior. | Define matrix, expected rank metadata, threshold semantics, optional-data behavior, runtime budget, and validation. |
| SuiteSparse rank-deficient QR corpus evidence | Days 8-9 | Day 8-9 Sprint 126 corpus gate, full-rank controls, checked-in Matrix Market inventory, and optional-large conventions. | Do not treat product QR diagnostics alone as independent expected-rank metadata. | Provide expected-rank metadata, threshold semantics, support tier, diagnostics, skip behavior, runtime budget, and validation. |
| Additional SuiteSparse minimum-norm evidence | Days 10-11 | Sprint 125 `west0067` 30 x 67 smoke and Sprint 126 SuiteSparse minimum-norm deferral. | Do not repeat the existing `west0067` first-30-row smoke as new evidence. | Pin extraction rule, shape, nnz, RHS, rank/nullity if claimed, residual/norm metrics, skip behavior, and support tier. |
| Optional-large SuiteSparse QR or minimum-norm evidence | Days 8-11 | Existing `SPARSE_TEST_LARGE=1` conventions and SuiteSparse optional-data gates. | Do not register optional-large work as default reviewed evidence without missing-data and runtime proof. | Record optional gate, missing-data skip behavior, runtime/platform expectations, support tier, diagnostics, and validation. |
| Additional QR-vs-SVD minimum-norm fixtures | Days 12-13 | Sprint 125 2 x 4 bounded QR-vs-SVD cross-check and Sprint 126 explicit deferral. | Do not treat SVD pseudoinverse as a global QR oracle or broad dense-library parity claim. | Define fixture keys, QR residual/norm metrics, SVD tolerance, diagnostic role, and non-oracle wording per fixture. |
| Larger exact underdetermined minimum-norm lanes | Days 12-13 | `qr_underdetermined_minnorm_2x4`, `qr_minnorm_5x10_exact_values`, and owner-local COLAMD exact/minnorm tests. | Do not repeat 2 x 4 or 5 x 10 exact evidence as larger-shape proof. | Choose non-duplicate shapes with closed-form expected values and explicit residual, value, and norm tolerances. |
| Generic QR/SVD helper movement | Days 12-14 and Sprint 128 handoff | Sprint 125 helper owner map, Sprint 126 exact/cross-check gates, and current owner-local assertion style. | Do not add generic helpers that hide behavior-specific tolerances or ownership. | Use behavior-specific helper names, call-site tolerances, focused QR solve/COLAMD/SVD validation, and full quality gate for C/header changes. |
| Raw Q-column, wide economy, sparse-mode Q/economy, and SuiteSparse Q/economy follow-through | Day 14 handoff to Sprint 128 | Sprint 124 projector policy, Sprint 125/126 corpus support rules, and Q/economy evidence policy. | Do not absorb Q/economy implementation into Sprint 127 residual/corpus/minimum-norm work. | Hand named output-shape semantics, projection metrics, support-tier rules, and claim boundaries to Sprint 128. |

## Completed Work Duplicate Fence

| Completed Work | Sprint 127 Handling |
| --- | --- |
| Sprint 121 SVD/QR/rank fixture taxonomy | Use as taxonomy input; do not redesign unless a Sprint 127 decision exposes a concrete gap. |
| Sprint 122 bounded SVD/QR/partial-SVD external oracle lane designs | Use as trust-boundary input; do not reopen earlier decisions. |
| Sprint 123 QR compatible, rank-deficient, underdetermined, and helper decisions | Use as completed evidence and ownership context; do not duplicate as new Sprint 127 evidence. |
| Sprint 124 Q-basis/economy policy and helper decisions | Use as Sprint 128 handoff context; do not pull raw Q/economy work into Sprint 127 except as prerequisite artifact. |
| `qr_rankdef_duplicate_5x4_rank_only` | Use as completed rank-only evidence; do not relabel as residual, nullspace, threshold, SuiteSparse, or minimum-norm proof. |
| `qr_rankdef_duplicate_5x4_residual_only` | Use as completed Sprint 125 residual-only evidence; do not duplicate as compatible, wide, dependent-row, or SuiteSparse residual evidence. |
| `qr_rankdef_dependent_row_4x3_residual_only` | Use as completed Sprint 126 dependent-row residual evidence; do not duplicate as compatible zero-residual, wide residual, nullspace, threshold, or SuiteSparse evidence. |
| `qr_rankdef_duplicate_5x4_nullspace_projector` | Use as completed one-dimensional projector evidence; do not relabel as wide, near-threshold, dependent-row, or SuiteSparse subspace evidence. |
| `qr_rank1_4x3_nullspace_projector` | Use as completed multi-dimensional projector evidence; do not relabel as wide-shape, dependent-row, near-threshold, SuiteSparse, sparse/economy, or raw-basis evidence. |
| `qr_rank_threshold_diag4_family` | Use as completed diagonal threshold evidence; do not relabel as scaled, perturbed, dependent-row, wide, default-threshold, or SuiteSparse threshold evidence. |
| `qr_rank_threshold_diag4_scaled_family` | Use as completed scaled diagonal threshold evidence; do not relabel as perturbed duplicate-column, dependent-row, wide, default-threshold, or SuiteSparse threshold evidence. |
| `qr_underdetermined_minnorm_2x4` | Use as completed exact underdetermined minimum-norm evidence; do not relabel as larger underdetermined, SuiteSparse, optional-large, or QR-vs-SVD proof. |
| `qr_minnorm_5x10_exact_values` | Use as completed exact 5 x 10 minimum-norm evidence; do not duplicate as larger-shape, SuiteSparse, optional-large, or QR-vs-SVD proof. |
| Sprint 125 COLAMD/fallback/rank-deficient/refinement/zero-row minimum-norm evidence | Use as owner-local baseline; do not create generic minimum-norm helper proof or broad QR minimum-norm claims. |
| Sprint 125 QR-vs-SVD bounded cross-check decision | Use as non-oracle rule; additional fixtures must preserve bounded diagnostic wording. |
| Sprint 125 `west0067` 30 x 67 minimum-norm submatrix smoke | Use as checked-in SuiteSparse minimum-norm baseline; do not claim optional-large, rank-deficient SuiteSparse, or corpus-wide behavior. |
| Sprint 126 SuiteSparse rank-deficient QR and minimum-norm deferrals | Use as metadata gates; do not weaken missing metadata into ambiguous claims. |
| `qr_economy_projector_5x3` | Use as Sprint 128 Q/economy input; not Sprint 127 residual corpus or minimum-norm scope. |
| `partial_svd_vector_residual_diag6_k2` | Use as Sprint 129 partial-SVD input; not Sprint 127 QR scope. |

## Validation Boundary

| Scenario | Required Validation |
| --- | --- |
| Documentation-only Day 1 work | `git diff --check` and focused markdown whitespace scan for Sprint 127 markdown files. |
| Future `.c` or `.h` changes | `make format && make lint && make test`. |
| Future Python helper changes | `python3 -m py_compile`, focused helper invocation, affected test executable, and protocol-output proof. |
| Future QR or minimum-norm test changes | Focused test executable proof plus full quality if C or header files changed. |
| Future Makefile/CMake/CTest membership changes | Source-list and CMake/CTest impact proof, including platform count notes where applicable. |
| Future SuiteSparse optional-corpus evidence | Optional-data present/missing behavior, skip-path proof, support-tier diagnostics, runtime/platform notes, and bounded claim note. |
| Future maintainer/public wording changes | Evidence-to-claim traceability, link/path hygiene, claim-boundary scan, and non-claim update. |

## Non-Claim Boundary

Sprint 127 Day 1 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad dense-library parity;
- broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, nullspace, minimum-norm, Q-basis, economy,
  sparse-mode, reorder, backend, corpus, optional-data, platform, or
  performance parity;
- compatible zero-residual, wide residual-only, wide-shape,
  dependent-row, near-threshold, SuiteSparse, optional-large, or
  default-threshold evidence beyond completed Sprint 126 fixtures and
  deferrals;
- raw nullspace basis equality, unique basis orientation, Q-sign,
  Q-orientation, or broad projection/subspace parity;
- global QR rank-threshold, default-threshold, or numerical-rank policy;
- broad QR minimum-norm, optional-large SuiteSparse, rank-deficient
  SuiteSparse, larger underdetermined, QR-vs-SVD-pseudoinverse, or
  SVD-pseudoinverse oracle parity;
- generic QR/SVD helper API or helper consolidation claim;
- package, ABI, platform, public API, CMake, Makefile, CI, CTest,
  performance, scalability, memory, or state-of-the-art claims.

## Downstream Handoff Needs

| Downstream Need | Day 1 Owner Boundary |
| --- | --- |
| Day 2 residual semantics policy | Must start from completed Sprint 125-126 residual-only evidence, then prove compatible zero-residual or wide residual-only candidates add distinct trust. |
| Day 4 nullspace/subspace expansion policy | Must not start implementation until rank/nullity metadata, projection metrics, sparse/economy semantics, and support tier are explicit. |
| Day 6 threshold-family follow-through policy | Must reuse completed diagonal and scaled threshold evidence while avoiding global threshold and default-threshold claims. |
| Day 8 SuiteSparse QR corpus gate | Must decide expected-rank metadata, threshold semantics, support tier, skip behavior, runtime budget, diagnostics, and validation before implementation. |
| Day 10 SuiteSparse and optional-large minimum-norm gate | Must decide extraction rule, residual, norm, rank, nullity, corpus metadata, optional-data behavior, runtime, and support tier before implementation. |
| Day 12 exact/cross-check/helper gate | Must preserve bounded non-oracle wording, closed-form expected values, explicit fixture keys, and behavior-specific helper ownership. |
| Day 14 validation and handoff | Must validate all accepted work and hand Q/economy/helper prerequisites to Sprint 128 without expanding claims. |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 127 project-plan item has a day-level owner. | Complete | See Sprint 127 project-plan item map. |
| Completed Sprint 121-126 evidence is not duplicated or silently reopened. | Complete | See completed work duplicate fence. |
| Dependency order is explicit before any new evidence is accepted. | Complete | See residual debt dedupe map and downstream handoff needs. |
| Sprint 127 working notes exist. | Complete | `WORKING_NOTES.md` created. |
| Sprint 127 artifact directory exists. | Complete | `artifacts/day1-deferred-dedupe-baseline.md` created. |
| Validation expectations are explicit. | Complete | See validation boundary. |
