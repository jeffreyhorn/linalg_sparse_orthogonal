# Sprint 128 Day 1 Residual Dedupe Baseline

## Purpose

Day 1 establishes the Sprint 128 working structure and converts Sprint 127's
compatible/wide residual, wide nullspace, threshold-family, SuiteSparse corpus,
optional-large, minimum-norm, QR-vs-SVD, and helper carry-forward debt into
dependency-ordered proof owners. The goal is to make each residual lane visible
without duplicating completed Sprint 121-127 evidence.

## Inputs Reviewed

| Input | Relevant Content |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 128 | Defines seven Sprint 128 items and the 166-hour project-plan estimate. |
| `docs/planning/EPIC_11/SPRINT_128/PLAN.md` | Splits Sprint 128 into 14 days with no day above 12 hours. |
| `docs/planning/EPIC_11/SPRINT_127/RETROSPECTIVE.md` | Names the residual deferred debt and explicit non-claim register. |
| `docs/planning/EPIC_11/SPRINT_127/WORKING_NOTES.md` | Captures completed bounded fixtures, validation rules, duplicate fences, and future-owner handoffs. |
| Sprint 127 Day 1 artifact | Previous residual dedupe map and duplicate fences. |
| Sprint 127 Day 2-3 artifacts | Compatible zero-residual and wide residual-only trust policy plus explicit deferrals. |
| Sprint 127 Day 4-5 artifacts | Nullspace/subspace expansion policy and completed dependent-row projector evidence. |
| Sprint 127 Day 6-7 artifacts | Threshold-family policy and completed perturbed duplicate-column threshold evidence. |
| Sprint 127 Day 8-9 artifacts | SuiteSparse rank-deficient QR corpus gate and explicit deferral requirements. |
| Sprint 127 Day 10-11 artifacts | SuiteSparse and optional-large minimum-norm gate and explicit deferral requirements. |
| Sprint 127 Day 12-13 artifacts | Exact minimum-norm and QR-vs-SVD/helper gate plus completed 3 x 6 exact minimum-norm evidence. |
| Sprint 127 Day 14 artifact | Final validation, maintainer evidence, non-claim register, and Sprint 128 handoff boundaries. |
| Sprint 121-126 artifacts | Earlier QR/SVD/rank taxonomy, bounded external-reference lanes, nullspace policies, threshold policies, SuiteSparse gates, minimum-norm owner maps, QR-vs-SVD rules, and helper ownership decisions. |

## Day 1 Created Structure

| Path | Purpose |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_128/PLAN.md` | Day-by-day Sprint 128 execution plan. |
| `docs/planning/EPIC_11/SPRINT_128/WORKING_NOTES.md` | Running Sprint 128 notes, validation rules, day ownership, and scope boundaries. |
| `docs/planning/EPIC_11/SPRINT_128/artifacts/day1-residual-dedupe-baseline.md` | Sprint intake, residual dedupe map, duplicate fence, validation boundary, and completion check. |

## Sprint 128 Project-Plan Item Map

| Item | Item Name | Day Owner | Day 1 Interpretation |
| --- | --- | --- | --- |
| 1 | Sprint 127 Residual Dedupe and Dependency Map | Day 1 | Map Sprint 127 carry-forward work against completed Sprint 121-127 evidence before any new lane is accepted. |
| 2 | Compatible and Wide Residual Evidence Gate | Days 2-3 | Add or explicitly defer compatible zero-residual and wide residual-only QR evidence only if the lane adds distinct trust, has pinned output semantics, and cannot be misread as nullspace or minimum-norm evidence. |
| 3 | Wide and Near-Threshold Nullspace/Subspace Gate | Days 4-5 | Add or explicitly defer wide-shape, near-threshold, and SuiteSparse nullspace/subspace evidence only through projector or two-way projection metrics with pinned rank/nullity metadata. |
| 4 | Remaining Threshold-Family Gate | Days 6-7 | Add or explicitly defer dependent-row, wide, default-threshold, and SuiteSparse threshold families with fixture-local expected ranks, diagnostics, support tiers, and non-global interpretation. |
| 5 | SuiteSparse Rank-Deficient QR Corpus Gate | Days 8-9 | Gate SuiteSparse rank-deficient QR corpus evidence behind independent expected-rank metadata, threshold semantics, support tier, diagnostics, skip behavior, runtime budget, and validation requirements. |
| 6 | SuiteSparse and Optional-Large Minimum-Norm Gate | Days 10-11 | Add or explicitly defer SuiteSparse and optional-large QR/minimum-norm evidence only after extraction, shape, nnz, RHS, rank/nullity, residual/norm metrics, skip behavior, runtime, platform, and support tier are pinned. |
| 7 | Exact Minimum-Norm and Helper Movement Gate | Days 12-14 | Add or explicitly defer additional QR-vs-SVD minimum-norm fixtures, larger exact underdetermined lanes, and helper movement only as bounded evidence with explicit fixture keys, tolerances, non-oracle wording, and behavior-specific helper boundaries. |

## Residual Debt Dedupe Map

| Sprint 127 Deferred Work | Sprint 128 Owner | Completed Evidence to Reuse | Duplicate Fence | Promotion Gate |
| --- | --- | --- | --- | --- |
| Compatible zero-residual rank-deficient QR residual evidence | Days 2-3 | Deterministic compatible QR solve behavior, Sprint 125 duplicate-column residual-only trust gate, Sprint 126 dependent-row residual-only evidence, and Sprint 127 explicit deferral. | Do not relabel existing compatible solve residuals, duplicate-column residual-only evidence, or dependent-row residual-only evidence as compatible zero-residual proof. | Prove the zero-residual lane adds diagnostics beyond deterministic compatible solves and cannot be misread as minimum-norm, solution-selection, or nullspace evidence. |
| Wide residual-only QR evidence | Days 2-3 | Existing wide QR, underdetermined minimum-norm, Q/economy policy artifacts, and Sprint 127 wide residual deferral. | Do not conflate wide residual evidence with underdetermined solution selection, minimum-norm optimality, Q/economy shape, raw basis, or sparse-mode behavior. | Define output semantics, solution-selection policy, Q/economy boundaries, residual-only proof value, diagnostics, and non-claims. |
| Wide-shape nullspace/subspace evidence | Days 4-5 | Sprint 125-127 nullspace policies, duplicate-column projector evidence, rank-1/nullity-2 projector evidence, and dependent-row projector evidence. | Do not treat minimum-norm, wide residual, or Q/economy behavior as nullspace/subspace evidence. | Pin shape, rank, nullity, projection metric, sparse/economy output semantics, tolerance, and failure diagnostics. |
| Near-threshold nullspace/subspace evidence | Days 4-5 | `qr_rank_threshold_diag4_family`, `qr_rank_threshold_diag4_scaled_family`, `qr_rank_threshold_duplicate_5x4_perturbed_family`, and projector metric policy artifacts. | Do not mix threshold-rank evidence with raw basis equality, unique nullspace claims, or global numerical rank policy. | Pin threshold, expected rank/nullity, projection metric, tolerance, diagnostic role, and non-global interpretation. |
| SuiteSparse nullspace/subspace evidence | Days 4-5 and 8-9 | SuiteSparse corpus gates, full-rank controls, checked-in Matrix Market inventory, and minimum-norm submatrix smoke. | Do not treat full-rank controls, product QR diagnostics, or one submatrix smoke as corpus-wide subspace evidence. | Define corpus availability, independent expected rank/nullity metadata, support tier, skip behavior, runtime, projection metric, diagnostics, and validation. |
| Dependent-row threshold families | Days 6-7 | Dependent-row residual fixture, dependent-row projector fixture, and threshold policy artifacts. | Do not derive a generic numerical-rank or default-threshold policy from one dependent-row fixture. | Define primary claim, expected ranks, threshold semantics, diagnostics, support tier, and failure interpretation. |
| Wide threshold families | Days 6-7 | Existing wide QR tests and underdetermined/minimum-norm policy artifacts. | Do not imply wide solve, minimum-norm, Q/economy, raw basis, or sparse-mode behavior from threshold evidence. | Define shape, expected ranks, threshold semantics, output boundaries, diagnostics, and non-claims. |
| Default-threshold families | Days 6-7 | Existing explicit-threshold families and default-threshold non-claims. | Do not turn fixture-local default behavior into a global QR rank-threshold policy. | Define primary default-threshold claim, expected ranks, diagnostics, support tier, failure interpretation, and public non-claim wording. |
| SuiteSparse threshold families | Days 6-9 | SuiteSparse rank-deficient QR corpus gates and support-tier policy. | Do not claim broad SuiteSparse threshold, corpus, optional-data, platform, or runtime behavior. | Define matrix, independent expected-rank metadata, threshold semantics, optional-data behavior, runtime budget, support tier, and validation. |
| SuiteSparse rank-deficient QR corpus evidence | Days 8-9 | Sprint 126-127 corpus gates, full-rank controls, checked-in Matrix Market inventory, and optional-large conventions. | Do not treat product QR diagnostics alone as independent expected-rank metadata. | Provide expected-rank metadata, threshold semantics, support tier, diagnostics, skip behavior, runtime budget, and validation. |
| Additional SuiteSparse minimum-norm evidence | Days 10-11 | Sprint 125 `west0067` 30 x 67 smoke and Sprint 126-127 SuiteSparse minimum-norm deferrals. | Do not repeat the existing `west0067` first-30-row smoke as new evidence. | Pin extraction rule, shape, nnz, RHS, rank/nullity if claimed, residual/norm metrics, skip behavior, runtime, and support tier. |
| Optional-large SuiteSparse QR or minimum-norm evidence | Days 8-11 | Existing `SPARSE_TEST_LARGE=1` conventions and SuiteSparse optional-data gates. | Do not register optional-large work as default reviewed evidence without missing-data, runtime, and platform proof. | Record optional gate, missing-data skip behavior, runtime/platform expectations, support tier, diagnostics, and validation. |
| Additional QR-vs-SVD minimum-norm fixtures | Days 12-13 | Sprint 125 2 x 4 bounded QR-vs-SVD cross-check and Sprint 127 explicit deferral. | Do not treat SVD pseudoinverse as a global QR oracle or broad dense-library parity claim. | Define fixture keys, QR residual/norm metrics, SVD tolerance, diagnostic role, and non-oracle wording per fixture. |
| Larger exact underdetermined minimum-norm lanes | Days 12-13 | `qr_underdetermined_minnorm_2x4`, `qr_minnorm_5x10_exact_values`, `qr_minnorm_3x6_exact_values`, and owner-local COLAMD exact/minimum-norm tests. | Do not repeat 2 x 4, 3 x 6, or 5 x 10 exact evidence as larger-shape proof. | Choose non-duplicate shapes with closed-form expected values and explicit residual, value, and norm tolerances. |
| Generic QR/SVD helper movement | Days 12-14 and Sprint 129 handoff | Sprint 125 helper owner map, Sprint 126-127 exact/cross-check gates, and current owner-local assertion style. | Do not add generic helpers that hide behavior-specific tolerances or ownership. | Use behavior-specific helper names, call-site tolerances, focused QR solve/COLAMD/SVD validation, and full quality gate for C/header changes. |
| Raw Q-column, wide economy, sparse-mode Q/economy, and SuiteSparse Q/economy follow-through | Day 14 handoff to Sprint 129 | Sprint 124 projector policy, Sprint 125-127 corpus support rules, Q/economy evidence policy, and Sprint 127 handoff. | Do not absorb Q/economy implementation into Sprint 128 residual/corpus/minimum-norm work. | Hand named output-shape semantics, projection metrics, support-tier rules, and claim boundaries to Sprint 129. |

## Completed Work Duplicate Fence

| Completed Work | Sprint 128 Handling |
| --- | --- |
| Sprint 121 SVD/QR/rank fixture taxonomy | Use as taxonomy input; do not redesign unless a Sprint 128 decision exposes a concrete gap. |
| Sprint 122 bounded SVD/QR/partial-SVD external oracle lane designs | Use as trust-boundary input; do not reopen earlier decisions. |
| Sprint 123 QR compatible, rank-deficient, underdetermined, and helper decisions | Use as completed evidence and ownership context; do not duplicate as new Sprint 128 evidence. |
| Sprint 124 Q-basis/economy policy and helper decisions | Use as Sprint 129 handoff context; do not pull raw Q/economy work into Sprint 128 except as prerequisite artifact. |
| `qr_rankdef_duplicate_5x4_rank_only` | Use as completed rank-only evidence; do not relabel as residual, nullspace, threshold, SuiteSparse, or minimum-norm proof. |
| `qr_rankdef_duplicate_5x4_residual_only` | Use as completed Sprint 125 residual-only evidence; do not duplicate as compatible, wide, dependent-row, or SuiteSparse residual evidence. |
| `qr_rankdef_dependent_row_4x3_residual_only` | Use as completed Sprint 126 dependent-row residual evidence; do not duplicate as compatible zero-residual, wide residual, nullspace, threshold, or SuiteSparse evidence. |
| `qr_rankdef_duplicate_5x4_nullspace_projector` | Use as completed one-dimensional projector evidence; do not relabel as wide, near-threshold, dependent-row, or SuiteSparse subspace evidence. |
| `qr_rank1_4x3_nullspace_projector` | Use as completed multi-dimensional projector evidence; do not relabel as wide-shape, near-threshold, SuiteSparse, sparse/economy, or raw-basis evidence. |
| `qr_rankdef_dependent_row_4x3_nullspace_projector` | Use as completed Sprint 127 dependent-row projector evidence; do not relabel as wide-shape, near-threshold, SuiteSparse, or raw-basis evidence. |
| `qr_rank_threshold_diag4_family` | Use as completed diagonal threshold evidence; do not relabel as scaled, perturbed, dependent-row, wide, default-threshold, or SuiteSparse threshold evidence. |
| `qr_rank_threshold_diag4_scaled_family` | Use as completed scaled diagonal threshold evidence; do not relabel as perturbed duplicate-column, dependent-row, wide, default-threshold, or SuiteSparse threshold evidence. |
| `qr_rank_threshold_duplicate_5x4_perturbed_family` | Use as completed Sprint 127 perturbed duplicate-column threshold evidence; do not relabel as dependent-row, wide, default-threshold, SuiteSparse, or nullspace evidence. |
| `qr_underdetermined_minnorm_2x4` | Use as completed exact underdetermined minimum-norm evidence; do not relabel as larger underdetermined, SuiteSparse, optional-large, or QR-vs-SVD proof. |
| `qr_minnorm_5x10_exact_values` | Use as completed exact 5 x 10 minimum-norm evidence; do not duplicate as larger-shape, SuiteSparse, optional-large, or QR-vs-SVD proof. |
| `qr_minnorm_3x6_exact_values` | Use as completed Sprint 127 exact 3 x 6 minimum-norm evidence; do not duplicate as larger-shape, SuiteSparse, optional-large, or QR-vs-SVD proof. |
| Sprint 125 COLAMD/fallback/rank-deficient/refinement/zero-row minimum-norm evidence | Use as owner-local baseline; do not create generic minimum-norm helper proof or broad QR minimum-norm claims. |
| Sprint 125 QR-vs-SVD bounded cross-check decision | Use as non-oracle rule; additional fixtures must preserve bounded diagnostic wording. |
| Sprint 125 `west0067` 30 x 67 minimum-norm submatrix smoke | Use as checked-in SuiteSparse minimum-norm baseline; do not claim optional-large, rank-deficient SuiteSparse, or corpus-wide behavior. |
| Sprint 126-127 SuiteSparse rank-deficient QR and minimum-norm deferrals | Use as metadata gates; do not weaken missing metadata into ambiguous claims. |
| `qr_economy_projector_5x3` | Use as Sprint 129 Q/economy input; not Sprint 128 residual corpus or minimum-norm scope. |
| `partial_svd_vector_residual_diag6_k2` | Use as Sprint 130 partial-SVD input; not Sprint 128 QR residual/corpus scope. |

## Validation Boundary

| Scenario | Required Validation |
| --- | --- |
| Documentation-only Day 1 work | `git diff --check` and focused markdown whitespace scan for Sprint 128 markdown files. |
| Future `.c` or `.h` changes | `make format && make lint && make test`. |
| Future Python helper changes | `python3 -m py_compile`, focused helper invocation, affected test executable, and protocol-output proof. |
| Future QR or minimum-norm test changes | Focused test executable proof plus full quality if C or header files changed. |
| Future Makefile/CMake/CTest membership changes | Source-list and CMake/CTest impact proof, including platform count notes where applicable. |
| Future SuiteSparse optional-corpus evidence | Optional-data present/missing behavior, skip-path proof, support-tier diagnostics, runtime/platform notes, and bounded claim note. |
| Future maintainer/public wording changes | Evidence-to-claim traceability, link/path hygiene, claim-boundary scan, and non-claim update. |

## Non-Claim Boundary

Sprint 128 Day 1 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad dense-library parity;
- broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, nullspace, minimum-norm, Q-basis, economy,
  sparse-mode, reorder, backend, corpus, optional-data, platform, or
  performance parity;
- compatible zero-residual, wide residual-only, wide-shape, near-threshold,
  SuiteSparse, optional-large, default-threshold, or additional threshold
  evidence beyond completed Sprint 127 fixtures and deferrals;
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
| Day 2 residual semantics policy | Must start from completed Sprint 125-127 residual-only evidence, then prove compatible zero-residual or wide residual-only candidates add distinct trust. |
| Day 4 wide/near-threshold subspace policy | Must not start implementation until rank/nullity metadata, projection metrics, sparse/economy semantics, support tier, and threshold semantics are explicit. |
| Day 6 remaining threshold-family policy | Must reuse completed diagonal, scaled, and perturbed duplicate-column threshold evidence while avoiding global threshold and default-threshold claims. |
| Day 8 SuiteSparse QR corpus gate | Must decide independent expected-rank metadata, threshold semantics, support tier, skip behavior, runtime budget, diagnostics, and validation before implementation. |
| Day 10 SuiteSparse and optional-large minimum-norm policy | Must decide extraction rule, residual, norm, rank, nullity, corpus metadata, optional-data behavior, runtime, platform, and support tier before implementation. |
| Day 12 exact/cross-check/helper gate | Must preserve bounded non-oracle wording, closed-form expected values, explicit fixture keys, and behavior-specific helper ownership. |
| Day 14 validation and handoff | Must validate all accepted work and hand Q/economy/helper prerequisites to Sprint 129 without expanding claims. |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 128 project-plan item has a day-level owner. | Complete | See Sprint 128 project-plan item map. |
| Completed Sprint 121-127 evidence is not duplicated or silently reopened. | Complete | See completed work duplicate fence. |
| Dependency order is explicit before any new evidence is accepted. | Complete | See residual debt dedupe map and downstream handoff needs. |
| Sprint 128 working notes exist. | Complete | `WORKING_NOTES.md` created. |
| Sprint 128 artifact directory exists. | Complete | `artifacts/day1-residual-dedupe-baseline.md` created. |
| Validation expectations are explicit. | Complete | See validation boundary. |
