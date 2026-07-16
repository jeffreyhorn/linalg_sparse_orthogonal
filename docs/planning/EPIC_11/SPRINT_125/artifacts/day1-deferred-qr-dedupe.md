# Sprint 125 Day 1 Deferred QR Dedupe Map

## Purpose

Day 1 establishes the Sprint 125 working structure and converts Sprint 124's
rank-deficient QR and minimum-norm carry-forward debt into dependency-ordered
proof owners. The goal is to make every residual-only, nullspace/subspace,
near-rank-deficient threshold, SuiteSparse corpus, and minimum-norm behavior
lane visible without duplicating completed Sprint 121-124 evidence.

## Inputs Reviewed

| Input | Relevant Content |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 125 | Defines seven Sprint 125 items and the 164-hour project-plan estimate. |
| `docs/planning/EPIC_11/SPRINT_125/PLAN.md` | Splits Sprint 125 into 14 days with no day above 12 hours. |
| `docs/planning/EPIC_11/SPRINT_124/RETROSPECTIVE.md` | Names the residual rank-deficient QR and minimum-norm deferred debt and the explicit non-claim register. |
| `docs/planning/EPIC_11/SPRINT_124/WORKING_NOTES.md` | Captures completed bounded fixtures, validation rules, and future-owner handoffs. |
| Sprint 124 Day 2 artifact | Rank-deficient QR rank policy, rank-threshold rules, nullspace separation, and rank/minimum-norm boundaries. |
| Sprint 124 Day 3 artifact | Completed `qr_rankdef_duplicate_5x4_rank_only` rank-only fixture and deferred residual/nullspace/threshold/SuiteSparse work. |
| Sprint 124 Day 4 artifact | Minimum-norm behavior contract and owner separation across QR solve, COLAMD, fallback, rank-deficient, refinement, QR-vs-SVD-pseudoinverse, and SuiteSparse behavior. |
| Sprint 124 Day 5 artifact | Completed `qr_underdetermined_minnorm_2x4` exact fixture and deferred broader minimum-norm behavior evidence. |
| Sprint 124 Day 13-14 artifacts | Validation gate, maintainer evidence, solver-selection no-update rationale, residual queue, and handoff boundaries. |
| Sprint 121-123 artifacts | Earlier QR/SVD/rank taxonomy, bounded external-reference lanes, helper ownership decisions, and duplicate fences. |

## Day 1 Created Structure

| Path | Purpose |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_125/PLAN.md` | Day-by-day Sprint 125 execution plan. |
| `docs/planning/EPIC_11/SPRINT_125/WORKING_NOTES.md` | Running Sprint 125 notes, validation rules, day ownership, and scope boundaries. |
| `docs/planning/EPIC_11/SPRINT_125/artifacts/day1-deferred-qr-dedupe.md` | Sprint intake, deferred QR/minimum-norm dedupe map, duplicate fence, and completion check. |

## Sprint 125 Project-Plan Item Map

| Item | Item Name | Day Owner | Day 1 Interpretation |
| --- | --- | --- | --- |
| 1 | Deferred QR Dedupe Map | Day 1 | Map Sprint 124 deferred rank-deficient and minimum-norm work against completed Sprint 121-124 evidence before any new lane is accepted. |
| 2 | Rank-Deficient Residual Evidence | Days 2-3 | Add or explicitly defer residual-only rank-deficient QR evidence only if it adds trust beyond deterministic tests and does not imply nullspace or minimum-norm behavior. |
| 3 | Nullspace and Subspace Policy | Days 4-5 | Define sign, ordering, nullity, projection/subspace metrics, and fixture-local tolerance rules before accepting external nullspace/subspace evidence. |
| 4 | Near-Rank-Deficient Threshold Evidence | Days 6-7 | Define threshold families, expected ranks, stability policy, and non-global interpretation before accepting threshold evidence. |
| 5 | SuiteSparse Rank-Deficient QR Evidence | Days 8-9 | Define optional corpus, platform skip behavior, support tier, diagnostics, and claim boundaries before accepting SuiteSparse evidence. |
| 6 | Minimum-Norm Behavior Evidence | Days 10-12 | Add or explicitly defer COLAMD, fallback, rank-deficient, refinement, QR-vs-SVD-pseudoinverse, and SuiteSparse minimum-norm evidence under behavior-specific owners. |
| 7 | Validation and Claim Gate | Days 13-14 | Validate accepted work, refresh evidence tables, preserve broad non-claims, and hand remaining dependencies to Sprint 126+. |

## Deferred Debt Dedupe Map

| Sprint 124 Deferred Work | Sprint 125 Owner | Completed Evidence to Reuse | Duplicate Fence | Promotion Gate |
| --- | --- | --- | --- | --- |
| Rank-deficient QR residual-only evidence | Days 2-3 | Deterministic rank-deficient solve/residual tests and `qr_rankdef_duplicate_5x4_rank_only` rank-only external fixture. | Do not relabel rank-only equality or deterministic solve residuals as new residual-only external evidence. | Show residual-only evidence adds trust beyond deterministic checks and explicitly excludes nullspace and minimum-norm claims. |
| Rank-deficient QR nullspace/subspace evidence | Days 4-5 | Deterministic QR nullspace tests plus Sprint 124 rank policy and rank-only fixture. | Do not compare raw nullspace vectors or imply unique basis orientation from rank evidence. | Define sign, ordering, nullity, projection/subspace metric, fixture-local tolerance, and diagnostics. |
| Near-rank-deficient QR threshold evidence | Days 6-7 | Deterministic diagonal threshold fixture and Sprint 124 rank-threshold policy. | Do not create a global QR numerical-rank threshold or duplicate exact structural rank-only evidence. | Define threshold family, expected ranks, scale/stability policy, and fixture-local interpretation. |
| SuiteSparse rank-deficient QR evidence | Days 8-9 | Existing SuiteSparse conventions and Sprint 124 optional-corpus non-claims. | Do not treat missing optional data, platform skips, or bounded corpus smoke as broad SuiteSparse support. | Define corpus availability, platform skip, support tier, diagnostics, validation, and claim boundaries. |
| COLAMD/reordered minimum-norm evidence | Days 10-11 | Deterministic `test_colamd.c` minimum-norm/COLAMD coverage and `qr_underdetermined_minnorm_2x4` exact fixture. | Do not hide ordering behavior behind a generic minimum-norm helper or reuse the 2x4 exact fixture as COLAMD proof. | Define ordering options, expected residual/norm behavior, diagnostics, and non-superiority wording. |
| Fallback minimum-norm evidence | Days 10-11 | Deterministic fallback coverage in `test_colamd.c`. | Do not conflate ordinary QR fallback with underdetermined minimum-norm optimality. | Define overdetermined/square fallback contract, expected result, and residual comparison. |
| Rank-deficient minimum-norm evidence | Days 10-11 | Sprint 124 rank policy, deterministic rank-deficient minimum-norm coverage, and exact 2x4 minimum-norm fixture. | Do not claim nullspace, pseudoinverse, or global rank policy from one minimum-norm check. | Combine rank threshold, consistency, residual, norm, and nullspace boundaries under a named owner. |
| Refinement minimum-norm evidence | Days 10-11 | Deterministic refinement coverage in `test_colamd.c`. | Do not imply convergence rate, superiority, or broad refinement behavior. | Define before/after residual expectations, iteration budget, and failure interpretation. |
| QR-vs-SVD-pseudoinverse evidence | Day 12 | Deterministic QR-vs-pinv check and SVD pseudoinverse tests. | Do not treat SVD pseudoinverse as a global QR oracle or dense-library parity claim. | Define whether the comparison is oracle, bounded cross-check, diagnostic, or deferral. |
| SuiteSparse minimum-norm evidence | Day 12 | Existing optional SuiteSparse submatrix smoke coverage and optional-data conventions. | Do not claim SuiteSparse-wide minimum-norm support or equal platform behavior. | Define corpus availability, platform skip, support tier, diagnostics, and bounded claim language. |

## Completed Work Duplicate Fence

| Completed Work | Sprint 125 Handling |
| --- | --- |
| Sprint 121 SVD/QR/rank fixture taxonomy | Use as taxonomy input; do not redesign unless a Sprint 125 decision exposes a concrete gap. |
| Sprint 122 bounded SVD/QR/partial-SVD external oracle lane designs | Use as trust-boundary input; do not reopen earlier decisions. |
| Sprint 123 QR compatible and incompatible external lanes | Use as completed QR solve evidence; do not duplicate as rank-deficient residual evidence. |
| `qr_rankdef_duplicate_5x4_rank_only` | Use as completed rank-only external evidence; do not relabel as residual, nullspace, threshold, SuiteSparse, or minimum-norm proof. |
| `qr_underdetermined_minnorm_2x4` | Use as completed exact underdetermined minimum-norm evidence; do not relabel as COLAMD, fallback, rank-deficient, refinement, QR-vs-SVD, or SuiteSparse proof. |
| `qr_economy_projector_5x3` | Use as Sprint 126 Q/economy input; not Sprint 125 rank-deficient residual or minimum-norm scope. |
| `partial_svd_vector_residual_diag6_k2` | Use as Sprint 127 partial-SVD input; not Sprint 125 QR scope. |
| Sprint 124 rank-deficient QR policy design | Use as policy input; Sprint 125 should only extend it where residual-only, nullspace, threshold, or SuiteSparse evidence requires more detail. |
| Sprint 124 QR minimum-norm behavior contract | Use as owner-boundary input; Sprint 125 should extend behavior-specific evidence, not create generic helpers. |
| Sprint 124 maintainer evidence and solver-selection claim gate | Use as current public-claim baseline; public wording changes require Day 13 evidence-to-claim traceability. |
| Sprint 124 final validation package and closeout evidence | Use as baseline validation and residual source; do not rerun full quality unless Sprint 125 code changes require it. |

## Validation Boundary

| Scenario | Required Validation |
| --- | --- |
| Documentation-only Day 1 work | `git diff --check` and focused whitespace scan for Sprint 125 markdown files. |
| Future `.c` or `.h` changes | `make format && make lint && make test`. |
| Future Python helper changes | `python3 -m py_compile`, focused helper invocation, affected test executable, and protocol-output proof. |
| Future QR or minimum-norm test changes | Focused test executable proof plus full quality if C or header files changed. |
| Future Makefile/CMake/CTest membership changes | Source-list and CMake/CTest impact proof, including platform count notes where applicable. |
| Future SuiteSparse optional-corpus evidence | Optional-data present/missing behavior, skip-path proof, support-tier diagnostics, and bounded claim note. |
| Future maintainer/public wording changes | Evidence-to-claim traceability, link/path hygiene, claim-boundary scan, and non-claim update. |

## Non-Claim Boundary

Sprint 125 Day 1 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad dense-library parity;
- broad QR factorization, QR solve, rank-deficient solve, nullspace,
  minimum-norm, Q-basis, economy, sparse-mode, reorder, backend, corpus, or
  performance parity;
- residual-only rank-deficient QR evidence beyond currently completed internal
  and rank-only fixtures;
- raw nullspace basis equality, unique basis orientation, Q-sign, Q-orientation,
  or subspace external parity;
- global near-rank-deficient threshold policy;
- broad QR minimum-norm, COLAMD, fallback, rank-deficient, refinement,
  QR-vs-SVD-pseudoinverse, or SuiteSparse minimum-norm parity;
- package, ABI, platform, public API, CMake, Makefile, CI, CTest, performance,
  scalability, memory, or state-of-the-art claims.

## Downstream Handoff Needs

| Downstream Need | Day 1 Owner Boundary |
| --- | --- |
| Day 2 residual-only trust gate | Must start from completed deterministic and rank-only evidence, then prove any residual-only lane adds distinct trust. |
| Day 4 nullspace/subspace policy | Must not start implementation until sign, ordering, nullity, projection/subspace metric, tolerance, and diagnostics are explicit. |
| Day 6 threshold family design | Must reuse existing deterministic threshold fixture while avoiding global threshold claims. |
| Day 8 SuiteSparse corpus policy | Must decide optional corpus and skip semantics before selecting a matrix or support-tier wording. |
| Day 10 minimum-norm owner map | Must preserve behavior-specific owners rather than generic minimum-norm helper movement. |
| Day 13 validation and claim gate | Must validate all accepted changes and preserve non-claims unless evidence supports bounded wording. |
| Sprint 126 Q-basis/economy work | Receives Q/economy-related debt; Sprint 125 should not absorb raw Q-column or economy-mode work. |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 125 project-plan item has a day-level owner. | Complete | See Sprint 125 project-plan item map. |
| Completed Sprint 121-124 evidence is not duplicated or silently reopened. | Complete | See completed work duplicate fence. |
| Dependency order is explicit before any new evidence is accepted. | Complete | See deferred debt dedupe map and downstream handoff needs. |
| Sprint 125 working notes exist. | Complete | `WORKING_NOTES.md` created. |
| Sprint 125 artifact directory exists. | Complete | `artifacts/day1-deferred-qr-dedupe.md` created. |
| Validation expectations are explicit. | Complete | See validation boundary. |
