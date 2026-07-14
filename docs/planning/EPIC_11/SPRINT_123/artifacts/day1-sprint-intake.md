# Sprint 123 Day 1 Sprint Intake and Residual Proof Map

## Purpose

Day 1 establishes the Sprint 123 working structure and converts Sprint 122's
residual deferred debt into explicit proof owners. The goal is to make every
SVD, QR, partial-SVD, helper, maintainer-evidence, and solver-selection
follow-through item owned without duplicating the bounded oracle and
helper-boundary work already completed in Sprint 122.

## Inputs Reviewed

| Input | Relevant Content |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 123 | Defines seven Sprint 123 items for residual SVD fixtures, QR behavior evidence, partial-SVD semantics, helper migration, maintainer evidence tables, solver-selection claim gates, and validation. |
| `docs/planning/EPIC_11/SPRINT_123/PLAN.md` | Splits Sprint 123 into 14 days with no day above 12 hours and a 166-hour total. |
| `docs/planning/EPIC_11/SPRINT_122/RETROSPECTIVE.md` | Names Sprint 122 residual deferred debt, non-claims, and completed-work fences. |
| `docs/planning/EPIC_11/SPRINT_122/WORKING_NOTES.md` | Provides day-by-day context for bounded SVD, QR, partial-SVD, helper, and claim-gate decisions. |
| Sprint 122 Day 3-4 artifacts | SVD external fixture inventory and completed `svd_rankdef_duplicate_5x4` decision. |
| Sprint 122 Day 5-6 artifacts | QR external lane requirements and completed `qr_overdetermined_incompatible_4x2` decision. |
| Sprint 122 Day 7-8 artifacts | Partial-SVD external semantics and completed `partial_svd_diag6_k2` top-k decision. |
| Sprint 122 Day 9-10 artifacts | Minimum-norm and Bidiagonal/Golub-Kahan helper ownership boundaries. |
| Sprint 122 Day 11-12 artifacts | Solver-selection evidence inventory and no public wording expansion rationale. |
| Sprint 122 Day 13-14 artifacts | Validation package, non-claim register, residual queue, and closeout evidence. |
| Sprint 121 artifacts | Fixture taxonomy, SVD/QR audits, helper extraction boundaries, and deterministic fixture expansion baseline. |

## Day 1 Created Structure

| Path | Purpose |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_123/PLAN.md` | Day-by-day Sprint 123 execution plan. |
| `docs/planning/EPIC_11/SPRINT_123/WORKING_NOTES.md` | Running Sprint 123 notes, validation rules, day ownership, and scope boundaries. |
| `docs/planning/EPIC_11/SPRINT_123/artifacts/day1-sprint-intake.md` | Sprint intake, residual proof map, duplicate fence, and completion check. |

## Sprint 123 Project-Plan Item Map

| Item | Item Name | Day Owner | Day 1 Interpretation |
| --- | --- | --- | --- |
| 1 | Broader SVD External Fixture Matrix | Days 2-4 | Decide whether to broaden SVD external fixtures beyond Sprint 122's bounded lanes, then implement a bounded batch or publish deferral gates. |
| 2 | QR External Behavior Evidence Batch | Days 5-8 | Decide and implement or defer QR compatible, rank-deficient, underdetermined/minimum-norm, and Q/economy evidence with behavior-specific semantics. |
| 3 | Partial-SVD External Semantics Batch | Days 9-10 | Expand or defer partial-SVD evidence beyond top-k values while keeping vector, subspace, convergence, repeated-spectrum, and rank-deficient semantics explicit. |
| 4 | Minimum-Norm Helper Migration Decision | Day 11 | Revisit helper migration only if QR/COLAMD/SVD-pseudoinverse/refinement/fallback/SuiteSparse ownership remains visible. |
| 5 | Bidiagonal/Golub-Kahan Helper Extraction Decision | Day 12 | Extract or defer only through a dedicated helper owner that preserves transpose, Householder, `U`/`V`, and bidiagonal QR semantics. |
| 6 | Maintainer Evidence Table Refresh | Day 13 | Refresh maintainer evidence with Sprint 122 and Sprint 123 oracle ownership, trust boundaries, validation commands, and non-claims. |
| 7 | Solver-Selection Claim Refresh Gate | Day 14 | Refresh public solver-selection wording only if broader evidence supports it; otherwise publish a no-update rationale and residual claim gates. |

## Residual Proof-Owner Map

| Sprint 122 Residual | Sprint 123 Owner | Dependency / Promotion Gate | Duplicate Fence |
| --- | --- | --- | --- |
| Broader SVD external fixture matrix | Days 2-4 | Fixture taxonomy, reference trust model, vector/rank/pseudoinverse/low-rank semantics, tolerance policy, skip handling, and failure interpretation. | Do not redo `svd_rect_fullrank_6x4` or `svd_rankdef_duplicate_5x4`; treat them as completed bounded lanes. |
| QR external compatible, rank-deficient, underdetermined/minimum-norm, and Q/economy evidence | Days 5-8 | Behavior-specific fixtures, reference semantics, basis/tolerance rules, and preserved QR/minimum-norm ownership. | Do not redo `qr_overdetermined_incompatible_4x2`; do not relabel deterministic QR fixtures as external parity. |
| Partial-SVD vector, subspace, convergence-budget, repeated/clustered spectrum, rectangular, and rank-deficient external semantics | Days 9-10 | Sign/subspace metric, convergence budget, degenerate spectra policy, and value/vector failure interpretation. | Do not redo `partial_svd_diag6_k2`; top-k value evidence is already complete. |
| Minimum-norm helper migration | Day 11 | Behavior-specific helper names and unchanged QR/COLAMD/SVD-pseudoinverse/refinement/fallback/SuiteSparse scenario ownership. | Do not move helpers into generic utilities if that hides scenario-specific tolerances or fallback behavior. |
| Bidiagonal/Golub-Kahan helper extraction | Day 12 | Dedicated helper owner preserving wide-transpose, implicit Householder reconstruction, explicit `U`/`V` reconstruction, and bidiagonal QR iteration semantics. | Do not merge specialized Bidiagonal/GK semantics into generic SVD helpers. |
| Maintainer evidence-table refresh | Day 13 | Named test owners, trust boundaries, validation commands, and non-claims. | Do not claim broad parity or support-level behavior from bounded fixture evidence. |
| Public solver-selection wording refresh | Day 14 | Earned claim table showing broader evidence than Sprint 122's bounded fixture lanes. | Do not broaden public wording without evidence-to-claim traceability. |

## Completed Work Duplicate Fence

| Completed Work | Sprint 123 Handling |
| --- | --- |
| Sprint 121 SVD/QR/rank fixture taxonomy | Use as input; do not redesign taxonomy unless a Sprint 123 decision exposes a specific gap. |
| Sprint 121 deterministic SVD, QR, rank-deficient, least-squares, pseudoinverse, low-rank, and partial-SVD fixture expansion | Use as internal evidence baseline; do not treat it as external oracle parity. |
| Sprint 122 residual owner map | Use as source; do not repeat ownership mapping except to assign Sprint 123 day owners. |
| `svd_rect_fullrank_6x4` external SVD lane | Completed prior lane; use as baseline only. |
| `svd_rankdef_duplicate_5x4` external SVD lane | Completed Sprint 122 lane; do not duplicate. |
| `qr_overdetermined_incompatible_4x2` external QR lane | Completed Sprint 122 lane; do not duplicate. |
| `partial_svd_diag6_k2` external partial-SVD top-k lane | Completed Sprint 122 lane; do not duplicate. |
| Sprint 122 minimum-norm helper ownership decision | Use as boundary input; only revisit if migration can preserve behavior-specific ownership. |
| Sprint 122 Bidiagonal/Golub-Kahan helper boundary decision | Use as boundary input; only extract through a dedicated semantic owner. |
| Sprint 122 solver-selection no-update rationale | Use as claim gate; public wording changes require additional evidence. |
| Sprint 122 validation and closeout package | Use as source of residuals and non-claims. |

## Scope Boundaries

- Sprint 123 may add bounded external oracle evidence only after each lane has
  explicit fixture, trust, tolerance, skip, basis/vector/subspace, and failure
  semantics.
- Sprint 123 may defer work when the future owner, dependency, and promotion
  gate are explicit.
- Sprint 123 must not claim broad LAPACK, NumPy, SciPy, SuiteSparse, PETSc,
  Trilinos, Eigen, ARPACK, vendor-backend, ecosystem, dense-library,
  singular-vector, Q-basis, Ritz-vector, subspace, partial-SVD convergence,
  low-rank, minimum-norm global optimality, cross-solver equivalence, package,
  ABI, platform, public API, performance, scalability, or state-of-the-art
  parity.
- Sprint 123 must not update public solver-selection wording unless broader
  evidence supports a user-facing claim.

## Validation Boundary

| Scenario | Required Validation |
| --- | --- |
| Documentation-only Day 1 work | `git diff --check` and focused whitespace scan for Sprint 123 markdown files. |
| Future `.c` or `.h` changes | `make format && make lint && make test`. |
| Future script/helper changes | Focused syntax check plus affected behavior check. |
| Future test membership or CMake changes | Include source-list, CMake/CTest count impact, and platform-specific reviewed-surface notes. |
| Future external oracle lane | Include helper invocation, skip-path proof, failure interpretation, and affected executable validation. |
| Future public wording change | Include evidence-to-claim traceability, link/path hygiene, and non-claim scan. |

## Downstream Sprint 124 Handoff Needs

| Sprint 124 Need | Sprint 123 Day 1 Owner |
| --- | --- |
| Corpus taxonomy inputs | Days 2-10 must publish accepted/deferred fixture classes and trust boundaries. |
| External-reference script inventory | Days 4, 8, and 10 must record any script changes or explicit no-change rationale. |
| Expected-failure and skip interpretation | Every accepted or deferred oracle lane must publish failure and skip semantics. |
| Report-index evidence fields | Day 13 must refresh maintainer evidence tables with lane ownership and validation commands. |
| Claim-boundary inputs | Day 14 must publish final claim gates and non-claims. |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 123 project-plan item has a day-level owner. | Complete | See project-plan item map. |
| Sprint 122 completed work is not silently reopened. | Complete | See completed work duplicate fence. |
| Downstream Sprint 124 corpus/report work has clear inputs. | Complete | See downstream handoff needs. |
| Sprint 123 working notes exist. | Complete | `WORKING_NOTES.md` created. |
| Sprint 123 artifact directory exists. | Complete | `artifacts/day1-sprint-intake.md` created. |
| Validation expectations are explicit. | Complete | See validation boundary. |
