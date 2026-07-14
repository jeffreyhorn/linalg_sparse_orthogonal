# Sprint 124 Day 1 Sprint Intake and Residual Dependency Map

## Purpose

Day 1 establishes the Sprint 124 working structure and converts Sprint 123's
residual deferred debt into explicit dependency-ordered proof owners. The goal
is to make every QR rank-deficient, QR minimum-norm, QR Q-basis/economy,
partial-SVD vector/subspace, partial-SVD residual, helper-ownership, and
claim-gate follow-through item owned without duplicating completed Sprint
121-123 oracle and helper-boundary work.

## Inputs Reviewed

| Input | Relevant Content |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 124 | Defines seven Sprint 124 items for residual QR, partial-SVD, helper ownership, validation, maintainer evidence, and solver-selection claim gates. |
| `docs/planning/EPIC_11/SPRINT_124/PLAN.md` | Splits Sprint 124 into 14 days with no day above 12 hours and a 166-hour total. |
| `docs/planning/EPIC_11/SPRINT_123/RETROSPECTIVE.md` | Names Sprint 123 residual deferred debt, non-claims, and completed-work fences. |
| `docs/planning/EPIC_11/SPRINT_123/WORKING_NOTES.md` | Provides day-by-day context for bounded QR, partial-SVD, helper, maintainer-evidence, and solver-selection decisions. |
| Sprint 123 Day 5-8 artifacts | QR behavior requirements, compatible/rank-deficient decision, minimum-norm/Q/economy deferral decision, and compatible QR implementation. |
| Sprint 123 Day 9-10 artifacts | Partial-SVD external semantics design and completed top-k tall fixture implementation. |
| Sprint 123 Day 11-12 artifacts | Minimum-norm helper migration decision and Bidiagonal/Golub-Kahan helper extraction decision. |
| Sprint 123 Day 13-14 artifacts | Maintainer evidence refresh, solver-selection no-update rationale, validation package, residual queue, and non-claim register. |
| Sprint 122 artifacts | Earlier SVD, QR, partial-SVD, helper, and solver-selection ownership boundaries. |
| Sprint 121 artifacts | Fixture taxonomy, QR/SVD/rank audits, matrix families, helper extraction boundaries, and deterministic fixture expansion baseline. |

## Day 1 Created Structure

| Path | Purpose |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_124/PLAN.md` | Day-by-day Sprint 124 execution plan. |
| `docs/planning/EPIC_11/SPRINT_124/WORKING_NOTES.md` | Running Sprint 124 notes, validation rules, day ownership, and scope boundaries. |
| `docs/planning/EPIC_11/SPRINT_124/artifacts/day1-sprint-intake.md` | Sprint intake, residual dependency map, duplicate fence, and completion check. |

## Sprint 124 Project-Plan Item Map

| Item | Item Name | Day Owner | Day 1 Interpretation |
| --- | --- | --- | --- |
| 1 | Rank-Deficient QR Oracle Design | Days 2-3 | Define rank-threshold, nullspace, pseudoinverse, tolerance, skip, and failure policies before implementing or explicitly deferring rank-deficient QR external evidence. |
| 2 | QR Minimum-Norm Oracle Design | Days 4-5 | Define behavior-specific QR minimum-norm evidence across QR solve, COLAMD, SVD-pseudoinverse, fallback, refinement, and optional SuiteSparse paths before implementation or deferral. |
| 3 | QR Q-Basis and Economy Oracle Design | Days 6-7 | Define sign, orientation, projection, subspace, and economy-shape semantics before implementing or explicitly deferring basis/economy evidence. |
| 4 | Partial-SVD Vector and Subspace Semantics | Days 8-9 | Define sign-invariant vector, projection, subspace, tolerance, residual, and failure semantics before implementation or explicit deferral. |
| 5 | Partial-SVD Residual Semantics Batch | Days 10-11 | Decide repeated-spectrum, clustered-spectrum, rank-deficient, convergence-budget, and low-rank optimality evidence without claiming broad partial-SVD parity. |
| 6 | Helper Ownership Follow-Through | Day 12 | Revisit minimum-norm and Bidiagonal/Golub-Kahan helper movement only with behavior-specific helper names and dedicated ownership. |
| 7 | Validation, Docs, and Claim Gate | Days 13-14 | Validate accepted work, refresh maintainer evidence, and update solver-selection wording only if the evidence supports a public claim. |

## Residual Dependency Map

| Sprint 123 Residual | Sprint 124 Owner | Dependency / Promotion Gate | Duplicate Fence |
| --- | --- | --- | --- |
| Rank-deficient QR external oracle evidence | Days 2-3 | Rank-threshold, nullspace, pseudoinverse, minimum-norm separation, tolerance, skip, diagnostics, and failure-interpretation policy. | Do not redo QR compatible or incompatible external lanes; do not treat residual-only rank evidence as nullspace or minimum-norm proof. |
| QR minimum-norm external oracle evidence | Days 4-5 | Behavior-specific owner spanning QR solve, COLAMD, SVD-pseudoinverse, fallback, refinement, optional SuiteSparse paths, norm/residual comparison, and skip behavior. | Do not migrate into generic helpers that hide QR/COLAMD/SVD-pseudoinverse/refinement/fallback/SuiteSparse scenario assertions. |
| QR Q-basis and economy external evidence | Days 6-7 | Sign/orientation policy, projection/subspace metric, economy-shape policy, and basis-dependent failure interpretation. | Do not claim vector equality or broad QR basis parity from solve residual evidence. |
| Partial-SVD vector/subspace evidence | Days 8-9 | Sign-invariant vector comparison, projection/subspace metrics, tolerance rules, residual meaning, and failure interpretation. | Do not redo completed top-k singular-value fixtures or treat singular-value agreement as vector/subspace evidence. |
| Partial-SVD residual scenarios | Days 10-11 | Repeated-spectrum, clustered-spectrum, rank-deficient, convergence-budget, and low-rank optimality scenario gates with explicit trust boundaries. | Do not claim broad partial-SVD convergence, vector/subspace, or low-rank optimality parity from bounded fixtures. |
| Minimum-norm helper migration | Day 12 | Behavior-specific helper naming, scenario-local assertions, and owner visibility across QR solve/COLAMD/SVD-pseudoinverse/refinement/fallback/SuiteSparse behavior. | Do not move helpers if the movement hides behavior-specific diagnostics or tolerance policy. |
| Bidiagonal/Golub-Kahan helper extraction | Day 12 | Dedicated semantic owner preserving wide transpose, implicit Householder reconstruction, explicit `U`/`V`, wide GK skips, and bidiagonal QR iteration semantics. | Do not merge specialized Bidiagonal/Golub-Kahan semantics into generic SVD helpers. |
| Maintainer evidence and solver-selection wording | Days 13-14 | Validation evidence, affected-owner map, non-claim register, and evidence-to-claim traceability. | Do not broaden public wording without earned evidence and explicit claim boundaries. |

## Completed Work Duplicate Fence

| Completed Work | Sprint 124 Handling |
| --- | --- |
| Sprint 121 SVD/QR/rank fixture taxonomy | Use as input; do not redesign taxonomy unless a Sprint 124 decision exposes a concrete gap. |
| Sprint 121 deterministic SVD, QR, rank-deficient, least-squares, pseudoinverse, low-rank, and partial-SVD fixture expansion | Use as internal evidence baseline; do not treat it as external oracle parity. |
| `svd_rect_fullrank_6x4` external SVD lane | Completed prior lane; use as baseline only. |
| `svd_rankdef_duplicate_5x4` external SVD lane | Completed prior lane; do not duplicate. |
| `svd_wide_fullrank_4x6` external SVD lane | Completed Sprint 123 lane; do not duplicate. |
| `qr_overdetermined_incompatible_4x2` external QR lane | Completed prior lane; do not duplicate. |
| `qr_overdetermined_compatible_5x3` external QR lane | Completed Sprint 123 lane; do not duplicate. |
| `partial_svd_diag6_k2` external partial-SVD top-k lane | Completed prior lane; do not duplicate. |
| `partial_svd_tall5x3_k2` external partial-SVD top-k lane | Completed Sprint 123 lane; do not duplicate. |
| Sprint 123 minimum-norm helper migration decision | Use as boundary input; only revisit if migration can preserve behavior-specific ownership. |
| Sprint 123 Bidiagonal/Golub-Kahan helper decision | Use as boundary input; only extract through a dedicated semantic owner. |
| Sprint 123 maintainer evidence refresh | Use as baseline; update only for Sprint 124 ownership or validation changes. |
| Sprint 123 solver-selection no-update rationale | Use as claim gate; public wording changes require additional evidence. |
| Sprint 123 validation and closeout package | Use as source of residuals and non-claims. |

## Scope Boundaries

- Sprint 124 may add bounded external oracle evidence only after each lane has
  explicit trust, tolerance, skip, rank, basis, vector/subspace, residual, and
  failure semantics.
- Sprint 124 may defer work when the future owner, dependency, and promotion
  gate are explicit.
- Sprint 124 must not claim broad LAPACK, NumPy, SciPy, SuiteSparse, PETSc,
  Trilinos, Eigen, ARPACK, vendor-backend, ecosystem, dense-library,
  singular-vector, Q-basis, Ritz-vector, subspace, rank-deficient QR,
  minimum-norm, partial-SVD convergence, low-rank, global optimality,
  cross-solver equivalence, package, ABI, platform, public API, performance,
  scalability, or state-of-the-art parity.
- Sprint 124 must not update public solver-selection wording unless new
  evidence supports a user-facing claim.

## Validation Boundary

| Scenario | Required Validation |
| --- | --- |
| Documentation-only Day 1 work | `git diff --check` and focused whitespace scan for Sprint 124 markdown files. |
| Future `.c` or `.h` changes | `make format && make lint && make test`. |
| Future script/helper changes | Focused syntax check plus affected behavior check. |
| Future test membership or CMake changes | Include source-list, CMake/CTest count impact, and platform-specific reviewed-surface notes. |
| Future external oracle lane | Include helper invocation, skip-path proof, failure interpretation, and affected executable validation. |
| Future maintainer/public wording change | Include evidence-to-claim traceability, link/path hygiene, and non-claim scan. |

## Downstream Sprint 125 Handoff Needs

| Sprint 125 Need | Sprint 124 Day 1 Owner |
| --- | --- |
| Corpus taxonomy inputs | Days 2-11 must publish accepted/deferred fixture classes and trust boundaries. |
| External-reference script inventory | Days 3, 5, 7, 9, and 11 must record any script changes or explicit no-change rationale. |
| Expected-failure and skip interpretation | Every accepted or deferred oracle lane must publish failure and skip semantics. |
| Report-index evidence fields | Day 13 must refresh maintainer evidence tables with lane ownership and validation commands. |
| Claim-boundary inputs | Day 14 must publish final claim gates and non-claims. |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 124 project-plan item has a day-level owner. | Complete | See project-plan item map. |
| Sprint 121-123 completed work is not silently reopened. | Complete | See completed work duplicate fence. |
| Downstream Sprint 125 corpus/report work has clear inputs. | Complete | See downstream handoff needs. |
| Sprint 124 working notes exist. | Complete | `WORKING_NOTES.md` created. |
| Sprint 124 artifact directory exists. | Complete | `artifacts/day1-sprint-intake.md` created. |
| Validation expectations are explicit. | Complete | See validation boundary. |
