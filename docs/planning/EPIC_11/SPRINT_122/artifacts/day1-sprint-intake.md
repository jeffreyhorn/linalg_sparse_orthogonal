# Sprint 122 Day 1 Sprint Intake and Residual Source Map

## Purpose

Day 1 establishes the Sprint 122 working structure and converts Sprint 121
residual deferred debt into an explicit source map. The goal is to make every
SVD, QR, partial-SVD, helper-ownership, and solver-selection follow-through item
owned without duplicating work already completed in Sprint 121.

## Inputs Reviewed

| Input | Relevant Content |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 122 | Defines seven Sprint 122 items for residual oracle decisions, helper boundaries, solver-selection claim gates, and validation. |
| `docs/planning/EPIC_11/SPRINT_122/PLAN.md` | Splits Sprint 122 into 14 days with no day above 12 hours and a 166-hour total. |
| `docs/planning/EPIC_11/SPRINT_121/RETROSPECTIVE.md` | Names the residual deferred debt and completed-work fences. |
| `docs/planning/EPIC_11/SPRINT_121/WORKING_NOTES.md` | Provides day-by-day evidence, deferral, validation, and non-claim context. |
| Sprint 121 Day 2 artifact | SVD, partial-SVD, rank, pseudoinverse, low-rank, Golub-Kahan, and bidiagonal evidence audit. |
| Sprint 121 Day 3 artifact | QR, least-squares, rank-deficient, minimum-norm, nullspace, and refinement evidence audit. |
| Sprint 121 Day 4 artifact | Matrix taxonomy classes and proof-owner metadata. |
| Sprint 121 Day 5-7 artifacts | Helper extraction plan and completed SVD/QR helper extraction boundaries. |
| Sprint 121 Day 8-10 artifacts | Rank-deficient, least-squares, pseudoinverse, low-rank, and partial-SVD fixture expansions. |
| Sprint 121 Day 11-12 artifacts | Bounded SVD external-reference pilot design and implementation. |
| Sprint 121 Day 13-14 artifacts | Validation package, deferred validation queue, closeout index, and non-claim list. |

## Day 1 Created Structure

| Path | Purpose |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_122/PLAN.md` | Day-by-day Sprint 122 execution plan. |
| `docs/planning/EPIC_11/SPRINT_122/WORKING_NOTES.md` | Running Sprint 122 notes, validation rules, day ownership, and scope boundaries. |
| `docs/planning/EPIC_11/SPRINT_122/artifacts/day1-sprint-intake.md` | Sprint intake, residual source map, duplicate fence, and completion check. |

## Sprint 122 Project-Plan Item Map

| Item | Item Name | Day Owner | Day 1 Interpretation |
| --- | --- | --- | --- |
| 1 | Residual Oracle Dedupe and Owner Map | Days 1-2 | Convert Sprint 121 residual deferred debt into explicit owners and duplicate fences. |
| 2 | Additional SVD External Fixture Decision | Days 3-4 | Decide whether any SVD external fixture beyond `svd_rect_fullrank_6x4` adds evidence. |
| 3 | QR External Dense-Reference Lane Design | Days 5-6 | Design or defer QR external reference work only after fixture, tolerance, skip, and failure semantics are explicit. |
| 4 | Partial-SVD External Parity Design | Days 7-8 | Treat partial-SVD external comparison separately from full-SVD parity because vector, subspace, convergence, and ordering semantics differ. |
| 5 | Helper Ownership Boundary Decisions | Days 9-10 | Decide minimum-norm and Bidiagonal/Golub-Kahan helper ownership without hiding scenario-specific semantics. |
| 6 | Solver-Selection Claim Gate | Days 11-12 | Update public wording only if the evidence gate supports it; otherwise document why wording remains unchanged. |
| 7 | Validation and Closeout | Days 13-14 | Validate selected work, preserve non-claims, and record future residuals. |

## Residual Source Map

| Residual | Sprint 121 Source | Sprint 122 Owner | Duplicate Fence |
| --- | --- | --- | --- |
| Additional SVD external fixtures | Retrospective residual debt; Day 11-12 SVD external pilot; Day 13-14 validation and closeout queue | Days 3-4 | Do not redo the fixed 6x4 full-SVD singular-value pilot. |
| QR external dense-reference lane | Retrospective residual debt; Day 3 QR audit; Day 4 taxonomy; Day 7 QR helper extraction; Day 9 least-squares expansion; Day 13-14 queue | Days 5-6 | Do not relabel deterministic QR and least-squares fixtures as external parity. |
| Partial-SVD external parity | Retrospective residual debt; Day 2 SVD audit; Day 4 taxonomy; Day 10 low-rank/partial-SVD expansion; Day 13-14 queue | Days 7-8 | Do not treat internal full-SVD comparisons as external partial-SVD parity. |
| Minimum-norm helper ownership migration | Retrospective residual debt; Day 3 QR audit; Day 5 helper plan; Day 7 QR helper extraction; Day 9 least-squares/pseudoinverse expansion | Day 9 | Do not move historically owned minimum-norm helpers unless a clearer QR/minimum-norm owner is established. |
| Bidiagonal/Golub-Kahan helper extraction | Retrospective residual debt; Day 2 SVD audit; Day 5 helper plan; Day 6 SVD helper extraction | Day 10 | Do not merge specialized transpose and reconstruction semantics into generic SVD helpers without proof. |
| Solver-selection wording gate | Retrospective residual debt; Day 11-12 external pilot non-claims; Day 13 validation package; Day 14 closeout | Days 11-12 | Do not broaden public claims before external/support-level evidence exists. |

## Completed Work Duplicate Fence

| Completed Sprint 121 Work | Sprint 122 Handling |
| --- | --- |
| SVD, partial-SVD, low-rank, rank, and pseudoinverse audit | Use as input; do not repeat the audit. |
| QR, least-squares, rank-deficient, and minimum-norm audit | Use as input; do not repeat the audit. |
| Matrix taxonomy design | Use fixture and evidence-class names; do not redesign taxonomy. |
| Bounded helper extraction plan | Use as helper-boundary input; do not restart the plan. |
| First SVD helper extraction batch | Preserve unless Day 10 proves a specialized boundary should change. |
| First QR helper extraction batch | Preserve unless Day 9 proves minimum-norm ownership should move. |
| Rank-deficient, least-squares, pseudoinverse, low-rank, and partial-SVD fixture expansion | Treat as internal evidence baseline, not external oracle parity. |
| Bounded SVD external-reference pilot | Treat `svd_rect_fullrank_6x4` as already implemented and validated. |
| Sprint 121 validation and closeout package | Use as source of residuals and non-claims. |

## Scope Boundaries

- Sprint 122 may add bounded external oracle evidence only after each lane has
  explicit fixture size, tolerance, skip behavior, failure interpretation, and
  non-claim wording.
- Sprint 122 must not claim broad LAPACK, SciPy, NumPy, SuiteSparse, PETSc,
  Trilinos, Eigen, dense-library, singular-vector, subspace, QR, partial-SVD,
  low-rank, pseudoinverse, package, ABI, platform, performance, public API, or
  state-of-the-art parity.
- Deferral is acceptable when the evidence owner, reason, and future trigger are
  explicit.

## Validation Boundary

| Scenario | Required Validation |
| --- | --- |
| Documentation-only Day 1 work | `git diff --check` and focused whitespace scan for Sprint 122 markdown files. |
| Future `.c` or `.h` changes | `make format && make lint && make test`. |
| Future test membership or CMake changes | Include CMake/CTest count impact and platform-specific reviewed-surface notes. |
| Future external oracle lane | Include skip-path proof, failure interpretation, and affected executable validation. |
| Future public wording change | Include evidence-to-claim traceability and non-claim scan. |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 122 project-plan item has a day-level owner. | Complete | See project-plan item map. |
| Sprint 121 residual inputs are identified. | Complete | See residual source map and inputs reviewed. |
| Completed Sprint 121 work is not silently reopened. | Complete | See duplicate fence. |
| Sprint 122 working notes exist. | Complete | `WORKING_NOTES.md` created. |
| Sprint 122 artifact directory exists. | Complete | `artifacts/day1-sprint-intake.md` created. |
| Validation expectations are explicit. | Complete | See validation boundary. |
