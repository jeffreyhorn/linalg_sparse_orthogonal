# Day 1 Residual Debt Intake & Dependency Ordering

## Purpose

Day 1 converts the Sprint 109 project-plan section and Sprint 108 residual
deferred debt into an actionable, bounded work package. The main risks are
duplicating Sprint 108 helper cleanup, moving behavior-sensitive eigensolver
code without source-list parity, and treating `src/sparse_matrix.c` shell work
as line-count cleanup instead of public-behavior ownership. This artifact
records the Sprint 109 owner inventory, exclusions, dependency order, and
validation expectations before boundary or implementation work begins.

## Source Inputs

- `docs/planning/EPIC_10/PROJECT_PLAN.md`, Sprint 109 section.
- `docs/planning/EPIC_10/SPRINT_108/RETROSPECTIVE.md`, residual deferred debt.
- `docs/planning/EPIC_10/SPRINT_108/WORKING_NOTES.md`.
- Sprint 108 source-boundary and closeout artifacts:
  - `day11-eigensolver-source-feasibility-boundary.md`
  - `day12-eigensolver-feasibility-closeout.md`
  - `day13-matrix-shell-public-behavior-review.md`
  - `day14-validation-metrics-closeout.md`

## Sprint 109 Workstream Inventory

| Workstream | Owner | Carry-Forward Work | Explicit Guardrail |
|---|---|---|---|
| Residual intake and ordering | Sprint 108 residual queue | Re-read residuals, exclude completed helpers, and order eigensolver, matrix-shell, and giant-test work. | No later item may be required by an earlier day. |
| Dense Jacobi source boundary | `src/sparse_eigs.c` and eigensolver internals | Revalidate `s21_dense_sym_jacobi` as the only plausible private helper source candidate. | No movement before Make/CMake/manifest/source-list parity and focused validation are planned. |
| Dense Jacobi extraction or deferral | eigensolver private source owner | Move only `s21_dense_sym_jacobi` if low risk; otherwise publish an explicit no-split deferral. | No public header, install-header, helper-target, or CTest drift. |
| Eigensolver behavior audit | grow-m, refinement, dispatch/defaults, handle/workspace, shift-invert, shared Lanczos paths | Add behavior evidence and explicit no-go conditions for future movement. | No additional eigensolver code movement in Sprint 109 without stronger evidence. |
| Matrix-shell boundary contract | `src/sparse_matrix.c` | Choose one future public-behavior owner and define private-header dependencies, source-list requirements, focused public tests, and solver smoke gates. | Do not move matrix-shell code unless independently low risk. |
| Giant-test cleanup pass | `tests/test_ldlt_csc.c`, `tests/test_qr.c`, `tests/test_iterative.c`, `tests/test_svd.c` | Select one bounded helper family after excluding Sprint 108 helper work. | Preserve proof assertions at call sites and avoid new compiled helper targets. |
| Validation and closeout | all touched Sprint 109 surfaces | Run checks appropriate to touched files and publish metrics/residuals. | No accidental public API, install-header, source-list, helper-target, or CTest drift. |

## Live Owner Snapshot

| Owner | Current Lines | Sprint 109 Disposition |
|---|---:|---|
| `src/sparse_eigs.c` | 1,538 | Source-boundary candidate only for `s21_dense_sym_jacobi`; behavior-sensitive paths remain audit-first. |
| `src/sparse_matrix.c` | 1,359 | Public-behavior contract only unless later evidence proves a low-risk owner split. |
| `tests/test_ldlt_csc.c` | 3,896 | Eligible for one future helper family only after excluding Sprint 108 residual helper work. |
| `tests/test_qr.c` | 3,210 | Eligible for one future fixture/proof cleanup only after excluding Sprint 108 tall fixture helper work. |
| `tests/test_iterative.c` | 2,849 | Eligible for one convergence-sensitive cleanup only if options and comparisons remain visible. |
| `tests/test_svd.c` | 2,890 | Eligible for one validation-lane cleanup only after excluding Sprint 108 full-UV fixture work. |

## Completed Sprint 108 Work Excluded From Sprint 109

These items are already complete and must not be reintroduced as Sprint 109
scope:

- Sprint 108 residual intake and live proof-owner re-rank;
- LDLT CSC residual assertion helper:
  `assert_s20_solve_residual_below`;
- QR tall diagonal-dominant fixture helper:
  `make_qr_tall_diagonal_dominant`;
- iterative diagonal-preconditioner fixture helper:
  `make_iterative_diagonal_precond_matrix`;
- SVD full-UV fixture helper:
  `make_svd_full_uv_fixture_16x8`;
- eigensolver feasibility boundary and closeout handoff;
- matrix-shell public-behavior review;
- Sprint 108 final validation, metrics, and drift checks.

## Dependency-Ordered Work Queue

| Order | Work | Reason This Comes Before Later Work |
|---:|---|---|
| 1 | Residual intake and exclusions | Prevents duplicate helper work and stale Sprint 108 assumptions. |
| 2 | Dense Jacobi boundary revalidation | Defines whether source movement is even allowed. |
| 3 | Source-list parity and validation prep | Required before any internal helper source is added. |
| 4 | Dense Jacobi extraction or deferral | The only approved implementation-source movement candidate. |
| 5 | Dense Jacobi focused validation | Closes build/test parity before broader eigensolver audit claims. |
| 6 | Grow-m/refinement/shared-kernel audit | Separates behavior-sensitive no-go paths from future candidates. |
| 7 | Dispatch/handle/shift-invert audit | Completes eigensolver behavior evidence after source candidate disposition. |
| 8 | Matrix-shell candidate contract | Converts central shell risk into one future public-behavior owner. |
| 9 | Matrix-shell validation/no-move decision | Prevents unsupported matrix-shell movement in Sprint 109. |
| 10 | Giant-test cleanup candidate selection | Chooses one bounded helper family after source-boundary risks are fenced. |
| 11 | Giant-test cleanup follow-through | Implements only the approved family with proof visibility retained. |
| 12 | Focused integration and drift check | Confirms touched surfaces before full quality gate. |
| 13 | Full quality gate and metrics | Produces branch-wide evidence for code/build/test changes. |
| 14 | Residual closeout | Publishes downstream work only after validation and metrics are known. |

## Day-Level Ownership

| Day | Focus | Sprint 109 Item(s) | Primary Output |
|---:|---|---|---|
| 1 | Scope and residual intake | 1 | Working notes and intake artifact. |
| 2 | Dense Jacobi source-boundary revalidation | 2 | Boundary artifact and go/no-go criteria. |
| 3 | Source-list parity and validation harness prep | 2 | Build/test/source-list checklist. |
| 4 | Dense Jacobi extraction or no-split deferral | 3 | Private helper source change or deferral artifact. |
| 5 | Dense Jacobi cross-lane validation | 3 | Focused eigensolver validation artifact. |
| 6 | Grow-m/refinement/shared-kernel audit | 4 | Behavior-sensitive boundary audit. |
| 7 | Dispatch/handle/shift-invert audit | 4 | Public workflow and direct-solver interaction audit. |
| 8 | Matrix-shell candidate boundary contract | 5 | Future owner contract. |
| 9 | Matrix-shell validation and no-move decision | 5 | Public behavior validation notes. |
| 10 | Giant-test cleanup candidate selection | 6 | Cleanup boundary artifact. |
| 11 | Giant-test cleanup follow-through | 6 | Bounded proof-owner cleanup or deferral. |
| 12 | Focused integration and drift check | 7 | Focused validation and no-drift evidence. |
| 13 | Full quality gate and metrics | 7 | Full validation and maintainability metrics. |
| 14 | Residual closeout | 7 | Downstream residual queue and retrospective input. |

## Validation Expectations

| Scenario | Required Validation |
|---|---|
| Documentation-only day | `git diff --check` and trailing-whitespace scan over touched docs. |
| Test `.c` day | Focused touched test suite plus `make format && make lint && make test`. |
| Test helper/header day | Focused impacted tests plus `make format && make lint && make test`. |
| Implementation source day | Focused family tests, source-list/build checks if membership changes, and `make format && make lint && make test`. |
| Build-system/source-list day | Make, CMake, source-list parity, focused build/test surfaces, and full quality gate. |
| Public header/install day | Public API/install/export review, downstream/package checks as applicable, and full quality gate. |
| Mixed day | Apply the strongest requirement from any touched surface. |

## Initial Non-Goals

- No public API change.
- No install-header change.
- No new compiled test helper target.
- No reviewed CTest count change unless explicitly planned and reviewed.
- No broad eigensolver split beyond the dense Jacobi candidate.
- No matrix-shell source movement without an independently low-risk public
  behavior owner.
- No broad solver-family rewrite from giant-test fixture cleanup.

## Completion Criteria Status

- Every Sprint 109 project-plan item has day-level ownership.
- No Sprint 109 item depends on work scheduled later in the sprint.
- Completed Sprint 108 helper work is listed as excluded.
- Validation expectations are explicit before source-boundary or cleanup work
  starts.
