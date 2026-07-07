# Day 1 Residual Debt Intake & Duplicate-Work Fence

## Purpose

Day 1 converts the Sprint 110 project-plan section and Sprint 109 residual
deferred debt into an actionable, bounded work package. The main risks are
duplicating completed Sprint 109 dense Jacobi, Matrix Market owner-selection,
QR exact-RHS, and validation work; moving Matrix Market I/O before Matrix
builder ownership is resolved; moving behavior-sensitive eigensolver code
without direct validation; and hiding proof values in the remaining giant tests.
This artifact records the Sprint 110 owner inventory, exclusions, dependency
order, day-level ownership, and validation expectations before boundary or
implementation work begins.

## Source Inputs

- `docs/planning/EPIC_10/PROJECT_PLAN.md`, Sprint 110 section.
- `docs/planning/EPIC_10/SPRINT_109/RETROSPECTIVE.md`, residual deferred debt.
- `docs/planning/EPIC_10/SPRINT_109/WORKING_NOTES.md`.
- Sprint 109 source-boundary, proof-owner, and closeout artifacts:
  - `day4-dense-jacobi-extraction.md`
  - `day5-dense-jacobi-cross-lane-validation.md`
  - `day6-growm-refinement-shared-kernel-audit.md`
  - `day7-dispatch-handle-shift-invert-audit.md`
  - `day8-matrix-shell-candidate-boundary-contract.md`
  - `day10-giant-test-cleanup-boundary.md`
  - `day11-giant-test-cleanup-follow-through.md`
  - `day12-focused-integration-drift-check.md`
  - `day13-full-quality-gate-metrics.md`
  - `day14-sprint-closeout-residual-queue.md`

## Sprint 110 Workstream Inventory

| Workstream | Owner | Carry-Forward Work | Explicit Guardrail |
|---|---|---|---|
| Residual intake and duplicate-work fence | Sprint 109 residual queue | Re-read residuals, exclude completed work, and order Matrix I/O, eigensolver, proof-owner, and SVD work. | No later item may be required by an earlier day. |
| Matrix builder ownership | `SparseBuildEntry`, `sparse_matrix_build_from_entries`, and `src/sparse_matrix.c` | Decide whether builder helpers become a private source owner or remain in the central matrix shell. | Builder ownership must be resolved before Matrix Market movement. |
| Matrix Market source split | `sparse_load_mm`, `sparse_save_mm`, and future `src/sparse_matrix_io.c` candidate | Move Matrix Market load/save only if builder ownership and focused validation gates prove the split low risk. | No public API, install-header, source-list, helper-target, or reviewed CTest drift. |
| Eigensolver behavior owner validation | defaults, dispatch, workspace, refinement, shift-invert, or shared Lanczos behavior | Select at most one behavior owner beyond dense Jacobi and validate directly, or publish a no-move contract. | Dense Jacobi extraction is complete and must not be repeated. |
| Direct and iterative proof-owner cleanup | `tests/test_qr.c`, `tests/test_ldlt_csc.c`, and `tests/test_iterative.c` | Perform one bounded cleanup family while preserving least-squares, refinement, oracle, convergence, and residual proof values. | Do not introduce a broad cross-solver helper or hide proof assertions. |
| SVD proof-loop cleanup | `tests/test_svd.c` | Review storage-layout, stride, rank, orthogonality, and reconstruction proof loops and extract only one safe setup helper family. | SVD proof values must remain visible at call sites. |
| Validation and residual handoff | all touched Sprint 110 surfaces | Run checks appropriate to touched files and publish metrics/residuals. | No accidental public API, install-header, source-list, helper-target, or CTest drift. |

## Live Owner Snapshot

| Owner | Current Lines | Sprint 110 Disposition |
|---|---:|---|
| `src/sparse_matrix.c` | 1,359 | Matrix builder and Matrix Market source-boundary candidate; no broad matrix-shell movement. |
| `src/sparse_eigs.c` | 1,412 | Behavior-sensitive validation candidate only; dense Jacobi extraction is complete. |
| `src/sparse_eigs_dense_internal.c` | 129 | Existing dense Jacobi private helper owner; no Sprint 110 movement planned. |
| `tests/test_qr.c` | 3,234 | Eligible only for non-duplicate sequential RHS/proof setup cleanup. |
| `tests/test_ldlt_csc.c` | 3,896 | Eligible only for one dense-reference oracle-lane cleanup if proof visibility remains strong. |
| `tests/test_iterative.c` | 2,849 | Eligible only for one per-solver exact-RHS cleanup family. |
| `tests/test_svd.c` | 2,890 | Eligible only for one setup helper family after proof-loop review. |

## Completed Sprint 109 Work Excluded From Sprint 110

These items are already complete and must not be reintroduced as Sprint 110
scope:

- Sprint 109 residual intake and dependency ordering;
- dense Jacobi boundary review;
- dense Jacobi extraction into `src/sparse_eigs_dense_internal.c`;
- dense Jacobi Makefile, CMake, manifest, and source-list registration;
- focused dense Jacobi Make and CMake validation;
- eigensolver grow-m/refinement/shared-kernel no-move audit;
- eigensolver dispatch/defaults/handle/workspace/shift-invert no-move audit;
- Matrix Market future-owner selection;
- QR exact-RHS helper cleanup in `tests/test_qr.c`;
- Sprint 109 focused integration, full validation, metrics, and drift checks.

## Residual Work Carried Forward

The unresolved Sprint 109 debt is intentionally narrower than the completed
Sprint 109 work:

- Matrix builder ownership must be resolved before Matrix Market source
  movement.
- Matrix Market load/save can move toward `src/sparse_matrix_io.c` only after
  builder ownership and focused validation gates are known.
- Eigensolver behavior-sensitive movement remains audit-first; Sprint 110 may
  select at most one behavior owner beyond dense Jacobi.
- QR cleanup must avoid duplicating `make_qr_exact_rhs` and must preserve
  least-squares/refinement proof values.
- LDLT CSC oracle cleanup remains a dedicated lane because it couples external
  references, Windows skips, factorization behavior, and dense solve
  comparison.
- Iterative exact-RHS cleanup must remain solver-family-specific.
- SVD cleanup must preserve storage-layout, stride, rank, orthogonality, and
  reconstruction proof values.

## Dependency-Ordered Work Queue

| Order | Work | Reason This Comes Before Later Work |
|---:|---|---|
| 1 | Residual intake and exclusions | Prevents duplicate Sprint 109 cleanup and stale assumptions. |
| 2 | Matrix builder dependency audit | Identifies whether builder helpers can support a private source owner. |
| 3 | Matrix builder ownership decision | Required before any Matrix Market movement or no-split closure. |
| 4 | Matrix Market boundary plan | Converts the builder decision into an implementation or deferral path. |
| 5 | Matrix Market source split follow-through | Executes only the approved movement or closes the deferral. |
| 6 | Matrix Market focused validation | Proves file I/O and loaded-matrix solver behavior before downstream claims. |
| 7 | Eigensolver behavior-owner selection | Picks one behavior owner or explicitly defers all candidates. |
| 8 | Eigensolver behavior-owner validation | Validates the selected owner or publishes a no-move contract. |
| 9 | Direct/iterative proof-owner boundary selection | Chooses one safe cleanup family after Matrix/eigs risks are fenced. |
| 10 | Direct/iterative proof-owner cleanup | Implements only the selected family with proof values visible. |
| 11 | SVD proof-loop boundary review | Defines claim-critical proof values before SVD cleanup. |
| 12 | SVD proof-loop cleanup or deferral | Applies only the selected safe helper family. |
| 13 | Integrated validation and metrics | Produces branch-wide evidence for code/build/test/doc changes. |
| 14 | Sprint closeout and residual handoff | Publishes downstream work only after validation and metrics are known. |

## Day-Level Ownership

| Day | Focus | Sprint 110 Item(s) | Primary Output |
|---:|---|---|---|
| 1 | Scope, residual intake, exclusions, dependency ordering, and validation expectations. | 1 | Working notes and intake artifact. |
| 2 | Matrix builder dependency audit. | 2 | Builder dependency and behavior-coupling artifact. |
| 3 | Matrix builder ownership decision. | 2 | Builder go/no-go ownership artifact. |
| 4 | Matrix Market source-boundary plan. | 3 | Matrix I/O movement checklist or no-split setup. |
| 5 | Matrix Market split follow-through or no-split closure. | 3 | Source movement or deferral artifact. |
| 6 | Matrix Market focused validation. | 3 | Matrix tests, solver-smoke, and drift evidence. |
| 7 | Eigensolver behavior-owner selection. | 4 | Selected owner or no-move rationale. |
| 8 | Eigensolver behavior-owner validation. | 4 | Direct validation or no-move contract. |
| 9 | QR, LDLT CSC, and iterative proof-owner boundary selection. | 5 | Proof-owner cleanup boundary artifact. |
| 10 | Direct/iterative proof-owner cleanup. | 5 | Bounded cleanup or deferral. |
| 11 | SVD proof-loop boundary review. | 6 | SVD proof-loop map and helper-family decision. |
| 12 | SVD cleanup or deferral. | 6 | SVD helper cleanup or explicit deferral. |
| 13 | Integrated validation and metrics. | 7 | Quality gate, no-drift evidence, and metrics. |
| 14 | Sprint closeout and residual handoff. | 7 | Residual queue and retrospective input. |

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
- No duplicate dense Jacobi extraction work.
- No duplicate Matrix Market owner-selection work.
- No duplicate QR exact-RHS helper cleanup.
- No Matrix Market source split before builder ownership is decided.
- No broad matrix-shell extraction.
- No broad eigensolver behavior split.
- No broad cross-solver proof helper abstraction.

## Completion Criteria Status

- Every Sprint 110 project-plan item has day-level ownership.
- No Sprint 110 item depends on work scheduled later in the sprint.
- Completed Sprint 109 work is listed as excluded.
- Validation expectations are explicit before source-boundary or cleanup work
  starts.
