# Sprint 177 Day 1: Sprint Intake And Source Baseline

## Purpose

Day 1 establishes the Sprint 177 baseline before residual selection begins.
It records the source-plan authority, sprint output path, starting branch,
baseline inputs, and initial documentation/evidence surfaces for the rest of
the sprint.

## Source Authority

The active Sprint 177 project-plan section is:

- `docs/planning/EPIC_16/PROJECT_PLAN.md`
- section: `Sprint 177: Epic 16 Baseline, Evidence Matrix & Closure Gates`

The sprint artifact path is:

- `docs/planning/EPIC_16/SPRINT_177/`

Sprint 177 artifacts in this directory follow Epic 16 scope.

## Starting Snapshot

| Field | Value |
| --- | --- |
| Branch | `sprint-177` |
| Starting commit | `bd639e2d5b4ef79bf5637708f4c816a77caa75ef` |
| Source project plan | `docs/planning/EPIC_16/PROJECT_PLAN.md` |
| Sprint plan path | `docs/planning/EPIC_16/SPRINT_177/PLAN.md` |
| Working notes path | `docs/planning/EPIC_16/SPRINT_177/WORKING_NOTES.md` |
| Artifact directory | `docs/planning/EPIC_16/SPRINT_177/artifacts/` |

## Recent Prior PR Context

| Commit | Context |
| --- | --- |
| `bd639e2d` | Added Epic 16 planning review, todo, and project plan. |
| `6c610d8a` | Merged PR #195 from Sprint 176. |
| `160fb358` | Addressed PR #195 review comments for CG/MINRES NULL-handle precedence. |
| `47649dde` | Addressed PR #195 review comments for GMRES NULL-handle precedence and allocation-failure wording. |
| `dadaac44` | Addressed PR #195 review comments for allocation-failure hook countdown semantics. |

## Baseline Source Documents

| Surface | Baseline files |
| --- | --- |
| Epic 16 plan | `docs/planning/EPIC_16/PROJECT_PLAN.md` |
| Epic 16 review | `docs/planning/EPIC_16/reviews/review-codex-2026-08-23.md` |
| Epic 16 todo | `docs/planning/EPIC_16/reviews/todo-codex-2026-08-23.md` |
| Prior retrospectives | `docs/planning/EPIC_13/EPIC_13_RETROSPECTIVE.md`, `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md`, `docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md` |
| Public adoption docs | `README.md`, `INSTALL.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/api_reference.md` |
| Maintainer docs | `docs/maintainer_guide.md`, `benchmarks/README.md` |
| Build/package roots | `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in` |
| Workflows | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` |

## Baseline Evidence Surfaces

| Area | Current owner surfaces | Day 1 note |
| --- | --- | --- |
| Allocation failure | `src/sparse_alloc_internal.*`, `tests/test_iterative.c`, `tests/test_iterative_handle_helpers.h`, `make iterative-allocation-failure-gate` | Current proof is family-local to CG, GMRES, and MINRES repeated-run handles. |
| Generated API | `Doxyfile`, `docs/api_reference.md`, `scripts/check_api_docs_coverage.py`, `scripts/check_api_docs_local_only.sh` | Current generated API HTML status is local-only. |
| Package/install | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh`, `scripts/package_manager_deferral_check.sh` | Static-first install is maintained; package-manager providers remain guarded non-claims. |
| Report index | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py` | Selected report freshness is stronger than broad report freshness. |
| External comparison | `scripts/run_external_comparison.py`, `tests/test_run_external_comparison.py` | Selected QR, partial-SVD, and LU comparison families are fixture-local. |
| Performance | `benchmarks/`, `scripts/bench_canonical_report.sh`, `scripts/performance_sentinels.sh`, `tests/test_bench_canonical_freshness.py` | Performance evidence remains methodology-bound and narrow. |
| Platform CI | `.github/workflows/*.yml` | Linux is strongest; macOS/Windows carry selected reviewed lanes and retained non-claims. |
| Public headers | `include/`, `docs/api_reference.md`, prior header guards | Several high-value headers are improved, but coherence remains uneven across the full API. |
| Maintainability | large `src/` and `tests/` files, Make/CMake source lists, source-list checks | Large proof-owner files and duplicated target metadata remain planning candidates. |

## Day 1 Decisions

- Treat `docs/planning/EPIC_16/PROJECT_PLAN.md` as the Sprint 177 source
  authority.
- Keep all Sprint 177 outputs under `docs/planning/EPIC_16/SPRINT_177/`.
- Do not select Epic 16 closure targets on Day 1. Target selection depends on
  Day 2 residual extraction, Day 3 classification, and Day 6 evidence matrix
  population.
- Keep broad state-of-the-art, broad external parity, portable performance,
  package-manager provider, shared-library, dynamic ABI, runtime-loader, broad
  Windows, and broad generated-report claims as non-claims unless later sprint
  gates earn evidence.

## Day 1 Deliverables

- `docs/planning/EPIC_16/SPRINT_177/WORKING_NOTES.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day1-sprint-intake.md`
- Source-plan/sprint-path note
- Starting branch and prior PR context
- Initial baseline evidence and documentation surface inventory

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 177 scope is tied to the Epic 16 project plan | Complete | Source authority recorded above and in working notes. |
| Sprint output path is explicitly recorded | Complete | Source/path note records `EPIC_16/SPRINT_177` as the sprint output path. |
| No closure target is selected before residual audit begins | Complete | Day 1 records baseline only; target selection is deferred to Day 7. |
