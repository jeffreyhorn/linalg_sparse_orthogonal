# Day 1: Windows QR Intake

## Purpose

Day 1 established the Sprint 203 baseline for promoting or re-deferring the
selected `qr-incompatible-ls` comparison target on Windows. No implementation,
manifest promotion, or workflow promotion was attempted.

## Inputs Reviewed

| Input | Finding |
| --- | --- |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Sprint 203 requires MSVC/CMake proof, selected generator/path fixes if needed, manifest promotion only with evidence, tests, docs, and validation. |
| `docs/planning/EPIC_17/SPRINT_191/RETROSPECTIVE.md` | `qr-incompatible-ls` exists as one bounded local-only selected comparison family with six expected rows and Linux/macOS selected workflow coverage. |
| `docs/planning/EPIC_18/SPRINT_199/RETROSPECTIVE.md` | Windows selected Cholesky evidence was reviewed but manifest promotion was re-deferred because workflow success alone did not align all metadata and generated claim surfaces. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | E18-RQ-006 remains open until MSVC/CMake proof, artifact review, path normalization, manifest metadata, and claim-safe docs agree. |
| `tests/corpus/manifests/selected_report_targets.tsv` | `SRT-COMP-QR-INCOMPATIBLE-LS` currently lists Linux and macOS only, with local-only support and retained Windows/package/ABI/performance/state-of-the-art non-claims. |
| `.github/workflows/windows-ci.yml` | Windows selected comparison workflow ownership currently covers the bounded Sprint 190 Cholesky path, not QR incompatible promotion. |

## Initial Scope Map

| Item | Day 1 disposition |
| --- | --- |
| 203.1 MSVC Probe | Needs a Day 2 canonical command and Day 3 execution or hosted evidence review. |
| 203.2 Generator Fixes | Blocked on probe results; no generator fix should be made before failure/proof classification. |
| 203.3 Manifest Promotion | Blocked on exact hosted Windows evidence and source-of-truth metadata alignment. |
| 203.4 Normalizer And Workflow Tests | Needs Windows QR path, row filtering, dependency status, stale output, and upload drift coverage. |
| 203.5 Docs Calibration | Must wait for promotion or re-deferral decision, then update public and maintainer claim surfaces together. |
| 203.6 Validation | Focused Python, selected freshness, QR, Windows guard, docs, and full C gate rules are identified. |

## Current Selected QR Incompatible Target

| Field | Current value |
| --- | --- |
| Target id | `SRT-COMP-QR-INCOMPATIBLE-LS` |
| Target key | `qr-incompatible-ls` |
| Subfamily | `qr_incompatible_ls` |
| Fixture | `qr_overdetermined_incompatible_4x2` |
| Generator command | `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` |
| Artifact directory | `build/comparison/qr_incompatible_ls/` |
| Required files | `project_observations.tsv`; `baseline_observations.tsv`; `dependency_status.tsv`; `study.tsv`; `summary.md`; `manifest.tsv` |
| Expected row count | 6 |
| Current selected platforms | Linux and macOS |
| Current Windows status | Unpromoted; no selected Windows metadata |

## Day 1 Risks

| Risk | Day 1 control |
| --- | --- |
| Over-reading Windows workflow existence as freshness promotion | Require hosted artifact evidence and manifest/docs alignment before promotion. |
| Windows path separators or absolute paths breaking filtering | Plan explicit Windows path normalization and selected filtering tests. |
| Broad QR parity wording leaking into docs | Keep the target fixture-local with broad QR, ecosystem, package, ABI, performance, and state-of-the-art non-claims. |
| Fixes broadening all comparison targets | Require target-scoped design and preservation tests before edits. |
| Existing Windows Cholesky evidence drifting | Track Cholesky workflow and artifact metadata as preservation surfaces. |

## Day 1 Non-Goals

- No broad Windows report freshness.
- No Windows selected oracle or benchmark freshness.
- No broad QR, broad least-squares, or external-library parity.
- No package-manager, ABI, release, performance superiority, platform parity,
  or state-of-the-art claim.
- No unselected comparison or broad report-index promotion.

## Day 1 Completion

Day 1 created the Sprint 203 working-notes scaffold, item-to-evidence map,
current QR incompatible comparison inventory, owner-surface inventory,
validation matrix, risk register, explicit non-goals, and open questions for
the Day 2 MSVC probe design.
