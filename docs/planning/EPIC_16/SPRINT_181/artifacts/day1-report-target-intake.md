# Sprint 181 Day 1: Report Target Intake

## Purpose

Day 1 establishes Sprint 181 scope, artifact layout, inherited evidence, and
manifest success criteria for the selected report target manifest work.

Sprint 181 implements the Epic 16 project-plan section "Sprint 181: Selected
Report Target Manifest". The sprint goal is to centralize selected report
target metadata so workflows, guards, docs, and freshness checks stop
duplicating target lists by hand.

## Project-Plan Scope

| Item | Day 1 intake position |
| --- | --- |
| 181.1 Target Inventory | Start with a shared inventory field list and current owner files. |
| 181.2 Manifest Schema | Defer implementation, but define the metadata categories the schema must cover. |
| 181.3 Guard Refactor | Defer implementation until duplicated target-list owners are inventoried. |
| 181.4 Workflow Scope Checks | Preserve exact YAML job and artifact upload block scope as a design constraint. |
| 181.5 Documentation Alignment | Track maintainer guide, report-index docs, README, benchmark docs, and INSTALL wording as claim surfaces. |
| 181.6 Validation | Use report normalizer tests, workflow guard tests, selected freshness checks, Python compile checks, and whitespace review as the baseline validation set. |

## Inherited Acceptance Gate

Sprint 181 implements Sprint 177 Day 8 Gate 4: Selected Report Target
Manifest.

| Gate field | Requirement |
| --- | --- |
| Residual | S177-R10 workflow and selected report target-list duplication. |
| Matrix rows | ESM-006 selected oracle freshness, ESM-007 selected comparison freshness, ESM-009 selected performance, and ESM-013 registration/workflow drift. |
| Owner files | New selected-target manifest, `Makefile`, report normalizer and generator scripts, benchmark report scripts, workflow guard tests, `.github/workflows/*.yml`, README, maintainer guide, and benchmark docs. |
| Required evidence | A source-controlled manifest owns selected oracle, comparison, performance, artifact, expected-row, support-tier, and workflow upload metadata with duplicate detection. |
| Pass definition | Workflows, guards, and docs read or validate against manifest-owned expectations; duplicates fail clearly; upload blocks remain fail-closed; selected target changes require one manifest update. |
| Protected non-claims | No broad report-index freshness, unselected oracle/comparison/performance freshness, release proof, package/ABI support, platform parity, or state-of-the-art claim. |

## Current Evidence Baseline

| Evidence | Baseline |
| --- | --- |
| Selected oracle freshness | Local QR/partial-SVD selected oracle freshness is maintained by `make report-index-oracle-freshness` and mirrored by reviewed Linux hosted report-freshness evidence. |
| Selected comparison freshness | Local selected QR, partial-SVD, and LU comparison freshness is maintained by `make report-index-comparison-freshness` and mirrored by reviewed Linux/macOS hosted selected comparison lanes. |
| Selected performance/report targets | Benchmark, sentinel, guardrail, dead-code, and coverage report rows exist with local-only or advisory semantics in current report-family metadata. |
| Report-family metadata | `tests/corpus/manifests/report_families.tsv` records report families, subfamilies, row meanings, row origins, statuses, support tiers, freshness policies, generator commands, artifact patterns, claim scopes, non-claims, owners, and introduction points. |
| Existing normalizer | `scripts/normalize_report_index.py` owns current normalization, family filtering, required-generated handling, freshness checks, and diagnostics. |
| Existing workflow guard | `tests/test_selected_comparison_workflow.py` owns selected comparison workflow checks, but Sprint 177 identifies target-list duplication as residual drift risk. |

## Sprint 180 Boundary

Sprint 180 selected a local Homebrew formula/tap proof path, but public
package-manager support remains unavailable. Sprint 181 should not change
package-manager, ABI, platform, performance, release, or state-of-the-art
claims. Any report-target manifest work that touches public docs must preserve
the current support-tier boundaries.

## Artifact Layout

Sprint 181 uses:

- `docs/planning/EPIC_16/SPRINT_181/PLAN.md`
- `docs/planning/EPIC_16/SPRINT_181/WORKING_NOTES.md`
- `docs/planning/EPIC_16/SPRINT_181/artifacts/`

Day artifacts should record the exact owner files inspected, decisions made,
validation run, and residual risks for the next day.

## Initial Inventory Fields

| Field | Intended use |
| --- | --- |
| `family` | Top-level report family or evidence category. |
| `subfamily` | Existing report-family subdivision or selected target grouping. |
| `target_key` | Stable unique identifier for one selected target. |
| `row_meaning` | Current report-index row meaning. |
| `row_origin` | Source-controlled, generated-local, hosted-CI, documentation, or advisory origin. |
| `support_tier` | Support tier used by docs, report index, and workflow guard checks. |
| `freshness_policy` | Freshness behavior expected for the selected target. |
| `generator_command` | Command that generates or validates the target. |
| `artifact_pattern` | Required artifact path or glob for generated or source-controlled evidence. |
| `expected_rows` | Expected generated row count where a target has count semantics. |
| `workflow_job` | Exact workflow job owning hosted validation, when applicable. |
| `workflow_artifact` | Exact artifact upload name/path scope, when applicable. |
| `claim_scope` | Positive claim allowed by this target. |
| `non_claims` | Explicit unsupported interpretations. |
| `owner` | Maintainer role or owner file set responsible for updates. |
| `introduced_in` | Sprint/day or artifact that introduced the target. |

## Day 1 Decisions

- Use Sprint 177 Gate 4 as the Sprint 181 acceptance gate.
- Keep manifest schema design separate from Day 1 intake; Day 1 only defines
  candidate fields and evidence owners.
- Treat `report_families.tsv` as an existing source of report-family metadata,
  not automatically as the final selected-target manifest.
- Require Day 2 to inventory duplicated selected target lists before any guard
  refactor.
- Preserve current report non-claims while centralizing metadata.

## Validation

Day 1 is documentation-only. Validation:

- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 181 scope is tied to the Epic 16 project plan. | Complete | Project-plan scope and Sprint 181 plan references above. |
| Inherited evidence and acceptance-gate requirements are explicit. | Complete | Sprint 177 Gate 4 and evidence baseline sections above. |
| Manifest design work starts from shared target metadata fields. | Complete | Initial inventory field table above. |
