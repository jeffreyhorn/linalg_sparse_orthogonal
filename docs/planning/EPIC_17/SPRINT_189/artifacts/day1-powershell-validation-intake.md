# Sprint 189 Day 1: PowerShell Validation Intake

## Purpose

Establish the Sprint 189 baseline for PowerShell validation ownership, owner
surfaces, local tool availability, current Windows report freshness deferral,
and non-claim boundaries before implementation begins.

## Planning Inputs

| Source | Day 1 use |
| --- | --- |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Defines Sprint 189 items 189.1 through 189.6 and the 166-hour scope. |
| `docs/planning/EPIC_17/SPRINT_189/PLAN.md` | Defines the 14-day execution plan. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day8-windows-acceptance-gates.md` | Defines accepted and rejected Sprint 189 outcomes. |
| `docs/planning/EPIC_16/SPRINT_182/artifacts/windows-report-freshness-deferral-decision.md` | Keeps Windows report freshness formally deferred. |

## Owner Surface Inventory

| Surface | Files | Baseline finding |
| --- | --- | --- |
| Hosted Windows workflow | `.github/workflows/windows-ci.yml` | Contains reviewed CMake-first/static-first Windows jobs using `shell: pwsh`. |
| Windows report deferral | Sprint 182 deferral artifact, README, INSTALL, maintainer guide | Report freshness remains deferred and must not be promoted by Sprint 189. |
| Selected report manifest | `tests/corpus/manifests/selected_report_targets.tsv` | Contains 7 selected rows and no `windows` workflow platform entries. |
| Workflow freshness guard | `tests/test_selected_comparison_workflow.py` | Guards that Windows CI does not run selected report freshness commands or upload selected report artifacts while deferral is active. |
| Manifest/schema validation | `scripts/validate_corpus_schema.py`, `tests/test_selected_report_targets_manifest.py` | Baseline checks pass. |
| User/maintainer docs | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Existing docs retain Windows CMake-first/static-first support and report freshness non-claims. |

## Local Tool Availability

| Tool | Status | Meaning |
| --- | --- | --- |
| `pwsh` | Not found | Local PowerShell validation is currently unavailable and must be documented as unavailable evidence, not pass evidence. |
| `powershell` | Not found | No local Windows PowerShell fallback exists. |
| `python3` | Available | Local schema, manifest, and workflow guards can run. |
| `gh` | Available | Hosted workflow dispatch and inspection are possible later in the sprint. |

## Baseline Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline is valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report manifest baseline is valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing workflow guard preserves Windows report freshness deferral. |
| Selected manifest Windows platform scan | 0 rows | No selected row currently promotes Windows report freshness. |
| `command -v pwsh` | Not found | The future validation command needs explicit local unavailable semantics. |

## Accepted Sprint 189 Outcomes

Sprint 189 can close in either of these states:

1. PowerShell validation ownership passes locally when `pwsh` exists and in
   hosted Windows CI.
2. Local `pwsh` remains unavailable, but hosted Windows owns the selected
   validation surface and local absence is documented as unavailable evidence.

## Rejected Outcomes

Sprint 189 must reject:

- treating missing local `pwsh` as passed validation;
- promoting Windows report freshness;
- uploading selected Windows report freshness artifacts;
- adding `windows` to selected report `workflow_platforms`;
- weakening the CMake-first/static-first Windows support boundary;
- claiming Windows Makefile parity or Windows `pkg-config` execution parity;
- claiming package-manager support, shared-library package support, dynamic
  ABI support, runtime-loader behavior, or broad Windows parity.

## Day 2 Handoff

Day 2 should build the detailed PowerShell surface map by auditing:

1. `.github/workflows/windows-ci.yml` PowerShell blocks and `shell: pwsh`
   declarations;
2. report-adjacent commands and artifact names guarded by
   `tests/test_selected_comparison_workflow.py`;
3. selected report manifest fields for workflow file, job, artifact, platform,
   support tier, claim scope, and non-claims;
4. maintainer and user documentation that discusses Windows PowerShell,
   report freshness, hosted evidence, or local unavailable checks.

## Validation Scope

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
