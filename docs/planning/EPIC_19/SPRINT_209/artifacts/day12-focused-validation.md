# Sprint 209 Day 12: Focused Validation

## Purpose

Run focused validation for the Sprint 209 QR incompatible re-deferral path and
separate local pass evidence from hosted-only evidence that remains unavailable.

## Pass Evidence

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Passed. | Generated `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv` under `build/comparison/qr_incompatible_ls/`; reported project-vs-baseline comparison passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed. | Reported all six generated QR incompatible rows fresh to current `HEAD` and `freshness ok (17 rows)`. |
| `python3 tests/test_normalize_report_index.py` | Passed. | Artifact inspection and freshness diagnostics remain covered. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. | Manifest row remains Linux/macOS-only and `local_only` for QR incompatible. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. | Windows workflow QR lane stays exact and bounded. |
| `python3 tests/test_validate_windows_powershell.py` | Passed. | PowerShell workflow, manifest deferral, and claim-boundary guard tests passed. |
| `make windows-powershell-guard` | Passed. | Integrated Windows/PowerShell guard target passed. |
| `make docs-check` | Passed. | Local generated API docs coverage passed. |
| `make support-docs-guard` | Passed. | Support/readiness docs guard passed. |
| `python3 tests/test_run_external_comparison.py` | Passed. | External comparison generator regressions passed. |
| `python3 -m py_compile scripts/normalize_report_index.py scripts/validate_windows_powershell.py tests/test_normalize_report_index.py tests/test_selected_comparison_workflow.py tests/test_selected_report_targets_manifest.py tests/test_validate_windows_powershell.py tests/test_run_external_comparison.py` | Passed. | Modified Python scripts and tests compile. |
| `git diff --check` | Passed. | Current diff has no whitespace errors. |

## Unavailable Evidence

| Evidence | Status | Claim impact |
| --- | --- | --- |
| Hosted Windows CI run and uploaded `sprint209-windows-selected-comparison-qr-incompatible` artifact for branch `sprint-209` | Unavailable in branch-local evidence. | Selected Windows QR incompatible freshness remains re-deferred; local generator/freshness and workflow guard passes are not hosted promotion proof. |

## Residual Blockers

| Blocker | Required closure |
| --- | --- |
| No hosted Windows QR artifact has been inspected. | A hosted Windows run must produce the exact six QR incompatible files, freshness must pass on that run, and the artifact must be reviewed before any manifest promotion. |
| QR selected manifest remains Linux/macOS-only and `local_only`. | Promotion must update workflow metadata, support tier, generated non-claim wording, docs, and guards together. |

## Closeout

Day 12 completes focused validation for the current re-deferral path. No C
source or public header changed, so `make format && make lint && make test` is
not required by the sprint rule for this day.
