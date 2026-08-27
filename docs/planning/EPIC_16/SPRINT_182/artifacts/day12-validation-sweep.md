# Sprint 182 Day 12: Validation Sweep

## Purpose

Run the feasible local validation sweep for the formal Windows report freshness
deferral path and record residual validation gaps.

## Validation Summary

| Command | Result | Notes |
| --- | --- | --- |
| `python3 -m py_compile scripts/validate_corpus_schema.py scripts/normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py` | Pass | Python syntax is valid for the touched schema, normalizer, and focused tests. |
| `python3 scripts/validate_corpus_schema.py` | Pass | Corpus schema and selected target manifest validate. |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass | Selected manifest diagnostics still reject Windows selected platforms while deferral is active. |
| `python3 tests/test_selected_comparison_workflow.py` | Pass | Linux, macOS, and Windows workflow guards pass, including Day 11 Windows deferral diagnostics. |
| `python3 tests/test_normalize_report_index.py` | Pass | Report-index regression coverage passes. |
| `python3 scripts/normalize_report_index.py --family corpus --family oracle --check` | Pass | Normalized corpus/oracle rows construct successfully. |
| `python3 scripts/normalize_report_index.py --family oracle --check-freshness` | Pass | Emits expected stale local oracle warnings against current `HEAD`; warnings are non-required diagnostics in this sweep. |
| `python3 scripts/normalize_report_index.py --family comparison --check-freshness` | Pass | Emits advisory local comparison freshness diagnostics. |
| `python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --check-freshness` | Pass | Emits advisory absent local report diagnostics and source-controlled package rows. |
| `bash scripts/static_package_deferral_check.sh` | Pass | Static package deferral and Windows package non-claim wording remain intact. |
| `bash scripts/package_manager_deferral_check.sh` | Pass | Package-manager provider deferral and public non-claims remain intact. |
| `git diff --check` | Pending | Run after recording the Day 12 artifact and notes. |

## PowerShell Check

`pwsh` is not installed in this local environment, so no local PowerShell parse
check was run for `.github/workflows/windows-ci.yml`. The workflow text guards
remain the local validation mechanism for the Windows job and deferral contract.

## Freshness Diagnostics

The non-required freshness commands intentionally report existing local
generated-row state rather than forcing regeneration:

- oracle rows report stale generated local artifacts against the current
  branch `HEAD`;
- comparison rows report advisory local measurement freshness;
- coverage and dead-code rows report absent local advisory reports;
- package rows remain source-controlled evidence.

These diagnostics are acceptable for the Sprint 182 deferral path because
Windows report freshness was not promoted and no generated Windows report
artifact is expected.

## Residual Risk

- Hosted Windows execution still requires GitHub Actions on `windows-2022`.
- Local PowerShell syntax validation is unavailable in this environment.
- Required selected freshness regeneration remains owned by the existing
  Linux/macOS selected freshness commands, not by the Windows deferral path.

## Completion Criteria

- Feasible local checks pass.
- Validation covers the formal Windows report freshness deferral path.
- Unsupported platform, package, performance, external-library, ABI, and
  release claims remain guarded.
