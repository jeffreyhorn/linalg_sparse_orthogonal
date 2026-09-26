# Sprint 209 Day 5: Workflow Implementation

## Purpose

Implement the bounded Windows/MSVC `qr-incompatible-ls` selected comparison
proof path designed on Day 4, while keeping broad Windows report freshness and
manifest promotion unclaimed until hosted artifact evidence exists.

## Implemented Workflow Lane

Day 5 adds one new job to `.github/workflows/windows-ci.yml`.

| Field | Value |
| --- | --- |
| Job id | `selected-qr-incompatible-comparison-freshness` |
| Job name | `Windows selected QR incompatible comparison freshness (MSVC)` |
| Runner | `windows-2022` |
| Timeout | `20` minutes |
| Build model | VS 2022 CMake configure plus Release `sparse_lu_ortho` target build. |
| Scope | Selected `qr-incompatible-ls` generated comparison proof only. |

## Implemented Commands

```text
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target sparse_lu_ortho
python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib
python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
```

## Implemented Artifact Upload

| Field | Value |
| --- | --- |
| Artifact name | `sprint209-windows-selected-comparison-qr-incompatible` |
| Upload action | `actions/upload-artifact@v4` |
| Missing-file behavior | `if-no-files-found: error` |
| Path scope | Exact six files under `build/comparison/qr_incompatible_ls/`. |

The uploaded files are:

```text
build/comparison/qr_incompatible_ls/project_observations.tsv
build/comparison/qr_incompatible_ls/baseline_observations.tsv
build/comparison/qr_incompatible_ls/dependency_status.tsv
build/comparison/qr_incompatible_ls/study.tsv
build/comparison/qr_incompatible_ls/summary.md
build/comparison/qr_incompatible_ls/manifest.tsv
```

## Guard Updates

| File | Guard change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Recognizes the Sprint 209 QR job as an owned selected lane, validates exact command and artifact membership, rejects broad/stale QR paths, and keeps stray artifact publication fail-closed. |
| `tests/test_selected_comparison_workflow.py` | Allows QR tokens only in the owned job and rejects duplicate or unowned QR commands, wrong artifact names, broad upload paths, and missing required files. |
| `tests/test_validate_windows_powershell.py` | Adds QR validator regressions for command, artifact, timeout, fail-closed upload, broad path, and missing-file drift. |

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `make windows-powershell-guard` | Passed. |

## Claim Boundary

Day 5 only adds the hosted proof path. It does not promote
`SRT-COMP-QR-INCOMPATIBLE-LS` to Windows in
`tests/corpus/manifests/selected_report_targets.tsv`, does not update the
generated support tier, and does not claim broad Windows report freshness.
Those decisions remain dependent on Day 6 hosted artifact inspection and Day 7
manifest decision evidence.
