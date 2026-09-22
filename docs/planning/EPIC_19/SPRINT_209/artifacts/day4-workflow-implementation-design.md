# Sprint 209 Day 4: Workflow Implementation Design

## Purpose

Define the exact Windows workflow implementation needed to generate hosted
MSVC/CMake evidence for `qr-incompatible-ls` without broadening Windows report
freshness or unrelated selected-target claims.

## Inputs

| Input | Day 4 interpretation |
| --- | --- |
| Day 2 MSVC probe design | Provides the exact QR generator and selected freshness commands. |
| Day 3 hosted evidence inventory | Confirms the current hosted Windows workflow has only the bounded Cholesky selected comparison artifact. |
| `.github/workflows/windows-ci.yml` | Existing selected comparison job is Cholesky-only and should remain independently owned. |
| `scripts/validate_windows_powershell.py` | Current Windows validator permits the Cholesky lane and rejects QR drift. |
| `tests/test_selected_comparison_workflow.py` | Current workflow tests enforce Cholesky-only Windows selected comparison behavior and must be revised for one owned QR lane. |

## Proposed Workflow Change

Add a new sibling job to `.github/workflows/windows-ci.yml`:

| Field | Required value |
| --- | --- |
| Job id | `selected-qr-incompatible-comparison-freshness` |
| Job name | `Windows selected QR incompatible comparison freshness (MSVC)` |
| Runner | `windows-2022` |
| Timeout | `20` minutes |
| Purpose | Generate and freshness-check only the selected `qr-incompatible-ls` comparison rows on hosted Windows/MSVC. |

The existing `selected-comparison-freshness` Cholesky job should remain
unchanged except for any shared comment wording needed to explain the two
bounded selected lanes.

## Required Steps

| Step | Shell | Command |
| --- | --- | --- |
| Checkout | action | `actions/checkout@v4` |
| Configure selected QR incompatible comparison library | `pwsh` | `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` |
| Build selected QR incompatible comparison library | `pwsh` | `cmake --build build --config Release --target sparse_lu_ortho` |
| Generate selected QR incompatible comparison report | `cmd` | `python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib` |
| Check selected QR incompatible comparison freshness | `cmd` | `python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` |
| Upload selected QR incompatible comparison freshness artifact | action | `actions/upload-artifact@v4` with exact file membership and fail-closed behavior. |

## Artifact Contract

| Field | Required value |
| --- | --- |
| Artifact name | `sprint209-windows-selected-comparison-qr-incompatible` |
| Missing-file behavior | `if-no-files-found: error` |
| Allowed path root | `build/comparison/qr_incompatible_ls/` only |
| Broad paths forbidden | `build/comparison/**`, `build/comparison/`, repository root, `build/**`, and any non-QR comparison subfamily. |

The upload must include exactly:

```text
build/comparison/qr_incompatible_ls/project_observations.tsv
build/comparison/qr_incompatible_ls/baseline_observations.tsv
build/comparison/qr_incompatible_ls/dependency_status.tsv
build/comparison/qr_incompatible_ls/study.tsv
build/comparison/qr_incompatible_ls/summary.md
build/comparison/qr_incompatible_ls/manifest.tsv
```

## PowerShell Validation Updates

Day 5 should update `scripts/validate_windows_powershell.py` to:

| Area | Required behavior |
| --- | --- |
| Owned QR lane constants | Add constants for job id, artifact name, target key, subfamily, commands, and six required upload paths. |
| Job parser checks | Require runner, timeout, configure step, build step, generator step, freshness step, upload step, and expected shells. |
| Claim-boundary markers | Require docs to describe the QR lane as selected-target-only and to keep broad Windows report freshness deferred. |
| Drift rejection | Reject QR commands outside the owned job, stale Sprint 203 artifact names, broad upload paths, missing required files, duplicate QR target commands, and missing fail-closed upload behavior. |
| Hosted parser mode | Keep `--require-pwsh` as the hosted parse gate for owned Windows snippets. |

## Workflow Regression Updates

Day 5 should update `tests/test_selected_comparison_workflow.py` so it rejects:

- missing `selected-qr-incompatible-comparison-freshness`;
- wrong runner or missing `timeout-minutes: 20`;
- missing or altered `qr-incompatible-ls` generator command;
- missing CMake probe generator, arch, config, or MSVC library path;
- missing `--selected-target qr-incompatible-ls`;
- wrong or stale QR artifact name;
- broad comparison upload paths;
- omitted required QR artifact file;
- QR commands outside the owned QR job.

The same test file should continue to protect the Cholesky lane independently.

## Manifest And Claim Sequencing

Do not promote `tests/corpus/manifests/selected_report_targets.tsv` on workflow
implementation alone. Manifest promotion requires a hosted Windows run that:

- completes the new QR job successfully;
- uploads `sprint209-windows-selected-comparison-qr-incompatible`;
- contains the six expected QR incompatible row IDs;
- has source commit and clean worktree metadata tied to the reviewed branch;
- passes selected freshness for `qr-incompatible-ls`.

Until that evidence exists, the manifest should keep `SRT-COMP-QR-INCOMPATIBLE-LS`
Linux/macOS-only, `local_only`, and bounded by `no Windows report freshness`.

## Day 4 Decision

Sprint 209 should implement one bounded hosted Windows/MSVC QR incompatible
selected comparison lane on Day 5. The lane is sufficient to request hosted
evidence, but it is not by itself enough to claim promotion. Promotion remains
dependent on Day 6 artifact inspection and Day 7 manifest decision evidence.
