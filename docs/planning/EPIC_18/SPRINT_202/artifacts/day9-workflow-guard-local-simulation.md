# Sprint 202 Day 9: Workflow Guard And Local Simulation

## Summary

Day 9 hardened the selected macOS benchmark freshness workflow guard around
the Day 8 hosted lane. The pass added macOS-specific negative fixtures for
workflow job drift, runner drift, missing generation/checker wiring, upload
path drift, and manifest platform mapping drift.

## Guarded Surface

The selected lane remains:

- workflow: `.github/workflows/macos-ci.yml`;
- job: `selected-performance-freshness`;
- selected row: `SRT-BENCH-REFACTOR-CSC-NOS4`;
- artifact: `build/bench-reports/canonical/bench_refactor_csc.csv`;
- checker: `scripts/check_bench_canonical_freshness.py --mode hosted`;
- upload artifact: `sprint202-macos-selected-performance-freshness`.

## Added Drift Fixtures

| Drift case | Guard evidence |
| --- | --- |
| Missing macOS selected performance job | `test_macos_performance_workflow_missing_job_fails_clearly` removes the job and expects a clear missing-job diagnostic. |
| Wrong hosted runner | `test_macos_performance_workflow_wrong_runner_fails_clearly` changes `runs-on: macos-latest` to another runner and expects a macOS runner diagnostic. |
| Missing benchmark generation step | `test_macos_performance_workflow_missing_generation_step_fails_clearly` removes `run: make bench-canonical-report` and expects a generation-command diagnostic. |
| Missing hosted checker mode | `test_macos_performance_workflow_missing_checker_mode_fails_clearly` changes `--mode hosted` and expects a hosted-mode diagnostic. |
| Wrong selected upload path | `test_macos_performance_workflow_wrong_upload_path_fails_clearly` swaps the selected `bench_refactor_csc.csv` path for an unselected CSV and expects the selected artifact path diagnostic. |
| Missing macOS manifest platform mapping | `test_macos_performance_manifest_missing_platform_fails_clearly` removes the macOS platform mapping while preserving list shape and expects a missing-platform diagnostic. |

## Local Simulation

The Day 9 local simulation remains scoped to the selected macOS metadata
contract:

```sh
BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance \
SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest \
SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
SPARSE_CANONICAL_BUILD_MODE=serial \
SPARSE_CANONICAL_CPU_MODEL=local-macos-simulation \
make bench-canonical-report &&
python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
```

The simulation proves that the generated local report can satisfy hosted-mode
metadata semantics for the selected row without asserting portable performance
or timing thresholds.

## Hosted Residual Checklist

The following remain hosted-only confirmations:

- GitHub Actions starts `selected-performance-freshness` on `macos-latest`;
- the hosted runner exposes a usable CPU model through `sysctl`;
- uploaded artifact `sprint202-macos-selected-performance-freshness` contains
  only `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`;
- the hosted checker passes with `--mode hosted`;
- the workflow summary reports exactly one selected row and does not promote
  broad performance claims.

## Claim Boundary

Day 9 strengthens local/static validation for one hosted selected macOS
benchmark freshness lane only. It does not claim Windows selected benchmark
freshness, broad benchmark-family publication, performance thresholds,
portable performance, package-manager support, ABI support, or
state-of-the-art sparse linear algebra status.
