# Sprint 202 Day 8: Hosted Workflow Lane Implementation

## Summary

Day 8 audited the selected macOS hosted benchmark freshness lane implemented
for Sprint 202 and ran a local hosted-mode simulation using the macOS metadata
contract.

The implemented lane is:

- workflow: `.github/workflows/macos-ci.yml`;
- job: `selected-performance-freshness`;
- selected row: `SRT-BENCH-REFACTOR-CSC-NOS4`;
- artifact: `build/bench-reports/canonical/bench_refactor_csc.csv`;
- checker: `scripts/check_bench_canonical_freshness.py --mode hosted`.

## Workflow Evidence

| Requirement | Evidence |
| --- | --- |
| Selected platform | macOS hosted runner through `runs-on: macos-latest`. |
| Runtime budget | `timeout-minutes: 10`. |
| Benchmark generation | `make bench-canonical-report`. |
| Freshness validation | `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted`. |
| Metadata label | `BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance`. |
| Hosted scope | `SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected`. |
| Claim boundary | `SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free`. |
| Runner context | `SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest`. |
| Build flags | `SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags`. |
| Build mode | `SPARSE_CANONICAL_BUILD_MODE=serial`. |
| CPU metadata | `sysctl -n machdep.cpu.brand_string`, with fallback to `unknown`. |

## Upload Scope

The selected hosted artifact upload is limited to:

- `build/bench-reports/canonical/bench_refactor_csc.csv`;
- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`.

The lane does not upload unselected canonical benchmark CSVs and does not use
broad benchmark glob paths.

## Local Simulation

Passed:

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

Observed selected row fields included:

- `support_tier=hosted_selected`;
- `claim_boundary=hosted_selected_threshold_free`;
- `runner_context=github-actions-macos-latest`;
- `report_label=sprint202-macos-hosted-performance`;
- `baseline=n/a`;
- `threshold=n/a`;
- `warmup=none_configured`;
- `variance=not_computed_single_sample`;
- `methodology_notes=threshold_free_local_measurement;not_portable_performance_claim`.

## Claim Boundary

The Day 8 lane supports only hosted selected benchmark freshness for the named
macOS row. It does not claim portable performance, timing thresholds, broad
benchmark publication, benchmark superiority, Windows selected benchmark
freshness, package/ABI proof, runtime-loader proof, OpenMP speedup, backend
superiority, or state-of-the-art status.

## Residual

Actual hosted CI execution must still be reviewed after the branch is pushed
and the macOS workflow runs.
