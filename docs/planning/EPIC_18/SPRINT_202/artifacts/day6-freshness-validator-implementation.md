# Sprint 202 Day 6: Freshness Validator Implementation

## Summary

Day 6 implemented the selected macOS hosted benchmark freshness plumbing for
the existing `SRT-BENCH-REFACTOR-CSC-NOS4` row.

The implementation adds one hosted macOS lane for the already-selected
`bench_refactor_csc` / `nos4.mtx --repeat 1` artifact and preserves the
threshold-free freshness contract.

## Changed Surfaces

| Surface | Change |
| --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` | Added macOS workflow metadata to the existing selected benchmark row. |
| `.github/workflows/macos-ci.yml` | Added the `selected-performance-freshness` hosted macOS job. |
| `tests/test_selected_comparison_workflow.py` | Added macOS selected performance workflow and unselected-upload regressions. |
| `tests/test_bench_canonical_freshness.py` | Added Linux/macOS manifest assertions, macOS-style hosted metadata coverage, and selected `relative_path` drift coverage. |

## macOS Hosted Lane

The new lane:

- runs on `macos-latest`;
- generates `make bench-canonical-report`;
- checks hosted freshness with
  `scripts/check_bench_canonical_freshness.py --mode hosted`;
- emits hosted selected threshold-free metadata;
- uploads only:
  - `build/bench-reports/canonical/bench_refactor_csc.csv`;
  - `build/bench-reports/canonical/index.tsv`;
  - `build/bench-reports/canonical/manifest.txt`.

The lane explicitly excludes timing thresholds, portable performance,
external-library comparisons, broad benchmark-family publication, package/ABI
claims, Windows selected benchmark freshness, and state-of-the-art claims.

## Manifest Result

The selected benchmark target remains one row and one selected artifact. The
workflow metadata now maps the same row to two hosted platforms:

| Platform | Workflow | Job | Artifact |
| --- | --- | --- | --- |
| Linux | `.github/workflows/ci.yml` | `hosted-performance-freshness` | `sprint168-selected-performance-freshness` |
| macOS | `.github/workflows/macos-ci.yml` | `selected-performance-freshness` | `sprint202-macos-selected-performance-freshness` |

## Validation

Passed:

```sh
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_performance_docs.py
git diff --check
```

Also confirmed no `.c` or `.h` files were modified, so the full C quality gate
was not required for Day 6.

## Residuals

Remaining Sprint 202 work:

- add any additional Day 7 negative fixtures required by review;
- update public and maintainer documentation wording;
- run broader docs and report validation;
- review hosted CI evidence after the new macOS lane runs.
