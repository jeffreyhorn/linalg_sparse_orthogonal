# Day 12: Integrated Validation

## Purpose

Run the selected Sprint 175 freshness lane and its focused report-index,
workflow, manifest, documentation-claim, and staging checks together.

## Selected Lane Validation

Ran:

```sh
make report-index-comparison-freshness
```

Result: passed.

The target regenerated all four selected comparison families:

- `qr-minnorm`;
- `qr-compatible-ls`;
- `partial-svd-diag6-k2`;
- `lu-nonsym-square-5`.

The final report-index freshness check reported:

```text
normalize-report-index: freshness ok (32 rows)
```

Those 32 rows are four source-controlled comparison contract rows plus 28
generated selected comparison rows.

## Focused Validation Results

| Check | Result |
| --- | --- |
| `make report-index-comparison-freshness` | Passed. |
| `python3 tests/test_run_external_comparison.py` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 scripts/run_external_comparison.py --self-check` | Passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | Passed. |
| `bash scripts/package_manager_deferral_check.sh` | Passed. |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `git diff --check` | Passed. |

## Generated Output Staging

`make report-index-comparison-freshness` regenerated selected comparison files
under `build/comparison/`:

- `project_observations.tsv`;
- `baseline_observations.tsv`;
- `dependency_status.tsv`;
- `study.tsv`;
- `summary.md`;
- `manifest.tsv`.

for each selected target directory:

- `build/comparison/qr_minnorm/`;
- `build/comparison/qr_compatible_ls/`;
- `build/comparison/partial_svd_diag6_k2/`;
- `build/comparison/lu_nonsym_square_5/`.

`git status --short --ignored build/comparison` reports `!! build/`, so the
generated output remains ignored and is not staged for source control. Hosted
Linux/macOS evidence remains workflow-artifact-only.

## Remaining Platform Limitations

The following limitations remain intentionally deferred:

- Windows report freshness;
- selected oracle freshness on macOS;
- hosted publication of all generated reports;
- hosted generated API HTML publication;
- broad report-index freshness;
- unselected comparison family freshness;
- package-manager provider support;
- shared-library ABI support;
- runtime-loader support;
- release evidence;
- performance superiority;
- external-library parity;
- state-of-the-art sparse linear algebra status.

## Full C Quality Gate

No `.c` or `.h` files were modified for Day 12. The full C quality gate
(`make format && make lint && make test`) is therefore not required by the
Sprint instruction for this day.
