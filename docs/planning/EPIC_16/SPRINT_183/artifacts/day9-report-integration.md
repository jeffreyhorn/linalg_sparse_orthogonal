# Sprint 183 Day 9: Report Integration

## Scope

Day 9 registered the selected Cholesky SPD tridiagonal comparison as
manifest-backed report metadata and integrated it into local selected
comparison freshness generation.

## Implemented Surfaces

| Surface | Result |
| --- | --- |
| Report family manifest | Added `comparison/cholesky_spd_tridiag_5` to `tests/corpus/manifests/report_families.tsv`. |
| Selected target manifest | Added `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` to `tests/corpus/manifests/selected_report_targets.tsv`. |
| Selected row IDs | Registered the six solve-shaped Cholesky rows for project status, baseline status, residual norm, solution norm, solution values, and project-vs-baseline max absolute delta. |
| Required generated files | Required `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv` under `build/comparison/cholesky_spd_tridiag_5/`. |
| Freshness generation | Added `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` to `make report-index-comparison-freshness`. |
| Runner metadata check | Removed the Day 8 temporary `require_report_family_metadata=False` bypass for the Cholesky target. |
| Normalizer tests | Updated selected comparison row IDs, artifact diagnostics, expected subfamilies, fixture rows, and Cholesky assertions in `tests/test_normalize_report_index.py`. |

## Manifest Contract

| Field | Value |
| --- | --- |
| Target ID | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| Target key | `cholesky-spd-tridiag-5` |
| Family | `comparison` |
| Subfamily | `cholesky_spd_tridiag_5` |
| Support tier | `local_only` |
| Freshness policy | `generated_compare_inputs` |
| Command | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` |
| Artifact | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| Expected rows | `6` |
| Workflow platforms | `linux;macos` |

Windows report freshness remains deferred. Day 9 did not add `windows` to the
selected comparison target or workflow platform metadata.

## Generated Freshness Result

`make report-index-comparison-freshness` regenerated all selected comparison
families and normalized the comparison report index with freshness checks. The
normalizer reported 39 comparison rows fresh, including the six
`cholesky_spd_tridiag_5` selected rows.

Generated files under `build/comparison/cholesky_spd_tridiag_5/` remain local
build output and are ignored by git.

## Day 10 Handoff

Day 10 should update the Linux and macOS selected comparison workflow upload
file lists and `tests/test_selected_comparison_workflow.py` guards so hosted
selected comparison artifacts include the six Cholesky files without broadening
to `build/comparison/**` and without promoting Windows freshness.

## Validation

| Command | Status |
| --- | --- |
| `python3 scripts/validate_corpus_schema.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `python3 tests/test_normalize_report_index.py` | Pass |
| `make report-index-comparison-freshness` | Pass |
| `git status --short -- build/comparison build/report-index` | Pass |
| `git diff --check` | Pass |
