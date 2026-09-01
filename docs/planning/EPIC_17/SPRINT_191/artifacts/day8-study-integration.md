# Sprint 191 Day 8: Runner Study Integration

## Summary

Day 8 integrated the selected `qr-incompatible-ls` family into the generated
study, summary, manifest, freshness, and hosted artifact surfaces.

The target remains bounded to one deterministic fixture:
`qr_overdetermined_incompatible_4x2`.

## Implementation

| Surface | Change |
| --- | --- |
| `scripts/run_external_comparison.py` | Kept the Day 5 target and Day 7 project probe, and assigned `qr_incompatible_ls` a Sprint 191 generated-row stage tag. |
| `Makefile` | Regenerates `qr-incompatible-ls` during `report-index-comparison-freshness`. |
| `tests/corpus/manifests/report_families.tsv` | Added `comparison / qr_incompatible_ls` report-family metadata. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Added `SRT-COMP-QR-INCOMPATIBLE-LS` with exact generated files and expected row IDs. |
| `scripts/normalize_report_index.py` | Added the selected row IDs and study artifact to selected comparison diagnostics. |
| `.github/workflows/ci.yml` | Summarizes and uploads the Linux `qr_incompatible_ls` artifacts by exact path. |
| `.github/workflows/macos-ci.yml` | Summarizes and uploads the macOS `qr_incompatible_ls` artifacts by exact path. |
| `tests/corpus/README.md` | Updated selected comparison documentation from five to six generated families. |

## Generated Artifacts

`python3 scripts/run_external_comparison.py --target qr-incompatible-ls`
writes:

| Artifact | Path |
| --- | --- |
| Project observations | `build/comparison/qr_incompatible_ls/project_observations.tsv` |
| Baseline observations | `build/comparison/qr_incompatible_ls/baseline_observations.tsv` |
| Dependency status | `build/comparison/qr_incompatible_ls/dependency_status.tsv` |
| Study rows | `build/comparison/qr_incompatible_ls/study.tsv` |
| Summary | `build/comparison/qr_incompatible_ls/summary.md` |
| Manifest | `build/comparison/qr_incompatible_ls/manifest.tsv` |

## Selected Rows

The study contributes six generated selected comparison rows:

| Row ID | Meaning |
| --- | --- |
| `comparison_qr_overdetermined_incompatible_4x2_project_status_v1` | Project probe status. |
| `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1` | Baseline helper status. |
| `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1` | Nonzero residual norm. |
| `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1` | Solution norm. |
| `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1` | Solution vector values. |
| `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1` | Project-vs-baseline solution delta. |

## Freshness Result

`make report-index-comparison-freshness` now regenerates six selected
comparison families:

- `qr-minnorm`
- `qr-compatible-ls`
- `qr-incompatible-ls`
- `partial-svd-diag6-k2`
- `lu-nonsym-square-5`
- `cholesky-spd-tridiag-5`

The normalized comparison freshness check passed with 46 rows.

## Claim Boundary

This is fixture-local evidence only. It does not claim broad QR parity, broad
least-squares parity, raw QR basis identity, Q sign or orientation identity,
global rank-threshold behavior, broad rank-deficient solve behavior,
NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity, Windows report freshness,
package-manager proof, shared-library ABI proof, performance superiority, or
state-of-the-art status.

## Validation

| Command | Result |
| --- | --- |
| `python3 scripts/run_external_comparison.py --self-check` | Pass |
| `python3 scripts/validate_corpus_schema.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `python3 tests/test_selected_comparison_workflow.py` | Pass |
| `python3 tests/test_normalize_report_index.py` | Pass |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Pass |
| `make report-index-comparison-freshness` | Pass, 46 normalized rows |

No `.c` or `.h` files changed, so the full C quality gate is not required for
Day 8.

## Day 9 Handoff

Day 9 should review whether target-specific freshness checks need a direct
`--selected-target qr-incompatible-ls` regression and confirm that Windows
selected comparison metadata remains bounded to the existing Sprint 190
Cholesky lane.
