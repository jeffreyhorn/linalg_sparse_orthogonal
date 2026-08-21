# Day 7: Normalization Implementation

## Purpose

Implement the Sprint 175 selected normalization lane designed on Day 6:
reviewed macOS selected comparison freshness, plus the Linux hosted selected
comparison inventory reconciliation needed to keep hosted selected evidence
consistent with the four-target Make freshness gate.

## Implemented Workflow Changes

### macOS Selected Comparison Freshness

Added a new job to `.github/workflows/macos-ci.yml`:

- job id: `selected-comparison-freshness`;
- job name: `macOS reviewed selected comparison freshness`;
- runner: `macos-latest`;
- timeout: `15` minutes;
- command: `make report-index-comparison-freshness`;
- artifact name: `sprint175-macos-selected-comparison-freshness`.

The job intentionally depends only on checkout, Make, Python 3, the default
hosted macOS C compiler, and the repository's maintained comparison target.
It does not install package, documentation, coverage, static-analysis, CMake,
or benchmark tooling.

### macOS Summary Assertions

The macOS summary step now validates all four selected comparison targets:

| Target | Directory | Expected rows |
| --- | --- | ---: |
| `qr-minnorm` | `build/comparison/qr_minnorm` | 6 |
| `qr-compatible-ls` | `build/comparison/qr_compatible_ls` | 6 |
| `partial-svd-diag6-k2` | `build/comparison/partial_svd_diag6_k2` | 10 |
| `lu-nonsym-square-5` | `build/comparison/lu_nonsym_square_5` | 6 |

The summary fails closed if:

- a selected `study.tsv`, `dependency_status.tsv`, or `manifest.tsv` is
  missing;
- a target has an unexpected selected-row count;
- a target has fewer pass rows than expected selected rows;
- manifest `source_commit`, `source_branch`, or `platform` metadata is
  missing.

The aggregate expected hosted generated row count is:

```text
selected_targets=4 total_selected_rows=28 total_pass_rows=28
```

### macOS Artifact Upload

The macOS job uploads exactly six generated files for each selected target:

- `project_observations.tsv`;
- `baseline_observations.tsv`;
- `dependency_status.tsv`;
- `study.tsv`;
- `summary.md`;
- `manifest.tsv`.

The upload uses `if-no-files-found: error` and `retention-days: 7`, so missing
generated output fails the job instead of silently producing partial evidence.

## Linux Reconciliation

Updated `.github/workflows/ci.yml` so the existing Linux reviewed hosted
selected comparison freshness path now matches the four-target Make target.

The Linux summary now includes `lu-nonsym-square-5`, validates exact row counts
for all four selected targets, validates manifest provenance/platform fields,
and uploads the six generated LU files from:

```text
build/comparison/lu_nonsym_square_5/
```

The reconciled Linux hosted artifact name is:

```text
sprint175-linux-selected-comparison-freshness
```

This reconciliation preserves the existing Linux hosted behavior while closing
the Sprint 174 LU inventory mismatch identified on Day 3. It is not the Sprint
175 promotion lane; the selected Sprint 175 promotion remains the new macOS
hosted selected comparison job.

## Generated Output Staging

Generated comparison reports remain local ignored artifacts under
`build/comparison/*`. Day 7 does not commit generated comparison outputs.

The new hosted behavior publishes generated selected comparison outputs only
as GitHub Actions artifacts for the reviewed hosted Linux and macOS lanes.
There is no source-controlled publication of generated TSV or Markdown report
outputs.

## Claim Boundaries Preserved

Day 7 preserves these non-claims:

- no Windows report freshness;
- no broad macOS platform parity;
- no hosted publication of all generated reports;
- no hosted generated API HTML publication;
- no unselected comparison family freshness;
- no broad QR, partial-SVD, LU, or external-library parity;
- no package-manager support;
- no shared-library ABI support;
- no runtime-loader support;
- no release evidence;
- no performance superiority;
- no state-of-the-art sparse linear algebra claim.

## Focused Assertions Added

The workflow summaries are the new focused assertions for this implementation.
They check target inventory, selected row counts, pass row counts, and manifest
provenance for every selected generated comparison family.

No C source or public header files were changed.

## Local Validation Results

Day 7 implementation was checked with:

| Check | Result |
| --- | --- |
| `make report-index-comparison-freshness` | Passed; regenerated all four selected comparison families and `normalize-report-index` reported freshness ok for 32 rows. |
| `python3 tests/test_run_external_comparison.py` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `python3 scripts/run_external_comparison.py --self-check` | Passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | Passed; freshness ok for 32 rows. |
| workflow selected comparison inventory check | Passed for `.github/workflows/ci.yml` and `.github/workflows/macos-ci.yml`. |
| `git diff --check` | Passed. |

Because no `.c` or `.h` files were modified, the full C quality gate is not
required for Day 7.
