# Day 8: Gate Integration

## Purpose

Wire the Sprint 175 selected macOS comparison freshness promotion into
source-controlled gate checks so the hosted workflow support tier cannot drift
silently after Day 7 implementation.

## Integration Point

The selected integration point is the GitHub Actions workflow pair:

- `.github/workflows/ci.yml` for the existing Linux reviewed hosted selected
  comparison freshness lane;
- `.github/workflows/macos-ci.yml` for the new macOS reviewed selected
  comparison freshness lane.

No Make target, generated report manifest, or normalized report-index row set
changed on Day 8. The selected Make target already regenerates the four
selected comparison families and the report-index freshness gate already checks
the 32 selected comparison rows.

## Source-Controlled Gate Added

Added `tests/test_selected_comparison_workflow.py`.

The test enforces that both Linux and macOS workflow lanes include:

- `make report-index-comparison-freshness`;
- all four selected targets:
  - `qr-minnorm`;
  - `qr-compatible-ls`;
  - `partial-svd-diag6-k2`;
  - `lu-nonsym-square-5`;
- expected generated row counts of `6`, `6`, `10`, and `6`;
- all six uploaded generated files for each selected target:
  - `project_observations.tsv`;
  - `baseline_observations.tsv`;
  - `dependency_status.tsv`;
  - `study.tsv`;
  - `summary.md`;
  - `manifest.tsv`;
- `if-no-files-found: error`;
- fail-closed summary assertions for selected-row counts, pass-row counts, and
  manifest provenance fields.

The test also checks the macOS workflow comment for explicit non-claims:

- Windows report freshness;
- external-library parity;
- package/ABI support;
- performance superiority;
- state-of-the-art status.

## Support-Tier Comment Review

The workflow comments remain bounded:

- Linux remains the source-of-truth reviewed hosted selected oracle/comparison
  lane plus supplemental Linux signals.
- macOS now has reviewed hosted selected comparison freshness only for the
  maintained selected generated comparison artifacts.
- Windows report freshness remains unsupported by Sprint 175.
- Unselected report families remain local-only, supplemental, or advisory.

## Expected Counts

The hosted selected comparison generated row count remains:

```text
selected_targets=4
total_selected_rows=28
total_pass_rows=28
```

The normalized comparison freshness gate still reports 32 selected comparison
rows because it includes four source-controlled comparison contract rows plus
28 generated comparison rows.

## Generated Output Staging

Day 8 keeps generated reports under `build/comparison/*` as ignored local
artifacts. The hosted Linux and macOS lanes upload generated selected
comparison files as workflow artifacts only.

## Validation Results

Day 8 validation:

| Check | Result |
| --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| workflow selected comparison inventory check | Passed for Linux and macOS workflows. |
| `python3 -m py_compile tests/test_selected_comparison_workflow.py` | Passed. |
| `bash scripts/package_manager_deferral_check.sh` | Passed. |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `git diff --check` | Passed. |

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 8.
