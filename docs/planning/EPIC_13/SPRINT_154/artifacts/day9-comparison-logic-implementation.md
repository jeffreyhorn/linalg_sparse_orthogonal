# Day 9: Comparison Logic Implementation

## Scope

Day 9 completed the first selected project-vs-baseline comparison layer for
the Sprint 154 QR minimum-norm target:
`qr_underdetermined_minnorm_2x4`.

The comparison remains local generated artifact evidence. It supports only the
fixture-local claim that the selected project QR minimum-norm solve agrees with
the selected source-controlled dense reference helper for the selected metrics
and tolerances.

## Implementation

Updated `scripts/run_external_comparison.py` to add:

- `study.tsv` row emission using the Day 5 schema;
- deterministic stable row ids for the six selected metric rows;
- project-vs-baseline residual norm comparison;
- project-vs-baseline solution norm comparison;
- project-vs-baseline solution value comparison;
- project-vs-baseline maximum absolute solution delta;
- selected-row validation for missing, duplicate, and non-pass rows;
- `summary.md` human-readable narrow study scaffold;
- manifest paths for `study.tsv` and `summary.md`;
- `--self-check` smoke coverage for selected-row validation and deferred
  optional dependency semantics.

The existing Day 8 trace outputs remain in place:

- `project_observations.tsv`;
- `baseline_observations.tsv`;
- `dependency_status.tsv`;
- `manifest.tsv`.

## Generated Study Rows

The Day 9 run emits six selected rows in
`build/comparison/qr_minnorm/study.tsv`:

| Row id | Metric | Status |
| --- | --- | --- |
| `comparison_qr_underdetermined_minnorm_2x4_project_status_v1` | `project_status` | `pass` |
| `comparison_qr_underdetermined_minnorm_2x4_baseline_status_v1` | `baseline_status` | `pass` |
| `comparison_qr_underdetermined_minnorm_2x4_residual_norm_v1` | `residual_norm` | `pass` |
| `comparison_qr_underdetermined_minnorm_2x4_solution_norm_v1` | `solution_norm` | `pass` |
| `comparison_qr_underdetermined_minnorm_2x4_solution_values_v1` | `solution_values` | `pass` |
| `comparison_qr_underdetermined_minnorm_2x4_project_vs_baseline_max_abs_delta_v1` | `project_vs_baseline_max_abs_delta` | `pass` |

Observed local deltas:

| Metric | Delta | Tolerance |
| --- | --- | --- |
| `residual_norm` | `1.5700924586837752e-16` | `1e-10` |
| `solution_norm` | `1.1102230246251565e-16` | `1e-10` |
| `solution_values` | `1.1102230246251565e-16` | `1e-10` per component |
| `project_vs_baseline_max_abs_delta` | `1.1102230246251565e-16` | `1e-10` |

## Smoke Checks

Added `python3 scripts/run_external_comparison.py --self-check`, which checks:

- complete selected rows validate successfully;
- missing selected rows produce `missing_selected_row`;
- duplicate selected rows produce `duplicate_selected_row`;
- non-pass selected rows produce `metric_tolerance_miss`;
- mismatched vector lengths produce `metric_comparison_malformed`;
- NumPy and SciPy optional package rows remain `defer`, not pass evidence.

## Validation

Ran:

```sh
python3 scripts/run_external_comparison.py --self-check
python3 scripts/run_external_comparison.py --target qr-minnorm
```

Results:

- self-check passed;
- project-side probe passed;
- baseline helper passed;
- all six selected comparison rows passed;
- `study.tsv`, `summary.md`, and manifest references were generated.

## Non-Claims

Day 9 does not claim:

- broad QR parity;
- NumPy or SciPy parity;
- external-library ecosystem parity;
- package-manager proof;
- hosted CI proof;
- shared-library or ABI support;
- performance superiority;
- state-of-the-art status.

## Day 10 Handoff

Day 10 should decide how generated comparison rows enter report-index
semantics. The default safe option remains artifact-only until freshness,
normalization, stale-output, and local-only support-tier rules are explicit.
