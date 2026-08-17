# Day 9 Report Integration Design

## Summary

Day 9 defines the normalized report and freshness design for the Sprint 160
two-family QR comparison surface.

The implementation already generates and freshness-checks both selected
comparison targets. Day 10 should tighten report diagnostics and normalized
wording so reviewers can inspect both selected families without relying on
stale local context or minimum-norm-only documentation.

## Selected Comparison Families

| Target | Subfamily | Fixture | Operation | Artifact |
| --- | --- | --- | --- | --- |
| `qr-minnorm` | `qr_minnorm` | `qr_underdetermined_minnorm_2x4` | `minnorm_solve` | `build/comparison/qr_minnorm/study.tsv` |
| `qr-compatible-ls` | `qr_compatible_ls` | `qr_overdetermined_compatible_5x3` | `least_squares_solve` | `build/comparison/qr_compatible_ls/study.tsv` |

Both families remain fixture-local generated comparison evidence against the
source-controlled dense QR reference helper. They do not create broad QR,
external-library, hosted CI, platform, package, ABI, performance, release, or
state-of-the-art claims.

## Normalized Row Design

The selected comparison set is exactly 12 generated rows plus the two
source-controlled report-family contract rows:

| Row group | Row count | Normalized row behavior |
| --- | ---: | --- |
| `comparison/qr_minnorm` contract | 1 | Source-controlled contract row with `freshness_status=source_controlled`. |
| `comparison/qr_minnorm` generated rows | 6 | Selected generated rows loaded from `build/comparison/qr_minnorm/study.tsv`. |
| `comparison/qr_compatible_ls` contract | 1 | Source-controlled contract row with `freshness_status=source_controlled`. |
| `comparison/qr_compatible_ls` generated rows | 6 | Selected generated rows loaded from `build/comparison/qr_compatible_ls/study.tsv`. |

Each generated row keeps the native `comparison_row_id` as the normalized
`row_id`. This keeps the row IDs stable for freshness diagnostics, tests, and
reviewer inspection.

## Selected Row IDs

### QR Minimum-Norm

- `comparison_qr_underdetermined_minnorm_2x4_project_status_v1`
- `comparison_qr_underdetermined_minnorm_2x4_baseline_status_v1`
- `comparison_qr_underdetermined_minnorm_2x4_residual_norm_v1`
- `comparison_qr_underdetermined_minnorm_2x4_solution_norm_v1`
- `comparison_qr_underdetermined_minnorm_2x4_solution_values_v1`
- `comparison_qr_underdetermined_minnorm_2x4_project_vs_baseline_max_abs_delta_v1`

### QR Compatible Least-Squares

- `comparison_qr_overdetermined_compatible_5x3_project_status_v1`
- `comparison_qr_overdetermined_compatible_5x3_baseline_status_v1`
- `comparison_qr_overdetermined_compatible_5x3_residual_norm_v1`
- `comparison_qr_overdetermined_compatible_5x3_solution_norm_v1`
- `comparison_qr_overdetermined_compatible_5x3_solution_values_v1`
- `comparison_qr_overdetermined_compatible_5x3_project_vs_baseline_max_abs_delta_v1`

## Freshness Requirement Decision

`make report-index-comparison-freshness` is the required selected comparison
freshness gate for Sprint 160.

The Make target must:

1. Build the static library if needed.
2. Regenerate `qr-minnorm`.
3. Regenerate `qr-compatible-ls`.
4. Run `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`.

The required freshness policy is fail-closed:

| State | Required behavior |
| --- | --- |
| All 12 selected generated rows present, current, unique, and `pass` | Freshness passes. |
| Missing selected row | Freshness error. |
| Unexpected selected row | Freshness error. |
| Duplicate selected row | Freshness error. |
| Stale `source_commit` | Freshness error. |
| Selected generated row with non-pass status | Freshness error. |
| Selected generated row with `skip` or `defer` status | Freshness diagnostic and no proof. |
| Optional NumPy/SciPy dependency defers | Visible context only; not selected pass evidence. |

## Support-Tier Classification

Both selected families remain `local_only` generated report metadata.

This classification is intentional even though the selected gate may run in a
reviewed hosted lane:

- `support_tier=local_only` describes the generated row semantics and claim
  boundary.
- A hosted CI run can prove only that the selected gate ran and passed on that
  reviewed Linux surface.
- Hosted execution does not convert the rows into broad platform proof, package
  proof, external-library parity, release proof, or state-of-the-art evidence.

## Deterministic Reviewer Output

Reviewers should be able to inspect these files after running the selected
gate:

```text
build/comparison/qr_minnorm/project_observations.tsv
build/comparison/qr_minnorm/baseline_observations.tsv
build/comparison/qr_minnorm/dependency_status.tsv
build/comparison/qr_minnorm/study.tsv
build/comparison/qr_minnorm/summary.md
build/comparison/qr_minnorm/manifest.tsv
build/comparison/qr_compatible_ls/project_observations.tsv
build/comparison/qr_compatible_ls/baseline_observations.tsv
build/comparison/qr_compatible_ls/dependency_status.tsv
build/comparison/qr_compatible_ls/study.tsv
build/comparison/qr_compatible_ls/summary.md
build/comparison/qr_compatible_ls/manifest.tsv
```

`summary.md` should remain descriptive and deterministic enough for inspection:
it should name the target, fixture, baseline helper, project status, baseline
status, row count, and pass/fail status without adding broad QR or
external-library claims.

## Day 10 Implementation Checklist

Day 10 should make the report integration match this design by:

1. Updating selected-comparison freshness diagnostics in
   `scripts/normalize_report_index.py` so row-set and non-pass errors name both
   selected artifacts, not only `build/comparison/qr_minnorm/study.tsv`.
2. Updating focused normalizer tests to assert the two-artifact diagnostic
   wording.
3. Updating maintainer/public documentation that still describes
   `make report-index-comparison-freshness` as QR minimum-norm-only.
4. Keeping both report-family rows `local_only`.
5. Preserving the Day 8 runner tests and the existing normalizer row-state
   failure coverage.

## Validation Plan

Day 10 should run:

```sh
python3 -m py_compile scripts/run_external_comparison.py scripts/normalize_report_index.py tests/test_normalize_report_index.py tests/test_run_external_comparison.py
python3 scripts/run_external_comparison.py --self-check
python3 tests/test_run_external_comparison.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
make report-index-comparison-freshness
git diff --check
```

If `.c` or `.h` files change unexpectedly, run:

```sh
make format && make lint && make test
```
