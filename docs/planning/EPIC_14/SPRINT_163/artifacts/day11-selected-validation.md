# Sprint 163 Day 11 Selected Benchmark And Sentinel Validation

## Purpose

Day 11 validates the selected Sprint 163 benchmark/report and sentinel surfaces
after the Day 6 through Day 10 script and documentation changes. The goal is to
prove the selected local commands emit the required methodology fields and keep
hard gates separate from threshold-free reports.

## Commands Run

```sh
bash -n scripts/bench_canonical_report.sh scripts/performance_sentinels.sh
make bench-canonical-report
make performance-sentinels
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel \
  --output build/report-index/normalized-index.tsv
```

## Command Results

| Command | Result | Notes |
| --- | --- | --- |
| `bash -n scripts/bench_canonical_report.sh scripts/performance_sentinels.sh` | Pass | Shell syntax accepted for both selected report scripts. |
| `make bench-canonical-report` | Pass | Wrote canonical CSVs, `index.tsv`, and `manifest.txt` under `build/bench-reports/canonical/`. |
| `make performance-sentinels` | Pass | Wrote `sentinels.tsv`, `manifest.txt`, `wall_check.txt`, `bench_chol_csc_nos4.csv`, and `bench_refactor_csc_kkt.csv`. |
| `python3 tests/test_normalize_report_index.py` | Pass | Focused normalizer regression test reported `test-normalize-report-index: ok`. |
| `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --output build/report-index/normalized-index.tsv` | Pass | Wrote 26 normalized benchmark/sentinel rows. |

## Canonical Report Validation

Generated file inspected:

- `build/bench-reports/canonical/index.tsv`

Summary:

- row count: `4`
- missing required methodology fields: `none`
- statuses: `measurement`
- claim boundaries: `local_threshold_free`
- repeat semantics: `benchmark_default`, `configured_repeat_1`

Required methodology fields confirmed:

- `report_family`
- `status`
- `support_tier`
- `claim_boundary`
- `fixture_or_workload`
- `repeat_semantics`
- `warmup`
- `variance`
- `baseline`
- `threshold`
- `methodology_notes`

Interpretation:

- canonical rows are threshold-free local measurements;
- `baseline=n/a` and `threshold=n/a` confirm they are not hard timing gates;
- `warmup=not_recorded` and `variance=not_recorded` prevent statistical
  overclaims.

## Sentinel Report Validation

Generated file inspected:

- `build/bench-reports/sentinels/sentinels.tsv`

Summary:

- total sentinel rows: `19`
- missing appended methodology fields: `none`
- S5 rows: `3`
  - statuses: `pass`
  - claim boundaries: `local_wall_gate`
- S2 rows: `8`
  - statuses: `report`
  - claim boundaries: `local_threshold_free`
- S3 rows: `8`
  - statuses: `report`
  - claim boundaries: `local_threshold_free`

Required appended methodology fields confirmed:

- `baseline_provenance`
- `repeat_semantics`
- `warmup`
- `variance`
- `methodology_notes`

Interpretation:

- S5 remains the only selected hard local timing gate;
- S5 status is meaningful only with baseline, threshold, fixture, command,
  baseline provenance, and local machine context;
- S2 and S3 remain threshold-free backend-context rows and do not pass, fail,
  or prove backend superiority.

## Local-Only Limitations

All generated artifacts inspected during Day 11 are local outputs under ignored
`build/` paths. They are validation evidence for this branch and current local
environment only. They are not:

- hosted CI proof;
- package proof;
- ABI proof;
- runtime-loader proof;
- broad platform proof;
- portable performance proof;
- OpenMP speedup proof;
- backend superiority proof;
- state-of-the-art proof.

## Validation-Driven Fixes

No Day 11 script or documentation fixes were required after the selected
validation commands passed.

## Completion Check

- Selected local validation passed.
- Generated rows match the methodology contract.
- Local-only limitations are documented.
