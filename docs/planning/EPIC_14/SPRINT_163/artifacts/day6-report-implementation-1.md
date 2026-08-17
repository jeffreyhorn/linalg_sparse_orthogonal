# Sprint 163 Day 6 Report Implementation I

## Purpose

Day 6 implements the first selected report enhancement from the Day 5 schema
gap analysis. The implementation is intentionally scoped to canonical benchmark
report output so selected threshold-free rows can carry the methodology fields
required by the Day 4 contract before broader report or documentation changes.

## Changed Files

- `scripts/bench_canonical_report.sh`
- `docs/planning/EPIC_14/SPRINT_163/WORKING_NOTES.md`
- `docs/planning/EPIC_14/SPRINT_163/artifacts/day6-report-implementation-1.md`

## Implementation Summary

`scripts/bench_canonical_report.sh` now appends methodology fields to
`build/bench-reports/canonical/index.tsv` while preserving the existing leading
columns:

- `report_family`
- `status`
- `support_tier`
- `claim_boundary`
- `fixture_or_workload`
- `matrix_size`
- `repeat_semantics`
- `warmup`
- `variance`
- `baseline`
- `threshold`
- `backend_context`
- `methodology_notes`

The script also writes the same canonical methodology constants to
`manifest.txt` and strengthens the manifest notes with non-superiority,
non-portability, package, ABI, runtime-loader, external-library, OpenMP
speedup, and backend-superiority caveats.

## Row Classification Added

| Field | Value | Meaning |
| --- | --- | --- |
| `report_family` | `benchmark` | Aligns canonical rows with report-family terminology. |
| `status` | `measurement` | Keeps canonical rows advisory/threshold-free rather than pass/fail. |
| `support_tier` | `local_only` | Matches the existing report-family manifest boundary. |
| `claim_boundary` | `local_threshold_free` | Blocks hard-gate or superiority interpretation. |
| `baseline` | `n/a` | Canonical rows are not thresholded gates. |
| `threshold` | `n/a` | Canonical rows are not thresholded gates. |
| `repeat_semantics` | `configured_repeat_1` or `benchmark_default` | Makes current repeat context explicit. |
| `warmup` | `not_recorded` | Prevents implied warmup evidence. |
| `variance` | `not_recorded` | Prevents implied variance evidence. |
| `backend_context` | `n/a` | Keeps canonical report rows separate from sentinel backend-context rows. |
| `methodology_notes` | `threshold_free_local_measurement;not_portable_performance_claim` | Captures compact row-level caveats for generated indexes. |

## Behavior Preserved

- `make bench-canonical-report` still emits:
  - `bench_refactor_csc.csv`
  - `bench_chol_csc.csv`
  - `bench_iterative_reuse.csv`
  - `bench_eigs_reuse.csv`
  - `index.tsv`
  - `manifest.txt`
- Existing leading `index.tsv` columns remain in place.
- Existing canonical benchmark commands are unchanged.
- `make bench`, `make bench-fast`, `make performance-sentinels`, and
  unselected exploratory benchmark commands were not changed.

## Focused Validation

The following focused checks passed:

```sh
bash -n scripts/bench_canonical_report.sh
make bench-canonical-report
python3 scripts/normalize_report_index.py --family benchmark --output build/report-index/normalized-index.tsv
```

Observed generated report behavior:

- `build/bench-reports/canonical/index.tsv` includes the appended methodology
  fields.
- The first canonical row reports `report_family=benchmark`,
  `status=measurement`, `support_tier=local_only`,
  `claim_boundary=local_threshold_free`, `repeat_semantics=configured_repeat_1`,
  `baseline=n/a`, `threshold=n/a`, `warmup=not_recorded`, and
  `variance=not_recorded`.
- `build/bench-reports/canonical/manifest.txt` includes the strengthened
  methodology-bound local measurement caveat.
- The normalized report index still loads benchmark rows successfully.

## Remaining Day 7 Work

- Add sentinel methodology fields such as `baseline_provenance`,
  `repeat_semantics`, `warmup`, and `variance`.
- Preserve S5 nonzero failure behavior.
- Confirm S5 remains a hard local wall gate while S2/S3 remain threshold-free
  backend-context rows.
- Re-run sentinel and combined normalizer checks.

## Completion Check

- Selected canonical reports can emit required methodology fields.
- Unselected rows were not silently promoted.
- Focused script checks passed.
